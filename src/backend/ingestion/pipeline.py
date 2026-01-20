"""Main ingestion pipeline orchestrator."""

import logging
import asyncio
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Literal, Union
from dataclasses import dataclass, field

from .unstructured import UnstructuredClient
from .chunker import SectionChunkBuilder, SectionNode
from .page_builder import PageBuilder
from .narrative_generator import NarrativeGenerator
from .classifier import DocumentClassifier
from .embedder import Embedder, get_embedder
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..database.models import DataRoom, Folder, Document, Page, Section, Chunk
from ..extraction.linker import EntityLinker
from ..config import get_settings

logger = logging.getLogger(__name__)


# Pipeline type for factory method
PipelineType = Literal["legacy", "graphrag"]


@dataclass
class IngestionProgress:
    """Tracks ingestion progress."""

    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, running, extracting_entities, completed, failed
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Entity extraction tracking
    total_documents_for_extraction: int = 0
    extracted_documents: int = 0
    entities_found: int = 0

    @property
    def file_progress(self) -> float:
        """Get file processing progress (0-1)."""
        return self.processed_files / self.total_files if self.total_files > 0 else 0

    @property
    def chunk_progress(self) -> float:
        """Get chunk processing progress (0-1)."""
        return self.processed_chunks / self.total_chunks if self.total_chunks > 0 else 0

    @property
    def extraction_progress(self) -> float:
        """Get entity extraction progress (0-1)."""
        return self.extracted_documents / self.total_documents_for_extraction if self.total_documents_for_extraction > 0 else 0


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    success: bool
    dataroom_id: str
    documents_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0


class IngestionPipeline:
    """Orchestrates the document ingestion process."""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        unstructured_client: Optional[UnstructuredClient] = None,
        embedder: Optional[Embedder] = None,
        max_concurrent_files: Optional[int] = None,
        extract_entities: bool = False,
        generate_narratives: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            neo4j_client: Neo4j client. Defaults to global instance.
            unstructured_client: Unstructured.io client.
            embedder: Embedding generator.
            max_concurrent_files: Max files to process in parallel. Defaults to settings.
            extract_entities: Whether to extract entities after document processing.
            generate_narratives: Whether to generate folder narratives and section descriptions.
        """
        settings = get_settings()
        self.neo4j = neo4j_client or get_neo4j_client()
        self.unstructured = unstructured_client or UnstructuredClient(
            max_workers=max_concurrent_files or settings.max_concurrent_files
        )
        self.embedder = embedder or get_embedder()
        self.classifier = DocumentClassifier()
        self.section_builder = SectionChunkBuilder()
        self.page_builder = PageBuilder()
        self.narrative_generator = NarrativeGenerator() if generate_narratives else None
        self.max_concurrent_files = max_concurrent_files or settings.max_concurrent_files
        self.extract_entities = extract_entities
        self.generate_narratives = generate_narratives
        self.entity_linker = EntityLinker(self.neo4j) if extract_entities else None

        self._progress = IngestionProgress()
        self._progress_callback: Optional[Callable[[IngestionProgress], None]] = None
        self._progress_lock = threading.Lock()

    @property
    def progress(self) -> IngestionProgress:
        """Get current progress."""
        return self._progress

    def set_progress_callback(
        self, callback: Callable[[IngestionProgress], None]
    ):
        """Set callback for progress updates.

        Args:
            callback: Function called with IngestionProgress on updates.
        """
        self._progress_callback = callback

    def _update_progress(self, **kwargs):
        """Update progress and notify callback. Thread-safe."""
        with self._progress_lock:
            for key, value in kwargs.items():
                setattr(self._progress, key, value)
            if self._progress_callback:
                self._progress_callback(self._progress)

    async def ingest_folder(
        self,
        dataroom_id: str,
        folder_path: Path,
        recursive: bool = True,
    ) -> IngestionResult:
        """Ingest all documents from a folder.

        Args:
            dataroom_id: ID of the target data room.
            folder_path: Path to the folder to ingest.
            recursive: Whether to process subfolders.

        Returns:
            IngestionResult with statistics.
        """
        start_time = datetime.utcnow()
        self._progress = IngestionProgress(
            status="running",
            start_time=start_time,
        )
        errors = []

        try:
            # Discover files
            files = self._discover_files(folder_path, recursive)
            self._update_progress(total_files=len(files))
            logger.info(f"Found {len(files)} files to process")

            if not files:
                return IngestionResult(
                    success=True,
                    dataroom_id=dataroom_id,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                )

            # Create folder nodes
            await self._create_folder_nodes(dataroom_id, folder_path, files)

            # Generate folder narratives if enabled
            if self.generate_narratives and self.narrative_generator:
                await self._generate_folder_narratives(dataroom_id, folder_path, files)

            # Process files in parallel with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_files)
            processed_count = 0
            total_chunks = 0
            results_lock = asyncio.Lock()

            async def process_with_semaphore(file_path: Path) -> tuple[Path, int, Optional[str]]:
                """Process a single file with semaphore limiting concurrency."""
                nonlocal processed_count, total_chunks

                async with semaphore:
                    self._update_progress(current_file=file_path.name)

                    try:
                        chunks = await self._process_document(
                            dataroom_id, folder_path, file_path
                        )
                        chunk_count = len(chunks)

                        # Update progress atomically
                        async with results_lock:
                            processed_count += 1
                            total_chunks += chunk_count
                            self._update_progress(
                                processed_files=processed_count,
                                total_chunks=total_chunks,
                            )

                        return file_path, chunk_count, None

                    except Exception as e:
                        error_msg = f"Error processing {file_path.name}: {e}"
                        logger.error(error_msg)

                        # Persist error to Document node if it exists
                        self._persist_document_error(
                            dataroom_id, str(file_path), str(e)
                        )

                        async with results_lock:
                            processed_count += 1
                            self._update_progress(processed_files=processed_count)

                        return file_path, 0, error_msg

            # Launch all tasks and gather results
            logger.info(f"Processing {len(files)} files with max {self.max_concurrent_files} concurrent")
            tasks = [process_with_semaphore(fp) for fp in files]
            results = await asyncio.gather(*tasks)

            # Collect errors from results
            for file_path, chunk_count, error in results:
                if error:
                    errors.append(error)

            self._update_progress(
                processed_files=len(files),
                total_chunks=total_chunks,
                processed_chunks=total_chunks,
                status="extracting_entities" if self.extract_entities else "completed",
                end_time=None if self.extract_entities else datetime.utcnow(),
            )

            # Extract entities from processed documents
            entities_extracted = 0
            if self.extract_entities and self.entity_linker:
                entities_extracted = await self._extract_entities(dataroom_id, errors)

            self._update_progress(
                status="completed",
                end_time=datetime.utcnow(),
            )

            # Update data room stats
            await self._update_dataroom_stats(dataroom_id)

            duration = (datetime.utcnow() - start_time).total_seconds()
            return IngestionResult(
                success=len(errors) == 0,
                dataroom_id=dataroom_id,
                documents_processed=len(files) - len(errors),
                chunks_created=total_chunks,
                entities_extracted=entities_extracted,
                errors=errors,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.exception(f"Ingestion failed: {e}")
            self._update_progress(
                status="failed",
                error_message=str(e),
                end_time=datetime.utcnow(),
            )
            return IngestionResult(
                success=False,
                dataroom_id=dataroom_id,
                errors=[str(e)],
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _discover_files(
        self, folder_path: Path, recursive: bool
    ) -> list[Path]:
        """Discover all supported files in a folder.

        Args:
            folder_path: Root folder path.
            recursive: Whether to search recursively.

        Returns:
            List of file paths.
        """
        files = []
        pattern = "**/*" if recursive else "*"

        for file_path in folder_path.glob(pattern):
            if file_path.is_file() and self.unstructured.is_supported(file_path):
                files.append(file_path)

        # Sort by path for consistent ordering
        return sorted(files)

    async def _create_folder_nodes(
        self,
        dataroom_id: str,
        root_path: Path,
        files: list[Path],
    ):
        """Create folder nodes in Neo4j.

        Args:
            dataroom_id: Data room ID.
            root_path: Root folder path.
            files: List of files (to determine folder structure).
        """
        # Get unique folder paths
        folders = set()
        for file_path in files:
            rel_path = file_path.parent.relative_to(root_path)
            current = root_path
            for part in rel_path.parts:
                current = current / part
                folders.add(current)

        # Create folder nodes
        for folder_path in sorted(folders):
            rel_path = folder_path.relative_to(root_path)
            parent_rel = folder_path.parent.relative_to(root_path) if folder_path != root_path else None

            folder = Folder.create(
                dataroom_id=dataroom_id,
                path=str(rel_path),
                name=folder_path.name,
            )

            # Find parent folder ID
            parent_folder_id = None
            if parent_rel and str(parent_rel) != ".":
                parent_folder_id = f"folder:{dataroom_id}:{hashlib.sha256(str(parent_rel).encode()).hexdigest()[:12]}"

            folder.parent_folder_id = parent_folder_id

            # Upsert to Neo4j
            self.neo4j.create_node(
                labels=["Folder"],
                properties=folder.to_neo4j_properties(),
            )

            # Create relationship to parent or data room
            if parent_folder_id:
                self.neo4j.create_relationship(
                    start_id=parent_folder_id,
                    start_label="Folder",
                    end_id=folder.id,
                    end_label="Folder",
                    rel_type="CONTAINS",
                )
            else:
                self.neo4j.create_relationship(
                    start_id=dataroom_id,
                    start_label="DataRoom",
                    end_id=folder.id,
                    end_label="Folder",
                    rel_type="HAS_ROOT",
                )

    async def _generate_folder_narratives(
        self,
        dataroom_id: str,
        root_path: Path,
        files: list[Path],
    ):
        """Generate narratives for all folders in the data room.

        Args:
            dataroom_id: Data room ID.
            root_path: Root folder path.
            files: List of files.
        """
        if not self.narrative_generator:
            return

        logger.info("Generating folder narratives...")
        self._update_progress(current_file="Generating folder narratives...")

        # Build file tree
        file_tree = self.narrative_generator.build_file_tree(root_path, files)

        # Get data room name
        dr_result = self.neo4j.execute_read(
            "MATCH (dr:DataRoom {id: $id}) RETURN dr.name as name",
            {"id": dataroom_id}
        )
        dataroom_name = dr_result[0]["name"] if dr_result else "Data Room"

        # Get all folders for this data room
        folders_result = self.neo4j.execute_read(
            "MATCH (f:Folder {dataroom_id: $dataroom_id}) RETURN f.id as id, f.path as path",
            {"dataroom_id": dataroom_id}
        )

        for folder in folders_result or []:
            narrative = self.narrative_generator.generate_folder_narrative(
                folder["path"],
                file_tree,
                dataroom_name,
            )

            if narrative:
                self.neo4j.execute_write(
                    "MATCH (f:Folder {id: $id}) SET f.narrative = $narrative",
                    {"id": folder["id"], "narrative": narrative}
                )

    async def _process_document(
        self,
        dataroom_id: str,
        root_path: Path,
        file_path: Path,
    ) -> list[Chunk]:
        """Process a single document.

        Args:
            dataroom_id: Data room ID.
            root_path: Root folder path.
            file_path: Path to the document.

        Returns:
            List of created chunks.
        """
        logger.info(f"Processing: {file_path.name}")

        # Compute content hash for deduplication
        content_hash = self._compute_file_hash(file_path)

        # Get folder ID
        rel_path = file_path.parent.relative_to(root_path)
        folder_id = f"folder:{dataroom_id}:{hashlib.sha256(str(rel_path).encode()).hexdigest()[:12]}"

        # Parse with Unstructured.io
        elements = await self.unstructured.parse_document(file_path)

        # Classify document
        full_text = "\n".join([el.text for el in elements])
        doc_type, confidence = self.classifier.classify(full_text, file_path.name)

        # Create document node
        document = Document.create(
            dataroom_id=dataroom_id,
            folder_id=folder_id,
            full_path=str(file_path),
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            file_type=file_path.suffix.lower(),
            doc_type=doc_type,
            doc_type_confidence=confidence,
            content_hash=content_hash,
            ingestion_status="processing",
        )

        self.neo4j.create_node(
            labels=["Document"],
            properties=document.to_neo4j_properties(),
        )

        # Create relationship to folder
        self.neo4j.create_relationship(
            start_id=folder_id,
            start_label="Folder",
            end_id=document.id,
            end_label="Document",
            rel_type="CONTAINS",
        )

        # Build Pages
        pages, page_id_map = self.page_builder.extract_pages(
            elements, dataroom_id, document.id
        )

        # Build Sections
        sections, section_nodes, section_page_map = self.section_builder.build_sections(
            elements, dataroom_id, document.id, page_id_map
        )

        # Build Chunks
        chunks = self.section_builder.build_chunks(
            section_nodes, dataroom_id, document.id, page_id_map
        )

        # Generate section descriptions for top-level sections
        if self.generate_narratives and self.narrative_generator:
            for section, node in zip(sections, section_nodes):
                if section.hierarchy_level == 0:
                    content_preview = self.section_builder.get_section_content_preview(node)
                    description = self.narrative_generator.generate_section_description(
                        section.title,
                        content_preview,
                        str(doc_type.value) if hasattr(doc_type, 'value') else str(doc_type),
                        file_path.name,
                    )
                    if description:
                        section.description = description

        # Save document graph
        await self._save_document_graph(
            document, pages, page_id_map, sections, section_page_map, chunks
        )

        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_texts(chunk_texts, show_progress=True)

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            self.neo4j.execute_write(
                "MATCH (c:Chunk {id: $id}) SET c.embedding = $embedding",
                {"id": chunk.id, "embedding": embedding}
            )

        # Update document status
        self.neo4j.execute_write(
            """
            MATCH (d:Document {id: $doc_id})
            SET d.ingestion_status = 'completed',
                d.chunk_count = $chunk_count,
                d.page_count = $page_count,
                d.updated_at = datetime()
            """,
            {"doc_id": document.id, "chunk_count": len(chunks), "page_count": len(pages)},
        )

        logger.info(f"Created {len(chunks)} chunks, {len(sections)} sections, {len(pages)} pages for {file_path.name}")
        return chunks

    async def _save_document_graph(
        self,
        document: Document,
        pages: list[Page],
        page_id_map: dict[int, str],
        sections: list[Section],
        section_page_map: dict[str, set[str]],
        chunks: list[Chunk],
    ):
        """Save the document graph structure to Neo4j.

        Args:
            document: Document model.
            pages: List of Page models.
            page_id_map: Mapping from page number to page ID.
            sections: List of Section models.
            section_page_map: Mapping from section ID to set of page IDs.
            chunks: List of Chunk models.
        """
        # Save Page nodes with NEXT relationships
        sorted_pages = sorted(pages, key=lambda p: p.page_number)
        for i, page in enumerate(sorted_pages):
            self.neo4j.create_node(
                labels=["Page"],
                properties=page.to_neo4j_properties(),
            )

            # Document -[HAS_PAGE]-> Page
            self.neo4j.create_relationship(
                start_id=document.id,
                start_label="Document",
                end_id=page.id,
                end_label="Page",
                rel_type="HAS_PAGE",
            )

            # Page -[NEXT]-> Page
            if i > 0:
                self.neo4j.create_relationship(
                    start_id=sorted_pages[i - 1].id,
                    start_label="Page",
                    end_id=page.id,
                    end_label="Page",
                    rel_type="NEXT",
                )

        # Save Section nodes with relationships
        for section in sections:
            self.neo4j.create_node(
                labels=["Section"],
                properties=section.to_neo4j_properties(),
            )

            # Section -[CONTAINS]-> Section (nested sections)
            if section.parent_section_id:
                self.neo4j.create_relationship(
                    start_id=section.parent_section_id,
                    start_label="Section",
                    end_id=section.id,
                    end_label="Section",
                    rel_type="CONTAINS",
                )

            # Page -[HAS_SECTION]-> Section (for each page the section spans)
            page_ids = section_page_map.get(section.id, set())
            for page_id in page_ids:
                self.neo4j.create_relationship(
                    start_id=page_id,
                    start_label="Page",
                    end_id=section.id,
                    end_label="Section",
                    rel_type="HAS_SECTION",
                )

        # Save Chunk nodes with relationships
        chunks_by_section: dict[Optional[str], list[Chunk]] = {}
        for chunk in chunks:
            section_id = chunk.section_id
            if section_id not in chunks_by_section:
                chunks_by_section[section_id] = []
            chunks_by_section[section_id].append(chunk)

        for section_id, section_chunks in chunks_by_section.items():
            sorted_chunks = sorted(section_chunks, key=lambda c: c.sequence_order)

            for i, chunk in enumerate(sorted_chunks):
                self.neo4j.create_node(
                    labels=["Chunk"],
                    properties=chunk.to_neo4j_properties(),
                )

                # Section -[CONTAINS]-> Chunk
                if section_id:
                    self.neo4j.create_relationship(
                        start_id=section_id,
                        start_label="Section",
                        end_id=chunk.id,
                        end_label="Chunk",
                        rel_type="CONTAINS",
                    )

                # Chunk -[ON_PAGE]-> Page
                if chunk.page_id:
                    self.neo4j.create_relationship(
                        start_id=chunk.id,
                        start_label="Chunk",
                        end_id=chunk.page_id,
                        end_label="Page",
                        rel_type="ON_PAGE",
                    )

                # Chunk -[NEXT]-> Chunk (within same section)
                if i > 0:
                    self.neo4j.create_relationship(
                        start_id=sorted_chunks[i - 1].id,
                        start_label="Chunk",
                        end_id=chunk.id,
                        end_label="Chunk",
                        rel_type="NEXT",
                    )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents.

        Args:
            file_path: Path to the file.

        Returns:
            Hex-encoded hash string.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _persist_document_error(
        self, dataroom_id: str, full_path: str, error_message: str
    ):
        """Persist error information to a Document node.

        Updates existing Document node or creates a minimal one if needed.

        Args:
            dataroom_id: Data room ID.
            full_path: Full path to the document file.
            error_message: Error message to store.
        """
        try:
            # Try to update existing Document node by full_path
            result = self.neo4j.execute_write(
                """
                MATCH (d:Document {full_path: $full_path})
                WHERE d.id STARTS WITH 'doc:' + $dataroom_id + ':'
                SET d.ingestion_status = 'failed',
                    d.error_message = $error_message,
                    d.updated_at = datetime()
                RETURN d.id
                """,
                {
                    "dataroom_id": dataroom_id,
                    "full_path": full_path,
                    "error_message": error_message,
                },
            )

            if not result:
                # Document node doesn't exist yet - create a minimal one
                file_path = Path(full_path)
                doc = Document.create(
                    dataroom_id=dataroom_id,
                    folder_id="",  # Unknown at this point
                    full_path=full_path,
                    filename=file_path.name,
                    file_size=file_path.stat().st_size if file_path.exists() else 0,
                    file_type=file_path.suffix.lower(),
                    doc_type="Unknown",
                    doc_type_confidence=0.0,
                    content_hash="",
                    ingestion_status="failed",
                    error_message=error_message,
                )
                self.neo4j.create_node(
                    labels=["Document"],
                    properties=doc.to_neo4j_properties(),
                )
                logger.debug(f"Created failed Document node for {file_path.name}")
            else:
                logger.debug(f"Updated Document node with error for {full_path}")

        except Exception as e:
            # Don't let error persistence failures mask the original error
            logger.warning(f"Failed to persist document error to Neo4j: {e}")

    async def _extract_entities(
        self, dataroom_id: str, errors: list[str]
    ) -> int:
        """Extract entities from all documents in the data room.

        Args:
            dataroom_id: Data room ID.
            errors: List to append errors to.

        Returns:
            Total number of entities extracted.
        """
        if not self.entity_linker:
            return 0

        logger.info(f"Starting entity extraction for data room {dataroom_id}")

        # Get all successfully processed documents
        docs_result = self.neo4j.execute_read(
            """
            MATCH (d:Document {dataroom_id: $dataroom_id})
            WHERE d.ingestion_status = 'completed'
            RETURN d.id as doc_id, d.filename as filename
            """,
            {"dataroom_id": dataroom_id},
        )

        total_docs = len(docs_result) if docs_result else 0
        self._update_progress(
            current_file="Extracting entities...",
            total_documents_for_extraction=total_docs,
            extracted_documents=0,
            entities_found=0,
        )

        total_entities = 0
        for i, doc in enumerate(docs_result or []):
            doc_id = doc["doc_id"]
            filename = doc["filename"]

            try:
                self._update_progress(
                    current_file=f"Extracting entities: {filename} ({i+1}/{total_docs})",
                    extracted_documents=i,
                )
                entities_count = self.entity_linker.process_document_chunks(
                    doc_id, dataroom_id
                )
                total_entities += entities_count
                self._update_progress(
                    entities_found=total_entities,
                    extracted_documents=i + 1,
                )
                logger.info(f"Extracted {entities_count} entities from {filename}")

            except Exception as e:
                error_msg = f"Entity extraction failed for {filename}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                self._update_progress(extracted_documents=i + 1)

        # Create entity relationships based on co-occurrence
        if total_entities > 0:
            try:
                self._update_progress(current_file="Creating entity relationships...")
                self.entity_linker.create_entity_relationships(dataroom_id)
            except Exception as e:
                logger.error(f"Failed to create entity relationships: {e}")

        logger.info(f"Entity extraction complete: {total_entities} total entities")
        return total_entities

    async def _update_dataroom_stats(self, dataroom_id: str):
        """Update data room statistics in Neo4j.

        Args:
            dataroom_id: Data room ID.
        """
        self.neo4j.execute_write(
            """
            MATCH (dr:DataRoom {id: $dataroom_id})
            OPTIONAL MATCH (d:Document {dataroom_id: $dataroom_id})
            OPTIONAL MATCH (c:Chunk {dataroom_id: $dataroom_id})
            OPTIONAL MATCH (e:Entity {dataroom_id: $dataroom_id})
            WITH dr, count(DISTINCT d) as doc_count, count(DISTINCT c) as chunk_count, count(DISTINCT e) as entity_count
            SET dr.document_count = doc_count,
                dr.chunk_count = chunk_count,
                dr.entity_count = entity_count,
                dr.ingestion_status = 'completed',
                dr.updated_at = datetime()
            """,
            {"dataroom_id": dataroom_id},
        )

    async def extract_entities_for_dataroom(self, dataroom_id: str) -> int:
        """Run entity extraction on all completed documents in a data room.

        Can be called independently after document ingestion completes.

        Args:
            dataroom_id: Data room ID.

        Returns:
            Number of entities extracted.
        """
        # Ensure entity linker is available
        if not self.entity_linker:
            self.entity_linker = EntityLinker(self.neo4j)

        errors: list[str] = []
        self._update_progress(status="extracting_entities")

        entities_extracted = await self._extract_entities(dataroom_id, errors)

        # Update data room stats
        await self._update_dataroom_stats(dataroom_id)

        self._update_progress(status="completed", end_time=datetime.utcnow())

        if errors:
            logger.warning(f"Entity extraction completed with {len(errors)} errors")

        return entities_extracted

    def shutdown(self):
        """Shutdown the pipeline and cleanup resources."""
        self.unstructured.shutdown()


def create_pipeline(
    pipeline_type: PipelineType = "legacy",
    neo4j_client: Optional[Neo4jClient] = None,
    extract_entities: bool = False,
    generate_narratives: bool = True,
    **kwargs,
) -> Union["IngestionPipeline", "GraphRAGPipeline"]:
    """Factory function to create an ingestion pipeline.

    This function provides a unified way to create either the legacy
    ingestion pipeline or the new GraphRAG-based pipeline.

    Args:
        pipeline_type: Type of pipeline to create:
            - "legacy": Original pipeline with Unstructured.io + custom entity extraction
            - "graphrag": New pipeline using neo4j-graphrag library
        neo4j_client: Neo4j client instance.
        extract_entities: Whether to extract entities (legacy only).
        generate_narratives: Whether to generate folder/section narratives (legacy only).
        **kwargs: Additional arguments passed to the pipeline constructor.

    Returns:
        Either IngestionPipeline or GraphRAGPipeline instance.

    Raises:
        ValueError: If pipeline_type is not recognized.
    """
    if pipeline_type == "legacy":
        return IngestionPipeline(
            neo4j_client=neo4j_client,
            extract_entities=extract_entities,
            generate_narratives=generate_narratives,
            **kwargs,
        )
    elif pipeline_type == "graphrag":
        # Import here to avoid circular imports and make graphrag optional
        from .graphrag_pipeline import GraphRAGPipeline

        return GraphRAGPipeline(
            neo4j_client=neo4j_client,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Use 'legacy' or 'graphrag'.")
