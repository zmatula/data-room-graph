"""Main ingestion pipeline orchestrator."""

import logging
import asyncio
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field

from .unstructured import UnstructuredClient
from .chunker import HierarchyBuilder
from .classifier import DocumentClassifier
from .embedder import Embedder, get_embedder
from ..database.neo4j_client import Neo4jClient, get_neo4j_client
from ..database.models import DataRoom, Folder, Document, Chunk
from ..extraction.linker import EntityLinker
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionProgress:
    """Tracks ingestion progress."""

    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def file_progress(self) -> float:
        """Get file processing progress (0-1)."""
        return self.processed_files / self.total_files if self.total_files > 0 else 0

    @property
    def chunk_progress(self) -> float:
        """Get chunk processing progress (0-1)."""
        return self.processed_chunks / self.total_chunks if self.total_chunks > 0 else 0


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
    ):
        """Initialize the pipeline.

        Args:
            neo4j_client: Neo4j client. Defaults to global instance.
            unstructured_client: Unstructured.io client.
            embedder: Embedding generator.
            max_concurrent_files: Max files to process in parallel. Defaults to settings.
            extract_entities: Whether to extract entities after document processing.
        """
        settings = get_settings()
        self.neo4j = neo4j_client or get_neo4j_client()
        self.unstructured = unstructured_client or UnstructuredClient(
            max_workers=max_concurrent_files or settings.max_concurrent_files
        )
        self.embedder = embedder or get_embedder()
        self.classifier = DocumentClassifier()
        self.chunker = HierarchyBuilder()
        self.max_concurrent_files = max_concurrent_files or settings.max_concurrent_files
        self.extract_entities = extract_entities
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

        # Create chunks
        chunks = self.chunker.create_chunks(elements, dataroom_id, document.id)

        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_texts(chunk_texts, show_progress=True)

        # Update chunks with embeddings and save to Neo4j
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

            self.neo4j.create_node(
                labels=["Chunk"],
                properties=chunk.to_neo4j_properties(),
            )

            # Create relationship to document
            self.neo4j.create_relationship(
                start_id=document.id,
                start_label="Document",
                end_id=chunk.id,
                end_label="Chunk",
                rel_type="HAS_ROOT" if not chunk.parent_chunk_id else "CONTAINS",
            )

            # Create parent-child relationship
            if chunk.parent_chunk_id:
                self.neo4j.create_relationship(
                    start_id=chunk.parent_chunk_id,
                    start_label="Chunk",
                    end_id=chunk.id,
                    end_label="Chunk",
                    rel_type="CONTAINS",
                )

        # Create NEXT relationships between sibling chunks
        sorted_chunks = sorted(chunks, key=lambda c: c.sequence_order)
        for i in range(len(sorted_chunks) - 1):
            current = sorted_chunks[i]
            next_chunk = sorted_chunks[i + 1]
            if current.parent_chunk_id == next_chunk.parent_chunk_id:
                self.neo4j.create_relationship(
                    start_id=current.id,
                    start_label="Chunk",
                    end_id=next_chunk.id,
                    end_label="Chunk",
                    rel_type="NEXT",
                )

        # Update document status
        self.neo4j.execute_write(
            """
            MATCH (d:Document {id: $doc_id})
            SET d.ingestion_status = 'completed',
                d.chunk_count = $chunk_count,
                d.updated_at = datetime()
            """,
            {"doc_id": document.id, "chunk_count": len(chunks)},
        )

        logger.info(f"Created {len(chunks)} chunks for {file_path.name}")
        return chunks

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
        self._update_progress(current_file="Extracting entities...")

        # Get all successfully processed documents
        docs_result = self.neo4j.execute_read(
            """
            MATCH (d:Document {dataroom_id: $dataroom_id})
            WHERE d.ingestion_status = 'completed'
            RETURN d.id as doc_id, d.filename as filename
            """,
            {"dataroom_id": dataroom_id},
        )

        total_entities = 0
        for i, doc in enumerate(docs_result or []):
            doc_id = doc["doc_id"]
            filename = doc["filename"]

            try:
                self._update_progress(
                    current_file=f"Extracting entities: {filename} ({i+1}/{len(docs_result)})"
                )
                entities_count = self.entity_linker.process_document_chunks(
                    doc_id, dataroom_id
                )
                total_entities += entities_count
                logger.info(f"Extracted {entities_count} entities from {filename}")

            except Exception as e:
                error_msg = f"Entity extraction failed for {filename}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

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
