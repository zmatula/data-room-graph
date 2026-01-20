"""Neo4j GraphRAG pipeline wrapper.

This module provides the integration layer between the preprocessing output
and neo4j-graphrag's SimpleKGPipeline. It handles:
1. Custom text splitter that preserves metadata
2. Pipeline configuration with PE schema and prompt
3. Results transformation to IngestionResult format
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime

from neo4j import Driver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunks, TextChunk
from neo4j_graphrag.llm import AnthropicLLM

from .pe_schema import PE_SCHEMA, LEXICAL_GRAPH_CONFIG
from .pe_prompt import get_extraction_prompt, get_extraction_examples
from .preprocessing import (
    DocumentPreprocessor,
    PreprocessedDocument,
    extract_metadata_for_graphrag,
)
from .cross_type_resolver import CrossTypeEntityResolver
from .validation import ExtractionValidator, ValidationResult
from ..config import get_settings
from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGIngestionProgress:
    """Progress tracking for GraphRAG ingestion."""

    total_documents: int = 0
    processed_documents: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, preprocessing, extracting, resolving, completed, failed
    error_message: str = ""
    entities_extracted: int = 0
    relationships_created: int = 0

    @property
    def progress(self) -> float:
        """Get overall progress (0-1)."""
        if self.total_documents == 0:
            return 0
        return self.processed_documents / self.total_documents


@dataclass
class GraphRAGIngestionResult:
    """Result of GraphRAG ingestion."""

    success: bool
    dataroom_id: str
    documents_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0


class MetadataPreservingSplitter(TextSplitter):
    """Custom text splitter that preserves chunk metadata from preprocessing.

    This splitter doesn't actually split text - it uses the pre-split chunks
    from the preprocessing phase and attaches their metadata.
    """

    def __init__(self):
        """Initialize the splitter."""
        self._chunk_texts: list[str] = []
        self._metadata_map: dict[int, dict] = {}

    def set_chunks(
        self,
        chunk_texts: list[str],
        metadata_map: dict[int, dict],
    ):
        """Set the pre-split chunks and their metadata.

        Args:
            chunk_texts: List of chunk text strings.
            metadata_map: Mapping from chunk index to metadata dict.
        """
        self._chunk_texts = chunk_texts
        self._metadata_map = metadata_map

    async def run(self, text: str) -> TextChunks:
        """Return pre-split chunks with metadata.

        Args:
            text: Full text (ignored, we use pre-split chunks).

        Returns:
            TextChunks with preserved metadata.
        """
        chunks = []
        for i, chunk_text in enumerate(self._chunk_texts):
            metadata = self._metadata_map.get(i, {})
            chunk = TextChunk(
                text=chunk_text,
                index=i,
                metadata=metadata,
            )
            chunks.append(chunk)

        return TextChunks(chunks=chunks)


class GraphRAGPipeline:
    """Wrapper around neo4j-graphrag's SimpleKGPipeline.

    This class integrates the preprocessing layer with neo4j-graphrag,
    providing a unified interface for document ingestion.
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        preprocessor: Optional[DocumentPreprocessor] = None,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "claude-sonnet-4-20250514",
        perform_entity_resolution: bool = True,
        perform_cross_type_resolution: bool = True,
        perform_validation: bool = True,
        auto_fix_validation_issues: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            neo4j_client: Neo4j client for database operations.
            preprocessor: Document preprocessor.
            embedding_model: OpenAI embedding model name.
            llm_model: Anthropic LLM model name for extraction.
            perform_entity_resolution: Whether to run entity resolution.
            perform_cross_type_resolution: Whether to run cross-type entity resolution.
            perform_validation: Whether to run extraction validation.
            auto_fix_validation_issues: Whether to auto-fix high-confidence validation issues.
        """
        settings = get_settings()
        self.neo4j = neo4j_client or get_neo4j_client()
        self.preprocessor = preprocessor or DocumentPreprocessor()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.perform_entity_resolution = perform_entity_resolution
        self.perform_cross_type_resolution = perform_cross_type_resolution
        self.perform_validation = perform_validation
        self.auto_fix_validation_issues = auto_fix_validation_issues

        # Post-processing components
        self.cross_type_resolver = CrossTypeEntityResolver(neo4j_client=self.neo4j)
        self.validator = ExtractionValidator(neo4j_client=self.neo4j)

        # Progress tracking
        self._progress = GraphRAGIngestionProgress()
        self._progress_callback: Optional[Callable[[GraphRAGIngestionProgress], None]] = None

        # Initialize neo4j-graphrag components lazily
        self._kg_builder: Optional[SimpleKGPipeline] = None
        self._splitter = MetadataPreservingSplitter()

    @property
    def progress(self) -> GraphRAGIngestionProgress:
        """Get current progress."""
        return self._progress

    def set_progress_callback(
        self,
        callback: Callable[[GraphRAGIngestionProgress], None],
    ):
        """Set callback for progress updates.

        Args:
            callback: Function called with progress updates.
        """
        self._progress_callback = callback

    def _update_progress(self, **kwargs):
        """Update progress and notify callback."""
        for key, value in kwargs.items():
            setattr(self._progress, key, value)
        if self._progress_callback:
            self._progress_callback(self._progress)

    def _get_kg_builder(self) -> SimpleKGPipeline:
        """Get or create the SimpleKGPipeline.

        Returns:
            Configured SimpleKGPipeline instance.
        """
        if self._kg_builder is not None:
            return self._kg_builder

        settings = get_settings()

        # Initialize LLM
        llm = AnthropicLLM(
            model_name=self.llm_model,
            model_params={"max_tokens": 4000},
            api_key=settings.anthropic_api_key,
        )

        # Initialize embedder
        embedder = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Get extraction prompt with examples
        prompt_template = get_extraction_prompt(include_examples=True)
        examples = get_extraction_examples()

        # Create the pipeline
        self._kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=self.neo4j.driver,
            embedder=embedder,
            text_splitter=self._splitter,
            schema=PE_SCHEMA,
            prompt_template=prompt_template,
            from_pdf=False,  # We preprocess with Unstructured
            perform_entity_resolution=self.perform_entity_resolution,
            on_error="IGNORE",
            lexical_graph_config=LEXICAL_GRAPH_CONFIG,
        )

        return self._kg_builder

    async def ingest_document(
        self,
        dataroom_id: str,
        file_path: Path,
        folder_id: Optional[str] = None,
    ) -> GraphRAGIngestionResult:
        """Ingest a single document using GraphRAG pipeline.

        Args:
            dataroom_id: ID of the target data room.
            file_path: Path to the document file.
            folder_id: Optional folder ID for the document.

        Returns:
            GraphRAGIngestionResult with statistics.
        """
        start_time = datetime.utcnow()
        self._progress = GraphRAGIngestionProgress(
            total_documents=1,
            status="preprocessing",
            current_file=file_path.name,
        )
        errors = []

        try:
            # Phase 1: Preprocess document
            self._update_progress(status="preprocessing")
            preprocessed = await self.preprocessor.preprocess(file_path)

            if not preprocessed.chunk_texts:
                logger.warning(f"No chunks extracted from {file_path.name}")
                return GraphRAGIngestionResult(
                    success=True,
                    dataroom_id=dataroom_id,
                    documents_processed=1,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                )

            # Phase 2: Configure splitter with preprocessed chunks
            metadata_map = extract_metadata_for_graphrag(preprocessed)
            self._splitter.set_chunks(preprocessed.chunk_texts, metadata_map)

            # Phase 3: Run neo4j-graphrag pipeline
            self._update_progress(status="extracting")
            kg_builder = self._get_kg_builder()

            # Run the pipeline on the full text
            # The splitter will return our pre-processed chunks
            result = await kg_builder.run_async(text=preprocessed.full_text)

            # Phase 4: Post-process - add dataroom context
            self._update_progress(status="resolving")
            await self._add_dataroom_context(
                dataroom_id,
                preprocessed,
                folder_id,
            )

            self._update_progress(
                status="completed",
                processed_documents=1,
                entities_extracted=result.get("entities", 0) if result else 0,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            return GraphRAGIngestionResult(
                success=True,
                dataroom_id=dataroom_id,
                documents_processed=1,
                chunks_created=preprocessed.chunk_count,
                entities_extracted=result.get("entities", 0) if result else 0,
                relationships_created=result.get("relationships", 0) if result else 0,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.exception(f"GraphRAG ingestion failed for {file_path}: {e}")
            self._update_progress(
                status="failed",
                error_message=str(e),
            )
            return GraphRAGIngestionResult(
                success=False,
                dataroom_id=dataroom_id,
                errors=[str(e)],
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def ingest_folder(
        self,
        dataroom_id: str,
        folder_path: Path,
        recursive: bool = True,
    ) -> GraphRAGIngestionResult:
        """Ingest all documents from a folder using GraphRAG pipeline.

        Args:
            dataroom_id: ID of the target data room.
            folder_path: Path to the folder to ingest.
            recursive: Whether to process subfolders.

        Returns:
            GraphRAGIngestionResult with statistics.
        """
        start_time = datetime.utcnow()

        # Discover files
        files = self._discover_files(folder_path, recursive)
        self._progress = GraphRAGIngestionProgress(
            total_documents=len(files),
            status="preprocessing",
        )

        if not files:
            return GraphRAGIngestionResult(
                success=True,
                dataroom_id=dataroom_id,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

        total_chunks = 0
        total_entities = 0
        total_relationships = 0
        errors = []

        for i, file_path in enumerate(files):
            self._update_progress(
                current_file=file_path.name,
                processed_documents=i,
            )

            try:
                result = await self.ingest_document(
                    dataroom_id,
                    file_path,
                    folder_id=None,  # TODO: Compute folder ID
                )

                total_chunks += result.chunks_created
                total_entities += result.entities_extracted
                total_relationships += result.relationships_created
                errors.extend(result.errors)

            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Post-processing: Cross-type entity resolution
        if self.perform_cross_type_resolution and total_entities > 0:
            self._update_progress(
                status="resolving",
                current_file="Running cross-type entity resolution...",
            )
            try:
                matches = await self.cross_type_resolver.resolve(dataroom_id)
                logger.info(f"Cross-type resolution merged {len(matches)} duplicate entities")
            except Exception as e:
                error_msg = f"Cross-type resolution failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Post-processing: Extraction validation
        validation_result: Optional[ValidationResult] = None
        if self.perform_validation and total_entities > 0:
            self._update_progress(
                status="validating",
                current_file="Running extraction validation...",
            )
            try:
                validation_result = await self.validator.validate_dataroom(
                    dataroom_id,
                    auto_fix=self.auto_fix_validation_issues,
                )
                logger.info(
                    f"Validation complete: {validation_result.issues_found} issues, "
                    f"{validation_result.issues_fixed} fixed"
                )
            except Exception as e:
                error_msg = f"Extraction validation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        self._update_progress(
            status="completed",
            processed_documents=len(files),
        )

        duration = (datetime.utcnow() - start_time).total_seconds()
        return GraphRAGIngestionResult(
            success=len(errors) == 0,
            dataroom_id=dataroom_id,
            documents_processed=len(files) - len(errors),
            chunks_created=total_chunks,
            entities_extracted=total_entities,
            relationships_created=total_relationships,
            errors=errors,
            duration_seconds=duration,
        )

    def _discover_files(
        self,
        folder_path: Path,
        recursive: bool,
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
            if file_path.is_file() and self.preprocessor.unstructured.is_supported(file_path):
                files.append(file_path)

        return sorted(files)

    async def _add_dataroom_context(
        self,
        dataroom_id: str,
        preprocessed: PreprocessedDocument,
        folder_id: Optional[str],
    ):
        """Add dataroom context to created nodes.

        This post-processing step adds dataroom_id and other context
        to the nodes created by neo4j-graphrag.

        Args:
            dataroom_id: Data room ID.
            preprocessed: Preprocessed document.
            folder_id: Optional folder ID.
        """
        # Update Document nodes with dataroom context
        update_doc_query = """
        MATCH (d:Document)
        WHERE d.path = $file_path OR d.text CONTAINS $filename
        SET d.dataroom_id = $dataroom_id,
            d.folder_id = $folder_id,
            d.doc_type = $doc_type,
            d.doc_type_confidence = $confidence
        """

        self.neo4j.execute_write(
            update_doc_query,
            {
                "file_path": str(preprocessed.file_path),
                "filename": preprocessed.file_path.name,
                "dataroom_id": dataroom_id,
                "folder_id": folder_id or "",
                "doc_type": preprocessed.doc_type.value,
                "confidence": preprocessed.doc_type_confidence,
            },
        )

        # Update Chunk nodes with dataroom context
        update_chunk_query = """
        MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document {dataroom_id: $dataroom_id})
        SET c.dataroom_id = $dataroom_id
        """

        self.neo4j.execute_write(
            update_chunk_query,
            {"dataroom_id": dataroom_id},
        )

        # Update Entity nodes with dataroom context
        update_entity_query = """
        MATCH (e:Entity)
        WHERE NOT EXISTS(e.dataroom_id)
        MATCH (e)<-[:MENTIONS]-(c:Chunk {dataroom_id: $dataroom_id})
        SET e.dataroom_id = $dataroom_id
        """

        self.neo4j.execute_write(
            update_entity_query,
            {"dataroom_id": dataroom_id},
        )

    def shutdown(self):
        """Shutdown pipeline resources."""
        self.preprocessor.shutdown()
