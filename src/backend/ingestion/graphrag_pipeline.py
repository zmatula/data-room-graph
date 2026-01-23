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
from neo4j_graphrag.llm import OpenAILLM

from .pe_schema import PE_SCHEMA, LEXICAL_GRAPH_CONFIG
from .pe_prompt import get_extraction_prompt, get_extraction_examples
from .preprocessing import (
    DocumentPreprocessor,
    PreprocessedDocument,
    extract_metadata_for_graphrag,
)
from .validation import ExtractionValidator, ValidationResult
from .entity_consolidator import EntityConsolidator, ConsolidationResult
from .relationship_extractor import RelationshipExtractor, RelationshipExtractionResult
from .vlm_extractor import VLMEntityExtractor, VLMExtractionResult
from ..config import get_settings
from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGIngestionProgress:
    """Progress tracking for GraphRAG ingestion."""

    total_documents: int = 0
    processed_documents: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, preprocessing, extracting, consolidating, validating, completed, failed
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
        llm_model: str = "gpt-4.1",
        perform_entity_resolution: bool = True,
        perform_entity_consolidation: bool = True,
        consolidation_model: str = "claude-opus-4-20250514",
        perform_validation: bool = True,
        auto_fix_validation_issues: bool = False,
        use_vlm: bool = False,
        vlm_model: str = "gpt-4o",
        parsing_strategy: str = "hi_res",
    ):
        """Initialize the pipeline.

        Args:
            neo4j_client: Neo4j client for database operations.
            preprocessor: Document preprocessor.
            embedding_model: OpenAI embedding model name.
            llm_model: OpenAI LLM model name for extraction.
            perform_entity_resolution: Whether to run entity resolution (neo4j-graphrag built-in).
            perform_entity_consolidation: Whether to run LLM-based entity consolidation (second pass).
            consolidation_model: Claude model to use for entity consolidation.
            perform_validation: Whether to run extraction validation.
            auto_fix_validation_issues: Whether to auto-fix high-confidence validation issues.
            use_vlm: Whether to use Vision Language Model for document parsing (better for logos/images).
            vlm_model: VLM model to use (default: 'gpt-4o').
            parsing_strategy: Unstructured.io parsing strategy ('hi_res', 'fast', 'ocr_only', 'auto', 'vlm').
        """
        settings = get_settings()
        self.neo4j = neo4j_client or get_neo4j_client()
        self.preprocessor = preprocessor or DocumentPreprocessor()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.perform_entity_resolution = perform_entity_resolution
        self.perform_entity_consolidation = perform_entity_consolidation
        self.consolidation_model = consolidation_model
        self.perform_validation = perform_validation
        self.auto_fix_validation_issues = auto_fix_validation_issues
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.parsing_strategy = parsing_strategy

        # Post-processing components
        self.consolidator = EntityConsolidator(
            neo4j_client=self.neo4j,
            model=consolidation_model,
        )
        self.relationship_extractor = RelationshipExtractor(
            neo4j_client=self.neo4j,
            model="claude-sonnet-4-20250514",
        )
        self.vlm_extractor = VLMEntityExtractor(
            model=vlm_model,
            max_pages=20,
        )
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
        # Using OpenAI with response_format for guaranteed JSON output,
        # which eliminates format errors from entity extraction
        llm = OpenAILLM(
            model_name=self.llm_model,
            model_params={
                "max_tokens": 4000,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            api_key=settings.openai_api_key,
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
            preprocessed = await self.preprocessor.preprocess(
                file_path,
                strategy=self.parsing_strategy,
                use_vlm=self.use_vlm,
                vlm_model=self.vlm_model,
            )

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

            # Extract counts from PipelineResult if available
            # The result is a Pydantic model, not a dict
            entities_count = 0
            relationships_count = 0
            if result:
                # Try to access run_result which contains the actual data
                if hasattr(result, 'run_result') and result.run_result:
                    run_result = result.run_result
                    if hasattr(run_result, 'get'):
                        entities_count = run_result.get("entities", 0)
                        relationships_count = run_result.get("relationships", 0)
                    elif hasattr(run_result, 'entities'):
                        entities_count = getattr(run_result, 'entities', 0) or 0
                        relationships_count = getattr(run_result, 'relationships', 0) or 0

            self._update_progress(
                status="completed",
                processed_documents=1,
                entities_extracted=entities_count,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            return GraphRAGIngestionResult(
                success=True,
                dataroom_id=dataroom_id,
                documents_processed=1,
                chunks_created=preprocessed.chunk_count,
                entities_extracted=entities_count,
                relationships_created=relationships_count,
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

        # Query actual entity count from Neo4j (since neo4j-graphrag doesn't return counts reliably)
        actual_entity_count = self._count_entities(dataroom_id)
        logger.info(f"Entities extracted for dataroom {dataroom_id}: {actual_entity_count}")

        # Post-processing: VLM image extraction for PDFs (extracts text from logos/graphics)
        vlm_result: Optional[VLMExtractionResult] = None
        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        if pdf_files:
            self._update_progress(
                status="vlm_extraction",
                current_file="Running VLM image extraction...",
            )
            logger.info(f"=== STARTING VLM EXTRACTION for {len(pdf_files)} PDFs ===")
            try:
                for pdf_file in pdf_files:
                    logger.info(f"VLM processing: {pdf_file.name}")
                    vlm_result = self.vlm_extractor.extract_from_pdf(pdf_file)
                    if vlm_result.entities_found > 0:
                        # Inject VLM-extracted entities into Neo4j
                        await self._inject_vlm_entities(dataroom_id, vlm_result)
                        logger.info(
                            f"VLM extraction from {pdf_file.name}: {vlm_result.entities_found} entities found"
                        )
                    else:
                        logger.info(f"VLM extraction from {pdf_file.name}: no entities found")
                # Update entity count after VLM extraction
                actual_entity_count = self._count_entities(dataroom_id)
                logger.info(f"=== VLM EXTRACTION COMPLETE === Total entities after VLM: {actual_entity_count}")
            except Exception as e:
                error_msg = f"VLM extraction failed: {e}"
                logger.exception(error_msg)
                errors.append(error_msg)

        # Post-processing: LLM-based entity consolidation (second pass)
        consolidation_result: Optional[ConsolidationResult] = None
        if self.perform_entity_consolidation and actual_entity_count > 0:
            self._update_progress(
                status="consolidating",
                current_file="Running LLM entity consolidation...",
            )
            try:
                consolidation_result = await self.consolidator.consolidate(dataroom_id)
                logger.info(
                    f"Consolidation complete: reviewed {consolidation_result.entities_reviewed} entities, "
                    f"merged {consolidation_result.entities_merged} duplicates"
                )
            except Exception as e:
                error_msg = f"Entity consolidation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Post-processing: Second-pass LLM relationship extraction
        relationship_result: Optional[RelationshipExtractionResult] = None
        if actual_entity_count > 0:
            self._update_progress(
                status="extracting_relationships",
                current_file="Running LLM relationship extraction...",
            )
            try:
                relationship_result = await self.relationship_extractor.extract_relationships(dataroom_id)
                total_relationships += relationship_result.relationships_created
                logger.info(
                    f"Relationship extraction complete: {relationship_result.chunks_analyzed} chunks analyzed, "
                    f"{relationship_result.relationships_created} relationships created"
                )
            except Exception as e:
                error_msg = f"Relationship extraction failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Post-processing: Extraction validation
        validation_result: Optional[ValidationResult] = None
        if self.perform_validation and actual_entity_count > 0:
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

    def _count_entities(self, dataroom_id: str) -> int:
        """Count entities in a dataroom.

        Args:
            dataroom_id: Dataroom ID to count entities for.

        Returns:
            Number of entities (Vehicle, Company, Person).
        """
        query = """
        MATCH (e)
        WHERE e.dataroom_id = $dataroom_id
          AND (e:Vehicle OR e:Company OR e:Person)
        RETURN count(e) AS count
        """
        result = self.neo4j.execute_read(query, {"dataroom_id": dataroom_id})
        return result[0]["count"] if result else 0

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
        to the nodes created by neo4j-graphrag, and creates the
        DataRoom -> Document -> Chunk hierarchy.

        Args:
            dataroom_id: Data room ID.
            preprocessed: Preprocessed document.
            folder_id: Optional folder ID.
        """
        # Ensure DataRoom node exists
        ensure_dataroom_query = """
        MERGE (dr:DataRoom {id: $dataroom_id})
        ON CREATE SET dr.created_at = datetime()
        SET dr.updated_at = datetime()
        """

        self.neo4j.execute_write(
            ensure_dataroom_query,
            {"dataroom_id": dataroom_id},
        )

        # Update Document nodes with dataroom context and link to DataRoom
        # neo4j-graphrag stores documents with path="document.txt" for inline text,
        # so we match documents without dataroom_id that have chunks related to them.
        update_doc_query = """
        MATCH (d:Document)
        WHERE d.dataroom_id IS NULL
        SET d.dataroom_id = $dataroom_id,
            d.doc_type = $doc_type,
            d.doc_type_confidence = $confidence,
            d.original_path = $file_path,
            d.filename = $filename
        WITH d
        MATCH (dr:DataRoom {id: $dataroom_id})
        MERGE (dr)-[:CONTAINS_DOCUMENT]->(d)
        """

        self.neo4j.execute_write(
            update_doc_query,
            {
                "file_path": str(preprocessed.file_path),
                "filename": preprocessed.file_path.name,
                "dataroom_id": dataroom_id,
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
        # Entity types: Vehicle, Company, Person
        # neo4j-graphrag uses MENTIONED_IN relationship (entity)-[:MENTIONED_IN]->(chunk)
        update_entity_query = """
        MATCH (e)
        WHERE e.dataroom_id IS NULL
          AND (e:Vehicle OR e:Company OR e:Person)
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk {dataroom_id: $dataroom_id})
        SET e.dataroom_id = $dataroom_id
        """

        self.neo4j.execute_write(
            update_entity_query,
            {"dataroom_id": dataroom_id},
        )

    async def _inject_vlm_entities(
        self,
        dataroom_id: str,
        vlm_result: VLMExtractionResult,
    ):
        """Inject VLM-extracted entities into Neo4j.

        This creates entities that were found in images but missed by text extraction.
        The entities will be marked with source='vlm_extraction' for tracking.

        Args:
            dataroom_id: Dataroom ID.
            vlm_result: VLM extraction result with entities.
        """
        for entity in vlm_result.entities:
            # Check if entity already exists (case-insensitive)
            check_query = """
            MATCH (e)
            WHERE e.dataroom_id = $dataroom_id
              AND (e:Vehicle OR e:Company OR e:Person)
              AND toLower(e.name) = toLower($name)
            RETURN count(e) AS count
            """
            result = self.neo4j.execute_read(check_query, {
                "dataroom_id": dataroom_id,
                "name": entity.name,
            })

            if result and result[0]["count"] > 0:
                logger.debug(f"VLM entity already exists: {entity.name}")
                continue

            # Determine entity label and properties based on type
            if entity.entity_type == "fund":
                label = "Vehicle"
                properties = {
                    "name": entity.name,
                    "dataroom_id": dataroom_id,
                    "vehicle_type": "Fund",
                    "is_closed_end_fund": True,
                    "private_or_public": "PRIVATE",
                    "extraction_source": "vlm_extraction",
                }
            elif entity.entity_type == "person":
                label = "Person"
                properties = {
                    "name": entity.name,
                    "dataroom_id": dataroom_id,
                    "extraction_source": "vlm_extraction",
                }
            else:  # company (default)
                label = "Company"
                # Determine if portfolio company based on context
                is_portfolio = (
                    "portfolio" in entity.source.lower() or
                    "logo" in entity.source.lower() or
                    "graphic" in entity.source.lower()
                )
                properties = {
                    "name": entity.name,
                    "dataroom_id": dataroom_id,
                    "is_portfolio_company": is_portfolio,
                    "extraction_source": "vlm_extraction",
                }

            # Create the entity with neo4j-graphrag labels for visualization compatibility
            # __Entity__ and __KGBuilder__ labels are required for the graph visualization
            create_query = f"""
            CREATE (e:{label}:__Entity__:__KGBuilder__ $properties)
            RETURN elementId(e) AS id
            """
            self.neo4j.execute_write(create_query, {"properties": properties})
            logger.info(f"Created VLM entity: {entity.name} ({label})")

            # For portfolio companies, create INVESTS_IN relationship from fund
            if is_portfolio:
                await self._create_portfolio_relationship(dataroom_id, entity.name)

    async def _create_portfolio_relationship(
        self,
        dataroom_id: str,
        company_name: str,
    ):
        """Create INVESTS_IN relationship from fund to portfolio company.

        This uses LLM to determine if a portfolio company should be linked to a fund.
        It finds any Vehicle in the dataroom and creates the relationship.

        Args:
            dataroom_id: Dataroom ID.
            company_name: Name of the portfolio company.
        """
        # Find funds (Vehicles) in this dataroom
        find_funds_query = """
        MATCH (v:Vehicle)
        WHERE v.dataroom_id = $dataroom_id
        RETURN v.name AS name
        """
        funds = self.neo4j.execute_read(find_funds_query, {"dataroom_id": dataroom_id})

        if not funds:
            logger.debug(f"No funds found for portfolio relationship: {company_name}")
            return

        # Create INVESTS_IN relationship from each fund to this portfolio company
        for fund in funds:
            create_rel_query = """
            MATCH (v:Vehicle)
            WHERE v.dataroom_id = $dataroom_id AND v.name = $fund_name
            MATCH (c:Company)
            WHERE c.dataroom_id = $dataroom_id AND c.name = $company_name
            MERGE (v)-[r:INVESTS_IN]->(c)
            SET r.extraction_source = 'vlm_portfolio_inference'
            RETURN v.name AS fund, c.name AS company
            """
            result = self.neo4j.execute_write(create_rel_query, {
                "dataroom_id": dataroom_id,
                "fund_name": fund["name"],
                "company_name": company_name,
            })
            if result:
                logger.info(f"Created INVESTS_IN: {fund['name']} -> {company_name}")

    def shutdown(self):
        """Shutdown pipeline resources."""
        self.preprocessor.shutdown()
