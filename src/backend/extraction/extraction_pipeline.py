"""Standalone pipeline for extracting entities from ingested documents."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from .linker import EntityLinker
from ..database.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class ExtractionProgress:
    """Tracks entity extraction progress."""

    total_documents: int = 0
    processed_documents: int = 0
    total_entities: int = 0
    current_document: str = ""
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def progress(self) -> float:
        """Get processing progress (0-1)."""
        if self.total_documents == 0:
            return 0
        return self.processed_documents / self.total_documents


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""

    success: bool
    dataroom_id: str
    documents_processed: int = 0
    entities_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0


class EntityExtractionPipeline:
    """Standalone pipeline for extracting entities from ingested documents.

    This pipeline can be run independently after document ingestion completes,
    allowing users to separate the document ingestion and entity extraction steps.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize the extraction pipeline.

        Args:
            neo4j_client: Neo4j client. Defaults to global instance.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.linker = EntityLinker(self.neo4j)

        self._progress = ExtractionProgress()
        self._progress_callback: Optional[Callable[[ExtractionProgress], None]] = None
        self._cancelled = False

    @property
    def progress(self) -> ExtractionProgress:
        """Get current progress."""
        return self._progress

    def set_progress_callback(
        self, callback: Callable[[ExtractionProgress], None]
    ):
        """Set callback for progress updates.

        Args:
            callback: Function called with ExtractionProgress on updates.
        """
        self._progress_callback = callback

    def cancel(self):
        """Request cancellation of extraction."""
        self._cancelled = True
        logger.info("Entity extraction cancellation requested")

    def _update_progress(self, **kwargs):
        """Update progress and notify callback."""
        for key, value in kwargs.items():
            setattr(self._progress, key, value)
        if self._progress_callback:
            self._progress_callback(self._progress)

    async def extract_for_dataroom(
        self,
        dataroom_id: str,
    ) -> ExtractionResult:
        """Extract entities from all documents in a data room.

        Args:
            dataroom_id: Data room ID.

        Returns:
            ExtractionResult with statistics.
        """
        start_time = datetime.utcnow()
        self._cancelled = False
        self._progress = ExtractionProgress(
            status="running",
            start_time=start_time,
        )
        errors: list[str] = []

        try:
            # Get all successfully processed documents
            docs_result = self.neo4j.execute_read(
                """
                MATCH (d:Document {dataroom_id: $dataroom_id})
                WHERE d.ingestion_status = 'completed'
                RETURN d.id as doc_id, d.filename as filename
                ORDER BY d.filename
                """,
                {"dataroom_id": dataroom_id},
            )

            total_docs = len(docs_result) if docs_result else 0
            self._update_progress(total_documents=total_docs)

            if total_docs == 0:
                logger.warning(f"No completed documents found in data room {dataroom_id}")
                return ExtractionResult(
                    success=True,
                    dataroom_id=dataroom_id,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                )

            logger.info(f"Starting entity extraction for {total_docs} documents in {dataroom_id}")

            total_entities = 0
            for i, doc in enumerate(docs_result):
                if self._cancelled:
                    logger.info("Entity extraction cancelled")
                    self._update_progress(
                        status="cancelled",
                        end_time=datetime.utcnow(),
                    )
                    return ExtractionResult(
                        success=False,
                        dataroom_id=dataroom_id,
                        documents_processed=i,
                        entities_extracted=total_entities,
                        errors=["Extraction cancelled by user"],
                        duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                    )

                doc_id = doc["doc_id"]
                filename = doc["filename"]

                self._update_progress(
                    current_document=filename,
                    processed_documents=i,
                )

                try:
                    entities_count = self.linker.process_document_chunks(
                        doc_id, dataroom_id
                    )
                    total_entities += entities_count
                    self._update_progress(total_entities=total_entities)
                    logger.info(f"Extracted {entities_count} entities from {filename}")

                except Exception as e:
                    error_msg = f"Entity extraction failed for {filename}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Create entity relationships based on co-occurrence
            if total_entities > 0:
                try:
                    self._update_progress(current_document="Creating entity relationships...")
                    self.linker.create_entity_relationships(dataroom_id)
                except Exception as e:
                    error_msg = f"Failed to create entity relationships: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Update data room stats
            self._update_dataroom_stats(dataroom_id)

            self._update_progress(
                processed_documents=total_docs,
                status="completed",
                end_time=datetime.utcnow(),
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Entity extraction complete: {total_entities} entities from {total_docs} documents"
            )

            return ExtractionResult(
                success=len(errors) == 0,
                dataroom_id=dataroom_id,
                documents_processed=total_docs,
                entities_extracted=total_entities,
                errors=errors,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.exception(f"Entity extraction failed: {e}")
            self._update_progress(
                status="failed",
                error_message=str(e),
                end_time=datetime.utcnow(),
            )
            return ExtractionResult(
                success=False,
                dataroom_id=dataroom_id,
                errors=[str(e)],
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _update_dataroom_stats(self, dataroom_id: str):
        """Update data room entity count in Neo4j.

        Args:
            dataroom_id: Data room ID.
        """
        self.neo4j.execute_write(
            """
            MATCH (dr:DataRoom {id: $dataroom_id})
            OPTIONAL MATCH (e:Entity {dataroom_id: $dataroom_id})
            WITH dr, count(DISTINCT e) as entity_count
            SET dr.entity_count = entity_count,
                dr.updated_at = datetime()
            """,
            {"dataroom_id": dataroom_id},
        )

    def get_extraction_status(self, dataroom_id: str) -> dict:
        """Check extraction status for a data room.

        Args:
            dataroom_id: Data room ID.

        Returns:
            Dict with extraction status info.
        """
        result = self.neo4j.execute_read(
            """
            MATCH (dr:DataRoom {id: $dataroom_id})
            OPTIONAL MATCH (d:Document {dataroom_id: $dataroom_id})
            WHERE d.ingestion_status = 'completed'
            OPTIONAL MATCH (e:Entity {dataroom_id: $dataroom_id})
            RETURN
                count(DISTINCT d) as document_count,
                count(DISTINCT e) as entity_count,
                dr.entity_count as stored_entity_count
            """,
            {"dataroom_id": dataroom_id},
        )

        if result:
            row = result[0]
            doc_count = row.get("document_count", 0)
            entity_count = row.get("entity_count", 0)
            return {
                "document_count": doc_count,
                "entity_count": entity_count,
                "has_documents": doc_count > 0,
                "has_entities": entity_count > 0,
                "extraction_needed": doc_count > 0 and entity_count == 0,
            }

        return {
            "document_count": 0,
            "entity_count": 0,
            "has_documents": False,
            "has_entities": False,
            "extraction_needed": False,
        }
