"""Background worker for running the ingestion pipeline in a QThread."""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from ...backend.ingestion.pipeline import (
    IngestionPipeline,
    IngestionProgress,
    IngestionResult,
    create_pipeline,
    PipelineType,
)
from ...backend.ingestion.graphrag_pipeline import GraphRAGPipeline, GraphRAGIngestionProgress

logger = logging.getLogger(__name__)


def _create_worker_event_loop() -> asyncio.AbstractEventLoop:
    """Create a standard asyncio event loop for use in worker threads.

    When QtAsyncio is active, it replaces the default event loop policy.
    Worker threads need a standard event loop, not a QtAsyncio one.
    """
    # Use the appropriate selector event loop for the platform
    if sys.platform == "win32":
        # On Windows, use the selector event loop (more compatible)
        loop = asyncio.SelectorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    return loop


@dataclass
class IngestionTask:
    """Task parameters for ingestion."""

    dataroom_id: str
    folder_path: str
    pipeline_type: PipelineType = "graphrag"  # Use GraphRAG by default for entity extraction
    extract_entities: bool = True  # Enable entity extraction


class IngestionWorker(QObject):
    """Worker that runs the async ingestion pipeline in a background thread.

    This worker creates its own asyncio event loop and runs the IngestionPipeline
    asynchronously, emitting Qt signals for thread-safe UI updates.
    """

    # Signals for thread-safe communication with main thread
    progress_updated = Signal(object)  # IngestionProgress
    ingestion_completed = Signal(object)  # IngestionResult
    ingestion_failed = Signal(str)  # error message

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the worker.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._cancelled = False
        self._task: Optional[IngestionTask] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_task(self, task: IngestionTask):
        """Set the ingestion task to run.

        Args:
            task: The ingestion task parameters.
        """
        self._task = task

    @Slot()
    def start_ingestion(self):
        """Start the ingestion process.

        This method is called when the thread starts. It creates a standard
        asyncio event loop (not QtAsyncio) and runs the pipeline.
        """
        if not self._task:
            self.ingestion_failed.emit("No task configured")
            return

        self._cancelled = False

        try:
            # Create a standard event loop for this worker thread
            # (bypasses QtAsyncio which only works on main thread)
            self._loop = _create_worker_event_loop()
            asyncio.set_event_loop(self._loop)

            # Create the pipeline using factory function
            pipeline = create_pipeline(
                pipeline_type=self._task.pipeline_type,
                extract_entities=self._task.extract_entities,
            )

            # Set progress callback (works for both pipeline types)
            if hasattr(pipeline, 'set_progress_callback'):
                pipeline.set_progress_callback(self._on_progress_wrapper)

            # Run the async ingestion
            folder_path = Path(self._task.folder_path)
            logger.info(
                f"Starting {self._task.pipeline_type} pipeline ingestion for {folder_path}"
            )
            result = self._loop.run_until_complete(
                pipeline.ingest_folder(self._task.dataroom_id, folder_path)
            )

            if not self._cancelled:
                # Both pipeline types now return compatible IngestionResult
                # GraphRAGIngestionResult has same structure as IngestionResult
                self.ingestion_completed.emit(result)

        except Exception as e:
            logger.exception(f"Ingestion failed: {e}")
            if not self._cancelled:
                self.ingestion_failed.emit(str(e))

        finally:
            if self._loop:
                self._loop.close()
                self._loop = None

    def _on_progress(self, progress: IngestionProgress):
        """Handle progress updates from the legacy pipeline.

        Args:
            progress: Current ingestion progress.
        """
        if not self._cancelled:
            self.progress_updated.emit(progress)

    def _on_progress_wrapper(self, progress):
        """Handle progress updates from either pipeline type.

        Converts GraphRAG progress to legacy format for UI compatibility.

        Args:
            progress: Progress object (IngestionProgress or GraphRAGIngestionProgress).
        """
        if self._cancelled:
            return

        # Convert GraphRAG progress to legacy format if needed
        if isinstance(progress, GraphRAGIngestionProgress):
            legacy_progress = IngestionProgress(
                total_files=progress.total_documents,
                processed_files=progress.processed_documents,
                total_chunks=0,  # Not tracked in GraphRAG progress
                current_file=progress.current_file,
                status=progress.status,
                error_message=progress.error_message,
                entities_found=progress.entities_extracted,
            )
            self.progress_updated.emit(legacy_progress)
        else:
            self.progress_updated.emit(progress)

    def cancel(self):
        """Request cancellation of the ingestion."""
        self._cancelled = True
        logger.info("Ingestion cancellation requested")
