"""Background worker for running the ingestion pipeline in a QThread."""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from ...backend.ingestion.pipeline import IngestionPipeline, IngestionProgress, IngestionResult

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

            # Create the pipeline
            pipeline = IngestionPipeline()
            pipeline.set_progress_callback(self._on_progress)

            # Run the async ingestion
            folder_path = Path(self._task.folder_path)
            result = self._loop.run_until_complete(
                pipeline.ingest_folder(self._task.dataroom_id, folder_path)
            )

            if not self._cancelled:
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
        """Handle progress updates from the pipeline.

        Args:
            progress: Current ingestion progress.
        """
        if not self._cancelled:
            self.progress_updated.emit(progress)

    def cancel(self):
        """Request cancellation of the ingestion."""
        self._cancelled = True
        logger.info("Ingestion cancellation requested")
