"""Background worker for running entity extraction in a QThread."""

import asyncio
import logging
import sys
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from ...backend.extraction.extraction_pipeline import (
    EntityExtractionPipeline,
    ExtractionProgress,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


def _create_worker_event_loop() -> asyncio.AbstractEventLoop:
    """Create a standard asyncio event loop for use in worker threads.

    When QtAsyncio is active, it replaces the default event loop policy.
    Worker threads need a standard event loop, not a QtAsyncio one.
    """
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    return loop


class ExtractionWorker(QObject):
    """Worker that runs the entity extraction pipeline in a background thread.

    This worker creates its own asyncio event loop and runs the extraction
    asynchronously, emitting Qt signals for thread-safe UI updates.
    """

    # Signals for thread-safe communication with main thread
    progress_updated = Signal(object)  # ExtractionProgress
    extraction_completed = Signal(object)  # ExtractionResult
    extraction_failed = Signal(str)  # error message

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the worker.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._cancelled = False
        self._dataroom_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pipeline: Optional[EntityExtractionPipeline] = None

    def set_dataroom(self, dataroom_id: str):
        """Set the data room to extract entities from.

        Args:
            dataroom_id: Data room ID.
        """
        self._dataroom_id = dataroom_id

    @Slot()
    def start_extraction(self):
        """Start the entity extraction process.

        This method is called when the thread starts. It creates a standard
        asyncio event loop (not QtAsyncio) and runs the extraction pipeline.
        """
        if not self._dataroom_id:
            self.extraction_failed.emit("No data room configured")
            return

        self._cancelled = False

        try:
            # Create a standard event loop for this worker thread
            self._loop = _create_worker_event_loop()
            asyncio.set_event_loop(self._loop)

            # Create the pipeline
            self._pipeline = EntityExtractionPipeline()
            self._pipeline.set_progress_callback(self._on_progress)

            # Run the async extraction
            result = self._loop.run_until_complete(
                self._pipeline.extract_for_dataroom(self._dataroom_id)
            )

            if not self._cancelled:
                self.extraction_completed.emit(result)

        except Exception as e:
            logger.exception(f"Entity extraction failed: {e}")
            if not self._cancelled:
                self.extraction_failed.emit(str(e))

        finally:
            if self._loop:
                self._loop.close()
                self._loop = None
            self._pipeline = None

    def _on_progress(self, progress: ExtractionProgress):
        """Handle progress updates from the pipeline.

        Args:
            progress: Current extraction progress.
        """
        if not self._cancelled:
            self.progress_updated.emit(progress)

    def cancel(self):
        """Request cancellation of the extraction."""
        self._cancelled = True
        if self._pipeline:
            self._pipeline.cancel()
        logger.info("Entity extraction cancellation requested")
