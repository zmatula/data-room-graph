"""Background worker for running preflight validation in a QThread."""

import asyncio
import logging
import sys
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from ...backend.services.preflight import PreflightValidator, PreflightReport

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


class PreflightWorker(QObject):
    """Worker that runs preflight validation in a background thread.

    This worker creates its own asyncio event loop and runs the PreflightValidator
    asynchronously, emitting Qt signals for thread-safe UI updates.
    """

    # Signals for thread-safe communication with main thread
    validation_completed = Signal(object)  # PreflightReport
    validation_failed = Signal(str)  # error message

    def __init__(self, folder_path: Optional[str] = None, parent: Optional[QObject] = None):
        """Initialize the worker.

        Args:
            folder_path: Optional folder path to validate.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._folder_path = folder_path
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @Slot()
    def run_validation(self):
        """Run the preflight validation.

        This method is called when the thread starts. It creates a standard
        asyncio event loop (not QtAsyncio) and runs the validator.
        """
        try:
            # Create a standard event loop for this worker thread
            # (bypasses QtAsyncio which only works on main thread)
            self._loop = _create_worker_event_loop()
            asyncio.set_event_loop(self._loop)

            # Create the validator and run validation
            validator = PreflightValidator()
            report = self._loop.run_until_complete(
                validator.validate_all(self._folder_path)
            )

            self.validation_completed.emit(report)

        except Exception as e:
            logger.exception(f"Preflight validation failed: {e}")
            self.validation_failed.emit(str(e))

        finally:
            if self._loop:
                self._loop.close()
                self._loop = None
