"""Manager for coordinating ingestion between UI and background worker."""

import logging
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ...backend.ingestion.pipeline import IngestionProgress, IngestionResult
from ...backend.extraction.extraction_pipeline import ExtractionProgress, ExtractionResult
from ...backend.services.preflight import PreflightReport
from ..workers.ingestion_worker import IngestionWorker, IngestionTask
from ..workers.preflight_worker import PreflightWorker
from ..workers.extraction_worker import ExtractionWorker

logger = logging.getLogger(__name__)


class IngestionManager(QObject):
    """Coordinates ingestion between the UI and the background worker.

    This manager handles the lifecycle of the QThread and worker, translates
    progress updates into UI-friendly signals, and manages cleanup.
    """

    # Signals for UI updates - Ingestion
    preflight_started = Signal()
    preflight_completed = Signal(object)  # PreflightReport
    ingestion_started = Signal(str)  # folder_path
    progress_changed = Signal(int, int, str)  # value, max, message
    detailed_progress = Signal(object)  # IngestionProgress
    ingestion_finished = Signal(object)  # IngestionResult
    ingestion_error = Signal(str)  # error message

    # Signals for UI updates - Extraction
    extraction_started = Signal(str)  # dataroom_id
    extraction_progress = Signal(object)  # ExtractionProgress
    extraction_finished = Signal(object)  # ExtractionResult
    extraction_error = Signal(str)  # error message

    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the manager.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[IngestionWorker] = None
        self._is_running = False
        self._start_time: Optional[datetime] = None
        self._current_folder: Optional[str] = None

        # Preflight state
        self._preflight_thread: Optional[QThread] = None
        self._preflight_worker: Optional[PreflightWorker] = None
        self._pending_dataroom_id: Optional[str] = None
        self._pending_folder_path: Optional[str] = None

        # Extraction state
        self._extraction_thread: Optional[QThread] = None
        self._extraction_worker: Optional[ExtractionWorker] = None
        self._extraction_running = False

    @property
    def is_running(self) -> bool:
        """Check if ingestion is currently running."""
        return self._is_running

    @property
    def start_time(self) -> Optional[datetime]:
        """Get the start time of the current ingestion."""
        return self._start_time

    @property
    def current_folder(self) -> Optional[str]:
        """Get the folder path being ingested."""
        return self._current_folder

    def start_ingestion(self, dataroom_id: str, folder_path: str):
        """Start ingestion of a folder (with preflight validation first).

        Args:
            dataroom_id: ID of the target data room.
            folder_path: Path to the folder to ingest.
        """
        if self._is_running:
            logger.warning("Ingestion already in progress")
            return

        logger.info(f"Starting preflight for: {folder_path} -> {dataroom_id}")

        # Store pending task for after preflight completes
        self._pending_dataroom_id = dataroom_id
        self._pending_folder_path = folder_path

        # Run preflight first
        self._start_preflight(folder_path)

    def _start_preflight(self, folder_path: str):
        """Start preflight validation.

        Args:
            folder_path: Path to the folder to validate.
        """
        self.preflight_started.emit()

        # Create thread and worker
        self._preflight_thread = QThread()
        self._preflight_worker = PreflightWorker(folder_path)

        # Move worker to thread
        self._preflight_worker.moveToThread(self._preflight_thread)

        # Connect signals
        self._preflight_thread.started.connect(self._preflight_worker.run_validation)
        self._preflight_worker.validation_completed.connect(self._on_preflight_completed)
        self._preflight_worker.validation_failed.connect(self._on_preflight_failed)

        # Start the thread
        self._preflight_thread.start()

    @Slot(object)
    def _on_preflight_completed(self, report: PreflightReport):
        """Handle preflight validation completion.

        Args:
            report: The preflight validation report.
        """
        self._cleanup_preflight()
        self.preflight_completed.emit(report)

        if report.ready:
            # Preflight passed - start actual ingestion
            self._start_ingestion_worker()
        else:
            # Preflight failed - clear pending state
            logger.warning(f"Preflight failed: {report.blocking_errors}")
            self._pending_dataroom_id = None
            self._pending_folder_path = None

    @Slot(str)
    def _on_preflight_failed(self, error: str):
        """Handle preflight validation failure.

        Args:
            error: Error message.
        """
        self._cleanup_preflight()
        logger.error(f"Preflight validation error: {error}")

        # Emit a failed preflight report
        from ...backend.services.preflight import PreflightResult
        report = PreflightReport(
            ready=False,
            results=[PreflightResult("preflight", "error", error)],
            blocking_errors=[f"Preflight validation error: {error}"],
        )
        self.preflight_completed.emit(report)

        # Clear pending state
        self._pending_dataroom_id = None
        self._pending_folder_path = None

    def _cleanup_preflight(self):
        """Clean up preflight thread and worker."""
        if self._preflight_thread:
            self._preflight_thread.quit()
            self._preflight_thread.wait()
            self._preflight_thread.deleteLater()
            self._preflight_thread = None

        if self._preflight_worker:
            self._preflight_worker.deleteLater()
            self._preflight_worker = None

    def _start_ingestion_worker(self):
        """Start the actual ingestion worker after preflight passes."""
        if not self._pending_dataroom_id or not self._pending_folder_path:
            logger.error("No pending ingestion task")
            return

        dataroom_id = self._pending_dataroom_id
        folder_path = self._pending_folder_path

        # Clear pending state
        self._pending_dataroom_id = None
        self._pending_folder_path = None

        logger.info(f"Starting ingestion: {folder_path} -> {dataroom_id}")
        self._is_running = True
        self._start_time = datetime.now()
        self._current_folder = folder_path

        # Create thread and worker
        self._thread = QThread()
        self._worker = IngestionWorker()
        self._worker.set_task(IngestionTask(dataroom_id=dataroom_id, folder_path=folder_path))

        # Move worker to thread
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.start_ingestion)
        self._worker.progress_updated.connect(self._on_progress_updated)
        self._worker.ingestion_completed.connect(self._on_ingestion_completed)
        self._worker.ingestion_failed.connect(self._on_ingestion_failed)

        # Cleanup connections
        self._worker.ingestion_completed.connect(self._cleanup)
        self._worker.ingestion_failed.connect(self._cleanup)

        # Start the thread
        self._thread.start()

        # Emit started signal
        self.ingestion_started.emit(folder_path)

    def cancel(self):
        """Cancel the current ingestion."""
        if not self._is_running:
            return

        logger.info("Cancelling ingestion")
        if self._worker:
            self._worker.cancel()

    @Slot(object)
    def _on_progress_updated(self, progress: IngestionProgress):
        """Handle progress updates from the worker.

        Args:
            progress: Current ingestion progress.
        """
        # Emit detailed progress for the progress panel
        self.detailed_progress.emit(progress)

        # Emit simplified progress for the drop zone
        if progress.total_files > 0:
            message = f"Processing: {progress.current_file}"
            self.progress_changed.emit(
                progress.processed_files,
                progress.total_files,
                message,
            )

    @Slot(object)
    def _on_ingestion_completed(self, result: IngestionResult):
        """Handle ingestion completion.

        Args:
            result: The ingestion result.
        """
        logger.info(
            f"Ingestion completed: {result.documents_processed} documents, "
            f"{result.chunks_created} chunks"
        )
        self._is_running = False
        self.ingestion_finished.emit(result)

    @Slot(str)
    def _on_ingestion_failed(self, error: str):
        """Handle ingestion failure.

        Args:
            error: Error message.
        """
        logger.error(f"Ingestion failed: {error}")
        self._is_running = False
        self.ingestion_error.emit(error)

    @Slot()
    def _cleanup(self):
        """Clean up thread and worker after completion."""
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

        self._start_time = None
        self._current_folder = None

    # =========================================================================
    # Entity Extraction Methods
    # =========================================================================

    @property
    def is_extraction_running(self) -> bool:
        """Check if entity extraction is currently running."""
        return self._extraction_running

    def start_extraction(self, dataroom_id: str):
        """Start entity extraction for a data room.

        Args:
            dataroom_id: ID of the data room to extract entities from.
        """
        if self._extraction_running:
            logger.warning("Entity extraction already in progress")
            return

        if self._is_running:
            logger.warning("Cannot start extraction while ingestion is running")
            return

        logger.info(f"Starting entity extraction for: {dataroom_id}")
        self._extraction_running = True

        # Create thread and worker
        self._extraction_thread = QThread()
        self._extraction_worker = ExtractionWorker()
        self._extraction_worker.set_dataroom(dataroom_id)

        # Move worker to thread
        self._extraction_worker.moveToThread(self._extraction_thread)

        # Connect signals
        self._extraction_thread.started.connect(self._extraction_worker.start_extraction)
        self._extraction_worker.progress_updated.connect(self._on_extraction_progress)
        self._extraction_worker.extraction_completed.connect(self._on_extraction_completed)
        self._extraction_worker.extraction_failed.connect(self._on_extraction_failed)

        # Cleanup connections
        self._extraction_worker.extraction_completed.connect(self._cleanup_extraction)
        self._extraction_worker.extraction_failed.connect(self._cleanup_extraction)

        # Start the thread
        self._extraction_thread.start()

        # Emit started signal
        self.extraction_started.emit(dataroom_id)

    def cancel_extraction(self):
        """Cancel the current entity extraction."""
        if not self._extraction_running:
            return

        logger.info("Cancelling entity extraction")
        if self._extraction_worker:
            self._extraction_worker.cancel()

    @Slot(object)
    def _on_extraction_progress(self, progress: ExtractionProgress):
        """Handle extraction progress updates.

        Args:
            progress: Current extraction progress.
        """
        self.extraction_progress.emit(progress)

    @Slot(object)
    def _on_extraction_completed(self, result: ExtractionResult):
        """Handle extraction completion.

        Args:
            result: The extraction result.
        """
        logger.info(
            f"Entity extraction completed: {result.entities_extracted} entities "
            f"from {result.documents_processed} documents"
        )
        self._extraction_running = False
        self.extraction_finished.emit(result)

    @Slot(str)
    def _on_extraction_failed(self, error: str):
        """Handle extraction failure.

        Args:
            error: Error message.
        """
        logger.error(f"Entity extraction failed: {error}")
        self._extraction_running = False
        self.extraction_error.emit(error)

    @Slot()
    def _cleanup_extraction(self):
        """Clean up extraction thread and worker after completion."""
        if self._extraction_thread:
            self._extraction_thread.quit()
            self._extraction_thread.wait()
            self._extraction_thread.deleteLater()
            self._extraction_thread = None

        if self._extraction_worker:
            self._extraction_worker.deleteLater()
            self._extraction_worker = None
