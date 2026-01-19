"""Main application window with splitter layout."""

import logging
import os
import subprocess
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QLabel,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon

from .widgets.dataroom_list import DataRoomListWidget
from .widgets.dashboard import DashboardWidget
from .widgets.drop_zone import DropZoneWidget
from .widgets.chat import ChatWidget
from .widgets.progress_panel import ProgressPanel
from .managers.ingestion_manager import IngestionManager
from ..backend.config import get_settings
from ..backend.ingestion.pipeline import IngestionProgress, IngestionResult
from ..backend.extraction.extraction_pipeline import ExtractionProgress, ExtractionResult
from ..backend.services.preflight import PreflightReport

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    # Signal emitted when a data room is selected
    dataroom_selected = Signal(str)  # dataroom_id

    def __init__(self, neo4j_available: bool = True, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            neo4j_available: Whether Neo4j connection is available.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.neo4j_available = neo4j_available
        self.current_dataroom_id: Optional[str] = None

        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_ingestion_manager()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Data Room Graph")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # Left panel: Data room list
        self.dataroom_list = DataRoomListWidget()
        self.dataroom_list.setMinimumWidth(200)
        self.dataroom_list.setMaximumWidth(350)
        self.main_splitter.addWidget(self.dataroom_list)

        # Center area: Stacked widget for different views
        self.center_stack = QStackedWidget()
        self.main_splitter.addWidget(self.center_stack)

        # Welcome/empty state view
        self.welcome_widget = self._create_welcome_widget()
        self.center_stack.addWidget(self.welcome_widget)

        # Dashboard view (shown when data room is selected)
        self.dashboard_widget = DashboardWidget()
        self.center_stack.addWidget(self.dashboard_widget)

        # Right panel: Vertical splitter for agent console and graph
        self.right_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.right_splitter)

        # Chat/Agent console
        self.chat_widget = ChatWidget()
        self.right_splitter.addWidget(self.chat_widget)

        # Drop zone (for folder drag-drop)
        self.drop_zone = DropZoneWidget()
        self.right_splitter.addWidget(self.drop_zone)

        # Progress panel (for detailed ingestion progress)
        self.progress_panel = ProgressPanel()
        self.right_splitter.addWidget(self.progress_panel)

        # Set splitter sizes (proportions)
        self.main_splitter.setSizes([250, 700, 400])
        self.right_splitter.setSizes([400, 150, 200])

        # Set stretch factors
        self.main_splitter.setStretchFactor(0, 0)  # Left: fixed
        self.main_splitter.setStretchFactor(1, 1)  # Center: stretch
        self.main_splitter.setStretchFactor(2, 0)  # Right: fixed

    def _create_welcome_widget(self) -> QWidget:
        """Create the welcome/empty state widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Welcome to Data Room Graph")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #e8e8f0;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        subtitle = QLabel(
            "Create a new data room or select an existing one to get started."
        )
        subtitle.setStyleSheet("font-size: 14px; color: #a0a0b8; margin-top: 10px;")
        layout.addWidget(subtitle, alignment=Qt.AlignCenter)

        hint = QLabel("Press Ctrl+N or click 'New Data Room' to begin")
        hint.setStyleSheet("font-size: 12px; color: #8b5cf6; margin-top: 20px;")
        layout.addWidget(hint, alignment=Qt.AlignCenter)

        if not self.neo4j_available:
            warning = QLabel(
                "Warning: Neo4j is not connected. Run scripts/setup_neo4j.py to start."
            )
            warning.setStyleSheet(
                "font-size: 12px; color: #ef4444; margin-top: 20px; "
                "padding: 10px; background-color: #3d1515; border: 1px solid #ef4444; border-radius: 4px;"
            )
            layout.addWidget(warning, alignment=Qt.AlignCenter)

        return widget

    def _setup_toolbar(self):
        """Setup the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # New data room action
        new_action = QAction("New Data Room", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_dataroom)
        toolbar.addAction(new_action)

        toolbar.addSeparator()

        # Refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._on_refresh)
        toolbar.addAction(refresh_action)

        toolbar.addSeparator()

        # Extract Entities action
        self.extract_entities_action = QAction("Extract Entities", self)
        self.extract_entities_action.setShortcut("Ctrl+E")
        self.extract_entities_action.triggered.connect(self._on_extract_entities)
        self.extract_entities_action.setEnabled(False)  # Disabled until data room selected
        toolbar.addAction(self.extract_entities_action)

        toolbar.addSeparator()

        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._on_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()

        # Open Logs action
        logs_action = QAction("Open Logs", self)
        logs_action.triggered.connect(self._on_open_logs)
        toolbar.addAction(logs_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Connection status indicator
        self.connection_label = QLabel()
        self._update_connection_status()
        self.statusbar.addPermanentWidget(self.connection_label)

    def _update_connection_status(self):
        """Update the Neo4j connection status indicator."""
        if self.neo4j_available:
            self.connection_label.setText("Neo4j: Connected")
            self.connection_label.setStyleSheet("color: #22c55e;")
        else:
            self.connection_label.setText("Neo4j: Disconnected")
            self.connection_label.setStyleSheet("color: #ef4444;")

    def _setup_ingestion_manager(self):
        """Setup the ingestion manager and connect its signals."""
        self.ingestion_manager = IngestionManager(self)

        # Connect preflight signals
        self.ingestion_manager.preflight_started.connect(self._on_preflight_started)
        self.ingestion_manager.preflight_completed.connect(self._on_preflight_completed)

        # Connect manager signals to handlers
        self.ingestion_manager.ingestion_started.connect(self._on_ingestion_started)
        self.ingestion_manager.progress_changed.connect(self._on_ingestion_progress)
        self.ingestion_manager.detailed_progress.connect(self._on_detailed_progress)
        self.ingestion_manager.ingestion_finished.connect(self._on_ingestion_finished)
        self.ingestion_manager.ingestion_error.connect(self._on_ingestion_error)

        # Connect extraction signals
        self.ingestion_manager.extraction_started.connect(self._on_extraction_started)
        self.ingestion_manager.extraction_progress.connect(self._on_extraction_progress)
        self.ingestion_manager.extraction_finished.connect(self._on_extraction_finished)
        self.ingestion_manager.extraction_error.connect(self._on_extraction_error)

        # Connect progress panel cancel to manager
        self.progress_panel.cancel_requested.connect(self.ingestion_manager.cancel)

    def _connect_signals(self):
        """Connect widget signals."""
        # Data room selection
        self.dataroom_list.dataroom_selected.connect(self._on_dataroom_selected)
        self.dataroom_list.dataroom_created.connect(self._on_dataroom_created)

        # Drop zone
        self.drop_zone.folder_dropped.connect(self._on_folder_dropped)

        # Dashboard extraction request
        self.dashboard_widget.extract_entities_requested.connect(self._on_extract_entities_requested)

    @Slot(str)
    def _on_dataroom_selected(self, dataroom_id: str):
        """Handle data room selection.

        Args:
            dataroom_id: ID of the selected data room.
        """
        logger.info(f"Data room selected: {dataroom_id}")
        self.current_dataroom_id = dataroom_id

        # Update dashboard
        self.dashboard_widget.set_dataroom(dataroom_id)

        # Update chat widget
        self.chat_widget.set_dataroom(dataroom_id)

        # Update drop zone
        self.drop_zone.set_dataroom(dataroom_id)

        # Enable extract entities toolbar action
        self.extract_entities_action.setEnabled(True)

        # Show dashboard view
        self.center_stack.setCurrentWidget(self.dashboard_widget)

        # Emit signal
        self.dataroom_selected.emit(dataroom_id)

        self.statusbar.showMessage(f"Data room loaded: {dataroom_id}", 3000)

    @Slot(str)
    def _on_dataroom_created(self, dataroom_id: str):
        """Handle new data room creation.

        Args:
            dataroom_id: ID of the created data room.
        """
        logger.info(f"Data room created: {dataroom_id}")
        self._on_dataroom_selected(dataroom_id)

    @Slot(str)
    def _on_folder_dropped(self, folder_path: str):
        """Handle folder drop for ingestion.

        Args:
            folder_path: Path to the dropped folder.
        """
        if not self.current_dataroom_id:
            QMessageBox.warning(
                self,
                "No Data Room Selected",
                "Please select or create a data room before dropping folders.",
            )
            return

        if self.ingestion_manager.is_running:
            QMessageBox.warning(
                self,
                "Busy",
                "Ingestion already in progress. Please wait for it to complete.",
            )
            return

        logger.info(f"Folder dropped for ingestion: {folder_path}")
        self.statusbar.showMessage(f"Validating services...")

        # Start ingestion pipeline (will run preflight first)
        self.ingestion_manager.start_ingestion(
            self.current_dataroom_id, folder_path
        )

    @Slot()
    def _on_preflight_started(self):
        """Handle preflight validation started."""
        self.progress_panel.start_preflight()
        self.drop_zone.set_ingesting(True, "Validating services...")
        self.statusbar.showMessage("Validating services...")

    @Slot(object)
    def _on_preflight_completed(self, report: PreflightReport):
        """Handle preflight validation completed.

        Args:
            report: The preflight validation report.
        """
        self.progress_panel.show_preflight_results(report)

        if not report.ready:
            # Preflight failed - show error dialog
            self.drop_zone.set_ingesting(False, "Ready to ingest documents")

            # Build detailed error message
            error_lines = ["Service validation failed:\n"]
            for result in report.results:
                if result.status == "error":
                    error_lines.append(f"\n{result.service.upper()}: {result.message}")
                    if result.details and "suggestion" in result.details:
                        error_lines.append(f"   Suggestion: {result.details['suggestion']}")

            QMessageBox.warning(
                self,
                "Preflight Validation Failed",
                "".join(error_lines),
            )
            self.statusbar.showMessage("Validation failed", 5000)
        else:
            self.statusbar.showMessage("Services validated, starting ingestion...")

    @Slot()
    def _on_new_dataroom(self):
        """Handle new data room action."""
        self.dataroom_list.show_create_dialog()

    @Slot()
    def _on_refresh(self):
        """Handle refresh action."""
        self.dataroom_list.refresh()
        if self.current_dataroom_id:
            self.dashboard_widget.refresh()
        self.statusbar.showMessage("Refreshed", 2000)

    @Slot()
    def _on_settings(self):
        """Handle settings action."""
        # TODO: Show settings dialog
        QMessageBox.information(
            self, "Settings", "Settings dialog coming soon."
        )

    @Slot()
    def _on_open_logs(self):
        """Open the logs folder in the system file explorer."""
        settings = get_settings()
        logs_path = settings.logs_dir

        try:
            if sys.platform == "win32":
                os.startfile(logs_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(logs_path)], check=True)
            else:
                subprocess.run(["xdg-open", str(logs_path)], check=True)
            self.statusbar.showMessage(f"Opened logs folder: {logs_path}", 3000)
        except Exception as e:
            logger.error(f"Failed to open logs folder: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open logs folder:\n{logs_path}\n\nError: {e}",
            )

    @Slot(str)
    def _on_ingestion_started(self, folder_path: str):
        """Handle ingestion started.

        Args:
            folder_path: Path to the folder being ingested.
        """
        self.drop_zone.set_ingesting(True, "Initializing...")
        self.progress_panel.start(folder_path, self.ingestion_manager.start_time)
        self.statusbar.showMessage(f"Ingesting: {folder_path}")

    @Slot(int, int, str)
    def _on_ingestion_progress(self, value: int, maximum: int, message: str):
        """Handle simplified progress updates for drop zone.

        Args:
            value: Current progress value.
            maximum: Maximum progress value.
            message: Status message.
        """
        self.drop_zone.set_progress(value, maximum, message)

    @Slot(object)
    def _on_detailed_progress(self, progress: IngestionProgress):
        """Handle detailed progress updates for progress panel.

        Args:
            progress: Current ingestion progress.
        """
        self.progress_panel.update_progress(progress)

    @Slot(object)
    def _on_ingestion_finished(self, result: IngestionResult):
        """Handle ingestion completion.

        Args:
            result: The ingestion result.
        """
        self.drop_zone.set_ingesting(False, "Ready to ingest documents")
        self.progress_panel.finish(result)

        # Refresh the dashboard to show new data
        if self.current_dataroom_id:
            self.dashboard_widget.refresh()

        # Show summary message
        if result.success:
            self.statusbar.showMessage(
                f"Ingestion complete: {result.documents_processed} documents, "
                f"{result.chunks_created} chunks",
                5000,
            )
        else:
            error_count = len(result.errors)
            self.statusbar.showMessage(
                f"Ingestion completed with {error_count} error(s)",
                5000,
            )

    @Slot(str)
    def _on_ingestion_error(self, error: str):
        """Handle ingestion failure.

        Args:
            error: Error message.
        """
        self.drop_zone.set_ingesting(False, "Ready to ingest documents")
        self.progress_panel.show_error(error)
        self.statusbar.showMessage("Ingestion failed", 5000)

        QMessageBox.critical(
            self,
            "Ingestion Failed",
            f"An error occurred during ingestion:\n\n{error}",
        )

    # =========================================================================
    # Entity Extraction Handlers
    # =========================================================================

    @Slot()
    def _on_extract_entities(self):
        """Handle Extract Entities toolbar action."""
        if self.current_dataroom_id:
            self._start_extraction(self.current_dataroom_id)

    @Slot(str)
    def _on_extract_entities_requested(self, dataroom_id: str):
        """Handle extraction request from dashboard.

        Args:
            dataroom_id: Data room ID to extract entities from.
        """
        self._start_extraction(dataroom_id)

    def _start_extraction(self, dataroom_id: str):
        """Start entity extraction for a data room.

        Args:
            dataroom_id: Data room ID.
        """
        if self.ingestion_manager.is_running:
            QMessageBox.warning(
                self,
                "Busy",
                "Ingestion is in progress. Please wait for it to complete.",
            )
            return

        if self.ingestion_manager.is_extraction_running:
            QMessageBox.warning(
                self,
                "Busy",
                "Entity extraction already in progress.",
            )
            return

        logger.info(f"Starting entity extraction for: {dataroom_id}")
        self.ingestion_manager.start_extraction(dataroom_id)

    @Slot(str)
    def _on_extraction_started(self, dataroom_id: str):
        """Handle extraction started.

        Args:
            dataroom_id: Data room ID being processed.
        """
        self.dashboard_widget.set_extraction_running(True)
        self.extract_entities_action.setEnabled(False)
        self.statusbar.showMessage(f"Extracting entities from {dataroom_id}...")

    @Slot(object)
    def _on_extraction_progress(self, progress: ExtractionProgress):
        """Handle extraction progress updates.

        Args:
            progress: Current extraction progress.
        """
        if progress.total_documents > 0:
            message = (
                f"Extracting: {progress.current_document} "
                f"({progress.processed_documents}/{progress.total_documents})"
            )
            self.statusbar.showMessage(message)

    @Slot(object)
    def _on_extraction_finished(self, result: ExtractionResult):
        """Handle extraction completion.

        Args:
            result: The extraction result.
        """
        self.dashboard_widget.set_extraction_running(False)
        self.extract_entities_action.setEnabled(True)

        # Refresh the dashboard to show new entities
        if self.current_dataroom_id:
            self.dashboard_widget.refresh()

        # Show summary message
        if result.success:
            self.statusbar.showMessage(
                f"Entity extraction complete: {result.entities_extracted} entities "
                f"from {result.documents_processed} documents",
                5000,
            )
        else:
            error_count = len(result.errors)
            self.statusbar.showMessage(
                f"Entity extraction completed with {error_count} error(s)",
                5000,
            )
            if result.errors:
                QMessageBox.warning(
                    self,
                    "Extraction Completed with Errors",
                    f"Entity extraction completed but encountered {error_count} error(s):\n\n"
                    + "\n".join(result.errors[:5]),
                )

    @Slot(str)
    def _on_extraction_error(self, error: str):
        """Handle extraction failure.

        Args:
            error: Error message.
        """
        self.dashboard_widget.set_extraction_running(False)
        self.extract_entities_action.setEnabled(True)
        self.statusbar.showMessage("Entity extraction failed", 5000)

        QMessageBox.critical(
            self,
            "Extraction Failed",
            f"An error occurred during entity extraction:\n\n{error}",
        )

    def closeEvent(self, event):
        """Handle window close event."""
        # Confirm exit if ingestion is in progress
        if self.ingestion_manager.is_running:
            reply = QMessageBox.question(
                self,
                "Ingestion in Progress",
                "An ingestion is currently in progress. "
                "Are you sure you want to exit?\n\n"
                "The ingestion will be cancelled.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            # Cancel the ingestion
            self.ingestion_manager.cancel()

        # Confirm exit if extraction is in progress
        if self.ingestion_manager.is_extraction_running:
            reply = QMessageBox.question(
                self,
                "Extraction in Progress",
                "Entity extraction is currently in progress. "
                "Are you sure you want to exit?\n\n"
                "The extraction will be cancelled.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            # Cancel the extraction
            self.ingestion_manager.cancel_extraction()

        event.accept()
