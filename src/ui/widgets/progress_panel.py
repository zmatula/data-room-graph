"""Progress panel widget for displaying detailed ingestion progress."""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from ...backend.config import get_settings
from ...backend.ingestion.pipeline import IngestionProgress, IngestionResult
from ...backend.services.preflight import PreflightReport, PreflightResult

logger = logging.getLogger(__name__)


class ProgressPanel(QFrame):
    """Widget displaying detailed ingestion progress.

    Shows current file, file/chunk progress bars, elapsed time,
    status indicator, error list, and cancel button.
    """

    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the progress panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._start_time: Optional[datetime] = None
        self._is_running = False
        self._errors: list[str] = []
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Setup the widget UI."""
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet(
            """
            ProgressPanel {
                background-color: #1a1a2e;
                border: 1px solid #3d3d6b;
                border-radius: 8px;
            }
            QLabel {
                color: #e8e8f0;
            }
            QProgressBar {
                border: 1px solid #3d3d6b;
                border-radius: 4px;
                background-color: #16213e;
                text-align: center;
                color: #e8e8f0;
            }
            QProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #6b7280;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header with status and elapsed time
        header_layout = QHBoxLayout()

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.status_label)

        header_layout.addStretch()

        self.elapsed_label = QLabel("--:--")
        self.elapsed_label.setStyleSheet("color: #a0a0b8; font-size: 12px;")
        header_layout.addWidget(self.elapsed_label)

        layout.addLayout(header_layout)

        # Current file label
        self.file_label = QLabel("No file")
        self.file_label.setStyleSheet("color: #a0a0b8; font-size: 12px;")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        # File progress
        file_progress_layout = QHBoxLayout()
        file_progress_layout.addWidget(QLabel("Files:"))
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setMinimumHeight(20)
        self.file_progress_bar.setFormat("%v / %m")
        file_progress_layout.addWidget(self.file_progress_bar, stretch=1)
        layout.addLayout(file_progress_layout)

        # Chunk progress
        chunk_progress_layout = QHBoxLayout()
        chunk_progress_layout.addWidget(QLabel("Chunks:"))
        self.chunk_progress_bar = QProgressBar()
        self.chunk_progress_bar.setMinimumHeight(20)
        self.chunk_progress_bar.setFormat("%v / %m")
        chunk_progress_layout.addWidget(self.chunk_progress_bar, stretch=1)
        layout.addLayout(chunk_progress_layout)

        # Error section (more prominent, with Open Log Folder button)
        self.error_widget = QWidget()
        self.error_widget.setStyleSheet(
            """
            QWidget#error_widget {
                background-color: #2d1515;
                border: 1px solid #ef4444;
                border-radius: 6px;
                padding: 4px;
            }
            """
        )
        self.error_widget.setObjectName("error_widget")
        error_layout = QVBoxLayout(self.error_widget)
        error_layout.setContentsMargins(8, 8, 8, 8)
        error_layout.setSpacing(6)

        # Error header with Open Log Folder button
        error_header_layout = QHBoxLayout()
        self.error_header = QLabel("Errors: 0")
        self.error_header.setStyleSheet(
            "color: #ef4444; font-weight: bold; font-size: 13px;"
        )
        error_header_layout.addWidget(self.error_header)
        error_header_layout.addStretch()

        self.open_logs_button = QPushButton("Open Log Folder")
        self.open_logs_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3d3d6b;
                color: #e8e8f0;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4d4d7b;
            }
            """
        )
        self.open_logs_button.clicked.connect(self._on_open_logs_clicked)
        error_header_layout.addWidget(self.open_logs_button)
        error_layout.addLayout(error_header_layout)

        # Scrollable error list (taller for better visibility)
        self.error_scroll = QScrollArea()
        self.error_scroll.setWidgetResizable(True)
        self.error_scroll.setMinimumHeight(60)
        self.error_scroll.setMaximumHeight(120)
        self.error_scroll.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #5d2525;
                border-radius: 4px;
                background-color: #1a1010;
            }
            """
        )

        self.error_list_widget = QWidget()
        self.error_list_layout = QVBoxLayout(self.error_list_widget)
        self.error_list_layout.setContentsMargins(8, 8, 8, 8)
        self.error_list_layout.setSpacing(4)
        self.error_list_layout.addStretch()

        self.error_scroll.setWidget(self.error_list_widget)
        error_layout.addWidget(self.error_scroll)

        self.error_widget.setVisible(False)
        layout.addWidget(self.error_widget)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Set initial state
        self._reset()

    def _setup_timer(self):
        """Setup the timer for updating elapsed time."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed_time)

    def _reset(self):
        """Reset the panel to idle state."""
        self.file_progress_bar.setValue(0)
        self.file_progress_bar.setMaximum(100)
        self.chunk_progress_bar.setValue(0)
        self.chunk_progress_bar.setMaximum(100)
        self.file_label.setText("No file")
        self.elapsed_label.setText("--:--")
        self.error_widget.setVisible(False)
        self._errors.clear()
        self._clear_error_list()

    def _clear_error_list(self):
        """Clear all error labels from the error list."""
        while self.error_list_layout.count() > 1:  # Keep the stretch
            item = self.error_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def start(self, folder_path: str, start_time: Optional[datetime] = None):
        """Start showing progress for an ingestion.

        Args:
            folder_path: Path to the folder being ingested.
            start_time: Optional start time (defaults to now).
        """
        self._reset()
        self._is_running = True
        self._start_time = start_time or datetime.now()

        folder_name = Path(folder_path).name
        self.status_label.setText(f"Ingesting: {folder_name}")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #f59e0b;"
        )

        self.cancel_button.setEnabled(True)
        self._timer.start(1000)  # Update every second

    def update_progress(self, progress: IngestionProgress):
        """Update the progress display.

        Args:
            progress: Current ingestion progress.
        """
        # Update file progress
        self.file_progress_bar.setMaximum(max(progress.total_files, 1))
        self.file_progress_bar.setValue(progress.processed_files)

        # Update chunk progress
        if progress.total_chunks > 0:
            self.chunk_progress_bar.setMaximum(progress.total_chunks)
            self.chunk_progress_bar.setValue(progress.processed_chunks)

        # Update current file
        if progress.current_file:
            self.file_label.setText(f"Processing: {progress.current_file}")
        else:
            self.file_label.setText("Preparing...")

        # Handle status changes
        if progress.status == "completed":
            self._on_completed()
        elif progress.status == "failed":
            self._on_failed(progress.error_message)

    def finish(self, result: IngestionResult):
        """Show completion state.

        Args:
            result: The ingestion result.
        """
        self._is_running = False
        self._timer.stop()
        self.cancel_button.setEnabled(False)

        if result.success:
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #22c55e;"
            )
            self.file_label.setText(
                f"Processed {result.documents_processed} documents, "
                f"created {result.chunks_created} chunks"
            )
        else:
            self.status_label.setText("Completed with errors")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #f59e0b;"
            )

        # Show errors if any
        if result.errors:
            self._show_errors(result.errors)

    def show_error(self, error: str):
        """Show an error state.

        Args:
            error: The error message.
        """
        self._is_running = False
        self._timer.stop()
        self.cancel_button.setEnabled(False)

        self.status_label.setText("Failed")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #ef4444;"
        )
        self.file_label.setText(error)

        self._show_errors([error])

    def _show_errors(self, errors: list[str]):
        """Show errors in the error list.

        Args:
            errors: List of error messages.
        """
        self._errors = errors
        self.error_header.setText(f"Errors: {len(errors)}")
        self.error_widget.setVisible(True)

        self._clear_error_list()
        for error in errors:
            error_label = QLabel(f"• {error}")
            error_label.setStyleSheet("color: #ef4444; font-size: 11px;")
            error_label.setWordWrap(True)
            self.error_list_layout.insertWidget(
                self.error_list_layout.count() - 1,  # Before the stretch
                error_label,
            )

    def _on_completed(self):
        """Handle completion state from progress update."""
        self._is_running = False
        self._timer.stop()
        self.cancel_button.setEnabled(False)

        self.status_label.setText("Completed")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #22c55e;"
        )

    def _on_failed(self, error_message: str):
        """Handle failure state from progress update.

        Args:
            error_message: The error message.
        """
        self.show_error(error_message)

    @Slot()
    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        if self._start_time and self._is_running:
            elapsed = datetime.now() - self._start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.elapsed_label.setText(f"{minutes:02d}:{seconds:02d}")

    @Slot()
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.cancel_requested.emit()

    @Slot()
    def _on_open_logs_clicked(self):
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
        except Exception as e:
            logger.error(f"Failed to open logs folder: {e}")

    def start_preflight(self):
        """Show preflight validation in progress."""
        self._reset()
        self.status_label.setText("Validating services...")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #3b82f6;"
        )
        self.file_label.setText("Checking Neo4j, Unstructured.io, OpenAI...")
        self.cancel_button.setEnabled(False)

    def show_preflight_results(self, report: PreflightReport):
        """Display preflight validation results.

        Args:
            report: The preflight validation report.
        """
        if report.ready:
            self.status_label.setText("Validation passed")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #22c55e;"
            )
            # Build summary of successful checks
            ok_services = [r.service for r in report.results if r.status == "ok"]
            self.file_label.setText(f"Services ready: {', '.join(ok_services)}")
        else:
            self.status_label.setText("Validation failed")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #ef4444;"
            )
            self._show_preflight_errors(report)

    def _show_preflight_errors(self, report: PreflightReport):
        """Show detailed preflight error information.

        Args:
            report: The preflight validation report.
        """
        errors = []
        for result in report.results:
            if result.status == "error":
                error_msg = f"{result.service}: {result.message}"
                if result.details and "suggestion" in result.details:
                    error_msg += f" ({result.details['suggestion']})"
                errors.append(error_msg)

        if errors:
            self.file_label.setText("Service validation failed")
            self._show_errors(errors)
