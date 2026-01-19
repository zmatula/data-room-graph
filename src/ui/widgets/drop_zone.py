"""Drag-and-drop zone widget for folder ingestion."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent

logger = logging.getLogger(__name__)


class DropZoneWidget(QFrame):
    """Widget for drag-and-drop folder ingestion."""

    # Signal emitted when a folder is dropped
    folder_dropped = Signal(str)  # folder_path

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._dataroom_id: Optional[str] = None
        self._is_ingesting = False
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumHeight(150)
        self.setMaximumHeight(250)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Icon/visual indicator
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 48px;")
        self.icon_label.setText("\U0001F4C1")  # Folder emoji
        layout.addWidget(self.icon_label)

        # Main label
        self.main_label = QLabel("Drop folders here to ingest")
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e8e8f0;")
        layout.addWidget(self.main_label)

        # Status label
        self.status_label = QLabel("Select a data room first")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #a0a0b8; font-size: 12px;")
        layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self._update_style()

    def _update_style(self):
        """Update the widget style based on state."""
        if not self._dataroom_id:
            self.setStyleSheet(
                """
                DropZoneWidget {
                    background-color: #16213e;
                    border: 2px dashed #3d3d6b;
                    border-radius: 8px;
                }
                QLabel {
                    color: #6b6b85;
                }
                """
            )
            self.status_label.setText("Select a data room first")
        elif self._is_ingesting:
            self.setStyleSheet(
                """
                DropZoneWidget {
                    background-color: #3d2c0a;
                    border: 2px solid #f59e0b;
                    border-radius: 8px;
                }
                QLabel {
                    color: #f59e0b;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                DropZoneWidget {
                    background-color: #1f1f3d;
                    border: 2px dashed #8b5cf6;
                    border-radius: 8px;
                }
                DropZoneWidget:hover {
                    background-color: #252547;
                    border: 2px solid #a78bfa;
                }
                QLabel {
                    color: #e8e8f0;
                }
                """
            )
            self.status_label.setText("Ready to ingest documents")

    def set_dataroom(self, dataroom_id: Optional[str]):
        """Set the current data room.

        Args:
            dataroom_id: ID of the current data room.
        """
        self._dataroom_id = dataroom_id
        self._update_style()

    def set_ingesting(self, is_ingesting: bool, message: str = ""):
        """Set the ingestion state.

        Args:
            is_ingesting: Whether ingestion is in progress.
            message: Status message to display.
        """
        self._is_ingesting = is_ingesting
        self.progress_bar.setVisible(is_ingesting)

        if is_ingesting:
            self.main_label.setText("Ingesting...")
            self.status_label.setText(message or "Processing documents")
            self.icon_label.setText("\u23F3")  # Hourglass
        else:
            self.main_label.setText("Drop folders here to ingest")
            self.status_label.setText(message or "Ready to ingest documents")
            self.icon_label.setText("\U0001F4C1")  # Folder

        self._update_style()

    def set_progress(self, value: int, maximum: int = 100, message: str = ""):
        """Update the progress bar.

        Args:
            value: Current progress value.
            maximum: Maximum progress value.
            message: Status message.
        """
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if not self._dataroom_id or self._is_ingesting:
            event.ignore()
            return

        if event.mimeData().hasUrls():
            # Check if any URL is a directory
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    event.acceptProposedAction()
                    self.setStyleSheet(
                        """
                        DropZoneWidget {
                            background-color: #252547;
                            border: 2px solid #a78bfa;
                            border-radius: 8px;
                        }
                        QLabel {
                            color: #a78bfa;
                        }
                        """
                    )
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self._update_style()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        if not self._dataroom_id or self._is_ingesting:
            event.ignore()
            return

        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    logger.info(f"Folder dropped: {path}")
                    self.folder_dropped.emit(str(path))
                    event.acceptProposedAction()
                    self._update_style()
                    return

        event.ignore()
        self._update_style()
