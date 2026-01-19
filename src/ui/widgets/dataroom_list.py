"""Data room list sidebar widget."""

import logging
from typing import Optional
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QLineEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QTextEdit,
    QMessageBox,
    QMenu,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction

from ...backend.config import get_settings
from ...backend.database.models import DataRoom
from ...backend.database.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


class CreateDataRoomDialog(QDialog):
    """Dialog for creating a new data room."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the dialog.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Create Data Room")
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QFormLayout(self)

        # Name field
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Acme Capital Fund IV")
        layout.addRow("Name:", self.name_input)

        # Description field
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText(
            "Optional description of the data room..."
        )
        self.description_input.setMaximumHeight(100)
        layout.addRow("Description:", self.description_input)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _on_accept(self):
        """Handle accept button."""
        if not self.name_input.text().strip():
            QMessageBox.warning(
                self, "Validation Error", "Please enter a name for the data room."
            )
            return
        self.accept()

    def get_data(self) -> dict:
        """Get the dialog data.

        Returns:
            Dictionary with name and description.
        """
        return {
            "name": self.name_input.text().strip(),
            "description": self.description_input.toPlainText().strip() or None,
        }


class DataRoomListWidget(QWidget):
    """Widget for displaying and managing data rooms."""

    # Signals
    dataroom_selected = Signal(str)  # dataroom_id
    dataroom_created = Signal(str)  # dataroom_id

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._datarooms: dict[str, DataRoom] = {}
        self._setup_ui()
        self._load_datarooms()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Data Rooms")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #e8e8f0;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        # Add button
        add_button = QPushButton("+")
        add_button.setFixedSize(24, 24)
        add_button.setToolTip("Create new data room")
        add_button.clicked.connect(self.show_create_dialog)
        header_layout.addWidget(add_button)

        layout.addLayout(header_layout)

        # Search field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search data rooms...")
        self.search_input.textChanged.connect(self._on_search)
        layout.addWidget(self.search_input)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Status label
        self.status_label = QLabel("0 data rooms")
        self.status_label.setStyleSheet("color: #a0a0b8; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _load_datarooms(self):
        """Load data rooms from storage."""
        settings = get_settings()
        datarooms_file = settings.datarooms_dir / "datarooms.json"

        if datarooms_file.exists():
            try:
                with open(datarooms_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        dr = DataRoom(**item)
                        self._datarooms[dr.id] = dr
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing data rooms file: {e}")
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Failed to parse data rooms file. The file may be corrupted.\n\n{e}",
                )
            except Exception as e:
                logger.error(f"Error loading data rooms: {e}")
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Failed to load data rooms: {e}",
                )

        self._update_list()

    def _save_datarooms(self):
        """Save data rooms to storage."""
        settings = get_settings()
        datarooms_file = settings.datarooms_dir / "datarooms.json"

        try:
            data = [dr.model_dump(mode="json") for dr in self._datarooms.values()]
            with open(datarooms_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving data rooms: {e}")

    def _update_list(self):
        """Update the list widget with current data rooms."""
        self.list_widget.clear()

        search_text = self.search_input.text().lower()

        for dr in self._datarooms.values():
            if search_text and search_text not in dr.name.lower():
                continue

            item = QListWidgetItem(dr.name)
            item.setData(Qt.UserRole, dr.id)
            item.setToolTip(dr.description or f"ID: {dr.id}")
            self.list_widget.addItem(item)

        count = self.list_widget.count()
        self.status_label.setText(f"{count} data room{'s' if count != 1 else ''}")

    def show_create_dialog(self):
        """Show the create data room dialog."""
        dialog = CreateDataRoomDialog(self)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_data()
            self._create_dataroom(data["name"], data["description"])

    def _create_dataroom(self, name: str, description: Optional[str] = None):
        """Create a new data room.

        Args:
            name: Name of the data room.
            description: Optional description.
        """
        try:
            dr = DataRoom.create(name=name, description=description)
            self._datarooms[dr.id] = dr
            self._save_datarooms()

            # Create DataRoom node in Neo4j
            try:
                neo4j = get_neo4j_client()
                neo4j.create_node(
                    labels=["DataRoom"],
                    properties=dr.to_neo4j_properties(),
                )
                logger.info(f"Created DataRoom node in Neo4j: {dr.id}")
            except Exception as e:
                logger.warning(f"Failed to create DataRoom node in Neo4j: {e}")

            self._update_list()

            # Select the new data room
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.data(Qt.UserRole) == dr.id:
                    self.list_widget.setCurrentItem(item)
                    break

            self.dataroom_created.emit(dr.id)
            logger.info(f"Created data room: {dr.id}")

        except Exception as e:
            logger.error(f"Error creating data room: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to create data room: {e}"
            )

    def _delete_dataroom(self, dataroom_id: str):
        """Delete a data room.

        Args:
            dataroom_id: ID of the data room to delete.
        """
        if dataroom_id in self._datarooms:
            dr = self._datarooms[dataroom_id]

            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f'Are you sure you want to delete "{dr.name}"?\n\n'
                "This will remove all associated data from the graph database.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                # Delete from Neo4j first
                try:
                    neo4j = get_neo4j_client()
                    deleted_counts = neo4j.delete_dataroom(dataroom_id)
                    total_deleted = sum(deleted_counts.values())
                    logger.info(f"Deleted {total_deleted} nodes from Neo4j for dataroom {dataroom_id}: {deleted_counts}")
                except Exception as e:
                    logger.error(f"Error deleting dataroom from Neo4j: {e}")
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Failed to delete data from Neo4j database:\n{e}\n\n"
                        "The data room will be removed from the list but data may remain in the database.",
                    )

                # Remove from local storage
                del self._datarooms[dataroom_id]
                self._save_datarooms()
                self._update_list()

                logger.info(f"Deleted data room: {dataroom_id}")

    def refresh(self):
        """Refresh the data room list."""
        self._load_datarooms()

    def get_dataroom(self, dataroom_id: str) -> Optional[DataRoom]:
        """Get a data room by ID.

        Args:
            dataroom_id: ID of the data room.

        Returns:
            DataRoom if found, None otherwise.
        """
        return self._datarooms.get(dataroom_id)

    @Slot(str)
    def _on_search(self, text: str):
        """Handle search text change."""
        self._update_list()

    @Slot(QListWidgetItem)
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle list item click."""
        dataroom_id = item.data(Qt.UserRole)
        self.dataroom_selected.emit(dataroom_id)

    @Slot(QListWidgetItem)
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle list item double-click."""
        # TODO: Open data room settings/details
        pass

    def _show_context_menu(self, position):
        """Show context menu for list item."""
        item = self.list_widget.itemAt(position)
        if not item:
            return

        dataroom_id = item.data(Qt.UserRole)

        menu = QMenu(self)

        open_action = QAction("Open", self)
        open_action.triggered.connect(
            lambda: self.dataroom_selected.emit(dataroom_id)
        )
        menu.addAction(open_action)

        menu.addSeparator()

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(
            lambda: self._delete_dataroom(dataroom_id)
        )
        menu.addAction(delete_action)

        menu.exec(self.list_widget.mapToGlobal(position))
