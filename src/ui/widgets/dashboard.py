"""Dashboard widget showing data room status and inventory."""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QPushButton,
)
from PySide6.QtCore import Qt, Slot, Signal

from ...backend.database import get_neo4j_client
from .graph_view import GraphViewWidget

logger = logging.getLogger(__name__)


class StatCard(QFrame):
    """A card widget displaying a statistic."""

    def __init__(
        self,
        title: str,
        value: str = "0",
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ):
        """Initialize the stat card.

        Args:
            title: Card title.
            value: Main value to display.
            subtitle: Optional subtitle/description.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            StatCard {
                background-color: #1f1f3d;
                border: 1px solid #3d3d6b;
                border-radius: 8px;
                padding: 16px;
            }
            StatCard:hover {
                border-color: #8b5cf6;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #a0a0b8; font-size: 12px;")
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #e8e8f0;"
        )
        layout.addWidget(self.value_label)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("color: #6b6b85; font-size: 11px;")
        if not subtitle:
            self.subtitle_label.hide()
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str):
        """Set the displayed value."""
        self.value_label.setText(value)

    def set_subtitle(self, subtitle: str):
        """Set the subtitle text."""
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))


class DashboardWidget(QWidget):
    """Dashboard widget showing data room overview."""

    # Signal emitted when user requests entity extraction
    extract_entities_requested = Signal(str)  # dataroom_id

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._dataroom_id: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        self.header_label = QLabel("Data Room Dashboard")
        self.header_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e8e8f0;"
        )
        layout.addWidget(self.header_label)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.doc_count_card = StatCard("Documents", "0", "Total ingested")
        stats_layout.addWidget(self.doc_count_card)

        self.chunk_count_card = StatCard("Chunks", "0", "Text segments")
        stats_layout.addWidget(self.chunk_count_card)

        self.entity_count_card = StatCard("Entities", "0", "Extracted entities")
        stats_layout.addWidget(self.entity_count_card)

        self.status_card = StatCard("Status", "Ready", "")
        stats_layout.addWidget(self.status_card)

        layout.addLayout(stats_layout)

        # Extract Entities button (shown when docs exist but no entities)
        self.extract_entities_btn = QPushButton("Extract Entities")
        self.extract_entities_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
            QPushButton:pressed {
                background-color: #6d28d9;
            }
            QPushButton:disabled {
                background-color: #4b5563;
                color: #9ca3af;
            }
            """
        )
        self.extract_entities_btn.clicked.connect(self._on_extract_entities)
        self.extract_entities_btn.hide()  # Initially hidden
        layout.addWidget(self.extract_entities_btn)

        # Tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget, 1)

        # Documents tab
        self.documents_table = self._create_documents_table()
        self.tab_widget.addTab(self.documents_table, "Documents")

        # Entities tab
        self.entities_table = self._create_entities_table()
        self.tab_widget.addTab(self.entities_table, "Entities")

        # Graph tab
        self.graph_view = GraphViewWidget()
        self.tab_widget.addTab(self.graph_view, "Graph")

        # Activity tab
        activity_widget = QWidget()
        activity_layout = QVBoxLayout(activity_widget)
        activity_layout.addWidget(
            QLabel("Recent activity will appear here...")
        )
        activity_layout.addStretch()
        self.tab_widget.addTab(activity_widget, "Activity")

        # Refresh graph when switching to Graph tab
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_documents_table(self) -> QTableWidget:
        """Create the documents table."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Filename", "Type", "Chunks", "Status", "Modified"]
        )

        # Configure columns
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        return table

    def _create_entities_table(self) -> QTableWidget:
        """Create the entities table."""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            ["Name", "Type", "Mentions", "Confidence"]
        )

        # Configure columns
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        return table

    def set_dataroom(self, dataroom_id: str):
        """Set the current data room.

        Args:
            dataroom_id: ID of the data room to display.
        """
        self._dataroom_id = dataroom_id
        self.header_label.setText(f"Data Room: {dataroom_id}")
        self.graph_view.set_dataroom(dataroom_id)
        self.refresh()

    def refresh(self):
        """Refresh the dashboard data from Neo4j."""
        if not self._dataroom_id:
            return

        try:
            neo4j = get_neo4j_client()

            # Fetch counts using direct property queries (more reliable)
            stats_result = neo4j.execute_read(
                """
                MATCH (dr:DataRoom {id: $dataroom_id})
                OPTIONAL MATCH (d:Document {dataroom_id: $dataroom_id})
                OPTIONAL MATCH (c:Chunk {dataroom_id: $dataroom_id})
                OPTIONAL MATCH (e:Entity {dataroom_id: $dataroom_id})
                RETURN
                    count(DISTINCT d) as doc_count,
                    count(DISTINCT c) as chunk_count,
                    count(DISTINCT e) as entity_count,
                    dr.ingestion_status as status
                """,
                {"dataroom_id": self._dataroom_id},
            )

            if stats_result:
                row = stats_result[0]
                doc_count = row.get("doc_count", 0)
                chunk_count = row.get("chunk_count", 0)
                entity_count = row.get("entity_count", 0)

                self.doc_count_card.set_value(str(doc_count))
                self.chunk_count_card.set_value(str(chunk_count))
                self.entity_count_card.set_value(str(entity_count))
                status = row.get("status", "Ready") or "Ready"
                self.status_card.set_value(status.capitalize())

                # Show extract button if docs exist but no entities
                if doc_count > 0 and entity_count == 0:
                    self.extract_entities_btn.show()
                    self.extract_entities_btn.setText("Extract Entities")
                    self.extract_entities_btn.setEnabled(True)
                else:
                    self.extract_entities_btn.hide()

            # Fetch documents using dataroom_id property
            self.documents_table.setRowCount(0)
            docs_result = neo4j.execute_read(
                """
                MATCH (d:Document {dataroom_id: $dataroom_id})
                RETURN d.filename as filename,
                       d.doc_type as doc_type,
                       d.chunk_count as chunk_count,
                       d.ingestion_status as status,
                       d.updated_at as updated_at
                ORDER BY d.filename ASC
                LIMIT 100
                """,
                {"dataroom_id": self._dataroom_id},
            )

            logger.info(f"Dashboard refresh: found {len(docs_result or [])} documents for {self._dataroom_id}")

            for doc in docs_result or []:
                updated = doc.get("updated_at", "")
                if updated and "T" in str(updated):
                    updated = str(updated).split("T")[0]
                self.add_document_row(
                    filename=doc.get("filename", "Unknown"),
                    doc_type=doc.get("doc_type", "Unknown"),
                    chunk_count=doc.get("chunk_count", 0) or 0,
                    status=doc.get("status", "unknown") or "unknown",
                    modified=str(updated),
                )

            # Fetch entities
            self.entities_table.setRowCount(0)
            entities_result = neo4j.execute_read(
                """
                MATCH (e:Entity {dataroom_id: $dataroom_id})
                RETURN e.canonical_name as name,
                       e.entity_type as entity_type,
                       e.mention_count as mentions,
                       e.confidence as confidence
                ORDER BY e.mention_count DESC
                LIMIT 100
                """,
                {"dataroom_id": self._dataroom_id},
            )

            for entity in entities_result or []:
                self.add_entity_row(
                    name=entity.get("name", "Unknown"),
                    entity_type=entity.get("entity_type", "Unknown"),
                    mentions=entity.get("mentions", 0) or 0,
                    confidence=entity.get("confidence", 0.0) or 0.0,
                )

            logger.info(f"Dashboard refreshed: {len(docs_result or [])} docs, {len(entities_result or [])} entities")

            # Refresh graph view if it's the current tab
            if self.tab_widget.currentWidget() == self.graph_view:
                self.graph_view.refresh()

        except Exception as e:
            logger.error(f"Failed to refresh dashboard: {e}", exc_info=True)

    def update_stats(
        self,
        documents: int = 0,
        chunks: int = 0,
        entities: int = 0,
        status: str = "Ready",
    ):
        """Update the dashboard statistics.

        Args:
            documents: Document count.
            chunks: Chunk count.
            entities: Entity count.
            status: Current status.
        """
        self.doc_count_card.set_value(str(documents))
        self.chunk_count_card.set_value(str(chunks))
        self.entity_count_card.set_value(str(entities))
        self.status_card.set_value(status)

    def add_document_row(
        self,
        filename: str,
        doc_type: str,
        chunk_count: int,
        status: str,
        modified: str,
    ):
        """Add a row to the documents table.

        Args:
            filename: Document filename.
            doc_type: Document type classification.
            chunk_count: Number of chunks.
            status: Processing status.
            modified: Last modified date.
        """
        row = self.documents_table.rowCount()
        self.documents_table.insertRow(row)

        self.documents_table.setItem(row, 0, QTableWidgetItem(filename))
        self.documents_table.setItem(row, 1, QTableWidgetItem(doc_type))
        self.documents_table.setItem(row, 2, QTableWidgetItem(str(chunk_count)))
        self.documents_table.setItem(row, 3, QTableWidgetItem(status))
        self.documents_table.setItem(row, 4, QTableWidgetItem(modified))

    def add_entity_row(
        self,
        name: str,
        entity_type: str,
        mentions: int,
        confidence: float,
    ):
        """Add a row to the entities table.

        Args:
            name: Entity name.
            entity_type: Entity type.
            mentions: Number of mentions.
            confidence: Confidence score.
        """
        row = self.entities_table.rowCount()
        self.entities_table.insertRow(row)

        self.entities_table.setItem(row, 0, QTableWidgetItem(name))
        self.entities_table.setItem(row, 1, QTableWidgetItem(entity_type))
        self.entities_table.setItem(row, 2, QTableWidgetItem(str(mentions)))
        self.entities_table.setItem(
            row, 3, QTableWidgetItem(f"{confidence:.1%}")
        )

    def _on_extract_entities(self):
        """Handle Extract Entities button click."""
        if self._dataroom_id:
            self.extract_entities_requested.emit(self._dataroom_id)

    def set_extraction_running(self, running: bool):
        """Update UI to reflect extraction running state.

        Args:
            running: Whether extraction is currently running.
        """
        if running:
            self.extract_entities_btn.setText("Extracting...")
            self.extract_entities_btn.setEnabled(False)
        else:
            self.extract_entities_btn.setText("Extract Entities")
            self.extract_entities_btn.setEnabled(True)

    @Slot(int)
    def _on_tab_changed(self, index: int):
        """Handle tab change.

        Args:
            index: Index of the newly selected tab.
        """
        # Refresh graph when switching to Graph tab
        if self.tab_widget.widget(index) == self.graph_view and self._dataroom_id:
            self.graph_view.refresh()
