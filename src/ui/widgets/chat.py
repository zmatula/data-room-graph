"""Chat widget for agent interactions."""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QComboBox,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject

# Agent calls use QThread workers to avoid event loop conflicts

from ...backend.agents.investor import InvestorAgent
from ...backend.agents.data_engineer import DataEngineerAgent
from ...backend.agents.enrichment import EnrichmentAgent

logger = logging.getLogger(__name__)


class AgentWorker(QObject):
    """Worker for running async agents in a separate thread (fallback mode)."""

    finished = Signal(str, list)  # message, citations
    error = Signal(str)

    def __init__(self, agent, message: str):
        super().__init__()
        self.agent = agent
        self.message = message

    def run(self):
        """Run the agent in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self.agent.run(self.message))
                self.finished.emit(response.message, response.citations)
            finally:
                loop.close()
        except Exception as e:
            logger.exception("Agent error")
            self.error.emit(str(e))


class MessageBubble(QFrame):
    """A chat message bubble widget."""

    def __init__(
        self,
        message: str,
        is_user: bool = True,
        timestamp: Optional[datetime] = None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the message bubble.

        Args:
            message: Message text.
            is_user: Whether this is a user message.
            timestamp: Message timestamp.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._setup_ui(message, is_user, timestamp)

    def _setup_ui(self, message: str, is_user: bool, timestamp: Optional[datetime]):
        """Setup the bubble UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(message_label)

        # Timestamp
        if timestamp:
            time_label = QLabel(timestamp.strftime("%H:%M"))
            time_label.setStyleSheet("color: #6b6b85; font-size: 10px;")
            layout.addWidget(time_label, alignment=Qt.AlignRight)

        # Style based on sender
        if is_user:
            self.setStyleSheet(
                """
                MessageBubble {
                    background-color: #8b5cf6;
                    border-radius: 12px;
                    margin-left: 40px;
                }
                QLabel {
                    color: #ffffff;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                MessageBubble {
                    background-color: #1f1f3d;
                    border: 1px solid #3d3d6b;
                    border-radius: 12px;
                    margin-right: 40px;
                }
                QLabel {
                    color: #e8e8f0;
                }
                """
            )


class ChatWidget(QWidget):
    """Chat widget for agent interactions."""

    # Signal emitted when a message is sent
    message_sent = Signal(str, str)  # agent_type, message

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._dataroom_id: Optional[str] = None
        self._agents: dict = {}
        self._current_worker: Optional[AgentWorker] = None
        self._current_thread: Optional[QThread] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with agent selector
        header_layout = QHBoxLayout()

        header_label = QLabel("Agent Console")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #e8e8f0;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.agent_selector = QComboBox()
        self.agent_selector.addItems([
            "Investment Professional",
            "Data Engineer",
            "Enrichment Agent",
        ])
        self.agent_selector.setMinimumWidth(180)
        header_layout.addWidget(self.agent_selector)

        layout.addLayout(header_layout)

        # Chat history area
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_area.setStyleSheet(
            """
            QScrollArea {
                background-color: #16213e;
                border: 1px solid #3d3d6b;
                border-radius: 4px;
            }
            QWidget {
                background-color: #16213e;
            }
            """
        )

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch()

        self.chat_area.setWidget(self.chat_container)
        layout.addWidget(self.chat_area, 1)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about the data room...")
        self.input_field.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send)
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Status label
        self.status_label = QLabel("Select a data room to start chatting")
        self.status_label.setStyleSheet("color: #a0a0b8; font-size: 11px;")
        layout.addWidget(self.status_label)

    def set_dataroom(self, dataroom_id: str):
        """Set the current data room.

        Args:
            dataroom_id: ID of the current data room.
        """
        self._dataroom_id = dataroom_id
        self.send_button.setEnabled(True)
        self.status_label.setText(f"Connected to: {dataroom_id}")
        self.clear_chat()

        # Initialize agents for this data room
        self._agents = {
            "investment_professional": InvestorAgent(dataroom_id),
            "data_engineer": DataEngineerAgent(dataroom_id),
            "enrichment_agent": EnrichmentAgent(dataroom_id),
        }

        # Add welcome message
        self.add_message(
            "Hello! I'm your Investment Professional agent. "
            "Ask me anything about the documents in this data room.",
            is_user=False,
        )

    def clear_chat(self):
        """Clear the chat history."""
        # Remove all widgets except the stretch
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def add_message(self, message: str, is_user: bool = True):
        """Add a message to the chat.

        Args:
            message: Message text.
            is_user: Whether this is a user message.
        """
        bubble = MessageBubble(
            message=message,
            is_user=is_user,
            timestamp=datetime.now(),
        )

        # Insert before the stretch
        self.chat_layout.insertWidget(
            self.chat_layout.count() - 1,
            bubble,
            alignment=Qt.AlignRight if is_user else Qt.AlignLeft,
        )

        # Scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def set_thinking(self, is_thinking: bool):
        """Set the thinking/loading state.

        Args:
            is_thinking: Whether the agent is thinking.
        """
        self.input_field.setEnabled(not is_thinking)
        self.send_button.setEnabled(not is_thinking and bool(self._dataroom_id))

        if is_thinking:
            self.status_label.setText("Agent is thinking...")
        else:
            self.status_label.setText(f"Connected to: {self._dataroom_id}")

    @Slot()
    def _on_send(self):
        """Handle send button click."""
        message = self.input_field.text().strip()
        if not message or not self._dataroom_id:
            return

        # Add user message
        self.add_message(message, is_user=True)
        self.input_field.clear()

        # Get selected agent
        agent_type = self.agent_selector.currentText().lower().replace(" ", "_")

        # Emit signal
        self.message_sent.emit(agent_type, message)

        # Get the agent
        agent = self._agents.get(agent_type)
        if not agent:
            self.add_message(f"Agent '{agent_type}' not available.", is_user=False)
            return

        self.set_thinking(True)

        # Call agent using QThread worker
        self._call_agent_threaded(agent, message)

    def _call_agent_threaded(self, agent, message: str):
        """Call agent in a separate thread (fallback mode)."""
        self._current_thread = QThread()
        self._current_worker = AgentWorker(agent, message)
        self._current_worker.moveToThread(self._current_thread)

        self._current_thread.started.connect(self._current_worker.run)
        self._current_worker.finished.connect(self._handle_agent_response)
        self._current_worker.error.connect(self._handle_agent_error)
        self._current_worker.finished.connect(self._cleanup_thread)
        self._current_worker.error.connect(self._cleanup_thread)

        self._current_thread.start()

    def _cleanup_thread(self):
        """Clean up the worker thread."""
        if self._current_thread:
            self._current_thread.quit()
            self._current_thread.wait()
            self._current_thread = None
            self._current_worker = None

    @Slot(str, list)
    def _handle_agent_response(self, message: str, citations: list):
        """Handle successful agent response."""
        self.set_thinking(False)

        # Format response with citations
        response_text = message
        if citations:
            response_text += "\n\n📚 Sources:\n"
            for i, citation in enumerate(citations, 1):
                response_text += f"  [{i}] {citation.document_name}"
                if citation.page:
                    response_text += f", p.{citation.page}"
                response_text += "\n"

        self.add_message(response_text, is_user=False)

    @Slot(str)
    def _handle_agent_error(self, error: str):
        """Handle agent error."""
        self.set_thinking(False)
        self.add_message(f"Error: {error}", is_user=False)
