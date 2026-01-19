"""Main entry point for the Data Room Graph application."""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Note: QtAsyncio was removed due to conflicts with worker threads
# that use asyncio-based libraries (like unstructured-client SDK).
# The chat widget uses QThread-based fallback for async agent calls.

from .main_window import MainWindow
from ..backend.config import get_settings
from ..backend.database import Neo4jClient, SchemaManager
from ..backend.logging_config import setup_logging

logger = logging.getLogger(__name__)


def setup_high_dpi():
    """Configure high DPI settings for crisp rendering on Windows."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def setup_neo4j() -> bool:
    """Initialize Neo4j connection and schema.

    Returns:
        True if connection successful, False otherwise.
    """
    try:
        client = Neo4jClient()
        if not client.verify_connectivity():
            logger.error("Failed to connect to Neo4j. Is the database running?")
            return False

        schema_manager = SchemaManager(client)
        schema_manager.initialize_schema()
        logger.info("Neo4j schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up Neo4j: {e}")
        return False


def main():
    """Application entry point."""
    # Initialize settings and logging early
    settings = get_settings()
    settings.ensure_app_data_dir()
    setup_logging(settings.logs_dir)

    logger.info("Starting Data Room Graph application...")
    logger.info(f"App data directory: {settings.app_data_dir}")

    # Setup high DPI before creating QApplication
    setup_high_dpi()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Data Room Graph")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("DataRoomGraph")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Load stylesheet
    style_path = Path(__file__).parent / "resources" / "styles.qss"
    if style_path.exists():
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())

    # Try to setup Neo4j (warn if not available but don't block)
    neo4j_available = setup_neo4j()
    if not neo4j_available:
        logger.warning(
            "Neo4j not available. Some features will be disabled. "
            "Run 'scripts/setup_neo4j.py' to start Neo4j."
        )

    # Create and show main window
    window = MainWindow(neo4j_available=neo4j_available)
    window.show()

    # Cleanup handler
    def cleanup():
        from ..backend.database import close_neo4j_client
        close_neo4j_client()
        logger.info("Application closed")

    app.aboutToQuit.connect(cleanup)

    # Run standard Qt event loop
    # Worker threads handle async operations with their own event loops
    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
