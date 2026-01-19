"""Logging configuration with file rotation for Data Room Graph."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: Path, console_level: int = logging.INFO) -> None:
    """Configure application logging with file rotation.

    Creates two log files:
    - app.log: All messages at INFO level and above (5MB x 5 backups)
    - error.log: ERROR level messages only (5MB x 5 backups)

    Args:
        logs_dir: Directory to store log files.
        console_level: Logging level for console output.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log format with timestamp, module, level, file:line, and message
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all; handlers filter

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Main app log (INFO and above, 5MB x 5 backups)
    app_log_path = logs_dir / "app.log"
    app_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)

    # Error log (ERROR level only, 5MB x 5 backups)
    error_log_path = logs_dir / "error.log"
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Log startup message
    logging.info(f"Logging initialized. Log directory: {logs_dir}")
