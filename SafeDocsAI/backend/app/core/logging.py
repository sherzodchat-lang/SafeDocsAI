"""
Logging configuration for AndozAI.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None
) -> None:
    """
    Setup application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    # Create handlers
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name."""
    return logging.getLogger(name)
