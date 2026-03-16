"""
Local logging configuration — no external service required.

Writes structured JSON logs to logs/greenvest.jsonl (one JSON object per line).
Also pretty-prints to console in development.

Usage:
    from greenvest.logging_config import configure_logging
    configure_logging()   # call once at startup
"""
import logging
import os
import sys
from pathlib import Path

import structlog


def configure_logging(log_dir: str = "logs") -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Ensure logs directory exists
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / "greenvest.jsonl"

    # File handler — JSON, one line per log entry
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    # Console handler — human-readable in dev
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
        format="%(message)s",
    )

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # File: JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)

    # Console: coloured key=value pairs
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)
