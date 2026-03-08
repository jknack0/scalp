"""Structured logging setup using structlog."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def configure_logging(
    log_level: str | None = None,
    log_file: str | None = "logs/bot.log",
    json_console: bool = False,
) -> None:
    """Configure structlog with dual output: JSON file + colored console.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path for JSON log file. None disables file logging.
        json_console: If True, console also outputs JSON (for systemd/journal).
    """
    import os
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors for all outputs
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Root logger setup
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler
    console_renderer = (
        structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty(), pad_level=False)
        if not json_console
        else structlog.processors.JSONRenderer()
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # File handler (JSON)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10_485_760,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Get a logger bound with a component name."""
    return structlog.get_logger(component=component)
