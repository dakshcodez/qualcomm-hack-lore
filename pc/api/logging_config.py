"""Phase 2 — structured logging setup for the FastAPI backend: rotating
file handler + console handler, with per-request IDs threaded through log
lines via a LoggerAdapter, so logs stay greppable under demo pressure.

Input: optional log file path / level / rotation-size overrides.
Output: configure_logging() returns the configured "lore" logger;
get_request_logger() returns a per-request LoggerAdapter.
Side effects: attaches handlers to the "lore" logger (idempotent — repeat
calls reset handlers rather than stacking them); creates the log file's
parent directory if needed.
"""

import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "lore"
DEFAULT_LOG_FILE = "lore.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 3

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


class _DefaultRequestIdFilter(logging.Filter):
    """Guarantees every record has a request_id attribute (falls back to
    "-" for log lines emitted outside a request context via get_request_logger),
    so the shared formatter never raises a KeyError."""

    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(
    log_file=DEFAULT_LOG_FILE,
    level=logging.DEBUG,
    max_bytes=DEFAULT_MAX_BYTES,
    backup_count=DEFAULT_BACKUP_COUNT,
):
    """Configure the "lore" logger with a rotating file handler (if
    log_file is not None) plus a console handler.

    Side effects: clears and replaces any handlers already attached to the
    "lore" logger (safe to call more than once); creates log_file's parent
    directory if it doesn't exist.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)
    request_id_filter = _DefaultRequestIdFilter()

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    logger.addHandler(console_handler)

    return logger


def new_request_id():
    """Generate a short, greppable per-request ID."""
    return uuid.uuid4().hex[:12]


def get_request_logger(request_id=None):
    """Return a LoggerAdapter on the "lore" logger that stamps every log
    line with `request_id` (auto-generated via new_request_id() if omitted).
    """
    request_id = request_id or new_request_id()
    return logging.LoggerAdapter(logging.getLogger(LOGGER_NAME), {"request_id": request_id})
