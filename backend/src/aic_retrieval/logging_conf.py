"""Project logging setup.

Replaces the ad-hoc ``print(...)`` calls in the old scripts. Call
``setup_logging()`` once at process start (CLI / API startup); use
``get_logger(__name__)`` everywhere else.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger once (idempotent)."""
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
