"""Logging configuration utility for Senti Market Intelligence."""

import logging
import sys
from typing import Optional
from config.settings import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Configures application-wide logging with standard output formatting."""
    if log_level is None:
        log_level = get_settings().LOG_LEVEL

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger instance."""
    return logging.getLogger(name)
