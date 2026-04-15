"""Structured logging with Rich."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "success": "bold green",
    }
)

_console = Console(theme=_THEME, stderr=True)
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a named logger with Rich formatting."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=True,
            markup=True,
        )
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _loggers[name] = logger
    return logger


def set_global_level(level: int) -> None:
    """Set logging level for all registered loggers."""
    for logger in _loggers.values():
        logger.setLevel(level)


def get_console() -> Console:
    """Return the shared Rich console."""
    return _console

