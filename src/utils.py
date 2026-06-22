"""Shared utilities: configuration loading and structured logging.

Kept dependency-light so every other module can import it without pulling in
heavy ML libraries at import time.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = parent of the `src` directory this file lives in.
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and resolve all relative paths against the repo root."""
    cfg_path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path to an absolute Path under the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger that writes professional, timestamped lines."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
