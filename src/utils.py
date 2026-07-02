"""Shared utilities for reproducible experiments."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def setup_logging(level: str = "INFO") -> None:
    """Configure a compact, timestamped logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible data splits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: str | Path) -> Path:
    """Return an expanded absolute path."""
    return Path(path).expanduser().resolve()

