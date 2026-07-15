"""Shared utilities for reproducible experiments."""

from __future__ import annotations

import logging
import os
import random
import csv
from pathlib import Path

import numpy as np
import torch


def setup_logging(level: str = "INFO") -> None:
    """Configure a compact, timestamped logger."""
    normalized_level = level.upper()
    if normalized_level not in logging.getLevelNamesMapping():
        raise ValueError(f"Unknown logging level: {level!r}")
    numeric_level = logging.getLevelNamesMapping()[normalized_level]
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


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Write dictionary rows to CSV using the first row's keys as fieldnames."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
