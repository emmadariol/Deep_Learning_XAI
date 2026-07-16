"""Reusable command-line configuration validators."""

from __future__ import annotations

import argparse
import math

import torch


STANDARD_LOG_LEVELS = frozenset(
    {"CRITICAL", "FATAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "NOTSET"}
)


def _integer(value: str | int, *, minimum: int, expectation: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {value!r}") from error
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {value!r}")
    return parsed


def positive_int(value: str | int) -> int:
    return _integer(value, minimum=1, expectation="a positive integer")


def nonnegative_int(value: str | int) -> int:
    return _integer(value, minimum=0, expectation="a non-negative integer")


def at_least_two_int(value: str | int) -> int:
    return _integer(value, minimum=2, expectation="an integer greater than or equal to 2")


def _finite_float(value: str | float, *, expectation: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {value!r}") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {value!r}")
    return parsed


def positive_float(value: str | float) -> float:
    parsed = _finite_float(value, expectation="a positive finite number")
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive finite number, got {value!r}")
    return parsed


def nonnegative_float(value: str | float) -> float:
    parsed = _finite_float(value, expectation="a non-negative finite number")
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative finite number, got {value!r}"
        )
    return parsed


def open_unit_float(value: str | float) -> float:
    parsed = _finite_float(value, expectation="a finite number in (0, 1)")
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(f"expected a finite number in (0, 1), got {value!r}")
    return parsed


def open_percentage_float(value: str | float) -> float:
    parsed = _finite_float(value, expectation="a finite percentage in (0, 100)")
    if not 0.0 < parsed < 100.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite percentage in (0, 100), got {value!r}"
        )
    return parsed


def log_level(value: str) -> str:
    normalized = value.upper()
    if normalized not in STANDARD_LOG_LEVELS:
        raise argparse.ArgumentTypeError(f"unknown logging level: {value!r}")
    return normalized


def device_spec(value: str) -> str:
    if value == "auto":
        return value
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid PyTorch device: {value!r}") from error
    if device.type not in {"cpu", "cuda", "mps"}:
        raise argparse.ArgumentTypeError(
            f"unsupported device type {device.type!r}; use auto, cpu, cuda or mps"
        )
    return value
