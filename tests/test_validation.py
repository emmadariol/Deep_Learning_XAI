"""Regression tests for command-line validators on supported Python versions."""

from __future__ import annotations

import argparse
import unittest

from src.validation import log_level


class LogLevelValidationTests(unittest.TestCase):
    def test_standard_level_is_normalized_without_new_logging_apis(self) -> None:
        self.assertEqual(log_level("info"), "INFO")
        self.assertEqual(log_level("warning"), "WARNING")

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            log_level("verbose")


if __name__ == "__main__":
    unittest.main()
