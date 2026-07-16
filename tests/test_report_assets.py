"""Regression tests for the maintained explainability HTML report."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "docs" / "intuitive_explainability.html"


class ExplainabilityReportTests(unittest.TestCase):
    def test_every_local_image_asset_exists(self) -> None:
        html = REPORT.read_text(encoding="utf-8")
        sources = re.findall(r'<img\s+[^>]*src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        self.assertTrue(sources, "Report should contain explanatory images.")
        missing = [
            source
            for source in sources
            if not source.startswith(("http://", "https://", "data:"))
            and not (REPORT.parent / source).resolve().is_file()
        ]
        self.assertEqual(missing, [], f"Missing report image assets: {missing}")

    def test_report_mentions_only_maintained_explanation_methods(self) -> None:
        html = REPORT.read_text(encoding="utf-8").lower()
        removed_methods = (
            "score-cam",
            "expected gradients",
            "smoothgrad",
            "ablation-cam",
            "occlusion attribution",
        )
        for method in removed_methods:
            self.assertNotIn(method, html)
        for method in ("grad-cam", "integrated gradients", "tcav", "concept bottleneck"):
            self.assertIn(method, html)


if __name__ == "__main__":
    unittest.main()
