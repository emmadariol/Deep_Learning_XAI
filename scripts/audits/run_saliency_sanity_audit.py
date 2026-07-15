"""Compatibility entry point for the retired local-attribution sanity script.

The maintained project compares two pixel-level methods, Grad-CAM and
Integrated Gradients, and two concept-level methods, TCAV and the Concept
Bottleneck Model. Use ``run_advanced_attribution_audit.py`` for local
attribution stability and faithfulness checks.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This legacy audit entry point has been retired. Use "
        "scripts/audits/run_advanced_attribution_audit.py --methods gradcam "
        "integrated_gradients."
    )


if __name__ == "__main__":
    main()
