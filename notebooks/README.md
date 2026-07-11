# Notebook Layout

The original phase-by-phase notebooks were consolidated to reduce clutter.

Use these notebooks for normal work:

1. `01_data_baseline_xai.ipynb` - Phases 1-3: data, baseline, first XAI maps.
2. `02_stress_concepts_tcav.ipynb` - Phases 4-7: perturbation stress tests, metrics, concepts and TCAV.
3. `03_bottleneck_sanity_report.ipynb` - Phases 8-9: Concept Bottleneck and saliency sanity checks.

Each notebook now includes explanatory markdown before and after the main code cells. Heavy computation is guarded by explicit `RUN_*` flags.

The old notebooks are preserved in `archive_phase_notebooks/` for traceability.
