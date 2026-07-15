# Notebook Layout

The original phase-by-phase notebooks were removed after consolidation. Use
these notebooks for normal work:

1. `01_data_baseline_xai.ipynb` - Phases 1-3: data, baseline, first XAI maps.
2. `02_stress_concepts_tcav.ipynb` - Phases 4-7 plus the advanced attribution audit.
3. `03_bottleneck_sanity_report.ipynb` - Phases 8-9: Concept Bottleneck and saliency sanity checks.
4. `04_real_forward_inspection.ipynb` - Real ResNet50 forward-pass inspection and activation trace export.
5. `05_blog_figures.ipynb` - Lightweight SVG figures used by the explanatory report.

Each notebook now includes explanatory markdown before and after the main code cells. Heavy computation is guarded by explicit `RUN_*` flags.
