# Notebook Layout

The original phase-by-phase notebooks were removed after consolidation. Use
these notebooks for normal work:

1. `01_data_baseline_xai.ipynb` - Data preparation, baseline training, and first XAI maps.
2. `02_stress_concepts_tcav.ipynb` - Background stress, concept profiles, TCAV, and the advanced attribution audit.
3. `03_bottleneck_sanity_report.ipynb` - Concept Bottleneck training and saliency sanity checks.
4. `04_blog_figures.ipynb` - Real ResNet50 forward inspection and lightweight figures used by the explanatory report.

Each notebook now includes explanatory markdown before and after the main code cells. Heavy computation is guarded by explicit `RUN_*` flags.
