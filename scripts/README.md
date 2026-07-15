# Script guide

Run every command from the project root. The notebooks are the usual guided
workflow; these scripts are the reproducible command-line entry points used by
the notebooks and for unattended runs.

| Folder | Purpose | Commands |
| --- | --- | --- |
| `data/` | Prepare AwA2 data and validate a manifest/data loader. | `prepare_awa2.py`, `general_tests.py` |
| `training/` | Train the image-classification baseline. | `train_baseline.py` |
| `experiments/` | Produce XAI examples, stress metrics, concept analyses, TCAV and CBM results. | `run_xai.py`, `run_background_stress_metrics.py`, `analyze_concept_profiles.py`, `run_tcav.py`, `train_cbm.py` |
| `audits/` | Evaluate attribution quality and saliency sanity. | `run_advanced_attribution_audit.py`, `run_saliency_sanity_audit.py` |
| `tools/` | Inspect a real model forward pass. | `run_forward_inspection.py` |

For the documented arguments and output files, see the phase sections in the
root [`README.md`](../README.md). Each command also supports `--help`.
