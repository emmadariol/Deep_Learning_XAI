# Script guide

Run every command from the project root. Standard workflows are orchestrated by
the pipeline wrapper:

```bash
python scripts/run_pipeline.py --profile outputs
```

See [`../PIPELINE.md`](../PIPELINE.md) for the workflow guide. The phase scripts
below expose the individual entry points.

| Folder | Purpose | Commands |
| --- | --- | --- |
| `scripts/` | Run multi-stage workflows. | `run_pipeline.py` |
| `data/` | Prepare AwA2 data and validate a manifest/data loader. | `prepare_awa2.py`, `general_tests.py` |
| `training/` | Train the image-classification baseline. | `train_baseline.py` |
| `experiments/` | Produce XAI examples, stress metrics, concept analyses, TCAV, CBM, and misclassification audits. | `run_xai.py`, `run_background_stress_metrics.py`, `analyze_concept_profiles.py`, `run_tcav.py`, `run_tcav_stress.py`, `train_cbm.py`, `run_misclassification_audit.py` |
| `audits/` | Evaluate attribution stability and faithfulness. | `run_advanced_attribution_audit.py` |

Use these scripts only for custom phase-level runs. Each command supports
`--help` for its arguments and output paths.
