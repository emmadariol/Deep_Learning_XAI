# The Illusion of Saliency Maps

Stress testing saliency and concept-based explanations on AwA2.

## Directory Setup

```text
Deep_Learning_XAI/
  data/
    AWA2/
      JPEGImages/
  docs/
  notebooks/
  outputs/
    checkpoints/
    figures/
    reports/
  sample_data/
  scripts/
    data/
    training/
    experiments/
    audits/
  src/
  requirements.txt
```

AwA2 requires roughly 13 GB of storage. The output-generation workflow expects
the portable subset manifest to already exist at:

```text
data/AWA2_subset_background20/awa2_manifest_subset.csv
```

The dataset is created only by the dedicated data profile.

## Environment

The command examples assume that `python` points to the project environment. If
the virtual environment is not active, use `.venv/bin/python` or activate it
first:

```bash
source .venv/bin/activate
```

For a fresh local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally lists only the direct, portable project
dependencies.

## How To Run The Project

The project can be executed in two ways.

### Option A: reproducible pipeline

Use the pipeline for command-line runs of the maintained workflow. Dataset
creation, baseline training and output generation are separate profiles.

Create or validate the portable dataset subset only when needed:

```bash
python scripts/run_pipeline.py --profile data --source-root /path/to/AwA2
```

Train the baseline only when the checkpoint needs to be created or replaced:

```bash
python scripts/run_pipeline.py --profile train
```

Generate the project outputs from the existing dataset and checkpoint:

```bash
python scripts/run_pipeline.py --profile outputs
```

Available profiles:

```text
data: portable dataset subset
train: baseline ResNet50 checkpoint
outputs: figures and reports from the existing checkpoint
full: outputs plus TCAV and the Concept Bottleneck Model phase
```

To recompute outputs instead of reusing existing files:

```bash
python scripts/run_pipeline.py --profile outputs --force
```

See [`PIPELINE.md`](PIPELINE.md) for the short workflow guide.

### Option B: guided notebooks

Use the notebooks when the goal is to inspect intermediate outputs, follow the
analysis step by step, or regenerate figures manually.

```text
notebooks/01_data_baseline_xai.ipynb
notebooks/02_stress_concepts_tcav.ipynb
notebooks/03_bottleneck_sanity_report.ipynb
notebooks/04_blog_figures.ipynb
```

## Workflow Summary

The project compares visual saliency explanations with concept-based analyses.
The baseline model is a ResNet50 trained on AwA2. Local explanations are
generated with Grad-CAM and Integrated Gradients, then evaluated under
background perturbations, deletion/insertion tests and misclassification
audits.

Concept-level analysis uses AwA2 semantic attributes in three ways:

```text
concept profiles: class-level attribute patterns and prediction transitions
TCAV: concept directions in activation space, validated against random controls
CBM: an interpretable image -> concepts -> class bottleneck model
```

The main outputs are written under:

```text
outputs/checkpoints/
outputs/figures/
outputs/reports/
```

Individual scripts are kept for custom experiments, but they are not the
recommended entry point for normal project execution. See
[`scripts/README.md`](scripts/README.md) only when a specific phase needs to be
run by itself.
