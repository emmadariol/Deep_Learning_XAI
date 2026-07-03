# Saliency Stress Test

Minimal PyTorch pipeline for Oxford-IIIT Pet classification, to support later Grad-CAM and Integrated Gradients analysis.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Prepare Oxford-IIIT Pet:

```bash
python3 scripts/prepare_oxford_pets.py --data-root data/OxfordPets --download
```

Optional debug manifest:

```bash
python3 scripts/prepare_oxford_pets.py \
  --data-root data/OxfordPets \
  --max-classes 8 \
  --max-images-per-class 80 \
  --manifest-name oxford_pets_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

## Train Baseline

Full manifest:

```bash
python3 scripts/train_baseline.py \
  --manifest data/OxfordPets/oxford_pets_manifest.csv \
  --batch-size 32 \
  --epochs 20 \
  --patience 5 \
  --lr 1e-4
```

Debug manifest:

```bash
python3 scripts/train_baseline.py \
  --manifest data/OxfordPets/oxford_pets_manifest_debug.csv \
  --batch-size 16 \
  --epochs 5 \
  --patience 2
```

## Current Code

```text
src/data.py                  Dataset and DataLoaders
src/model.py                 ResNet50 classifier
src/train.py                 Training loop and checkpointing
src/utils.py                 Logging, seeding, path helpers
scripts/prepare_oxford_pets.py
scripts/train_baseline.py
notebooks/project_flow.ipynb Minimal function-call flow
```

## XAI Examples

After training the baseline, generate Grad-CAM and Integrated Gradients examples:

```bash
python3 scripts/run_xai.py \
  --manifest data/OxfordPets/oxford_pets_manifest.csv \
  --checkpoint outputs/checkpoints/best_resnet50_oxford_pets.pt \
  --output outputs/figures/xai_examples.png \
  --max-images 4
```

Outputs:

```text
outputs/checkpoints/best_resnet50_oxford_pets.pt
outputs/reports/training_history_resnet50_oxford_pets.csv
```
