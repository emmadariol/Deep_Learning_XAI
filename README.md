# The Illusion of Saliency Maps

Stress testing Grad-CAM and Integrated Gradients on AwA2.

## Directory Setup

```text
Deep_Learning_XAI/
  configs/
  data/
    AWA2/
      JPEGImages/
  outputs/
    checkpoints/
    figures/
    reports/
  scripts/
  src/
```

AwA2 requires roughly 13 GB of storage. You can either copy `JPEGImages/`
manually into `data/AWA2/JPEGImages/`, or use the preparation script with
`--download`.

## Phase 1

Prepare the full manifest:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2
```

Prepare a lightweight debug manifest:

```bash
python scripts/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

Optional download:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2 --download
```

Run the DataLoader smoke test:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

Run the smoke test on the subset:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest_debug.csv
```
