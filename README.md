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
  notebooks/
  scripts/
  src/
```

AwA2 requires roughly 13 GB of storage. You can either copy `JPEGImages/`
manually into `data/AWA2/JPEGImages/`, or use the preparation script with
`--download`.

Do not commit the full dataset. Keep raw images in `data/`, on an external
drive, or on shared storage; those paths are intentionally ignored by Git.

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

Create a full portable AwA2 copy resized to `128x128`:

```bash
python scripts/create_awa2_subset.py \
  --source-root /path/to/AwA2 \
  --output-root data/AWA2_resized_128 \
  --preset none \
  --num-classes 50 \
  --max-images-per-class 100000 \
  --resize-size 128 \
  --resize-method pad \
  --jpeg-quality 92 \
  --seed 42 \
  --make-zip
```

This keeps all available classes and images, only resizing the image files.
AwA2 metadata/features files are not modified by this image-subset script; keep
the original metadata alongside the resized image folder if needed.

Create a reduced portable subset resized to `128x128`:

```bash
python scripts/create_awa2_subset.py \
  --source-root /path/to/AwA2 \
  --output-root data/AWA2_subset_background20 \
  --preset background20 \
  --max-images-per-class 200 \
  --resize-size 128 \
  --resize-method pad \
  --jpeg-quality 92 \
  --seed 42 \
  --make-zip
```

The subset contains copied images plus `awa2_manifest_subset.csv`,
`class_to_idx_subset.csv`, and `subset_summary.json`. With the command above,
images are saved as `128x128` JPEGs using aspect-ratio preserving padding. The
manifest uses paths relative to the subset folder, and the project DataLoader
resolves those paths from the manifest location.

Optional download:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2 --download
```

Create a tiny synthetic dataset for local smoke tests:

```bash
python scripts/create_sample_awa2.py --output-root sample_data/AWA2
python scripts/prepare_awa2.py \
  --data-root sample_data/AWA2 \
  --manifest-dir sample_data/AWA2 \
  --manifest-name awa2_manifest_sample.csv \
  --class-map-name class_to_idx_sample.csv
python scripts/check_dataloader.py \
  --manifest sample_data/AWA2/awa2_manifest_sample.csv
```

Run the DataLoader smoke test:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

Run the smoke test on the subset:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest_debug.csv
```

Run the smoke test on a portable subset:

```bash
python scripts/check_dataloader.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv
```

Notebook versions:

```text
notebooks/01_phase1_prepare_awa2.ipynb
notebooks/02_phase1_dataloader_smoke_test.ipynb
```

Phase 1 is data-only. Gradients are intentionally not tracked here; they will
be enabled explicitly in the later XAI phase.
