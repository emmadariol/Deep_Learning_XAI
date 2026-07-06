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

Run the DataLoader sanity check:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

Run the sanity check on the debug manifest:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest_debug.csv
```

Run the sanity check on a portable subset:

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

## Phase 2 Baseline Training

Quick CPU/GPU sanity run without downloading pretrained weights:

```bash
python scripts/train_baseline.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --no-pretrained
```

Baseline training with ImageNet pretrained ResNet50:

```bash
python scripts/train_baseline.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --batch-size 32 \
  --epochs 5
```

Outputs:

```text
outputs/checkpoints/best_resnet50_awa2.pt
outputs/reports/training_history.csv
```

## Phase 3 XAI Extraction

After training, generate a small XAI grid:

```bash
python scripts/run_xai.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/xai_examples.png \
  --max-images 4 \
  --ig-steps 16
```

For a code-only sanity run from a weak checkpoint, allow misclassified examples:

```bash
python scripts/run_xai.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/xai_smoke_test.png \
  --max-images 2 \
  --ig-steps 4 \
  --allow-incorrect
```

Implemented attribution methods:

```text
input gradient: explicit d(class score) / d(image)
Grad-CAM: gradient of the class score at model.layer4[-1]
Integrated Gradients: explicit gradient loop from a blurred baseline to the image
```

## Phase 4 Background Stress Test

AwA2 does not provide segmentation masks. The Phase 4 implementation therefore
uses explicit approximate masks:

```text
center_ellipse: preserve the central elliptical region and perturb the outside
center_box: preserve the central rectangular region and perturb the outside
global: perturb the whole image as a fallback
```

Run the default background stress test:

```bash
python scripts/run_stress_test.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --figure-output outputs/figures/phase4_stress_test.png \
  --csv-output outputs/reports/phase4_stress_test.csv \
  --max-images 6 \
  --mask-strategy center_ellipse
```

Implemented perturbations:

```text
gaussian_noise: add Gaussian noise only to approximate background pixels
color_shift: invert RGB values only on approximate background pixels
background_swap: replace approximate background pixels with uniform random noise
```

If the approximation is too weak for a specific image, run the fallback global
test:

```bash
python scripts/run_stress_test.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --figure-output outputs/figures/phase4_stress_test_global.png \
  --csv-output outputs/reports/phase4_stress_test_global.csv \
  --max-images 6 \
  --mask-strategy global
```

Outputs:

```text
outputs/figures/phase4_stress_test.png
outputs/reports/phase4_stress_test.csv
```
