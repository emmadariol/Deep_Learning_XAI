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

## Phase 5 Saliency Metrics

Phase 5 recomputes saliency maps on the original and perturbed images from the
Phase 4 setup, then measures explanation degradation.

Implemented metrics:

```text
IoU top 20%: overlap between the most salient pixels before/after perturbation
Spearman: rank correlation between flattened saliency maps
confidence_delta: prediction confidence change after perturbation
prediction_changed: whether the predicted class changed
```

Run Phase 5:

```bash
python scripts/run_phase5_metrics.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --csv-output outputs/reports/phase5_saliency_metrics.csv \
  --figure-output outputs/figures/phase5_saliency_comparison.png \
  --max-images 4 \
  --xai-methods gradcam integrated_gradients \
  --ig-steps 16 \
  --mask-strategy center_ellipse
```

For a faster run:

```bash
python scripts/run_phase5_metrics.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --csv-output outputs/reports/phase5_saliency_metrics_fast.csv \
  --figure-output outputs/figures/phase5_saliency_comparison_fast.png \
  --max-images 2 \
  --xai-methods gradcam \
  --mask-strategy center_ellipse
```

Inspect the generated metric CSV in a more intuitive way:

```bash
python scripts/inspect_phase5_metrics.py \
  --csv outputs/reports/phase5_saliency_metrics.csv \
  --output-dir outputs/reports
```

This creates summary CSVs and plots such as:

```text
outputs/reports/phase5_metric_summary.csv
outputs/reports/phase5_prediction_transitions.csv
outputs/reports/phase5_mean_iou_by_method.png
outputs/reports/phase5_iou_vs_spearman.png
```

## Phase 6 Concept-Level Analysis

Phase 6 moves from pixel-level saliency to AwA2 semantic concepts. It reads
AwA2 attributes such as stripes, horns, hooves, furry, aquatic and color
attributes, then connects prediction flips to concept-level class differences.

Run Phase 6:

```bash
python scripts/run_phase6_concepts.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --stress-csv outputs/reports/phase5_saliency_metrics.csv \
  --class-profile-output outputs/reports/phase6_class_concepts.csv \
  --transition-output outputs/reports/phase6_concept_transitions.csv \
  --heatmap-output outputs/figures/phase6_class_concept_heatmap.png \
  --transition-figure-output outputs/figures/phase6_concept_transition_examples.png
```

Outputs:

```text
outputs/reports/phase6_class_concepts.csv
outputs/reports/phase6_concept_transitions.csv
outputs/figures/phase6_class_concept_heatmap.png
outputs/figures/phase6_concept_transition_examples.png
```

Notebook:

```text
notebooks/06_phase6_concepts.ipynb
```

This phase is the bridge toward TCAV: before training Concept Activation
Vectors, the project now has an explicit concept vocabulary and a way to inspect
whether saliency failures correspond to semantic class confusions.

## Phase 7 TCAV

Phase 7 implements Testing with Concept Activation Vectors. It uses the AwA2
semantic attributes from Phase 6 to select positive and negative concept
examples, extracts pooled internal activations from the ResNet layer, trains a
linear CAV, and measures whether each concept direction increases a target class
score.

Run Phase 7:

```bash
python scripts/run_phase7_tcav.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --concepts stripes furry hooves horns flippers \
  --layer layer3 \
  --score-output outputs/reports/phase7_tcav_scores.csv \
  --cav-output outputs/reports/phase7_cav_summary.csv \
  --heatmap-output outputs/figures/phase7_tcav_heatmap.png \
  --bar-output outputs/figures/phase7_tcav_top_scores.png
```

Outputs:

```text
outputs/reports/phase7_tcav_scores.csv
outputs/reports/phase7_cav_summary.csv
outputs/figures/phase7_tcav_heatmap.png
outputs/figures/phase7_tcav_top_scores.png
```

Notebook:

```text
notebooks/07_phase7_tcav.ipynb
```

Interpretation:

```text
high TCAV score -> moving along the concept direction tends to increase the target class score
low TCAV score  -> the class score is not consistently sensitive to that concept direction
```

Use `layer3` by default for TCAV. The final `layer4` output in ResNet50 is
followed only by average pooling and the linear classifier, so class-score
gradients can become nearly constant per class and TCAV scores can collapse to
0/1.

## Phase 8 Concept Bottleneck Model

Phase 8 trains a simple interpretable-by-design model:

```text
image -> predicted AwA2 concepts -> class
```

The project uses AwA2 class-level semantic attributes as concept supervision.
This means every image from a class receives the same concept vector. It is a
useful bottleneck baseline, but it should be interpreted as class-level concept
supervision rather than image-level concept annotation.

Run Phase 8:

```bash
python scripts/run_phase8_cbm.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --backbone-checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --checkpoint-output outputs/checkpoints/phase8_cbm.pt \
  --history-output outputs/reports/phase8_cbm_history.csv \
  --concept-metrics-output outputs/reports/phase8_concept_metrics.csv \
  --predictions-output outputs/reports/phase8_cbm_predictions.csv \
  --intervention-output outputs/reports/phase8_concept_interventions.csv \
  --training-figure-output outputs/figures/phase8_cbm_training.png \
  --concept-figure-output outputs/figures/phase8_concept_prediction_metrics.png \
  --intervention-figure-output outputs/figures/phase8_concept_interventions.png \
  --top-concepts 20 \
  --epochs 5
```

Outputs:

```text
outputs/checkpoints/phase8_cbm.pt
outputs/reports/phase8_cbm_history.csv
outputs/reports/phase8_concept_metrics.csv
outputs/reports/phase8_cbm_predictions.csv
outputs/reports/phase8_concept_interventions.csv
outputs/figures/phase8_cbm_training.png
outputs/figures/phase8_concept_prediction_metrics.png
outputs/figures/phase8_concept_interventions.png
```

Notebook:

```text
notebooks/08_phase8_concept_bottleneck.ipynb
```

Interpretation:

```text
high concept accuracy -> the image encoder can recover the selected semantic attributes
high class accuracy   -> the predicted concepts are sufficient for classification
large intervention    -> manually changing a concept strongly affects a class probability
```
