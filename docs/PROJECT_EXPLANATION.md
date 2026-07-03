# The Illusion of Saliency Maps

## 1. Project Summary

This project builds a PyTorch experimental pipeline to critically evaluate two post-hoc Explainable AI methods for Computer Vision:

- Grad-CAM
- Integrated Gradients

The case study uses Animals with Attributes 2, abbreviated as AwA2, a dataset containing animal images from 50 classes. The objective is not only to classify the animals correctly, but to stress test the explanations produced by the model and verify whether the saliency maps are actually tied to animal morphology or instead to spurious correlations, especially the background.

The experimental hypothesis is:

> if a saliency map changes drastically when only the background is perturbed while the model prediction stays unchanged, then the explanation is not stable with respect to the main semantic content of the image.

In other words, the project aims to show quantitatively and visually that some post-hoc explanations can create a false sense of reliability: they may appear to point to the right object, while still being sensitive to non-causal signals.

## 2. Motivation

Saliency maps are often used to interpret deep learning models for image classification. In an ideal case, if a ResNet predicts "zebra", we would expect the saliency map to highlight:

- the animal body;
- the head;
- the legs;
- coat texture;
- relevant morphological patterns.

However, convolutional models can learn spurious correlations:

- grass associated with herbivores;
- snow associated with polar animals;
- water associated with marine mammals;
- forest backgrounds associated with some wild animals;
- background colors and textures associated with a class.

This is particularly important for AwA2 because many animal classes are photographed in recurring natural contexts. If the model uses the background as a shortcut, a saliency map can become visually convincing but conceptually fragile.

## 3. Technical Goal

The technical goal is to build a complete pipeline that:

1. prepares the AwA2 dataset;
2. trains a ResNet50 baseline on the 50 classes;
3. generates Grad-CAM and Integrated Gradients for correctly predicted images;
4. selectively perturbs the background;
5. recomputes predictions and saliency maps;
6. measures how much the explanations change;
7. produces CSV reports and visual comparison grids for analysis or a blog post.

The central aim is not to achieve the highest possible accuracy, but to obtain a baseline that is strong enough to make the XAI maps and stress test meaningful.

## 4. Project Structure

The intended structure is:

```text
Deep_Learning_XAI/
  configs/
  data/
    AWA2/
      JPEGImages/
      awa2_manifest.csv
      class_to_idx.csv
      awa2_manifest_debug.csv
      class_to_idx_debug.csv
  docs/
    PROJECT_EXPLANATION.md
  notebooks/
    01_phase1_prepare_awa2.ipynb
    02_phase1_dataloader_smoke_test.ipynb
  outputs/
    checkpoints/
    figures/
    reports/
  scripts/
    prepare_awa2.py
    check_dataloader.py
    train_baseline.py
    run_xai.py
  src/
    __init__.py
    data.py
    model.py
    train.py
    xai.py
    utils.py
  README.md
  requirements.txt
```

At the moment, Phases 1-3 have been implemented:

- manifest preparation;
- custom PyTorch Dataset;
- standard ResNet transforms;
- DataLoader construction;
- DataLoader sanity checking;
- lightweight debug subset mode.
- ResNet50 baseline training;
- checkpoint saving;
- Grad-CAM extraction;
- input-gradient saliency;
- Integrated Gradients from a blurred baseline;
- notebook walkthroughs for the current data, training and XAI pipeline.

The next phases are the stress-test interventions and quantitative degradation metrics.

## 5. Dataset: AwA2

AwA2 contains JPEG images organized by class:

```text
data/AWA2/JPEGImages/
  antelope/
  grizzly+bear/
  killer+whale/
  ...
```

Each subdirectory represents a class. The script `scripts/prepare_awa2.py` scans these directories and produces a CSV manifest with the columns:

```text
filepath,label,class_name,split
```

Conceptual example:

```text
/path/to/JPEGImages/zebra/zebra_10001.jpg,49,zebra,train
```

The class-to-index mapping is saved separately in:

```text
data/AWA2/class_to_idx.csv
```

This makes the encoding of the 50 classes explicit and reproducible.

## 6. Debug Subset Strategy

AwA2 is roughly 13 GB. Using the full dataset immediately would slow down development, debugging, and iteration. For this reason, Phase 1 includes a subset mode:

```bash
python scripts/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

This mode makes it possible to:

- validate the code quickly;
- inspect transforms and DataLoaders;
- test training on a small number of classes;
- develop Grad-CAM and Integrated Gradients without excessive compute cost.

The selection is deterministic and seed-driven, so debug experiments are reproducible.

## 7. Phase 1: Data Preparation and DataLoader

### 7.1 Preparation Script

File:

```text
scripts/prepare_awa2.py
```

Responsibilities:

- find `JPEGImages/`;
- optionally download AwA2;
- collect images by class;
- generate train/validation/test splits;
- create the CSV manifest;
- create the class-to-index mapping;
- support debug subsets.

Default split:

```text
train: 70%
val:   15%
test:  15%
```

The split is performed within each class, so every split preserves a balanced distribution over the selected classes.

### 7.2 PyTorch Dataset

File:

```text
src/data.py
```

Main class:

```python
AwA2Dataset
```

Each item returns:

```python
image_tensor, label, class_name, filepath
```

The `filepath` is kept because it will be useful in the XAI and stress-test phases, where images, heatmaps, and reports must be associated with the original sample.

### 7.3 Transforms

The transforms follow the standard preprocessing used for ImageNet-pretrained ResNet models:

```text
Resize(128)
CenterCrop(128)
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

The project keeps ImageNet mean and standard deviation because Phase 2 uses a
ResNet50 architecture. The resized subset uses `128x128` inputs to keep the
training and XAI loops lightweight.

### 7.4 Sanity Check

File:

```text
scripts/check_dataloader.py
```

Checks:

- number of images per split;
- number of classes;
- batch shape;
- normalized tensor statistics;
- denormalized tensor statistics;
- pre-normalization range.
- examples from the stable class mapping.

Notebook versions are also provided:

```text
notebooks/01_phase1_prepare_awa2.ipynb
notebooks/02_phase1_dataloader_smoke_test.ipynb
```

Phase 1 is intentionally data-only. No gradients are tracked here; gradients will be enabled explicitly in the later XAI phase when Grad-CAM and Integrated Gradients are implemented.

## 8. Phase 2: Baseline Training and Fine-Tuning

This phase has a minimal implementation.

Goal:

- load a ResNet50 pretrained on ImageNet;
- replace the final layer with a 50-class head;
- freeze the early blocks;
- train the later blocks, especially `layer3`, `layer4`, and `fc`;
- optimize Cross-Entropy loss;
- save the best checkpoint;
- use Early Stopping.

Implemented architecture:

```text
src/model.py
src/train.py
scripts/train_baseline.py
```

Planned checkpoint:

```text
outputs/checkpoints/best_resnet50_awa2.pt
```

Minimum metrics:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- best epoch;
- early stopping counter.

## 9. Phase 3: XAI Extraction

This phase has a minimal implementation.

Methods:

- Grad-CAM;
- Integrated Gradients.

For Grad-CAM, the target layer will be:

```python
model.layer4[-1]
```

For Integrated Gradients, the baseline will not be a black image. Instead, it will be a blurred baseline produced by applying extreme Gaussian Blur to the original image.

Reason:

- a black baseline introduces an artificial reference point;
- it can drastically alter brightness and color distribution;
- for natural images, a blurred baseline preserves average color and illumination;
- the comparison becomes more coherent with the visual content of the image.

Output:

```text
outputs/figures/xai_examples/
```

Each figure should compare:

- original image;
- Grad-CAM overlay;
- Integrated Gradients overlay;
- true class;
- predicted class;
- confidence.

## 10. Phase 4: Stress Test

This phase will be implemented after confirmation.

The objective is to perturb the background while preserving the animal as much as possible.

Main approach:

- use `torchvision.models.detection.maskrcnn_resnet50_fpn`;
- try to obtain an approximate animal mask;
- apply perturbations only to pixels outside the mask.

Planned perturbations:

1. Gaussian Noise on the background;
2. Color Shift through RGB channel inversion on the background;
3. Background Swap with uniform noise.

Known issue:

Mask R-CNN is trained on COCO, which does not contain all AwA2 classes. Some animals may not be segmented correctly.

Fallback:

- if the mask fails, use controlled global perturbations;
- record in the report that the sample used the fallback;
- still analyze saliency stability with respect to non-semantic input changes.

This does not invalidate the project: if the map changes substantially even under non-semantic perturbations, the critical point remains valid.

## 11. Phase 5: Quantitative Metrics

This phase will be implemented after confirmation.

For every original and perturbed image:

1. compute the model prediction;
2. check whether the prediction is preserved;
3. compute the original saliency;
4. compute the perturbed saliency;
5. compare the two maps.

### 11.1 Saliency IoU

Take the top 20% most salient pixels in the original map and in the perturbed map. Convert each map into a binary mask.

Formula:

```text
IoU = area(intersection) / area(union)
```

Interpretation:

- high IoU: the explanation remains spatially similar;
- low IoU: the explanation moves;
- low IoU with unchanged prediction: possible explanation instability.

### 11.2 Spearman Rank Correlation

Flatten the saliency tensors and compute Spearman rank correlation.

Interpretation:

- high correlation: the pixel-importance ordering remains similar;
- low or negative correlation: the importance hierarchy changes;
- low correlation with unchanged prediction: the explanation is not robust.

## 12. Final Report

Planned output:

```text
outputs/reports/stress_test_results.csv
```

Planned columns:

```text
image_id
filepath
class_name
true_label
pred_original
pred_perturbed
confidence_original
confidence_perturbed
prediction_preserved
perturbation_type
mask_status
xai_method
saliency_iou_top20
spearman_correlation
notes
```

Planned figures:

```text
outputs/figures/stress_test_grids/
```

Each grid should show:

- original image;
- perturbed image;
- original Grad-CAM;
- perturbed Grad-CAM;
- original Integrated Gradients;
- perturbed Integrated Gradients.

## 13. Success Criteria

The project succeeds if it produces evidence that:

- the model often preserves the same prediction after background perturbation;
- the XAI maps change substantially;
- IoU between original and perturbed saliency maps decreases;
- Spearman correlation between original and perturbed saliency maps decreases;
- visualizations show saliency shifts toward non-semantic or unstable regions.

The goal is not to prove that Grad-CAM or Integrated Gradients are always useless. The claim is narrower: in this setting, they can be fragile and misleading if interpreted naively as causal explanations.

## 14. Experimental Risks

### 14.1 Low Accuracy

If the baseline does not learn well enough, the XAI maps will be less informative. Mitigations:

- use more images;
- increase epochs;
- unfreeze more layers;
- check the learning rate;
- verify class mapping and normalization.

### 14.2 Failed Segmentation

Mask R-CNN may fail to segment many AwA2 animals. Mitigations:

- use the global fallback;
- save `mask_status`;
- do not hide the failure, but include it in the critical analysis.

### 14.3 Integrated Gradients Cost

Integrated Gradients requires many forward passes. Mitigations:

- use a GPU;
- limit the number of images;
- temporarily reduce `n_steps` in debug mode;
- use small batches;
- compute Integrated Gradients only for correctly predicted images.

### 14.4 Over-Interpreting Saliency

Saliency maps are not automatically causal explanations. The project must avoid overly strong claims. The correct framing is:

> the observed saliency maps are unstable under controlled perturbations, so they should not be naively interpreted as proof that the model uses animal morphology.

## 15. Phase 1 Commands

Prepare the full manifest:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2
```

Prepare the debug manifest:

```bash
python scripts/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

Sanity check the full manifest:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

Sanity check the debug manifest:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest_debug.csv
```

## 16. Current Status

Implemented:

- directory setup;
- Phase 1;
- debug subset mode;
- resized portable subset generation;
- initial documentation;
- Phase 2 baseline training;
- Phase 3 XAI extraction with explicit gradients.

Not implemented yet:

- Phase 4 stress test;
- Phase 5 metrics and report.

The next phase, after confirmation, will be Phase 4.
