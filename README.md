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
    tools/
  src/
  requirements.txt
```

The scripts are grouped by responsibility; see [`scripts/README.md`](scripts/README.md) for the map of commands.

AwA2 requires roughly 13 GB of storage. You can either copy `JPEGImages/`
manually into `data/AWA2/JPEGImages/`, or use the preparation script with
`--download`.

## Environment

The command examples below assume that `python` points to the project
environment. If the virtual environment is not active, use `.venv/bin/python`
or activate it first:

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

## Notebook Workflow

The notebooks are consolidated into five maintained entry points:

```text
notebooks/01_data_baseline_xai.ipynb
notebooks/02_stress_concepts_tcav.ipynb
notebooks/03_bottleneck_sanity_report.ipynb
notebooks/04_real_forward_inspection.ipynb
notebooks/05_blog_figures.ipynb
```

Use the maintained notebooks for normal analysis. Heavy execution is behind
explicit flags, so opening a notebook does not accidentally retrain the model
or recompute expensive XAI maps. The forward-inspection notebook exports a
compact trace for the HTML activation simulator, and the blog-figures notebook
generates lightweight assets for the explanatory report.

## Data Preparation

Prepare the full manifest:

```bash
python scripts/data/prepare_awa2.py --data-root data/AWA2
```

Prepare a lightweight debug manifest:

```bash
python scripts/data/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

Create a full portable AwA2 copy resized to `128x128`:

```bash
python scripts/data/prepare_awa2.py --mode subset \
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
python scripts/data/prepare_awa2.py --mode subset \
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

Subset preparation is idempotent only when the request is identical. Passing
`--allow-existing` compares source, classes, seed, ratios, copy/resize mode,
size and JPEG settings with `subset_summary.json`, then validates every reused
destination image. A mismatch fails clearly instead of silently mixing dataset
configurations. Ratios equal to zero are preserved exactly.

Optional download:

```bash
python scripts/data/prepare_awa2.py --data-root data/AWA2 --download
```

Run the general data-pipeline smoke tests on any project manifest:

```bash
python scripts/data/general_tests.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv
```

The checks validate the manifest schema, labels and class mapping, referenced
image files, one batch from each train/validation/test split, and the image
normalization round-trip.

Data preparation is data-only. Gradients are intentionally not tracked here; they will
be enabled explicitly in the later XAI phase.

## Baseline Training

Quick CPU/GPU sanity run without downloading pretrained weights:

```bash
python scripts/training/train_baseline.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --no-pretrained
```

Baseline training with ImageNet pretrained ResNet50:

```bash
python scripts/training/train_baseline.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --batch-size 32 \
  --epochs 20 \
  --early-stopping-patience 3
```

Outputs:

```text
outputs/checkpoints/best_resnet50_awa2.pt
outputs/reports/training_history.csv
```

Training uses augmentation only for the training split and deterministic
Resize/CenterCrop preprocessing for validation and test. The best checkpoint
stores the class mapping, model and optimizer configuration, seed and transform
description. Frozen BatchNorm statistics remain fixed. Checkpoints are loaded
with PyTorch tensor-only safe loading; legacy checkpoints without metadata are
accepted with an explicit warning and shape validation.

## XAI Examples

After training, generate a small local-attribution grid:

```bash
python scripts/experiments/run_xai.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/xai_examples.png \
  --max-images 4 \
  --ig-steps 16
```

Audit examples are selected reproducibly with class-balanced reservoir sampling,
so ordered manifests do not concentrate the analysis on the first class.

Project explainability methods:

```text
Pixel-level local attribution:
  Grad-CAM: Captum LayerGradCam at model.layer4[-1]
  Integrated Gradients: Captum IntegratedGradients from a blurred image baseline

Concept-level explanation:
  TCAV: concept directions in activation space with validation and random-concept baselines
  Concept Bottleneck Model: class prediction mediated by AwA2 semantic attributes
```

`scripts/experiments/run_xai.py` writes a visual grid for Grad-CAM and Integrated
Gradients. The concept-level analyses are produced by the TCAV and CBM scripts.

## Background Stress and Saliency Metrics

AwA2 does not provide segmentation masks. This implementation therefore
uses explicit approximate masks:

```text
center_ellipse: preserve the central elliptical region and perturb the outside
center_box: preserve the central rectangular region and perturb the outside
global: perturb the whole image as a fallback
```

Implemented perturbations:

```text
gaussian_noise: add Gaussian noise only to approximate background pixels
color_shift: invert RGB values only on approximate background pixels
background_swap: replace approximate background pixels with uniform random noise
```

This analysis runs the perturbation suite once: it saves a raw grid with the
original image, approximate background mask and perturbations, then recomputes
the saliency maps and measures their degradation. Use `--mask-strategy global`
as the fallback when the approximate foreground mask is unsuitable for an
image.

Implemented metrics:

```text
IoU top 20%: overlap between the most salient pixels before/after perturbation
Spearman: rank correlation between flattened saliency maps
confidence_delta: change in the fixed saliency-target probability (perturbed - original)
confidence_drop: decrease in the fixed saliency-target probability (original - perturbed)
original/perturbed_confidence: top-1 confidence before/after perturbation
prediction_changed: whether the predicted class changed
```

Run the stress metrics:

```bash
python scripts/experiments/run_background_stress_metrics.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --csv-output outputs/reports/phase5_saliency_metrics.csv \
  --perturbation-figure-output outputs/figures/phase5_perturbations.png \
  --figure-output outputs/figures/phase5_saliency_comparison.png \
  --max-images 4 \
  --xai-methods gradcam integrated_gradients \
  --ig-steps 16 \
  --mask-strategy center_ellipse
```

For a faster run:

```bash
python scripts/experiments/run_background_stress_metrics.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --csv-output outputs/reports/phase5_saliency_metrics_fast.csv \
  --perturbation-figure-output outputs/figures/phase5_perturbations_fast.png \
  --figure-output outputs/figures/phase5_saliency_comparison_fast.png \
  --max-images 2 \
  --xai-methods gradcam \
  --mask-strategy center_ellipse
```

The stress-metrics run produces both visual outputs:

```text
outputs/figures/phase5_perturbations.png
outputs/figures/phase5_saliency_comparison.png
outputs/reports/phase5_saliency_metrics.csv
```

## Contrastive Misclassification Audit

The ordinary error grid targets only the class selected by the network. The
contrastive audit instead fixes the original wrong class and the ground-truth
class on each misclassified image, then follows both scores through the same
interventions.

For logits `z`, the central quantity is the decision margin:

```text
margin(x) = z_wrong(x) - z_true(x)
```

The original error has a positive margin. A positive `margin_reduction` after
an intervention means that the wrong class lost relative evidence. The audit
combines four checks:

```text
target contrast: Grad-CAM and IG for wrong and true targets on the same image
background response: fixed true/wrong probabilities, logits and margin
deletion faithfulness: replace target-ranked pixels with a blurred baseline
CBM comparator: decompose the CBM's linear wrong-vs-true margin by concept
```

Run it with the already trained baseline and CBM checkpoints:

```bash
python scripts/experiments/run_misclassification_audit.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --cbm-checkpoint outputs/checkpoints/phase8_cbm_notebook.pt \
  --metadata-root data/AWA2 \
  --max-images 4 \
  --ig-steps 16 \
  --figure-directory outputs/figures/misclassification_audit
```

This command does not retrain either model. It produces:

```text
outputs/reports/misclassification_decision_audit.csv
outputs/reports/misclassification_audit_summary.csv
outputs/reports/misclassification_deletion_audit.csv
outputs/reports/misclassification_concept_evidence.csv
outputs/reports/misclassification_concept_summary.csv
outputs/figures/misclassification_audit/target_contrast_gradcam.png
outputs/figures/misclassification_audit/target_contrast_integrated_gradients.png
outputs/figures/misclassification_audit/perturbation_decision_margins.png
outputs/figures/misclassification_audit/deletion_curves.png
outputs/figures/misclassification_audit/cbm_concept_evidence.png
```

The CBM figure explains the CBM's semantic pathway, not the separate direct
ResNet. TCAV remains the direct concept probe of the ResNet, but its main score
is cohort-level. The error audit therefore uses Grad-CAM, Integrated Gradients
and score interventions for local ResNet diagnosis, with the CBM as a parallel
semantic comparator.

## Concept Profiles and Prediction Transitions

This analysis moves from pixel-level saliency to AwA2 semantic concepts. It reads
AwA2 attributes such as stripes, horns, hooves, furry, aquatic and color
attributes, then connects prediction flips to concept-level class differences.

Analyze concept profiles:

```bash
python scripts/experiments/analyze_concept_profiles.py \
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

Notebook: `notebooks/02_stress_concepts_tcav.ipynb`

This phase is the bridge toward TCAV: before training Concept Activation
Vectors, the project now has an explicit concept vocabulary and a way to inspect
whether saliency failures correspond to semantic class confusions.

## TCAV

This analysis implements a repeated and validated Testing with Concept
Activation Vectors protocol. AwA2 attributes define concept-positive and
concept-negative classes. The evaluated target class is excluded from CAV
training, deterministic layer activations are cached once, and class-disjoint
CAV train/validation splits are used whenever the concept has enough classes.
Every concept-target pair is fitted over multiple seeds and compared with
matched random CAV controls. The report includes held-out accuracy, run-to-run
variability, confidence intervals, paired permutation tests and corrected
p-values.

Run TCAV:

```bash
python scripts/experiments/run_tcav.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --concepts stripes furry hooves horns flippers \
  --layer layer3 \
  --num-cav-runs 20 \
  --min-valid-runs 5 \
  --score-output outputs/reports/phase7_tcav_scores.csv \
  --run-output outputs/reports/phase7_tcav_runs.csv \
  --cav-output outputs/reports/phase7_cav_summary.csv \
  --coverage-output outputs/reports/phase7_concept_coverage.csv \
  --cav-artifact-output outputs/reports/phase7_cav_vectors.npz \
  --heatmap-output outputs/figures/phase7_tcav_heatmap.png \
  --bar-output outputs/figures/phase7_tcav_top_scores.png
```

Outputs:

```text
outputs/reports/phase7_tcav_scores.csv
outputs/reports/phase7_tcav_runs.csv
outputs/reports/phase7_cav_summary.csv
outputs/reports/phase7_concept_coverage.csv
outputs/reports/phase7_cav_vectors.npz
outputs/reports/phase7_cav_vectors.json
outputs/figures/phase7_tcav_heatmap.png
outputs/figures/phase7_tcav_top_scores.png
```

Notebook: `notebooks/02_stress_concepts_tcav.ipynb`

Interpretation:

```text
high TCAV score        -> the target logit often increases along the learned direction
positive effect size   -> real CAV sensitivity exceeds the matched random-CAV baseline
low corrected p-value  -> the repeated effect is distinguishable from random controls
high held-out accuracy -> the concept direction generalizes beyond its CAV training rows
```

Use `layer3` by default for TCAV. The final `layer4` output in ResNet50 is
followed only by average pooling and the linear classifier, so class-score
gradients can become nearly constant per class and TCAV scores can collapse to
0/1.

After generating the CAV artifact, audit concept sensitivity under the same
background interventions used for saliency maps:

```bash
python scripts/experiments/run_tcav_stress.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --cav-artifact outputs/reports/phase7_cav_vectors.npz \
  --run-output outputs/reports/tcav_stress_runs.csv \
  --summary-output outputs/reports/tcav_stress_summary.csv \
  --figure-output outputs/figures/tcav_stress_effects.png
```

The stress audit reuses each fitted CAV and recomputes only target-logit
gradients. Therefore a score change reflects changed model sensitivity in a
fixed concept direction, not a redefinition of the direction itself.

## Concept Bottleneck Model

This training command fits a simple interpretable-by-design model:

```text
image -> predicted AwA2 concepts -> class
```

The project uses AwA2 class-level semantic attributes as concept supervision.
This means every image from a class receives the same concept vector. It is a
useful bottleneck baseline, but it should be interpreted as class-level concept
supervision rather than image-level concept annotation.

Train the CBM:

```bash
python scripts/experiments/train_cbm.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --backbone-checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --checkpoint-path outputs/checkpoints/phase8_cbm.pt \
  --history-output outputs/reports/phase8_cbm_history.csv \
  --summary-output outputs/reports/phase8_cbm_summary.csv \
  --concept-metrics-output outputs/reports/phase8_concept_metrics.csv \
  --concept-confusion-output outputs/reports/phase8_concept_confusion_matrix.csv \
  --predictions-output outputs/reports/phase8_cbm_predictions.csv \
  --error-analysis-output outputs/reports/phase8_cbm_error_analysis.csv \
  --error-summary-output outputs/reports/phase8_cbm_error_summary.csv \
  --intervention-output outputs/reports/phase8_oracle_prototype_interventions.csv \
  --image-intervention-output outputs/reports/phase8_image_concept_interventions.csv \
  --training-figure-output outputs/figures/phase8_cbm_training.png \
  --summary-figure-output outputs/figures/phase8_cbm_summary.png \
  --concept-figure-output outputs/figures/phase8_concept_prediction_metrics.png \
  --concept-confusion-figure-output outputs/figures/phase8_concept_confusion_matrix.png \
  --intervention-figure-output outputs/figures/phase8_oracle_prototype_interventions.png \
  --image-intervention-figure-output outputs/figures/phase8_image_concept_interventions.png \
  --error-figure-output outputs/figures/phase8_cbm_error_analysis.png \
  --top-concepts 20 \
  --epochs 5
```

Outputs:

```text
outputs/checkpoints/phase8_cbm.pt
outputs/reports/phase8_cbm_history.csv
outputs/reports/phase8_cbm_summary.csv
outputs/reports/phase8_concept_metrics.csv
outputs/reports/phase8_concept_confusion_matrix.csv
outputs/reports/phase8_cbm_predictions.csv
outputs/reports/phase8_cbm_error_analysis.csv
outputs/reports/phase8_cbm_error_summary.csv
outputs/reports/phase8_oracle_prototype_interventions.csv
outputs/reports/phase8_image_concept_interventions.csv
outputs/figures/phase8_cbm_training.png
outputs/figures/phase8_cbm_summary.png
outputs/figures/phase8_concept_prediction_metrics.png
outputs/figures/phase8_concept_confusion_matrix.png
outputs/figures/phase8_oracle_prototype_interventions.png
outputs/figures/phase8_image_concept_interventions.png
outputs/figures/phase8_cbm_error_analysis.png
```

Notebook: `notebooks/03_bottleneck_sanity_report.ipynb`

Interpretation:

```text
high concept accuracy -> the image encoder can recover the selected semantic attributes
high class accuracy   -> the predicted concepts are sufficient for classification
large image-specific intervention -> correcting one predicted concept changes the true-class probability
large prototype intervention      -> the concept-to-class head is sensitive around an AwA2 class prototype
```

The CBM refuses to continue with a missing backbone checkpoint unless
`--use-imagenet-pretrained` is explicitly selected. Its best checkpoint embeds
the class mapping, ordered concept names and indices, architecture settings,
seed and preprocessing description. The training loop also keeps every frozen
BatchNorm module in evaluation mode so its running statistics do not drift.

## Advanced Attribution Audit

This audit evaluates whether attribution maps are faithful, stable,
class-specific and semantically allocated. It is stricter than visual inspection
because it treats each saliency map as a measurable hypothesis.

Run the audit:

```bash
python scripts/audits/run_advanced_attribution_audit.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --methods gradcam integrated_gradients \
  --num-examples 4 \
  --ig-steps 12 \
  --report-output outputs/reports/advanced_attribution_audit.csv \
  --summary-output outputs/reports/advanced_attribution_audit_summary.csv \
  --figure-dir outputs/figures/advanced_attribution_audit
```

Implemented diagnostics:

```text
deletion/insertion curves -> faithfulness of the saliency ranking
animal/background ratio   -> amount of attribution assigned to approximate foreground vs background
class-discriminativeness  -> difference between top-1 and top-2 target explanations
sensitivity to noise      -> explanation stability under small input perturbations
saliency entropy          -> concentration or diffuseness of the saliency distribution
IG baseline comparison    -> dependence on blurred vs black Integrated Gradients baselines
```

Outputs:

```text
outputs/reports/advanced_attribution_audit.csv
outputs/reports/advanced_attribution_audit_summary.csv
outputs/figures/advanced_attribution_audit/*_deletion_insertion.png
outputs/figures/advanced_attribution_audit/*_class_discriminativeness.png
```

Interpretation:

```text
low deletion AUC + high insertion AUC -> attribution identifies evidence the classifier uses
high background saliency ratio        -> possible reliance on contextual shortcuts
high top-1/top-2 map similarity       -> attribution is weakly class-discriminative
low noise-stability metrics           -> explanation is fragile even when prediction is stable
low blurred-vs-black IG similarity    -> Integrated Gradients is strongly baseline-dependent
```

## Real Forward Inspection

This utility records real intermediate tensors from a trained ResNet50 on one
AwA2 image. It attaches hooks to the main ResNet stages, prints compact tensor
statistics, saves a visual summary and exports a JSON trace that can be loaded
by `docs/resnet_activation_simulator.html`.

Run the inspection:

```bash
python scripts/tools/run_forward_inspection.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/real_forward_inspection.png \
  --trace-json outputs/reports/real_forward_trace.json \
  --split test \
  --sample-index 0
```

Outputs:

```text
outputs/figures/real_forward_inspection.png
outputs/reports/real_forward_trace.json
```

Notebook:

```text
notebooks/04_real_forward_inspection.ipynb
```
