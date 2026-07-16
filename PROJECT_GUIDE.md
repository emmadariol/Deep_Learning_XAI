# Complete Technical Guide to the Explainability Project

## The Illusion of Saliency Maps: local attribution, stress testing, and concept-level explanations on AwA2

This document explains the complete project implemented in this repository. It is both a conceptual guide and an operational reference. It covers the research question, AwA2 data preparation, manifest-based loading, ResNet50 architecture and training, prediction flow, Grad-CAM, Integrated Gradients, background perturbations, quantitative attribution audits, semantic attribute processing, TCAV, the Concept Bottleneck Model, the forward-pass simulator, notebooks, output files, current results, reproducibility, limitations, and the conclusions that the project can legitimately support.

The maintained explanation methods are:

| Explanation level | Method | Main question |
| --- | --- | --- |
| Local and spatial | Grad-CAM | Which coarse regions support a selected class through the last convolutional representation? |
| Local and input-level | Integrated Gradients | Which input dimensions contribute to a target score relative to a blurred reference image? |
| Concept-level and post-hoc | Validated TCAV | Is a class score consistently more sensitive to a learned human concept direction than to matched random directions? |
| Concept-level and by design | Concept Bottleneck Model | Can class prediction be forced through an explicit vector of human-readable concepts? |

The central rule of the project is:

> A heatmap is a hypothesis about model evidence, not proof of model reasoning.

The hypothesis must be tested through controlled interventions, prediction changes, map stability, faithfulness curves, target-class comparisons, concept probes, and explicit model structure.

### Finalized implementation status

- The maintained explanation set contains exactly four methods: Grad-CAM, Integrated Gradients, validated TCAV, and the Concept Bottleneck Model.
- The data pipeline now derives and validates the label-to-class mapping from the manifest, uses stochastic augmentation only for training, preserves deterministic validation/test transforms, and validates existing subset configurations instead of silently reusing incompatible data.
- Frozen BatchNorm modules remain in evaluation mode during baseline and CBM training, so their running statistics do not drift.
- Baseline and CBM checkpoints include the class mapping, model configuration, seed, transforms, and, for the CBM, the ordered concept vocabulary and backbone provenance. Tensor checkpoints are loaded through the safe weights-only path where supported.
- The validated TCAV implementation supports held-out CAV validation, repeated seeds, matched random controls, permutation inference, multiple-testing correction, and reusable CAV artifacts. The TCAV CSV currently present in `outputs/reports`, however, is still the earlier exploratory schema and is identified as legacy throughout the report pipeline.
- The CBM was retrained from the AwA2 baseline checkpoint. The current results expose a strong distribution mismatch when exact AwA2 class targets are injected into a class head trained on predicted concept vectors; this intervention must not be described as an oracle upper bound.

---

## Table of contents

1. Research objective
2. Explainability terminology
3. Complete experimental logic
4. Repository architecture
5. AwA2 dataset
6. Manifest and class mapping
7. Dataset preparation
8. PyTorch Dataset and DataLoaders
9. Image preprocessing
10. ResNet50 architecture
11. Baseline training
12. Prediction flow
13. Saliency maps and heatmaps
14. Grad-CAM
15. Integrated Gradients
16. Tensor diagnostics
17. Background perturbation stress test
18. Stress-test metrics
19. Advanced attribution audit
20. AwA2 semantic concepts
21. Concept profiles and prediction transitions
22. Validated TCAV concept sensitivity
23. Concept Bottleneck Model
24. Concept interventions
25. Real forward-pass inspection
26. Notebook workflow
27. Complete command-line workflow
28. Output files
29. Current result snapshot
30. Interpretation of correct and incorrect predictions
31. Reproducibility
32. Validation and tests
33. Known limitations
34. Claims supported by the project
35. Recommended presentation flow
36. Reproducible execution and report publication
37. Glossary
38. References

---

## 1. Research objective

The project investigates a recurring problem in explainable computer vision:

> A classifier can produce the correct animal label while using contextual correlations, and an attractive saliency map can hide this failure.

Animal datasets contain recurring environmental correlations. Marine animals often appear with water. Polar bears often appear with snow or ice. Antelopes and zebras often appear on grassland. A deep classifier can combine real animal morphology with contextual shortcuts such as color, vegetation, water, sky, snow, or photographic composition.

The project tests three related hypotheses:

1. A correct prediction does not guarantee that the model used the correct semantic evidence.
2. A visually plausible saliency map can be unstable or weakly faithful.
3. Pixel attribution alone is insufficient for a complete explainability analysis.

The research direction is therefore cumulative:

~~~text
AwA2 images and semantic metadata
        |
        v
Manifest, stable labels, deterministic splits
        |
        v
ImageNet-pretrained ResNet50 fine-tuning
        |
        v
Predictions, logits, and softmax probabilities
        |
        +----------------------+
        |                      |
        v                      v
Grad-CAM              Integrated Gradients
        |                      |
        +-----------+----------+
                    |
                    v
Background perturbation stress test
                    |
                    v
Stability, faithfulness, and target-specificity metrics
                    |
                    v
AwA2 semantic concept profiles
                    |
        +-----------+-----------+
        |                       |
        v                       v
Validated TCAV probe   Concept Bottleneck Model
        |                       |
        +-----------+-----------+
                    |
                    v
Conservative final interpretation
~~~

The project does not try to prove that Grad-CAM or Integrated Gradients are useless. It demonstrates why local attribution methods must be audited instead of accepted from visual appearance alone.

---

## 2. Explainability terminology

### 2.1 Explainability

Explainability is the set of methods and evidence used to help a human inspect, understand, challenge, compare, or intervene on model behavior.

There is no single universal explanation. Every method defines a different notion of importance.

### 2.2 Interpretability

Interpretability usually refers to how directly a human can understand the internal decision mechanism. A linear model over named concepts is more directly interpretable than a ResNet50 over millions of opaque features.

### 2.3 Local explanation

A local explanation describes one input and one selected output. Grad-CAM and Integrated Gradients are local methods.

A local explanation must specify:

- the input image;
- the model checkpoint;
- the target class;
- the explained layer or baseline;
- the normalization and visualization rule.

### 2.4 Global explanation

A global explanation describes behavior across a set of inputs, classes, perturbations, or concepts. Aggregated stability metrics, TCAV matrices, and CBM concept statistics are more global than a single heatmap.

### 2.5 Post-hoc explanation

A post-hoc method is applied to an already trained model without changing its prediction mechanism.

In this project:

- Grad-CAM is post-hoc;
- Integrated Gradients is post-hoc;
- TCAV is post-hoc.

### 2.6 Interpretable by design

An interpretable-by-design model includes a human-readable structure in its prediction path.

The Concept Bottleneck Model uses:

~~~text
image -> predicted semantic concepts -> animal class
~~~

The concept vector is therefore part of the model, rather than an explanation attached afterward.

### 2.7 Attribution is not causality

An attribution value is not automatically:

- a segmentation mask;
- a causal effect;
- a direct representation of model reasoning;
- proof that the highlighted pixels are necessary;
- proof that unhighlighted pixels are irrelevant;
- a guarantee that the explanation is stable.

This distinction motivates the perturbation and faithfulness experiments.

---

## 3. Complete experimental logic

The project asks four progressively stronger questions.

### Question 1: where is evidence located?

Grad-CAM and Integrated Gradients produce spatial maps for a selected class.

### Question 2: is the map stable and faithful?

Background perturbations, top-k IoU, Spearman correlation, deletion, insertion, noise sensitivity, and class-discriminativeness test the map.

### Question 3: which semantic concepts are represented?

AwA2 attributes and validated TCAV directional sensitivity connect model activations to concepts such as stripes, horns, hooves, fur, or flippers.

### Question 4: can prediction be mediated by concepts?

The Concept Bottleneck Model changes the prediction path so the class head receives only predicted concepts.

These questions are complementary. A local heatmap cannot replace concept analysis, and concept sensitivity cannot replace a spatial explanation.

---

## 4. Repository architecture

~~~text
Deep_Learning_XAI/
  data/
    AWA2/
    AWA2_subset_background20/
  docs/
    intuitive_explainability.html
    resnet_activation_simulator.html
    assets/xai-report/
  notebooks/
    01_data_baseline_xai.ipynb
    02_stress_concepts_tcav.ipynb
    03_bottleneck_sanity_report.ipynb
    04_real_forward_inspection.ipynb
    05_blog_figures.ipynb
  outputs/
    checkpoints/
    figures/
    reports/
  scripts/
    data/
    training/
    experiments/
    audits/
    tools/
  src/
    data.py
    model.py
    train.py
    xai.py
    perturb.py
    metrics.py
    attribution_audit.py
    concepts.py
    tcav.py
    bottleneck.py
    forward_inspection.py
    utils.py
  tests/
  README.md
  requirements.txt
~~~

### Responsibility boundaries

| Location | Responsibility |
| --- | --- |
| src | Reusable model, data, XAI, metric, concept, and inspection logic |
| scripts | Reproducible command-line entry points |
| notebooks | Guided analysis and immediate visualization |
| outputs | Generated checkpoints, CSV reports, and figures |
| docs | Interactive final report and simulator |
| tests | Regression checks for critical data and training behavior |

The source code defines the maintained implementation. The outputs directory can contain old generated files from earlier experiments. Generated legacy files are not evidence that an old algorithm remains supported.

---

## 5. AwA2 dataset

Animals with Attributes 2, abbreviated AwA2, provides animal images and class-level semantic attributes.

### 5.1 Image data

The full image dataset contains 50 animal classes under:

~~~text
JPEGImages/<class_name>/
~~~

The complete archive requires roughly 13 GB of storage.

### 5.2 Maintained 20-class subset

The project commonly uses a portable subset containing:

~~~text
antelope
blue+whale
bobcat
buffalo
dolphin
elephant
giant+panda
giraffe
grizzly+bear
hippopotamus
humpback+whale
leopard
lion
polar+bear
seal
sheep
tiger
walrus
wolf
zebra
~~~

The current validated subset contains approximately:

| Split | Images |
| --- | ---: |
| Train | 2782 |
| Validation | 596 |
| Test | 596 |
| Total | 3974 |

### 5.3 Semantic metadata

AwA2 provides files such as:

~~~text
classes.txt
predicates.txt
predicate-matrix-continuous.txt
predicate-matrix-binary.txt
~~~

The predicate matrix has shape:

~~~text
number of classes x number of semantic concepts
~~~

Concept examples include:

~~~text
stripes, furry, hooves, horns, flippers, paws, claws,
ocean, water, vegetation, hunter, stalker, swims, tail
~~~

### 5.4 Critical semantic limitation

AwA2 concepts are class-level attributes.

Every image with the same class label receives the same concept vector. The metadata does not state whether a specific image visibly contains a concept.

For example, every zebra image receives the zebra attribute prototype even if:

- the stripes are partly hidden;
- the legs are occluded;
- the animal is very small;
- the image is dominated by background.

This limitation affects both TCAV sample construction and CBM supervision.

---

## 6. Manifest and class mapping

### 6.1 What a manifest is

A manifest is a CSV index connecting each image to its label, class name, and split.

Required columns:

~~~csv
filepath,label,class_name,split
JPEGImages/antelope/antelope_0001.jpg,0,antelope,train
JPEGImages/antelope/antelope_0002.jpg,0,antelope,val
JPEGImages/zebra/zebra_0001.jpg,19,zebra,test
~~~

The manifest separates dataset organization from model logic.

### 6.2 Why the manifest matters

Without a manifest, labels can accidentally depend on directory traversal order. A checkpoint output index can then be interpreted as the wrong animal.

The manifest fixes:

- the image path;
- the integer label;
- the human class name;
- the train, validation, or test split.

### 6.3 Class map

The adjacent class map contains:

~~~csv
class_name,label
antelope,0
...
zebra,19
~~~

The project validates both directions:

- one label cannot map to two class names;
- one class name cannot map to two labels;
- labels must be contiguous from zero;
- the class map must match the manifest.

### 6.4 Portable paths

A full manifest can contain absolute paths. A portable subset uses paths relative to the manifest directory.

Relative paths are resolved against the manifest folder. The entire subset can therefore be moved without rewriting every row.

---

## 7. Dataset preparation

The preparation script supports two modes.

### 7.1 Full-manifest mode

Manifest mode indexes an existing AwA2 image tree.

It can:

- locate JPEGImages recursively;
- optionally download the official archive;
- select all or a limited number of classes;
- limit images per class;
- create deterministic train, validation, and test splits;
- write the manifest and class map.

### 7.2 Subset mode

Subset mode creates a self-contained directory containing:

~~~text
JPEGImages/
awa2_manifest_subset.csv
class_to_idx_subset.csv
subset_summary.json
README_subset.md
~~~

It supports:

- explicit class names;
- class lists from a file;
- background20 and debug10 presets;
- random class selection;
- copy, hardlink, or symlink;
- optional image resizing;
- optional ZIP generation.

### 7.3 Resize behavior

The documented subset uses:

~~~text
stored size:     128 x 128
resize method:   aspect-ratio preserving padding
JPEG quality:    92
padding RGB:     124, 116, 104
~~~

Padding preserves geometry. Crop can remove image edges. Stretch can distort morphology.

The later ResNet preprocessing still transforms stored images to 224 x 224. Upsampling a 128 x 128 stored image cannot restore lost detail.

### 7.4 Split ratios

Defaults:

~~~text
train = 0.70
val   = 0.15
test  = 0.15
~~~

Ratios must be non-negative and sum to one.

Splitting is performed independently per class, preserving representation of each selected animal.

### 7.5 Determinism

The script uses a seeded Python random generator. The same source files, class selection, limits, and seed produce the same image selection and split.

### 7.6 Idempotent subset generation

Without allow-existing, subset generation refuses to write into a non-empty destination.

With allow-existing, the script first compares the stored summary with the
requested source directory, selected classes, seed, ratios, copy mode, resize
size and method, JPEG quality and per-class limit. Any mismatch fails before
files are reused. Existing copies and links are checked against their source;
resized images are decoded and checked for expected dimensions and RGB mode.

### 7.7 Zero-ratio handling

Counts use a largest-remainder allocation per class. A split whose requested
ratio is zero always receives zero images. Non-empty allocation is attempted
only for splits with positive ratios and only when the class has enough images.

### 7.8 Image validation

Every selected source image is decoded through Pillow during preparation,
including copy and link modes. Unreadable files fail with their exact path.
Reused resized destinations are additionally verified and checked for expected
dimensions and RGB mode.

---

## 8. PyTorch Dataset and DataLoaders

### 8.1 Dataset output

ImageManifestDataset returns:

~~~python
(image_tensor, integer_label, class_name, filepath)
~~~

Each image is opened through Pillow and converted to RGB.

### 8.2 Tensor shape

One image has shape:

~~~text
3 x 224 x 224
~~~

One batch has shape:

~~~text
B x 3 x 224 x 224
~~~

PyTorch uses channel-first order:

~~~text
channel 0 = red
channel 1 = green
channel 2 = blue
~~~

### 8.3 DataLoader behavior

| Split | Shuffle | Augmentation |
| --- | --- | --- |
| Train | Yes | Stochastic |
| Validation | No | Deterministic |
| Test | No | Deterministic |

Drop-last is false, so all samples are used.

Pinned host memory and non-blocking device transfer are enabled when appropriate for CUDA.

---

## 9. Image preprocessing

### 9.1 Training transform

~~~python
RandomResizedCrop(224, scale=(0.75, 1.0))
RandomHorizontalFlip(p=0.5)
ToTensor()
Normalize(ImageNet mean, ImageNet std)
~~~

RandomResizedCrop varies framing and object scale. Horizontal flipping reduces orientation dependence.

### 9.2 Validation and test transform

~~~python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(ImageNet mean, ImageNet std)
~~~

Evaluation is deterministic so metrics do not change because of random crops.

### 9.3 ImageNet normalization

For each channel:

$$
x_{norm}=\frac{x-\mu}{\sigma}
$$

with:

~~~text
mean = 0.485, 0.456, 0.406
std  = 0.229, 0.224, 0.225
~~~

These are the statistics expected by ImageNet-pretrained ResNet50 weights.

Visualization applies:

$$
x=x_{norm}\sigma+\mu
$$

The project logs and checks the inverse-normalization range.

---

## 10. ResNet50 architecture

### 10.1 Library implementation

The network is provided by:

~~~python
torchvision.models.resnet50
~~~

The project does not manually reimplement ResNet50.

### 10.2 Residual learning

A residual block computes:

$$
y=\operatorname{ReLU}(F(x)+S(x))
$$

where:

- F is the learned residual transformation;
- S is the identity or a projection shortcut.

Residual shortcuts improve gradient propagation in deep networks.

### 10.3 Bottleneck block

A ResNet50 bottleneck contains:

1. a 1 x 1 convolution;
2. Batch Normalization and ReLU;
3. a 3 x 3 convolution;
4. Batch Normalization and ReLU;
5. a 1 x 1 expansion convolution;
6. Batch Normalization;
7. shortcut addition;
8. final ReLU.

The torchvision implementation places downsampling stride in the 3 x 3 convolution of the first bottleneck in later stages. This variant is often called ResNet v1.5.

### 10.4 Exact tensor flow

For a 224 x 224 input:

| Stage | Operation | Output shape | Blocks |
| --- | --- | --- | ---: |
| Input | normalized RGB | B x 3 x 224 x 224 | - |
| conv1 | 7 x 7, 64 filters, stride 2, padding 3 | B x 64 x 112 x 112 | - |
| maxpool | 3 x 3, stride 2, padding 1 | B x 64 x 56 x 56 | - |
| layer1 | bottleneck residual stage | B x 256 x 56 x 56 | 3 |
| layer2 | first block downsamples | B x 512 x 28 x 28 | 4 |
| layer3 | first block downsamples | B x 1024 x 14 x 14 | 6 |
| layer4 | first block downsamples | B x 2048 x 7 x 7 | 3 |
| avgpool | global spatial average | B x 2048 x 1 x 1 | - |
| flatten | remove spatial singleton axes | B x 2048 | - |
| fc | linear classifier | B x C | - |

The model receives a full 224 x 224 image. The 14 x 14 and 7 x 7 grids are internal representations created by stride and pooling.

There is no decoder and no upsampling inside the classifier. Grad-CAM upsampling happens only after attribution.

### 10.5 Classifier replacement

The original 1000-class ImageNet fully connected layer is replaced with:

~~~python
Linear(2048, number_of_manifest_classes)
~~~

The number of classes is inferred from contiguous manifest labels.

### 10.6 Logits and softmax

The final linear layer produces logits z.

Softmax computes:

$$
p(y=c\mid x)=\frac{e^{z_c}}{\sum_j e^{z_j}}
$$

The top prediction is:

$$
\hat y=\arg\max_c p(y=c\mid x)
$$

A high softmax value means one logit dominates the alternatives. It does not prove correctness, calibration, or valid evidence.

---

## 11. Baseline training

### 11.1 Transfer learning

Baseline training uses ImageNet-pretrained weights unless no-pretrained is explicitly requested.

The actual training script fine-tunes:

~~~text
layer4
fc
~~~

Earlier stages remain frozen.

### 11.2 Cross-Entropy loss

The classifier uses nn.CrossEntropyLoss:

$$
\mathcal{L}_{CE}(x,y)=
-\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)
$$

Cross-Entropy expects raw logits. Softmax must not be applied before the loss.

### 11.3 Optimizer

The project uses AdamW over trainable parameters only.

Defaults:

~~~text
learning rate = 1e-4
weight decay  = 1e-4
batch size    = 32
epochs        = 5
~~~

### 11.4 Training epoch

Each training batch performs:

1. move images and labels to the selected device;
2. clear gradients;
3. forward pass;
4. Cross-Entropy calculation;
5. backward pass;
6. optimizer step;
7. accumulate loss and accuracy.

Validation performs a forward pass without gradients or parameter updates.

### 11.5 Frozen Batch Normalization

Freezing parameters does not automatically freeze BatchNorm running means and variances.

After model.train, the baseline training loop forces a BatchNorm module back to evaluation mode when all its local parameters are frozen.

This prevents hidden updates in early frozen stages.

### 11.6 Overfitting controls

The baseline uses:

- transfer learning;
- frozen early stages;
- random crop augmentation;
- horizontal flipping;
- AdamW weight decay;
- deterministic validation;
- best-validation checkpoint selection.

The standard baseline ResNet50 does not add dropout.

### 11.7 Early stopping and best checkpoint

A checkpoint is saved whenever validation accuracy improves. In addition,
patience-based early stopping terminates training after the configured number of
validation epochs without an improvement larger than `min_delta`. A patience
value of zero disables stopping while retaining best-model checkpointing.

### 11.8 Baseline checkpoint contents

The baseline checkpoint contains:

~~~text
model_state_dict
epoch
val_acc
val_loss
metadata.schema
metadata.idx_to_class
metadata.class_to_idx
metadata.model_config
metadata.training_config
metadata.transforms
metadata.seed
~~~

Optimizer state is intentionally omitted because this checkpoint is an
evaluation artifact rather than a resumable training snapshot. The loader uses
`weights_only=True` and validates the embedded class mapping when the caller
provides the expected manifest mapping. Legacy state-dict checkpoints can still
be evaluated, but produce an explicit metadata warning.

---

## 12. Prediction flow

For each image:

~~~text
normalized tensor
    -> ResNet50
    -> class logits
    -> softmax probabilities
    -> top-1 label and confidence
~~~

### 12.1 Correct examples

A correct example satisfies:

~~~text
predicted label = true label
~~~

The standard XAI script selects correct examples by default.

### 12.2 Incorrect examples

An incorrect example satisfies:

~~~text
predicted label != true label
~~~

For an error, two target explanations are possible:

- predicted-class explanation: why did the wrong class receive support?
- true-class explanation: what evidence supported or failed to support the correct class?

These are different questions. The target label must always be shown.

### 12.3 Class-balanced selection

The selector groups candidate indices by class, shuffles reproducibly, interleaves classes, and applies a per-class cap.

This prevents ordered manifests from selecting only the first class.

---

## 13. Saliency maps and heatmaps

A saliency map is a numerical spatial tensor assigning attribution to positions for a selected output.

A heatmap is the colored rendering of that tensor.

An overlay blends the heatmap with the original image.

~~~text
saliency map = numerical values
heatmap      = color representation
overlay      = heatmap plus RGB image
~~~

The colors are relative inside each independently normalized map. Red does not have a universal magnitude across images or methods.

---

## 14. Grad-CAM

### 14.1 Purpose

Grad-CAM explains a target class through deep convolutional feature maps.

The maintained target layer is:

~~~python
model.layer4[-1]
~~~

Its native map is approximately 7 x 7.

### 14.2 Mathematical definition

Let A superscript k be feature map k and S subscript c be the target-class logit.

The channel weight is:

$$
\alpha_k^c=
\frac{1}{Z}\sum_i\sum_j
\frac{\partial S_c}{\partial A_{ij}^k}
$$

The spatial explanation is:

$$
L_{GradCAM}^c=
\operatorname{ReLU}
\left(\sum_k\alpha_k^c A^k\right)
$$

ReLU retains positive support for the selected class and discards negative evidence.

### 14.3 Implementation flow

1. Clone the normalized input.
2. Enable input gradients.
3. Run the forward pass.
4. Backpropagate the selected class score.
5. capture gradients at layer4[-1].
6. average each channel gradient spatially;
7. weight the feature maps;
8. apply ReLU;
9. bilinearly upsample to 224 x 224;
10. min-max normalize to zero-one.

Captum LayerGradCam and LayerAttribution interpolation perform the core operations.

### 14.4 Interpretation

A high Grad-CAM value means that deep feature maps active at that location had a positive gradient-weighted relationship with the target logit.

It does not mean the original pixel alone caused the prediction.

### 14.5 Main limitations

- Native spatial resolution is coarse.
- Upsampling makes the map look smoother than its real 7 x 7 information content.
- ReLU hides negative evidence.
- The result depends on the selected layer.
- Different class targets can produce similar maps.
- A plausible animal-shaped map can still fail deletion or stability tests.

---

## 15. Integrated Gradients

### 15.1 Purpose

Integrated Gradients accumulates target gradients along a path from a reference image to the real image.

It addresses saturation, where the final local gradient can be small even though a feature was important earlier along the input path.

### 15.2 Mathematical definition

For input component i:

$$
IG_i(x)=
(x_i-x'_i)
\int_0^1
\frac{\partial F_c(x'+\alpha(x-x'))}{\partial x_i}
\,d\alpha
$$

where:

- x is the real image;
- x prime is the baseline;
- F subscript c is the target-class score;
- alpha moves from baseline to input.

The integral is approximated with a finite number of steps.

### 15.3 Blurred-image baseline

The project does not use a black baseline for the main IG map.

It creates an image-specific strongly blurred reference.

Default blur radius:

~~~text
18.0
~~~

Procedure:

1. reverse ImageNet normalization;
2. clamp values to zero-one;
3. convert to a Pillow RGB image;
4. apply Gaussian blur;
5. convert back to a tensor;
6. reapply ImageNet normalization.

This preserves average brightness and color better than a black image while removing much texture and morphology.

### 15.4 Implementation flow

1. Generate the blurred baseline.
2. Interpolate path points between baseline and input.
3. Evaluate target gradients at path points.
4. Numerically integrate.
5. Multiply by input minus baseline.
6. take absolute attribution magnitude;
7. sum across RGB channels;
8. min-max normalize the spatial map.

Defaults:

~~~text
main XAI steps         = 50
stress-test IG steps   = 16
internal batch size    = 4
blur radius            = 18.0
~~~

The reduced stress-test step count is a computation-quality tradeoff.

### 15.5 Completeness

Signed IG approximately satisfies:

$$
\sum_i IG_i(x)
\approx
F_c(x)-F_c(x')
$$

The displayed map uses absolute values and channel summation, so the normalized visualization no longer preserves signed completeness.

### 15.6 Main limitations

- Results depend strongly on the baseline.
- Results depend on integration resolution.
- Absolute visualization hides positive and negative contributions.
- Linear path points can be visually unrealistic.
- Independent min-max normalization can exaggerate a weak map.
- Fine-grained attributions can be noisy.

---

## 16. Tensor diagnostics

The project logs intermediate tensor statistics:

~~~text
shape
minimum
maximum
mean
standard deviation
~~~

Expected Grad-CAM map:

~~~text
B x 1 x 224 x 224
~~~

Expected Integrated Gradients tensors:

~~~text
blurred baseline: B x 3 x 224 x 224
raw attribution:  B x 3 x 224 x 224
saliency map:     B x 1 x 224 x 224
~~~

### 16.1 Map normalization

For each sample:

$$
\widetilde M=
\frac{M-\min(M)}
{\max(M)-\min(M)+\epsilon}
$$

Per-sample normalization enables visualization but removes absolute-scale comparability.

Raw attribution norms and tensor statistics should be inspected before comparing methods.

### 16.2 Input gradients

The source module retains a plain input-gradient function for debugging. It is not one of the four maintained explanation methods used for the main project conclusions.

---

## 17. Background perturbation stress test

### 17.1 Purpose

The stress test changes pixels treated as background while preserving an approximate central foreground.

It asks:

- does the prediction change?
- does the probability of the original target change?
- does the saliency map move?
- do different attribution methods react differently?

### 17.2 Why approximate masks are needed

AwA2 does not provide segmentation masks.

The current implementation uses:

| Strategy | Protected region | Purpose |
| --- | --- | --- |
| center_ellipse | centered ellipse | default animal proxy |
| center_box | centered rectangle | alternative proxy |
| global | no protected region | fallback global intervention |

### 17.3 Ellipse mask

Coordinates are normalized to minus one through one.

For foreground scale s:

$$
\left(\frac{x}{s}\right)^2+
\left(\frac{y}{s}\right)^2
\leq 1
$$

Default:

~~~text
s = 0.68
~~~

The complement is treated as background.

If the animal is off-center or extends outside the ellipse, animal pixels are perturbed. The mask must always be displayed and inspected.

### 17.4 General perturbation equation

Let M subscript bg be the background mask and q be a replacement rule:

$$
x_{pert}=
(1-M_{bg})\odot x+
M_{bg}\odot q(x)
$$

Perturbations are applied in denormalized RGB space and then re-normalized.

### 17.5 Gaussian noise

For background pixels:

$$
q(x)=
\operatorname{clip}(x+\epsilon,0,1),
\qquad
\epsilon\sim\mathcal N(0,\sigma^2)
$$

Default:

~~~text
sigma = 0.25
~~~

### 17.6 Color shift

The implemented color shift is RGB inversion:

$$
q(x)=1-x
$$

It strongly changes color while preserving spatial boundaries.

### 17.7 Background swap

Background pixels are replaced with uniform random values:

$$
q(x)\sim\mathcal U(0,1)
$$

This destroys natural background texture and is the strongest synthetic intervention.

### 17.8 Reproducibility

A seeded torch Generator creates random perturbations.

### 17.9 Fixed explanation target

Original and perturbed saliency maps are computed for the original predicted class.

This prevents target switching from being confused with explanation instability.

The perturbed top-1 prediction is recorded separately.

---

## 18. Stress-test metrics

### 18.1 Prediction change

~~~text
prediction_changed =
original_prediction != perturbed_prediction
~~~

A high rate can indicate context dependence, mask failure, severe distribution shift, or a combination.

### 18.2 Fixed-target probability

For original target class c:

$$
\Delta p_c=
p_c(x_{pert})-p_c(x)
$$

The reported drop is:

$$
confidence\_drop=
p_c(x)-p_c(x_{pert})
$$

This compares the same target before and after perturbation.

### 18.3 Top-20% saliency IoU

The implementation selects the exact top fraction with a stable descending sort.

For original support S subscript o and perturbed support S subscript p:

$$
IoU=
\frac{|S_o\cap S_p|}
{|S_o\cup S_p|}
$$

Interpretation:

~~~text
high IoU = similar most-salient spatial support
low IoU  = the highlighted region moved
~~~

IoU ignores ordering inside the selected set.

### 18.4 Spearman rank correlation

The complete maps are flattened. Tie-aware ranks are calculated and correlated:

$$
\rho=
\operatorname{corr}
(R(M_o),R(M_p))
$$

Interpretation:

~~~text
rho near 1  = similar global ranking
rho near 0  = little rank agreement
rho near -1 = reversed ranking
~~~

Constant or non-finite maps produce undefined correlation and return NaN.

### 18.5 Why both metrics are needed

IoU measures overlap of the strongest support.

Spearman measures the complete ordering.

A map can preserve its broad ranking while moving its top hotspot, or preserve top overlap while changing the remainder.

These are stability metrics, not direct proof of faithfulness.

---

## 19. Advanced attribution audit

### 19.1 Deletion

Start from the original image and progressively replace the most salient pixels with a blurred baseline.

A faithful ranking should cause the target probability to fall quickly.

Lower deletion AUC is generally favorable.

### 19.2 Insertion

Start from the blurred baseline and progressively restore the most salient original pixels.

A faithful ranking should recover target probability quickly.

Higher insertion AUC is generally favorable.

### 19.3 Faithfulness gap

$$
gap=
AUC_{insertion}-
AUC_{deletion}
$$

A larger positive gap is favorable, but remains conditional on the replacement baseline.

### 19.4 Approximate animal/background allocation

Using the geometric mask:

$$
r_{animal}=
\frac{\sum M\odot M_{animal}}
{\sum M}
$$

$$
r_{background}=
\frac{\sum M\odot M_{background}}
{\sum M}
$$

These values describe a geometric proxy, not true segmentation.

### 19.5 Saliency entropy

The map is converted to a spatial probability distribution:

$$
p_i=\frac{M_i}{\sum_j M_j}
$$

Normalized entropy:

$$
H(M)=
-\frac{\sum_i p_i\log p_i}
{\log N}
$$

High entropy means diffuse saliency. Low entropy means concentrated saliency. Neither is universally correct.

### 19.6 Sensitivity to small noise

Small Gaussian noise is added to normalized input tensors.

Original and noisy maps are compared with IoU and Spearman. The audit also checks whether the model prediction remained unchanged.

If prediction stays fixed but explanation changes substantially, the explanation is less stable than the decision.

### 19.7 Class discriminativeness

For the same image, explanations are computed for top-1 and top-2 predicted classes.

High map similarity can indicate generic object localization instead of class-specific evidence.

Related classes can legitimately share evidence, so this is a diagnostic rather than a binary test.

### 19.8 IG baseline comparison

Integrated Gradients is computed using:

- the maintained blurred baseline;
- a normalized black baseline.

Low similarity exposes reference dependence. It does not establish that the black baseline is correct.

---

## 20. AwA2 semantic concepts

### 20.1 Metadata discovery

The loader searches common AwA2 layouts for class, predicate, and matrix files.

### 20.2 Shape validation

The matrix must match:

~~~text
number of AwA2 classes x number of predicates
~~~

### 20.3 Continuous and binary matrices

The continuous matrix is used by default.

If it is absent, the loader can fall back to the binary matrix.

### 20.4 Alignment to the manifest

AwA2 class names and manifest names are normalized.

Concept matrix rows are reordered to match manifest label order.

This is essential because metadata order and custom subset label order are not assumed to match.

### 20.5 Per-concept normalization

For class c and concept k:

$$
\widetilde a_{c,k}=
\frac{a_{c,k}-\min_c a_{c,k}}
{\max_c a_{c,k}-\min_c a_{c,k}+\epsilon}
$$

Every selected class receives a concept vector in zero-one space.

### 20.6 Meaning of a concept value

A value is an AwA2 class prototype strength.

It is not:

- a model probability;
- an image-specific annotation;
- proof that the concept is visible;
- proof that the classifier uses the concept.

---

## 21. Concept profiles and prediction transitions

### 21.1 Class profile

For each class, concepts are sorted by normalized strength.

The heatmap selects highly variable concepts because concepts with no variation cannot separate classes.

### 21.2 Prediction transition

When stress changes prediction from class a to class b:

$$
\Delta c=c_b-c_a
$$

The project reports:

- concept cosine similarity;
- mean absolute concept delta;
- gained concepts with positive delta;
- lost concepts with negative delta;
- shared high concepts.

### 21.3 Sign correctness

Gained concepts include only strictly positive deltas.

Lost concepts include only strictly negative deltas.

### 21.4 Interpretation boundary

This analysis describes semantic differences between class prototypes.

It does not prove the ResNet used those concepts. TCAV connects concepts to internal sensitivity, while the CBM makes concepts part of prediction.

---

## 22. Validated TCAV concept sensitivity

TCAV means Testing with Concept Activation Vectors.

It asks whether movement toward a human concept direction inside a layer tends to increase a target class score.

### 22.1 Selected layer

The project uses layer3 by default.

Its output is approximately:

~~~text
B x 1024 x 14 x 14
~~~

Spatial average pooling creates a 1024-dimensional vector.

The final layer4 is followed only by global average pooling and a linear classifier. Its class gradients can become nearly constant. Layer3 provides a richer nonlinear path.

### 22.2 Positive and negative examples

For concept k:

~~~text
positive strength threshold = 0.75
negative strength threshold = 0.25
~~~

If too few examples exist, adaptive thresholds use the 75th and 25th percentiles.

Sets are:

- disjoint;
- shuffled with a seed;
- capped at 200 examples by default.

Because labels are class-level, concept groups can also encode class identity and correlated context.

### 22.3 Activation extraction

A forward hook records the selected layer output.

Average pooling computes:

$$
a(x)=
\frac{1}{HW}
\sum_{i,j}A(x)_{:,i,j}
$$

### 22.4 CAV training

Positive and negative activation vectors are standardized:

$$
\widetilde a=
\frac{a-\mu}{\sigma}
$$

A linear binary classifier is trained with:

~~~text
loss          = BCEWithLogitsLoss
optimizer     = AdamW
epochs        = 200
learning rate = 1e-2
weight decay  = 1e-4
~~~

The learned normal vector is converted back to raw activation coordinates and normalized.

### 22.5 Directional derivative

For concept direction v subscript C:

$$
S_{C,k}(x)=
\nabla_a f_k(a(x))
\cdot v_C
$$

Positive sensitivity means a local movement toward the concept direction increases the target logit.

### 22.6 TCAV score

For target-class examples X subscript k:

$$
TCAV_{C,k}=
\frac{1}{|X_k|}
\sum_{x\in X_k}
\mathbf{1}
[S_{C,k}(x)>0]
$$

The script also records mean and standard deviation of the raw directional derivative.

### 22.7 Correct interpretation

~~~text
high score = target logit often increases along the learned concept direction
low score  = positive sensitivity is not consistent
~~~

A high score does not prove necessity, sufficiency, causality, or concept disentanglement.

### 22.8 Validation and leakage control

The maintained implementation does not accept CAV training accuracy as
evidence. For every concept-target pair it:

1. excludes all training images of the evaluated target class from the CAV data;
2. creates positive and negative sets from AwA2 attribute thresholds;
3. prefers class-disjoint CAV train/validation splits;
4. falls back to an image-level split only when class coverage is insufficient;
5. rejects a CAV below the configured held-out validation accuracy;
6. records concept coverage and the split strategy in CSV outputs.

Target-class exclusion is important because a separator can otherwise learn the
target animal identity and be mislabeled as a semantic concept direction.

### 22.9 Repetition, controls and inference

Each concept-target pair is repeated over multiple deterministic seeds. Every
real CAV is paired with a random CAV trained on disjoint, size-matched random
groups. The project reports:

- mean and standard deviation of real TCAV scores;
- a 95 percent normal-approximation confidence interval;
- mean random-CAV score;
- real-minus-random effect size;
- paired sign-flip permutation p-value;
- Benjamini-Hochberg or Bonferroni corrected p-value;
- the complete per-run table;
- a versioned NPZ plus JSON CAV bank.

The default run requests 20 CAV fits and requires at least five valid fits for
each reported pair. A pair can be semantically discussed only after checking
held-out CAV accuracy, run variability, random-control effect and corrected
significance together.

### 22.10 TCAV background stress audit

`run_tcav_stress.py` reuses the validated CAV bank and recomputes the gradient
of the same target logit after Gaussian background noise, background color
inversion and random background replacement. For fixed direction `v_C`:

$$
\Delta TCAV_{C,k}=TCAV_{C,k}(x')-TCAV_{C,k}(x)
$$

It also records the change in mean directional derivative, prediction-change
rate and target-probability change. A paired random-CAV delta controls whether
the apparent concept drift is larger than generic directional instability.

This is an intervention on image context, not a causal intervention on the
human concept itself.

---

## 23. Concept Bottleneck Model

### 23.1 Purpose

The CBM changes the prediction path:

~~~text
image
  -> ResNet features
  -> predicted concepts
  -> class prediction
~~~

The class head cannot directly access the 2048-dimensional image feature vector.

### 23.2 Concept target dataset

For image label y, the wrapped dataset returns the normalized AwA2 class prototype:

~~~python
(image, concept_target, class_label, class_name, filepath)
~~~

Concept supervision is class-level, not image-level.

### 23.3 Concept vocabulary selection

The default bottleneck contains 20 concepts.

Selection order:

1. user-requested concepts;
2. always include stripes, furry, hooves, horns, flippers;
3. fill remaining positions with highest-variance concepts.

High variance improves class discrimination but does not guarantee independence or causal meaning.

### 23.4 Architecture

~~~text
normalized image
    -> ResNet50 backbone
    -> 2048-dimensional feature vector
    -> dropout, p = 0.15
    -> linear concept head
    -> concept logits
    -> sigmoid
    -> concept probabilities
    -> linear class head
    -> class logits
    -> softmax for reporting
~~~

Mathematically:

$$
h=\operatorname{ResNet}(x)
$$

$$
z_c=W_c h+b_c
$$

$$
\widehat c=\sigma(z_c)
$$

$$
z_y=W_y\widehat c+b_y
$$

$$
p(y\mid x)=\operatorname{softmax}(z_y)
$$

### 23.5 Joint loss

Concept loss:

$$
\mathcal L_{concept}=
BCEWithLogits(z_c,c)
$$

Class loss:

$$
\mathcal L_{class}=
CE(z_y,y)
$$

Combined objective:

$$
\mathcal L=
\lambda_{class}\mathcal L_{class}+
\lambda_{concept}\mathcal L_{concept}
$$

Defaults:

~~~text
lambda class   = 1.0
lambda concept = 1.0
~~~

Equal scalar weights do not guarantee equal gradient influence because the losses can have different scales.

### 23.6 Training defaults

~~~text
epochs                    = 5
batch size                = 16
learning rate             = 1e-4
weight decay              = 1e-4
dropout                   = 0.15
trainable backbone stage  = layer4
optimizer                 = AdamW
~~~

The best checkpoint is selected by validation class accuracy.

### 23.7 Backbone initialization

The intended run loads the trained baseline checkpoint into the CBM backbone, excluding the baseline fully connected classifier.

An alternative is explicit ImageNet initialization.

The maintained loader fails immediately when the requested checkpoint is
missing and ImageNet initialization was not explicitly requested. It never
permits a mostly frozen randomly initialized backbone.

A formal run must use:

~~~text
a valid baseline checkpoint
or
explicit ImageNet pretraining
~~~

### 23.8 Metrics

The CBM reports:

- total loss;
- class loss;
- concept loss;
- class accuracy;
- concept mean absolute error;
- concept binary accuracy at threshold 0.5;
- per-concept MAE;
- per-concept Pearson correlation;
- TP, FP, FN, TN;
- precision, recall, specificity, F1;
- agreement with the baseline;
- image-level class and concept errors.
- class-head accuracy under AwA2 class-target injection;
- the target-injection-minus-predicted-CBM accuracy gap;
- image-specific concept correction effects;
- oracle-prototype class-head sensitivity.

The class head is optimized on predicted concept vectors. Exact AwA2 class
targets can therefore be outside the distribution seen by that head. The
target-injection accuracy is a distribution-shift diagnostic, not a guaranteed
oracle upper bound. A negative gap indicates that the semantic interface and
the downstream class head are not aligned under direct target replacement.

### 23.9 Concept error

For concept k:

$$
e_k=\widehat c_k-c_k
$$

Positive error means overprediction.

Negative error means underprediction.

The project records the largest absolute concept errors for each image and groups class errors by true-to-predicted transition.

### 23.10 Frozen BatchNorm handling

Calling `model.train()` normally switches every BatchNorm layer to training
mode, even when its affine parameters are frozen. The CBM training loop now
reapplies `set_frozen_batchnorm_eval(model.backbone)` after every train-mode
transition. Running mean and variance therefore remain unchanged for frozen
BatchNorm modules, while trainable BatchNorm layers retain normal behavior.

### 23.11 Checkpoint semantics

The best CBM checkpoint embeds:

- ordered concept names;
- concept indices;
- class mapping;
- seed;
- transforms;
- architecture configuration;
- the source of backbone initialization.

The checkpoint is loaded with tensor-only safe loading. These fields prevent a
concept head from being interpreted with a different class order or concept
vocabulary.

---

## 24. Concept interventions

The class head can classify a manually supplied concept vector.

For one concept k:

1. keep all other concept values fixed;
2. set concept k to zero;
3. compute target probability;
4. set concept k to one;
5. compute target probability.

The intervention effect is:

$$
\Delta p_k=
p(y\mid c_k=1,c_{-k})-
p(y\mid c_k=0,c_{-k})
$$

Positive delta means increasing the concept increases the target probability.

Negative delta means increasing the concept decreases the target probability.

### 24.1 Oracle-prototype sensitivity

One maintained output starts from the oracle AwA2 class prototype:

~~~text
oracle-prototype intervention
~~~

It is explicitly labeled as a class-head sensitivity test. It is not an
image-specific correction experiment.

### 24.2 Image-specific intervention

The second maintained output:

1. run one image through the CBM;
2. save its predicted concept vector;
3. edit selected concepts;
4. rerun only the class head;
5. test whether the true-class probability increases;
6. records whether the class prediction is corrected.

The script ranks corrections by their increase in the true-class probability
and stores the original and intervened predictions, concept values, concept
error and probability delta. Because the replacement value is an AwA2
class-level target, it is an oracle correction under class-level supervision,
not an annotation of what is visibly present in that individual image.

---

## 25. Real forward-pass inspection

Forward hooks capture real tensors from:

~~~text
conv1
maxpool
layer1
layer2
layer3
layer4
avgpool
~~~

For one dataset image, the inspector records:

- input statistics;
- intermediate activation tensors;
- logits;
- softmax probabilities;
- predicted label and confidence;
- top class probabilities;
- Grad-CAM;
- activation summary maps.

### 25.1 Activation summary map

For a convolutional tensor:

1. take absolute activation;
2. average across channels;
3. resize to input resolution;
4. min-max normalize.

This is an activation-energy visualization. It is not a semantic interpretation of every neuron.

### 25.2 Simulator payload

The JSON trace contains:

- the original denormalized image as embedded PNG;
- a compact 14 x 14 RGB preview;
- 7 x 7 activation previews;
- 7 x 7 Grad-CAM preview;
- top softmax probabilities;
- tensor shape and statistics.

The compact grids are browser visualizations. The Python model still processes the full 224 x 224 tensor.

---

## 26. Notebook workflow

### 26.1 01_data_baseline_xai.ipynb

Purpose:

- validate project paths;
- inspect manifest and DataLoaders;
- load or train ResNet50;
- print predictions;
- inspect correct and incorrect examples;
- compute Grad-CAM and IG;
- optionally print raw gradient diagnostics.

Current control flags:

~~~python
RUN_TRAINING = False
RUN_XAI = CHECKPOINT.exists()
RUN_GRADIENT_DIAGNOSTICS = False
~~~

### 26.2 02_stress_concepts_tcav.ipynb

Purpose:

- create perturbations;
- calculate prediction and saliency stability;
- run the advanced attribution audit;
- inspect AwA2 class concepts;
- run validated TCAV analysis and inspect its random-control inference.

Current flags:

~~~python
RUN_METRICS = CHECKPOINT.exists()
RUN_ADVANCED_AUDIT = CHECKPOINT.exists()
RUN_CONCEPTS = False
RUN_TCAV = False
~~~

### 26.3 03_bottleneck_sanity_report.ipynb

Purpose:

- train and evaluate the CBM;
- inspect concept metrics;
- inspect concept confusion counts;
- compare CBM and baseline predictions;
- analyze errors through concept errors;
- inspect oracle-prototype interventions;
- inspect image-specific concept corrections;
- read both current and legacy CBM summary schemas without an uninformative `KeyError`;
- display advanced attribution results.

Current flags:

~~~python
RUN_CBM = True
RUN_SANITY = True
RUN_ADVANCED_ATTRIBUTION_AUDIT = True
~~~

These expensive operations should be reviewed before running.

### 26.4 04_real_forward_inspection.ipynb

Purpose:

- load one real AwA2 image;
- capture real ResNet activations;
- print probabilities and tensor diagnostics;
- generate the forward figure;
- export simulator JSON.

### 26.5 05_blog_figures.ipynb

Purpose:

- create lightweight explanatory assets;
- derive metric, case-study, and concept-summary figures from existing CSV files;
- copy maintained assets into `docs/assets/xai-report/`;
- update documentation figures without rerunning model experiments;
- use the validated TCAV columns when available;
- fall back to raw exploratory TCAV scores and CAV training accuracy when the legacy CSV is present, with an explicit warning and without fabricating error bars, random-control effects, held-out validation, or significance.

---

## 27. Complete command-line workflow

Run from the WSL project root:

~~~bash
cd /home/emma/DeepLearning/Deep_Learning_XAI
source /home/emma/DeepLearning/.venvDeepLearning/bin/activate
~~~

### 27.1 Install dependencies

~~~bash
python -m pip install -r requirements.txt
~~~

### 27.2 Create the 20-class subset

~~~bash
python scripts/data/prepare_awa2.py --mode subset \
  --source-root data/AWA2 \
  --output-root data/AWA2_subset_background20 \
  --preset background20 \
  --max-images-per-class 200 \
  --resize-size 128 \
  --resize-method pad \
  --jpeg-quality 92 \
  --seed 42
~~~

### 27.3 Validate the data pipeline

~~~bash
python scripts/data/general_tests.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv
~~~

### 27.4 Train the baseline

~~~bash
python scripts/training/train_baseline.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint-path outputs/checkpoints/best_resnet50_awa2.pt \
  --history-path outputs/reports/training_history.csv \
  --batch-size 32 \
  --epochs 5 \
  --lr 1e-4 \
  --weight-decay 1e-4
~~~

### 27.5 Generate local explanations

~~~bash
python scripts/experiments/run_xai.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/xai_examples.png \
  --max-images 8 \
  --max-per-class 1 \
  --ig-steps 50
~~~

### 27.6 Run background stress metrics

~~~bash
python scripts/experiments/run_background_stress_metrics.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --csv-output outputs/reports/phase5_saliency_metrics.csv \
  --perturbation-figure-output outputs/figures/phase5_perturbations.png \
  --figure-output outputs/figures/phase5_saliency_comparison.png \
  --max-images 20 \
  --xai-methods gradcam integrated_gradients \
  --ig-steps 16 \
  --mask-strategy center_ellipse \
  --seed 42
~~~

Four examples are sufficient for debugging, not final statistics.

### 27.7 Analyze concept profiles

~~~bash
python scripts/experiments/analyze_concept_profiles.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --metadata-root data/AWA2 \
  --stress-csv outputs/reports/phase5_saliency_metrics.csv \
  --class-profile-output outputs/reports/phase6_class_concepts.csv \
  --transition-output outputs/reports/phase6_concept_transitions.csv \
  --heatmap-output outputs/figures/phase6_class_concept_heatmap.png \
  --transition-figure-output outputs/figures/phase6_concept_transition_examples.png
~~~

### 27.8 Run validated TCAV

~~~bash
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
~~~

Then test whether concept sensitivity changes under background interventions:

~~~bash
python scripts/experiments/run_tcav_stress.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --cav-artifact outputs/reports/phase7_cav_vectors.npz \
  --run-output outputs/reports/tcav_stress_runs.csv \
  --summary-output outputs/reports/tcav_stress_summary.csv \
  --figure-output outputs/figures/tcav_stress_effects.png
~~~

### 27.9 Train the CBM

~~~bash
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
~~~

Verify that the backbone checkpoint exists before running.

### 27.10 Run the advanced attribution audit

~~~bash
python scripts/audits/run_advanced_attribution_audit.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --methods gradcam integrated_gradients \
  --num-examples 20 \
  --ig-steps 16 \
  --report-output outputs/reports/advanced_attribution_audit.csv \
  --summary-output outputs/reports/advanced_attribution_audit_summary.csv \
  --figure-dir outputs/figures/advanced_attribution_audit
~~~

### 27.11 Export a real forward trace

~~~bash
python scripts/tools/run_forward_inspection.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv \
  --checkpoint outputs/checkpoints/best_resnet50_awa2.pt \
  --output outputs/figures/real_forward_inspection.png \
  --trace-json outputs/reports/real_forward_trace.json \
  --split test \
  --sample-index 0
~~~

---

## 28. Output files

### 28.1 Baseline

~~~text
outputs/checkpoints/best_resnet50_awa2.pt
outputs/reports/training_history.csv
~~~

### 28.2 Local attribution

~~~text
outputs/figures/xai_examples*.png
outputs/figures/xai_correct_examples*.png
outputs/figures/xai_incorrect_examples*.png
~~~

### 28.3 Stress test

~~~text
outputs/figures/phase5_perturbations*.png
outputs/figures/phase5_saliency_comparison*.png
outputs/reports/phase5_saliency_metrics*.csv
~~~

### 28.4 Concepts and TCAV

~~~text
outputs/reports/phase6_class_concepts*.csv
outputs/reports/phase6_concept_transitions*.csv
outputs/reports/phase7_cav_summary*.csv
outputs/reports/phase7_tcav_scores*.csv
outputs/figures/phase7_tcav_heatmap*.png
outputs/figures/phase7_tcav_top_scores*.png
~~~

### 28.5 Concept Bottleneck Model

~~~text
outputs/checkpoints/phase8_cbm*.pt
outputs/reports/phase8_cbm_summary*.csv
outputs/reports/phase8_concept_metrics*.csv
outputs/reports/phase8_concept_confusion_matrix*.csv
outputs/reports/phase8_cbm_predictions*.csv
outputs/reports/phase8_cbm_error_analysis*.csv
outputs/reports/phase8_oracle_prototype_interventions*.csv
outputs/reports/phase8_image_concept_interventions*.csv
~~~

### 28.6 Advanced audit

~~~text
outputs/reports/advanced_attribution_audit*.csv
outputs/figures/advanced_attribution_audit/
~~~

### 28.7 Documentation assets

`05_blog_figures.ipynb` publishes self-contained report assets so the HTML never
depends on the Git-ignored `outputs/` directory:

~~~text
docs/blog_project_pipeline.svg
docs/blog_method_taxonomy.svg
docs/blog_evidence_ladder.svg
docs/assets/xai-report/metric-summary.png
docs/assets/xai-report/case-summary.png
docs/assets/xai-report/concept-validation-summary.png
docs/assets/xai-report/cbm-summary.png
docs/assets/xai-report/concept-interventions.png
~~~

### 28.8 Legacy generated artifacts

Some current generated outputs were produced by older exploratory methods.

They are not part of the maintained four-method project.

For final reporting:

1. regenerate current outputs into clean filenames;
2. copy only maintained assets into docs/assets;
3. do not cite stale CSVs or figures;
4. verify that HTML image references are local and valid.

---

## 29. Current result snapshot

The following values come from CSV files currently present in outputs/reports. They explain the current run but are not all publication-ready.

### 29.1 Baseline training

| Epoch | Train accuracy | Validation accuracy | Validation loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.654 | 0.856 | 0.499 |
| 2 | 0.876 | 0.884 | 0.396 |
| 3 | 0.937 | 0.879 | 0.377 |
| 4 | 0.969 | 0.879 | 0.399 |
| 5 | 0.980 | 0.864 | 0.443 |

Validation accuracy peaks at epoch 2.

Training accuracy continues to increase while validation performance declines. This indicates overfitting pressure.

Best-validation checkpointing should retain epoch 2.

The CBM comparison report measures baseline test accuracy at approximately 0.857 over 596 test samples.

### 29.2 Background stress snapshot

The current stress summary contains only four examples. It is useful as a case-study snapshot but must not be generalized to the complete dataset.

| Method | Perturbation | Prediction-change rate | Mean IoU | Mean Spearman |
| --- | --- | ---: | ---: | ---: |
| Grad-CAM | Gaussian noise | 0.00 | 0.550 | 0.647 |
| Grad-CAM | Color shift | 0.00 | 0.617 | 0.768 |
| Grad-CAM | Background swap | 0.75 | 0.503 | 0.595 |
| Integrated Gradients | Gaussian noise | 0.00 | 0.363 | 0.489 |
| Integrated Gradients | Color shift | 0.00 | 0.344 | 0.577 |
| Integrated Gradients | Background swap | 0.75 | 0.338 | 0.405 |

Within this tiny snapshot:

- background swap changes predictions most frequently;
- IG maps are less spatially stable than Grad-CAM;
- the sample is too small for a dataset-level conclusion.

### 29.3 Advanced attribution snapshot

The current advanced summary also contains only four examples.

| Metric | Grad-CAM | Integrated Gradients |
| --- | ---: | ---: |
| Deletion AUC | 0.136 | 0.085 |
| Insertion AUC | 0.383 | 0.269 |
| Faithfulness gap | 0.247 | 0.184 |
| Approximate animal ratio | 0.774 | 0.555 |
| Approximate background ratio | 0.226 | 0.445 |
| Noise-stability IoU | 0.861 | 0.455 |
| Noise-stability Spearman | 0.923 | 0.734 |
| Top-1/top-2 map IoU | 0.285 | 0.441 |
| Top-1/top-2 map Spearman | 0.510 | 0.748 |

Within this snapshot:

- Grad-CAM is more stable to noise;
- Grad-CAM assigns more mass to the central foreground proxy;
- IG assigns much more mass to the approximate background;
- both methods show substantial similarity between competing class explanations.

IG blurred-versus-black baseline comparison:

~~~text
top-20% IoU = 0.198
Spearman     = 0.215
~~~

Within this four-example snapshot, the low overlap is a clear warning that the IG visualization depends strongly on baseline choice. It is not a population-level estimate.

### 29.4 Legacy TCAV snapshot requiring regeneration

The values below were produced by the earlier single-CAV implementation. They
are retained only as historical context and must not be reported as results of
the validated protocol. Regenerate the TCAV notebook before publication.

All legacy concept separators reported training accuracy 1.0. This could mean
easy separation, class leakage, high-dimensional overfitting, or genuine
concept structure; the old outputs could not distinguish these explanations.

Some plausible values:

~~~text
stripes -> tiger    0.933
stripes -> zebra    0.867
horns   -> antelope 0.800
hooves  -> antelope 0.767
~~~

Some surprising values are also present, such as strong positive horn sensitivity for dolphin.

The new run must replace these values with held-out accuracy, repeated scores,
matched random controls and corrected significance.

### 29.5 Concept Bottleneck snapshot

~~~text
classes                              = 20
concepts                             = 20
test class accuracy                  = 0.799
baseline test accuracy               = 0.857
CBM/baseline agreement               = 0.728
concept MAE                          = 0.211
concept binary accuracy              = 0.754
class head on AwA2 target injection  = 0.201
target injection minus CBM accuracy  = -0.597
backbone source                      = awa2_checkpoint
~~~

The CBM loses approximately 5.9 percentage points of class accuracy in exchange
for an explicit semantic path. Its concept predictions remain substantially,
but not uniformly, aligned with the class-level AwA2 attributes.

Concept performance is uneven:

~~~text
flippers binary accuracy approximately 0.893
furry binary accuracy approximately 0.649
~~~

The low 0.201 accuracy after injecting exact AwA2 class targets is not an oracle
upper bound. It shows that the class head learned the distribution of predicted
concept vectors and transfers poorly to exact class prototypes. The associated
negative gap is therefore evidence of concept-to-class distribution mismatch.

The image-specific intervention report starts from each image's predicted
concept vector and changes one selected concept to its AwA2 class-level target.
The current blog summary reports that approximately 2.9% of the selected
originally incorrect images are corrected by at least one such edit. This small
rate must be interpreted with the limited evaluated sample and class-level
concept supervision in mind.

---

## 30. Interpretation of correct and incorrect predictions

### 30.1 Correct prediction and animal-focused map

Conservative statement:

> The explanation is visually plausible for this image and selected target.

Required follow-up:

- deletion and insertion;
- background stress;
- class discriminativeness;
- concept consistency.

### 30.2 Correct prediction and background-focused map

Possible explanations:

- model context dependence;
- attribution-method failure;
- inaccurate foreground proxy;
- valid joint use of animal and environment.

The overlay alone cannot decide among them.

### 30.3 Incorrect prediction

For true antelope and predicted buffalo:

Predicted-target explanation asks:

> Which evidence supported buffalo?

True-target explanation asks:

> Which evidence supported antelope, and why was it insufficient?

Inspect:

- top probabilities;
- Grad-CAM for both targets;
- IG for both targets;
- context versus morphology;
- class concept prototype difference;
- CBM overpredicted and underpredicted concepts;
- perturbation prediction transitions.

### 30.4 Grad-CAM and IG disagreement

Disagreement is expected.

Grad-CAM defines importance through deep feature maps.

IG defines importance through an input-to-baseline path.

Do not choose the visually more attractive map. Compare stability, faithfulness, target specificity, and baseline dependence.

### 30.5 Confidence is not explanation quality

High softmax confidence does not prove:

- correctness;
- calibration;
- semantic validity;
- faithful attribution;
- robustness.

---

## 31. Reproducibility

The seed utility configures:

~~~text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
PYTHONHASHSEED
cuDNN deterministic mode
cuDNN benchmark disabled
~~~

Default seed:

~~~text
42
~~~

Exact reproducibility can still depend on:

- PyTorch version;
- torchvision version;
- Captum version;
- CUDA and driver;
- hardware;
- DataLoader workers;
- manifest contents;
- checkpoint metadata;
- command parameters.

For a formal experiment, archive:

~~~text
manifest
class map
subset_summary.json
checkpoint
training history
command line
seed
git commit
dependency versions
CSV outputs
figures
~~~

---

## 32. Validation and tests

### 32.1 General data test

The data smoke test validates:

- required CSV columns;
- non-empty fields;
- non-negative integer labels;
- one-to-one label and class mapping;
- image paths;
- train, validation, and test coverage;
- contiguous labels;
- class-map agreement;
- one batch per split;
- B x 3 x 224 x 224 tensor shape;
- finite tensors;
- inverse-normalization range.

### 32.2 Unit tests

Current tests cover:

- duplicate class names with inconsistent labels;
- different train and evaluation transforms;
- frozen BatchNorm evaluation behavior;
- frozen BatchNorm running-stat stability;
- correct gained and lost concept signs;
- Python 3.10-compatible log-level validation without relying on `logging.getLevelNamesMapping()`.

### 32.3 Validation commands

~~~bash
python -m unittest discover -v
python -m compileall -q src scripts
python scripts/data/general_tests.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv
git diff --check
~~~

Passing these tests confirms software consistency, not scientific validity.

---

## 33. Known limitations

### 33.1 Geometric foreground proxy

The central ellipse is not a segmentation mask.

It can perturb the animal and preserve background.

Improvement:

- manually validate masks;
- evaluate a segmented subset;
- report mask-quality categories.

### 33.2 Small current samples

Several current reports use four examples.

Improvement:

- class-balanced evaluation;
- larger samples;
- per-class metrics;
- bootstrap confidence intervals.

### 33.3 Residual TCAV limitations

The script now includes class-aware validation, repeated seeds, random CAVs,
permutation inference and multiple-testing correction. Remaining limitations
come mainly from the data: AwA2 concepts are class-level, positive examples can
share habitat and taxonomy, and a linear direction need not isolate one causal
factor. Report concept coverage, split fallback rate and random-control effect
with every result.

### 33.4 Class-level concepts

AwA2 attributes are not image-specific.

Improvement:

- annotate an image-level subset;
- use an image-level attribute dataset;
- report visibility and occlusion.

### 33.5 CBM backbone safety

The loader now fails unless a valid baseline checkpoint exists or explicit
ImageNet initialization is selected. The residual requirement is to report the
chosen backbone source from the checkpoint metadata.

### 33.6 CBM frozen BatchNorm

Frozen BatchNorm modules are restored to evaluation mode after every
`model.train()` call. Regression tests verify that their running statistics do
not change.

### 33.7 Concept interventions

Both oracle-prototype sensitivity and image-specific oracle corrections are
implemented and stored separately. Their remaining limitation is the use of
class-level AwA2 targets rather than image-level concept annotations.

### 33.8 Checkpoint metadata

CBM checkpoints package class mapping, ordered concepts, transforms, seed,
architecture configuration and backbone source. Optimizer state is not needed
for evaluation but would still be required to resume an interrupted CBM run.

### 33.9 Subset idempotency

Existing subsets are reused only after configuration identity and destination
compatibility checks. Explicit zero ratios are preserved. For copied files the
source and destination bytes are compared; resized files are decoded and
validated by size and color mode.

### 33.10 Per-map min-max normalization

A weak map can still become visually colorful.

Improvement:

- raw attribution norms;
- signed maps;
- shared color scales;
- IG convergence delta.

### 33.11 Synthetic distribution shift

Noise and inversion create unnatural images.

Improvement:

- natural background replacement;
- validated segmentation;
- retain synthetic perturbations as severity controls.

### 33.12 Calibration

Softmax confidence is not calibrated.

Improvement:

- expected calibration error;
- reliability diagrams;
- temperature scaling.

---

## 34. Claims supported by the project

### 34.1 Supported claims

The project can show that:

- local methods define importance differently;
- predictions and explanations can change after background interventions;
- stability and faithfulness are distinct;
- Grad-CAM is coarse because it originates from deep low-resolution maps;
- IG depends on its baseline;
- semantic attributes organize class-level differences;
- internal representations can be probed along concept directions;
- a CBM creates an explicit semantic prediction path;
- concept errors can be connected to CBM class errors.

### 34.2 Unsupported or premature claims

The project should not claim that:

- a heatmap reveals true reasoning;
- low stability proves exclusive background use;
- geometric masks perfectly isolate animals;
- a TCAV pair is meaningful without held-out CAV accuracy, random controls and corrected significance;
- class-level concepts are visible in every image;
- an oracle-prototype intervention is equivalent to correcting an individual image prediction;
- one attribution method is universally superior;
- visual plausibility establishes causality.

### 34.3 Final position

> Explainability in computer vision is not a single visualization. It is an audit combining local attribution, controlled interventions, stability and faithfulness metrics, concept probes, interpretable model structure, and explicit uncertainty about what each method can establish.

---

## 35. Recommended presentation flow

For a professor, report, or blog:

1. Define explainability, attribution, saliency map, and heatmap.
2. Introduce contextual shortcuts.
3. Explain AwA2 images and attributes.
4. Show the manifest and stable labels.
5. Show ResNet50 tensor flow.
6. State Cross-Entropy, AdamW, augmentation, and checkpoint policy.
7. Establish baseline performance and overfitting evidence.
8. Derive Grad-CAM.
9. Derive Integrated Gradients and the blurred baseline.
10. Show correct and incorrect examples with explicit targets.
11. Show the approximate mask before perturbation results.
12. Define all three perturbations mathematically.
13. Present prediction changes, IoU, and Spearman.
14. Add deletion, insertion, entropy, noise sensitivity, and class specificity.
15. Move from pixels to semantic concepts.
16. Explain validated TCAV, target-class exclusion, repeated CAVs, random controls and corrected significance.
17. Explain the CBM semantic bottleneck.
18. Compare baseline, CBM, and concept metrics.
19. Separate oracle-prototype class-head sensitivity from image-specific oracle concept corrections.
20. End with conservative conclusions.

The narrative should move from visual plausibility toward increasingly demanding evidence.

---

## 36. Reproducible execution and report publication

The final numerical report must be regenerated in dependency order. Notebook
flags are disabled by default for expensive operations so opening the project
cannot accidentally retrain a model or launch thousands of gradient passes.

### 36.1 Baseline and local explanations

Open `notebooks/01_data_baseline_xai.ipynb` and execute it from the first cell.
Set `RUN_TRAINING = True` for a publication run. This produces a new baseline
checkpoint with class mapping, seed, transforms, model configuration and early
stopping metadata. The same checkpoint must be used by every later notebook.

### 36.2 Stress metrics and validated TCAV

Open `notebooks/02_stress_concepts_tcav.ipynb` and set:

~~~python
RUN_CONCEPTS = True
RUN_TCAV = True
RUN_TCAV_STRESS = True
~~~

The TCAV command writes:

~~~text
phase7_concept_coverage_notebook.csv
phase7_tcav_runs_notebook.csv
phase7_tcav_scores_notebook.csv
phase7_cav_summary_notebook.csv
phase7_cav_vectors_notebook.npz
phase7_cav_vectors_notebook.json
~~~

The stress command consumes the CAV bank and writes per-run and aggregate
changes in fixed concept-direction sensitivity after background interventions.

### 36.3 Concept Bottleneck Model

Open `notebooks/03_bottleneck_sanity_report.ipynb` and set `RUN_CBM = True`.
The run requires the baseline checkpoint or an explicit ImageNet fallback. It
produces standard CBM metrics, class-head accuracy under AwA2 target injection,
oracle-prototype sensitivity and image-specific concept-correction outputs under
distinct names. The notebook reports a negative target-injection gap as a
distribution-mismatch diagnostic rather than presenting it as an oracle bound.

### 36.4 Publish figures into the HTML report

Run `notebooks/05_blog_figures.ipynb` only after the preceding CSV and PNG
outputs exist. It derives KPI figures from CSV and copies publication assets
into `docs/assets/xai-report/`. With the validated TCAV schema it plots repeated
effects, held-out CAV accuracy and random controls. With the current legacy
schema it generates a clearly labeled exploratory panel using only raw TCAV
scores and CAV training accuracy. It never synthesizes missing inferential
statistics. In particular it refreshes:

~~~text
metric-summary.png
case-summary.png
tcav-heatmap.png
tcav-top-scores.png
cbm-summary.png
concept-interventions.png
concept-validation-summary.png
~~~

It also regenerates the explanatory SVG assets:

~~~text
docs/blog_project_pipeline.svg
docs/blog_method_taxonomy.svg
docs/blog_evidence_ladder.svg
~~~

The maintained report is `docs/intuitive_explainability.html`. It never loads
figures directly from the Git-ignored `outputs/` tree. Unit tests verify that
all local image references resolve and that only Grad-CAM, Integrated
Gradients, validated TCAV and the Concept Bottleneck Model are presented as
maintained explanation methods.

### 36.5 Required final validation

~~~bash
python -m unittest discover -v
python -m compileall -q src scripts
python scripts/data/general_tests.py \
  --manifest data/AWA2_subset_background20/awa2_manifest_subset.csv
git diff --check
~~~

Do not combine metrics from checkpoints with different class mappings or
training configurations. Regenerating only the final figure notebook does not
make legacy TCAV or CBM CSV files methodologically current.

---

## 37. Glossary

**Activation**  
The output tensor produced by a network layer.

**Attribution**  
A numerical importance assignment for a selected model output.

**Baseline**  
A reference input used by path or replacement methods.

**Batch Normalization**  
A layer using affine parameters and running activation statistics.

**CAV**  
Concept Activation Vector, a learned direction in internal activation space.

**Checkpoint**  
A serialized model state and associated metadata.

**Concept**  
A human-readable semantic property such as stripes or horns.

**Concept Bottleneck**  
An explicit concept vector through which class prediction passes.

**Convolution**  
A learned local spatial filter.

**Faithfulness**  
How well an explanation reflects evidence actually used by the model.

**Feature map**  
One spatial channel of a convolutional activation tensor.

**Gradient**  
A derivative measuring local output sensitivity.

**Heatmap**  
A color rendering of a numerical spatial map.

**Intervention**  
An intentional change followed by measurement of model response.

**IoU**  
Intersection over Union between two spatial supports.

**Logit**  
An unnormalized class score before softmax.

**Manifest**  
A CSV mapping paths to labels, names, and splits.

**Post-hoc explanation**  
An explanation applied after model training.

**Residual connection**  
A shortcut added to a learned residual transformation.

**Saliency map**  
A numerical spatial attribution tensor.

**Softmax**  
A transformation converting logits to positive values summing to one.

**Spearman correlation**  
A rank-based correlation between saliency maps.

**TCAV score**  
The fraction of target examples with positive directional sensitivity to a concept.

**Transfer learning**  
Adapting a model initialized on another dataset.

---

## 38. References

- He, Zhang, Ren, and Sun. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
- Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017. https://arxiv.org/abs/1610.02391
- Sundararajan, Taly, and Yan. Axiomatic Attribution for Deep Networks. ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
- Kim et al. Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors. ICML 2018. https://proceedings.mlr.press/v80/kim18d.html
- Koh et al. Concept Bottleneck Models. ICML 2020. https://proceedings.mlr.press/v119/koh20a.html
- Captum documentation. https://captum.ai/

---

## Final summary

The project begins with a conventional deep image classifier but does not stop at accuracy.

It inspects how ResNet50 transforms RGB images, explains selected outputs with Grad-CAM and Integrated Gradients, perturbs approximate background regions, measures map stability and faithfulness, translates class behavior into AwA2 semantic concepts, implements validated TCAV concept directions with random controls, and trains a Concept Bottleneck Model that makes concepts part of prediction. The implementation status and generated-result status remain separate: the current TCAV artifacts are legacy exploratory outputs, whereas the CBM metrics and intervention outputs have been regenerated with the maintained pipeline.

Every result remains conditional on:

- target class;
- selected layer;
- IG baseline;
- mask quality;
- perturbation realism;
- map normalization;
- concept definition;
- data annotation level;
- sample size;
- checkpoint identity.

A professional explainability analysis makes those conditions explicit and tests whether its conclusions survive when they change.
