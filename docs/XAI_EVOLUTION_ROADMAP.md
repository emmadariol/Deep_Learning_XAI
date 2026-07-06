# From Saliency Maps to Concept-Based Explainability

## Project Direction

This project starts from a simple question:

> When a computer vision model classifies an animal, is it really using the animal's morphology, or is it relying on contextual shortcuts such as background, texture, color, or dataset bias?

The first part of the project studies post-hoc saliency maps:

- Input Gradients
- Grad-CAM
- Integrated Gradients

The next part should evolve toward more structured explainability:

- perturbation-based stress tests
- concept-based explanations
- TCAV
- Concept Bottleneck Models

The narrative arc is:

```text
pixels -> saliency maps -> stress tests -> concepts -> TCAV -> bottleneck models
```

## Phase 1: AwA2 Data Pipeline

Status: implemented.

Goal:

- prepare the AwA2 dataset;
- create reproducible CSV manifests;
- build a custom PyTorch Dataset;
- build train, validation and test DataLoaders;
- support full, debug and portable resized subsets.

Main files:

```text
scripts/prepare_awa2.py
scripts/create_awa2_subset.py
scripts/check_dataloader.py
src/data.py
```

This phase makes the dataset reproducible and inspectable.

## Phase 2: ResNet50 Baseline

Status: implemented.

Goal:

- load a ResNet50 architecture from torchvision;
- optionally use ImageNet pretrained weights;
- replace the final classification layer for the AwA2 classes;
- train or load a checkpoint;
- save the best model checkpoint.

Main files:

```text
src/model.py
src/train.py
scripts/train_baseline.py
```

The baseline is not the final research contribution. It is the model we interrogate.

## Phase 3: Saliency Maps

Status: implemented.

Methods:

- Input Gradients
- Grad-CAM
- Integrated Gradients

Main files:

```text
src/xai.py
scripts/run_xai.py
notebooks/phase2_phase3_awa2.ipynb
```

Goal:

- generate visual explanations for correctly classified images;
- generate visual explanations for misclassified images;
- explicitly report which class each wrong prediction was confused with;
- compare what the model appears to use when it succeeds and when it fails.

Scientific message:

> Saliency maps are intuitive, but they are not automatically causal explanations.

They answer:

```text
Where does the model appear to look?
```

They do not fully answer:

```text
Which semantic concept caused the prediction?
```

## Phase 4: Saliency Stress Test

Status: implemented as an approximate background perturbation stress test.

Goal:

- perturb the image in controlled ways;
- keep the animal content as stable as possible;
- recompute predictions and saliency maps;
- measure whether explanations remain stable.

Possible perturbations:

- Gaussian noise;
- color shift;
- blur;
- random background replacement;
- occlusion;
- crop-based context removal.

Because AwA2 does not provide segmentation masks, the current implementation uses:

1. a centered elliptical foreground approximation;
2. a centered rectangular foreground approximation;
3. a global perturbation fallback.

Main files:

```text
src/perturb.py
scripts/run_stress_test.py
notebooks/04_phase4_stress_test.ipynb
```

The key question is:

```text
If the prediction stays similar but the saliency map changes drastically,
how reliable was the explanation?
```

## Phase 5: Quantitative Metrics

Status: implemented for saliency degradation under Phase 4 perturbations.

Metrics to implement:

- IoU between top-k salient pixels;
- Spearman rank correlation between saliency maps;
- confidence drop;
- prediction flip rate;

Main files:

```text
src/metrics.py
scripts/run_phase5_metrics.py
notebooks/05_phase5_metrics.ipynb
```

Still useful future extensions:

- feature drift in internal ResNet layers;
- saliency mass shift under perturbation;
- aggregation plots across classes and perturbation types.

Goal:

```text
turn visual observations into quantitative evidence
```

This phase closes the critique of saliency maps.

## Phase 6: From Pixels to Concepts

Status: future phase.

The next conceptual step is to move from:

```text
pixel-level explanation
```

to:

```text
concept-level explanation
```

AwA2 is especially useful here because it contains semantic animal attributes.

Examples:

- stripes;
- furry;
- hooves;
- horns;
- tail;
- aquatic;
- black;
- white;
- brown;
- fast;
- domestic.

The question changes from:

```text
Which pixels are important?
```

to:

```text
Which human-understandable concepts are important?
```

## Phase 7: TCAV

Status: future phase.

TCAV means:

```text
Testing with Concept Activation Vectors
```

Core idea:

1. choose a concept, for example `stripes`;
2. collect positive examples for that concept;
3. collect random or negative examples;
4. extract internal activations from the model;
5. train a linear separator in activation space;
6. use the separator direction as a Concept Activation Vector;
7. measure whether moving along that concept direction increases the class score.

Example:

```text
class: zebra
concept: stripes
layer: layer4
question: how sensitive is the zebra prediction to the stripes concept?
```

Interpretation:

```text
high TCAV score -> the concept is influential for that class
low TCAV score  -> the concept is not very influential
```

Narrative transition:

```text
Grad-CAM asks: where does the model look?
TCAV asks: which concept matters?
```

## Phase 8: Concept Bottleneck Models

Status: future phase.

A standard classifier works like this:

```text
image -> class
```

A Concept Bottleneck Model works like this:

```text
image -> concepts -> class
```

For AwA2:

```text
image -> animal attributes -> animal class
```

Example:

```text
image -> has_stripes, has_hooves, has_tail, black_white -> zebra
```

Advantages:

- the prediction path is more interpretable;
- concepts can be inspected directly;
- concepts can be intervened on;
- errors can be diagnosed at the concept level.

This is stronger than saliency because it supports intervention:

```text
If the model predicts zebra but the concept "stripes" is low,
we can manually change the concept and observe whether the class changes.
```

## Final Comparison

The final project can compare three explainability families:

| Method family | Explanation type | Strength | Weakness |
|---|---|---|---|
| Saliency maps | pixel-level | intuitive and visual | fragile, not necessarily causal |
| TCAV | concept-level | semantically meaningful | depends on concept quality |
| Concept Bottleneck | interpretable by design | supports interventions | requires concept annotations |

## Blog or Report Structure

Suggested title:

```text
From Saliency Illusions to Concept-Based Explainability
```

Suggested sections:

1. Why visual explanations are tempting.
2. Grad-CAM and Integrated Gradients.
3. The illusion of saliency maps.
4. AwA2 and animal classification.
5. Correct vs incorrect explanations.
6. Stress testing saliency maps.
7. Moving from pixels to concepts.
8. TCAV.
9. Concept Bottleneck Models.
10. Final comparison and lessons learned.

## Concrete Roadmap

Next implementation steps:

```text
Phase 4:
implement perturbations and stress tests

Phase 5:
implement IoU, Spearman, confidence drop and prediction flip metrics

Phase 6:
load AwA2 semantic attributes as concept labels

Phase 7:
extract layer activations and implement TCAV

Phase 8:
train a simple image -> attributes -> class bottleneck model

Phase 9:
write the final comparison: saliency vs TCAV vs bottleneck
```

## Central Thesis

The project should not simply say that saliency maps are useless.

The stronger and more accurate thesis is:

> Saliency maps are useful exploratory tools, but they are incomplete explanations. Their visual plausibility can hide instability and non-causal behavior. Concept-based methods such as TCAV and Concept Bottleneck Models provide a more structured path toward explanations that are closer to human reasoning and more suitable for intervention.
