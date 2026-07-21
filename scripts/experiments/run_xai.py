"""Run attribution methods on selected AwA2 images."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders, infer_num_classes, load_class_names
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.utils import set_seed, setup_logging
from src.validation import (
    device_spec,
    log_level,
    nonnegative_float,
    nonnegative_int,
    positive_int,
)
from src.xai import (
    gradcam_saliency,
    integrated_gradients_maps,
    log_tensor_stats,
    save_xai_grid,
)

LOGGER = logging.getLogger("run_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AwA2 attribution examples.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2" / "awa2_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "xai_examples_awa2.png",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--max-images", type=positive_int, default=4)
    parser.add_argument("--max-per-class", type=positive_int, default=1)
    parser.add_argument("--ig-steps", type=positive_int, default=50)
    parser.add_argument("--ig-internal-batch-size", type=positive_int, default=4)
    parser.add_argument("--blur-radius", type=nonnegative_float, default=18.0)
    parser.add_argument(
        "--selection",
        choices=("correct", "incorrect", "all"),
        default="correct",
        help="Choose correctly classified, misclassified, or all test examples.",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def collect_correct_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names_by_label: dict[int, str] | None = None,
    max_images: int = 4,
    max_per_class: int | None = None,
    idx_to_class: dict[int, str] | None = None,
    allow_incorrect: bool = False,
    only_incorrect: bool = False,
    target_class: str | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float], list[str]]:
    """Select a reproducible, class-balanced sample from eligible examples."""
    if class_names_by_label is None:
        class_names_by_label = idx_to_class
    if class_names_by_label is None:
        raise ValueError("class_names_by_label or idx_to_class must be provided.")
    if max_images < 1:
        raise ValueError("max_images must be positive.")
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be positive or None.")

    candidate_limit = max_images if max_per_class is None else max_per_class
    reservoir_rng = random.Random(seed)
    candidates_by_class: dict[
        int,
        list[tuple[torch.Tensor, torch.Tensor, str, str, float, str]],
    ] = {}
    eligible_seen_by_class: dict[int, int] = {}

    evaluation_loader = loader
    known_class_count: int | None = None
    randomized_input_order = False
    dataset_samples = getattr(loader.dataset, "samples", None)
    if dataset_samples and all(hasattr(sample, "label") for sample in dataset_samples):
        indices_by_class: dict[int, list[int]] = {}
        for index, sample in enumerate(dataset_samples):
            indices_by_class.setdefault(int(sample.label), []).append(index)
        if target_class is not None:
            target_labels = [
                label
                for label in indices_by_class
                if class_names_by_label.get(label) == target_class
            ]
            if not target_labels:
                raise RuntimeError(f"No samples found for target class: {target_class}")
            indices_by_class = {label: indices_by_class[label] for label in target_labels}

        index_rng = random.Random(seed)
        class_order = sorted(indices_by_class)
        index_rng.shuffle(class_order)
        for indices in indices_by_class.values():
            index_rng.shuffle(indices)
        interleaved_indices = [
            indices_by_class[label][round_index]
            for round_index in range(max(map(len, indices_by_class.values())))
            for label in class_order
            if round_index < len(indices_by_class[label])
        ]
        evaluation_loader = DataLoader(
            Subset(loader.dataset, interleaved_indices),
            # Keep selection independent of the caller's throughput setting.
            batch_size=min(8, len(interleaved_indices)),
            shuffle=False,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=False,
        )
        known_class_count = len(indices_by_class)
        randomized_input_order = True

    model.eval()
    with torch.no_grad():
        for batch in evaluation_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            batch_true_names = list(batch[2])
            image_paths = list(batch[3])

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            if only_incorrect:
                selected = preds != labels
            elif allow_incorrect:
                selected = torch.ones_like(labels, dtype=torch.bool)
            else:
                selected = preds == labels
            if target_class is not None:
                target_mask = torch.tensor(
                    [
                        class_names_by_label[int(label.item())] == target_class
                        for label in labels
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                selected = selected & target_mask

            for idx in selected.nonzero(as_tuple=False).flatten().tolist():
                label_value = int(labels[idx].item())
                candidate = (
                    images[idx].detach().cpu(),
                    labels[idx].detach().cpu(),
                    batch_true_names[idx],
                    class_names_by_label[int(preds[idx].item())],
                    float(conf[idx].detach().cpu().item()),
                    image_paths[idx],
                )
                seen_count = eligible_seen_by_class.get(label_value, 0) + 1
                eligible_seen_by_class[label_value] = seen_count
                bucket = candidates_by_class.setdefault(label_value, [])
                if len(bucket) < candidate_limit:
                    bucket.append(candidate)
                elif not randomized_input_order:
                    replacement_index = reservoir_rng.randrange(seen_count)
                    if replacement_index < candidate_limit:
                        bucket[replacement_index] = candidate

            if randomized_input_order and known_class_count is not None:
                target_count = min(
                    max_images,
                    known_class_count * candidate_limit,
                )
                nonempty_class_count = sum(
                    bool(bucket) for bucket in candidates_by_class.values()
                )
                collected_count = sum(
                    len(bucket) for bucket in candidates_by_class.values()
                )
                if (
                    nonempty_class_count >= min(target_count, known_class_count)
                    and collected_count >= target_count
                ):
                    break

    if not candidates_by_class:
        raise RuntimeError("No selected examples found in the test split.")

    selection_rng = random.Random(seed)
    class_labels = sorted(candidates_by_class)
    selection_rng.shuffle(class_labels)
    for label in class_labels:
        selection_rng.shuffle(candidates_by_class[label])

    selected_candidates: list[
        tuple[torch.Tensor, torch.Tensor, str, str, float, str]
    ] = []
    round_index = 0
    while len(selected_candidates) < max_images:
        added_in_round = False
        for label in class_labels:
            bucket = candidates_by_class[label]
            if round_index >= len(bucket):
                continue
            selected_candidates.append(bucket[round_index])
            added_in_round = True
            if len(selected_candidates) >= max_images:
                break
        if not added_in_round:
            break
        round_index += 1

    if len(selected_candidates) < max_images:
        LOGGER.warning(
            "Selected %d of %d requested examples after eligibility and per-class limits.",
            len(selected_candidates),
            max_images,
        )

    images_out = [candidate[0] for candidate in selected_candidates]
    labels_out = [candidate[1] for candidate in selected_candidates]
    true_names = [candidate[2] for candidate in selected_candidates]
    pred_names = [candidate[3] for candidate in selected_candidates]
    confidences = [candidate[4] for candidate in selected_candidates]
    image_paths_out = [candidate[5] for candidate in selected_candidates]

    for true_name, pred_name, confidence, image_path in zip(
        true_names,
        pred_names,
        confidences,
        image_paths_out,
    ):
        LOGGER.info(
            "Selected example true=%s pred=%s confidence=%.4f correct=%s image=%s",
            true_name,
            pred_name,
            confidence,
            pred_name == true_name,
            image_path,
        )

    return (
        torch.stack(images_out, dim=0).to(device),
        torch.stack(labels_out, dim=0).to(device),
        true_names,
        pred_names,
        confidences,
        image_paths_out,
    )



def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)
    class_names_by_label = load_class_names(manifest)

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using manifest: %s", manifest)
    LOGGER.info("Using checkpoint: %s", checkpoint)
    LOGGER.info("Detected num_classes=%d", num_classes)

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(
        num_classes=num_classes,
        pretrained=False,
        trainable_modules=("layer4", "fc"),
    )
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    images, labels, true_names, pred_names, confidences, _paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names_by_label=class_names_by_label,
        max_images=args.max_images,
        max_per_class=args.max_per_class,
        allow_incorrect=args.selection == "all",
        only_incorrect=args.selection == "incorrect",
        seed=args.seed,
    )
    class_labels_by_name = {name: label for label, name in class_names_by_label.items()}
    attribution_targets = torch.tensor(
        [class_labels_by_name[name] for name in pred_names],
        device=device,
        dtype=labels.dtype,
    )

    logits = model(images)
    log_tensor_stats("logits", logits)

    target_layer = model.layer4[-1]
    LOGGER.info("Computing Grad-CAM on layer: %s", target_layer.__class__.__name__)
    gradcam_maps = gradcam_saliency(model, images, attribution_targets, target_layer)

    LOGGER.info("Computing Integrated Gradients with blurred image baselines")
    ig_maps, ig_attributions, ig_baselines = integrated_gradients_maps(
        model=model,
        inputs=images,
        targets=attribution_targets,
        steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size,
        blur_radius=args.blur_radius,
    )

    log_tensor_stats("gradcam.maps", gradcam_maps)
    log_tensor_stats("ig.attributions", ig_attributions)
    log_tensor_stats("ig.baselines", ig_baselines)
    log_tensor_stats("ig.maps", ig_maps)

    save_xai_grid(
        images=images,
        gradcam_maps=gradcam_maps,
        ig_maps=ig_maps,
        true_names=true_names,
        predicted_names=pred_names,
        confidences=confidences,
        output_path=args.output,
    )
    LOGGER.info("Saved attribution grid to %s", args.output)


if __name__ == "__main__":
    main()
