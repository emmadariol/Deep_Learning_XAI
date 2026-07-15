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
    at_least_two_int,
    device_spec,
    log_level,
    nonnegative_float,
    nonnegative_int,
    positive_int,
)
from src.xai import (
    ScoreCAM,
    expected_gradients_maps,
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
    parser.add_argument("--expected-gradient-samples", type=positive_int, default=24)
    parser.add_argument("--expected-gradient-internal-batch-size", type=positive_int, default=8)
    parser.add_argument("--expected-gradient-baselines", type=at_least_two_int, default=16)
    parser.add_argument("--scorecam-max-channels", type=positive_int, default=64)
    parser.add_argument("--scorecam-batch-size", type=positive_int, default=16)
    parser.add_argument("--blur-radius", type=nonnegative_float, default=18.0)
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
            selected = torch.ones_like(labels, dtype=torch.bool) if allow_incorrect else preds == labels

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


def collect_reference_images(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int = 16,
) -> torch.Tensor:
    """Collect normalized images from a loader for Expected Gradients baselines."""
    if max_images < 2:
        raise ValueError("Expected Gradients requires at least two reference images.")
    images_out: list[torch.Tensor] = []
    for batch in loader:
        images = batch[0]
        for image in images:
            images_out.append(image.detach().cpu())
            if len(images_out) >= max_images:
                reference = torch.stack(images_out, dim=0).to(device)
                log_tensor_stats("expected_gradients.reference_images", reference)
                return reference
    if not images_out:
        raise RuntimeError("Could not collect reference images for Expected Gradients.")
    if len(images_out) < 2:
        raise RuntimeError("The reference loader yielded fewer than two images.")
    reference = torch.stack(images_out, dim=0).to(device)
    log_tensor_stats("expected_gradients.reference_images", reference)
    return reference


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

    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    images, labels, true_names, pred_names, confidences, _image_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names_by_label=class_names_by_label,
        max_images=args.max_images,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    log_tensor_stats("xai.inputs", images)
    LOGGER.info("Selected target labels: %s", [int(label.item()) for label in labels])

    gradcam_maps = gradcam_saliency(model, images, labels, model.layer4[-1])
    scorecam = ScoreCAM(
        model=model,
        target_layer=model.layer4[-1],
        max_channels=args.scorecam_max_channels,
        batch_size=args.scorecam_batch_size,
        blur_radius=args.blur_radius,
    )
    try:
        scorecam_maps = scorecam(images, labels)
    finally:
        scorecam.close()

    ig_maps, _ig_attributions, _ig_baseline = integrated_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size,
        blur_radius=args.blur_radius,
    )
    expected_baselines = collect_reference_images(
        loaders["train"],
        device=device,
        max_images=args.expected_gradient_baselines,
    )
    expected_maps, _expected_attributions, _expected_baseline_pool = expected_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        baselines=expected_baselines,
        n_samples=args.expected_gradient_samples,
        internal_batch_size=args.expected_gradient_internal_batch_size,
        seed=args.seed,
    )

    save_xai_grid(
        images=images,
        gradcam_maps=gradcam_maps,
        ig_maps=ig_maps,
        scorecam_maps=scorecam_maps,
        expected_gradients_maps=expected_maps,
        true_names=true_names,
        pred_names=pred_names,
        confidences=confidences,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
