"""Run Phase 5 saliency degradation metrics after background perturbations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stress_test import names_from_labels
from scripts.run_xai import collect_correct_examples, load_checkpoint, load_idx_to_class
from scripts.train_baseline import infer_num_classes
from src.data import build_dataloaders, denormalize_batch
from src.metrics import (
    saliency_iou_at_top_percent,
    spearman_rank_correlation,
    write_metrics_csv,
)
from src.model import build_resnet50_classifier, get_device
from src.perturb import apply_perturbation_suite, predict_batch
from src.utils import set_seed, setup_logging
from src.xai import (
    GradCAM,
    ScoreCAM,
    expected_gradients,
    input_gradient_saliency,
    integrated_gradients,
    overlay_heatmap,
)

LOGGER = logging.getLogger("run_phase5_metrics")


def draw_panel_label(axis: plt.Axes, text: str) -> None:
    """Draw a readable label inside a matplotlib image panel."""
    axis.text(
        0.02,
        0.98,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saliency maps before and after Phase 4 perturbations."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2_subset_background20" / "awa2_manifest_subset.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase5_saliency_metrics.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase5_saliency_comparison.png",
    )
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        choices=["input_gradient", "gradcam", "scorecam", "integrated_gradients", "expected_gradients"],
        default=["gradcam", "scorecam", "integrated_gradients", "expected_gradients"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--expected-gradient-samples", type=int, default=12)
    parser.add_argument("--scorecam-max-channels", type=int, default=48)
    parser.add_argument("--scorecam-batch-size", type=int, default=16)
    parser.add_argument("--top-percent", type=float, default=20.0)
    parser.add_argument("--allow-incorrect", action="store_true")
    parser.add_argument(
        "--mask-strategy",
        choices=["center_ellipse", "center_box", "global"],
        default="center_ellipse",
    )
    parser.add_argument("--foreground-scale", type=float, default=0.68)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def compute_saliency_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    ig_steps: int,
    expected_gradient_samples: int = 12,
    scorecam_max_channels: int | None = 48,
    scorecam_batch_size: int = 16,
) -> torch.Tensor:
    """Compute one saliency method for a batch."""
    if method == "input_gradient":
        return input_gradient_saliency(model, images, targets)
    if method == "gradcam":
        gradcam = GradCAM(model, model.layer4[-1])
        try:
            return gradcam(images, targets)
        finally:
            gradcam.close()
    if method == "scorecam":
        scorecam = ScoreCAM(
            model,
            model.layer4[-1],
            max_channels=scorecam_max_channels,
            batch_size=scorecam_batch_size,
        )
        try:
            return scorecam(images, targets)
        finally:
            scorecam.close()
    if method == "integrated_gradients":
        return integrated_gradients(model, images, targets, steps=ig_steps)
    if method == "expected_gradients":
        return expected_gradients(
            model,
            images,
            targets,
            n_samples=expected_gradient_samples,
            internal_batch_size=4,
        )
    raise ValueError(f"Unsupported XAI method: {method}")


def build_metric_rows(
    true_names: list[str],
    target_names: list[str],
    original_prediction_names: list[str],
    original_confidences: torch.Tensor,
    perturbed_predictions: dict[str, torch.Tensor],
    perturbed_confidences: dict[str, torch.Tensor],
    saliency_maps: dict[str, dict[str, torch.Tensor]],
    top_percent: float,
    idx_to_class: dict[int, str],
) -> list[dict[str, object]]:
    """Build per-sample, per-perturbation, per-XAI metric rows."""
    rows: list[dict[str, object]] = []
    original_confidences_cpu = original_confidences.detach().cpu()

    for xai_method, method_maps in saliency_maps.items():
        original_maps = method_maps["original"]
        for perturbation_name, perturbed_maps in method_maps.items():
            if perturbation_name == "original":
                continue
            iou = saliency_iou_at_top_percent(
                original_maps,
                perturbed_maps,
                top_percent=top_percent,
            ).detach().cpu()
            spearman = spearman_rank_correlation(
                original_maps,
                perturbed_maps,
            ).detach().cpu()
            perturbed_names = names_from_labels(
                perturbed_predictions[perturbation_name],
                idx_to_class,
            )
            perturbed_conf_cpu = perturbed_confidences[perturbation_name].detach().cpu()
            for index, perturbed_name in enumerate(perturbed_names):
                rows.append(
                    {
                        "index": index,
                        "true_class": true_names[index],
                        "target_class_for_saliency": target_names[index],
                        "xai_method": xai_method,
                        "perturbation": perturbation_name,
                        "original_prediction": original_prediction_names[index],
                        "perturbed_prediction": perturbed_name,
                        "prediction_changed": original_prediction_names[index] != perturbed_name,
                        "original_confidence": float(original_confidences_cpu[index].item()),
                        "perturbed_confidence": float(perturbed_conf_cpu[index].item()),
                        "confidence_delta": float(
                            perturbed_conf_cpu[index].item()
                            - original_confidences_cpu[index].item()
                        ),
                        f"iou_top_{int(top_percent)}pct": float(iou[index].item()),
                        "spearman": float(spearman[index].item()),
                    }
                )
    return rows


def save_saliency_comparison_grid(
    images: torch.Tensor,
    perturbed_batches: dict[str, torch.Tensor],
    perturbed_predictions: dict[str, torch.Tensor],
    saliency_maps: dict[str, dict[str, torch.Tensor]],
    true_names: list[str],
    original_prediction_names: list[str],
    idx_to_class: dict[int, str],
    output_path: str | Path,
    preferred_method: str = "gradcam",
) -> None:
    """Save a visual comparison for one saliency method."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if preferred_method not in saliency_maps:
        preferred_method = next(iter(saliency_maps))

    method_maps = saliency_maps[preferred_method]
    perturbation_names = list(perturbed_batches)
    row_count = images.size(0)
    col_count = 2 + len(perturbation_names)

    original_denorm = denormalize_batch(images.detach().cpu()).clamp(0.0, 1.0)
    perturbed_denorm = {
        name: denormalize_batch(batch.detach().cpu()).clamp(0.0, 1.0)
        for name, batch in perturbed_batches.items()
    }
    perturbed_prediction_names = {
        name: names_from_labels(predictions, idx_to_class)
        for name, predictions in perturbed_predictions.items()
    }

    fig, axes = plt.subplots(row_count, col_count, figsize=(3.4 * col_count, 3.0 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(row_count):
        original_np = original_denorm[row].permute(1, 2, 0).numpy()
        original_map = method_maps["original"][row, 0].detach().cpu().numpy()
        axes[row, 0].imshow(original_np)
        original_label = (
            f"true: {true_names[row]}\n"
            f"original pred: {original_prediction_names[row]}"
        )
        draw_panel_label(axes[row, 0], original_label)
        axes[row, 0].set_title("original image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_heatmap(original_np, original_map))
        draw_panel_label(axes[row, 1], f"saliency target:\n{original_prediction_names[row]}")
        axes[row, 1].set_title(f"{preferred_method}: original")
        axes[row, 1].axis("off")

        for col, perturbation_name in enumerate(perturbation_names, start=2):
            perturbed_np = perturbed_denorm[perturbation_name][row].permute(1, 2, 0).numpy()
            perturbed_map = method_maps[perturbation_name][row, 0].detach().cpu().numpy()
            perturbed_name = perturbed_prediction_names[perturbation_name][row]
            axes[row, col].imshow(overlay_heatmap(perturbed_np, perturbed_map))
            draw_panel_label(
                axes[row, col],
                f"pred after:\n{original_prediction_names[row]} -> {perturbed_name}",
            )
            axes[row, col].set_title(f"{preferred_method}: {perturbation_name}")
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved saliency comparison grid: %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    idx_to_class = load_idx_to_class(manifest)
    num_classes = infer_num_classes(manifest)

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

    images, _labels, true_names, _predicted_names, _confidences, _image_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        idx_to_class=idx_to_class,
        max_images=args.max_images,
        allow_incorrect=args.allow_incorrect,
    )

    original_predictions, original_confidences = predict_batch(model, images)
    original_prediction_names = names_from_labels(original_predictions, idx_to_class)
    target_labels = original_predictions
    target_names = original_prediction_names

    _background_mask, perturbed_batches = apply_perturbation_suite(
        inputs=images,
        mask_strategy=args.mask_strategy,
        foreground_scale=args.foreground_scale,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    perturbed_predictions: dict[str, torch.Tensor] = {}
    perturbed_confidences: dict[str, torch.Tensor] = {}
    for perturbation_name, perturbed_images in perturbed_batches.items():
        predictions, confidences = predict_batch(model, perturbed_images)
        perturbed_predictions[perturbation_name] = predictions
        perturbed_confidences[perturbation_name] = confidences

    saliency_maps: dict[str, dict[str, torch.Tensor]] = {}
    for xai_method in args.xai_methods:
        LOGGER.info("Computing %s saliency maps", xai_method)
        method_maps: dict[str, torch.Tensor] = {
            "original": compute_saliency_maps(
                model=model,
                images=images,
                targets=target_labels,
                method=xai_method,
                ig_steps=args.ig_steps,
                expected_gradient_samples=args.expected_gradient_samples,
                scorecam_max_channels=args.scorecam_max_channels,
                scorecam_batch_size=args.scorecam_batch_size,
            ).detach().cpu()
        }
        for perturbation_name, perturbed_images in perturbed_batches.items():
            method_maps[perturbation_name] = compute_saliency_maps(
                model=model,
                images=perturbed_images,
                targets=target_labels,
                method=xai_method,
                ig_steps=args.ig_steps,
                expected_gradient_samples=args.expected_gradient_samples,
                scorecam_max_channels=args.scorecam_max_channels,
                scorecam_batch_size=args.scorecam_batch_size,
            ).detach().cpu()
        saliency_maps[xai_method] = method_maps

    rows = build_metric_rows(
        true_names=true_names,
        target_names=target_names,
        original_prediction_names=original_prediction_names,
        original_confidences=original_confidences,
        perturbed_predictions=perturbed_predictions,
        perturbed_confidences=perturbed_confidences,
        saliency_maps=saliency_maps,
        top_percent=args.top_percent,
        idx_to_class=idx_to_class,
    )
    write_metrics_csv(rows, args.csv_output)

    save_saliency_comparison_grid(
        images=images,
        perturbed_batches=perturbed_batches,
        perturbed_predictions=perturbed_predictions,
        saliency_maps=saliency_maps,
        true_names=true_names,
        original_prediction_names=original_prediction_names,
        idx_to_class=idx_to_class,
        output_path=args.figure_output,
        preferred_method="gradcam",
    )

    LOGGER.info("Phase 5 complete: figure=%s csv=%s", args.figure_output, args.csv_output)


if __name__ == "__main__":
    main()
