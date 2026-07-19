"""Measure saliency degradation after background perturbations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_xai import collect_correct_examples
from src.data import (
    build_dataloaders,
    denormalize_batch,
    infer_num_classes,
    load_idx_to_class,
    names_from_labels,
)
from src.metrics import (
    percentage_token,
    saliency_iou_at_top_percent,
    spearman_rank_correlation,
)
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.perturb import (
    apply_perturbation_suite,
    predict_batch_probabilities,
    probabilities_for_targets,
    save_perturbation_grid,
)
from src.utils import set_seed, setup_logging, write_csv
from src.validation import (
    device_spec,
    log_level,
    nonnegative_float,
    nonnegative_int,
    open_percentage_float,
    open_unit_float,
    positive_int,
)
from src.xai import (
    gradcam_saliency,
    integrated_gradients,
    overlay_heatmap,
)

LOGGER = logging.getLogger("run_background_stress_metrics")


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
        description="Compare saliency maps before and after background perturbations."
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
        "--perturbation-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase5_perturbations.png",
        help=(
            "Output grid with the original image, approximate background mask, and "
            "raw perturbations."
        ),
    )
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        choices=["gradcam", "integrated_gradients"],
        default=["gradcam", "integrated_gradients"],
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--max-images", type=positive_int, default=4)
    parser.add_argument("--ig-steps", type=positive_int, default=16)
    parser.add_argument("--ig-internal-batch-size", type=positive_int, default=4)
    parser.add_argument("--top-percent", type=open_percentage_float, default=20.0)
    parser.add_argument("--allow-incorrect", action="store_true")
    parser.add_argument(
        "--mask-strategy",
        choices=["center_ellipse", "center_box", "global"],
        default="center_ellipse",
    )
    parser.add_argument("--foreground-scale", type=open_unit_float, default=0.68)
    parser.add_argument("--noise-std", type=nonnegative_float, default=0.25)
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def compute_saliency_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    ig_steps: int,
    ig_internal_batch_size: int = 4,
) -> torch.Tensor:
    """Compute one maintained saliency method for a batch."""
    if method == "gradcam":
        return gradcam_saliency(model, images, targets, model.layer4[-1])
    if method == "integrated_gradients":
        return integrated_gradients(
            model,
            images,
            targets,
            steps=ig_steps,
            internal_batch_size=ig_internal_batch_size,
        )
    raise ValueError(f"Unsupported XAI method: {method}")

def build_metric_rows(
    true_names: list[str],
    target_names: list[str],
    original_prediction_names: list[str],
    original_confidences: torch.Tensor,
    original_target_probabilities: torch.Tensor,
    perturbed_predictions: dict[str, torch.Tensor],
    perturbed_confidences: dict[str, torch.Tensor],
    perturbed_target_probabilities: dict[str, torch.Tensor],
    saliency_maps: dict[str, dict[str, torch.Tensor]],
    top_percent: float,
    idx_to_class: dict[int, str],
) -> list[dict[str, object]]:
    """Build metrics while comparing the same target class before and after."""
    rows: list[dict[str, object]] = []
    original_confidences_cpu = original_confidences.detach().cpu()
    original_target_probabilities_cpu = original_target_probabilities.detach().cpu()
    top_percent_token = percentage_token(top_percent)

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
            perturbed_target_prob_cpu = perturbed_target_probabilities[
                perturbation_name
            ].detach().cpu()
            for index, perturbed_name in enumerate(perturbed_names):
                original_target_probability = float(
                    original_target_probabilities_cpu[index].item()
                )
                perturbed_target_probability = float(
                    perturbed_target_prob_cpu[index].item()
                )
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
                        "original_target_probability": original_target_probability,
                        "perturbed_target_probability": perturbed_target_probability,
                        "confidence_delta": (
                            perturbed_target_probability - original_target_probability
                        ),
                        "confidence_drop": (
                            original_target_probability - perturbed_target_probability
                        ),
                        f"iou_top_{top_percent_token}pct": float(iou[index].item()),
                        "spearman": float(spearman[index].item()),
                    }
                )
    return rows


def save_saliency_comparison_grid(
    images: torch.Tensor,
    perturbed_batches: dict[str, torch.Tensor],
    perturbed_predictions: dict[str, torch.Tensor],
    saliency_maps: dict[str, dict[str, torch.Tensor]],
    predicted_target_saliency_maps: dict[str, dict[str, torch.Tensor]],
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
    predicted_target_method_maps = predicted_target_saliency_maps.get(preferred_method, {})
    perturbation_names = list(perturbed_batches)
    row_count = images.size(0)
    has_background_swap_predicted_target = "background_swap" in predicted_target_method_maps
    col_count = 2 + len(perturbation_names) + int(has_background_swap_predicted_target)

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

        if has_background_swap_predicted_target:
            col = 2 + len(perturbation_names)
            perturbed_np = perturbed_denorm["background_swap"][row].permute(1, 2, 0).numpy()
            perturbed_name = perturbed_prediction_names["background_swap"][row]
            predicted_target_map = (
                predicted_target_method_maps["background_swap"][row, 0].detach().cpu().numpy()
            )
            axes[row, col].imshow(overlay_heatmap(perturbed_np, predicted_target_map))
            draw_panel_label(
                axes[row, col],
                f"saliency target:\n{perturbed_name}",
            )
            axes[row, col].set_title(
                f"{preferred_method}: background_swap\npredicted target"
            )
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
        seed=args.seed,
    )

    original_predictions, original_confidences, original_probabilities = (
        predict_batch_probabilities(model, images)
    )
    original_prediction_names = names_from_labels(original_predictions, idx_to_class)
    target_labels = original_predictions
    target_names = original_prediction_names
    original_target_probabilities = probabilities_for_targets(
        original_probabilities,
        target_labels,
    )

    background_mask, perturbed_batches = apply_perturbation_suite(
        inputs=images,
        mask_strategy=args.mask_strategy,
        foreground_scale=args.foreground_scale,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    perturbed_predictions: dict[str, torch.Tensor] = {}
    perturbed_confidences: dict[str, torch.Tensor] = {}
    perturbed_target_probabilities: dict[str, torch.Tensor] = {}
    for perturbation_name, perturbed_images in perturbed_batches.items():
        predictions, confidences, probabilities = predict_batch_probabilities(
            model,
            perturbed_images,
        )
        perturbed_predictions[perturbation_name] = predictions
        perturbed_confidences[perturbation_name] = confidences
        perturbed_target_probabilities[perturbation_name] = probabilities_for_targets(
            probabilities,
            target_labels,
        )

    save_perturbation_grid(
        original_images=images,
        background_mask=background_mask,
        perturbed_batches=perturbed_batches,
        true_names=true_names,
        original_pred_names=original_prediction_names,
        perturbed_pred_names={
            name: names_from_labels(predictions, idx_to_class)
            for name, predictions in perturbed_predictions.items()
        },
        output_path=args.perturbation_figure_output,
    )

    saliency_maps: dict[str, dict[str, torch.Tensor]] = {}
    predicted_target_saliency_maps: dict[str, dict[str, torch.Tensor]] = {}
    for xai_method in args.xai_methods:
        LOGGER.info("Computing %s saliency maps", xai_method)
        method_maps: dict[str, torch.Tensor] = {
            "original": compute_saliency_maps(
                model=model,
                images=images,
                targets=target_labels,
                method=xai_method,
                ig_steps=args.ig_steps,
                ig_internal_batch_size=args.ig_internal_batch_size,
            ).detach().cpu()
        }
        for perturbation_name, perturbed_images in perturbed_batches.items():
            method_maps[perturbation_name] = compute_saliency_maps(
                model=model,
                images=perturbed_images,
                targets=target_labels,
                method=xai_method,
                ig_steps=args.ig_steps,
                ig_internal_batch_size=args.ig_internal_batch_size,
            ).detach().cpu()
        saliency_maps[xai_method] = method_maps

        method_predicted_target_maps: dict[str, torch.Tensor] = {}
        if "background_swap" in perturbed_batches:
            LOGGER.info(
                "Computing %s saliency maps for background_swap predicted targets",
                xai_method,
            )
            method_predicted_target_maps["background_swap"] = compute_saliency_maps(
                model=model,
                images=perturbed_batches["background_swap"],
                targets=perturbed_predictions["background_swap"],
                method=xai_method,
                ig_steps=args.ig_steps,
                ig_internal_batch_size=args.ig_internal_batch_size,
            ).detach().cpu()
        predicted_target_saliency_maps[xai_method] = method_predicted_target_maps

    rows = build_metric_rows(
        true_names=true_names,
        target_names=target_names,
        original_prediction_names=original_prediction_names,
        original_confidences=original_confidences,
        original_target_probabilities=original_target_probabilities,
        perturbed_predictions=perturbed_predictions,
        perturbed_confidences=perturbed_confidences,
        perturbed_target_probabilities=perturbed_target_probabilities,
        saliency_maps=saliency_maps,
        top_percent=args.top_percent,
        idx_to_class=idx_to_class,
    )
    write_csv(rows, args.csv_output)

    save_saliency_comparison_grid(
        images=images,
        perturbed_batches=perturbed_batches,
        perturbed_predictions=perturbed_predictions,
        saliency_maps=saliency_maps,
        predicted_target_saliency_maps=predicted_target_saliency_maps,
        true_names=true_names,
        original_prediction_names=original_prediction_names,
        idx_to_class=idx_to_class,
        output_path=args.figure_output,
        preferred_method="gradcam",
    )
    figure_output = Path(args.figure_output).expanduser().resolve()
    for xai_method in saliency_maps:
        method_figure = figure_output.with_name(
            f"{figure_output.stem}_{xai_method}{figure_output.suffix}"
        )
        save_saliency_comparison_grid(
            images=images,
            perturbed_batches=perturbed_batches,
            perturbed_predictions=perturbed_predictions,
            saliency_maps=saliency_maps,
            predicted_target_saliency_maps=predicted_target_saliency_maps,
            true_names=true_names,
            original_prediction_names=original_prediction_names,
            idx_to_class=idx_to_class,
            output_path=method_figure,
            preferred_method=xai_method,
        )

    LOGGER.info(
        "Background stress metrics complete: perturbations=%s saliency=%s csv=%s",
        args.perturbation_figure_output,
        args.figure_output,
        args.csv_output,
    )


if __name__ == "__main__":
    main()
