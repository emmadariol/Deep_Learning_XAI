"""Run advanced attribution diagnostics for the AwA2 XAI project."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_xai import collect_correct_examples, infer_num_classes, load_checkpoint, load_class_names
from src.attribution_audit import (
    class_discriminativeness,
    compute_attribution,
    deletion_insertion_curves,
    integrated_gradients_baseline_comparison,
    predict_with_logits,
    region_saliency_scores,
    saliency_entropy,
    save_class_discriminativeness_grid,
    save_deletion_insertion_plot,
    sensitivity_to_noise,
    trapezoid_auc,
    write_csv,
)
from src.data import build_dataloaders, infer_class_map_path
from src.explainability_audit import rank_correlation, topk_iou
from src.model import build_resnet50_classifier, get_device
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("run_advanced_attribution_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run faithfulness, region-allocation, class-discriminativeness, "
            "sensitivity and IG-baseline diagnostics for attribution maps."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2_subset_background20" / "awa2_manifest_subset.csv",
    )
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "scorecam", "integrated_gradients", "expected_gradients"],
        choices=["gradcam", "scorecam", "integrated_gradients", "expected_gradients", "input_gradients"],
        help="Attribution methods to audit.",
    )
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--ig-internal-batch-size", type=int, default=4)
    parser.add_argument("--expected-gradient-samples", type=int, default=12)
    parser.add_argument("--expected-gradient-baselines", type=int, default=16)
    parser.add_argument("--scorecam-max-channels", type=int, default=48)
    parser.add_argument("--scorecam-batch-size", type=int, default=16)
    parser.add_argument("--blur-radius", type=float, default=18.0)
    parser.add_argument("--mask-strategy", type=str, default="center_ellipse")
    parser.add_argument("--foreground-scale", type=float, default=0.68)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--curve-steps", type=int, default=10)
    parser.add_argument("--sensitivity-noise-std", type=float, default=0.03)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "advanced_attribution_audit.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "advanced_attribution_audit_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "advanced_attribution_audit",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _name_list(labels: torch.Tensor, idx_to_class: dict[int, str]) -> list[str]:
    return [idx_to_class[int(label.item())] for label in labels.detach().cpu()]


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary: list[dict[str, object]] = []
    numeric_keys = [
        "original_target_probability",
        "deletion_auc",
        "insertion_auc",
        "faithfulness_gap",
        "animal_saliency_ratio",
        "background_saliency_ratio",
        "saliency_entropy",
        "sensitivity_iou_top20",
        "sensitivity_spearman",
        "class_discriminativeness_iou_top20",
        "class_discriminativeness_spearman",
        "ig_blur_vs_black_iou_top20",
        "ig_blur_vs_black_spearman",
    ]
    for method, method_rows in grouped.items():
        summary_row: dict[str, object] = {"method": method, "examples": len(method_rows)}
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in method_rows
                if row.get(key, "") not in {"", None}
            ]
            summary_row[f"mean_{key}"] = f"{_mean(values):.6f}" if values else ""
        summary.append(summary_row)
    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    class_map = args.class_map.expanduser().resolve() if args.class_map else infer_class_map_path(manifest)
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    idx_to_class = load_class_names(manifest)
    num_classes = infer_num_classes(manifest)

    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("checkpoint=%s", checkpoint)
    LOGGER.info("device=%s", device)
    LOGGER.info("methods=%s", args.methods)

    loaders = build_dataloaders(
        manifest_path=manifest,
        class_map_path=class_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    images, labels, true_names, predicted_names, confidences, image_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names_by_label=idx_to_class,
        max_images=args.num_examples,
        max_per_class=1,
    )
    images = images.to(device)
    labels = labels.to(device)
    _, top1_labels, _ = predict_with_logits(model, images)
    target_probabilities = torch.tensor(confidences, device=device)
    expected_gradient_baselines = []
    for batch in loaders["train"]:
        for image in batch[0]:
            expected_gradient_baselines.append(image.detach().cpu())
            if len(expected_gradient_baselines) >= args.expected_gradient_baselines:
                break
        if len(expected_gradient_baselines) >= args.expected_gradient_baselines:
            break
    expected_gradient_baseline_tensor = torch.stack(expected_gradient_baselines, dim=0).to(device)

    fractions = [index / args.curve_steps for index in range(args.curve_steps + 1)]
    rows: list[dict[str, object]] = []

    for method in args.methods:
        LOGGER.info("auditing method=%s", method)
        bundle = compute_attribution(
            model=model,
            inputs=images,
            targets=labels,
            method=method,
            ig_steps=args.ig_steps,
            ig_internal_batch_size=args.ig_internal_batch_size,
            blur_radius=args.blur_radius,
            scorecam_max_channels=args.scorecam_max_channels,
            scorecam_batch_size=args.scorecam_batch_size,
            expected_gradients_samples=args.expected_gradient_samples,
            expected_gradients_baselines=expected_gradient_baseline_tensor,
        )
        maps = bundle.maps
        animal_ratio, background_ratio, _background_mask = region_saliency_scores(
            maps=maps,
            images=images,
            mask_strategy=args.mask_strategy,
            foreground_scale=args.foreground_scale,
        )
        entropy = saliency_entropy(maps)
        deletion_scores, insertion_scores = deletion_insertion_curves(
            model=model,
            inputs=images,
            targets=labels,
            maps=maps,
            fractions=fractions,
            baseline=bundle.baseline,
        )
        deletion_auc = trapezoid_auc(fractions, deletion_scores)
        insertion_auc = trapezoid_auc(fractions, insertion_scores)

        sensitivity_reference, sensitivity_candidate, same_prediction = sensitivity_to_noise(
            model=model,
            inputs=images,
            targets=labels,
            method=method,
            noise_std=args.sensitivity_noise_std,
            ig_steps=max(4, args.ig_steps // 2),
            ig_internal_batch_size=args.ig_internal_batch_size,
            blur_radius=args.blur_radius,
            scorecam_max_channels=args.scorecam_max_channels,
            scorecam_batch_size=args.scorecam_batch_size,
            expected_gradients_samples=max(4, args.expected_gradient_samples // 2),
            expected_gradients_baselines=expected_gradient_baseline_tensor,
        )
        top1, top2, top1_maps, top2_maps, class_metrics = class_discriminativeness(
            model=model,
            inputs=images,
            method=method,
            ig_steps=max(4, args.ig_steps // 2),
            ig_internal_batch_size=args.ig_internal_batch_size,
            blur_radius=args.blur_radius,
            top_fraction=args.top_fraction,
            scorecam_max_channels=args.scorecam_max_channels,
            scorecam_batch_size=args.scorecam_batch_size,
            expected_gradients_samples=max(4, args.expected_gradient_samples // 2),
            expected_gradients_baselines=expected_gradient_baseline_tensor,
        )

        ig_baseline_metrics = None
        if method == "integrated_gradients":
            _blur_maps, _black_maps, ig_baseline_metrics = integrated_gradients_baseline_comparison(
                model=model,
                inputs=images,
                targets=labels,
                steps=max(4, args.ig_steps // 2),
                internal_batch_size=args.ig_internal_batch_size,
                blur_radius=args.blur_radius,
                top_fraction=args.top_fraction,
            )

        save_deletion_insertion_plot(
            fractions=fractions,
            deletion_scores=deletion_scores,
            insertion_scores=insertion_scores,
            method=method,
            output_path=args.figure_dir / f"{method}_deletion_insertion.png",
        )
        save_class_discriminativeness_grid(
            images=images,
            top1_maps=top1_maps,
            top2_maps=top2_maps,
            true_names=true_names,
            top1_names=_name_list(top1, idx_to_class),
            top2_names=_name_list(top2, idx_to_class),
            output_path=args.figure_dir / f"{method}_class_discriminativeness.png",
        )

        for index in range(images.size(0)):
            sensitivity_iou = topk_iou(
                sensitivity_reference[index],
                sensitivity_candidate[index],
                args.top_fraction,
            )
            sensitivity_spearman = rank_correlation(
                sensitivity_reference[index],
                sensitivity_candidate[index],
            )
            row = {
                "method": method,
                "index": index,
                "image_path": image_paths[index],
                "true_class": true_names[index],
                "predicted_class": predicted_names[index],
                "target_class": idx_to_class[int(labels[index].item())],
                "top1_class": idx_to_class[int(top1_labels[index].item())],
                "top2_class": idx_to_class[int(top2[index].item())],
                "original_target_probability": f"{target_probabilities[index].item():.6f}",
                "deletion_auc": f"{deletion_auc[index].item():.6f}",
                "insertion_auc": f"{insertion_auc[index].item():.6f}",
                "faithfulness_gap": f"{(insertion_auc[index] - deletion_auc[index]).item():.6f}",
                "animal_saliency_ratio": f"{animal_ratio[index].item():.6f}",
                "background_saliency_ratio": f"{background_ratio[index].item():.6f}",
                "saliency_entropy": f"{entropy[index].item():.6f}",
                "sensitivity_same_prediction": bool(same_prediction[index].item()),
                "sensitivity_iou_top20": f"{sensitivity_iou:.6f}",
                "sensitivity_spearman": f"{sensitivity_spearman:.6f}",
                "class_discriminativeness_iou_top20": f"{class_metrics[index, 0].item():.6f}",
                "class_discriminativeness_spearman": f"{class_metrics[index, 1].item():.6f}",
                "ig_blur_vs_black_iou_top20": "",
                "ig_blur_vs_black_spearman": "",
            }
            if ig_baseline_metrics is not None:
                row["ig_blur_vs_black_iou_top20"] = f"{ig_baseline_metrics[index, 0].item():.6f}"
                row["ig_blur_vs_black_spearman"] = f"{ig_baseline_metrics[index, 1].item():.6f}"
            rows.append(row)

    summary_rows = _summary_rows(rows)
    write_csv(rows, args.report_output)
    write_csv(summary_rows, args.summary_output)

    for row in summary_rows:
        LOGGER.info("summary %s", row)
    LOGGER.info("advanced attribution audit complete: report=%s summary=%s figures=%s", args.report_output, args.summary_output, args.figure_dir)


if __name__ == "__main__":
    main()
