"""Run Phase 4 background perturbation stress tests on AwA2."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_xai import collect_correct_examples, load_checkpoint, load_idx_to_class
from scripts.train_baseline import infer_num_classes
from src.data import build_dataloaders
from src.model import build_resnet50_classifier, get_device
from src.perturb import (
    apply_perturbation_suite,
    predict_batch,
    save_perturbation_grid,
    write_stress_report,
)
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("run_stress_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Phase 4 background perturbations and compare predictions."
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
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase4_stress_test.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase4_stress_test.csv",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument(
        "--allow-incorrect",
        action="store_true",
        help="Allow selecting misclassified images when few correct examples are available.",
    )
    parser.add_argument(
        "--mask-strategy",
        choices=["center_ellipse", "center_box", "global"],
        default="center_ellipse",
        help=(
            "Approximate background definition. AwA2 has no masks, so center_ellipse "
            "keeps the central object region clean and perturbs the outside."
        ),
    )
    parser.add_argument("--foreground-scale", type=float, default=0.68)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def names_from_labels(labels: torch.Tensor, idx_to_class: dict[int, str]) -> list[str]:
    return [idx_to_class[int(label.item())] for label in labels.detach().cpu()]


def build_report_rows(
    true_names: list[str],
    original_predictions: torch.Tensor,
    original_confidences: torch.Tensor,
    perturbed_predictions: dict[str, torch.Tensor],
    perturbed_confidences: dict[str, torch.Tensor],
    idx_to_class: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    original_prediction_names = names_from_labels(original_predictions, idx_to_class)
    original_confidences_cpu = original_confidences.detach().cpu()

    for method_name, predictions in perturbed_predictions.items():
        prediction_names = names_from_labels(predictions, idx_to_class)
        confidences_cpu = perturbed_confidences[method_name].detach().cpu()
        for index, predicted_name in enumerate(prediction_names):
            original_pred = original_prediction_names[index]
            rows.append(
                {
                    "index": index,
                    "true_class": true_names[index],
                    "perturbation": method_name,
                    "original_prediction": original_pred,
                    "perturbed_prediction": predicted_name,
                    "original_confidence": float(original_confidences_cpu[index].item()),
                    "perturbed_confidence": float(confidences_cpu[index].item()),
                    "confidence_delta": float(
                        confidences_cpu[index].item()
                        - original_confidences_cpu[index].item()
                    ),
                    "prediction_changed": original_pred != predicted_name,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    idx_to_class = load_idx_to_class(manifest)
    num_classes = infer_num_classes(manifest)

    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("checkpoint=%s", checkpoint)
    LOGGER.info("num_classes=%d device=%s", num_classes, device)
    LOGGER.info(
        "mask_strategy=%s foreground_scale=%.3f",
        args.mask_strategy,
        args.foreground_scale,
    )

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

    images, labels, true_names, _predicted_names, _confidences = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        idx_to_class=idx_to_class,
        max_images=args.max_images,
        allow_incorrect=args.allow_incorrect,
    )

    original_predictions, original_confidences = predict_batch(model, images)
    original_prediction_names = names_from_labels(original_predictions, idx_to_class)

    background_mask, perturbed_batches = apply_perturbation_suite(
        inputs=images,
        mask_strategy=args.mask_strategy,
        foreground_scale=args.foreground_scale,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    perturbed_predictions: dict[str, torch.Tensor] = {}
    perturbed_confidences: dict[str, torch.Tensor] = {}
    perturbed_prediction_names: dict[str, list[str]] = {}

    for method_name, perturbed_images in perturbed_batches.items():
        predictions, confidences = predict_batch(model, perturbed_images)
        perturbed_predictions[method_name] = predictions
        perturbed_confidences[method_name] = confidences
        perturbed_prediction_names[method_name] = names_from_labels(predictions, idx_to_class)
        change_rate = (predictions != original_predictions).float().mean().item()
        LOGGER.info("%s prediction_change_rate=%.4f", method_name, change_rate)

    rows = build_report_rows(
        true_names=true_names,
        original_predictions=original_predictions,
        original_confidences=original_confidences,
        perturbed_predictions=perturbed_predictions,
        perturbed_confidences=perturbed_confidences,
        idx_to_class=idx_to_class,
    )
    write_stress_report(rows, args.csv_output)

    save_perturbation_grid(
        original_images=images,
        background_mask=background_mask,
        perturbed_batches=perturbed_batches,
        true_names=true_names,
        original_pred_names=original_prediction_names,
        perturbed_pred_names=perturbed_prediction_names,
        output_path=args.figure_output,
    )

    LOGGER.info("Phase 4 complete: figure=%s csv=%s", args.figure_output, args.csv_output)


if __name__ == "__main__":
    main()
