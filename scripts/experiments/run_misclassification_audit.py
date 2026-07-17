"""Explain baseline errors through target contrast and controlled interventions."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_xai import collect_correct_examples
from src.attribution_audit import compute_attribution
from src.bottleneck import load_concept_bottleneck_checkpoint
from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    read_manifest_classes,
)
from src.data import build_dataloaders, infer_num_classes, load_idx_to_class
from src.misclassification import (
    TargetPairScores,
    background_saliency_fraction,
    concept_evidence_rows,
    replace_top_salient_pixels,
    saliency_pair_diagnostics,
    save_concept_evidence_figure,
    save_contrastive_attribution_figure,
    save_deletion_curves_figure,
    save_perturbation_margin_figure,
    score_target_pair,
)
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.perturb import apply_perturbation_suite
from src.utils import set_seed, setup_logging, write_csv
from src.validation import (
    device_spec,
    log_level,
    nonnegative_float,
    nonnegative_int,
    open_unit_float,
    positive_int,
)
from src.xai import blurred_baseline, log_tensor_stats

LOGGER = logging.getLogger("run_misclassification_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the fixed wrong and true classes on misclassified AwA2 images "
            "using attribution, background interventions and deletion tests."
        )
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
        "--cbm-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "phase8_cbm_notebook.pt",
    )
    parser.add_argument("--metadata-root", type=Path, default=PROJECT_ROOT / "data" / "AWA2")
    parser.add_argument("--matrix-kind", choices=("continuous", "binary"), default="continuous")
    parser.add_argument("--skip-cbm", action="store_true")
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        choices=("gradcam", "integrated_gradients"),
        default=("gradcam", "integrated_gradients"),
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=("gaussian_noise", "color_shift", "background_swap"),
        default=("gaussian_noise", "color_shift", "background_swap"),
    )
    parser.add_argument(
        "--mask-strategy",
        choices=("center_ellipse", "center_box", "global"),
        default="center_ellipse",
    )
    parser.add_argument("--foreground-scale", type=open_unit_float, default=0.68)
    parser.add_argument("--noise-std", type=nonnegative_float, default=0.25)
    parser.add_argument("--top-fraction", type=open_unit_float, default=0.20)
    parser.add_argument(
        "--deletion-fractions",
        nargs="+",
        type=float,
        default=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--max-images", type=positive_int, default=4)
    parser.add_argument("--max-per-class", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--ig-steps", type=positive_int, default=16)
    parser.add_argument("--ig-internal-batch-size", type=positive_int, default=4)
    parser.add_argument("--blur-radius", type=nonnegative_float, default=18.0)
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "misclassification_decision_audit.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "misclassification_audit_summary.csv",
    )
    parser.add_argument(
        "--deletion-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "misclassification_deletion_audit.csv",
    )
    parser.add_argument(
        "--concept-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "misclassification_concept_evidence.csv",
    )
    parser.add_argument(
        "--concept-summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "misclassification_concept_summary.csv",
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "misclassification_audit",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def _validate_deletion_fractions(values: list[float] | tuple[float, ...]) -> list[float]:
    fractions = sorted(set(float(value) for value in values))
    if not fractions or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("deletion fractions must be finite values in [0, 1].")
    if 0.0 not in fractions:
        fractions.insert(0, 0.0)
    return fractions


def _predict_condition(
    model: torch.nn.Module,
    images: torch.Tensor,
    true_targets: torch.Tensor,
    wrong_targets: torch.Tensor,
) -> tuple[TargetPairScores, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(images)
    scores = score_target_pair(logits, true_targets, wrong_targets)
    confidences, predictions = scores.probabilities.max(dim=1)
    return scores, predictions.detach(), confidences.detach()


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def build_summary_rows(decision_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in decision_rows:
        grouped[(str(row["xai_method"]), str(row["condition"]))].append(row)

    summary: list[dict[str, object]] = []
    for (method, condition), rows in sorted(grouped.items()):
        summary.append(
            {
                "xai_method": method,
                "condition": condition,
                "examples": len(rows),
                "prediction_change_rate": _mean(rows, "prediction_changed"),
                "correction_to_true_rate": _mean(rows, "corrected_to_true"),
                "mean_wrong_probability_delta": _mean(rows, "wrong_probability_delta"),
                "mean_true_probability_delta": _mean(rows, "true_probability_delta"),
                "mean_margin_delta": _mean(rows, "margin_delta"),
                "mean_margin_reduction": _mean(rows, "margin_reduction"),
                "mean_wrong_true_map_iou": _mean(rows, "wrong_true_map_iou"),
                "mean_wrong_true_map_spearman": _mean(rows, "wrong_true_map_spearman"),
                "mean_wrong_map_stability_iou": _mean(rows, "wrong_map_stability_iou"),
                "mean_true_map_stability_iou": _mean(rows, "true_map_stability_iou"),
                "mean_wrong_background_saliency": _mean(rows, "wrong_background_saliency"),
                "mean_true_background_saliency": _mean(rows, "true_background_saliency"),
            }
        )
    return summary


def build_decision_rows(
    conditions: dict[str, torch.Tensor],
    scores_by_condition: dict[str, TargetPairScores],
    predictions_by_condition: dict[str, torch.Tensor],
    confidences_by_condition: dict[str, torch.Tensor],
    maps: dict[str, dict[str, dict[str, torch.Tensor]]],
    background_mask: torch.Tensor,
    labels: torch.Tensor,
    wrong_targets: torch.Tensor,
    true_names: list[str],
    wrong_names: list[str],
    image_paths: list[str],
    idx_to_class: dict[int, str],
    top_fraction: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    original_scores = scores_by_condition["original"]
    original_predictions = predictions_by_condition["original"]

    for method, method_maps in maps.items():
        original_wrong_maps = method_maps["original"]["wrong"]
        original_true_maps = method_maps["original"]["true"]
        for condition in conditions:
            condition_wrong_maps = method_maps[condition]["wrong"]
            condition_true_maps = method_maps[condition]["true"]
            contrast_iou, contrast_spearman = saliency_pair_diagnostics(
                condition_wrong_maps,
                condition_true_maps,
                top_fraction,
            )
            wrong_stability_iou, wrong_stability_spearman = saliency_pair_diagnostics(
                original_wrong_maps,
                condition_wrong_maps,
                top_fraction,
            )
            true_stability_iou, true_stability_spearman = saliency_pair_diagnostics(
                original_true_maps,
                condition_true_maps,
                top_fraction,
            )
            wrong_background = background_saliency_fraction(condition_wrong_maps, background_mask)
            true_background = background_saliency_fraction(condition_true_maps, background_mask)
            condition_scores = scores_by_condition[condition]

            for index in range(labels.size(0)):
                predicted_label = int(predictions_by_condition[condition][index].item())
                original_label = int(original_predictions[index].item())
                wrong_probability_delta = float(
                    condition_scores.wrong_probabilities[index].item()
                    - original_scores.wrong_probabilities[index].item()
                )
                true_probability_delta = float(
                    condition_scores.true_probabilities[index].item()
                    - original_scores.true_probabilities[index].item()
                )
                margin_delta = float(
                    condition_scores.margins[index].item()
                    - original_scores.margins[index].item()
                )
                rows.append(
                    {
                        "image_index": index,
                        "filepath": image_paths[index],
                        "true_class": true_names[index],
                        "fixed_wrong_class": wrong_names[index],
                        "xai_method": method,
                        "condition": condition,
                        "actual_prediction": idx_to_class[predicted_label],
                        "actual_confidence": float(confidences_by_condition[condition][index].item()),
                        "prediction_changed": predicted_label != original_label,
                        "corrected_to_true": predicted_label == int(labels[index].item()),
                        "true_logit": float(condition_scores.true_logits[index].item()),
                        "wrong_logit": float(condition_scores.wrong_logits[index].item()),
                        "true_probability": float(condition_scores.true_probabilities[index].item()),
                        "wrong_probability": float(condition_scores.wrong_probabilities[index].item()),
                        "wrong_minus_true_margin": float(condition_scores.margins[index].item()),
                        "wrong_probability_delta": wrong_probability_delta,
                        "true_probability_delta": true_probability_delta,
                        "margin_delta": margin_delta,
                        "margin_reduction": -margin_delta,
                        "wrong_true_map_iou": float(contrast_iou[index].item()),
                        "wrong_true_map_spearman": float(contrast_spearman[index].item()),
                        "wrong_map_stability_iou": float(wrong_stability_iou[index].item()),
                        "wrong_map_stability_spearman": float(wrong_stability_spearman[index].item()),
                        "true_map_stability_iou": float(true_stability_iou[index].item()),
                        "true_map_stability_spearman": float(true_stability_spearman[index].item()),
                        "wrong_background_saliency": float(wrong_background[index].item()),
                        "true_background_saliency": float(true_background[index].item()),
                    }
                )
    return rows


def run_deletion_audit(
    model: torch.nn.Module,
    images: torch.Tensor,
    maps: dict[str, dict[str, dict[str, torch.Tensor]]],
    true_targets: torch.Tensor,
    wrong_targets: torch.Tensor,
    true_names: list[str],
    wrong_names: list[str],
    image_paths: list[str],
    idx_to_class: dict[int, str],
    fractions: list[float],
    blur_radius: float,
) -> list[dict[str, object]]:
    baseline = blurred_baseline(images, blur_radius=blur_radius)
    original_scores, _original_predictions, _original_confidences = _predict_condition(
        model,
        images,
        true_targets,
        wrong_targets,
    )
    rows: list[dict[str, object]] = []
    for method, method_maps in maps.items():
        for ranking_target in ("wrong", "true"):
            ranking_maps = method_maps["original"][ranking_target]
            for fraction in fractions:
                deleted = replace_top_salient_pixels(images, ranking_maps, fraction, baseline)
                scores, predictions, confidences = _predict_condition(
                    model,
                    deleted,
                    true_targets,
                    wrong_targets,
                )
                for index in range(images.size(0)):
                    predicted_label = int(predictions[index].item())
                    margin_delta = float(
                        scores.margins[index].item() - original_scores.margins[index].item()
                    )
                    rows.append(
                        {
                            "image_index": index,
                            "filepath": image_paths[index],
                            "true_class": true_names[index],
                            "fixed_wrong_class": wrong_names[index],
                            "xai_method": method,
                            "ranking_target": ranking_target,
                            "deleted_fraction": fraction,
                            "actual_prediction": idx_to_class[predicted_label],
                            "actual_confidence": float(confidences[index].item()),
                            "corrected_to_true": predicted_label == int(true_targets[index].item()),
                            "true_probability": float(scores.true_probabilities[index].item()),
                            "wrong_probability": float(scores.wrong_probabilities[index].item()),
                            "wrong_minus_true_margin": float(scores.margins[index].item()),
                            "wrong_probability_drop": float(
                                original_scores.wrong_probabilities[index].item()
                                - scores.wrong_probabilities[index].item()
                            ),
                            "true_probability_drop": float(
                                original_scores.true_probabilities[index].item()
                                - scores.true_probabilities[index].item()
                            ),
                            "margin_delta": margin_delta,
                            "margin_reduction": -margin_delta,
                        }
                    )
    return rows


def run_cbm_comparator(
    checkpoint_path: Path,
    metadata_root: Path,
    matrix_kind: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    wrong_targets: torch.Tensor,
    true_names: list[str],
    wrong_names: list[str],
    image_paths: list[str],
    idx_to_class: dict[int, str],
    manifest: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model, metadata = load_concept_bottleneck_checkpoint(
        checkpoint_path,
        device,
        expected_class_mapping=idx_to_class,
    )
    concept_names = [str(name) for name in metadata["concept_names"]]
    concept_indices = [int(index) for index in metadata.get("concept_indices", [])]
    if len(concept_indices) != len(concept_names):
        raise ValueError("CBM checkpoint concept indices and names disagree.")

    manifest_classes = read_manifest_classes(manifest)
    resolved_metadata_root = find_awa2_metadata_root(
        [metadata_root, manifest.parent, PROJECT_ROOT / "data" / "AWA2", PROJECT_ROOT / "data"]
    )
    bank = load_awa2_concepts(
        resolved_metadata_root,
        matrix_kind=str(metadata.get("matrix_kind", matrix_kind)),
    )
    bank = align_concept_bank_to_manifest(bank, manifest_classes)
    prototypes = torch.tensor(
        bank.normalized_matrix()[:, concept_indices],
        dtype=images.dtype,
        device=device,
    )
    true_prototypes = prototypes[labels]
    wrong_prototypes = prototypes[wrong_targets]

    with torch.no_grad():
        outputs = model(images)
        cbm_scores = score_target_pair(outputs.class_logits, labels, wrong_targets)
        cbm_confidences, cbm_predictions = cbm_scores.probabilities.max(dim=1)
        true_similarity = F.cosine_similarity(outputs.concept_probs, true_prototypes, dim=1)
        wrong_similarity = F.cosine_similarity(outputs.concept_probs, wrong_prototypes, dim=1)

    evidence_rows = concept_evidence_rows(
        model.class_head,
        outputs.concept_probs.detach(),
        true_prototypes,
        wrong_prototypes,
        labels,
        wrong_targets,
        concept_names,
    )
    for row in evidence_rows:
        index = int(row["image_index"])
        row.update(
            {
                "filepath": image_paths[index],
                "true_class": true_names[index],
                "fixed_wrong_class": wrong_names[index],
                "evidence_scope": "parallel_concept_bottleneck_model",
            }
        )

    summary_rows: list[dict[str, object]] = []
    for index in range(images.size(0)):
        cbm_prediction = int(cbm_predictions[index].item())
        summary_rows.append(
            {
                "image_index": index,
                "filepath": image_paths[index],
                "true_class": true_names[index],
                "wrong_class": wrong_names[index],
                "cbm_predicted_class": idx_to_class[cbm_prediction],
                "cbm_confidence": float(cbm_confidences[index].item()),
                "cbm_true_probability": float(cbm_scores.true_probabilities[index].item()),
                "cbm_wrong_probability": float(cbm_scores.wrong_probabilities[index].item()),
                "cbm_wrong_vs_true_margin": float(cbm_scores.margins[index].item()),
                "concept_cosine_to_true_prototype": float(true_similarity[index].item()),
                "concept_cosine_to_wrong_prototype": float(wrong_similarity[index].item()),
                "concept_similarity_gap_wrong_minus_true": float(
                    wrong_similarity[index].item() - true_similarity[index].item()
                ),
                "interpretation": "CBM comparator; not direct ResNet causal evidence",
            }
        )
    return evidence_rows, summary_rows


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)
    fractions = _validate_deletion_fractions(args.deletion_fractions)
    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    figure_directory = args.figure_directory.expanduser().resolve()
    figure_directory.mkdir(parents=True, exist_ok=True)
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
    load_checkpoint(model, checkpoint, device, expected_class_mapping=idx_to_class)
    model.to(device).eval()

    images, labels, true_names, _stored_predictions, _stored_confidences, image_paths = (
        collect_correct_examples(
            model=model,
            loader=loaders["test"],
            device=device,
            idx_to_class=idx_to_class,
            max_images=args.max_images,
            max_per_class=args.max_per_class,
            only_incorrect=True,
            seed=args.seed,
        )
    )
    with torch.no_grad():
        original_logits = model(images)
        original_probabilities = torch.softmax(original_logits, dim=1)
        wrong_targets = original_probabilities.argmax(dim=1)
    if torch.any(wrong_targets == labels):
        raise RuntimeError("Misclassification selection returned a correctly classified image.")
    wrong_names = [idx_to_class[int(label.item())] for label in wrong_targets]
    log_tensor_stats("misclassification.original_logits", original_logits)

    background_mask, perturbed = apply_perturbation_suite(
        inputs=images,
        mask_strategy=args.mask_strategy,
        foreground_scale=args.foreground_scale,
        methods=tuple(args.perturbations),
        noise_std=args.noise_std,
        seed=args.seed,
    )
    conditions = {"original": images, **perturbed}
    scores_by_condition: dict[str, TargetPairScores] = {}
    predictions_by_condition: dict[str, torch.Tensor] = {}
    confidences_by_condition: dict[str, torch.Tensor] = {}
    for condition, condition_images in conditions.items():
        scores, predictions, confidences = _predict_condition(
            model,
            condition_images,
            labels,
            wrong_targets,
        )
        scores_by_condition[condition] = scores
        predictions_by_condition[condition] = predictions
        confidences_by_condition[condition] = confidences

    maps: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for method in args.xai_methods:
        maps[method] = {}
        for condition, condition_images in conditions.items():
            maps[method][condition] = {}
            for role, targets in (("wrong", wrong_targets), ("true", labels)):
                LOGGER.info("attribution method=%s condition=%s target=%s", method, condition, role)
                bundle = compute_attribution(
                    model,
                    condition_images,
                    targets,
                    method=method,
                    ig_steps=args.ig_steps,
                    ig_internal_batch_size=args.ig_internal_batch_size,
                    blur_radius=args.blur_radius,
                )
                maps[method][condition][role] = bundle.maps.detach()
                log_tensor_stats(
                    f"misclassification.{method}.{condition}.{role}.maps",
                    bundle.maps,
                )

    decision_rows = build_decision_rows(
        conditions=conditions,
        scores_by_condition=scores_by_condition,
        predictions_by_condition=predictions_by_condition,
        confidences_by_condition=confidences_by_condition,
        maps=maps,
        background_mask=background_mask,
        labels=labels,
        wrong_targets=wrong_targets,
        true_names=true_names,
        wrong_names=wrong_names,
        image_paths=image_paths,
        idx_to_class=idx_to_class,
        top_fraction=args.top_fraction,
    )
    summary_rows = build_summary_rows(decision_rows)
    write_csv(decision_rows, args.decision_output)
    write_csv(summary_rows, args.summary_output)

    deletion_rows = run_deletion_audit(
        model=model,
        images=images,
        maps=maps,
        true_targets=labels,
        wrong_targets=wrong_targets,
        true_names=true_names,
        wrong_names=wrong_names,
        image_paths=image_paths,
        idx_to_class=idx_to_class,
        fractions=fractions,
        blur_radius=args.blur_radius,
    )
    write_csv(deletion_rows, args.deletion_output)

    for method in args.xai_methods:
        save_contrastive_attribution_figure(
            images=images,
            wrong_maps=maps[method]["original"]["wrong"],
            true_maps=maps[method]["original"]["true"],
            scores=scores_by_condition["original"],
            true_names=true_names,
            wrong_names=wrong_names,
            method=method,
            output_path=figure_directory / f"target_contrast_{method}.png",
        )
    save_perturbation_margin_figure(
        images=images,
        scores_by_condition=scores_by_condition,
        predictions_by_condition=predictions_by_condition,
        idx_to_class=idx_to_class,
        true_names=true_names,
        wrong_names=wrong_names,
        output_path=figure_directory / "perturbation_decision_margins.png",
    )
    save_deletion_curves_figure(
        deletion_rows,
        methods=args.xai_methods,
        output_path=figure_directory / "deletion_curves.png",
    )

    cbm_checkpoint = args.cbm_checkpoint.expanduser().resolve()
    if not args.skip_cbm and cbm_checkpoint.exists():
        concept_rows, concept_summary_rows = run_cbm_comparator(
            checkpoint_path=cbm_checkpoint,
            metadata_root=args.metadata_root,
            matrix_kind=args.matrix_kind,
            images=images,
            labels=labels,
            wrong_targets=wrong_targets,
            true_names=true_names,
            wrong_names=wrong_names,
            image_paths=image_paths,
            idx_to_class=idx_to_class,
            manifest=manifest,
            device=device,
        )
        write_csv(concept_rows, args.concept_output)
        write_csv(concept_summary_rows, args.concept_summary_output)
        save_concept_evidence_figure(
            concept_rows,
            concept_summary_rows,
            output_path=figure_directory / "cbm_concept_evidence.png",
        )
    elif not args.skip_cbm:
        LOGGER.warning("CBM checkpoint not found; skipping semantic comparator: %s", cbm_checkpoint)

    LOGGER.info(
        "Misclassification audit complete: examples=%d decision=%s deletion=%s figures=%s",
        images.size(0),
        args.decision_output,
        args.deletion_output,
        figure_directory,
    )


if __name__ == "__main__":
    main()
