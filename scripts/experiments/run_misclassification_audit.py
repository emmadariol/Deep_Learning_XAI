"""Explain baseline errors through target contrast and controlled interventions."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "deep_learning_xai"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg-cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

from scripts.experiments.run_xai import collect_correct_examples
from src.attribution_audit import compute_attribution
from src.bottleneck import ConceptBottleneckModel, load_concept_bottleneck_checkpoint
from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    normalize_class_name,
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
    parser.add_argument(
        "--legacy-concept-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / "phase8_concept_metrics_notebook.csv"
        ),
        help=(
            "Ordered concept report used only when loading a legacy CBM checkpoint "
            "without embedded semantic metadata."
        ),
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
    parser.add_argument(
        "--target-pairs",
        nargs="*",
        default=None,
        metavar="TRUE:WRONG",
        help=(
            "Optional fixed true/wrong class pairs for the contrastive audit. "
            "Exact true->wrong errors are preferred; if none exists for a pair, "
            "the test image with the highest wrong-class probability is used. "
            "Append @FILENAME to force a specific image."
        ),
    )
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _class_lookup(idx_to_class: dict[int, str]) -> dict[str, tuple[int, str]]:
    return {
        normalize_class_name(class_name): (label, class_name)
        for label, class_name in idx_to_class.items()
    }


def _parse_target_pairs(
    values: list[str] | None,
    idx_to_class: dict[int, str],
) -> list[tuple[int, str, int, str, str | None]]:
    if not values:
        return []
    lookup = _class_lookup(idx_to_class)
    pairs: list[tuple[int, str, int, str, str | None]] = []
    for value in values:
        pair_value, image_selector = (
            [part.strip() for part in value.split("@", 1)]
            if "@" in value
            else (value.strip(), None)
        )
        separator = "->" if "->" in pair_value else ":"
        if separator not in pair_value:
            raise ValueError(
                f"Invalid target pair {value!r}; expected TRUE:WRONG or TRUE->WRONG."
            )
        true_raw, wrong_raw = [part.strip() for part in pair_value.split(separator, 1)]
        true_key = normalize_class_name(true_raw)
        wrong_key = normalize_class_name(wrong_raw)
        if true_key not in lookup:
            raise ValueError(f"Unknown true class in target pair {value!r}: {true_raw!r}")
        if wrong_key not in lookup:
            raise ValueError(f"Unknown wrong class in target pair {value!r}: {wrong_raw!r}")
        true_label, true_name = lookup[true_key]
        wrong_label, wrong_name = lookup[wrong_key]
        if true_label == wrong_label:
            raise ValueError(f"Target pair must use two different classes: {value!r}")
        pairs.append((true_label, true_name, wrong_label, wrong_name, image_selector))
    return pairs


def _matches_image_selector(image_path: str, image_selector: str | None) -> bool:
    if image_selector is None:
        return True
    return image_selector == Path(image_path).name or image_selector in image_path


def collect_target_pair_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_pairs: list[tuple[int, str, int, str, str | None]],
    idx_to_class: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    """Select one test image for each requested true/wrong target pair."""
    best_exact: list[tuple[float, tuple[object, ...]] | None] = [None] * len(target_pairs)
    best_fallback: list[tuple[float, tuple[object, ...]] | None] = [None] * len(target_pairs)
    pairs_by_true_label: dict[int, list[int]] = {}
    for pair_index, (true_label, _true_name, _wrong_label, _wrong_name, _image_selector) in enumerate(target_pairs):
        pairs_by_true_label.setdefault(true_label, []).append(pair_index)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            image_paths = list(batch[3])
            probabilities = torch.softmax(model(images), dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for batch_index in range(images.size(0)):
                true_label = int(labels[batch_index].item())
                if true_label not in pairs_by_true_label:
                    continue
                predicted_label = int(predictions[batch_index].item())
                for pair_index in pairs_by_true_label[true_label]:
                    (
                        _pair_true_label,
                        true_name,
                        wrong_label,
                        wrong_name,
                        image_selector,
                    ) = target_pairs[pair_index]
                    if not _matches_image_selector(image_paths[batch_index], image_selector):
                        continue
                    wrong_probability = float(
                        probabilities[batch_index, wrong_label].detach().cpu().item()
                    )
                    candidate = (
                        images[batch_index].detach().cpu(),
                        labels[batch_index].detach().cpu(),
                        torch.tensor(wrong_label, dtype=torch.long),
                        true_name,
                        wrong_name,
                        image_paths[batch_index],
                        idx_to_class[predicted_label],
                        float(confidences[batch_index].detach().cpu().item()),
                    )
                    if image_selector is not None:
                        if predicted_label == wrong_label:
                            best_exact[pair_index] = (wrong_probability, candidate)
                        else:
                            best_fallback[pair_index] = (wrong_probability, candidate)
                        continue
                    fallback = best_fallback[pair_index]
                    if fallback is None or wrong_probability > fallback[0]:
                        best_fallback[pair_index] = (wrong_probability, candidate)
                    if predicted_label == wrong_label:
                        exact = best_exact[pair_index]
                        if exact is None or wrong_probability > exact[0]:
                            best_exact[pair_index] = (wrong_probability, candidate)

    selected: list[tuple[object, ...]] = []
    for pair_index, (_true_label, true_name, _wrong_label, wrong_name, image_selector) in enumerate(target_pairs):
        exact = best_exact[pair_index]
        fallback = best_fallback[pair_index]
        chosen = exact if exact is not None else fallback
        if chosen is None:
            detail = f" matching {image_selector!r}" if image_selector else ""
            raise RuntimeError(
                f"No test images found for requested pair {true_name}:{wrong_name}{detail}."
            )
        if exact is None:
            if image_selector is None:
                LOGGER.warning(
                    "No exact %s->%s error found; using highest wrong-class probability fallback.",
                    true_name,
                    wrong_name,
                )
            else:
                LOGGER.warning(
                    "Requested image %s for %s->%s is not an exact top-1 error.",
                    image_selector,
                    true_name,
                    wrong_name,
                )
        selected.append(chosen[1])

    images_out = [candidate[0] for candidate in selected]
    labels_out = [candidate[1] for candidate in selected]
    wrong_targets_out = [candidate[2] for candidate in selected]
    true_names = [str(candidate[3]) for candidate in selected]
    wrong_names = [str(candidate[4]) for candidate in selected]
    image_paths = [str(candidate[5]) for candidate in selected]
    predicted_names = [str(candidate[6]) for candidate in selected]
    confidences = [float(candidate[7]) for candidate in selected]

    for true_name, wrong_name, predicted_name, confidence, image_path in zip(
        true_names,
        wrong_names,
        predicted_names,
        confidences,
        image_paths,
        strict=True,
    ):
        LOGGER.info(
            "Selected target-pair example true=%s wrong=%s actual_pred=%s confidence=%.4f image=%s",
            true_name,
            wrong_name,
            predicted_name,
            confidence,
            image_path,
        )

    return (
        torch.stack(images_out, dim=0).to(device),
        torch.stack(labels_out, dim=0).to(device),
        torch.stack(wrong_targets_out, dim=0).to(device),
        true_names,
        wrong_names,
        image_paths,
    )


def load_legacy_cbm_checkpoint(
    checkpoint_path: Path,
    concept_report_path: Path,
    concept_bank,
    matrix_kind: str,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[ConceptBottleneckModel, dict[str, object]]:
    """Load an old state-only CBM after reconstructing and validating semantics."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Legacy CBM checkpoint must be a dictionary.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Legacy CBM checkpoint is missing model_state_dict.")
    class_weight = state_dict.get("class_head.weight")
    concept_weight = state_dict.get("concept_head.1.weight")
    if not isinstance(class_weight, torch.Tensor) or not isinstance(
        concept_weight, torch.Tensor
    ):
        raise ValueError("Legacy CBM checkpoint has an unsupported architecture.")
    num_classes, num_concepts = map(int, class_weight.shape)
    if int(concept_weight.shape[0]) != num_concepts:
        raise ValueError("Legacy CBM concept-head and class-head dimensions disagree.")
    if num_classes != len(idx_to_class):
        raise ValueError(
            "Legacy CBM class count does not match the current manifest mapping."
        )

    concept_rows = _read_csv(concept_report_path)
    if not concept_rows or "concept" not in concept_rows[0]:
        raise ValueError("Legacy concept report must contain a concept column.")
    concept_names = [row["concept"] for row in concept_rows]
    if len(concept_names) != num_concepts or len(set(concept_names)) != num_concepts:
        raise ValueError(
            "Legacy concept report does not uniquely match the checkpoint concept dimension."
        )
    concept_to_index = {
        str(name): index for index, name in enumerate(concept_bank.concept_names)
    }
    missing = [name for name in concept_names if name not in concept_to_index]
    if missing:
        raise ValueError(f"Legacy concept report contains unknown concepts: {missing}")
    concept_indices = [concept_to_index[name] for name in concept_names]

    model = ConceptBottleneckModel(
        num_classes=num_classes,
        num_concepts=num_concepts,
        pretrained=False,
        trainable_backbone_layers=("layer4",),
        dropout=0.15,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    metadata: dict[str, object] = {
        "concept_names": concept_names,
        "concept_indices": concept_indices,
        "idx_to_class": idx_to_class,
        "matrix_kind": matrix_kind,
    }
    LOGGER.warning(
        "Loaded legacy CBM checkpoint using validated concept order from %s.",
        concept_report_path,
    )
    return model, metadata


def run_cbm_comparator(
    checkpoint_path: Path,
    legacy_concept_report: Path,
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
    manifest_classes = read_manifest_classes(manifest)
    resolved_metadata_root = find_awa2_metadata_root(
        [metadata_root, manifest.parent, PROJECT_ROOT / "data" / "AWA2", PROJECT_ROOT / "data"]
    )
    bank = load_awa2_concepts(resolved_metadata_root, matrix_kind=matrix_kind)
    bank = align_concept_bank_to_manifest(bank, manifest_classes)
    try:
        model, metadata = load_concept_bottleneck_checkpoint(
            checkpoint_path,
            device,
            expected_class_mapping=idx_to_class,
        )
    except ValueError as error:
        if str(error) != "CBM checkpoint is missing model_config metadata.":
            raise
        model, metadata = load_legacy_cbm_checkpoint(
            checkpoint_path=checkpoint_path,
            concept_report_path=legacy_concept_report,
            concept_bank=bank,
            matrix_kind=matrix_kind,
            idx_to_class=idx_to_class,
            device=device,
        )
    concept_names = [str(name) for name in metadata["concept_names"]]
    concept_indices = [int(index) for index in metadata.get("concept_indices", [])]
    if len(concept_indices) != len(concept_names):
        raise ValueError("CBM checkpoint concept indices and names disagree.")

    if str(metadata.get("matrix_kind", matrix_kind)) != matrix_kind:
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

    target_pairs = _parse_target_pairs(args.target_pairs, idx_to_class)
    if target_pairs:
        images, labels, wrong_targets, true_names, wrong_names, image_paths = (
            collect_target_pair_examples(
                model=model,
                loader=loaders["test"],
                device=device,
                target_pairs=target_pairs,
                idx_to_class=idx_to_class,
            )
        )
    else:
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
    if target_pairs:
        with torch.no_grad():
            original_logits = model(images)
            original_probabilities = torch.softmax(original_logits, dim=1)
            original_predictions = original_probabilities.argmax(dim=1)
    else:
        original_predictions = wrong_targets
    if torch.any(wrong_targets == labels):
        raise RuntimeError("Target-pair selection returned a true class as the wrong target.")
    mismatched_fixed_targets = [
        (
            true_names[index],
            wrong_names[index],
            idx_to_class[int(original_predictions[index].item())],
        )
        for index in range(images.size(0))
        if int(original_predictions[index].item()) != int(wrong_targets[index].item())
    ]
    for true_name, wrong_name, actual_prediction in mismatched_fixed_targets:
        LOGGER.warning(
            "Fixed target %s->%s is not the model top-1 for the selected image; actual top-1 is %s.",
            true_name,
            wrong_name,
            actual_prediction,
        )
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
            background_mask=background_mask,
            top_fraction=args.top_fraction,
            actual_names=[
                idx_to_class[int(prediction.item())]
                for prediction in predictions_by_condition["original"]
            ],
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
            legacy_concept_report=args.legacy_concept_report,
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
