"""Decompose one wrong Concept Bottleneck Model prediction end to end."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from pathlib import Path

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

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")

from src.bottleneck import ConceptBottleneckModel, load_concept_bottleneck_checkpoint
from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    read_manifest_classes,
)
from src.data import build_resnet_transforms, load_idx_to_class
from src.misclassification import (
    concept_evidence_rows,
    save_cbm_error_decomposition_figure,
)
from src.model import get_device
from src.utils import setup_logging, write_csv
from src.validation import device_spec, log_level, nonnegative_int, positive_int

LOGGER = logging.getLogger("analyze_cbm_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one wrong CBM prediction and connect concept errors, exact "
            "linear-head contributions, and one-concept oracle corrections."
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
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "phase8_cbm_notebook.pt",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_cbm_error_analysis.csv",
        help="CBM error CSV used to select a case when --filepath is omitted.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2",
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
            "Ordered concept report used only when loading a legacy checkpoint "
            "without embedded semantic metadata."
        ),
    )
    parser.add_argument("--matrix-kind", choices=("continuous", "binary"), default="continuous")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--filepath",
        type=Path,
        default=None,
        help="Analyze this manifest image instead of selecting from the error report.",
    )
    parser.add_argument("--true-class", default=None, help="Optional error-report filter.")
    parser.add_argument("--predicted-class", default=None, help="Optional error-report filter.")
    parser.add_argument(
        "--contrast-class",
        default=None,
        help=(
            "Decompose this fixed class against the true class even when it is "
            "not the CBM top-1 prediction (for cross-model error comparisons)."
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=(
            "report_order",
            "cbm_confidence",
            "mean_abs_concept_error",
            "max_abs_concept_error",
        ),
        default="report_order",
    )
    parser.add_argument("--case-index", type=nonnegative_int, default=0)
    parser.add_argument("--top-concepts", type=positive_int, default=8)
    parser.add_argument(
        "--table-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_cbm_error_decomposition.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_cbm_error_decomposition_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase8_cbm_error_decomposition.png",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_false(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "no"}


def _manifest_samples(manifest: Path, split: str) -> list[dict[str, object]]:
    rows = _read_csv(manifest)
    required = {"filepath", "label", "class_name", "split"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Manifest is missing columns: {sorted(required.difference(rows[0]))}")

    samples: list[dict[str, object]] = []
    for row in rows:
        if row["split"] != split:
            continue
        filepath = Path(row["filepath"]).expanduser()
        if not filepath.is_absolute():
            filepath = manifest.parent / filepath
        samples.append(
            {
                "filepath": filepath.resolve(),
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "split": row["split"],
            }
        )
    if not samples:
        raise ValueError(f"No manifest samples found for split={split!r}.")
    return samples


def _match_manifest_sample(
    value: str | Path,
    samples: list[dict[str, object]],
    expected_class: str | None = None,
) -> dict[str, object]:
    reported = Path(value).expanduser()
    resolved = reported.resolve()
    exact = [sample for sample in samples if Path(sample["filepath"]) == resolved]
    if len(exact) == 1:
        return exact[0]

    by_name = [
        sample
        for sample in samples
        if Path(sample["filepath"]).name == reported.name
        and (expected_class is None or str(sample["class_name"]) == expected_class)
    ]
    if len(by_name) == 1:
        return by_name[0]
    if not by_name:
        raise FileNotFoundError(f"Image is not present in the selected manifest split: {value}")
    raise ValueError(f"Image name is ambiguous in the selected manifest split: {value}")


def select_error_case(
    error_rows: list[dict[str, str]],
    samples: list[dict[str, object]],
    true_class: str | None,
    predicted_class: str | None,
    rank_by: str,
    case_index: int,
) -> tuple[dict[str, object], dict[str, str]]:
    required = {"filepath", "true_class", "cbm_predicted_class", "cbm_correct"}
    if error_rows and not required.issubset(error_rows[0]):
        raise ValueError(f"Error report is missing columns: {sorted(required.difference(error_rows[0]))}")
    candidates = [row for row in error_rows if _is_false(row["cbm_correct"])]
    if true_class is not None:
        candidates = [row for row in candidates if row["true_class"] == true_class]
    if predicted_class is not None:
        candidates = [
            row for row in candidates if row["cbm_predicted_class"] == predicted_class
        ]
    if rank_by != "report_order":
        candidates.sort(key=lambda row: float(row[rank_by]), reverse=True)
    if case_index >= len(candidates):
        raise IndexError(
            f"case-index {case_index} is out of range for {len(candidates)} matching errors."
        )
    report_row = candidates[case_index]
    sample = _match_manifest_sample(
        report_row["filepath"],
        samples,
        expected_class=report_row["true_class"],
    )
    return sample, report_row


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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    manifest = args.manifest.expanduser().resolve()
    samples = _manifest_samples(manifest, args.split)

    report_row: dict[str, str] | None = None
    if args.filepath is not None:
        sample = _match_manifest_sample(args.filepath, samples, expected_class=args.true_class)
    else:
        sample, report_row = select_error_case(
            _read_csv(args.error_report),
            samples,
            true_class=args.true_class,
            predicted_class=args.predicted_class,
            rank_by=args.rank_by,
            case_index=args.case_index,
        )

    image_path = Path(sample["filepath"])
    true_target = int(sample["label"])
    true_class = str(sample["class_name"])
    idx_to_class = load_idx_to_class(manifest)
    device = get_device(args.device)
    metadata_root = find_awa2_metadata_root(
        [args.metadata_root, manifest.parent, PROJECT_ROOT / "data" / "AWA2", PROJECT_ROOT / "data"]
    )
    checkpoint_format = "metadata_embedded"
    concept_bank = None
    try:
        model, metadata = load_concept_bottleneck_checkpoint(
            args.checkpoint,
            device,
            expected_class_mapping=idx_to_class,
        )
    except ValueError as error:
        if str(error) != "CBM checkpoint is missing model_config metadata.":
            raise
        checkpoint_format = "legacy_semantics_reconstructed_from_concept_report"
        concept_bank = load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind)
        concept_bank = align_concept_bank_to_manifest(
            concept_bank,
            read_manifest_classes(manifest),
        )
        model, metadata = load_legacy_cbm_checkpoint(
            checkpoint_path=args.checkpoint,
            concept_report_path=args.legacy_concept_report,
            concept_bank=concept_bank,
            matrix_kind=args.matrix_kind,
            idx_to_class=idx_to_class,
            device=device,
        )

    concept_names = [str(name) for name in metadata["concept_names"]]
    concept_indices = [int(index) for index in metadata.get("concept_indices", [])]
    if len(concept_indices) != len(concept_names):
        raise ValueError("CBM checkpoint concept indices and names disagree.")
    if concept_bank is None:
        concept_bank = load_awa2_concepts(
            metadata_root,
            matrix_kind=str(metadata.get("matrix_kind", args.matrix_kind)),
        )
        concept_bank = align_concept_bank_to_manifest(
            concept_bank,
            read_manifest_classes(manifest),
        )
    prototypes = torch.tensor(
        concept_bank.normalized_matrix()[:, concept_indices],
        dtype=torch.float32,
        device=device,
    )

    image = build_resnet_transforms(train=False)(Image.open(image_path).convert("RGB"))
    image_batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_batch)
        probabilities = torch.softmax(outputs.class_logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    predicted_target = int(prediction[0].item())
    if predicted_target == true_target and args.contrast_class is None:
        raise ValueError(
            f"Selected image is currently classified correctly as {true_class}; "
            "choose a wrong prediction or a matching checkpoint."
        )
    predicted_class = idx_to_class[predicted_target]
    if report_row is not None and report_row["cbm_predicted_class"] != predicted_class:
        LOGGER.warning(
            "Report predicted %s but checkpoint currently predicts %s; using checkpoint output.",
            report_row["cbm_predicted_class"],
            predicted_class,
        )

    class_to_idx = {class_name: index for index, class_name in idx_to_class.items()}
    if args.contrast_class is None:
        contrast_target = predicted_target
    else:
        if args.contrast_class not in class_to_idx:
            raise ValueError(f"Unknown contrast class: {args.contrast_class!r}")
        contrast_target = class_to_idx[args.contrast_class]
        if contrast_target == true_target:
            raise ValueError("contrast-class must differ from the true class.")
    contrast_class = idx_to_class[contrast_target]

    true_prototype = prototypes[true_target : true_target + 1]
    contrast_prototype = prototypes[contrast_target : contrast_target + 1]
    evidence_rows = concept_evidence_rows(
        class_head=model.class_head,
        concept_probabilities=outputs.concept_probs.detach(),
        true_prototypes=true_prototype,
        wrong_prototypes=contrast_prototype,
        true_targets=torch.tensor([true_target], device=device),
        wrong_targets=torch.tensor([contrast_target], device=device),
        concept_names=concept_names,
    )

    for row in evidence_rows:
        row.update(
            {
                "filepath": str(image_path),
                "true_class": true_class,
                "cbm_predicted_class": predicted_class,
                "contrast_class": contrast_class,
                "checkpoint_format": checkpoint_format,
                "concept_prediction_delta": (
                    float(row["predicted_value"]) - float(row["true_prototype_value"])
                ),
                "corrected_predicted_class": idx_to_class[int(row["corrected_prediction_index"])],
                "corrected_is_true_class": (
                    int(row["corrected_prediction_index"]) == true_target
                ),
                "correction_recovers_true_class": (
                    predicted_target != true_target
                    and int(row["corrected_prediction_index"]) == true_target
                ),
            }
        )

    with torch.no_grad():
        oracle_logits = model.classify_concepts(true_prototype)
        oracle_probabilities = torch.softmax(oracle_logits, dim=1)
        oracle_confidence, oracle_prediction = oracle_probabilities.max(dim=1)
    oracle_prediction_index = int(oracle_prediction[0].item())
    contribution_sum = float(
        np.sum([float(row["wrong_vs_true_margin_contribution"]) for row in evidence_rows])
    )
    best_correction = max(
        evidence_rows,
        key=lambda row: float(row["correction_true_probability_delta"]),
    )
    recovering_corrections = [
        row for row in evidence_rows if bool(row["correction_recovers_true_class"])
    ]
    best_recovering_correction = (
        max(
            recovering_corrections,
            key=lambda row: float(row["correction_true_probability_delta"]),
        )
        if recovering_corrections
        else None
    )
    first = evidence_rows[0]
    oracle_correct = oracle_prediction_index == true_target
    summary: dict[str, object] = {
        "filepath": str(image_path),
        "true_class": true_class,
        "cbm_predicted_class": predicted_class,
        "contrast_class": contrast_class,
        "checkpoint_format": checkpoint_format,
        "cbm_confidence": float(confidence[0].item()),
        "cbm_true_probability": float(probabilities[0, true_target].item()),
        "cbm_predicted_probability": float(probabilities[0, predicted_target].item()),
        "cbm_contrast_probability": float(probabilities[0, contrast_target].item()),
        "wrong_vs_true_margin": float(first["original_cbm_margin"]),
        "wrong_vs_true_bias": float(first["wrong_vs_true_bias"]),
        "concept_contribution_sum": contribution_sum,
        "reconstructed_margin": float(first["reconstructed_cbm_margin"]),
        "margin_reconstruction_error": float(first["margin_reconstruction_error"]),
        "mean_abs_concept_error": float(
            np.mean([float(row["distance_to_true_prototype"]) for row in evidence_rows])
        ),
        "max_abs_concept_error": float(
            np.max([float(row["distance_to_true_prototype"]) for row in evidence_rows])
        ),
        "oracle_head_predicted_class": idx_to_class[oracle_prediction_index],
        "oracle_head_confidence": float(oracle_confidence[0].item()),
        "oracle_head_true_probability": float(oracle_probabilities[0, true_target].item()),
        "oracle_head_correct": oracle_correct,
        "diagnosis": (
            "cbm_top1_correct_fixed_contrast_analysis"
            if predicted_target == true_target
            else "concept_prediction_error_sufficient_under_oracle_intervention"
            if oracle_correct
            else "concept_error_plus_oracle_intervention_head_failure"
        ),
        "oracle_intervention_scope": (
            "AwA2 class-level profile stress test; the class head was trained on "
            "predicted concept vectors, so this is not a guaranteed in-distribution upper bound"
        ),
        "best_single_correction": str(best_correction["concept"]),
        "best_true_probability_delta": float(
            best_correction["correction_true_probability_delta"]
        ),
        "best_correction_predicted_class": str(
            best_correction["corrected_predicted_class"]
        ),
        "best_correction_recovers_true_class": bool(
            best_correction["correction_recovers_true_class"]
        ),
        "recovering_single_corrections": len(recovering_corrections),
        "best_recovering_correction": (
            str(best_recovering_correction["concept"])
            if best_recovering_correction is not None
            else ""
        ),
        "best_recovering_true_probability_delta": (
            float(best_recovering_correction["correction_true_probability_delta"])
            if best_recovering_correction is not None
            else ""
        ),
    }

    write_csv(evidence_rows, args.table_output)
    write_csv([summary], args.summary_output)
    save_cbm_error_decomposition_figure(
        image=image.detach().cpu(),
        rows=evidence_rows,
        summary=summary,
        output_path=args.figure_output,
        top_k=args.top_concepts,
    )
    LOGGER.info(
        "CBM case decomposed: true=%s top1=%s contrast=%s | diagnosis=%s | table=%s | figure=%s",
        true_class,
        predicted_class,
        contrast_class,
        summary["diagnosis"],
        args.table_output,
        args.figure_output,
    )


if __name__ == "__main__":
    main()
