"""Run Phase 6 concept-level analysis with AwA2 semantic attributes."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.concepts import (
    align_concept_bank_to_manifest,
    concept_transition_summary,
    find_awa2_metadata_root,
    load_awa2_concepts,
    read_manifest_classes,
    top_concepts_for_class,
)
from src.utils import setup_logging, write_csv

LOGGER = logging.getLogger("run_phase6_concepts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AwA2 predictions through semantic animal attributes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2_subset_background20" / "awa2_manifest_subset.csv",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help=(
            "Folder containing AwA2 classes.txt, predicates.txt, and predicate-matrix files. "
            "If omitted, the script searches near the manifest and data/AWA2."
        ),
    )
    parser.add_argument(
        "--stress-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase5_saliency_metrics.csv",
        help="Optional Phase 4/5 CSV with original_prediction and perturbed_prediction columns.",
    )
    parser.add_argument(
        "--class-profile-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase6_class_concepts.csv",
    )
    parser.add_argument(
        "--transition-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase6_concept_transitions.csv",
    )
    parser.add_argument(
        "--saliency-alignment-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase6_concept_saliency_alignment.csv",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase6_class_concept_heatmap.png",
    )
    parser.add_argument(
        "--transition-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase6_concept_transition_examples.png",
    )
    parser.add_argument(
        "--saliency-alignment-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase6_concept_saliency_alignment.png",
    )
    parser.add_argument(
        "--matrix-kind",
        choices=["continuous", "binary"],
        default="continuous",
    )
    parser.add_argument("--top-concepts", type=int, default=12)
    parser.add_argument("--max-classes-plot", type=int, default=20)
    parser.add_argument("--max-transitions-plot", type=int, default=8)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_class_profile_rows(concept_bank, top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name in concept_bank.class_names:
        top_concepts = top_concepts_for_class(concept_bank, class_name, top_k=top_k)
        rows.append(
            {
                "class_name": class_name,
                "top_concepts": "; ".join(
                    f"{concept}:{value:.3f}" for concept, value in top_concepts
                ),
            }
        )
    return rows


def read_prediction_transitions(csv_path: Path) -> list[tuple[str, str, str]]:
    """Read original/perturbed prediction transitions from Phase 4 or Phase 5 CSV."""
    if not csv_path.exists():
        LOGGER.warning("stress CSV not found; skipping transition analysis: %s", csv_path)
        return []

    transitions: list[tuple[str, str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"original_prediction", "perturbed_prediction", "perturbation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            source = row["original_prediction"]
            target = row["perturbed_prediction"]
            if source == target:
                continue
            transitions.append((source, target, row["perturbation"]))
    return transitions


def build_transition_rows(concept_bank, transitions: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = Counter(transitions)
    for (source, target, perturbation), count in counts.most_common():
        summary = concept_transition_summary(concept_bank, source, target)
        rows.append(
            {
                "count": count,
                "perturbation": perturbation,
                **summary,
            }
        )
    return rows


def build_saliency_alignment_rows(concept_bank, csv_path: Path) -> list[dict[str, object]]:
    """Join Phase 5 saliency drift metrics with AwA2 class-concept distances.

    AwA2 provides class-level attributes, not spatial concept masks. This report
    therefore measures semantic alignment indirectly: when saliency shifts under
    a perturbation, does the predicted class also move toward a conceptually
    different animal?
    """
    if not csv_path.exists():
        LOGGER.warning("stress CSV not found; skipping concept-saliency alignment: %s", csv_path)
        return []

    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "xai_method",
            "perturbation",
            "original_prediction",
            "perturbed_prediction",
            "prediction_changed",
            "spearman",
            "confidence_delta",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        iou_columns = [name for name in (reader.fieldnames or []) if name.startswith("iou_top_")]
        if not iou_columns:
            raise ValueError(f"{csv_path} is missing an iou_top_* column.")
        iou_column = iou_columns[0]

        for row in reader:
            source = row["original_prediction"]
            target = row["perturbed_prediction"]
            summary = concept_transition_summary(concept_bank, source, target)
            mean_abs_delta = float(summary["mean_abs_concept_delta"])
            concept_cosine = float(summary["concept_cosine"])
            saliency_iou = float(row[iou_column])
            spearman = float(row["spearman"])
            saliency_drift = 1.0 - saliency_iou
            confidence_drop = -float(row["confidence_delta"])
            rows.append(
                {
                    "xai_method": row["xai_method"],
                    "perturbation": row["perturbation"],
                    "source_class": source,
                    "target_class": target,
                    "prediction_changed": row["prediction_changed"],
                    "confidence_drop": confidence_drop,
                    "saliency_iou_top_pct": saliency_iou,
                    "saliency_spearman": spearman,
                    "saliency_drift": saliency_drift,
                    "concept_cosine": concept_cosine,
                    "mean_abs_concept_delta": mean_abs_delta,
                    "semantic_shift_score": mean_abs_delta * saliency_drift,
                    "gained_concepts": summary["gained_concepts"],
                    "lost_concepts": summary["lost_concepts"],
                }
            )
    return rows


def save_concept_heatmap(concept_bank, output_path: Path, max_classes: int, top_concepts: int) -> None:
    """Save a compact class-by-concept heatmap for the most variable concepts."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = concept_bank.normalized_matrix()
    variances = matrix.var(axis=0)
    concept_indices = np.argsort(variances)[::-1][:top_concepts]
    class_count = min(max_classes, len(concept_bank.class_names))
    image = matrix[:class_count, concept_indices]

    fig, ax = plt.subplots(figsize=(max(9, top_concepts * 0.55), max(5, class_count * 0.32)))
    im = ax.imshow(image, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("AwA2 concept profiles by class")
    ax.set_xlabel("Semantic concept")
    ax.set_ylabel("Animal class")
    ax.set_xticks(range(len(concept_indices)))
    ax.set_xticklabels(
        [concept_bank.concept_names[index] for index in concept_indices],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(class_count))
    ax.set_yticklabels(concept_bank.class_names[:class_count])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="normalized attribute strength")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def save_transition_plot(rows: list[dict[str, object]], output_path: Path, max_rows: int) -> None:
    """Save concept-distance bars for the most frequent prediction transitions."""
    if not rows:
        LOGGER.warning("No changed prediction transitions to plot.")
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:max_rows]
    labels = [
        f"{row['source_class']} -> {row['target_class']}\n{row['perturbation']}"
        for row in selected
    ]
    values = [float(row["mean_abs_concept_delta"]) for row in selected]
    colors = [float(row["concept_cosine"]) for row in selected]

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.62 * len(selected))))
    bars = ax.barh(labels[::-1], values[::-1], color=plt.cm.magma_r(colors[::-1]))
    ax.set_title("Concept change for prediction flips")
    ax.set_xlabel("Mean absolute concept delta")
    ax.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, selected[::-1]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"cos={float(row['concept_cosine']):.2f}, n={row['count']}",
            va="center",
            fontsize=8.5,
        )
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def save_saliency_alignment_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save a scatter plot of saliency drift versus semantic class shift."""
    if not rows:
        LOGGER.warning("No concept-saliency alignment rows to plot.")
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["xai_method"]) for row in rows})
    markers = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for marker, method in zip(markers, methods):
        method_rows = [row for row in rows if row["xai_method"] == method]
        ax.scatter(
            [float(row["mean_abs_concept_delta"]) for row in method_rows],
            [float(row["saliency_drift"]) for row in method_rows],
            s=58,
            alpha=0.78,
            marker=marker,
            label=method,
        )
    ax.set_title("Concept-saliency alignment under perturbation")
    ax.set_xlabel("Mean absolute concept delta between predictions")
    ax.set_ylabel("Saliency drift (1 - top-k IoU)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    manifest = args.manifest.expanduser().resolve()
    search_roots = [
        args.metadata_root,
        manifest.parent,
        PROJECT_ROOT / "data" / "AWA2",
        PROJECT_ROOT / "data",
    ]
    metadata_root = find_awa2_metadata_root([path for path in search_roots if path is not None])
    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("metadata_root=%s", metadata_root)

    manifest_classes = read_manifest_classes(manifest)
    concept_bank = load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind)
    concept_bank = align_concept_bank_to_manifest(concept_bank, manifest_classes)

    class_rows = build_class_profile_rows(concept_bank, top_k=args.top_concepts)
    write_csv(class_rows, args.class_profile_output)
    save_concept_heatmap(
        concept_bank,
        args.heatmap_output,
        max_classes=args.max_classes_plot,
        top_concepts=args.top_concepts,
    )

    transitions = read_prediction_transitions(args.stress_csv.expanduser().resolve())
    transition_rows = build_transition_rows(concept_bank, transitions)
    if transition_rows:
        write_csv(transition_rows, args.transition_output)
        save_transition_plot(
            transition_rows,
            args.transition_figure_output,
            max_rows=args.max_transitions_plot,
        )
    else:
        LOGGER.warning("No changed prediction transitions found; wrote only class profiles.")

    alignment_rows = build_saliency_alignment_rows(concept_bank, args.stress_csv.expanduser().resolve())
    if alignment_rows:
        write_csv(alignment_rows, args.saliency_alignment_output)
        save_saliency_alignment_plot(alignment_rows, args.saliency_alignment_figure_output)
    else:
        LOGGER.warning("Concept-saliency alignment not written.")

    LOGGER.info(
        "Phase 6 complete: class_profiles=%s transitions=%s alignment=%s",
        args.class_profile_output,
        args.transition_output if transition_rows else "not written",
        args.saliency_alignment_output if alignment_rows else "not written",
    )


if __name__ == "__main__":
    main()
