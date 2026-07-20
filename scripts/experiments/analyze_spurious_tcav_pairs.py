"""Audit TCAV concept-class pairs against AwA2 semantic priors."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    normalize_class_name,
    read_manifest_classes,
)

LOGGER = logging.getLogger("analyze_spurious_tcav_pairs")

DEFAULT_INVESTIGATION_PAIRS = (
    "flippers:polar+bear",
    "horns:dolphin",
    "hooves:dolphin",
    "flippers:tiger",
    "flippers:giant+panda",
    "hooves:antelope",
)

CONTEXT_HINTS = {
    "arctic",
    "coastal",
    "water",
    "ocean",
    "fish",
    "swims",
    "plankton",
    "skimmer",
    "newworld",
    "oldworld",
    "forest",
    "jungle",
    "mountains",
    "fields",
    "desert",
    "ground",
    "tree",
    "cave",
    "bush",
}


class CsvDialectError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TCAV concept sensitivity with AwA2 class-level concept priors "
            "to surface biologically implausible concept-species associations."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "AWA2_subset_background20"
        / "awa2_manifest_subset.csv",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Folder containing AwA2 metadata. If omitted, common project locations are searched.",
    )
    parser.add_argument(
        "--tcav-scores",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_scores_notebook.csv",
    )
    parser.add_argument(
        "--cav-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_cav_summary_notebook.csv",
        help="CAV summary CSV used to recover the positive classes behind each concept direction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_spurious_tcav_pairs.csv",
    )
    parser.add_argument(
        "--bridge-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_bridge_concepts.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_spurious_tcav_pairs.png",
    )
    parser.add_argument(
        "--bridge-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_tcav_bridge_concepts.png",
    )
    parser.add_argument("--matrix-kind", choices=["continuous", "binary"], default="continuous")
    parser.add_argument("--tcav-threshold", type=float, default=0.70)
    parser.add_argument("--oracle-low-threshold", type=float, default=0.25)
    parser.add_argument("--effect-threshold", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument("--bridge-top-k", type=int, default=8)
    parser.add_argument(
        "--investigate-pair",
        action="append",
        default=None,
        metavar="CONCEPT:CLASS",
        help="Concept-class pair to include in the bridge-concept audit. Can be passed multiple times.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def optional_bool(row: dict[str, str], key: str) -> bool | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def suspicion_score(
    tcav_score: float,
    oracle_value: float,
    random_tcav: float | None,
    effect_size: float | None,
    significant: bool | None,
) -> float:
    baseline_gap = effect_size
    if baseline_gap is None and random_tcav is not None:
        baseline_gap = tcav_score - random_tcav
    if baseline_gap is None:
        baseline_gap = tcav_score - 0.5

    significance_bonus = 0.15 if significant is True else 0.0
    oracle_penalty = max(0.0, 0.25 - oracle_value)
    return float(baseline_gap + max(0.0, tcav_score - oracle_value) + oracle_penalty + significance_bonus)


def concept_indices(concept_bank) -> dict[str, int]:
    return {normalize_class_name(name): index for index, name in enumerate(concept_bank.concept_names)}


def build_audit_rows(
    tcav_rows: list[dict[str, str]],
    concept_bank,
    tcav_threshold: float,
    oracle_low_threshold: float,
    effect_threshold: float,
) -> list[dict[str, object]]:
    class_to_index = concept_bank.class_to_index()
    concept_to_index = concept_indices(concept_bank)
    matrix = concept_bank.normalized_matrix()

    rows: list[dict[str, object]] = []
    for row in tcav_rows:
        concept = str(row["concept"])
        target_class = str(row["target_class"])
        class_key = normalize_class_name(target_class)
        concept_key = normalize_class_name(concept)
        if class_key not in class_to_index or concept_key not in concept_to_index:
            LOGGER.warning("Skipping unknown pair concept=%s target=%s", concept, target_class)
            continue

        tcav_score = float(row["tcav_score"])
        oracle_value = float(matrix[class_to_index[class_key], concept_to_index[concept_key]])
        random_tcav = optional_float(row, "random_tcav_mean")
        if random_tcav is None:
            random_tcav = optional_float(row, "random_tcav_score")
        effect_size = optional_float(row, "effect_size")
        significant = optional_bool(row, "significant")

        positive_effect = True if effect_size is None else effect_size >= effect_threshold
        candidate = (
            tcav_score >= tcav_threshold
            and oracle_value <= oracle_low_threshold
            and positive_effect
        )
        if candidate and significant is True:
            status = "high_risk_validated"
        elif candidate:
            status = "candidate"
        elif tcav_score >= tcav_threshold and oracle_value <= oracle_low_threshold:
            status = "high_tcav_low_oracle"
        else:
            status = "background"

        rows.append(
            {
                "concept": concept,
                "target_class": target_class,
                "tcav_score": round(tcav_score, 4),
                "random_tcav": "" if random_tcav is None else round(random_tcav, 4),
                "effect_size": "" if effect_size is None else round(effect_size, 4),
                "significant": "" if significant is None else significant,
                "awa2_oracle_concept_value": round(oracle_value, 4),
                "tcav_minus_oracle": round(tcav_score - oracle_value, 4),
                "suspicion_score": round(
                    suspicion_score(
                        tcav_score,
                        oracle_value,
                        random_tcav,
                        effect_size,
                        significant,
                    ),
                    4,
                ),
                "audit_status": status,
            }
        )

    return sorted(
        rows,
        key=lambda item: (
            str(item["audit_status"]) == "background",
            -float(item["suspicion_score"]),
        ),
    )


def parse_class_counts(summary: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in summary.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise CsvDialectError(f"Class-count entry lacks ':': {chunk!r}")
        name, count = chunk.rsplit(":", 1)
        counts[name.strip()] = int(count)
    return counts


def concept_positive_classes_from_summary(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    mapping: dict[str, dict[str, int]] = {}
    for row in rows:
        mapping[normalize_class_name(row["concept"])] = parse_class_counts(row["positive_classes"])
    return mapping


def infer_positive_classes(concept_bank, concept_name: str, threshold: float) -> dict[str, int]:
    concept_to_index = concept_indices(concept_bank)
    concept_key = normalize_class_name(concept_name)
    matrix = concept_bank.normalized_matrix()
    if concept_key not in concept_to_index:
        raise ValueError(f"Unknown concept {concept_name!r}")
    column = matrix[:, concept_to_index[concept_key]]
    return {
        class_name: 1
        for class_name, value in zip(concept_bank.class_names, column)
        if value >= threshold
    }


def weighted_average_profile(concept_bank, class_counts: dict[str, int]) -> np.ndarray:
    class_to_index = concept_bank.class_to_index()
    matrix = concept_bank.normalized_matrix()
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for class_name, count in class_counts.items():
        key = normalize_class_name(class_name)
        if key not in class_to_index:
            LOGGER.warning("Ignoring unknown CAV source class %s", class_name)
            continue
        vectors.append(matrix[class_to_index[key]])
        weights.append(float(count))
    if not vectors:
        raise ValueError("No valid source classes available for bridge audit.")
    return np.average(np.vstack(vectors), axis=0, weights=np.array(weights))


def parse_pair(pair: str) -> tuple[str, str]:
    if ":" not in pair:
        raise ValueError(f"Investigation pair must look like CONCEPT:CLASS, got {pair!r}")
    concept, target = pair.split(":", 1)
    return concept.strip(), target.strip()


def source_classes_text(class_counts: dict[str, int]) -> str:
    return "; ".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))


def bridge_hypothesis(concept_name: str, bridge_names: list[str]) -> str:
    context = [name for name in bridge_names if normalize_class_name(name) in CONTEXT_HINTS]
    if context:
        return "context_or_habitat_entanglement: " + ", ".join(context[:4])
    if bridge_names:
        return "shared_non_target_attributes: " + ", ".join(bridge_names[:4])
    return "no_clear_bridge"


def build_bridge_rows(
    audit_rows: list[dict[str, object]],
    concept_bank,
    cav_positive_classes: dict[str, dict[str, int]],
    requested_pairs: list[str],
    positive_threshold: float,
    top_k: int,
) -> list[dict[str, object]]:
    class_to_index = concept_bank.class_to_index()
    concept_to_index = concept_indices(concept_bank)
    matrix = concept_bank.normalized_matrix()
    audit_by_pair = {
        (normalize_class_name(str(row["concept"])), normalize_class_name(str(row["target_class"]))): row
        for row in audit_rows
    }

    rows: list[dict[str, object]] = []
    for raw_pair in requested_pairs:
        concept_name, target_class = parse_pair(raw_pair)
        concept_key = normalize_class_name(concept_name)
        target_key = normalize_class_name(target_class)
        if target_key not in class_to_index:
            LOGGER.warning("Skipping bridge audit for unknown target class %s", target_class)
            continue
        if concept_key not in concept_to_index:
            LOGGER.warning("Skipping bridge audit for unknown concept %s", concept_name)
            continue

        source_counts = cav_positive_classes.get(concept_key)
        if not source_counts:
            source_counts = infer_positive_classes(concept_bank, concept_name, positive_threshold)
        source_profile = weighted_average_profile(concept_bank, source_counts)
        target_profile = matrix[class_to_index[target_key]]
        bridge = np.minimum(source_profile, target_profile)
        concept_col = concept_to_index[concept_key]
        bridge[concept_col] = -np.inf
        top_indices = [int(index) for index in np.argsort(bridge)[::-1] if np.isfinite(bridge[index])][:top_k]
        bridge_names = [concept_bank.concept_names[index] for index in top_indices]
        audit = audit_by_pair.get((concept_key, target_key), {})
        rows.append(
            {
                "concept": concept_name,
                "target_class": target_class,
                "tcav_score": audit.get("tcav_score", ""),
                "awa2_oracle_concept_value": audit.get("awa2_oracle_concept_value", ""),
                "suspicion_score": audit.get("suspicion_score", ""),
                "source_positive_classes": source_classes_text(source_counts),
                "bridge_concepts": "; ".join(
                    f"{concept_bank.concept_names[index]}:{bridge[index]:.3f}"
                    for index in top_indices
                ),
                "target_values": "; ".join(
                    f"{concept_bank.concept_names[index]}:{target_profile[index]:.3f}"
                    for index in top_indices
                ),
                "source_average_values": "; ".join(
                    f"{concept_bank.concept_names[index]}:{source_profile[index]:.3f}"
                    for index in top_indices
                ),
                "bridge_hypothesis": bridge_hypothesis(concept_name, bridge_names),
            }
        )
    return rows


def save_audit_figure(rows: list[dict[str, object]], output_path: Path, top_n: int) -> None:
    selected = rows[:top_n]

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{row['concept']} -> {row['target_class']}" for row in selected]
    scores = [float(row["tcav_score"]) for row in selected]
    oracle_values = [float(row["awa2_oracle_concept_value"]) for row in selected]
    y_positions = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(selected))))
    bars = ax.barh(y_positions, scores, color="#0f766e", alpha=0.86, label="TCAV score")
    ax.scatter(oracle_values, y_positions, color="#9a5a2b", s=34, zorder=3, label="AwA2 oracle value")
    ax.axvline(0.70, color="#314047", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title("TCAV sensitivity versus AwA2 concept prior")
    ax.legend(loc="lower right")
    for bar, row in zip(bars, selected):
        ax.text(
            min(1.0, bar.get_width() + 0.015),
            bar.get_y() + bar.get_height() / 2,
            str(row["audit_status"]),
            va="center",
            fontsize=8,
            color="#314047",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_bridge_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_labels = [f"{row['concept']} -> {row['target_class']}" for row in rows]
    bridge_names: list[str] = []
    parsed_rows: list[dict[str, float]] = []
    for row in rows:
        parsed: dict[str, float] = {}
        for chunk in str(row["bridge_concepts"]).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, value = chunk.rsplit(":", 1)
            parsed[name] = float(value)
            if name not in bridge_names:
                bridge_names.append(name)
        parsed_rows.append(parsed)

    values = np.zeros((len(rows), len(bridge_names)), dtype=float)
    for row_index, parsed in enumerate(parsed_rows):
        for col_index, name in enumerate(bridge_names):
            values[row_index, col_index] = parsed.get(name, 0.0)

    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.42 * len(bridge_names)), max(4.8, 0.62 * len(rows)))
    )
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Shared attributes that can contaminate suspicious CAV directions")
    ax.set_xlabel("Potential bridge concept")
    ax.set_ylabel("TCAV pair under audit")
    ax.set_xticks(range(len(bridge_names)))
    ax.set_xticklabels(bridge_names, rotation=35, ha="right")
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if value >= 0.35:
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.62 else "#263238",
                    fontsize=7,
                )
    fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02, label="min(source avg, target)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


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
    metadata_root = find_awa2_metadata_root(
        [path for path in search_roots if path is not None]
    )
    concept_bank = align_concept_bank_to_manifest(
        load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind),
        read_manifest_classes(manifest),
    )
    rows = build_audit_rows(
        read_csv_rows(args.tcav_scores),
        concept_bank,
        tcav_threshold=args.tcav_threshold,
        oracle_low_threshold=args.oracle_low_threshold,
        effect_threshold=args.effect_threshold,
    )
    pairs = args.investigate_pair or list(DEFAULT_INVESTIGATION_PAIRS)
    bridge_rows = build_bridge_rows(
        rows,
        concept_bank,
        concept_positive_classes_from_summary(args.cav_summary),
        pairs,
        positive_threshold=args.tcav_threshold,
        top_k=args.bridge_top_k,
    )
    write_csv(rows, args.output)
    write_csv(bridge_rows, args.bridge_output)
    save_audit_figure(rows, args.figure_output, args.top_n)
    save_bridge_figure(bridge_rows, args.bridge_figure_output)
    LOGGER.info("Saved %d audited TCAV pairs to %s", len(rows), args.output)
    LOGGER.info("Saved %d bridge rows to %s", len(bridge_rows), args.bridge_output)
    LOGGER.info("Saved audit figure to %s", args.figure_output)
    LOGGER.info("Saved bridge figure to %s", args.bridge_figure_output)


if __name__ == "__main__":
    main()
