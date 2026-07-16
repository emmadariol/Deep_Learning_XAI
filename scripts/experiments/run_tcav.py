"""Run repeated, validated TCAV analysis with random-concept controls."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.concepts import (
    align_concept_bank_to_manifest,
    concept_coverage_rows,
    concept_group,
    find_awa2_metadata_root,
    load_awa2_concepts,
    normalize_class_name,
    read_manifest_classes,
)
from src.data import ImageManifestDataset, build_resnet_transforms, infer_num_classes
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.tcav import (
    ConceptSampleSelection,
    aggregate_tcav_run_rows,
    build_subset_loader,
    extract_activation_cache,
    extract_pooled_gradients,
    find_concept_index,
    score_cav_from_gradients,
    select_class_sample_indices,
    select_concept_sample_indices,
    select_random_control_indices,
    split_concept_selection,
    train_cav,
)
from src.utils import set_seed, setup_logging, write_csv
from src.validation import (
    device_spec,
    log_level,
    nonnegative_int,
    open_unit_float,
    positive_float,
    positive_int,
)

LOGGER = logging.getLogger("run_tcav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validated repeated TCAV with target-class exclusion, "
            "random CAV controls, variability and corrected significance."
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
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["stripes", "furry", "hooves", "horns", "flippers"],
    )
    parser.add_argument("--target-classes", nargs="*", default=None)
    parser.add_argument("--layer", type=str, default="layer3")
    parser.add_argument("--pool", choices=["avg", "flatten"], default="avg")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=0.75)
    parser.add_argument("--negative-threshold", type=float, default=0.25)
    parser.add_argument("--max-concept-examples", type=positive_int, default=200)
    parser.add_argument("--min-concept-examples", type=positive_int, default=8)
    parser.add_argument("--max-eval-per-class", type=positive_int, default=40)
    parser.add_argument("--max-target-classes", type=positive_int, default=10)
    parser.add_argument("--cav-epochs", type=positive_int, default=200)
    parser.add_argument("--cav-lr", type=positive_float, default=1e-2)
    parser.add_argument("--cav-weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-cav-runs", type=positive_int, default=20)
    parser.add_argument("--min-valid-runs", type=positive_int, default=5)
    parser.add_argument(
        "--cav-validation-fraction",
        type=open_unit_float,
        default=0.25,
    )
    parser.add_argument(
        "--minimum-cav-validation-accuracy",
        type=open_unit_float,
        default=0.55,
    )
    parser.add_argument(
        "--exclude-target-class",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefer-class-disjoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--multiple-testing",
        choices=["benjamini_hochberg", "bonferroni"],
        default="benjamini_hochberg",
    )
    parser.add_argument("--significance-alpha", type=open_unit_float, default=0.05)
    parser.add_argument("--max-permutations", type=positive_int, default=10000)
    parser.add_argument(
        "--matrix-kind",
        choices=["continuous", "binary"],
        default="continuous",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_scores.csv",
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_runs.csv",
    )
    parser.add_argument(
        "--cav-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_cav_summary.csv",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_concept_coverage.csv",
    )
    parser.add_argument(
        "--cav-artifact-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_cav_vectors.npz",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_tcav_heatmap.png",
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_tcav_top_scores.png",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def class_name_to_label(manifest_classes: list[str]) -> dict[str, int]:
    return {
        normalize_class_name(name): index
        for index, name in enumerate(manifest_classes)
    }


def choose_target_classes(
    concept_bank,
    concept_names: list[str],
    max_classes: int,
) -> list[str]:
    """Pick target classes across requested concepts, not only the first concept."""
    matrix = concept_bank.normalized_matrix()
    selected: list[str] = []
    rankings: list[list[str]] = []
    for concept_name in concept_names:
        try:
            concept_index = find_concept_index(concept_bank, concept_name)
        except ValueError as error:
            LOGGER.warning("Skipping target ranking for %s: %s", concept_name, error)
            continue
        rankings.append(
            [
                concept_bank.class_names[int(index)]
                for index in np.argsort(matrix[:, concept_index])[::-1]
            ]
        )
    for rank in range(len(concept_bank.class_names)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            class_name = ranking[rank]
            if class_name not in selected:
                selected.append(class_name)
            if len(selected) >= max_classes:
                return selected
    return selected or concept_bank.class_names[:max_classes]


def deterministic_dataset(manifest: Path, split: str) -> ImageManifestDataset:
    """Use evaluation preprocessing for concept activations and CAV validation."""
    return ImageManifestDataset(
        manifest_path=manifest,
        split=split,
        transform=build_resnet_transforms(train=False),
    )


def _take(cache: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        raise ValueError("Cannot take an empty activation subset.")
    return cache[torch.tensor(indices, dtype=torch.long)]


def _random_selection(
    dataset: Dataset,
    positive_indices: list[int],
    negative_indices: list[int],
    concept_name: str,
) -> ConceptSampleSelection:
    return ConceptSampleSelection(
        concept_name=f"random_control_for_{concept_name}",
        concept_index=-1,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        positive_threshold=float("nan"),
        negative_threshold=float("nan"),
        positive_classes="random group A",
        negative_classes="random group B",
    )


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def save_cav_artifact(
    vectors: dict[str, np.ndarray],
    metadata: list[dict[str, object]],
    output_path: Path,
    shared_metadata: dict[str, object],
) -> None:
    """Store numerical vectors in NPZ and semantics in adjacent trusted JSON."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **vectors)
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"schema": "validated-tcav-cav-bank-v1", **shared_metadata, "vectors": metadata},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    LOGGER.info("Saved CAV vectors: %s", output_path)
    LOGGER.info("Saved CAV metadata: %s", metadata_path)


def concept_summary_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["concept"])].append(row)
    rows: list[dict[str, object]] = []
    for concept, values in sorted(grouped.items()):
        rows.append(
            {
                "concept": concept,
                "concept_group": values[0]["concept_group"],
                "target_pairs": len({str(row["target_class"]) for row in values}),
                "valid_runs": len(values),
                "mean_train_accuracy": float(
                    np.mean([float(row["cav_train_accuracy"]) for row in values])
                ),
                "mean_validation_accuracy": float(
                    np.mean([float(row["cav_validation_accuracy"]) for row in values])
                ),
                "std_validation_accuracy": float(
                    np.std(
                        [float(row["cav_validation_accuracy"]) for row in values],
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0,
                "mean_random_validation_accuracy": float(
                    np.mean(
                        [
                            float(row["random_cav_validation_accuracy"])
                            for row in values
                        ]
                    )
                ),
                "class_disjoint_run_fraction": float(
                    np.mean(
                        [
                            str(row["split_strategy"]) == "class_disjoint"
                            for row in values
                        ]
                    )
                ),
            }
        )
    return rows


def save_tcav_heatmap(
    rows: list[dict[str, object]],
    concepts: list[str],
    target_classes: list[str],
    output_path: Path,
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((len(target_classes), len(concepts)), np.nan)
    significant = np.zeros_like(values, dtype=bool)
    class_to_row = {
        normalize_class_name(name): index for index, name in enumerate(target_classes)
    }
    concept_to_col = {
        normalize_class_name(name): index for index, name in enumerate(concepts)
    }
    for row in rows:
        class_key = normalize_class_name(str(row["target_class"]))
        concept_key = normalize_class_name(str(row["concept"]))
        if class_key in class_to_row and concept_key in concept_to_col:
            row_index = class_to_row[class_key]
            col_index = concept_to_col[concept_key]
            values[row_index, col_index] = float(row["tcav_score"])
            significant[row_index, col_index] = bool(row["significant"])

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.35 * len(concepts)), max(5.0, 0.5 * len(target_classes)))
    )
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Validated TCAV mean scores (* corrected significant vs random)")
    ax.set_xlabel("Concept")
    ax.set_ylabel("Target class")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=35, ha="right")
    ax.set_yticks(range(len(target_classes)))
    ax.set_yticklabels(target_classes)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isfinite(value):
                marker = "*" if significant[row_index, col_index] else ""
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}{marker}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="mean TCAV score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_tcav_barplot(
    rows: list[dict[str, object]],
    output_path: Path,
    top_n: int = 18,
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(
        rows,
        key=lambda row: float(row["effect_size"]),
        reverse=True,
    )[:top_n]
    labels = [f"{row['target_class']} | {row['concept']}" for row in selected]
    effects = [float(row["effect_size"]) for row in selected]
    errors = [float(row["tcav_std"]) for row in selected]
    colors = ["#0f766e" if bool(row["significant"]) else "#94a3b8" for row in selected]
    fig, ax = plt.subplots(figsize=(11.0, max(5.0, 0.43 * len(selected))))
    positions = np.arange(len(selected))
    ax.barh(
        positions,
        effects[::-1],
        xerr=errors[::-1],
        color=colors[::-1],
        alpha=0.92,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels[::-1])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("TCAV effect relative to matched random CAV controls")
    ax.set_xlabel("mean(real TCAV - random TCAV), error bar = real-score std")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    if args.min_valid_runs > args.num_cav_runs:
        raise ValueError("--min-valid-runs cannot exceed --num-cav-runs.")

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)
    manifest_classes = read_manifest_classes(manifest)
    label_by_class = class_name_to_label(manifest_classes)

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
        manifest_classes,
    )
    concepts = list(dict.fromkeys(args.concepts))
    target_classes = (
        list(dict.fromkeys(args.target_classes))
        if args.target_classes
        else choose_target_classes(concept_bank, concepts, args.max_target_classes)
    )

    coverage = concept_coverage_rows(
        concept_bank,
        concepts,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    write_csv(coverage, args.coverage_output)
    for row in coverage:
        LOGGER.info(
            "coverage concept=%s positive_classes=%s negative_classes=%s class_disjoint=%s",
            row["concept"],
            row["positive_class_count"],
            row["negative_class_count"],
            row["class_disjoint_ready"],
        )

    train_dataset = deterministic_dataset(manifest, args.train_split)
    eval_dataset = deterministic_dataset(manifest, args.eval_split)

    model = build_resnet50_classifier(
        num_classes=num_classes,
        pretrained=False,
        trainable_modules=("layer3", "layer4", "fc"),
    )
    load_checkpoint(
        model,
        checkpoint,
        device,
        expected_class_mapping={
            index: class_name for index, class_name in enumerate(manifest_classes)
        },
    )
    model.to(device)
    model.eval()

    LOGGER.info("Extracting deterministic activation cache from %d rows", len(train_dataset))
    activation_cache = extract_activation_cache(
        model=model,
        dataset=train_dataset,
        layer_name=args.layer,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        pool=args.pool,
    )

    gradient_cache: dict[str, tuple[torch.Tensor, int, int]] = {}
    for target_class in target_classes:
        key = normalize_class_name(target_class)
        if key not in label_by_class:
            LOGGER.warning("Skipping unknown target class: %s", target_class)
            continue
        eval_indices = select_class_sample_indices(
            eval_dataset,
            target_class,
            max_samples=args.max_eval_per_class,
            seed=args.seed,
        )
        if not eval_indices:
            LOGGER.warning("No evaluation images for target class: %s", target_class)
            continue
        loader = build_subset_loader(
            eval_dataset,
            eval_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        label = label_by_class[key]
        gradients = extract_pooled_gradients(
            model=model,
            dataloader=loader,
            layer_name=args.layer,
            target_label=label,
            device=device,
            pool=args.pool,
        )
        canonical_name = manifest_classes[label]
        gradient_cache[canonical_name] = (gradients, label, len(eval_indices))
        LOGGER.info(
            "Cached target gradients class=%s samples=%d features=%d",
            canonical_name,
            gradients.size(0),
            gradients.size(1),
        )

    run_rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    vector_metadata: list[dict[str, object]] = []
    train_samples = getattr(train_dataset, "samples")

    for concept_name in concepts:
        for target_class, (target_gradients, target_label, n_eval) in gradient_cache.items():
            excluded_names = {target_class} if args.exclude_target_class else set()
            valid_pair_rows: list[dict[str, object]] = []
            for run_index in range(args.num_cav_runs):
                run_seed = args.seed + run_index * 1009 + target_label * 37
                try:
                    selection = select_concept_sample_indices(
                        dataset=train_dataset,
                        concept_bank=concept_bank,
                        concept_name=concept_name,
                        positive_threshold=args.positive_threshold,
                        negative_threshold=args.negative_threshold,
                        max_positive=args.max_concept_examples,
                        max_negative=args.max_concept_examples,
                        min_examples=args.min_concept_examples,
                        seed=run_seed,
                        excluded_class_names=excluded_names,
                    )
                    split = split_concept_selection(
                        train_dataset,
                        selection,
                        validation_fraction=args.cav_validation_fraction,
                        seed=run_seed,
                        prefer_class_disjoint=args.prefer_class_disjoint,
                    )
                except ValueError as error:
                    LOGGER.warning(
                        "Skipping concept=%s target=%s run=%d: %s",
                        concept_name,
                        target_class,
                        run_index,
                        error,
                    )
                    continue

                cav = train_cav(
                    positive_activations=_take(
                        activation_cache.activations,
                        split.positive_train_indices,
                    ),
                    negative_activations=_take(
                        activation_cache.activations,
                        split.negative_train_indices,
                    ),
                    positive_validation_activations=_take(
                        activation_cache.activations,
                        split.positive_validation_indices,
                    ),
                    negative_validation_activations=_take(
                        activation_cache.activations,
                        split.negative_validation_indices,
                    ),
                    concept_name=concept_name,
                    layer_name=args.layer,
                    positive_classes=selection.positive_classes,
                    negative_classes=selection.negative_classes,
                    epochs=args.cav_epochs,
                    lr=args.cav_lr,
                    weight_decay=args.cav_weight_decay,
                    seed=run_seed,
                )
                if cav.validation_accuracy < args.minimum_cav_validation_accuracy:
                    LOGGER.warning(
                        "Rejecting weak CAV concept=%s target=%s run=%d val_acc=%.3f",
                        concept_name,
                        target_class,
                        run_index,
                        cav.validation_accuracy,
                    )
                    continue

                target_key = normalize_class_name(target_class)
                target_excluded_indices = {
                    index
                    for index, sample in enumerate(train_samples)
                    if normalize_class_name(str(sample.class_name)) == target_key
                }
                try:
                    random_positive, random_negative = select_random_control_indices(
                        dataset_size=len(train_dataset),
                        positive_count=cav.positive_count,
                        negative_count=cav.negative_count,
                        seed=run_seed + 500_003,
                        excluded_indices=target_excluded_indices,
                    )
                    random_selection = _random_selection(
                        train_dataset,
                        random_positive,
                        random_negative,
                        concept_name,
                    )
                    random_split = split_concept_selection(
                        train_dataset,
                        random_selection,
                        validation_fraction=args.cav_validation_fraction,
                        seed=run_seed + 700_001,
                        prefer_class_disjoint=False,
                    )
                    random_cav = train_cav(
                        positive_activations=_take(
                            activation_cache.activations,
                            random_split.positive_train_indices,
                        ),
                        negative_activations=_take(
                            activation_cache.activations,
                            random_split.negative_train_indices,
                        ),
                        positive_validation_activations=_take(
                            activation_cache.activations,
                            random_split.positive_validation_indices,
                        ),
                        negative_validation_activations=_take(
                            activation_cache.activations,
                            random_split.negative_validation_indices,
                        ),
                        concept_name=f"random_control_for_{concept_name}",
                        layer_name=args.layer,
                        positive_classes="random group A",
                        negative_classes="random group B",
                        epochs=args.cav_epochs,
                        lr=args.cav_lr,
                        weight_decay=args.cav_weight_decay,
                        seed=run_seed + 900_001,
                    )
                except ValueError as error:
                    LOGGER.warning(
                        "Skipping random control concept=%s target=%s run=%d: %s",
                        concept_name,
                        target_class,
                        run_index,
                        error,
                    )
                    continue

                real_score = score_cav_from_gradients(target_gradients, cav.vector)
                random_score = score_cav_from_gradients(
                    target_gradients,
                    random_cav.vector,
                )
                row = {
                    "concept": concept_name,
                    "concept_group": concept_group(concept_name),
                    "target_class": target_class,
                    "target_label": target_label,
                    "run": run_index,
                    "seed": run_seed,
                    "layer": args.layer,
                    "pool": args.pool,
                    "target_class_excluded": args.exclude_target_class,
                    "split_strategy": split.strategy,
                    "random_split_strategy": random_split.strategy,
                    "positive_count": cav.positive_count,
                    "negative_count": cav.negative_count,
                    "positive_train_count": cav.positive_train_count,
                    "negative_train_count": cav.negative_train_count,
                    "positive_validation_count": cav.positive_validation_count,
                    "negative_validation_count": cav.negative_validation_count,
                    "positive_threshold": selection.positive_threshold,
                    "negative_threshold": selection.negative_threshold,
                    "cav_train_accuracy": cav.train_accuracy,
                    "cav_validation_accuracy": cav.validation_accuracy,
                    "cav_train_loss": cav.final_loss,
                    "cav_validation_loss": cav.validation_loss,
                    "random_cav_train_accuracy": random_cav.train_accuracy,
                    "random_cav_validation_accuracy": random_cav.validation_accuracy,
                    "tcav_score": real_score.tcav_score,
                    "random_tcav_score": random_score.tcav_score,
                    "mean_directional_derivative": real_score.mean_directional_derivative,
                    "std_directional_derivative": real_score.std_directional_derivative,
                    "random_mean_directional_derivative": random_score.mean_directional_derivative,
                    "n_eval": n_eval,
                    "positive_classes": cav.positive_classes,
                    "negative_classes": cav.negative_classes,
                }
                valid_pair_rows.append(row)

                vector_key = (
                    f"cav_{_safe_key(concept_name)}_{_safe_key(target_class)}_"
                    f"run_{run_index:03d}"
                )
                random_key = f"random_{vector_key}"
                vectors[vector_key] = cav.vector.numpy()
                vectors[random_key] = random_cav.vector.numpy()
                vector_metadata.append(
                    {
                        "key": vector_key,
                        "random_key": random_key,
                        "concept": concept_name,
                        "target_class": target_class,
                        "run": run_index,
                        "seed": run_seed,
                        "validation_accuracy": cav.validation_accuracy,
                        "random_validation_accuracy": random_cav.validation_accuracy,
                        "split_strategy": split.strategy,
                    }
                )

            if len(valid_pair_rows) < args.min_valid_runs:
                LOGGER.warning(
                    "Dropping concept-target pair %s/%s: valid_runs=%d required=%d",
                    concept_name,
                    target_class,
                    len(valid_pair_rows),
                    args.min_valid_runs,
                )
                continue
            run_rows.extend(valid_pair_rows)

    if not run_rows:
        raise RuntimeError(
            "No concept-target pair produced enough validated CAV runs. "
            "Review concept coverage, target exclusion, thresholds and validation accuracy."
        )

    score_rows = aggregate_tcav_run_rows(
        run_rows,
        alpha=args.significance_alpha,
        correction=args.multiple_testing,
        permutation_seed=args.seed,
        max_permutations=args.max_permutations,
    )
    write_csv(run_rows, args.run_output)
    write_csv(score_rows, args.score_output)
    write_csv(concept_summary_rows(run_rows), args.cav_output)
    accepted_runs = {
        (str(row["concept"]), str(row["target_class"]), int(row["run"]))
        for row in run_rows
    }
    vector_metadata = [
        row
        for row in vector_metadata
        if (str(row["concept"]), str(row["target_class"]), int(row["run"]))
        in accepted_runs
    ]
    accepted_vector_keys = {
        key
        for row in vector_metadata
        for key in (str(row["key"]), str(row["random_key"]))
    }
    vectors = {
        key: value for key, value in vectors.items() if key in accepted_vector_keys
    }
    save_cav_artifact(
        vectors,
        vector_metadata,
        args.cav_artifact_output,
        {
            "manifest": str(manifest),
            "checkpoint": str(checkpoint),
            "metadata_root": str(metadata_root),
            "layer": args.layer,
            "pool": args.pool,
            "matrix_kind": args.matrix_kind,
            "seed": args.seed,
            "num_requested_runs": args.num_cav_runs,
            "target_class_excluded": args.exclude_target_class,
            "class_names": manifest_classes,
            "concept_names": concepts,
        },
    )

    scored_concepts = [
        concept
        for concept in concepts
        if any(str(row["concept"]) == concept for row in score_rows)
    ]
    scored_classes = [
        target
        for target in target_classes
        if any(
            normalize_class_name(str(row["target_class"]))
            == normalize_class_name(target)
            for row in score_rows
        )
    ]
    save_tcav_heatmap(
        score_rows,
        scored_concepts,
        scored_classes,
        args.heatmap_output,
    )
    save_tcav_barplot(score_rows, args.bar_output)

    significant_count = sum(bool(row["significant"]) for row in score_rows)
    LOGGER.info(
        "Validated TCAV complete pairs=%d significant=%d runs=%s scores=%s",
        len(score_rows),
        significant_count,
        args.run_output,
        args.score_output,
    )


if __name__ == "__main__":
    main()
