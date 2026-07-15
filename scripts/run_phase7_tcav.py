"""Run Phase 7 TCAV analysis on AwA2 semantic concepts."""

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

from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    normalize_class_name,
    read_manifest_classes,
)
from src.data import build_dataloaders, infer_num_classes
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.tcav import (
    build_subset_loader,
    compute_tcav_score,
    extract_pooled_activations,
    find_concept_index,
    select_class_sample_indices,
    select_concept_sample_indices,
    train_cav,
)
from src.utils import set_seed, setup_logging, write_csv

LOGGER = logging.getLogger("run_phase7_tcav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Concept Activation Vectors and compute TCAV scores."
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
            "Folder containing AwA2 classes.txt, predicates.txt and predicate matrices. "
            "If omitted, the script searches near the manifest and data/AWA2."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["stripes", "furry", "hooves", "horns", "flippers"],
        help="AwA2 semantic attributes to turn into CAVs.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=None,
        help="Animal classes to score. If omitted, classes are chosen from strongest concepts.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="layer3",
        help=(
            "Model layer used as TCAV bottleneck. layer3 is the default because "
            "the output of layer4 is followed only by avgpool+fc in ResNet50, "
            "which makes class-score gradients nearly constant per class."
        ),
    )
    parser.add_argument("--pool", choices=["avg", "flatten"], default="avg")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=0.75)
    parser.add_argument("--negative-threshold", type=float, default=0.25)
    parser.add_argument("--max-concept-examples", type=int, default=200)
    parser.add_argument("--min-concept-examples", type=int, default=4)
    parser.add_argument("--max-eval-per-class", type=int, default=40)
    parser.add_argument("--max-target-classes", type=int, default=10)
    parser.add_argument("--cav-epochs", type=int, default=200)
    parser.add_argument("--cav-lr", type=float, default=1e-2)
    parser.add_argument("--cav-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--score-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_scores.csv",
    )
    parser.add_argument(
        "--cav-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_cav_summary.csv",
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
    parser.add_argument("--matrix-kind", choices=["continuous", "binary"], default="continuous")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def class_name_to_label(manifest_classes: list[str]) -> dict[str, int]:
    return {normalize_class_name(name): index for index, name in enumerate(manifest_classes)}


def choose_target_classes(concept_bank, concept_names: list[str], max_classes: int) -> list[str]:
    """Pick compact default target classes across all requested concepts."""
    matrix = concept_bank.normalized_matrix()
    selected: list[str] = []
    ranked_classes_by_concept: list[list[str]] = []

    for concept_name in concept_names:
        try:
            concept_index = find_concept_index(concept_bank, concept_name)
        except ValueError as error:
            LOGGER.warning("skipping target-class ranking for concept=%s: %s", concept_name, error)
            continue
        ranked_classes_by_concept.append(
            [
                concept_bank.class_names[int(class_index)]
                for class_index in np.argsort(matrix[:, concept_index])[::-1]
            ]
        )

    for rank in range(len(concept_bank.class_names)):
        for ranked_classes in ranked_classes_by_concept:
            if rank >= len(ranked_classes):
                continue
            class_name = ranked_classes[rank]
            if class_name in selected:
                continue
            selected.append(class_name)
            if len(selected) >= max_classes:
                return selected

    if not selected:
        selected = concept_bank.class_names[:max_classes]
    return selected


def save_tcav_heatmap(
    rows: list[dict[str, object]],
    concepts: list[str],
    target_classes: list[str],
    output_path: Path,
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.full((len(target_classes), len(concepts)), np.nan, dtype=np.float64)
    class_to_row = {normalize_class_name(name): index for index, name in enumerate(target_classes)}
    concept_to_col = {normalize_class_name(name): index for index, name in enumerate(concepts)}

    for row in rows:
        class_key = normalize_class_name(str(row["target_class"]))
        concept_key = normalize_class_name(str(row["concept"]))
        if class_key in class_to_row and concept_key in concept_to_col:
            matrix[class_to_row[class_key], concept_to_col[concept_key]] = float(row["tcav_score"])

    fig, ax = plt.subplots(
        figsize=(max(7.0, 1.15 * len(concepts)), max(4.5, 0.45 * len(target_classes)))
    )
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Phase 7 TCAV scores")
    ax.set_xlabel("Concept")
    ax.set_ylabel("Target class")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=35, ha="right")
    ax.set_yticks(range(len(target_classes)))
    ax.set_yticklabels(target_classes)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )

    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="fraction positive")
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def save_tcav_barplot(rows: list[dict[str, object]], output_path: Path, top_n: int = 15) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = sorted(rows, key=lambda row: float(row["tcav_score"]), reverse=True)[:top_n]
    labels = [f"{row['target_class']} | {row['concept']}" for row in selected]
    values = [float(row["tcav_score"]) for row in selected]
    colors = [float(row["mean_directional_derivative"]) for row in selected]

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.42 * len(selected))))
    bars = ax.barh(labels[::-1], values[::-1], color=plt.cm.magma_r(np.linspace(0.18, 0.78, len(selected))))
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Highest TCAV class-concept sensitivities")
    ax.set_xlabel("TCAV score")
    ax.grid(axis="x", alpha=0.25)
    for bar, value, derivative in zip(bars, values[::-1], colors[::-1]):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ({derivative:.2e})",
            va="center",
            fontsize=8.5,
        )
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)
    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("checkpoint=%s", checkpoint)
    LOGGER.info("num_classes=%d device=%s", num_classes, device)

    search_roots = [
        args.metadata_root,
        manifest.parent,
        PROJECT_ROOT / "data" / "AWA2",
        PROJECT_ROOT / "data",
    ]
    metadata_root = find_awa2_metadata_root([path for path in search_roots if path is not None])
    LOGGER.info("metadata_root=%s", metadata_root)

    manifest_classes = read_manifest_classes(manifest)
    label_by_class = class_name_to_label(manifest_classes)
    concept_bank = load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind)
    concept_bank = align_concept_bank_to_manifest(concept_bank, manifest_classes)

    concepts = list(dict.fromkeys(args.concepts))
    if args.target_classes:
        target_classes = list(dict.fromkeys(args.target_classes))
    else:
        target_classes = choose_target_classes(
            concept_bank=concept_bank,
            concept_names=concepts,
            max_classes=args.max_target_classes,
        )

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.train_split not in loaders:
        raise ValueError(f"Unknown train split: {args.train_split}")
    if args.eval_split not in loaders:
        raise ValueError(f"Unknown eval split: {args.eval_split}")

    train_dataset = loaders[args.train_split].dataset
    eval_dataset = loaders[args.eval_split].dataset

    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False)
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    score_rows: list[dict[str, object]] = []
    cav_rows: list[dict[str, object]] = []

    for concept_name in concepts:
        LOGGER.info("processing concept=%s", concept_name)
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
                seed=args.seed,
            )
        except ValueError as error:
            LOGGER.warning("skipping concept=%s: %s", concept_name, error)
            continue

        positive_loader = build_subset_loader(
            train_dataset,
            selection.positive_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        negative_loader = build_subset_loader(
            train_dataset,
            selection.negative_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        positive_activations = extract_pooled_activations(
            model=model,
            dataloader=positive_loader,
            layer_name=args.layer,
            device=device,
            pool=args.pool,
        )
        negative_activations = extract_pooled_activations(
            model=model,
            dataloader=negative_loader,
            layer_name=args.layer,
            device=device,
            pool=args.pool,
        )
        cav = train_cav(
            positive_activations=positive_activations,
            negative_activations=negative_activations,
            concept_name=concept_name,
            layer_name=args.layer,
            positive_classes=selection.positive_classes,
            negative_classes=selection.negative_classes,
            epochs=args.cav_epochs,
            lr=args.cav_lr,
            weight_decay=args.cav_weight_decay,
            seed=args.seed,
        )
        cav_rows.append(
            {
                "concept": concept_name,
                "layer": args.layer,
                "pool": args.pool,
                "positive_count": cav.positive_count,
                "negative_count": cav.negative_count,
                "positive_threshold": selection.positive_threshold,
                "negative_threshold": selection.negative_threshold,
                "cav_train_accuracy": cav.train_accuracy,
                "cav_final_loss": cav.final_loss,
                "positive_classes": cav.positive_classes,
                "negative_classes": cav.negative_classes,
            }
        )

        for target_class in target_classes:
            target_key = normalize_class_name(target_class)
            if target_key not in label_by_class:
                LOGGER.warning("target class not found in manifest, skipping: %s", target_class)
                continue

            eval_indices = select_class_sample_indices(
                dataset=eval_dataset,
                class_name=target_class,
                max_samples=args.max_eval_per_class,
                seed=args.seed,
            )
            if not eval_indices:
                LOGGER.warning("no eval images for target class=%s", target_class)
                continue

            eval_loader = build_subset_loader(
                eval_dataset,
                eval_indices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            score = compute_tcav_score(
                model=model,
                dataloader=eval_loader,
                layer_name=args.layer,
                target_label=label_by_class[target_key],
                cav_vector=cav.vector,
                device=device,
                pool=args.pool,
            )
            score_rows.append(
                {
                    "concept": concept_name,
                    "target_class": manifest_classes[label_by_class[target_key]],
                    "target_label": label_by_class[target_key],
                    "layer": args.layer,
                    "pool": args.pool,
                    "tcav_score": score.tcav_score,
                    "mean_directional_derivative": score.mean_directional_derivative,
                    "std_directional_derivative": score.std_directional_derivative,
                    "n_eval": score.n_eval,
                    "cav_train_accuracy": cav.train_accuracy,
                    "positive_count": cav.positive_count,
                    "negative_count": cav.negative_count,
                }
            )

    if not cav_rows:
        raise RuntimeError("No CAVs were trained. Check concept names, thresholds and manifest size.")
    if not score_rows:
        raise RuntimeError("No TCAV scores were computed. Check target classes and eval split.")

    write_csv(cav_rows, args.cav_output)
    write_csv(score_rows, args.score_output)
    scored_concepts = [
        concept
        for concept in concepts
        if any(normalize_class_name(str(row["concept"])) == normalize_class_name(concept) for row in score_rows)
    ]
    scored_target_classes = [
        class_name
        for class_name in target_classes
        if any(
            normalize_class_name(str(row["target_class"])) == normalize_class_name(class_name)
            for row in score_rows
        )
    ]
    save_tcav_heatmap(score_rows, scored_concepts, scored_target_classes, args.heatmap_output)
    save_tcav_barplot(score_rows, args.bar_output)

    LOGGER.info(
        "Phase 7 complete: scores=%s cavs=%s heatmap=%s barplot=%s",
        args.score_output,
        args.cav_output,
        args.heatmap_output,
        args.bar_output,
    )


if __name__ == "__main__":
    main()
