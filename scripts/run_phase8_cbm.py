"""Run Phase 8 Concept Bottleneck Model training and analysis."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bottleneck import (
    ConceptBottleneckModel,
    ConceptTargetDataset,
    collect_prediction_rows,
    concept_names_from_indices,
    intervention_rows,
    load_backbone_checkpoint,
    per_concept_metrics,
    select_concept_indices,
    train_cbm,
    trainable_parameters,
    write_history_csv,
)
from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    read_manifest_classes,
)
from src.data import build_dataloaders
from src.model import get_device
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("run_phase8_cbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and inspect a simple AwA2 Concept Bottleneck Model."
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
        "--backbone-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
        help="Optional Phase 2 classifier checkpoint used to initialize the CBM backbone.",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "phase8_cbm.pt",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_cbm_history.csv",
    )
    parser.add_argument(
        "--concept-metrics-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_concept_metrics.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_cbm_predictions.csv",
    )
    parser.add_argument(
        "--intervention-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase8_concept_interventions.csv",
    )
    parser.add_argument(
        "--training-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase8_cbm_training.png",
    )
    parser.add_argument(
        "--concept-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase8_concept_prediction_metrics.png",
    )
    parser.add_argument(
        "--intervention-figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase8_concept_interventions.png",
    )
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=[],
        help="Optional concept names to force into the bottleneck vocabulary.",
    )
    parser.add_argument("--top-concepts", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--concept-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--trainable-backbone-layers", nargs="+", default=["layer4"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--max-prediction-batches", type=int, default=4)
    parser.add_argument("--max-intervention-classes", type=int, default=10)
    parser.add_argument("--top-interventions-per-class", type=int, default=6)
    parser.add_argument(
        "--use-imagenet-pretrained",
        action="store_true",
        help="Initialize the CBM backbone from torchvision ImageNet weights.",
    )
    parser.add_argument("--matrix-kind", choices=["continuous", "binary"], default="continuous")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("saved %s", output_path)


def build_concept_dataloaders(
    manifest: Path,
    concept_bank,
    concept_indices: list[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    base_loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loaders: dict[str, DataLoader] = {}
    for split, loader in base_loaders.items():
        dataset = ConceptTargetDataset(
            image_dataset=loader.dataset,
            concept_bank=concept_bank,
            concept_indices=concept_indices,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders


def choose_intervention_classes(concept_bank, concept_indices: list[int], max_classes: int) -> list[str]:
    """Select class examples that strongly express the selected concepts."""
    matrix = concept_bank.normalized_matrix()
    ranked: list[list[str]] = []
    for concept_index in concept_indices:
        ranked.append(
            [
                concept_bank.class_names[int(class_index)]
                for class_index in np.argsort(matrix[:, concept_index])[::-1]
            ]
        )

    selected: list[str] = []
    for rank in range(len(concept_bank.class_names)):
        for ranked_classes in ranked:
            class_name = ranked_classes[rank]
            if class_name in selected:
                continue
            selected.append(class_name)
            if len(selected) >= max_classes:
                return selected
    return selected


def load_best_checkpoint(model: ConceptBottleneckModel, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("loaded best CBM checkpoint: %s", checkpoint_path)


def save_training_curves(history_path: Path, output_path: Path) -> None:
    rows = read_csv_rows(history_path)
    epochs = sorted({int(row["epoch"]) for row in rows})
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val")
    }

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for split, split_rows in by_split.items():
        split_rows = sorted(split_rows, key=lambda row: int(row["epoch"]))
        axes[0].plot(
            epochs,
            [float(row["loss"]) for row in split_rows],
            marker="o",
            label=split,
        )
        axes[1].plot(
            epochs,
            [float(row["class_acc"]) for row in split_rows],
            marker="o",
            label=split,
        )
        axes[2].plot(
            epochs,
            [float(row["concept_mae"]) for row in split_rows],
            marker="o",
            label=split,
        )
    axes[0].set_title("CBM loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Class accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[2].set_title("Concept MAE")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MAE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def save_concept_metrics_plot(metrics_rows: list[dict[str, object]], output_path: Path) -> None:
    selected = sorted(metrics_rows, key=lambda row: float(row["mae"]))[:20]
    labels = [str(row["concept"]) for row in selected]
    mae_values = [float(row["mae"]) for row in selected]
    binary_acc = [float(row["binary_accuracy"]) for row in selected]

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, max(5.2, 0.34 * len(selected))))
    bars = ax.barh(labels[::-1], mae_values[::-1], color=plt.cm.viridis(binary_acc[::-1]))
    ax.set_title("Best predicted bottleneck concepts")
    ax.set_xlabel("mean absolute error")
    ax.grid(axis="x", alpha=0.25)
    for bar, acc in zip(bars, binary_acc[::-1]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"bin-acc={acc:.2f}",
            va="center",
            fontsize=8.5,
        )
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def save_intervention_plot(rows: list[dict[str, object]], output_path: Path, top_n: int = 18) -> None:
    selected = sorted(rows, key=lambda row: abs(float(row["intervention_delta"])), reverse=True)[:top_n]
    labels = [f"{row['target_class']} | {row['concept']}" for row in selected]
    values = [float(row["intervention_delta"]) for row in selected]
    colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in values]

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, max(5.2, 0.38 * len(selected))))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_title("Largest concept intervention effects")
    ax.set_xlabel("class probability change when concept is set 0 -> 1")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", output_path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    device = get_device(args.device)
    manifest_classes = read_manifest_classes(manifest)
    num_classes = len(manifest_classes)
    idx_to_class = {index: class_name for index, class_name in enumerate(manifest_classes)}

    search_roots = [
        args.metadata_root,
        manifest.parent,
        PROJECT_ROOT / "data" / "AWA2",
        PROJECT_ROOT / "data",
    ]
    metadata_root = find_awa2_metadata_root([path for path in search_roots if path is not None])
    concept_bank = load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind)
    concept_bank = align_concept_bank_to_manifest(concept_bank, manifest_classes)

    concept_indices = select_concept_indices(
        concept_bank=concept_bank,
        requested_concepts=args.concepts,
        top_k=args.top_concepts,
    )
    concept_names = concept_names_from_indices(concept_bank, concept_indices)
    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("metadata_root=%s", metadata_root)
    LOGGER.info("num_classes=%d num_concepts=%d device=%s", num_classes, len(concept_names), device)
    LOGGER.info("concepts=%s", ", ".join(concept_names))

    loaders = build_concept_dataloaders(
        manifest=manifest,
        concept_bank=concept_bank,
        concept_indices=concept_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ConceptBottleneckModel(
        num_classes=num_classes,
        num_concepts=len(concept_names),
        pretrained=args.use_imagenet_pretrained,
        trainable_backbone_layers=tuple(args.trainable_backbone_layers),
        dropout=args.dropout,
    )
    load_backbone_checkpoint(model, args.backbone_checkpoint, device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = train_cbm(
        model=model,
        dataloaders=loaders,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
        checkpoint_path=args.checkpoint_output,
        concept_loss_weight=args.concept_loss_weight,
        class_loss_weight=args.class_loss_weight,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    write_history_csv(history, args.history_output)
    save_training_curves(args.history_output, args.training_figure_output)

    load_best_checkpoint(model, args.checkpoint_output, device)
    test_metrics = per_concept_metrics(
        model=model,
        dataloader=loaders["test"],
        concept_names=concept_names,
        device=device,
        max_batches=args.max_test_batches,
    )
    write_csv(test_metrics, args.concept_metrics_output)
    save_concept_metrics_plot(test_metrics, args.concept_figure_output)

    prediction_rows = collect_prediction_rows(
        model=model,
        dataloader=loaders["test"],
        idx_to_class=idx_to_class,
        concept_names=concept_names,
        device=device,
        max_batches=args.max_prediction_batches,
    )
    write_csv(prediction_rows, args.predictions_output)

    intervention_classes = choose_intervention_classes(
        concept_bank=concept_bank,
        concept_indices=concept_indices,
        max_classes=args.max_intervention_classes,
    )
    interventions = intervention_rows(
        model=model,
        concept_bank=concept_bank,
        concept_indices=concept_indices,
        target_classes=intervention_classes,
        device=device,
        top_k_per_class=args.top_interventions_per_class,
    )
    write_csv(interventions, args.intervention_output)
    save_intervention_plot(interventions, args.intervention_figure_output)

    LOGGER.info(
        "Phase 8 complete: checkpoint=%s history=%s concept_metrics=%s predictions=%s interventions=%s",
        args.checkpoint_output,
        args.history_output,
        args.concept_metrics_output,
        args.predictions_output,
        args.intervention_output,
    )


if __name__ == "__main__":
    main()
