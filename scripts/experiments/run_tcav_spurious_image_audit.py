"""Image-level audit for suspicious TCAV concept-class pairs."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "deep_learning_xai"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg-cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.concepts import (
    align_concept_bank_to_manifest,
    find_awa2_metadata_root,
    load_awa2_concepts,
    normalize_class_name,
    read_manifest_classes,
)
from src.data import ImageManifestDataset, build_resnet_transforms, denormalize_batch, infer_num_classes
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.perturb import apply_perturbation_suite, predict_batch_probabilities
from src.tcav import (
    LayerActivationRecorder,
    build_subset_loader,
    pool_layer_tensor,
    select_class_sample_indices,
    select_concept_sample_indices,
    split_concept_selection,
    train_cav,
)
from src.utils import setup_logging, write_csv

LOGGER = logging.getLogger("run_tcav_spurious_image_audit")

DEFAULT_PAIRS = (
    "flippers:polar+bear",
    "horns:dolphin",
    "hooves:dolphin",
    "flippers:tiger",
    "hooves:antelope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image-level TCAV derivative and background-stress audit for suspicious pairs."
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
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        metavar="CONCEPT:CLASS",
        help="Concept-class pair to audit. Can be repeated.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_image_level_audit.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_tcav_image_level_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_tcav_image_level_summary.png",
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase7_tcav_image_level_examples.png",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--layer", default="layer3")
    parser.add_argument("--pool", choices=["avg", "flatten"], default="avg")
    parser.add_argument("--matrix-kind", choices=["continuous", "binary"], default="continuous")
    parser.add_argument("--positive-threshold", type=float, default=0.75)
    parser.add_argument("--negative-threshold", type=float, default=0.25)
    parser.add_argument("--max-concept-examples", type=int, default=160)
    parser.add_argument("--min-concept-examples", type=int, default=8)
    parser.add_argument("--cav-validation-fraction", type=float, default=0.25)
    parser.add_argument("--prefer-class-disjoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cav-epochs", type=int, default=140)
    parser.add_argument("--cav-lr", type=float, default=1e-2)
    parser.add_argument("--cav-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-target-images", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--foreground-scale", type=float, default=0.68)
    parser.add_argument("--mask-strategy", choices=["center_ellipse", "center_box"], default="center_ellipse")
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--top-examples", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def parse_pair(pair: str) -> tuple[str, str]:
    if ":" not in pair:
        raise ValueError(f"Pair must look like CONCEPT:CLASS, got {pair!r}")
    concept, target = pair.split(":", 1)
    return concept.strip(), target.strip()


def safe_key(value: str) -> str:
    return normalize_class_name(value).replace("_", "-")


def take_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_subset_activations(model, dataset, indices, args, device) -> torch.Tensor:
    loader = build_subset_loader(
        dataset,
        indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    from src.tcav import extract_pooled_activations

    return extract_pooled_activations(
        model=model,
        dataloader=loader,
        layer_name=args.layer,
        device=device,
        pool=args.pool,
    )


def train_concept_cav(model, train_dataset, concept_bank, concept_name, args, device):
    selection = select_concept_sample_indices(
        dataset=train_dataset,
        concept_bank=concept_bank,
        concept_name=concept_name,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        max_positive=args.max_concept_examples,
        max_negative=args.max_concept_examples,
        min_examples=args.min_concept_examples,
        seed=args.seed + abs(hash(concept_name)) % 10_000,
    )
    split = split_concept_selection(
        train_dataset,
        selection,
        validation_fraction=args.cav_validation_fraction,
        seed=args.seed + 177,
        prefer_class_disjoint=args.prefer_class_disjoint,
    )
    needed_indices = (
        split.positive_train_indices
        + split.negative_train_indices
        + split.positive_validation_indices
        + split.negative_validation_indices
    )
    activations = extract_subset_activations(model, train_dataset, needed_indices, args, device)
    offset = 0

    def take(count: int) -> torch.Tensor:
        nonlocal offset
        chunk = activations[offset : offset + count]
        offset += count
        return chunk

    pos_train = take(len(split.positive_train_indices))
    neg_train = take(len(split.negative_train_indices))
    pos_val = take(len(split.positive_validation_indices))
    neg_val = take(len(split.negative_validation_indices))
    cav = train_cav(
        positive_activations=pos_train,
        negative_activations=neg_train,
        positive_validation_activations=pos_val,
        negative_validation_activations=neg_val,
        concept_name=concept_name,
        layer_name=args.layer,
        positive_classes=selection.positive_classes,
        negative_classes=selection.negative_classes,
        epochs=args.cav_epochs,
        lr=args.cav_lr,
        weight_decay=args.cav_weight_decay,
        seed=args.seed,
    )
    return cav, selection, split


def directional_derivatives_for_batch(model, images, target_label: int, cav_vector: torch.Tensor, layer_name: str, pool: str, device):
    model.eval()
    images = images.to(device)
    with LayerActivationRecorder(model, layer_name) as recorder:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(images)
            activation = recorder.activation
            if activation is None or not activation.requires_grad:
                raise RuntimeError(f"Layer {layer_name!r} did not expose differentiable activation.")
            score = logits[:, target_label].sum()
            gradients = torch.autograd.grad(score, activation, retain_graph=False)[0]
    pooled = pool_layer_tensor(gradients.detach(), pool=pool).cpu().float()
    vector = cav_vector.detach().cpu().float().view(1, -1)
    directional = (pooled * vector).sum(dim=1)
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)
    target_probs = probabilities[:, target_label]
    top_probs, top_labels = probabilities.max(dim=1)
    return directional, target_probs, top_labels, top_probs


def sample_visual_stats(pil_image: Image.Image) -> dict[str, float]:
    image = np.asarray(pil_image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    height, width, _ = image.shape
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ellipse = ((xx - cx) / (0.68 * width / 2.0)) ** 2 + ((yy - cy) / (0.68 * height / 2.0)) ** 2 <= 1.0
    background = ~ellipse
    bg = image[background]
    blue_green = float((bg[:, 2].mean() + bg[:, 1].mean()) / 2.0)
    whiteness = float((bg.mean(axis=1) > 0.72).mean())
    saturation_proxy = float(np.abs(bg.max(axis=1) - bg.min(axis=1)).mean())
    return {
        "background_blue_green_mean": blue_green,
        "background_white_fraction": whiteness,
        "background_saturation_proxy": saturation_proxy,
    }


def class_label_map(class_names: list[str]) -> dict[str, int]:
    return {normalize_class_name(name): index for index, name in enumerate(class_names)}


def audit_pair(model, eval_dataset, cav, concept_name, target_class, target_label, class_names, args, device):
    indices = select_class_sample_indices(
        eval_dataset,
        target_class,
        max_samples=args.max_target_images,
        seed=args.seed + target_label,
    )
    loader = DataLoader(
        Subset(eval_dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    rows: list[dict[str, object]] = []
    sample_offset = 0
    for batch_index, batch in enumerate(loader):
        images, labels, _class_names, paths = batch
        original_deriv, original_prob, original_pred, original_pred_prob = directional_derivatives_for_batch(
            model,
            images,
            target_label,
            cav.vector,
            args.layer,
            args.pool,
            device,
        )
        mask, perturbed = apply_perturbation_suite(
            images.to(device),
            mask_strategy=args.mask_strategy,
            foreground_scale=args.foreground_scale,
            methods=("gaussian_noise", "color_shift", "background_swap"),
            noise_std=args.noise_std,
            seed=args.seed + batch_index,
        )
        perturbed_outputs = {}
        for method_name, perturbed_images in perturbed.items():
            perturbed_outputs[method_name] = directional_derivatives_for_batch(
                model,
                perturbed_images.detach().cpu(),
                target_label,
                cav.vector,
                args.layer,
                args.pool,
                device,
            )
        for local_index, path in enumerate(paths):
            pil = Image.open(path).convert("RGB")
            stats = sample_visual_stats(pil)
            row = {
                "concept": concept_name,
                "target_class": target_class,
                "sample_index": indices[sample_offset + local_index],
                "filepath": path,
                "true_class": str(_class_names[local_index]),
                "target_label": target_label,
                "original_directional_derivative": float(original_deriv[local_index].item()),
                "original_positive_direction": bool(original_deriv[local_index].item() > 0.0),
                "original_target_probability": float(original_prob[local_index].item()),
                "original_prediction": class_names[int(original_pred[local_index].item())],
                "original_prediction_probability": float(original_pred_prob[local_index].item()),
                **stats,
            }
            for method_name, outputs in perturbed_outputs.items():
                deriv, prob, pred, pred_prob = outputs
                value = float(deriv[local_index].item())
                row[f"{method_name}_directional_derivative"] = value
                row[f"{method_name}_directional_delta"] = value - float(original_deriv[local_index].item())
                row[f"{method_name}_positive_direction"] = bool(value > 0.0)
                row[f"{method_name}_target_probability"] = float(prob[local_index].item())
                row[f"{method_name}_target_probability_delta"] = float(prob[local_index].item()) - float(original_prob[local_index].item())
                row[f"{method_name}_prediction"] = class_names[int(pred[local_index].item())]
                row[f"{method_name}_prediction_probability"] = float(pred_prob[local_index].item())
            rows.append(row)
        sample_offset += len(paths)
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["concept"]), str(row["target_class"]))].append(row)
    summary = []
    for (concept, target), values in grouped.items():
        original = np.array([float(row["original_directional_derivative"]) for row in values])
        bg = np.array([float(row["background_swap_directional_derivative"]) for row in values])
        prob = np.array([float(row["original_target_probability"]) for row in values])
        bg_prob = np.array([float(row["background_swap_target_probability"]) for row in values])
        changed = np.array([row["background_swap_prediction"] != row["original_prediction"] for row in values], dtype=float)
        white = np.array([float(row["background_white_fraction"]) for row in values])
        blue_green = np.array([float(row["background_blue_green_mean"]) for row in values])
        def corr(a, b):
            if len(a) < 3 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
                return ""
            return round(float(np.corrcoef(a, b)[0, 1]), 4)
        summary.append(
            {
                "concept": concept,
                "target_class": target,
                "samples": len(values),
                "positive_direction_rate": round(float((original > 0.0).mean()), 4),
                "background_swap_positive_direction_rate": round(float((bg > 0.0).mean()), 4),
                "mean_directional_derivative": round(float(original.mean()), 8),
                "mean_background_swap_derivative": round(float(bg.mean()), 8),
                "mean_background_swap_derivative_delta": round(float((bg - original).mean()), 8),
                "mean_target_probability": round(float(prob.mean()), 6),
                "mean_background_swap_probability_delta": round(float((bg_prob - prob).mean()), 6),
                "background_swap_prediction_change_rate": round(float(changed.mean()), 4),
                "corr_derivative_background_white_fraction": corr(original, white),
                "corr_derivative_background_blue_green": corr(original, blue_green),
            }
        )
    return summary


def save_summary_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    summary = summarize_rows(rows)
    labels = [f"{row['concept']} -> {row['target_class']}" for row in summary]
    pos = np.array([float(row["positive_direction_rate"]) for row in summary])
    bg_pos = np.array([float(row["background_swap_positive_direction_rate"]) for row in summary])
    delta = np.array([float(row["mean_background_swap_derivative_delta"]) for row in summary])
    change = np.array([float(row["background_swap_prediction_change_rate"]) for row in summary])
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, max(4.8, 0.55 * len(labels))))
    axes[0].barh(y - 0.18, pos, height=0.34, label="original", color="#0f766e")
    axes[0].barh(y + 0.18, bg_pos, height=0.34, label="background swap", color="#9a5a2b")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Positive TCAV direction rate")
    axes[0].legend(fontsize=8)
    colors = ["#0f766e" if value >= 0 else "#9a5a2b" for value in delta]
    axes[1].barh(y, delta, color=colors)
    axes[1].axvline(0, color="#314047", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_title("Mean derivative change after background swap")
    axes[2].barh(y, change, color="#64748b")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_title("Prediction-change rate after background swap")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_examples_figure(rows: list[dict[str, object]], output_path: Path, top_examples: int) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["concept"]), str(row["target_class"]))].append(row)
    selected_rows = []
    for key, values in grouped.items():
        ranked = sorted(values, key=lambda row: float(row["original_directional_derivative"]), reverse=True)
        selected_rows.extend(ranked[:top_examples])
    if not selected_rows:
        return
    row_count = len(selected_rows)
    fig, axes = plt.subplots(row_count, 2, figsize=(7.5, max(3.0, 2.45 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, row in enumerate(selected_rows):
        image = Image.open(str(row["filepath"])).convert("RGB")
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].axis("off")
        axes[row_index, 0].set_title(
            f"{row['concept']} -> {row['target_class']}\norig d={float(row['original_directional_derivative']):.2e} p={float(row['original_target_probability']):.2f}",
            fontsize=8,
        )
        image_arr = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0
        h, w, _ = image_arr.shape
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - w / 2) / (0.68 * w / 2)) ** 2 + ((yy - h / 2) / (0.68 * h / 2)) ** 2 > 1.0
        overlay = image_arr.copy()
        overlay[mask] = 0.55 * overlay[mask] + np.array([0.85, 0.15, 0.05]) * 0.45
        axes[row_index, 1].imshow(overlay)
        axes[row_index, 1].axis("off")
        axes[row_index, 1].set_title(
            f"bg swap delta d={float(row['background_swap_directional_delta']):.2e}\nwhite={float(row['background_white_fraction']):.2f} blue/green={float(row['background_blue_green_mean']):.2f}",
            fontsize=8,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    device = get_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    metadata_root = find_awa2_metadata_root(
        [path for path in [args.metadata_root, manifest.parent, PROJECT_ROOT / "data" / "AWA2", PROJECT_ROOT / "data"] if path is not None]
    )
    class_names = read_manifest_classes(manifest)
    label_by_class = class_label_map(class_names)
    concept_bank = align_concept_bank_to_manifest(
        load_awa2_concepts(metadata_root, matrix_kind=args.matrix_kind),
        class_names,
    )
    train_dataset = ImageManifestDataset(manifest, split=args.train_split, transform=build_resnet_transforms(train=False))
    eval_dataset = ImageManifestDataset(manifest, split=args.eval_split, transform=build_resnet_transforms(train=False))
    model = build_resnet50_classifier(
        num_classes=infer_num_classes(manifest),
        pretrained=False,
        trainable_modules=("layer3", "layer4", "fc"),
    )
    load_checkpoint(
        model,
        args.checkpoint,
        device,
        expected_class_mapping={index: name for index, name in enumerate(class_names)},
    )
    model.to(device)
    model.eval()
    pairs = [parse_pair(pair) for pair in (args.pair or DEFAULT_PAIRS)]
    concepts = sorted({concept for concept, _target in pairs})
    cavs = {}
    for concept in concepts:
        LOGGER.info("Training CAV for concept=%s", concept)
        cavs[concept] = train_concept_cav(model, train_dataset, concept_bank, concept, args, device)[0]
    rows = []
    for concept, target in pairs:
        target_key = normalize_class_name(target)
        if target_key not in label_by_class:
            raise ValueError(f"Unknown target class {target!r}")
        LOGGER.info("Auditing pair %s -> %s", concept, target)
        rows.extend(
            audit_pair(
                model,
                eval_dataset,
                cavs[concept],
                concept,
                target,
                label_by_class[target_key],
                class_names,
                args,
                device,
            )
        )
    summary = summarize_rows(rows)
    write_csv(rows, args.csv_output)
    write_csv(summary, args.summary_output)
    save_summary_figure(rows, args.figure_output)
    save_examples_figure(rows, args.examples_output, args.top_examples)
    LOGGER.info("Saved image-level rows to %s", args.csv_output)
    LOGGER.info("Saved summary rows to %s", args.summary_output)
    LOGGER.info("Saved figures to %s and %s", args.figure_output, args.examples_output)


if __name__ == "__main__":
    main()
