"""Measure whether validated AwA2 TCAV evidence survives background changes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.concepts import normalize_class_name, read_manifest_classes
from src.data import ImageManifestDataset, build_resnet_transforms, infer_num_classes
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.perturb import apply_perturbation_suite
from src.tcav import (
    adjust_p_values,
    build_subset_loader,
    extract_pooled_gradients,
    paired_permutation_p_value,
    score_cav_from_gradients,
    select_class_sample_indices,
)
from src.utils import set_seed, setup_logging, write_csv
from src.validation import device_spec, log_level, nonnegative_int, open_unit_float, positive_int

LOGGER = logging.getLogger("run_tcav_stress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute directional concept sensitivity after controlled background "
            "perturbations using the validated CAV bank produced by run_tcav.py."
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
        "--cav-artifact",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase7_cav_vectors.npz",
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--concepts", nargs="*", default=None)
    parser.add_argument("--target-classes", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gaussian_noise", "color_shift", "background_swap"],
        default=["gaussian_noise", "color_shift", "background_swap"],
    )
    parser.add_argument(
        "--mask-strategy",
        choices=["center_ellipse", "center_box", "global"],
        default="center_ellipse",
    )
    parser.add_argument("--foreground-scale", type=float, default=0.68)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--max-eval-per-class", type=positive_int, default=40)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--multiple-testing",
        choices=["benjamini_hochberg", "bonferroni"],
        default="benjamini_hochberg",
    )
    parser.add_argument("--significance-alpha", type=open_unit_float, default=0.05)
    parser.add_argument("--max-permutations", type=positive_int, default=10000)
    parser.add_argument(
        "--run-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "tcav_stress_runs.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "tcav_stress_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "tcav_stress_effects.png",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def load_cav_bank(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    path = path.expanduser().resolve()
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"CAV artifact requires both {path} and {metadata_path}. Run run_tcav.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema") != "validated-tcav-cav-bank-v1":
        raise ValueError("Unsupported or unvalidated CAV artifact schema.")
    with np.load(path, allow_pickle=False) as archive:
        vectors = {
            key: torch.from_numpy(np.asarray(archive[key], dtype=np.float32)).view(-1)
            for key in archive.files
        }
    for row in metadata.get("vectors", []):
        if row["key"] not in vectors or row["random_key"] not in vectors:
            raise ValueError(f"CAV metadata references a missing vector: {row}")
    return vectors, metadata


def collect_images(loader: DataLoader) -> torch.Tensor:
    batches = [batch[0].detach().cpu() for batch in loader]
    if not batches:
        raise RuntimeError("No evaluation images were selected.")
    return torch.cat(batches, dim=0)


def prediction_statistics(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_label: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    target_probs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size].to(device)
            probabilities = torch.softmax(model(batch), dim=1)
            predictions.append(probabilities.argmax(dim=1).cpu())
            target_probs.append(probabilities[:, target_label].cpu())
    return torch.cat(predictions), torch.cat(target_probs)


def perturb_images(
    images: torch.Tensor,
    methods: tuple[str, ...],
    device: torch.device,
    batch_size: int,
    mask_strategy: str,
    foreground_scale: float,
    noise_std: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {method: [] for method in methods}
    for batch_index, start in enumerate(range(0, images.size(0), batch_size)):
        batch = images[start : start + batch_size].to(device)
        _mask, perturbed = apply_perturbation_suite(
            inputs=batch,
            mask_strategy=mask_strategy,
            foreground_scale=foreground_scale,
            methods=methods,
            noise_std=noise_std,
            seed=seed + batch_index,
        )
        for method in methods:
            outputs[method].append(perturbed[method].detach().cpu())
    return {method: torch.cat(batches, dim=0) for method, batches in outputs.items()}


def gradient_cache(
    model: torch.nn.Module,
    images: torch.Tensor,
    layer: str,
    pool: str,
    target_label: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return extract_pooled_gradients(
        model=model,
        dataloader=loader,
        layer_name=layer,
        target_label=target_label,
        device=device,
        pool=pool,
    )


def aggregate_rows(
    rows: list[dict[str, object]],
    alpha: float,
    correction: str,
    seed: int,
    max_permutations: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["concept"]), str(row["target_class"]), str(row["perturbation"]))].append(row)

    summary: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for group_index, ((concept, target_class, perturbation), values) in enumerate(sorted(grouped.items())):
        real_delta = [float(row["tcav_delta"]) for row in values]
        random_delta = [float(row["random_tcav_delta"]) for row in values]
        p_value = paired_permutation_p_value(
            real_delta,
            random_delta,
            seed=seed + group_index,
            max_permutations=max_permutations,
        )
        raw_p_values.append(p_value)
        real_array = np.asarray(real_delta, dtype=np.float64)
        random_array = np.asarray(random_delta, dtype=np.float64)
        summary.append(
            {
                "concept": concept,
                "target_class": target_class,
                "perturbation": perturbation,
                "runs": len(values),
                "original_tcav_mean": float(np.mean([float(row["original_tcav_score"]) for row in values])),
                "perturbed_tcav_mean": float(np.mean([float(row["perturbed_tcav_score"]) for row in values])),
                "tcav_delta_mean": float(real_array.mean()),
                "tcav_delta_std": float(real_array.std(ddof=1)) if len(values) > 1 else 0.0,
                "random_tcav_delta_mean": float(random_array.mean()),
                "stress_effect_vs_random": float((real_array - random_array).mean()),
                "directional_derivative_delta_mean": float(
                    np.mean([float(row["directional_derivative_delta"]) for row in values])
                ),
                "prediction_change_rate": float(values[0]["prediction_change_rate"]),
                "target_probability_delta": float(values[0]["target_probability_delta"]),
                "p_value": p_value,
            }
        )
    corrected = adjust_p_values(raw_p_values, method=correction)
    for row, adjusted in zip(summary, corrected, strict=True):
        row["adjusted_p_value"] = adjusted
        row["significant"] = adjusted < alpha
        row["multiple_testing"] = correction
        row["alpha"] = alpha
    return summary


def save_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    selected = sorted(rows, key=lambda row: abs(float(row["tcav_delta_mean"])), reverse=True)[:24]
    labels = [f"{row['target_class']} | {row['concept']} | {row['perturbation']}" for row in selected]
    values = [float(row["tcav_delta_mean"]) for row in selected]
    colors = ["#b91c1c" if value < 0 else "#0f766e" for value in values]
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(selected))))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_title("Change in validated TCAV score after background perturbation")
    ax.set_xlabel("perturbed TCAV - original TCAV")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    vectors, metadata = load_cav_bank(args.cav_artifact)

    artifact_manifest = Path(str(metadata["manifest"])).expanduser()
    if artifact_manifest.name != manifest.name:
        raise ValueError("The CAV bank and stress-test manifest do not match.")
    layer = str(metadata["layer"])
    pool = str(metadata["pool"])
    class_names = read_manifest_classes(manifest)
    artifact_classes = list(metadata.get("class_names", []))
    if artifact_classes != class_names:
        raise ValueError("The CAV class mapping differs from the stress-test manifest.")
    label_by_class = {normalize_class_name(name): index for index, name in enumerate(class_names)}

    selected_concepts = {normalize_class_name(name) for name in args.concepts} if args.concepts else None
    selected_targets = {normalize_class_name(name) for name in args.target_classes} if args.target_classes else None
    vector_rows = [
        row
        for row in metadata["vectors"]
        if (selected_concepts is None or normalize_class_name(str(row["concept"])) in selected_concepts)
        and (selected_targets is None or normalize_class_name(str(row["target_class"])) in selected_targets)
    ]
    if not vector_rows:
        raise ValueError("No CAV runs match the requested concepts and target classes.")

    model = build_resnet50_classifier(
        num_classes=infer_num_classes(manifest),
        pretrained=False,
        trainable_modules=("layer3", "layer4", "fc"),
    )
    load_checkpoint(
        model,
        args.checkpoint,
        device,
        expected_class_mapping={
            index: class_name for index, class_name in enumerate(class_names)
        },
    )
    model.to(device).eval()
    dataset = ImageManifestDataset(
        manifest_path=manifest,
        split=args.eval_split,
        transform=build_resnet_transforms(train=False),
    )

    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in vector_rows:
        by_target[str(row["target_class"])].append(row)

    run_rows: list[dict[str, object]] = []
    for target_offset, (target_class, cav_rows) in enumerate(sorted(by_target.items())):
        target_key = normalize_class_name(target_class)
        if target_key not in label_by_class:
            raise ValueError(f"Unknown target class in CAV artifact: {target_class}")
        target_label = label_by_class[target_key]
        indices = select_class_sample_indices(
            dataset,
            target_class,
            max_samples=args.max_eval_per_class,
            seed=args.seed,
        )
        images = collect_images(
            build_subset_loader(dataset, indices, args.batch_size, args.num_workers, device.type == "cuda")
        )
        original_predictions, original_target_probs = prediction_statistics(
            model, images, target_label, device, args.batch_size
        )
        variants = perturb_images(
            images=images,
            methods=tuple(args.methods),
            device=device,
            batch_size=args.batch_size,
            mask_strategy=args.mask_strategy,
            foreground_scale=args.foreground_scale,
            noise_std=args.noise_std,
            seed=args.seed + target_offset * 10_000,
        )
        original_gradients = gradient_cache(
            model, images, layer, pool, target_label, device, args.batch_size
        )

        for method, perturbed_images in variants.items():
            perturbed_predictions, perturbed_target_probs = prediction_statistics(
                model, perturbed_images, target_label, device, args.batch_size
            )
            perturbed_gradients = gradient_cache(
                model, perturbed_images, layer, pool, target_label, device, args.batch_size
            )
            prediction_change_rate = float(
                (perturbed_predictions != original_predictions).float().mean().item()
            )
            target_probability_delta = float(
                (perturbed_target_probs - original_target_probs).mean().item()
            )

            for cav_row in cav_rows:
                real_original = score_cav_from_gradients(original_gradients, vectors[str(cav_row["key"])])
                real_perturbed = score_cav_from_gradients(perturbed_gradients, vectors[str(cav_row["key"])])
                random_original = score_cav_from_gradients(original_gradients, vectors[str(cav_row["random_key"])])
                random_perturbed = score_cav_from_gradients(perturbed_gradients, vectors[str(cav_row["random_key"])])
                run_rows.append(
                    {
                        "concept": cav_row["concept"],
                        "target_class": target_class,
                        "run": cav_row["run"],
                        "seed": cav_row["seed"],
                        "perturbation": method,
                        "mask_strategy": args.mask_strategy,
                        "foreground_scale": args.foreground_scale,
                        "n_eval": images.size(0),
                        "original_tcav_score": real_original.tcav_score,
                        "perturbed_tcav_score": real_perturbed.tcav_score,
                        "tcav_delta": real_perturbed.tcav_score - real_original.tcav_score,
                        "random_original_tcav_score": random_original.tcav_score,
                        "random_perturbed_tcav_score": random_perturbed.tcav_score,
                        "random_tcav_delta": random_perturbed.tcav_score - random_original.tcav_score,
                        "original_mean_directional_derivative": real_original.mean_directional_derivative,
                        "perturbed_mean_directional_derivative": real_perturbed.mean_directional_derivative,
                        "directional_derivative_delta": (
                            real_perturbed.mean_directional_derivative
                            - real_original.mean_directional_derivative
                        ),
                        "prediction_change_rate": prediction_change_rate,
                        "target_probability_delta": target_probability_delta,
                    }
                )

    summary_rows = aggregate_rows(
        run_rows,
        alpha=args.significance_alpha,
        correction=args.multiple_testing,
        seed=args.seed,
        max_permutations=args.max_permutations,
    )
    write_csv(run_rows, args.run_output)
    write_csv(summary_rows, args.summary_output)
    save_figure(summary_rows, args.figure_output)
    LOGGER.info(
        "TCAV stress audit complete: runs=%d groups=%d summary=%s",
        len(run_rows),
        len(summary_rows),
        args.summary_output,
    )


if __name__ == "__main__":
    main()
