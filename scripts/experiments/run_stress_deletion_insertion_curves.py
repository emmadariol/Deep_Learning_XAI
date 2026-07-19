"""Generate deletion/insertion curves for each background stress test.

These figures preserve the meaning of the report's Figure 10 and Figure 11:
they are behavioral faithfulness curves, not bar summaries. For every
background intervention, the script computes the selected attribution method on
the perturbed images and then measures how the fixed target score changes when
the most salient pixels are deleted or inserted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_background_stress_metrics import compute_saliency_maps
from scripts.experiments.run_xai import collect_correct_examples
from src.attribution_audit import (
    deletion_insertion_curves,
    save_deletion_insertion_plot,
    trapezoid_auc,
)
from src.data import build_dataloaders, infer_num_classes, load_idx_to_class
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.perturb import (
    apply_perturbation_suite,
    predict_batch_probabilities,
)
from src.utils import set_seed, setup_logging, write_csv
from src.validation import (
    device_spec,
    log_level,
    nonnegative_float,
    nonnegative_int,
    open_unit_float,
    positive_int,
)
from src.xai import blurred_baseline


LOGGER = logging.getLogger("run_stress_deletion_insertion_curves")

METHOD_LABELS = {
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
}

PERTURBATION_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "color_shift": "Colour shift",
    "background_swap": "Background replacement",
}

PERTURBATION_SLUGS = {
    "gaussian_noise": "gaussian-noise",
    "color_shift": "colour-shift",
    "background_swap": "background-replacement",
}

METHOD_SLUGS = {
    "gradcam": "gradcam",
    "integrated_gradients": "integrated-gradients",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deletion/insertion curves on background-perturbed images."
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
        "--methods",
        nargs="+",
        choices=["gradcam", "integrated_gradients"],
        default=["gradcam", "integrated_gradients"],
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--max-images", type=positive_int, default=4)
    parser.add_argument("--ig-steps", type=positive_int, default=16)
    parser.add_argument("--ig-internal-batch-size", type=positive_int, default=4)
    parser.add_argument("--curve-steps", type=positive_int, default=10)
    parser.add_argument("--blur-radius", type=nonnegative_float, default=18.0)
    parser.add_argument("--mask-strategy", choices=["center_ellipse", "center_box", "global"], default="center_ellipse")
    parser.add_argument("--foreground-scale", type=open_unit_float, default=0.68)
    parser.add_argument("--noise-std", type=nonnegative_float, default=0.25)
    parser.add_argument("--allow-incorrect", action="store_true")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "stress_deletion_insertion",
    )
    parser.add_argument(
        "--docs-output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "xai-report",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "stress_deletion_insertion_curves.csv",
    )
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


def _copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
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
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    images, _labels, true_names, _predicted_names, _confidences, _image_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        idx_to_class=idx_to_class,
        max_images=args.max_images,
        allow_incorrect=args.allow_incorrect,
        seed=args.seed,
    )
    images = images.to(device)
    target_labels, _target_confidences, _target_probabilities = predict_batch_probabilities(
        model,
        images,
    )

    _background_mask, perturbed_batches = apply_perturbation_suite(
        inputs=images,
        mask_strategy=args.mask_strategy,
        foreground_scale=args.foreground_scale,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    fractions = [index / args.curve_steps for index in range(args.curve_steps + 1)]
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.docs_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for perturbation_name, perturbed_images in perturbed_batches.items():
        perturbation_label = PERTURBATION_LABELS.get(perturbation_name, perturbation_name)
        perturbation_slug = PERTURBATION_SLUGS.get(
            perturbation_name,
            perturbation_name.replace("_", "-"),
        )
        faithfulness_baseline = blurred_baseline(
            perturbed_images,
            blur_radius=args.blur_radius,
        )
        for method in args.methods:
            LOGGER.info(
                "Computing deletion/insertion curves: perturbation=%s method=%s",
                perturbation_name,
                method,
            )
            maps = compute_saliency_maps(
                model=model,
                images=perturbed_images,
                targets=target_labels,
                method=method,
                ig_steps=args.ig_steps,
                ig_internal_batch_size=args.ig_internal_batch_size,
            )
            deletion_scores, insertion_scores = deletion_insertion_curves(
                model=model,
                inputs=perturbed_images,
                targets=target_labels,
                maps=maps,
                fractions=fractions,
                baseline=faithfulness_baseline,
            )
            deletion_auc = trapezoid_auc(fractions, deletion_scores)
            insertion_auc = trapezoid_auc(fractions, insertion_scores)

            method_slug = METHOD_SLUGS[method]
            figure_path = args.figure_dir / f"{perturbation_slug}_{method_slug}_deletion_insertion.png"
            save_deletion_insertion_plot(
                fractions=fractions,
                deletion_scores=deletion_scores,
                insertion_scores=insertion_scores,
                method=f"{METHOD_LABELS[method]} under {perturbation_label}",
                output_path=figure_path,
            )
            docs_path = args.docs_output_dir / f"stress-curve-{perturbation_slug}-{method_slug}.png"
            _copy_bytes(figure_path, docs_path)

            for index, true_name in enumerate(true_names):
                rows.append(
                    {
                        "perturbation": perturbation_name,
                        "xai_method": method,
                        "index": index,
                        "true_class": true_name,
                        "target_class": idx_to_class[int(target_labels[index].item())],
                        "deletion_auc": float(deletion_auc[index].item()),
                        "insertion_auc": float(insertion_auc[index].item()),
                        "figure_path": str(figure_path),
                        "docs_asset": str(docs_path),
                    }
                )
            LOGGER.info("saved %s and %s", figure_path, docs_path)

    write_csv(rows, args.report_output)
    LOGGER.info("stress deletion/insertion curves complete: %s", args.report_output)


if __name__ == "__main__":
    main()
