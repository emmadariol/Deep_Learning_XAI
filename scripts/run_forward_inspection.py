"""Inspect a real ResNet50 forward pass on one AwA2 image."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ImageManifestDataset, build_resnet_transforms
from src.forward_inspection import (
    ForwardActivationInspector,
    print_trace_summary,
    save_trace_json,
    save_prediction_trace_figure,
)
from src.model import build_resnet50_classifier, get_device
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("run_forward_inspection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect real intermediate ResNet50 tensors for one AwA2 image.")
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "real_forward_inspection.png",
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "real_forward_trace.json",
        help="Compact JSON trace that can be loaded by docs/resnet_activation_simulator.html.",
    )
    parser.add_argument("--image-path", type=Path, default=None, help="Optional specific image to inspect.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def infer_num_classes(manifest_path: Path) -> int:
    labels: set[int] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.add(int(row["label"]))
    if not labels:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    return max(labels) + 1


def load_class_names(manifest_path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names[int(row["label"])] = row["class_name"]
    return names


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)


def find_manifest_sample_by_path(dataset: ImageManifestDataset, image_path: Path) -> int:
    resolved = image_path.expanduser().resolve()
    for index, sample in enumerate(dataset.samples):
        if sample.filepath.resolve() == resolved:
            return index
    raise ValueError(f"Image path is not present in the selected manifest split: {resolved}")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)
    class_names = load_class_names(manifest)

    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("checkpoint=%s", checkpoint)
    LOGGER.info("device=%s", device)

    dataset = ImageManifestDataset(
        manifest_path=manifest,
        split=args.split,
        transform=build_resnet_transforms(train=False),
    )
    sample_index = (
        find_manifest_sample_by_path(dataset, args.image_path)
        if args.image_path is not None
        else args.sample_index
    )
    image, label, true_name, image_path = dataset[sample_index]

    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    with ForwardActivationInspector(model) as inspector:
        trace = inspector.run(
            image=image,
            target_label=args.target_label,
            compute_gradcam=True,
        )

    LOGGER.info("selected image=%s true_label=%d true_name=%s", image_path, int(label), true_name)
    print(f"image={image_path}")
    print(f"true_label={int(label)} true_name={true_name}")
    print_trace_summary(trace, class_names)

    save_prediction_trace_figure(
        image=image,
        trace=trace,
        class_names=class_names,
        output_path=args.output,
        true_label=int(label),
        image_path=image_path,
    )
    save_trace_json(
        image=image,
        trace=trace,
        class_names=class_names,
        output_path=args.trace_json,
        true_label=int(label),
        image_path=image_path,
    )
    print(f"saved_figure={args.output.expanduser().resolve()}")
    print(f"saved_trace_json={args.trace_json.expanduser().resolve()}")


if __name__ == "__main__":
    main()
