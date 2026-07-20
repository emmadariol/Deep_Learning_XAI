"""Run project workflows through separated pipeline profiles.

The individual phase scripts remain the source of truth. This wrapper only
assembles their documented command-line calls into repeatable workflows.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: list[str]
    outputs: tuple[Path, ...] = ()


PROFILES: dict[str, tuple[str, ...]] = {
    "data": ("prepare", "validate"),
    "train": ("validate", "train"),
    "outputs": (
        "validate",
        "xai",
        "xai-errors",
        "stress",
        "concepts",
        "advanced-audit",
        "error-audit",
    ),
    "full": (
        "validate",
        "xai",
        "xai-errors",
        "stress",
        "concepts",
        "tcav",
        "tcav-stress",
        "cbm",
        "advanced-audit",
        "error-audit",
    ),
}

CHECKPOINT_REQUIRED_STAGES = {
    "xai",
    "xai-errors",
    "stress",
    "tcav",
    "tcav-stress",
    "cbm",
    "advanced-audit",
    "error-audit",
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def python_cmd(args: list[str]) -> list[str]:
    return [sys.executable, *args]


def add_common(command: list[str], args: argparse.Namespace) -> list[str]:
    return [
        *command,
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--log-level",
        args.log_level,
    ]


def build_stages(args: argparse.Namespace) -> dict[str, Stage]:
    manifest = project_path(args.manifest)
    metadata_root = project_path(args.metadata_root)

    checkpoint = project_path(args.checkpoint)
    history = project_path("outputs/reports/training_history.csv")
    train_extra = [
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
    ]
    xai_max_images = str(args.max_images)
    ig_steps = str(args.ig_steps)
    stress_methods = ["gradcam", "integrated_gradients"]

    cbm_checkpoint = project_path(args.cbm_checkpoint)

    prepare_command = [
        "scripts/data/prepare_awa2.py",
        "--mode",
        "subset",
        "--source-root",
        str(project_path(args.source_root)),
        "--output-root",
        str(manifest.parent),
        "--preset",
        "background20",
        "--max-images-per-class",
        str(args.max_images_per_class),
        "--resize-size",
        str(args.resize_size),
        "--resize-method",
        "pad",
        "--jpeg-quality",
        "92",
        "--allow-existing",
    ]

    stages = {
        "prepare": Stage(
            "prepare",
            "Create or validate the portable AwA2 subset manifest.",
            python_cmd(prepare_command),
            (manifest,),
        ),
        "validate": Stage(
            "validate",
            "Run manifest and dataloader checks.",
            python_cmd(
                [
                    "scripts/data/general_tests.py",
                    "--manifest",
                    str(manifest),
                    "--batch-size",
                    "4",
                    "--num-workers",
                    str(args.num_workers),
                    "--log-level",
                    args.log_level,
                ]
            ),
            (),
        ),
        "train": Stage(
            "train",
            "Train the ResNet50 baseline checkpoint.",
            add_common(
                python_cmd(
                    [
                        "scripts/training/train_baseline.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint-path",
                        str(checkpoint),
                        "--history-path",
                        str(history),
                        *train_extra,
                    ]
                ),
                args,
            ),
            (checkpoint, history),
        ),
        "xai": Stage(
            "xai",
            "Generate attribution examples for correct predictions.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_xai.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        "outputs/figures/xai_correct_examples.png",
                        "--selection",
                        "correct",
                        "--max-images",
                        xai_max_images,
                        "--ig-steps",
                        ig_steps,
                    ]
                ),
                args,
            ),
            (project_path("outputs/figures/xai_correct_examples.png"),),
        ),
        "xai-errors": Stage(
            "xai-errors",
            "Generate attribution examples for incorrect predictions.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_xai.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        "outputs/figures/xai_incorrect_examples.png",
                        "--selection",
                        "incorrect",
                        "--max-images",
                        xai_max_images,
                        "--ig-steps",
                        ig_steps,
                    ]
                ),
                args,
            ),
            (project_path("outputs/figures/xai_incorrect_examples.png"),),
        ),
        "stress": Stage(
            "stress",
            "Run background perturbation saliency metrics.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_background_stress_metrics.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--csv-output",
                        "outputs/reports/phase5_saliency_metrics.csv",
                        "--perturbation-figure-output",
                        "outputs/figures/phase5_perturbations.png",
                        "--figure-output",
                        "outputs/figures/phase5_saliency_comparison.png",
                        "--max-images",
                        xai_max_images,
                        "--xai-methods",
                        *stress_methods,
                        "--ig-steps",
                        ig_steps,
                        "--mask-strategy",
                        args.mask_strategy,
                    ]
                ),
                args,
            ),
            (
                project_path("outputs/reports/phase5_saliency_metrics.csv"),
                project_path("outputs/figures/phase5_saliency_comparison.png"),
            ),
        ),
        "concepts": Stage(
            "concepts",
            "Analyze AwA2 concept profiles and prediction transitions.",
            python_cmd(
                [
                    "scripts/experiments/analyze_concept_profiles.py",
                    "--manifest",
                    str(manifest),
                    "--metadata-root",
                    str(metadata_root),
                    "--stress-csv",
                    "outputs/reports/phase5_saliency_metrics.csv",
                    "--class-profile-output",
                    "outputs/reports/phase6_class_concepts.csv",
                    "--transition-output",
                    "outputs/reports/phase6_concept_transitions.csv",
                    "--heatmap-output",
                    "outputs/figures/phase6_class_concept_heatmap.png",
                    "--skip-transition-figure",
                    "--log-level",
                    args.log_level,
                ]
            ),
            (
                project_path("outputs/reports/phase6_class_concepts.csv"),
                project_path("outputs/reports/phase6_concept_transitions.csv"),
            ),
        ),
        "tcav": Stage(
            "tcav",
            "Run repeated TCAV with held-out CAV validation and random controls.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_tcav.py",
                        "--manifest",
                        str(manifest),
                        "--metadata-root",
                        str(metadata_root),
                        "--checkpoint",
                        str(checkpoint),
                        "--concepts",
                        "stripes",
                        "furry",
                        "hooves",
                        "horns",
                        "flippers",
                        "--layer",
                        "layer3",
                        "--num-cav-runs",
                        str(args.num_cav_runs),
                        "--min-valid-runs",
                        "5",
                        "--score-output",
                        "outputs/reports/phase7_tcav_scores.csv",
                        "--run-output",
                        "outputs/reports/phase7_tcav_runs.csv",
                        "--cav-output",
                        "outputs/reports/phase7_cav_summary.csv",
                        "--coverage-output",
                        "outputs/reports/phase7_concept_coverage.csv",
                        "--cav-artifact-output",
                        "outputs/reports/phase7_cav_vectors.npz",
                        "--heatmap-output",
                        "outputs/figures/phase7_tcav_heatmap.png",
                        "--bar-output",
                        "outputs/figures/phase7_tcav_top_scores.png",
                    ]
                ),
                args,
            ),
            (
                project_path("outputs/reports/phase7_tcav_scores.csv"),
                project_path("outputs/reports/phase7_cav_vectors.npz"),
            ),
        ),
        "tcav-stress": Stage(
            "tcav-stress",
            "Audit fixed TCAV directions under background perturbations.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_tcav_stress.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--cav-artifact",
                        "outputs/reports/phase7_cav_vectors.npz",
                        "--run-output",
                        "outputs/reports/tcav_stress_runs.csv",
                        "--summary-output",
                        "outputs/reports/tcav_stress_summary.csv",
                        "--figure-output",
                        "outputs/figures/tcav_stress_effects.png",
                    ]
                ),
                args,
            ),
            (project_path("outputs/reports/tcav_stress_summary.csv"),),
        ),
        "cbm": Stage(
            "cbm",
            "Train and evaluate the Concept Bottleneck Model.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/train_cbm.py",
                        "--manifest",
                        str(manifest),
                        "--metadata-root",
                        str(metadata_root),
                        "--backbone-checkpoint",
                        str(checkpoint),
                        "--checkpoint-path",
                        str(cbm_checkpoint),
                        "--history-output",
                        "outputs/reports/phase8_cbm_history.csv",
                        "--summary-output",
                        "outputs/reports/phase8_cbm_summary.csv",
                        "--concept-metrics-output",
                        "outputs/reports/phase8_concept_metrics.csv",
                        "--concept-confusion-output",
                        "outputs/reports/phase8_concept_confusion_matrix.csv",
                        "--predictions-output",
                        "outputs/reports/phase8_cbm_predictions.csv",
                        "--error-analysis-output",
                        "outputs/reports/phase8_cbm_error_analysis.csv",
                        "--error-summary-output",
                        "outputs/reports/phase8_cbm_error_summary.csv",
                        "--intervention-output",
                        "outputs/reports/phase8_oracle_prototype_interventions.csv",
                        "--image-intervention-output",
                        "outputs/reports/phase8_image_concept_interventions.csv",
                        "--training-figure-output",
                        "outputs/figures/phase8_cbm_training.png",
                        "--summary-figure-output",
                        "outputs/figures/phase8_cbm_summary.png",
                        "--concept-figure-output",
                        "outputs/figures/phase8_concept_prediction_metrics.png",
                        "--concept-confusion-figure-output",
                        "outputs/figures/phase8_concept_confusion_matrix.png",
                        "--intervention-figure-output",
                        "outputs/figures/phase8_oracle_prototype_interventions.png",
                        "--image-intervention-figure-output",
                        "outputs/figures/phase8_image_concept_interventions.png",
                        "--error-figure-output",
                        "outputs/figures/phase8_cbm_error_analysis.png",
                        "--top-concepts",
                        "20",
                        "--epochs",
                        str(args.cbm_epochs),
                    ]
                ),
                args,
            ),
            (cbm_checkpoint, project_path("outputs/reports/phase8_cbm_summary.csv")),
        ),
        "advanced-audit": Stage(
            "advanced-audit",
            "Run attribution faithfulness and stability diagnostics.",
            add_common(
                python_cmd(
                    [
                        "scripts/audits/run_advanced_attribution_audit.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--methods",
                        "gradcam",
                        "integrated_gradients",
                        "--num-examples",
                        xai_max_images,
                        "--ig-steps",
                        ig_steps,
                        "--report-output",
                        "outputs/reports/advanced_attribution_audit.csv",
                        "--summary-output",
                        "outputs/reports/advanced_attribution_audit_summary.csv",
                        "--figure-dir",
                        "outputs/figures/advanced_attribution_audit",
                    ]
                ),
                args,
            ),
            (project_path("outputs/reports/advanced_attribution_audit.csv"),),
        ),
        "error-audit": Stage(
            "error-audit",
            "Compare wrong and true targets for misclassified examples.",
            add_common(
                python_cmd(
                    [
                        "scripts/experiments/run_misclassification_audit.py",
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint),
                        "--cbm-checkpoint",
                        str(cbm_checkpoint),
                        "--metadata-root",
                        str(metadata_root),
                        "--max-images",
                        xai_max_images,
                        "--ig-steps",
                        ig_steps,
                        "--figure-directory",
                        "outputs/figures/misclassification_audit",
                        *(["--skip-cbm"] if args.profile != "full" else []),
                    ]
                ),
                args,
            ),
            (project_path("outputs/reports/misclassification_decision_audit.csv"),),
        ),
    }
    return stages


def selected_stage_names(args: argparse.Namespace) -> list[str]:
    if args.stages:
        names = args.stages
    else:
        names = list(PROFILES[args.profile])
    return names


def outputs_exist(stage: Stage) -> bool:
    return bool(stage.outputs) and all(path.exists() for path in stage.outputs)


def run_stage(stage: Stage, args: argparse.Namespace) -> None:
    quoted = " ".join(shlex.quote(part) for part in stage.command)
    print(f"\n==> {stage.name}: {stage.description}")
    if args.skip_existing and outputs_exist(stage):
        outputs = ", ".join(rel(path) for path in stage.outputs)
        print(f"Skipping {stage.name}; outputs already exist: {outputs}")
        return
    print(quoted)
    if args.dry_run:
        return
    subprocess.run(stage.command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    stage_choices = (
        "prepare",
        "validate",
        "train",
        "xai",
        "xai-errors",
        "stress",
        "concepts",
        "tcav",
        "tcav-stress",
        "cbm",
        "advanced-audit",
        "error-audit",
    )
    parser = argparse.ArgumentParser(
        description="Run AwA2 XAI project phases through pipeline profiles."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="outputs")
    parser.add_argument("--stages", nargs="+", choices=stage_choices)
    parser.add_argument("--source-root", default="data/AWA2")
    parser.add_argument("--manifest", default="data/AWA2_subset_background20/awa2_manifest_subset.csv")
    parser.add_argument("--metadata-root", default="data/AWA2")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_resnet50_awa2.pt")
    parser.add_argument("--cbm-checkpoint", default="outputs/checkpoints/phase8_cbm.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--early-stopping-patience", default=3, type=int)
    parser.add_argument("--max-images", default=4, type=int)
    parser.add_argument("--ig-steps", default=16, type=int)
    parser.add_argument("--mask-strategy", default="center_ellipse")
    parser.add_argument("--num-cav-runs", default=20, type=int)
    parser.add_argument("--cbm-epochs", default=5, type=int)
    parser.add_argument("--max-images-per-class", default=200, type=int)
    parser.add_argument("--resize-size", default=128, type=int)
    parser.add_argument(
        "--reuse-existing",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Reuse stage outputs that already exist.",
    )
    parser.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Recompute stages even when their outputs already exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = build_stages(args)
    names = selected_stage_names(args)

    manifest = project_path(args.manifest)
    if not args.dry_run and "prepare" not in names and not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Run --profile data first or create it manually."
        )

    checkpoint = project_path(args.checkpoint)
    needs_checkpoint = any(name in CHECKPOINT_REQUIRED_STAGES for name in names)
    if not args.dry_run and "train" not in names and needs_checkpoint and not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run --profile train first or pass --checkpoint."
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Profile: {args.profile}")
    print(f"Stages: {', '.join(names)}")
    for name in names:
        run_stage(stages[name], args)


if __name__ == "__main__":
    main()
