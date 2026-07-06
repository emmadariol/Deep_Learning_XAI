"""Inspect Phase 5 saliency metrics in a human-readable way."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot Phase 5 saliency degradation metrics."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/reports/phase5_saliency_metrics.csv"),
        help="Input CSV produced by scripts/run_phase5_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory where summary CSVs and plots are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of most unstable rows to print and save.",
    )
    return parser.parse_args()


def infer_iou_column(frame: pd.DataFrame) -> str:
    """Return the IoU column, accepting any configured top-percent suffix."""
    candidates = [column for column in frame.columns if column.startswith("iou_top_")]
    if not candidates:
        raise ValueError("Could not find an IoU column. Expected a name like iou_top_20pct.")
    return candidates[0]


def load_metrics(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Load metrics and add intuitive derived columns."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    iou_column = infer_iou_column(frame)
    numeric_columns = [
        "original_confidence",
        "perturbed_confidence",
        "confidence_delta",
        iou_column,
        "spearman",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["prediction_changed"].dtype == object:
        frame["prediction_changed"] = frame["prediction_changed"].astype(str).str.lower() == "true"

    frame["confidence_drop"] = frame["original_confidence"] - frame["perturbed_confidence"]
    frame["saliency_drift"] = (1.0 - frame[iou_column]) + (1.0 - frame["spearman"]) / 2.0
    frame["prediction_transition"] = (
        frame["original_prediction"].astype(str)
        + " -> "
        + frame["perturbed_prediction"].astype(str)
    )
    frame["case"] = (
        frame["true_class"].astype(str)
        + " -> "
        + frame["original_prediction"].astype(str)
        + " => "
        + frame["perturbed_prediction"].astype(str)
        + " / "
        + frame["perturbation"].astype(str)
        + " / "
        + frame["xai_method"].astype(str)
    )
    return frame, iou_column


def summarize(frame: pd.DataFrame, iou_column: str) -> pd.DataFrame:
    """Aggregate metrics by XAI method and perturbation."""
    return (
        frame.groupby(["xai_method", "perturbation"], as_index=False)
        .agg(
            examples=("index", "count"),
            prediction_change_rate=("prediction_changed", "mean"),
            mean_confidence_drop=("confidence_drop", "mean"),
            mean_iou=(iou_column, "mean"),
            std_iou=(iou_column, "std"),
            mean_spearman=("spearman", "mean"),
            std_spearman=("spearman", "std"),
            mean_saliency_drift=("saliency_drift", "mean"),
        )
        .sort_values(["mean_saliency_drift", "prediction_change_rate"], ascending=False)
    )


def summarize_prediction_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate which animal the model predicts after each perturbation."""
    return (
        frame.groupby(
            [
                "xai_method",
                "perturbation",
                "true_class",
                "original_prediction",
                "perturbed_prediction",
            ],
            as_index=False,
        )
        .agg(
            rows=("index", "count"),
            prediction_changed=("prediction_changed", "max"),
            mean_confidence_drop=("confidence_drop", "mean"),
            mean_saliency_drift=("saliency_drift", "mean"),
        )
        .sort_values(
            ["prediction_changed", "rows", "mean_saliency_drift"],
            ascending=[False, False, False],
        )
    )


def save_bar_plot(
    summary: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a grouped bar plot from the summary table."""
    pivot = summary.pivot(index="perturbation", columns="xai_method", values=value_column)
    ax = pivot.plot(kind="bar", figsize=(9.5, 4.8), rot=20)
    ax.set_title(title)
    ax.set_xlabel("Background perturbation")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="XAI method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_scatter(frame: pd.DataFrame, iou_column: str, output_path: Path) -> None:
    """Save IoU-vs-Spearman scatter plot."""
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    methods = sorted(frame["xai_method"].unique())
    markers = ["o", "s", "^", "D"]
    for marker, method in zip(markers, methods):
        subset = frame[frame["xai_method"] == method]
        ax.scatter(
            subset[iou_column],
            subset["spearman"],
            s=58,
            alpha=0.76,
            marker=marker,
            label=method,
        )
    ax.set_title("Saliency stability: IoU vs Spearman")
    ax.set_xlabel("IoU of top-saliency support")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.axvline(0.5, color="#b45309", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="#b45309", linestyle="--", linewidth=1)
    ax.grid(alpha=0.25)
    ax.legend(title="XAI method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_top_instability_plot(top_rows: pd.DataFrame, output_path: Path) -> None:
    """Save a horizontal ranking of the most unstable examples."""
    plot_rows = top_rows.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4.8, 0.42 * len(plot_rows))))
    ax.barh(plot_rows["case"], plot_rows["saliency_drift"], color="#0f766e")
    ax.set_title("Most unstable saliency explanations")
    ax.set_xlabel("Saliency drift score")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, iou_column = load_metrics(args.csv.expanduser().resolve())
    summary = summarize(frame, iou_column)
    transitions = summarize_prediction_transitions(frame)
    top_rows = frame.sort_values("saliency_drift", ascending=False).head(args.top_k)

    summary_path = output_dir / "phase5_metric_summary.csv"
    transitions_path = output_dir / "phase5_prediction_transitions.csv"
    top_path = output_dir / "phase5_most_unstable_examples.csv"
    summary.to_csv(summary_path, index=False)
    transitions.to_csv(transitions_path, index=False)
    top_rows.to_csv(top_path, index=False)

    save_bar_plot(
        summary,
        "mean_iou",
        "Mean IoU",
        "How much does the top-saliency region remain stable?",
        output_dir / "phase5_mean_iou_by_method.png",
    )
    save_bar_plot(
        summary,
        "mean_spearman",
        "Mean Spearman",
        "How much does the full saliency ranking remain stable?",
        output_dir / "phase5_mean_spearman_by_method.png",
    )
    save_bar_plot(
        summary,
        "mean_confidence_drop",
        "Mean confidence drop",
        "How much confidence is lost after perturbing the background?",
        output_dir / "phase5_confidence_drop_by_method.png",
    )
    save_scatter(frame, iou_column, output_dir / "phase5_iou_vs_spearman.png")
    save_top_instability_plot(top_rows, output_dir / "phase5_most_unstable_examples.png")

    print("\n=== Phase 5 CSV overview ===")
    print(f"rows: {len(frame)}")
    print(f"unique images: {frame['index'].nunique()}")
    print(f"xai methods: {', '.join(sorted(frame['xai_method'].unique()))}")
    print(f"perturbations: {', '.join(sorted(frame['perturbation'].unique()))}")
    print(f"prediction change rate: {frame['prediction_changed'].mean():.3f}")

    print("\n=== Summary by method and perturbation ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\n=== Animal predicted after perturbation ===")
    transition_columns = [
        "xai_method",
        "perturbation",
        "true_class",
        "original_prediction",
        "perturbed_prediction",
        "prediction_changed",
        "mean_confidence_drop",
        "mean_saliency_drift",
    ]
    print(
        transitions[transition_columns]
        .head(args.top_k)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    print("\n=== Most unstable examples ===")
    display_columns = [
        "true_class",
        "original_prediction",
        "perturbed_prediction",
        "prediction_transition",
        "xai_method",
        "perturbation",
        iou_column,
        "spearman",
        "confidence_drop",
        "saliency_drift",
        "prediction_changed",
    ]
    print(top_rows[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved prediction transitions: {transitions_path}")
    print(f"Saved unstable examples: {top_path}")
    print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()
