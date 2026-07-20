"""Create Figure-09-to-12-style dashboards for each background stress test.

The report section around Figure 09 uses four complementary diagnostics:

1. saliency mass on animal/background regions,
2. stability of the saliency map,
3. deletion/insertion AUC,
4. stress-test IoU and prediction-change behavior.

This script keeps that visual logic and expands it per perturbation. For each
background intervention it creates:

- a Figure-09-style summary dashboard,
- a Grad-CAM method detail figure,
- an Integrated Gradients method detail figure,
- a case-study figure linking prediction and explanation movement.

The script reads existing CSV outputs only; it does not run training,
perturbations, Grad-CAM or Integrated Gradients again.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
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
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PALETTE = {
    "navy": "#18324a",
    "teal": "#2a9d8f",
    "amber": "#e9a23b",
    "red": "#d95f5f",
    "blue": "#4f7cac",
    "soft": "#f4f8f7",
    "line": "#cfd8dc",
    "ink": "#263238",
    "muted": "#61717a",
}

METHOD_NAMES = {
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated\nGradients",
}

METHOD_TITLES = {
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
}

METHOD_ASSET_SLUGS = {
    "gradcam": "gradcam",
    "integrated_gradients": "integrated-gradients",
}

PERTURBATION_NAMES = {
    "gaussian_noise": "Gaussian noise",
    "color_shift": "Colour shift",
    "background_swap": "Background replacement",
}

PERTURBATION_SLUGS = {
    "gaussian_noise": "gaussian-noise",
    "color_shift": "colour-shift",
    "background_swap": "background-replacement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure-09-to-12-style metric dashboards for each stress test."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/reports/phase5_saliency_metrics_notebook.csv"),
        help="Stress metrics CSV produced by run_background_stress_metrics.py.",
    )
    parser.add_argument(
        "--advanced-summary-csv",
        type=Path,
        default=Path("outputs/reports/advanced_attribution_audit_notebook_summary.csv"),
        help="Advanced attribution audit summary with saliency mass and mean AUC metrics.",
    )
    parser.add_argument(
        "--advanced-detail-csv",
        type=Path,
        default=Path("outputs/reports/advanced_attribution_audit_notebook.csv"),
        help="Advanced attribution audit per-example metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--docs-output-dir",
        type=Path,
        default=Path("docs/assets/xai-report"),
        help="Directory where report-ready assets are copied.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/stress_metric_dashboard_summary.csv"),
        help="Aggregated per-perturbation metrics CSV.",
    )
    return parser.parse_args()


def find_iou_column(columns: list[str]) -> str:
    for column in columns:
        if column.startswith("iou_top_"):
            return column
    raise ValueError("The metrics CSV does not contain an iou_top_* column.")


def load_stress_metrics(metrics_csv: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(metrics_csv)
    iou_column = find_iou_column(list(df.columns))
    required = {
        "index",
        "true_class",
        "xai_method",
        "perturbation",
        "prediction_changed",
        "original_prediction",
        "perturbed_prediction",
        "original_target_probability",
        "perturbed_target_probability",
        "confidence_drop",
        "spearman",
        iou_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required stress metric columns: {sorted(missing)}")

    df = df[df["xai_method"].isin(METHOD_NAMES)].copy()
    if df.empty:
        raise ValueError("No maintained XAI methods found in the stress metrics CSV.")

    numeric_columns = [
        "original_target_probability",
        "perturbed_target_probability",
        "confidence_drop",
        "spearman",
        iou_column,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["prediction_changed"] = df["prediction_changed"].astype(str).str.lower().eq("true")
    df["saliency_drift"] = (1.0 - df[iou_column]) + (1.0 - df["spearman"]) / 2.0
    df["transition"] = df["original_prediction"].astype(str) + " -> " + df[
        "perturbed_prediction"
    ].astype(str)
    return df, iou_column


def summarize_stress_metrics(stress_metrics: pd.DataFrame, iou_column: str) -> pd.DataFrame:
    summary = (
        stress_metrics.groupby(["perturbation", "xai_method"], as_index=False)
        .agg(
            mean_iou=(iou_column, "mean"),
            mean_spearman=("spearman", "mean"),
            mean_confidence_drop=("confidence_drop", "mean"),
            prediction_change_rate=("prediction_changed", "mean"),
            n_examples=("index", "nunique"),
        )
        .sort_values(["perturbation", "xai_method"])
    )
    summary["method_label"] = summary["xai_method"].map(METHOD_NAMES)
    summary["perturbation_label"] = summary["perturbation"].map(PERTURBATION_NAMES).fillna(
        summary["perturbation"]
    )
    return summary


def load_advanced_summary(advanced_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(advanced_csv)
    required = {
        "method",
        "mean_animal_saliency_ratio",
        "mean_background_saliency_ratio",
        "mean_deletion_auc",
        "mean_insertion_auc",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required advanced-audit columns: {sorted(missing)}")

    df = df[df["method"].isin(METHOD_NAMES)].copy()
    if df.empty:
        raise ValueError("No maintained XAI methods found in the advanced audit summary.")

    numeric_columns = [
        "mean_animal_saliency_ratio",
        "mean_background_saliency_ratio",
        "mean_deletion_auc",
        "mean_insertion_auc",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["label"] = df["method"].map(METHOD_NAMES)
    return df.set_index("method").reindex(METHOD_NAMES).reset_index()


def load_advanced_detail(advanced_detail_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(advanced_detail_csv)
    required = {
        "method",
        "index",
        "true_class",
        "deletion_auc",
        "insertion_auc",
        "animal_saliency_ratio",
        "background_saliency_ratio",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required advanced-audit detail columns: {sorted(missing)}")
    df = df[df["method"].isin(METHOD_NAMES)].copy()
    numeric_columns = [
        "deletion_auc",
        "insertion_auc",
        "animal_saliency_ratio",
        "background_saliency_ratio",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df


def add_bar_labels(ax: plt.Axes, bars, offset: float = 0.025) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    for bar in bars:
        value = bar.get_height()
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset * span,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["ink"],
        )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.grid(axis=grid_axis, alpha=0.18)


def ordered_perturbations(stress_summary: pd.DataFrame) -> list[str]:
    available = set(stress_summary["perturbation"])
    ordered = [p for p in PERTURBATION_NAMES if p in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def plot_saliency_mass(ax: plt.Axes, advanced: pd.DataFrame) -> None:
    y = np.arange(len(advanced))
    animal = advanced["mean_animal_saliency_ratio"].to_numpy()
    background = advanced["mean_background_saliency_ratio"].to_numpy()
    ax.barh(y, animal, color=PALETTE["teal"], label="animal")
    ax.barh(y, background, left=animal, color=PALETTE["amber"], label="background")
    ax.set_yticks(y, advanced["label"])
    ax.set_xlim(0, 1)
    ax.set_title("Where the saliency mass is located", pad=24)
    ax.set_xlabel("fraction of total saliency")
    for i, (animal_value, background_value) in enumerate(zip(animal, background)):
        ax.text(
            animal_value / 2,
            i,
            f"{animal_value:.2f}",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )
        ax.text(
            animal_value + background_value / 2,
            i,
            f"{background_value:.2f}",
            va="center",
            ha="center",
            color=PALETTE["navy"],
            fontweight="bold",
        )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    style_axis(ax, grid_axis="x")


def plot_stability(ax: plt.Axes, stress_row: pd.DataFrame, perturbation_label: str) -> None:
    x = np.arange(len(stress_row))
    width = 0.38
    iou_bars = ax.bar(
        x - width / 2,
        stress_row["mean_iou"],
        width=width,
        color=PALETTE["blue"],
        label="Top-20 IoU",
    )
    spearman_bars = ax.bar(
        x + width / 2,
        stress_row["mean_spearman"],
        width=width,
        color=PALETTE["teal"],
        label="Spearman",
    )
    ax.set_xticks(x, stress_row["method_label"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Sensitivity stability under {perturbation_label.lower()}")
    ax.set_ylabel("higher is more stable")
    add_bar_labels(ax, iou_bars)
    add_bar_labels(ax, spearman_bars)
    ax.legend(frameon=False)
    style_axis(ax)


def plot_auc(ax: plt.Axes, advanced: pd.DataFrame) -> None:
    x = np.arange(len(advanced))
    width = 0.38
    deletion_bars = ax.bar(
        x - width / 2,
        advanced["mean_deletion_auc"],
        width=width,
        color=PALETTE["red"],
        label="deletion AUC",
    )
    insertion_bars = ax.bar(
        x + width / 2,
        advanced["mean_insertion_auc"],
        width=width,
        color=PALETTE["teal"],
        label="insertion AUC",
    )
    max_value = max(
        0.05,
        float(advanced["mean_deletion_auc"].max()),
        float(advanced["mean_insertion_auc"].max()),
    )
    ax.set_xticks(x, advanced["label"])
    ax.set_ylim(0, max_value * 1.35)
    ax.set_title("Behavioral faithfulness curves")
    ax.set_ylabel("AUC")
    add_bar_labels(ax, deletion_bars, offset=0.02)
    add_bar_labels(ax, insertion_bars, offset=0.02)
    ax.legend(frameon=False)
    style_axis(ax)


def plot_stress_response(ax: plt.Axes, stress_row: pd.DataFrame, perturbation_label: str) -> None:
    x = np.arange(len(stress_row))
    width = 0.38
    iou_bars = ax.bar(
        x - width / 2,
        stress_row["mean_iou"],
        width=width,
        color=PALETTE["blue"],
        label=f"{perturbation_label} IoU",
    )
    change_bars = ax.bar(
        x + width / 2,
        stress_row["prediction_change_rate"],
        width=width,
        color=PALETTE["amber"],
        label="prediction change rate",
    )
    ax.set_xticks(x, stress_row["method_label"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{perturbation_label} stress test")
    ax.set_ylabel("rate / overlap")
    add_bar_labels(ax, iou_bars)
    add_bar_labels(ax, change_bars)
    ax.legend(frameon=False)
    style_axis(ax)


def plot_single_dashboard(
    *,
    perturbation: str,
    stress_summary: pd.DataFrame,
    advanced: pd.DataFrame,
    output_path: Path,
) -> None:
    stress_row = (
        stress_summary[stress_summary["perturbation"].eq(perturbation)]
        .set_index("xai_method")
        .reindex(METHOD_NAMES)
        .reset_index()
    )
    if stress_row["mean_iou"].isna().any():
        raise ValueError(f"Missing metrics for perturbation: {perturbation}")

    perturbation_label = PERTURBATION_NAMES.get(perturbation, perturbation)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.patch.set_facecolor("white")

    plot_saliency_mass(axes[0, 0], advanced)
    plot_stability(axes[0, 1], stress_row, perturbation_label)
    plot_auc(axes[1, 0], advanced)
    plot_stress_response(axes[1, 1], stress_row, perturbation_label)

    fig.suptitle(
        f"Quantitative evidence under {perturbation_label.lower()}",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_method_detail(
    *,
    method: str,
    perturbation: str,
    stress_metrics: pd.DataFrame,
    advanced_detail: pd.DataFrame,
    iou_column: str,
    output_path: Path,
) -> None:
    stress = stress_metrics[
        stress_metrics["perturbation"].eq(perturbation)
        & stress_metrics["xai_method"].eq(method)
    ].copy()
    detail = advanced_detail[advanced_detail["method"].eq(method)].copy()
    stress_for_merge = stress[
        [
            "index",
            iou_column,
            "spearman",
            "original_target_probability",
            "perturbed_target_probability",
            "confidence_drop",
            "prediction_changed",
            "transition",
        ]
    ].rename(
        columns={
            "original_target_probability": "stress_original_target_probability",
            "perturbed_target_probability": "stress_perturbed_target_probability",
        }
    )
    merged = detail.merge(
        stress_for_merge,
        on="index",
        how="inner",
    ).sort_values("index")
    if merged.empty:
        raise ValueError(f"No merged method-detail data for {method}/{perturbation}.")

    labels = merged["true_class"].astype(str).tolist()
    x = np.arange(len(merged))
    width = 0.36
    perturbation_label = PERTURBATION_NAMES.get(perturbation, perturbation)
    method_title = METHOD_TITLES.get(method, method)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.barh(x, merged["animal_saliency_ratio"], color=PALETTE["teal"], label="animal")
    ax.barh(
        x,
        merged["background_saliency_ratio"],
        left=merged["animal_saliency_ratio"],
        color=PALETTE["amber"],
        label="background",
    )
    ax.set_yticks(x, labels)
    ax.set_xlim(0, 1)
    ax.set_title("Saliency mass per inspected image")
    ax.set_xlabel("fraction of total saliency")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="x")

    ax = axes[0, 1]
    deletion_bars = ax.bar(
        x - width / 2,
        merged["deletion_auc"],
        width=width,
        color=PALETTE["red"],
        label="deletion AUC",
    )
    insertion_bars = ax.bar(
        x + width / 2,
        merged["insertion_auc"],
        width=width,
        color=PALETTE["teal"],
        label="insertion AUC",
    )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    max_auc = max(float(merged["deletion_auc"].max()), float(merged["insertion_auc"].max()), 0.05)
    ax.set_ylim(0, max_auc * 1.35)
    ax.set_title("Deletion / insertion faithfulness by image")
    ax.set_ylabel("AUC")
    add_bar_labels(ax, deletion_bars, offset=0.02)
    add_bar_labels(ax, insertion_bars, offset=0.02)
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[1, 0]
    iou_bars = ax.bar(
        x - width / 2,
        merged[iou_column],
        width=width,
        color=PALETTE["blue"],
        label="Top-20 IoU",
    )
    spearman_bars = ax.bar(
        x + width / 2,
        merged["spearman"],
        width=width,
        color=PALETTE["teal"],
        label="Spearman",
    )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Explanation stability under {perturbation_label.lower()}")
    ax.set_ylabel("higher is more stable")
    add_bar_labels(ax, iou_bars)
    add_bar_labels(ax, spearman_bars)
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[1, 1]
    original_bars = ax.bar(
        x - width / 2,
        merged["stress_original_target_probability"],
        width=width,
        color=PALETTE["teal"],
        label="original target probability",
    )
    perturbed_bars = ax.bar(
        x + width / 2,
        merged["stress_perturbed_target_probability"],
        width=width,
        color=PALETTE["amber"],
        label="after intervention",
    )
    for idx, changed in enumerate(merged["prediction_changed"]):
        if changed:
            marker_y = min(
                0.93,
                max(
                    float(merged["stress_original_target_probability"].iloc[idx]),
                    float(merged["stress_perturbed_target_probability"].iloc[idx]),
                )
                + 0.12,
            )
            ax.text(
                idx,
                marker_y,
                "class\nchanged",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["red"],
                fontweight="bold",
            )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Target score before and after stress")
    ax.set_ylabel("target probability")
    add_bar_labels(ax, original_bars)
    add_bar_labels(ax, perturbed_bars)
    ax.legend(frameon=False)
    style_axis(ax)

    fig.suptitle(
        f"{method_title}: attribution behavior under {perturbation_label.lower()}",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def select_case(stress_metrics: pd.DataFrame, perturbation: str) -> pd.DataFrame:
    pool = stress_metrics[stress_metrics["perturbation"].eq(perturbation)].copy()
    if pool.empty:
        raise ValueError(f"No case pool for perturbation: {perturbation}")
    changed = pool[pool["prediction_changed"]].copy()
    selection_pool = changed if not changed.empty else pool
    case_index = selection_pool.groupby("index")["saliency_drift"].mean().idxmax()
    case = (
        pool[pool["index"].eq(case_index)]
        .set_index("xai_method")
        .reindex(METHOD_NAMES)
        .reset_index()
    )
    return case


def plot_case_study(
    *,
    perturbation: str,
    stress_metrics: pd.DataFrame,
    iou_column: str,
    output_path: Path,
) -> None:
    case = select_case(stress_metrics, perturbation)
    base = case.dropna(subset=["true_class"]).iloc[0]
    perturbation_label = PERTURBATION_NAMES.get(perturbation, perturbation)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Selected stress case", color=PALETTE["navy"], fontweight="bold")
    summary = [
        ("true class", str(base["true_class"])),
        ("original prediction", str(base["original_prediction"])),
        ("after intervention", str(base["perturbed_prediction"])),
        ("transition", str(base["transition"])),
    ]
    for i, (key, value) in enumerate(summary):
        y0 = 0.82 - i * 0.19
        patch = FancyBboxPatch(
            (0.02, y0 - 0.075),
            0.96,
            0.13,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            transform=ax.transAxes,
            fc=PALETTE["soft"],
            ec=PALETTE["line"],
        )
        ax.add_patch(patch)
        ax.text(
            0.07,
            y0 + 0.018,
            key.upper(),
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["muted"],
            fontweight="bold",
        )
        ax.text(
            0.07,
            y0 - 0.035,
            value,
            transform=ax.transAxes,
            fontsize=10,
            color=PALETTE["ink"],
            fontweight="bold",
        )

    ax = axes[1]
    values = [base["original_target_probability"], base["perturbed_target_probability"]]
    bars = ax.bar(["original", "perturbed"], values, color=[PALETTE["teal"], PALETTE["amber"]])
    ax.set_ylim(0, 1)
    ax.set_title("Fixed-target score")
    ax.set_ylabel("target probability")
    add_bar_labels(ax, bars)
    style_axis(ax)

    ax = axes[2]
    labels = [METHOD_NAMES[m] for m in METHOD_NAMES]
    x = np.arange(len(case))
    width = 0.36
    iou_bars = ax.bar(
        x - width / 2,
        case[iou_column],
        width=width,
        color=PALETTE["blue"],
        label="Top-20 IoU",
    )
    spearman_bars = ax.bar(
        x + width / 2,
        case["spearman"],
        width=width,
        color=PALETTE["teal"],
        label="Spearman",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Explanation stability for {perturbation_label.lower()}")
    ax.set_ylabel("higher is more stable")
    add_bar_labels(ax, iou_bars)
    add_bar_labels(ax, spearman_bars)
    ax.legend(frameon=False)
    style_axis(ax)

    fig.suptitle(
        f"Case study under {perturbation_label.lower()}",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def copy_to_docs(source: Path, docs_output_dir: Path, asset_name: str) -> None:
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, docs_output_dir / asset_name)


def main() -> None:
    args = parse_args()
    stress_metrics, iou_column = load_stress_metrics(args.metrics_csv)
    stress_summary = summarize_stress_metrics(stress_metrics, iou_column)
    advanced_summary = load_advanced_summary(args.advanced_summary_csv)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    stress_summary.to_csv(args.summary_output, index=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_output_dir.mkdir(parents=True, exist_ok=True)

    for perturbation in ordered_perturbations(stress_summary):
        perturbation_slug = PERTURBATION_SLUGS.get(perturbation, perturbation.replace("_", "-"))

        dashboard_output = args.output_dir / f"stress_metric_dashboard_{perturbation_slug}.png"
        plot_single_dashboard(
            perturbation=perturbation,
            stress_summary=stress_summary,
            advanced=advanced_summary,
            output_path=dashboard_output,
        )
        copy_to_docs(
            dashboard_output,
            args.docs_output_dir,
            f"stress-metric-{perturbation_slug}.png",
        )
        print(f"saved dashboard: {dashboard_output}")

        case_output = args.output_dir / f"stress_metric_{perturbation_slug}_case.png"
        plot_case_study(
            perturbation=perturbation,
            stress_metrics=stress_metrics,
            iou_column=iou_column,
            output_path=case_output,
        )
        copy_to_docs(
            case_output,
            args.docs_output_dir,
            f"stress-metric-{perturbation_slug}-case.png",
        )
        print(f"saved case study: {case_output}")

    print(f"saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
