'''
Plot per-feature histograms split by severity, side by side.

Input:
- CSV table produced by graph_feature_table.py

Output:
- One PNG per numeric feature.
- Each PNG contains subplots, one histogram per severity level.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCLUDE_COLUMNS = {"image_name", "severity_score"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot feature histograms by severity level.")
    parser.add_argument("--table-csv", type=Path, required=True, help="Path to feature table CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save histogram images")
    parser.add_argument("--bins", type=int, default=12, help="Number of bins per histogram")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of feature columns to plot. If omitted, all numeric feature columns are used.",
    )
    return parser.parse_args()


def severity_label(v: float) -> str:
    if np.isfinite(v) and abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}"
    return f"{v:.2f}"


def discover_feature_columns(df: pd.DataFrame, selected: list[str] | None) -> list[str]:
    if selected:
        missing = [c for c in selected if c not in df.columns]
        if missing:
            raise ValueError(f"Requested features not in table: {missing}")
        return selected

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]
    if not features:
        raise ValueError("No numeric feature columns found to plot.")
    return features


def plot_feature_by_severity(
    df: pd.DataFrame,
    feature: str,
    severities: list[float],
    bins: int,
    output_path: Path,
) -> None:
    n = len(severities)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    feature_values = df[feature].dropna().to_numpy(dtype=float)
    if feature_values.size == 0:
        plt.close(fig)
        return

    vmin = float(np.min(feature_values))
    vmax = float(np.max(feature_values))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    severity_total_counts = df.groupby("severity_score").size().to_dict()

    for ax, sev in zip(axes, severities):
        vals = df.loc[df["severity_score"] == sev, feature].dropna().to_numpy(dtype=float)
        total_n = int(severity_total_counts.get(sev, 0))
        valid_n = int(len(vals))
        ax.hist(vals, bins=bin_edges, color="#2a9d8f", edgecolor="black", alpha=0.85)
        ax.set_title(
            f"Severity {severity_label(sev)}\n"
            f"images={total_n}, valid={valid_n}"
        )
        ax.set_xlabel(feature)
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{feature} by Severity", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table_csv)

    if "severity_score" not in df.columns:
        raise ValueError("Input table must contain 'severity_score' column.")

    df = df.copy()
    df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce")
    df = df.dropna(subset=["severity_score"])

    severities = sorted(df["severity_score"].unique().tolist())
    features = discover_feature_columns(df, args.features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        output_path = args.output_dir / f"hist_{feature}.png"
        plot_feature_by_severity(df, feature, severities, args.bins, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
