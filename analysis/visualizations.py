#!/usr/bin/env python3
"""
Visualization suite for CFC experiment.

Generates:
  1. Condition comparison heatmap (conditions × DVs)
  2. Interaction plots (ring × ring effects)
  3. Radar charts (per-condition DV profiles)
  4. FULL vs INFO_MATCHED comparison
  5. Trust Gate amplification chart
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent

DVS = ["shannon_entropy", "hedging_index", "cdrc", "bridging_count", "gns", "cdi"]
DV_LABELS = {
    "shannon_entropy": "Entropy (H)",
    "hedging_index": "Hedging (HI)",
    "cdrc": "Domains (CDRC)",
    "bridging_count": "Bridging (BC)",
    "gns": "Novelty (GNS)",
    "cdi": "Collaboration (CDI)",
}

# Ordered by ring inclusion
CONDITION_ORDER = [
    "BASELINE", "T", "Q", "B", "S",
    "TQ", "TB", "QB", "TQB", "FULL", "INFO_MATCHED",
]


def load_metrics(metrics_path: Path = None) -> pd.DataFrame:
    if metrics_path is None:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        files = sorted(processed_dir.glob("metrics_*.json"))
        if not files:
            print("No metrics files found.")
            sys.exit(1)
        metrics_path = files[-1]

    with open(metrics_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def setup_style():
    """Configure matplotlib for clean, publication-quality figures."""
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#0a0a0a",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "axes.edgecolor": "#333333",
        "grid.color": "#222222",
        "font.family": "monospace",
        "font.size": 10,
        "figure.dpi": 150,
    })


def plot_condition_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap: conditions × DVs with z-scored values."""
    # Compute means per condition
    means = df.groupby("condition")[DVS].mean()

    # Reorder conditions
    ordered = [c for c in CONDITION_ORDER if c in means.index]
    means = means.loc[ordered]

    # Z-score for comparability across DVs
    z_means = (means - means.mean()) / means.std()
    z_means.columns = [DV_LABELS[c] for c in z_means.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(z_means, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "z-score (relative to condition means)"})
    ax.set_title("Context Field Conditioning — Condition × DV Heatmap", fontsize=13, pad=15)
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "condition_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  condition_heatmap.png")


def plot_radar_charts(df: pd.DataFrame, output_dir: Path):
    """Radar chart comparing key conditions."""
    conditions = ["BASELINE", "T", "FULL", "INFO_MATCHED"]
    conditions = [c for c in conditions if c in df["condition"].values]

    means = df.groupby("condition")[DVS].mean()

    # Normalize each DV to 0-1 range across all conditions
    mins = means.min()
    maxs = means.max()
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (means - mins) / ranges

    angles = np.linspace(0, 2 * np.pi, len(DVS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#666666", "#e74c3c", "#2ecc71", "#3498db"]
    for i, cond in enumerate(conditions):
        if cond not in normed.index:
            continue
        values = normed.loc[cond].values.tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=cond, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    labels = [DV_LABELS[dv] for dv in DVS]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("DV Profiles by Condition", pad=20, fontsize=13)

    plt.tight_layout()
    fig.savefig(output_dir / "radar_profiles.png", bbox_inches="tight")
    plt.close(fig)
    print("  radar_profiles.png")


def plot_structure_comparison(df: pd.DataFrame, output_dir: Path):
    """Side-by-side: FULL vs INFO_MATCHED for each DV."""
    full = df[df["condition"] == "FULL"]
    info = df[df["condition"] == "INFO_MATCHED"]

    if full.empty or info.empty:
        print("  (skipping structure comparison — missing FULL or INFO_MATCHED)")
        return

    fig, axes = plt.subplots(1, len(DVS), figsize=(16, 5))

    for i, dv in enumerate(DVS):
        ax = axes[i]
        data = pd.DataFrame({
            "FULL": full[dv].dropna().values[:min(len(full), len(info))],
            "INFO\nMATCHED": info[dv].dropna().values[:min(len(full), len(info))],
        })
        data.plot.box(ax=ax, color={"medians": "#e74c3c"})
        ax.set_title(DV_LABELS[dv], fontsize=10)
        ax.set_ylabel("")

    fig.suptitle("Structure Test: FULL (honeycomb) vs INFO_MATCHED (flat)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "structure_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  structure_comparison.png")


def plot_trust_gate_amplification(df: pd.DataFrame, output_dir: Path):
    """Bar chart: effect of adding Trust Gate to each ring."""
    pairs = [("Q", "TQ"), ("B", "TB")]
    baseline_mean = df[df["condition"] == "BASELINE"][DVS].mean()

    fig, axes = plt.subplots(1, len(DVS), figsize=(16, 5))

    for i, dv in enumerate(DVS):
        ax = axes[i]
        base = baseline_mean[dv]

        for without, with_trust in pairs:
            wo_mean = df[df["condition"] == without][dv].dropna().mean() - base
            wt_mean = df[df["condition"] == with_trust][dv].dropna().mean() - base

            x = pairs.index((without, with_trust))
            ax.bar(x * 3, wo_mean, color="#888888", width=0.8, label=without if i == 0 else "")
            ax.bar(x * 3 + 1, wt_mean, color="#e74c3c", width=0.8, label=with_trust if i == 0 else "")

        ax.set_title(DV_LABELS[dv], fontsize=10)
        ax.set_xticks([0.5, 3.5])
        ax.set_xticklabels(["Q→TQ", "B→TB"])
        ax.axhline(y=0, color="#444", linewidth=0.5)

    fig.suptitle("Trust Gate Amplification: Does Ring 1 Boost Other Rings?",
                 fontsize=13, y=1.02)
    axes[0].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "trust_gate_amplification.png", bbox_inches="tight")
    plt.close(fig)
    print("  trust_gate_amplification.png")


def plot_category_breakdown(df: pd.DataFrame, output_dir: Path):
    """Heatmap: condition × probe category for entropy."""
    pivot = df.pivot_table(
        values="shannon_entropy",
        index="condition",
        columns="probe_category",
        aggfunc="mean",
    )

    ordered = [c for c in CONDITION_ORDER if c in pivot.index]
    pivot = pivot.loc[ordered]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                linewidths=0.5, ax=ax)
    ax.set_title("Shannon Entropy: Condition × Probe Category", fontsize=13, pad=15)
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "entropy_by_category.png", bbox_inches="tight")
    plt.close(fig)
    print("  entropy_by_category.png")


def generate_all(df: pd.DataFrame, output_dir: Path):
    """Generate all visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Generating visualizations...")
    plot_condition_heatmap(df, output_dir)
    plot_radar_charts(df, output_dir)
    plot_structure_comparison(df, output_dir)
    plot_trust_gate_amplification(df, output_dir)
    plot_category_breakdown(df, output_dir)
    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    df = load_metrics(metrics_path)
    output_dir = PROJECT_ROOT / "analysis" / "results"
    generate_all(df, output_dir)
