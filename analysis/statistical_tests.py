#!/usr/bin/env python3
"""
Statistical analysis for CFC experiment.

Tests:
  1. One-way ANOVA across conditions for each DV
  2. Factorial interaction analysis (superadditivity test)
  3. Trust Gate prerequisite test (H3)
  4. FULL vs INFO_MATCHED comparison
  5. Effect sizes (Cohen's d) with Bonferroni correction
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent

DVS = ["shannon_entropy", "hedging_index", "cdrc", "bridging_count", "gns", "cdi"]
DV_LABELS = {
    "shannon_entropy": "Shannon Entropy (H)",
    "hedging_index": "Hedging Index (HI)",
    "cdrc": "Domain Count (CDRC)",
    "bridging_count": "Bridging Count (BC)",
    "gns": "Generative Novelty (GNS)",
    "cdi": "Collaboration Depth (CDI)",
}


def load_metrics(metrics_path: Path = None) -> pd.DataFrame:
    """Load processed metrics into a DataFrame."""
    if metrics_path is None:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        files = sorted(processed_dir.glob("metrics_*.json"))
        if not files:
            print("No metrics files found. Run compute_metrics.py first.")
            sys.exit(1)
        metrics_path = files[-1]

    with open(metrics_path) as f:
        data = json.load(f)

    return pd.DataFrame(data)


def cohens_d(group1, group2) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def test_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA across all conditions for each DV."""
    results = {}
    conditions = df["condition"].unique()

    for dv in DVS:
        groups = [df[df["condition"] == c][dv].dropna().values for c in conditions]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        f_stat, p_val = stats.f_oneway(*groups)
        # Eta-squared effect size
        grand_mean = df[dv].dropna().mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        results[dv] = {
            "F": float(f_stat),
            "p": float(p_val),
            "eta_squared": float(eta_sq),
            "n_conditions": len(groups),
            "significant": p_val < 0.05,
        }

    return results


def test_superadditivity(df: pd.DataFrame) -> dict:
    """Test H2: Are ring interactions superadditive?

    Test: effect(TQ) > effect(T) + effect(Q)
    Where effect(X) = mean(X) - mean(BASELINE)
    """
    results = {}
    baseline = df[df["condition"] == "BASELINE"]

    pairs = [
        ("TQ", "T", "Q"),
        ("TB", "T", "B"),
        ("QB", "Q", "B"),
        ("TQB", "TQ", "B"),
    ]

    for dv in DVS:
        base_mean = baseline[dv].dropna().mean()
        dv_results = []

        for combo, a, b in pairs:
            combo_vals = df[df["condition"] == combo][dv].dropna()
            a_vals = df[df["condition"] == a][dv].dropna()
            b_vals = df[df["condition"] == b][dv].dropna()

            if combo_vals.empty or a_vals.empty or b_vals.empty:
                continue

            combo_effect = combo_vals.mean() - base_mean
            a_effect = a_vals.mean() - base_mean
            b_effect = b_vals.mean() - base_mean
            additive_prediction = a_effect + b_effect
            superadditive = combo_effect > additive_prediction

            dv_results.append({
                "combination": combo,
                "components": f"{a}+{b}",
                "combo_effect": float(combo_effect),
                "additive_prediction": float(additive_prediction),
                "superadditive_ratio": float(combo_effect / additive_prediction) if additive_prediction != 0 else float("nan"),
                "is_superadditive": bool(superadditive),
            })

        results[dv] = dv_results

    return results


def test_trust_gate_prerequisite(df: pd.DataFrame) -> dict:
    """Test H3: Is Trust Gate a prerequisite for other rings?

    Compare: B vs TB, Q vs TQ, S vs TS (if available)
    If Trust Gate amplifies the effect, TB > B and TQ > Q.
    """
    results = {}
    comparisons = [
        ("Q", "TQ", "Trust amplifies Quantitative"),
        ("B", "TB", "Trust amplifies Bridges"),
    ]

    for dv in DVS:
        dv_results = []
        for without_trust, with_trust, label in comparisons:
            wt = df[df["condition"] == with_trust][dv].dropna()
            wo = df[df["condition"] == without_trust][dv].dropna()

            if wt.empty or wo.empty:
                continue

            t_stat, p_val = stats.ttest_ind(wt, wo)
            d = cohens_d(wt.values, wo.values)

            dv_results.append({
                "comparison": f"{with_trust} vs {without_trust}",
                "label": label,
                "with_trust_mean": float(wt.mean()),
                "without_trust_mean": float(wo.mean()),
                "t": float(t_stat),
                "p": float(p_val),
                "cohens_d": float(d),
                "trust_amplifies": bool(wt.mean() > wo.mean()),
            })

        results[dv] = dv_results

    return results


def test_structure_matters(df: pd.DataFrame) -> dict:
    """Test: FULL vs INFO_MATCHED.

    Same content, different structure. If FULL > INFO_MATCHED,
    structure is a causal variable.
    """
    results = {}
    full = df[df["condition"] == "FULL"]
    info = df[df["condition"] == "INFO_MATCHED"]

    for dv in DVS:
        full_vals = full[dv].dropna()
        info_vals = info[dv].dropna()

        if full_vals.empty or info_vals.empty:
            continue

        t_stat, p_val = stats.ttest_ind(full_vals, info_vals)
        d = cohens_d(full_vals.values, info_vals.values)

        results[dv] = {
            "full_mean": float(full_vals.mean()),
            "full_std": float(full_vals.std()),
            "info_matched_mean": float(info_vals.mean()),
            "info_matched_std": float(info_vals.std()),
            "t": float(t_stat),
            "p": float(p_val),
            "cohens_d": float(d),
            "structure_matters": bool(p_val < 0.05),
            "direction": "FULL > INFO" if full_vals.mean() > info_vals.mean() else "INFO >= FULL",
        }

    return results


def pairwise_effects(df: pd.DataFrame) -> dict:
    """Compute pairwise Cohen's d for all condition pairs."""
    conditions = sorted(df["condition"].unique())
    n_comparisons = len(list(combinations(conditions, 2)))
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    results = {}
    for dv in DVS:
        dv_pairs = []
        for c1, c2 in combinations(conditions, 2):
            g1 = df[df["condition"] == c1][dv].dropna().values
            g2 = df[df["condition"] == c2][dv].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue

            t_stat, p_val = stats.ttest_ind(g1, g2)
            d = cohens_d(g1, g2)
            dv_pairs.append({
                "pair": f"{c1} vs {c2}",
                "cohens_d": float(d),
                "p": float(p_val),
                "significant_bonferroni": p_val < bonferroni_alpha,
            })

        results[dv] = {
            "bonferroni_alpha": bonferroni_alpha,
            "n_comparisons": n_comparisons,
            "pairs": sorted(dv_pairs, key=lambda x: abs(x["cohens_d"]), reverse=True),
        }

    return results


def run_all_tests(df: pd.DataFrame) -> dict:
    """Run the complete analysis battery."""
    print("=" * 60)
    print("  CONTEXT FIELD CONDITIONING — Statistical Analysis")
    print("=" * 60)

    results = {}

    # 1. ANOVA
    print("\n--- One-way ANOVA ---")
    anova = test_anova(df)
    for dv, r in anova.items():
        sig = "*" if r["significant"] else ""
        print(f"  {DV_LABELS[dv]:30s}  F={r['F']:.3f}  p={r['p']:.4f}  η²={r['eta_squared']:.3f} {sig}")
    results["anova"] = anova

    # 2. Superadditivity
    print("\n--- Superadditivity Test (H2) ---")
    superadd = test_superadditivity(df)
    for dv, pairs in superadd.items():
        for p in pairs:
            mark = "✓" if p["is_superadditive"] else "✗"
            print(f"  {DV_LABELS[dv]:30s}  {p['combination']} vs {p['components']:6s} "
                  f"ratio={p['superadditive_ratio']:.2f} {mark}")
    results["superadditivity"] = superadd

    # 3. Trust Gate prerequisite
    print("\n--- Trust Gate Prerequisite (H3) ---")
    trust = test_trust_gate_prerequisite(df)
    for dv, comps in trust.items():
        for c in comps:
            mark = "✓" if c["trust_amplifies"] else "✗"
            print(f"  {DV_LABELS[dv]:30s}  {c['comparison']:12s}  d={c['cohens_d']:.3f}  p={c['p']:.4f} {mark}")
    results["trust_gate"] = trust

    # 4. Structure matters
    print("\n--- FULL vs INFO_MATCHED (Structure Test) ---")
    structure = test_structure_matters(df)
    for dv, r in structure.items():
        sig = "*" if r["structure_matters"] else ""
        print(f"  {DV_LABELS[dv]:30s}  d={r['cohens_d']:.3f}  p={r['p']:.4f}  {r['direction']} {sig}")
    results["structure_test"] = structure

    # 5. Pairwise effects
    print("\n--- Pairwise Effect Sizes (top 5 per DV) ---")
    pairwise = pairwise_effects(df)
    for dv, r in pairwise.items():
        top5 = r["pairs"][:5]
        print(f"  {DV_LABELS[dv]}:")
        for p in top5:
            sig = "*" if p["significant_bonferroni"] else ""
            print(f"    {p['pair']:30s}  d={p['cohens_d']:.3f}  p={p['p']:.4f} {sig}")
    results["pairwise"] = pairwise

    return results


if __name__ == "__main__":
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    df = load_metrics(metrics_path)

    print(f"Loaded {len(df)} trials across {df['condition'].nunique()} conditions")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print()

    results = run_all_tests(df)

    # Save full results
    output_dir = PROJECT_ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statistical_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFull results saved to {output_path}")
