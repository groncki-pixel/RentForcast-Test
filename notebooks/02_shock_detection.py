"""
02_shock_detection.py
=====================
Compare original vs May2026 Summary Sheet rankings. Identify shocked markets
(|rank change| >= 8) and decompose what category drove the change.

Inputs (from data/processed/):
  - summary_original.csv
  - summary_may2026.csv
  - weights_original.csv

Outputs:
  - outputs/reports/shock_detection.csv          (per-market table)
  - outputs/reports/shock_category_decomposition.csv
  - outputs/figures/shock_heatmap.png
  - outputs/figures/shock_movers_bar.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

CATEGORY_COLS = {
    "AskingRent": ["AskingRent_Forward1", "AskingRent_Forward3",
                   "AskingRent_Forward5", "AskingRent_Forward10"],
    "EffectiveRent": ["EffectiveRent_Forward1", "EffectiveRent_Forward3",
                      "EffectiveRent_Forward5", "EffectiveRent_Forward10"],
    "Employment": ["Employment_Hist10", "Employment_Hist5",
                   "Employment_Forward1", "Employment_Forward3",
                   "Employment_Forward5", "Employment_Forward10"],
    "Supply": ["Supply_Hist5", "Supply_Forward1", "Supply_Forward3",
               "Supply_Forward5", "Supply_Forward10"],
    "UPP": ["UPP_Hist10", "UPP_Forward5", "UPP_Forward10"],
    "Income": ["Income_Forward1", "Income_Forward3",
               "Income_Forward5", "Income_Forward10"],
    "Population": ["Population_Hist10", "Population_Hist5",
                   "Population_Forward1", "Population_Forward3",
                   "Population_Forward5", "Population_Forward10"],
    "SalePrice": ["SalePrice_Hist1", "SalePrice_Hist3",
                  "SalePrice_Hist5", "SalePrice_Hist10",
                  "SalePrice_Forward1", "SalePrice_Forward3",
                  "SalePrice_Forward5", "SalePrice_Forward10"],
}
ALL_SUB_COLS = sum(CATEGORY_COLS.values(), [])


def main() -> None:
    print("=" * 70)
    print("SCRIPT 2: SHOCK DETECTION")
    print("=" * 70)

    orig = pd.read_csv(os.path.join(PROCESSED, "summary_original.csv"))
    may = pd.read_csv(os.path.join(PROCESSED, "summary_may2026.csv"))
    weights = pd.read_csv(os.path.join(PROCESSED, "weights_original.csv"),
                          index_col=0).iloc[:, 0].to_dict()

    merged = orig.merge(may, on="Market", suffixes=("_orig", "_may"))
    print(f"  Merged on Market: {len(merged)} rows")

    merged["RankChange"] = merged["FinalRank_may"] - merged["FinalRank_orig"]
    merged["WeightedAvgChange"] = (merged["WeightedAverage_may"]
                                   - merged["WeightedAverage_orig"])
    merged["Shocked"] = merged["RankChange"].abs() >= 8

    # Category decomposition: average sub-ranking change per category
    cat_change_cols = []
    for cat, cols in CATEGORY_COLS.items():
        diff_vals = pd.DataFrame()
        for c in cols:
            diff_vals[c] = merged[f"{c}_may"] - merged[f"{c}_orig"]
        merged[f"{cat}_avgRankChange"] = diff_vals.mean(axis=1)
        cat_change_cols.append(f"{cat}_avgRankChange")

    # Score-change decomposition (% of WeightedAvg change attributable to category)
    # Each cell in summary is a rank 1-150; weighted_avg = sum(weight * mean(rank))
    # Per-category contribution to WA change ≈ weight * delta(mean_cat_rank)
    contrib_cols = []
    for cat, cols in CATEGORY_COLS.items():
        w = weights.get(cat, 0.0)
        merged[f"{cat}_contribution"] = w * merged[f"{cat}_avgRankChange"]
        contrib_cols.append(f"{cat}_contribution")
    merged["TotalContribution"] = merged[contrib_cols].sum(axis=1)

    # Percent share — guard against zero denom
    for cat in CATEGORY_COLS:
        denom = merged["TotalContribution"].replace(0, np.nan)
        merged[f"{cat}_share_pct"] = (
            merged[f"{cat}_contribution"] / denom * 100.0
        )

    # ---- Save per-market shock table ---------------------------------------
    out_cols = (
        ["Market", "FinalRank_orig", "FinalRank_may", "RankChange",
         "WeightedAverage_orig", "WeightedAverage_may", "WeightedAvgChange",
         "Shocked"]
        + cat_change_cols + contrib_cols
        + [f"{c}_share_pct" for c in CATEGORY_COLS]
    )
    shock_table = merged[out_cols].copy().sort_values(
        "RankChange", key=lambda s: s.abs(), ascending=False)
    shock_table.to_csv(os.path.join(REPORTS, "shock_detection.csv"), index=False)
    print(f"  Wrote shock_detection.csv ({len(shock_table)} rows)")

    # ---- Save category decomposition aggregate -----------------------------
    decomp = pd.DataFrame({
        "Category": list(CATEGORY_COLS.keys()),
        "Weight": [weights.get(c, 0.0) for c in CATEGORY_COLS],
        "MeanAbsRankChange": [
            merged[f"{c}_avgRankChange"].abs().mean() for c in CATEGORY_COLS
        ],
        "StdRankChange": [
            merged[f"{c}_avgRankChange"].std() for c in CATEGORY_COLS
        ],
        "MeanContribution": [
            merged[f"{c}_contribution"].abs().mean() for c in CATEGORY_COLS
        ],
    })
    decomp.to_csv(os.path.join(REPORTS, "shock_category_decomposition.csv"),
                  index=False)
    print(f"  Wrote shock_category_decomposition.csv")

    # ---- Heatmap of top 30 movers ------------------------------------------
    top30 = shock_table.head(30).set_index("Market")
    heat = top30[cat_change_cols].copy()
    heat.columns = [c.replace("_avgRankChange", "") for c in heat.columns]
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        heat, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
        cbar_kws={"label": "Avg sub-ranking change (positive = worse)"},
        ax=ax,
    )
    ax.set_title("Top 30 Markets by |Rank Change|: avg sub-ranking change by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Market")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "shock_heatmap.png"), dpi=140)
    plt.close(fig)
    print(f"  Wrote shock_heatmap.png")

    # ---- Bar chart of biggest movers ---------------------------------------
    risers = shock_table.nsmallest(15, "RankChange")
    fallers = shock_table.nlargest(15, "RankChange")
    movers = pd.concat([risers, fallers]).sort_values("RankChange")
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#2a9d8f" if v < 0 else "#e76f51" for v in movers["RankChange"]]
    ax.barh(movers["Market"], movers["RankChange"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Rank change (negative = improved, positive = worsened)")
    ax.set_title("Biggest ranking movers: 15 risers (green) and 15 fallers (red)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "shock_movers_bar.png"), dpi=140)
    plt.close(fig)
    print(f"  Wrote shock_movers_bar.png")

    # ---- Print key findings ------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    n_shocked = int(merged["Shocked"].sum())
    print(f"  Markets with |rank change| >= 8: {n_shocked} of {len(merged)}")
    print(f"  Mean |rank change|: {merged['RankChange'].abs().mean():.1f}")
    print(f"  Max |rank change|: {merged['RankChange'].abs().max():.0f}")
    print("\n  Top 10 risers (largest rank improvement):")
    for _, r in shock_table.nsmallest(10, "RankChange").iterrows():
        print(f"    {r['Market']:30s}  {r['FinalRank_orig']:>3.0f} -> "
              f"{r['FinalRank_may']:>3.0f}  ({r['RankChange']:+.0f})")
    print("\n  Top 10 fallers (largest rank deterioration):")
    for _, r in shock_table.nlargest(10, "RankChange").iterrows():
        print(f"    {r['Market']:30s}  {r['FinalRank_orig']:>3.0f} -> "
              f"{r['FinalRank_may']:>3.0f}  ({r['RankChange']:+.0f})")
    print("\n  Average |sub-rank change| by category:")
    for _, r in decomp.sort_values("MeanAbsRankChange", ascending=False).iterrows():
        print(f"    {r['Category']:14s} weight={r['Weight']:.2f}  "
              f"mean|delta|={r['MeanAbsRankChange']:5.2f}  "
              f"contribution={r['MeanContribution']:5.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
