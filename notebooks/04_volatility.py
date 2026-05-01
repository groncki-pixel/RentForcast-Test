"""
04_volatility.py
================
Measure how much each of the 40 sub-rankings shifted between the original
and May 2026 model vintages. Group by category to see which inputs are
volatile enough to be unreliable for ranking.

Outputs:
  outputs/reports/volatility_scorecard.csv
  outputs/reports/volatility_by_category.csv
  outputs/figures/volatility_by_category.png
  outputs/figures/volatility_by_subcolumn.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Reuse the same column-> category mapping as script 2
SUB_COLS = [
    ("AskingRent_Forward1", "AskingRent"), ("AskingRent_Forward3", "AskingRent"),
    ("AskingRent_Forward5", "AskingRent"), ("AskingRent_Forward10", "AskingRent"),
    ("EffectiveRent_Forward1", "EffectiveRent"),
    ("EffectiveRent_Forward3", "EffectiveRent"),
    ("EffectiveRent_Forward5", "EffectiveRent"),
    ("EffectiveRent_Forward10", "EffectiveRent"),
    ("Employment_Hist10", "Employment"), ("Employment_Hist5", "Employment"),
    ("Employment_Forward1", "Employment"), ("Employment_Forward3", "Employment"),
    ("Employment_Forward5", "Employment"), ("Employment_Forward10", "Employment"),
    ("Supply_Hist5", "Supply"), ("Supply_Forward1", "Supply"),
    ("Supply_Forward3", "Supply"), ("Supply_Forward5", "Supply"),
    ("Supply_Forward10", "Supply"),
    ("UPP_Hist10", "UPP"), ("UPP_Forward5", "UPP"), ("UPP_Forward10", "UPP"),
    ("Income_Forward1", "Income"), ("Income_Forward3", "Income"),
    ("Income_Forward5", "Income"), ("Income_Forward10", "Income"),
    ("Population_Hist10", "Population"), ("Population_Hist5", "Population"),
    ("Population_Forward1", "Population"),
    ("Population_Forward3", "Population"),
    ("Population_Forward5", "Population"),
    ("Population_Forward10", "Population"),
    ("SalePrice_Hist1", "SalePrice"), ("SalePrice_Hist3", "SalePrice"),
    ("SalePrice_Hist5", "SalePrice"), ("SalePrice_Hist10", "SalePrice"),
    ("SalePrice_Forward1", "SalePrice"), ("SalePrice_Forward3", "SalePrice"),
    ("SalePrice_Forward5", "SalePrice"), ("SalePrice_Forward10", "SalePrice"),
]


def main() -> None:
    print("=" * 70)
    print("SCRIPT 4: VOLATILITY")
    print("=" * 70)

    orig = pd.read_csv(os.path.join(PROCESSED, "summary_original.csv"))
    may = pd.read_csv(os.path.join(PROCESSED, "summary_may2026.csv"))
    df = orig.merge(may, on="Market", suffixes=("_o", "_m"))

    rows = []
    for sub, cat in SUB_COLS:
        diff = df[f"{sub}_m"] - df[f"{sub}_o"]
        rows.append({
            "SubColumn": sub,
            "Category": cat,
            "MeanAbsRankChange": diff.abs().mean(),
            "StdRankChange": diff.std(),
            "MedianAbsRankChange": diff.abs().median(),
            "P90AbsRankChange": diff.abs().quantile(0.9),
            "MaxAbsRankChange": diff.abs().max(),
            "PctMarketsMoved10plus": (diff.abs() >= 10).mean() * 100,
        })
    score = pd.DataFrame(rows).sort_values("MeanAbsRankChange", ascending=False)
    score.to_csv(os.path.join(REPORTS, "volatility_scorecard.csv"), index=False)
    print(f"  Wrote volatility_scorecard.csv ({len(score)} sub-columns)")

    # ---- By-category aggregate ---------------------------------------------
    by_cat = (score.groupby("Category")
              .agg(MeanAbsRankChange=("MeanAbsRankChange", "mean"),
                   StdRankChange=("StdRankChange", "mean"),
                   P90AbsRankChange=("P90AbsRankChange", "mean"),
                   PctMarketsMoved10plus=("PctMarketsMoved10plus", "mean"),
                   N_subcols=("SubColumn", "count"))
              .reset_index()
              .sort_values("MeanAbsRankChange", ascending=False))
    by_cat.to_csv(os.path.join(REPORTS, "volatility_by_category.csv"), index=False)

    # ---- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_cat["Category"], by_cat["MeanAbsRankChange"], color="#264653")
    ax.set_ylabel("Mean |rank change| (across all sub-columns and markets)")
    ax.set_title("Ranking volatility by category, original vs May 2026")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "volatility_by_category.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    score_sorted = score.sort_values("MeanAbsRankChange")
    palette = {"AskingRent": "#264653", "EffectiveRent": "#e76f51",
               "Employment": "#2a9d8f", "Supply": "#f4a261",
               "UPP": "#8e7dbe", "Income": "#43aa8b",
               "Population": "#577590", "SalePrice": "#bc4749"}
    colors = [palette.get(c, "grey") for c in score_sorted["Category"]]
    ax.barh(score_sorted["SubColumn"], score_sorted["MeanAbsRankChange"],
            color=colors)
    ax.set_xlabel("Mean |rank change|")
    ax.set_title("Ranking volatility by sub-column (color = category)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l)
               for l, c in palette.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "volatility_by_subcolumn.png"), dpi=140)
    plt.close(fig)
    print(f"  Wrote 2 figures.")

    # ---- Print findings -----------------------------------------------------
    print("\n" + "=" * 70)
    print("VOLATILITY FINDINGS")
    print("=" * 70)
    print("\n  By category (mean |rank change|):")
    for _, r in by_cat.iterrows():
        print(f"    {r['Category']:14s} mean|d|={r['MeanAbsRankChange']:5.2f}  "
              f"std={r['StdRankChange']:5.2f}  "
              f"% markets moved 10+={r['PctMarketsMoved10plus']:5.1f}%")

    print("\n  Top 10 most volatile sub-rankings:")
    for _, r in score.head(10).iterrows():
        print(f"    {r['SubColumn']:25s} ({r['Category']:13s}) "
              f"mean|d|={r['MeanAbsRankChange']:5.2f} "
              f"std={r['StdRankChange']:5.2f}")

    print("\n  Most stable sub-rankings (bottom 10):")
    for _, r in score.tail(10).iloc[::-1].iterrows():
        print(f"    {r['SubColumn']:25s} ({r['Category']:13s}) "
              f"mean|d|={r['MeanAbsRankChange']:5.2f}")

    er_vol = by_cat.loc[by_cat["Category"] == "EffectiveRent",
                        "MeanAbsRankChange"].iloc[0]
    pop_vol = by_cat.loc[by_cat["Category"] == "Population",
                         "MeanAbsRankChange"].iloc[0]
    emp_vol = by_cat.loc[by_cat["Category"] == "Employment",
                         "MeanAbsRankChange"].iloc[0]
    sp_vol = by_cat.loc[by_cat["Category"] == "SalePrice",
                        "MeanAbsRankChange"].iloc[0]
    print(f"\n  Volatility comparison:")
    print(f"    EffectiveRent: {er_vol:.2f}  (the variable in question)")
    print(f"    SalePrice    : {sp_vol:.2f}")
    print(f"    Employment   : {emp_vol:.2f}")
    print(f"    Population   : {pop_vol:.2f}")
    if er_vol > 2 * pop_vol or er_vol > 2 * emp_vol:
        print("    => Effective Rent IS materially more volatile than employment / "
              "population. Hypothesis SUPPORTED — consider down-weighting or "
              "smoothing.")
    else:
        print("    => Effective Rent is comparable in volatility to other forward variables.")

    print("\n  Recommendation: rank with high confidence on the most stable "
          "categories (Employment, Population) and treat the noisy categories "
          "(EffectiveRent forward, AskingRent, Income forward) with caution.")
    print("\nDone.")


if __name__ == "__main__":
    main()
