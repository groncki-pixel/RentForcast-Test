"""
05_weight_sensitivity.py
========================
Re-run the weighted average using May 2026 data under several alternative
weight schemes. Compare each scheme's top 20 against the baseline.

Schemes:
  a) baseline           — original weights {EffRent .2, UPP .2, SalePrice .2,
                          Employment .1, Population .1, Supply 0, Income 0,
                          AskingRent 0}
  b) equal              — 0.125 to each of the 8 categories
  c) drop_effective_rent — redistribute the 0.20 weight on EffRent
                          proportionally to the other non-zero categories
  d) double_employment  — set Employment 0.20, scale others to keep sum = 1
  e) inverse_volatility — weight ∝ 1 / volatility from script 4
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


def category_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean rank per category per market.

    Uses skipna so partially-missing categories (a few markets have NaN in
    long-horizon Population/UPP forwards) still produce a defined mean.
    """
    out = pd.DataFrame({"Market": df["Market"]})
    for cat, cols in CATEGORY_COLS.items():
        out[cat] = df[cols].mean(axis=1, skipna=True)
    return out


def apply_weights(cat_means: pd.DataFrame, weights: dict) -> pd.DataFrame:
    s = pd.Series(0.0, index=cat_means.index)
    for cat, w in weights.items():
        s = s + w * cat_means[cat]
    out = pd.DataFrame({"Market": cat_means["Market"], "WeightedAvg": s})
    out["Rank"] = out["WeightedAvg"].rank(method="min").astype("Int64")
    return out.sort_values("Rank", na_position="last")


def normalize(weights: dict) -> dict:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def main() -> None:
    print("=" * 70)
    print("SCRIPT 5: WEIGHT SENSITIVITY")
    print("=" * 70)

    df = pd.read_csv(os.path.join(PROCESSED, "summary_may2026.csv"))
    cat_means = category_means(df)

    # Baseline weights
    baseline = {"AskingRent": 0.0, "EffectiveRent": 0.20, "Employment": 0.10,
                "Supply": 0.0, "UPP": 0.20, "Income": 0.0,
                "Population": 0.10, "SalePrice": 0.20}
    # Sum = 0.80; original model treats this as the full weight (sum check
    # cell shows 1.0 because asking-rent group banner spans columns whose
    # weights sum to nothing). To match the file's published WeightedAverage
    # we'll keep the published weights as-is even though they sum to 0.8.

    schemes = {}

    # a) baseline
    schemes["baseline"] = baseline.copy()

    # b) equal — 1/8 to each category, sum = 1.0
    schemes["equal"] = {c: 1 / 8 for c in CATEGORY_COLS}

    # c) drop_effective_rent — 0 to EffRent, redistribute that 0.20 to the
    # other CURRENTLY-NON-ZERO categories proportionally
    nonzero_other = {c: w for c, w in baseline.items()
                     if c != "EffectiveRent" and w > 0}
    redistributed = {}
    total_other = sum(nonzero_other.values())
    extra = baseline["EffectiveRent"]
    for c in baseline:
        if c == "EffectiveRent":
            redistributed[c] = 0.0
        elif c in nonzero_other:
            redistributed[c] = baseline[c] + extra * (baseline[c] / total_other)
        else:
            redistributed[c] = baseline[c]
    schemes["drop_effective_rent"] = redistributed

    # d) double_employment — Employment 0.20, rest scaled to keep sum same
    de = baseline.copy()
    de["Employment"] = 0.20
    # Scale remaining non-zero non-Employment cats so total matches baseline sum
    target = sum(baseline.values())
    rest_total = target - 0.20
    rest = {c: v for c, v in baseline.items()
            if c != "Employment" and v > 0}
    rest_sum = sum(rest.values())
    for c in de:
        if c != "Employment" and c in rest:
            de[c] = rest[c] / rest_sum * rest_total
    schemes["double_employment"] = de

    # e) inverse_volatility — weight ∝ 1 / mean|rank change| per category
    vol = pd.read_csv(os.path.join(REPORTS, "volatility_by_category.csv"))
    inv_w = {}
    for _, r in vol.iterrows():
        v = r["MeanAbsRankChange"]
        inv_w[r["Category"]] = (1.0 / v) if v and v > 0 else 0.0
    inv_w = normalize(inv_w)
    schemes["inverse_volatility"] = inv_w

    # ---- Run each scheme ---------------------------------------------------
    rankings = {}
    for name, w in schemes.items():
        ranked = apply_weights(cat_means, w)
        rankings[name] = ranked

    # ---- Comparison table: top 20 across schemes ---------------------------
    top20_baseline = rankings["baseline"].head(20)["Market"].tolist()
    rows = []
    for name, ranked in rankings.items():
        ranked_indexed = ranked.set_index("Market")
        for i, mkt in enumerate(top20_baseline, start=1):
            rk = ranked_indexed.loc[mkt, "Rank"] if mkt in ranked_indexed.index else None
            rows.append({
                "BaselineTop20Pos": i, "Market": mkt, "Scheme": name,
                "Rank": int(rk) if rk is not None and pd.notna(rk) else None,
            })
    cmp_long = pd.DataFrame(rows)
    cmp_wide = cmp_long.pivot(index=["BaselineTop20Pos", "Market"],
                              columns="Scheme", values="Rank").reset_index()
    cmp_wide.to_csv(os.path.join(REPORTS, "weight_sensitivity_top20.csv"),
                    index=False)

    # Save full per-scheme rankings
    full_rows = []
    for name, ranked in rankings.items():
        for _, r in ranked.iterrows():
            full_rows.append({"Scheme": name, "Market": r["Market"],
                              "WeightedAvg": r["WeightedAvg"],
                              "Rank": r["Rank"]})
    pd.DataFrame(full_rows).to_csv(
        os.path.join(REPORTS, "weight_sensitivity_full.csv"), index=False)

    # Save the schemes themselves
    schemes_df = pd.DataFrame(schemes).T
    schemes_df["sum"] = schemes_df.sum(axis=1)
    schemes_df.to_csv(os.path.join(REPORTS, "weight_sensitivity_schemes.csv"))

    # ---- Plot top 20 ranks across schemes ----------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    schemes_order = ["baseline", "equal", "drop_effective_rent",
                     "double_employment", "inverse_volatility"]
    pivot = cmp_wide.set_index("Market")[schemes_order]
    pivot.plot(kind="line", marker="o", ax=ax, alpha=0.85, lw=1)
    ax.invert_yaxis()
    ax.set_ylabel("Rank under scheme (1 = best)")
    ax.set_title("Top 20 baseline markets — rank under each weighting scheme")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=60, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "weight_sensitivity_top20.png"), dpi=140)
    plt.close(fig)

    # ---- Print findings -----------------------------------------------------
    print("\n  Weight schemes:")
    print(schemes_df.to_string(float_format="%.4f"))
    print("\n  Inverse-volatility weights:")
    for c, w in sorted(inv_w.items(), key=lambda x: -x[1]):
        print(f"    {c:14s} {w:.4f}")
    print("\n  Top 10 markets under each scheme:")
    for name in schemes_order:
        top10 = rankings[name].head(10)["Market"].tolist()
        print(f"    {name:22s} {', '.join(top10[:5])} | {', '.join(top10[5:])}")

    # Movement summary
    print("\n  Rank moves of baseline top 5 under each alt scheme:")
    for mkt in rankings["baseline"].head(5)["Market"]:
        line = f"    {mkt:30s}"
        for name in schemes_order:
            rk = rankings[name].set_index("Market").loc[mkt, "Rank"]
            r = int(rk) if pd.notna(rk) else -1
            line += f"  {name[:6]}={r:>3d}"
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
