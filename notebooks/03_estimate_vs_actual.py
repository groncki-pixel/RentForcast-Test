"""
03_estimate_vs_actual.py
========================
Compare file 1 forward estimates (anchored 2025 Q2) against file 2 actuals
for the overlap window 2025 Q3 - 2026 Q1.

Methodology
-----------
File 1 *_pct_change tables are already rebased to 1.00 at 2025 Q2. To make
file 2 levels comparable, divide each market's level series by its 2025 Q2
level. The estimate-vs-actual error per quarter is then:
    error_q = forecast_index_q - actual_index_q
where both are unitless multiplicative growth from the 2025 Q2 baseline.

For variables only available as quarterly growth rates in file 2 (Effective
Rent Growth, Asking Rent Growth) we cumulatively compound from 2025 Q2 to
build an actual index series.

Outputs:
  outputs/reports/forecast_errors_<variable>.csv  per-market per-quarter errors
  outputs/reports/forecast_error_summary.csv      aggregate summary table
  outputs/figures/forecast_error_by_variable.png
  outputs/figures/forecast_error_top_markets.png
  outputs/figures/austin_indianapolis_compare.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
DISCREPANCY_LOG = os.path.join(REPORTS, "data_discrepancies_log.txt")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

BASELINE_QTR = "2025 Q2"
OVERLAP_QTRS = ["2025 Q3", "2025 Q4", "2026 Q1"]


def log_disc(msg: str) -> None:
    with open(DISCREPANCY_LOG, "a") as f:
        f.write(f"[03_estimate_vs_actual] {msg}\n")
    print(f"  [LOG] {msg}")


def load_orig_forecast(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED, name), index_col=0)
    # Rename "2025 Q2 EST" to "2025 Q2" so quarter columns align
    df.columns = [c.replace(" EST", "").strip() for c in df.columns]
    return df


def load_v3_levels(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED, name), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def rebase_to_baseline(levels: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Divide each row by its baseline-quarter level."""
    if baseline not in levels.columns:
        raise KeyError(f"Baseline {baseline} not in columns: {list(levels.columns)[:5]}...")
    return levels.div(levels[baseline], axis=0)


def cumulate_growth(growth: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Convert a quarterly growth-rate matrix to an index anchored at baseline.

    Index value at the baseline column = 1.00. Earlier columns are computed
    by dividing forward; later columns by cumulative compounding.
    """
    cols = list(growth.columns)
    if baseline not in cols:
        raise KeyError(f"Baseline {baseline} not in growth columns")
    bidx = cols.index(baseline)
    # Build index column-by-column. Start at 1.0 at baseline, then forward
    # multiply (1+g_{q+1}) and backward divide.
    idx = pd.DataFrame(index=growth.index, columns=cols, dtype=float)
    idx[baseline] = 1.0
    for i in range(bidx + 1, len(cols)):
        idx[cols[i]] = idx[cols[i - 1]] * (1.0 + growth[cols[i]])
    for i in range(bidx - 1, -1, -1):
        idx[cols[i]] = idx[cols[i + 1]] / (1.0 + growth[cols[i + 1]])
    return idx


def compute_errors(forecast: pd.DataFrame, actual: pd.DataFrame,
                   variable: str) -> pd.DataFrame:
    """Per-market, per-quarter error = forecast - actual for OVERLAP_QTRS."""
    fc = forecast.reindex(columns=OVERLAP_QTRS)
    ac = actual.reindex(columns=OVERLAP_QTRS)
    common = sorted(set(fc.index) & set(ac.index))
    errs = (fc.loc[common] - ac.loc[common]).copy()
    errs.columns = [f"err_{c}" for c in errs.columns]
    errs["MAE"] = errs.abs().mean(axis=1)
    errs["MeanError"] = errs[[c for c in errs.columns
                              if c.startswith("err_")]].mean(axis=1)
    errs["Variable"] = variable
    errs.index.name = "Market"
    return errs.reset_index()


def main() -> None:
    print("=" * 70)
    print("SCRIPT 3: ESTIMATE vs ACTUAL")
    print("=" * 70)
    with open(DISCREPANCY_LOG, "a") as f:
        f.write("\n----- 03_estimate_vs_actual (latest run) -----\n")

    # ---- Effective Rent: use Table 2 pct_change ----------------------------
    print(f"\n[1] Effective Rent (using indexed % change tables, baseline {BASELINE_QTR})")
    fc_er = load_orig_forecast("orig_effective_rent_forward_pct_change.csv")
    er_growth = load_v3_levels("v3_effective_rent_growth.csv")
    actual_er = cumulate_growth(er_growth, BASELINE_QTR)
    err_er = compute_errors(fc_er, actual_er, "EffectiveRent")
    err_er.to_csv(os.path.join(REPORTS, "forecast_errors_effective_rent.csv"),
                  index=False)
    print(f"    {len(err_er)} markets, mean MAE = {err_er['MAE'].mean():.4f}, "
          f"mean signed = {err_er['MeanError'].mean():+.4f}")

    # ---- Asking Rent --------------------------------------------------------
    print(f"\n[2] Asking Rent")
    fc_ar = load_orig_forecast("orig_asking_rent_pct_change.csv")
    ar_growth = load_v3_levels("v3_asking_rent_growth.csv")
    actual_ar = cumulate_growth(ar_growth, BASELINE_QTR)
    err_ar = compute_errors(fc_ar, actual_ar, "AskingRent")
    err_ar.to_csv(os.path.join(REPORTS, "forecast_errors_asking_rent.csv"),
                  index=False)
    print(f"    {len(err_ar)} markets, mean MAE = {err_ar['MAE'].mean():.4f}, "
          f"mean signed = {err_ar['MeanError'].mean():+.4f}")

    # ---- Employment: forecast pct_change vs rebased actual levels ----------
    print(f"\n[3] Employment")
    fc_em = load_orig_forecast("orig_forward_employment_pct_change.csv")
    em_lev = load_v3_levels("v3_employment.csv")
    actual_em = rebase_to_baseline(em_lev, BASELINE_QTR)
    err_em = compute_errors(fc_em, actual_em, "Employment")
    err_em.to_csv(os.path.join(REPORTS, "forecast_errors_employment.csv"),
                  index=False)
    print(f"    {len(err_em)} markets, mean MAE = {err_em['MAE'].mean():.4f}, "
          f"mean signed = {err_em['MeanError'].mean():+.4f}")

    # ---- Supply: forecast pct_change vs rebased Inventory Units ------------
    print(f"\n[4] Supply (Inventory Units)")
    fc_sp = load_orig_forecast("orig_supply_forward_pct_change.csv")
    inv_lev = load_v3_levels("v3_inventory_units.csv")
    actual_sp = rebase_to_baseline(inv_lev, BASELINE_QTR)
    err_sp = compute_errors(fc_sp, actual_sp, "Supply")
    err_sp.to_csv(os.path.join(REPORTS, "forecast_errors_supply.csv"),
                  index=False)
    print(f"    {len(err_sp)} markets, mean MAE = {err_sp['MAE'].mean():.4f}, "
          f"mean signed = {err_sp['MeanError'].mean():+.4f}")

    # ---- Sale Price: forecast pct_change vs rebased Market Sale Price Index --
    print(f"\n[5] Sale Price")
    fc_sl = load_orig_forecast("orig_forward_sale_price_pct_change.csv")
    sl_lev = load_v3_levels("v3_market_sale_price_index.csv")
    actual_sl = rebase_to_baseline(sl_lev, BASELINE_QTR)
    err_sl = compute_errors(fc_sl, actual_sl, "SalePrice")
    err_sl.to_csv(os.path.join(REPORTS, "forecast_errors_sale_price.csv"),
                  index=False)
    print(f"    {len(err_sl)} markets, mean MAE = {err_sl['MAE'].mean():.4f}, "
          f"mean signed = {err_sl['MeanError'].mean():+.4f}")

    # ---- Combine ------------------------------------------------------------
    all_err = pd.concat([err_er, err_ar, err_em, err_sp, err_sl],
                        ignore_index=True)
    all_err.to_csv(os.path.join(REPORTS, "forecast_errors_all.csv"), index=False)

    summary = (all_err.groupby("Variable")
               .agg(N=("Market", "count"),
                    MAE_mean=("MAE", "mean"),
                    MAE_p90=("MAE", lambda s: s.quantile(0.9)),
                    SignedError_mean=("MeanError", "mean"),
                    SignedError_std=("MeanError", "std"))
               .reset_index()
               .sort_values("MAE_mean", ascending=False))
    summary.to_csv(os.path.join(REPORTS, "forecast_error_summary.csv"),
                   index=False)

    # ---- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["Variable"], summary["MAE_mean"], color="#264653")
    ax.set_ylabel("Mean Absolute Error (index units)")
    ax.set_title("Forecast MAE by variable, 2025 Q3 - 2026 Q1")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "forecast_error_by_variable.png"), dpi=140)
    plt.close(fig)

    # Top 15 markets by MAE for Effective Rent (the focus)
    fig, ax = plt.subplots(figsize=(9, 7))
    top = err_er.nlargest(15, "MAE").sort_values("MAE")
    ax.barh(top["Market"], top["MAE"], color="#e76f51")
    ax.set_xlabel("MAE on indexed Effective Rent (vs 2025 Q2 = 1.0)")
    ax.set_title("Top 15 markets by Effective Rent forecast error (file 1 vs file 2 actuals)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "forecast_error_top_markets.png"), dpi=140)
    plt.close(fig)

    # ---- Austin / Indianapolis spotlight -----------------------------------
    spotlight = ["Austin - TX", "Indianapolis - IN"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    quarters = OVERLAP_QTRS
    for ax, mkt in zip(axes, spotlight):
        for label, fc, ac, color in [
            ("EffectiveRent", fc_er, actual_er, "#264653"),
            ("Employment", fc_em, actual_em, "#2a9d8f"),
            ("SalePrice", fc_sl, actual_sl, "#e76f51"),
        ]:
            if mkt in fc.index and mkt in ac.index:
                ax.plot(quarters, fc.loc[mkt, quarters].values, "--",
                        color=color, label=f"{label} forecast")
                ax.plot(quarters, ac.loc[mkt, quarters].values, "-",
                        color=color, label=f"{label} actual")
        ax.set_title(mkt)
        ax.set_xlabel("Quarter")
        ax.axhline(1.0, color="grey", lw=0.5, ls=":")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Index (2025 Q2 = 1.0)")
    fig.suptitle("Forecast vs Actual: Austin and Indianapolis")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "austin_indianapolis_compare.png"), dpi=140)
    plt.close(fig)

    # ---- Print findings -----------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("\n  Forecast error summary by variable:")
    print(summary.to_string(index=False, float_format="%.5f"))

    print("\n  Markets with worst Effective Rent forecast error (top 10):")
    for _, r in err_er.nlargest(10, "MAE").iterrows():
        print(f"    {r['Market']:30s} MAE={r['MAE']:.4f} "
              f"signed={r['MeanError']:+.4f}")

    aus_er = err_er[err_er["Market"] == "Austin - TX"].iloc[0] \
        if (err_er["Market"] == "Austin - TX").any() else None
    ind_er = err_er[err_er["Market"] == "Indianapolis - IN"].iloc[0] \
        if (err_er["Market"] == "Indianapolis - IN").any() else None
    if aus_er is not None:
        print(f"\n  Austin - TX EffectiveRent: signed error = {aus_er['MeanError']:+.4f}, "
              f"MAE = {aus_er['MAE']:.4f}")
        print(f"     (positive signed = forecast was OVER actual i.e. we expected "
              f"more rent growth than realized)")
    if ind_er is not None:
        print(f"  Indianapolis EffectiveRent: signed error = {ind_er['MeanError']:+.4f}, "
              f"MAE = {ind_er['MAE']:.4f}")

    overall_signed = all_err.groupby("Variable")["MeanError"].mean()
    print("\n  Systematic forecast bias (positive = forecast > actual):")
    for v, val in overall_signed.items():
        direction = "OVER" if val > 0 else "UNDER"
        print(f"    {v:14s} {val:+.4f}  ({direction}-forecast)")

    print("\nDone.")


if __name__ == "__main__":
    main()
