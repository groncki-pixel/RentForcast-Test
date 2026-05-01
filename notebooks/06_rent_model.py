"""
06_rent_model.py
================
Build a simple predictive model for effective rent growth using lagged
fundamentals from file 2 (the data refresh). Compare against the vendor's
forward estimates from file 1 over the test period 2024-2025.

Data setup
----------
- Target: quarterly Effective Rent Growth from v3_effective_rent_growth.csv
- Predictors (all quarterly, lag 1):
    * employment growth (from v3_employment level series)
    * population growth (v3_population)
    * inventory growth (v3_inventory_units, used as supply proxy)
    * income growth (v3_median_household_income)
    * asking rent growth (v3_asking_rent_growth)
    * sale price index growth (v3_market_sale_price_index)
- Train: 2016 Q2 - 2023 Q4
- Test:  2024 Q1 - 2025 Q1 (the quarters where actuals exist *and* the
  vendor's file-1 forecast started)
- Vendor benchmark: file 1 effective rent forward pct_change (rebased to
  1.00 at 2025 Q2). Available test quarters overlapping with our actuals:
  2025 Q3, 2025 Q4, 2026 Q1.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
DISCREPANCY_LOG = os.path.join(REPORTS, "data_discrepancies_log.txt")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def log_disc(msg: str) -> None:
    with open(DISCREPANCY_LOG, "a") as f:
        f.write(f"[06_rent_model] {msg}\n")
    print(f"  [LOG] {msg}")


def long_format(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide market x quarter to long Market/Quarter/value table."""
    out = df.reset_index().melt(id_vars="Market", var_name="Quarter",
                                value_name=value_name)
    return out


def levels_to_growth(levels: pd.DataFrame) -> pd.DataFrame:
    """Quarter-over-quarter pct change."""
    return levels.pct_change(axis=1)


def main() -> None:
    print("=" * 70)
    print("SCRIPT 6: PROPRIETARY RENT FORECAST MODEL")
    print("=" * 70)
    with open(DISCREPANCY_LOG, "a") as f:
        f.write("\n----- 06_rent_model (latest run) -----\n")

    # ---- Load file 2 inputs -------------------------------------------------
    eff_growth = pd.read_csv(os.path.join(PROCESSED, "v3_effective_rent_growth.csv"),
                             index_col=0)
    ask_growth = pd.read_csv(os.path.join(PROCESSED, "v3_asking_rent_growth.csv"),
                             index_col=0)
    emp_lev = pd.read_csv(os.path.join(PROCESSED, "v3_employment.csv"),
                          index_col=0)
    pop_lev = pd.read_csv(os.path.join(PROCESSED, "v3_population.csv"),
                          index_col=0)
    inv_lev = pd.read_csv(os.path.join(PROCESSED, "v3_inventory_units.csv"),
                          index_col=0)
    inc_lev = pd.read_csv(os.path.join(PROCESSED, "v3_median_household_income.csv"),
                          index_col=0)
    sp_lev = pd.read_csv(os.path.join(PROCESSED, "v3_market_sale_price_index.csv"),
                         index_col=0)

    # Convert level series to growth rates
    emp_g = levels_to_growth(emp_lev)
    pop_g = levels_to_growth(pop_lev)
    inv_g = levels_to_growth(inv_lev)
    inc_g = levels_to_growth(inc_lev)
    sp_g = levels_to_growth(sp_lev)

    # ---- Build a panel: market x quarter rows, predictor cols --------------
    panel = long_format(eff_growth, "y_eff_rent_g")
    for src, name in [
        (ask_growth, "x_ask_g"),
        (emp_g, "x_emp_g"),
        (pop_g, "x_pop_g"),
        (inv_g, "x_inv_g"),
        (inc_g, "x_inc_g"),
        (sp_g, "x_sp_g"),
    ]:
        panel = panel.merge(long_format(src, name),
                            on=["Market", "Quarter"], how="left")

    # Lag predictors by 1 quarter so we predict y_t from x_{t-1}
    panel = panel.sort_values(["Market", "Quarter"])
    for c in ["x_ask_g", "x_emp_g", "x_pop_g", "x_inv_g", "x_inc_g", "x_sp_g"]:
        panel[c] = panel.groupby("Market")[c].shift(1)

    # ---- Train / test split -------------------------------------------------
    train = panel[(panel["Quarter"] >= "2016 Q2") & (panel["Quarter"] <= "2023 Q4")]
    test = panel[(panel["Quarter"] >= "2024 Q1") & (panel["Quarter"] <= "2025 Q1")]

    feat_cols = ["x_ask_g", "x_emp_g", "x_pop_g", "x_inv_g", "x_inc_g", "x_sp_g"]
    train_clean = train.dropna(subset=feat_cols + ["y_eff_rent_g"])
    test_clean = test.dropna(subset=feat_cols + ["y_eff_rent_g"])

    print(f"  Train obs: {len(train_clean):,}  Test obs: {len(test_clean):,}")
    print(f"  Features: {feat_cols}")
    if len(train_clean) == 0 or len(test_clean) == 0:
        log_disc(f"Insufficient training/testing data: train={len(train_clean)}, test={len(test_clean)}")
        return

    X_tr = train_clean[feat_cols].values
    y_tr = train_clean["y_eff_rent_g"].values
    X_te = test_clean[feat_cols].values
    y_te = test_clean["y_eff_rent_g"].values

    # ---- Linear regression --------------------------------------------------
    lr = LinearRegression().fit(X_tr, y_tr)
    yhat_lr = lr.predict(X_te)
    lr_mae = mean_absolute_error(y_te, yhat_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_te, yhat_lr))

    # ---- Random forest ------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42,
                               n_jobs=-1).fit(X_tr, y_tr)
    yhat_rf = rf.predict(X_te)
    rf_mae = mean_absolute_error(y_te, yhat_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_te, yhat_rf))

    # ---- Naive baseline: assume next-q growth = last-q observed growth ------
    yhat_naive = test_clean["x_ask_g"].values  # x_ask_g is already lag-1
    naive_mae = mean_absolute_error(y_te, yhat_naive)
    naive_rmse = np.sqrt(mean_squared_error(y_te, yhat_naive))

    # ---- Vendor benchmark ---------------------------------------------------
    # File 1 forward: indexed pct change at 2025 Q2 = 1.0; quarter-over-quarter
    # change can be backed out as ratio of consecutive index values.
    fc_idx = pd.read_csv(
        os.path.join(PROCESSED, "orig_effective_rent_forward_pct_change.csv"),
        index_col=0)
    fc_idx.columns = [c.replace(" EST", "").strip() for c in fc_idx.columns]
    # vendor's QoQ growth implied = idx[q] / idx[q-1] - 1; only meaningful for
    # the forecast period (2025 Q2 onward). The overlap with our test window
    # (2024 Q1 - 2025 Q1) is therefore EMPTY: file 1 has no forecast values
    # before its baseline. So we can only compare our model to the vendor in
    # the forecast quarters 2025 Q3, 2025 Q4, 2026 Q1, against actuals from
    # file 2.
    overlap_qtrs = ["2025 Q3", "2025 Q4", "2026 Q1"]
    available = [q for q in overlap_qtrs if q in fc_idx.columns]
    vendor_qoq = pd.DataFrame(index=fc_idx.index)
    prev = fc_idx["2025 Q2"]
    for q in available:
        vendor_qoq[q] = fc_idx[q] / prev - 1.0
        prev = fc_idx[q]

    # Actual qoq growth from file 2 for the same quarters
    actual_qoq = eff_growth.reindex(columns=available)

    # Build evaluation rows
    bench_rows = []
    for mkt in fc_idx.index:
        if mkt not in actual_qoq.index:
            continue
        for q in available:
            ya = actual_qoq.loc[mkt, q]
            yv = vendor_qoq.loc[mkt, q]
            if pd.isna(ya) or pd.isna(yv):
                continue
            bench_rows.append({"Market": mkt, "Quarter": q,
                               "actual": ya, "vendor": yv})
    bench = pd.DataFrame(bench_rows)
    if len(bench):
        v_mae = mean_absolute_error(bench["actual"], bench["vendor"])
        v_rmse = np.sqrt(mean_squared_error(bench["actual"], bench["vendor"]))
    else:
        v_mae = v_rmse = float("nan")

    # Apply our model on the same overlap quarters using lag-1 fundamentals
    # available for those quarters
    overlap_panel = panel[panel["Quarter"].isin(available)].copy()
    overlap_clean = overlap_panel.dropna(subset=feat_cols)
    if len(overlap_clean):
        overlap_clean["yhat_lr"] = lr.predict(overlap_clean[feat_cols].values)
        overlap_clean["yhat_rf"] = rf.predict(overlap_clean[feat_cols].values)
        # Compare to actuals where available
        comp = overlap_clean.dropna(subset=["y_eff_rent_g"])
        if len(comp):
            our_lr_mae = mean_absolute_error(comp["y_eff_rent_g"], comp["yhat_lr"])
            our_rf_mae = mean_absolute_error(comp["y_eff_rent_g"], comp["yhat_rf"])
        else:
            our_lr_mae = our_rf_mae = float("nan")
    else:
        our_lr_mae = our_rf_mae = float("nan")

    # ---- Save metrics -------------------------------------------------------
    metrics = pd.DataFrame([
        {"Model": "LinearRegression (in-sample test 2024Q1-2025Q1)",
         "MAE": lr_mae, "RMSE": lr_rmse, "N": len(y_te)},
        {"Model": "RandomForest    (in-sample test 2024Q1-2025Q1)",
         "MAE": rf_mae, "RMSE": rf_rmse, "N": len(y_te)},
        {"Model": "Naive (lag asking-rent growth)",
         "MAE": naive_mae, "RMSE": naive_rmse, "N": len(y_te)},
        {"Model": "Vendor (file 1 forecast vs file 2 actual, 2025Q3-2026Q1)",
         "MAE": v_mae, "RMSE": v_rmse, "N": len(bench)},
        {"Model": "Our LR on vendor's forecast window (2025Q3-2026Q1)",
         "MAE": our_lr_mae, "RMSE": float("nan"),
         "N": int(comp.shape[0]) if len(overlap_clean) and len(comp) else 0},
        {"Model": "Our RF on vendor's forecast window (2025Q3-2026Q1)",
         "MAE": our_rf_mae, "RMSE": float("nan"),
         "N": int(comp.shape[0]) if len(overlap_clean) and len(comp) else 0},
    ])
    metrics.to_csv(os.path.join(REPORTS, "rent_model_metrics.csv"), index=False)

    # Coefficients / importances
    coeffs = pd.DataFrame({
        "feature": feat_cols,
        "lr_coef": lr.coef_,
        "rf_importance": rf.feature_importances_,
    })
    coeffs.to_csv(os.path.join(REPORTS, "rent_model_features.csv"), index=False)

    # Save predictions vs actual
    test_clean = test_clean.copy()
    test_clean["yhat_lr"] = yhat_lr
    test_clean["yhat_rf"] = yhat_rf
    test_clean[["Market", "Quarter", "y_eff_rent_g", "yhat_lr", "yhat_rf"]].to_csv(
        os.path.join(REPORTS, "rent_model_test_predictions.csv"), index=False)

    # ---- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, yhat_lr, alpha=0.4, s=12, label="Linear")
    ax.scatter(y_te, yhat_rf, alpha=0.4, s=12, label="RandomForest", color="#e76f51")
    lim = max(abs(y_te).max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.7)
    ax.set_xlabel("Actual quarterly effective rent growth")
    ax.set_ylabel("Predicted")
    ax.set_title("Model predictions vs actual, test window 2024Q1-2025Q1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "rent_model_pred_vs_actual.png"), dpi=140)
    plt.close(fig)

    # Bar chart of MAE comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ms = metrics.dropna(subset=["MAE"])
    ax.barh(ms["Model"], ms["MAE"], color="#264653")
    ax.set_xlabel("Mean Absolute Error (quarterly growth, fraction)")
    ax.set_title("Effective rent growth: model error comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "rent_model_mae_comparison.png"), dpi=140)
    plt.close(fig)

    # Feature importance bar
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(coeffs["feature"], coeffs["rf_importance"], color="#2a9d8f")
    ax.set_ylabel("Random Forest feature importance")
    ax.set_title("Predictors of effective rent growth (lag 1 quarter)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "rent_model_feature_importance.png"), dpi=140)
    plt.close(fig)

    # ---- Print findings -----------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL METRICS")
    print("=" * 70)
    print(metrics.to_string(index=False, float_format="%.5f"))
    print("\n  Linear regression coefficients:")
    print(coeffs.to_string(index=False, float_format="%.5f"))

    print("\n  Verdict:")
    if not np.isnan(our_rf_mae) and not np.isnan(v_mae):
        if our_rf_mae < v_mae:
            print(f"    Our random forest beats the vendor on the overlap window: "
                  f"{our_rf_mae:.5f} vs {v_mae:.5f}.")
            print(f"    APPROACH SHOWS PROMISE — proceed to refine.")
        else:
            print(f"    Vendor still wins on the overlap window: vendor MAE={v_mae:.5f} "
                  f"vs our RF MAE={our_rf_mae:.5f}.")
            print(f"    Approach is competitive but not dominant; needs feature "
                  f"engineering or a per-market model.")
    else:
        print("    Insufficient overlap data to compare against vendor.")

    print("\nDone.")


if __name__ == "__main__":
    main()
