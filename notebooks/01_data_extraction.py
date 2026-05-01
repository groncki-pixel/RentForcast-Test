"""
01_data_extraction.py
=====================
Read all 3 raw Excel files, parse the stacked tables, normalize market names,
and write clean CSVs into data/processed/.

Run from repo root:  python notebooks/01_data_extraction.py
"""

import os
import re
import sys
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

DISCREPANCY_LOG = os.path.join(REPORTS, "data_discrepancies_log.txt")

FILE_ORIG = os.path.join(RAW, "Full_Market_Leveler_Original.xlsx")
FILE_V3 = os.path.join(RAW, "Updated_Market_Leveler_v3.xlsx")
FILE_MAY = os.path.join(RAW, "Full_Market_Leveler_May2026.xlsx")


def log_disc(msg: str) -> None:
    with open(DISCREPANCY_LOG, "a") as f:
        f.write(f"[01_data_extraction] {msg}\n")
    print(f"  [LOG] {msg}")


# ----------------------------------------------------------------------------
# Market name normalization
# ----------------------------------------------------------------------------
_RE_USA_PAREN = re.compile(r"\s*\(USA\)\s*$")
_RE_USA_PLAIN = re.compile(r"\s+USA\s*$")
_RE_WS = re.compile(r"\s+")


def normalize_market(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip()
    s = _RE_USA_PAREN.sub("", s)
    s = _RE_USA_PLAIN.sub("", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


# ----------------------------------------------------------------------------
# Summary Sheet extraction (files 1 and 3)
# ----------------------------------------------------------------------------
SUMMARY_HEADERS_BY_COL = {
    1: "AskingRent_Forward1", 2: "AskingRent_Forward3",
    3: "AskingRent_Forward5", 4: "AskingRent_Forward10",
    5: "EffectiveRent_Forward1", 6: "EffectiveRent_Forward3",
    7: "EffectiveRent_Forward5", 8: "EffectiveRent_Forward10",
    9: "Employment_Hist10", 10: "Employment_Hist5",
    11: "Employment_Forward1", 12: "Employment_Forward3",
    13: "Employment_Forward5", 14: "Employment_Forward10",
    15: "Supply_Hist5", 16: "Supply_Forward1",
    17: "Supply_Forward3", 18: "Supply_Forward5",
    19: "Supply_Forward10",
    20: "UPP_Hist10", 21: "UPP_Forward5", 22: "UPP_Forward10",
    23: "Income_Forward1", 24: "Income_Forward3",
    25: "Income_Forward5", 26: "Income_Forward10",
    27: "Population_Hist10", 28: "Population_Hist5",
    29: "Population_Forward1", 30: "Population_Forward3",
    31: "Population_Forward5", 32: "Population_Forward10",
    33: "SalePrice_Hist1", 34: "SalePrice_Hist3",
    35: "SalePrice_Hist5", 36: "SalePrice_Hist10",
    37: "SalePrice_Forward1", 38: "SalePrice_Forward3",
    39: "SalePrice_Forward5", 40: "SalePrice_Forward10",
}

CATEGORY_TO_COLS = {
    "AskingRent": list(range(1, 5)),
    "EffectiveRent": list(range(5, 9)),
    "Employment": list(range(9, 15)),
    "Supply": list(range(15, 20)),
    "UPP": list(range(20, 23)),
    "Income": list(range(23, 27)),
    "Population": list(range(27, 33)),
    "SalePrice": list(range(33, 41)),
}


def extract_summary_sheet(path: str, label: str) -> tuple[pd.DataFrame, dict]:
    """Returns (markets_df, weights_dict).

    markets_df has columns: Market + 40 sub-ranking columns +
      SimpleAverage + WeightedAverage + FinalRank.
    weights_dict maps category -> weight (float).
    """
    raw = pd.read_excel(path, sheet_name="Summary Sheet", header=None)

    # Row 1 = weights (numeric values appear at specific cols)
    weights_row = raw.iloc[1]
    # Find the weight that "covers" each category: assume the weight aligns
    # with one of the columns in the category block; pick the first numeric
    # value in that block.
    weights = {}
    for cat, cols in CATEGORY_TO_COLS.items():
        vals = [weights_row[c] for c in cols
                if isinstance(weights_row[c], (int, float)) and pd.notna(weights_row[c])]
        weights[cat] = float(vals[0]) if vals else 0.0

    # Markets at rows 3..152, col 0
    body = raw.iloc[3:153].copy()
    out = pd.DataFrame()
    out["Market"] = body.iloc[:, 0].apply(normalize_market)

    for col, name in SUMMARY_HEADERS_BY_COL.items():
        out[name] = pd.to_numeric(body.iloc[:, col], errors="coerce")

    out["SimpleAverage"] = pd.to_numeric(body.iloc[:, 41], errors="coerce")
    out["WeightedAverage"] = pd.to_numeric(body.iloc[:, 42], errors="coerce")

    # Compute rank from WeightedAverage (lower = better)
    out["FinalRank"] = out["WeightedAverage"].rank(method="min").astype("Int64")

    out = out.reset_index(drop=True)
    print(f"  [{label} Summary] {len(out)} markets, weights={weights}")
    return out, weights


# ----------------------------------------------------------------------------
# Stacked-table extractor for "forward" / "historical" sheets in files 1 & 3
# ----------------------------------------------------------------------------
def find_geography_col(row_values) -> int:
    """Identify which column holds the 'Geography Name' / 'Market' label."""
    for i, v in enumerate(row_values):
        if isinstance(v, str) and v.strip() in ("Geography Name", "Market"):
            return i
    # Fallback: first non-null string column
    for i, v in enumerate(row_values):
        if isinstance(v, str):
            return i
    return 0


def extract_stacked_sheet(path: str, sheet: str) -> dict[str, pd.DataFrame]:
    """Split a forward/historical sheet into one DataFrame per stacked table.

    Returns a dict like {'levels': df1, 'pct_change': df2, 'aggregate': df3}
    where each df has Market as index and quarter columns as columns.
    All operations are positional (iloc) to tolerate duplicate / NaN headers.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    # Find header rows (rows where some col contains 'Geography Name'/'Market')
    header_rows = []
    for i in range(len(raw)):
        row_vals = raw.iloc[i].tolist()
        if any(isinstance(v, str) and v.strip() in ("Geography Name", "Market")
               for v in row_vals):
            header_rows.append(i)
    if not header_rows:
        return {}

    blank_rows = set()
    for i in range(len(raw)):
        if raw.iloc[i].isna().all():
            blank_rows.add(i)

    blocks = {}
    label_seq = ["levels", "pct_change", "aggregate", "extra1", "extra2"]
    for idx, hrow in enumerate(header_rows[:5]):
        end = len(raw)
        for br in sorted(blank_rows):
            if br > hrow:
                end = br
                break
        for hr in header_rows:
            if hr > hrow and hr < end:
                end = hr
                break

        header_vals = raw.iloc[hrow].tolist()
        geo_col = find_geography_col(header_vals)
        # Determine which columns to keep (everything to the right of geo_col,
        # except any "Concept Name" col and NaN-headed cols)
        keep_idx = []
        keep_names = []
        for ci, name in enumerate(header_vals):
            if ci <= geo_col:
                continue
            if isinstance(name, float) and pd.isna(name):
                continue
            sname = str(name).strip()
            if sname == "" or sname.lower() == "concept name":
                continue
            keep_idx.append(ci)
            keep_names.append(sname)

        # Build positional data slice
        body = raw.iloc[hrow + 1:end]
        market_col = body.iloc[:, geo_col].apply(normalize_market)
        mask = market_col != ""
        if mask.sum() == 0:
            continue
        data = pd.DataFrame({"Market": market_col[mask].values})
        for ci, n in zip(keep_idx, keep_names):
            data[n] = pd.to_numeric(body.iloc[:, ci][mask].values, errors="coerce")
        # Deduplicate column names (rare; if happens, suffix them)
        seen = {}
        new_cols = []
        for c in data.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        data.columns = new_cols
        data = data.drop_duplicates(subset=["Market"]).set_index("Market")

        label = label_seq[idx] if idx < len(label_seq) else f"block{idx}"
        blocks[label] = data

    return blocks


# ----------------------------------------------------------------------------
# File 2 sheet extractor (flat layout)
# ----------------------------------------------------------------------------
def extract_v3_sheet(path: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=0)
    # First column is market (sometimes named 'Market', sometimes 'x')
    raw.columns = [str(c) for c in raw.columns]
    market_col = raw.columns[0]
    raw["Market"] = raw[market_col].apply(normalize_market)
    raw = raw[raw["Market"] != ""].copy()
    keep_cols = [c for c in raw.columns
                 if c not in (market_col, "Market") and not c.startswith("Unnamed")]
    for c in keep_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    out = raw[["Market"] + keep_cols].drop_duplicates(subset=["Market"])
    out = out.set_index("Market")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("SCRIPT 1: DATA EXTRACTION")
    print("=" * 70)

    # ---- Reset extractor section of discrepancy log -------------------------
    with open(DISCREPANCY_LOG, "a") as f:
        f.write("\n----- 01_data_extraction (latest run) -----\n")

    # ---- Summary sheets -----------------------------------------------------
    print("\n[1/4] Extracting Summary Sheets from files 1 and 3...")
    summary_orig, weights_orig = extract_summary_sheet(FILE_ORIG, "ORIG")
    summary_may, weights_may = extract_summary_sheet(FILE_MAY, "MAY26")

    summary_orig.to_csv(os.path.join(PROCESSED, "summary_original.csv"), index=False)
    summary_may.to_csv(os.path.join(PROCESSED, "summary_may2026.csv"), index=False)

    # Persist weights
    pd.Series(weights_orig).to_csv(os.path.join(PROCESSED, "weights_original.csv"))
    pd.Series(weights_may).to_csv(os.path.join(PROCESSED, "weights_may2026.csv"))

    # Sanity: market sets
    o_set = set(summary_orig["Market"])
    m_set = set(summary_may["Market"])
    only_o = o_set - m_set
    only_m = m_set - o_set
    if only_o or only_m:
        log_disc(f"Summary market mismatch — only_orig={only_o}, only_may={only_m}")
    else:
        print(f"  OK: 150 markets matched between orig and may26 Summary Sheets.")

    # Sanity: weights identical
    if weights_orig != weights_may:
        log_disc(f"Weights differ! orig={weights_orig} may={weights_may}")
    else:
        print(f"  OK: weights identical across files.")

    # ---- File 2 (v3) sheets -------------------------------------------------
    print("\n[2/4] Extracting File 2 (Updated_Market_Leveler_v3.xlsx)...")
    v3_sheets = [
        "Asking Rent Growth", "Effective Rent Growth", "Employment",
        "Construction Starts", "Asset Value", "Median Household Income",
        "Population", "Market Sale Price Index", "Market Asking Rent",
        "Market Effective Rent", "Inventory Units", "Units Per Person (UPP)",
    ]
    v3_index = []
    for s in v3_sheets:
        df = extract_v3_sheet(FILE_V3, s)
        slug = s.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(PROCESSED, f"v3_{slug}.csv")
        df.to_csv(out_path)
        v3_index.append({
            "sheet": s, "rows": df.shape[0], "cols": df.shape[1],
            "first_period": df.columns[0] if len(df.columns) else "",
            "last_period": df.columns[-1] if len(df.columns) else "",
            "non_null_frac": float(df.notna().mean().mean()),
            "csv": os.path.basename(out_path),
        })
        print(f"  {s}: {df.shape[0]}x{df.shape[1]} non-null={df.notna().mean().mean():.1%} -> {os.path.basename(out_path)}")
    pd.DataFrame(v3_index).to_csv(os.path.join(PROCESSED, "v3_index.csv"), index=False)

    # ---- Stacked sheets in files 1 and 3 ------------------------------------
    print("\n[3/4] Extracting stacked Forward / Historical sheets...")
    stacked_sheets = [
        "Asking Rent", "Effective Rent-Forward", "Forward Employment",
        "Supply Forward", "Income Growth", "Population Growth Forward",
        "Forward Sale Price", "Forward UPP", "Historical Sale Price",
        "Supply Historical", "UPP Historical",
    ]
    stack_index_rows = []

    for path, label in [(FILE_ORIG, "orig"), (FILE_MAY, "may26")]:
        for sheet in stacked_sheets:
            try:
                blocks = extract_stacked_sheet(path, sheet)
            except Exception as e:
                log_disc(f"FAILED to read {label}:{sheet}: {e}")
                continue
            if not blocks:
                log_disc(f"No tables found in {label}:{sheet}")
                continue
            slug = sheet.lower().replace(" ", "_").replace("-", "_")
            for blk_name, df in blocks.items():
                # Skip blocks with zero columns (empty)
                if df.shape[1] == 0:
                    continue
                out_path = os.path.join(
                    PROCESSED, f"{label}_{slug}_{blk_name}.csv"
                )
                df.to_csv(out_path)
                stack_index_rows.append({
                    "file": label, "sheet": sheet, "block": blk_name,
                    "rows": df.shape[0], "cols": df.shape[1],
                    "first_col": str(df.columns[0]) if len(df.columns) else "",
                    "last_col": str(df.columns[-1]) if len(df.columns) else "",
                    "non_null_frac": float(df.notna().mean().mean()),
                    "csv": os.path.basename(out_path),
                })
                print(f"  [{label}] {sheet} -> {blk_name}: "
                      f"{df.shape[0]}x{df.shape[1]} -> {os.path.basename(out_path)}")
    pd.DataFrame(stack_index_rows).to_csv(
        os.path.join(PROCESSED, "stacked_index.csv"), index=False)

    # ---- Small / non-stacked sheets in file 1 (Effective Rent Historical etc) ---
    # File 1 Effective Rent Historical has a 5-yr and 10-yr mini-table side
    # by side. Extract both into a single CSV.
    print("\n[4/4] Extracting small sheets (file 1)...")

    def extract_orig_erh():
        df = pd.read_excel(FILE_ORIG, sheet_name="Effective Rent Historical", header=None)
        # Header at row 1: cols 0,1,2 = Market, 5 Yr Growth, Rank
        # cols 5,6,7 = Market, 10 Yr Growth, 10 Yr Rank
        body = df.iloc[2:152]
        out = pd.DataFrame({
            "Market": body.iloc[:, 0].apply(normalize_market),
            "EffRent_5YrGrowth": pd.to_numeric(body.iloc[:, 1], errors="coerce"),
            "EffRent_5YrRank": pd.to_numeric(body.iloc[:, 2], errors="coerce"),
            "EffRent_10YrGrowth": pd.to_numeric(body.iloc[:, 6], errors="coerce"),
            "EffRent_10YrRank": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
        })
        return out

    def extract_orig_he():
        df = pd.read_excel(FILE_ORIG, sheet_name="Historical Employment", header=None)
        body = df.iloc[3:153]
        out = pd.DataFrame({
            "Market": body.iloc[:, 1].apply(normalize_market),
            "Emp_5YrGrowth": pd.to_numeric(body.iloc[:, 2], errors="coerce"),
            "Emp_5YrRank": pd.to_numeric(body.iloc[:, 3], errors="coerce"),
            "Emp_10YrGrowth": pd.to_numeric(body.iloc[:, 6], errors="coerce"),
            "Emp_10YrRank": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
        })
        return out[out["Market"] != ""].reset_index(drop=True)

    def extract_orig_hpg():
        df = pd.read_excel(FILE_ORIG, sheet_name="Historical Population Growth", header=None)
        body = df.iloc[2:152]
        out = pd.DataFrame({
            "Market": body.iloc[:, 1].apply(normalize_market),
            "Pop_5YrGrowth": pd.to_numeric(body.iloc[:, 2], errors="coerce"),
            "Pop_5YrRank": pd.to_numeric(body.iloc[:, 3], errors="coerce"),
            "Pop_10YrGrowth": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
            "Pop_10YrRank": pd.to_numeric(body.iloc[:, 8], errors="coerce"),
        })
        return out[out["Market"] != ""].reset_index(drop=True)

    extract_orig_erh().to_csv(
        os.path.join(PROCESSED, "orig_effective_rent_historical_summary.csv"),
        index=False)
    extract_orig_he().to_csv(
        os.path.join(PROCESSED, "orig_historical_employment_summary.csv"),
        index=False)
    extract_orig_hpg().to_csv(
        os.path.join(PROCESSED, "orig_historical_population_summary.csv"),
        index=False)
    print("  orig small sheets extracted.")

    # File 3 historical sheets are full quarterly time series (flat).
    for sheet in ["Effective Rent Historical", "Historical Employment",
                  "Historical Population Growth"]:
        try:
            df = extract_v3_sheet(FILE_MAY, sheet)
            slug = sheet.lower().replace(" ", "_")
            df.to_csv(os.path.join(PROCESSED, f"may26_{slug}.csv"))
            print(f"  may26 {sheet}: {df.shape}")
        except Exception as e:
            log_disc(f"may26 {sheet} extraction failed: {e}")

    # ---- Summary printout ---------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    n_csvs = len([f for f in os.listdir(PROCESSED) if f.endswith(".csv")])
    print(f"  Total CSV files written: {n_csvs}")
    print(f"  Output directory: {PROCESSED}")

    # Sanity cross-check Austin / Indianapolis ranks
    aus_o = summary_orig.loc[summary_orig["Market"] == "Austin - TX",
                             "FinalRank"].values
    aus_m = summary_may.loc[summary_may["Market"] == "Austin - TX",
                            "FinalRank"].values
    ind_o = summary_orig.loc[summary_orig["Market"] == "Indianapolis - IN",
                             "FinalRank"].values
    ind_m = summary_may.loc[summary_may["Market"] == "Indianapolis - IN",
                            "FinalRank"].values
    print(f"\n  Austin: orig rank={aus_o[0] if len(aus_o) else 'n/a'} "
          f"-> may26 rank={aus_m[0] if len(aus_m) else 'n/a'}  (expected 1 -> 32)")
    print(f"  Indianapolis: orig rank={ind_o[0] if len(ind_o) else 'n/a'} "
          f"-> may26 rank={ind_m[0] if len(ind_m) else 'n/a'}  (expected 33 -> 6)")

    print("\nDone.")


if __name__ == "__main__":
    main()
