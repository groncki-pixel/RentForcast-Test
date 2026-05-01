# RentForecast Market Leveler Analysis

## Overview
Analysis of multifamily market ranking model ("Market Leveler") across two data vintages to identify:
1. Which markets experienced ranking shocks and what drove them
2. Whether forecast estimates were materially revised (data quality check)
3. Which input variables are too volatile to rank on reliably
4. Whether the current weightings are optimal or should be revised
5. Whether we can build a proprietary effective rent forecast

## Data Files
- Full_Market_Leveler_Original.xlsx — Original model (Q2 2025 estimates). Austin=#1, Indianapolis=#33
- Updated_Market_Leveler_v3.xlsx — Raw data refresh (Apr 2026 MSA DataExport). Actuals + revised forecasts
- Full_Market_Leveler_May2026.xlsx — Updated model (May 2026 data). Austin=#32, Indianapolis=#6

## Model Structure
150 US multifamily markets ranked across 8 categories with weighted scoring:

- Effective Rent (20%) — Forward 1/3/5/10 yr
- Units Per Person UPP (20%) — Historical 10, Forward 5/10
- Sale Price (20%) — Historical 1/3/5/10, Forward 1/3/5/10
- Employment (10%) — Historical 5/10, Forward 1/3/5/10
- Supply (10%) — Historical 5, Forward 1/3/5/10
- Income (~10%) — Forward 1/3/5/10
- Population (10%) — Historical 5/10, Forward 1/3/5/10
- Asking Rent (0% tracked) — Forward 1/3/5/10

## Analysis Pipeline

- 01_data_extraction.py — Parse all 3 files into clean dataframes
- 02_shock_detection.py — Ranking changes and decomposition
- 03_estimate_vs_actual.py — Forecast error analysis
- 04_volatility.py — Forecast revision dispersion
- 05_weight_sensitivity.py — Backtesting optimal weights
- 06_rent_model.py — Proprietary rent forecast (holy grail)

## Key Questions
- What data materially changed our rankings?
- Were our estimates garbage or did fundamentals actually shift?
- Should we discount forward estimates beyond X quarters?
- Is effective rent forward too volatile to rank on?
- Can we build our own effective rent forecast that beats the vendor?
