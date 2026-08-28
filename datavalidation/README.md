# Dataset Validation + Deterministic LSTM Baseline (COMP0197 Group Project)

This folder provides a **dataset validation / sanity-check pipeline** for time-series forecasting.
It is designed for the stage when the group has not finalized the dataset yet and needs a quick,
repeatable way to evaluate candidate datasets.

The script does:
1) **Load** a candidate dataset (local file or remote URL).
2) **Clean** missing values (drop very-missing columns, then fill).
3) **Inspect** the dataset with plots:
   - missing ratio per column
   - correlation heatmap (excluding time-index columns)
   - target-only time series plot
4) Run two baselines:
   - **Persistence baseline** (y[t-1] → y[t]) reported as MAE
   - **Deterministic LSTM baseline** (MSE training) reported as RMSE/MAE on the test split
5) Save diagnostic plots:
   - full test 1-step forecast curve
   - Actual vs Predicted (first N steps, default 200)
   - residual (prediction error) distribution

---

## Files

- `scripts/inspect_and_baseline.py`  
  Main script for dataset loading, inspection, and deterministic baseline training/evaluation.

- `results/`  
  Output folder created automatically (or a custom folder via `--out_dir`).

---

## Supported data inputs

### A) Local files
`--data` can be a local path to:
- `.csv`
- `.csv.zip` (zipped CSV)
- `.xlsx` / `.xls` (Excel)
- `.parquet`

Example:
```bash
python scripts/inspect_and_baseline.py \
  --data data/raw/my_dataset.csv \
  --timestamp_col timestamp \
  --target_col target \
  --lookback 48 \
  --epochs 10 \
  --out_dir results/my_dataset