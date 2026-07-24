import pandas as pd
import numpy as np

def compute_price_paths(pred_mean, all_preds, df_raw, test_df, price_col='GBP-USD', target_col='target', ci=95):
    """
    Compute actual next prices and prediction intervals from log-return predictions.
    """
    alpha = (100 - ci) / 2

    # Align anchor prices to test dates
    price_df = df_raw[["Date", price_col]].copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    test_dates = pd.to_datetime(test_df["Date"]).reset_index(drop=True)

    test_anchor_prices = (
        test_dates.to_frame(name="Date")
        .merge(price_df, on="Date", how="left")[price_col]
        .to_numpy()
    )

    targets = test_df[target_col].to_numpy()
    T = len(targets)

    # Actual next prices (ground truth)
    actual_next_price = test_anchor_prices * np.exp(targets)

    # Mean 1-step-ahead predictions
    walk_forward_pred = test_anchor_prices * np.exp(pred_mean)

    # MC Dropout uncertainty bands
    all_step_preds = (test_anchor_prices[:, None] * np.exp(all_preds.T)).T  # (mc_samples, T)
    lower_bound = np.percentile(all_step_preds, alpha, axis=0)
    upper_bound = np.percentile(all_step_preds, 100 - alpha, axis=0)

    return actual_next_price, walk_forward_pred, lower_bound, upper_bound