"""
Shared logic for the Sales Forecasting project.

Both the training notebook (sales_forecasting_notebook.ipynb) and the
Streamlit app (app.py) import from this module, so feature engineering
is guaranteed to be IDENTICAL at training time and prediction time.

If you ever change a feature here, both the notebook and the app
pick up the change automatically next time they run.
"""

import pandas as pd
import numpy as np

LAGS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]


def load_and_aggregate(file_or_path):
    """Load transaction-level CSV and aggregate to a daily sales series."""
    raw = pd.read_csv(file_or_path, encoding="latin1")
    raw["Order Date"] = pd.to_datetime(raw["Order Date"], format="%m/%d/%Y")

    daily = raw.groupby("Order Date")["Sales"].sum().rename("sales").to_frame()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx).fillna(0)
    daily.index.name = "date"
    daily = daily.reset_index()
    return raw, daily


def add_features(df):
    """Add calendar + lag + rolling features. Used identically for
    training data and for each recursive forecast step."""
    d = df.copy()
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    d["is_weekend"] = d["day_of_week"].isin([5, 6]).astype(int)
    d["is_month_start"] = d["date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["date"].dt.is_month_end.astype(int)
    for lag in LAGS:
        d[f"lag_{lag}"] = d["sales"].shift(lag)
    for w in ROLLING_WINDOWS:
        d[f"rolling_mean_{w}"] = d["sales"].shift(1).rolling(w).mean()
        d[f"rolling_std_{w}"] = d["sales"].shift(1).rolling(w).std()
    return d


def get_feature_cols(featured_df):
    return [c for c in featured_df.columns if c not in ["date", "sales"]]


def recursive_forecast(model, daily, feature_cols, horizon_days):
    """Forecast forward day by day. Each predicted day is appended to
    history so the next day's lag/rolling features can be computed."""
    history = daily.copy()
    future_preds = []
    for _ in range(horizon_days):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        temp = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [np.nan]})],
            ignore_index=True,
        )
        temp_featured = add_features(temp)
        row = temp_featured.iloc[[-1]][feature_cols]
        pred = max(model.predict(row)[0], 0)
        future_preds.append({"date": next_date, "sales": pred})
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [pred]})],
            ignore_index=True,
        )
    return pd.DataFrame(future_preds)
