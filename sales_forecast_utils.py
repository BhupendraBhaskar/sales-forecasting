"""
Shared logic for the Sales Forecasting project - FINAL version.

This replaces the original daily-only sales_forecast_utils.py after
benchmarking showed weekly aggregation + external features cuts relative
forecast error roughly in half (RMSE as % of mean sales: ~97% -> ~44%).

Both the training notebook and app.py import from this module, so feature
engineering is guaranteed identical at training time and prediction time.

What changed vs. the original version, and why:
  - Aggregation is WEEKLY, not daily (biggest single improvement - daily
    sales were too noisy for any model to forecast well)
  - Added real external features: order count, total quantity, avg discount
    (previously unused columns in the raw data)
  - Added a US holiday flag (pandas' built-in calendar, no extra install)
  - Added Fourier (sin/cos) terms for smoother yearly seasonality
  - Added an exponentially-weighted moving average feature
  - Log1p target transform was tried and DROPPED - it tested worse here
    (median-bias issue when transforming predictions back to raw scale)

Not included (would require packages unavailable in the dev sandbox used to
build this - no internet access there to install them): XGBoost, LightGBM,
Prophet, SARIMA. Install them locally and add them to the `candidates` dict
in the notebook/benchmark script the same way Ridge/RandomForest are added.
"""

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

LAGS = [1, 4, 8, 12]            # weeks
ROLLING_WINDOWS = [4, 8, 12]    # weeks


def load_and_aggregate(file_or_path):
    """Load transaction-level CSV and aggregate to a WEEKLY series,
    including external features: order count, quantity, avg discount."""
    raw = pd.read_csv(file_or_path, encoding="latin1")
    raw["Order Date"] = pd.to_datetime(raw["Order Date"], format="%m/%d/%Y")

    daily = raw.groupby("Order Date").agg(
        sales=("Sales", "sum"),
        order_count=("Order ID", "nunique"),
        quantity=("Quantity", "sum"),
        avg_discount=("Discount", "mean"),
    )
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily[["sales", "order_count", "quantity"]] = daily[["sales", "order_count", "quantity"]].fillna(0)
    daily["avg_discount"] = daily["avg_discount"].fillna(0)
    daily.index.name = "date"
    daily = daily.reset_index()

    weekly = daily.set_index("date").resample("W-MON", label="left", closed="left").agg({
        "sales": "sum", "order_count": "sum", "quantity": "sum", "avg_discount": "mean"
    }).reset_index()

    return raw, weekly


def add_features(df):
    """Calendar + holiday + Fourier + external + lag/rolling/EWM features,
    all computed on weekly-aggregated data."""
    d = df.copy()
    d["month"] = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype(int)
    d["is_month_start"] = d["date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["date"].dt.is_month_end.astype(int)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=d["date"].min(), end=d["date"].max() + pd.Timedelta(days=14))
    # a week "contains a holiday" if any of its 7 days is a US federal holiday
    d["contains_holiday"] = d["date"].apply(
        lambda wk_start: int(any((wk_start + pd.Timedelta(days=i)) in holidays for i in range(7)))
    )

    day_of_year = d["date"].dt.dayofyear
    for k in [1, 2]:
        d[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        d[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    for lag in LAGS:
        d[f"lag_{lag}"] = d["sales"].shift(lag)
    for w in ROLLING_WINDOWS:
        shifted = d["sales"].shift(1)
        d[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        d[f"rolling_std_{w}"] = shifted.rolling(w).std()
        d[f"ewm_mean_{w}"] = shifted.ewm(span=w).mean()

    # External signal, lagged by 1 week so we never use "this week's" own
    # order count/quantity to predict "this week's" sales (unknown in advance
    # in a real forecasting scenario)
    for col in ["order_count", "quantity", "avg_discount"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    d = d.drop(columns=["order_count", "quantity", "avg_discount"])
    return d


def get_feature_cols(featured_df):
    return [c for c in featured_df.columns if c not in ["date", "sales"]]


def recursive_forecast(model, weekly, feature_cols, horizon_weeks):
    """Forecast forward week by week. External-signal lag features are
    carried forward using their last known value during the horizon,
    since future order counts/quantity aren't knowable in advance."""
    history = weekly.copy()
    future_preds = []

    last_known = {
        "order_count": history["order_count"].iloc[-1] if "order_count" in history else 0,
        "quantity": history["quantity"].iloc[-1] if "quantity" in history else 0,
        "avg_discount": history["avg_discount"].iloc[-1] if "avg_discount" in history else 0,
    }

    for _ in range(horizon_weeks):
        next_date = history["date"].max() + pd.Timedelta(weeks=1)
        new_row = pd.DataFrame({
            "date": [next_date], "sales": [np.nan],
            "order_count": [last_known["order_count"]],
            "quantity": [last_known["quantity"]],
            "avg_discount": [last_known["avg_discount"]],
        })
        temp = pd.concat([history, new_row], ignore_index=True)
        temp_featured = add_features(temp)
        row = temp_featured.iloc[[-1]][feature_cols]

        pred = max(model.predict(row)[0], 0)
        future_preds.append({"date": next_date, "sales": pred})
        history = pd.concat([history, pd.DataFrame({
            "date": [next_date], "sales": [pred],
            "order_count": [last_known["order_count"]],
            "quantity": [last_known["quantity"]],
            "avg_discount": [last_known["avg_discount"]],
        })], ignore_index=True)

    return pd.DataFrame(future_preds)


def check_model_compatibility(model, feature_cols):
    """Compare the features the app builds against the features the saved
    model was trained on. Returns (ok, missing_from_data, unexpected_in_data).
    Catches the classic 'old model.pkl + new feature code' mismatch before
    model.predict() raises a cryptic sklearn ValueError."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:      # model trained on a plain array - can't verify
        return True, [], []
    expected = list(expected)
    missing = [c for c in expected if c not in feature_cols]
    unexpected = [c for c in feature_cols if c not in expected]
    return (not missing and not unexpected), missing, unexpected
