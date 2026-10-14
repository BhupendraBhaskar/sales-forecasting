"""
Streamlit Sales Forecasting App
=================================
Run with:  streamlit run app.py

Lets the user:
- Upload their own transaction-level CSV (or use the bundled Superstore data)
- View EDA (weekly trend, category/region breakdown)
- Pick a forecast horizon (in weeks)
- See the forecast chart + table
- Download the forecast as CSV

NOTE: forecasting is done on WEEKLY-aggregated sales, not daily. Benchmarking
showed weekly aggregation roughly halves relative forecast error compared to
daily (noisy day-to-day sales were hard for any model to predict well) - see
README.md "Results" section for the before/after comparison.
"""

import streamlit as st
import pandas as pd
import joblib
import os

from sales_forecast_utils import (
    load_and_aggregate as _load_and_aggregate_raw,
    add_features,
    get_feature_cols,
    recursive_forecast,
    check_model_compatibility,
)

st.set_page_config(page_title="Sales Forecasting", layout="wide")

MODEL_PATH = "final_model.pkl"


# ------------------------------------------------------------------
# Cached wrappers around the shared utils - caching lives here (app-only
# concern), the actual logic lives in sales_forecast_utils.py so the
# notebook uses the exact same code.
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_and_aggregate(file):
    return _load_and_aggregate_raw(file)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
horizon = st.sidebar.slider("Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=4)

model = load_model(MODEL_PATH)
if model is None:
    st.sidebar.error(f"Model file '{MODEL_PATH}' not found. Place it next to app.py.")
    st.stop()

# Use uploaded file if given, otherwise fall back to the bundled dataset
data_source = uploaded_file if uploaded_file is not None else "Superstore.csv"
if uploaded_file is None and not os.path.exists("Superstore.csv"):
    st.error("No file uploaded and no default 'Superstore.csv' found. Please upload a CSV.")
    st.stop()

raw, weekly = load_and_aggregate(data_source)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("📈 Sales Forecasting Dashboard")
st.caption("Upload your transaction data, or use the bundled dataset, to see historical trends and a forward forecast. Forecasting runs on weekly-aggregated sales.")

col1, col2, col3 = st.columns(3)
col1.metric("Total historical sales", f"${weekly['sales'].sum():,.0f}")
col2.metric("Weeks of data", f"{weekly.shape[0]:,}")
col3.metric("Avg weekly sales", f"${weekly['sales'].mean():,.0f}")

tab_eda, tab_forecast = st.tabs(["📊 Exploratory Data", "🔮 Forecast"])

# ---------------- EDA TAB ----------------
with tab_eda:
    st.subheader("Weekly Sales Trend")
    st.line_chart(weekly.set_index("date")["sales"])

    if "Category" in raw.columns:
        st.subheader("Sales by Category")
        cat_sales = raw.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(cat_sales)

    if "Region" in raw.columns:
        st.subheader("Sales by Region")
        region_sales = raw.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(region_sales)

    st.info("Forecasting is done on weekly-aggregated sales rather than daily — "
            "daily totals were too noisy for any model to predict well "
            "(see README.md for the before/after comparison).")

# ---------------- FORECAST TAB ----------------
with tab_forecast:
    st.subheader(f"Forecast - Next {horizon} Weeks")

    featured = add_features(weekly).dropna().reset_index(drop=True)
    feature_cols = get_feature_cols(featured)

    # Guard: the saved model must have been trained on exactly the features
    # this version of the code builds. Mismatch = stale final_model.pkl.
    ok, missing, unexpected = check_model_compatibility(model, feature_cols)
    if not ok:
        st.error(
            "final_model.pkl doesn't match the current feature code - it was "
            "most likely trained with an older version of the pipeline.\n\n"
            "Fix: run `python train_model.py` (or re-run the notebook) to "
            "rebuild final_model.pkl, then restart the app."
        )
        with st.expander("Technical details"):
            st.write("Features the model expects but the code no longer builds:", missing)
            st.write("Features the code builds but the model never saw:", unexpected)
        st.stop()

    with st.spinner("Generating forecast..."):
        future_df = recursive_forecast(model, weekly, feature_cols, horizon)

    combined = pd.concat([
        weekly.assign(type="Historical"),
        future_df.assign(type="Forecast"),
    ])
    chart_data = combined.pivot(index="date", columns="type", values="sales")
    st.line_chart(chart_data)

    st.subheader("Forecast Table")
    st.dataframe(future_df, use_container_width=True)

    csv_bytes = future_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast as CSV",
        data=csv_bytes,
        file_name="sales_forecast.csv",
        mime="text/csv",
    )

st.caption("Model: loaded from final_model.pkl, trained on weekly-aggregated historical sales with lag/rolling/EWM/calendar/holiday/external features.")
