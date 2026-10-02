"""
Streamlit Sales Forecasting App
=================================
Run with:  streamlit run app.py

Lets the user:
- Upload their own transaction-level CSV (or use the bundled Superstore data)
- View EDA (daily trend, category/region breakdown)
- Pick a forecast horizon
- See the forecast chart + table
- Download the forecast as CSV
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
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7)

model = load_model(MODEL_PATH)
if model is None:
    st.sidebar.error(f"Model file '{MODEL_PATH}' not found. Place it next to app.py.")
    st.stop()

# Use uploaded file if given, otherwise fall back to the bundled dataset
data_source = uploaded_file if uploaded_file is not None else "Superstore.csv"
if uploaded_file is None and not os.path.exists("Superstore.csv"):
    st.error("No file uploaded and no default 'Superstore.csv' found. Please upload a CSV.")
    st.stop()

raw, daily = load_and_aggregate(data_source)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("📈 Sales Forecasting Dashboard")
st.caption("Upload your transaction data, or use the bundled dataset, to see historical trends and a forward forecast.")

col1, col2, col3 = st.columns(3)
col1.metric("Total historical sales", f"${daily['sales'].sum():,.0f}")
col2.metric("Days of data", f"{daily.shape[0]:,}")
col3.metric("Avg daily sales", f"${daily['sales'].mean():,.0f}")

tab_eda, tab_forecast = st.tabs(["📊 Exploratory Data", "🔮 Forecast"])

# ---------------- EDA TAB ----------------
with tab_eda:
    st.subheader("Daily Sales Trend")
    st.line_chart(daily.set_index("date")["sales"])

    if "Category" in raw.columns:
        st.subheader("Sales by Category")
        cat_sales = raw.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(cat_sales)

    if "Region" in raw.columns:
        st.subheader("Sales by Region")
        region_sales = raw.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(region_sales)

    zero_pct = (daily["sales"] == 0).mean() * 100
    st.info(f"{zero_pct:.1f}% of days in this dataset have zero recorded sales — "
            f"keep this in mind when judging forecast accuracy.")

# ---------------- FORECAST TAB ----------------
with tab_forecast:
    st.subheader(f"Forecast - Next {horizon} Days")

    featured = add_features(daily).dropna().reset_index(drop=True)
    feature_cols = get_feature_cols(featured)

    # Guard: the uploaded data must have the same feature set the model was trained on
    missing = [c for c in feature_cols if c not in featured.columns]
    if missing:
        st.error(f"Uploaded data can't produce required features: {missing}")
        st.stop()

    with st.spinner("Generating forecast..."):
        future_df = recursive_forecast(model, daily, feature_cols, horizon)

    combined = pd.concat([
        daily.assign(type="Historical"),
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

st.caption("Model: loaded from final_model.pkl, trained on daily-aggregated historical sales with lag/rolling/calendar features.")
