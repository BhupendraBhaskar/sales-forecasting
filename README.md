# 📈 Sales Forecasting

An end-to-end weekly sales forecasting project on the Superstore retail dataset: exploratory analysis, feature engineering, tuned model benchmarking with time-series cross-validation, and an interactive Streamlit dashboard that forecasts sales forward.

## Highlights

- Aggregates ~10K transaction rows into a **weekly** sales series (weekly forecasting roughly halved relative error compared to daily, see [Results](#results))
- Features: calendar, US holiday flag, Fourier seasonality terms, lag / rolling / EWM features, and lagged order count, quantity and discount
- Compares Ridge, Random Forest and Gradient Boosting (tuned with `TimeSeriesSplit` + `GridSearchCV`) against naive and moving-average baselines
- Honest evaluation: chronological holdout, models scored only on data they never saw
- Streamlit app with EDA tab, forecast horizon slider, forecast table and CSV download
- Notebook and app share one module (`sales_forecast_utils.py`), so training and serving use identical feature code

## Project structure

```
.
├── app.py                                        # Streamlit dashboard
├── train_model.py                                # Rebuilds final_model.pkl locally
├── sales_forecast_utils.py                       # Shared feature engineering + forecasting logic
├── sales_forecasting_notebook.ipynb              # EDA, features, benchmarking, forecast
├── Superstore.csv                                # Dataset
├── final_model.pkl                               # Trained model (rebuild with train_model.py)
├── future_forecast.csv                           # Latest forecast output
├── requirements.txt
├── pipeline_comparison_A_to_D.py                 # Reproduces the before/after comparison below
├── sales_forecast_utils_v1_archive.py            # Original daily-only pipeline (reference)
└── sales_forecast_utils_v2_flexible_archive.py   # Daily/weekly variant used by the comparison script
```

## Setup

Download or clone this repository, open a terminal in the project folder, then:

```bash
git clone https://github.com/<BhupendraBhaskar>/<sales-forecasting>.git
cd <sales-forecasting>
pip install -r requirements.txt
```

To run the notebook you also need Jupyter (`pip install jupyter`).

## Usage

**1. Build the model** (do this first, and again whenever the feature code or your scikit-learn version changes):

```bash
python train_model.py
```

This tunes the models, prints holdout metrics, retrains the best one on all data and saves `final_model.pkl`.

**2. Launch the dashboard:**

```bash
streamlit run app.py
```

You can view weekly trends and category / region breakdowns, upload your own transaction CSV, pick a forecast horizon (4 to 52 weeks) and download the forecast as CSV.

**3. (Optional) Explore the notebook:**

```bash
jupyter notebook sales_forecasting_notebook.ipynb
```

## Results

Evaluated on a chronological holdout of the final 13 weeks (no shuffling). Models were fitted on the training split only.

| Model | Mean error (bias) | MAE | RMSE | RMSE / mean sales |
|---|---|---|---|---|
| Naive (last week) | n/a | 13,250 | 14,269 | 66.9% |
| 4-week moving average | n/a | 7,443 | 9,447 | 44.3% |
| Ridge Regression | 1,631 | 8,176 | 9,394 | 44.1% |
| Random Forest (tuned) | 4,095 | 8,020 | 9,658 | 45.3% |
| **Gradient Boosting (tuned)** | 2,757 | 7,497 | **9,358** | **43.9%** |

**How to read this honestly:**

- The model clearly beats the naive forecast, but it is **essentially tied with a plain 4-week moving average** (the moving average even has a slightly lower MAE). The top models differ by less than 1 percentage point on only 13 weeks, which is within noise.
- Mean error is positive for every model, meaning they slightly under-predict on average. With 13 weeks this is not statistically conclusive.
- Predictions mostly hover around the average and miss the largest weekly swings, especially in Q4.

### How the pipeline evolved

The project started with **daily** forecasting. Daily sales were extremely noisy (coefficient of variation about 1.4, about 15% of days with zero sales), so even the best model's RMSE was about 97% of mean daily sales.

| Pipeline | Best model | RMSE | RMSE as % of mean |
|---|---|---|---|
| (A) Original: daily, lag/rolling features only | Random Forest | 2,977 | 96.7% |
| (B) + external / holiday / Fourier / EWM features, still daily | Gradient Boosting | 2,883 | 93.6% |
| (C) Same as (B), **weekly aggregation** (current pipeline) | Gradient Boosting | 9,368 | **44.0%** |
| (D) Same as (C) + log1p target transform | Gradient Boosting | 10,472 | 49.1% |

Weekly aggregation was the change that mattered. The log transform was tried and dropped because it tested worse. Reproduce this table with `pipeline_comparison_A_to_D.py`. (Percentages are comparable across rows; raw RMSE values are not, because daily and weekly sales are on different scales.)

## Limitations

- This is a noisy retail series with about 4 years of history. Treat forecasts as directional, not precise.
- Results come from a single 13-week holdout. A walk-forward validation across more windows would give a more reliable estimate.
- During recursive forecasting, order count, quantity and discount are carried forward from their last known value, since future values are not known in advance.
- XGBoost, LightGBM, Prophet and SARIMA were not benchmarked. They are a natural next step.
- Never score the model with `final_model.pkl` on the holdout weeks. It was retrained on all data and has already seen them, so the numbers would look unrealistically good.

## Troubleshooting

**`ValueError: The feature names should match those that were passed during fit`**, or the app showing a "final_model.pkl doesn't match the current feature code" message: your saved model was trained with a different version of the feature code. Run `python train_model.py` to rebuild it, then restart the app. Make sure `app.py`, `sales_forecast_utils.py` and `final_model.pkl` all come from the same version.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib
