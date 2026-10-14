"""
Rebuild final_model.pkl on YOUR machine.

Run:  python train_model.py

Why this exists: a pickled model is tied to the scikit-learn version and the
feature code it was trained with. If the app complains that final_model.pkl
doesn't match the feature code (or you upgraded Python/scikit-learn), run this
script to regenerate it, then restart the Streamlit app.

Steps: load -> weekly features -> chronological split -> tuned models (with
TimeSeriesSplit CV) -> pick best on the HOLDOUT (train-only models) -> retrain
the winner on all data -> save final_model.pkl.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sales_forecast_utils import load_and_aggregate, add_features, get_feature_cols

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "Superstore.csv"
TEST_SIZE_WEEKS = 13

raw, weekly = load_and_aggregate(DATA_PATH)
featured = add_features(weekly).dropna().reset_index(drop=True)
feature_cols = get_feature_cols(featured)

cutoff = featured["date"].max() - pd.Timedelta(weeks=TEST_SIZE_WEEKS)
train = featured[featured["date"] <= cutoff]
test = featured[featured["date"] > cutoff]
X_train, y_train = train[feature_cols], train["sales"]
X_test, y_test = test[feature_cols], test["sales"]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

tscv = TimeSeriesSplit(n_splits=4)
rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {"n_estimators": [200, 400], "max_depth": [6, 10, None]},
    cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
).fit(X_train, y_train)
gbr = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {"n_estimators": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4]},
    cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
).fit(X_train, y_train)

candidates = {
    "Ridge Regression": Ridge(alpha=1.0).fit(X_train, y_train),
    "Random Forest (tuned)": rf.best_estimator_,
    "Gradient Boosting (tuned)": gbr.best_estimator_,
}

# Honest evaluation: these models were fitted on the TRAIN split only.
rows = {}
for name, m in candidates.items():
    p = m.predict(X_test)
    err = y_test.values - p
    rows[name] = {
        "Mean error (bias)": err.mean(),
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
    }
    rows[name]["RMSE % of mean"] = rows[name]["RMSE"] / y_test.mean() * 100

print("\nHoldout results (models trained on train split only):")
print(pd.DataFrame(rows).T.round(1).sort_values("RMSE"))

best_name = min(rows, key=lambda k: rows[k]["RMSE"])
best = candidates[best_name]
print(f"\nBest on holdout: {best_name}")

# Retrain the winner on ALL data for forecasting. Do NOT use this model to
# score the test weeks - it has already seen them.
final_model = type(best)(**best.get_params())
final_model.fit(featured[feature_cols], featured["sales"])
joblib.dump(final_model, "final_model.pkl")
print("Saved final_model.pkl")
