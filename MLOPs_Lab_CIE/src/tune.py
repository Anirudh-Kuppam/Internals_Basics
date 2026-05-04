import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, ParameterSampler, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import random

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("../data/training_data.csv")

X = data.drop("turbine_output_mwh", axis=1)
y = data["turbine_output_mwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Parameter Grid
# -----------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

# Random search setup
n_trials = 5
param_list = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_rmse = float("inf")
best_params = None
best_mae = None
best_cv_mae = None

mlflow.set_experiment("windcast-turbine-output-mwh")

# -----------------------------
# Parent Run
# -----------------------------
with mlflow.start_run(run_name="tuning-windcast"):

    for i, params in enumerate(param_list):

        with mlflow.start_run(run_name=f"trial_{i}", nested=True):

            model = GradientBoostingRegressor(**params)

            cv_mae_scores = []
            cv_rmse_scores = []

            # Cross-validation
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                mae = mean_absolute_error(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))

                cv_mae_scores.append(mae)
                cv_rmse_scores.append(rmse)

            avg_mae = np.mean(cv_mae_scores)
            avg_rmse = np.mean(cv_rmse_scores)

            # Log params + metrics
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", avg_mae)
            mlflow.log_metric("cv_rmse", avg_rmse)

            # Track best
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = params
                best_cv_mae = avg_mae

    # Train best model on full train set
    best_model = GradientBoostingRegressor(**best_params)
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, preds)

# -----------------------------
# Save JSON
# -----------------------------
output = {
    "search_type": "random",
    "n_folds": 5,
    "total_trials": n_trials,
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-windcast"
}

with open("../results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed. Results saved.")