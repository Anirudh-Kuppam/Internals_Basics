import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# Metrics Function
# -----------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    return mae, rmse, r2, mape

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("windcast-turbine-output-mwh")

results = []

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor()
}

best_model_name = None
best_rmse = float("inf")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        mae, rmse, r2, mape = evaluate(model, X_test, y_test)

        # Log params
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        # Tag
        mlflow.set_tag("priority", "high")

        # Save model
        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

# -----------------------------
# Save JSON
# -----------------------------
import json

output = {
    "experiment_name": "windcast-turbine-output-mwh",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed. Results saved.")

import joblib
joblib.dump(model, "../models/model.pkl")