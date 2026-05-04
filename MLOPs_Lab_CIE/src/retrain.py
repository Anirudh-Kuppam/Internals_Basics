import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load original data
# -----------------------------
train_data = pd.read_csv("../data/training_data.csv")
new_data = pd.read_csv("../data/new_data.csv")

# -----------------------------
# Combine data
# -----------------------------
combined_data = pd.concat([train_data, new_data], ignore_index=True)

# -----------------------------
# Prepare original model (champion)
# -----------------------------
X_train_orig = train_data.drop("turbine_output_mwh", axis=1)
y_train_orig = train_data["turbine_output_mwh"]

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_train_orig, y_train_orig, test_size=0.2, random_state=42
)

# Best params from Task 2 (update if needed)
champion_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

champion_model.fit(X_train_o, y_train_o)
champion_preds = champion_model.predict(X_test_o)
champion_mae = mean_absolute_error(y_test_o, champion_preds)

# -----------------------------
# Retrained model on combined data
# -----------------------------
X_combined = combined_data.drop("turbine_output_mwh", axis=1)
y_combined = combined_data["turbine_output_mwh"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

retrained_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

retrained_model.fit(X_train_c, y_train_c)
retrained_preds = retrained_model.predict(X_test_c)
retrained_mae = mean_absolute_error(y_test_c, retrained_preds)

# -----------------------------
# Compare and decide
# -----------------------------
improvement = champion_mae - retrained_mae

if retrained_mae < champion_mae:
    action = "promoted"
else:
    action = "kept_champion"

# -----------------------------
# Save JSON
# -----------------------------
output = {
    "original_data_rows": len(train_data),
    "new_data_rows": len(new_data),
    "combined_data_rows": len(combined_data),
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "mae"
}

with open("../results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed. Results saved.")