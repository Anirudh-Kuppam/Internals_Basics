import pandas as pd
import numpy as np
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Train model (reuse best params from Task 2)
# -----------------------------
data = pd.read_csv("../data/training_data.csv")

X = data.drop("turbine_output_mwh", axis=1)
y = data["turbine_output_mwh"]

# Best params from Task 2 (update if yours differ)
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

model.fit(X, y)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# Input validation
class InputData(BaseModel):
    wind_speed_kmph: float = Field(..., ge=5, le=60)
    blade_length_m: float = Field(..., ge=20, le=80)
    altitude_m: float = Field(..., ge=50, le=200)
    humidity_pct: float = Field(..., ge=20, le=90)

# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "GradientBoosting",
        "version": "1.0"
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/estimate")
def estimate(input_data: InputData):
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Save JSON output (for exam)
# -----------------------------
def save_result():
    test_input = {
        "wind_speed_kmph": 37.7,
        "blade_length_m": 59.5,
        "altitude_m": 126.6,
        "humidity_pct": 58.7
    }

    df = pd.DataFrame([test_input])
    prediction = model.predict(df)[0]

    output = {
        "health_endpoint": "/health",
        "predict_endpoint": "/estimate",
        "port": 8080,
        "health_response": {
            "status": "running",
            "model": "GradientBoosting",
            "version": "1.0"
        },
        "test_input": test_input,
        "prediction": float(prediction)
    }

    with open("../results/step3_s4.json", "w") as f:
        json.dump(output, f, indent=4)


# Run save when file executed directly
if __name__ == "__main__":
    save_result()