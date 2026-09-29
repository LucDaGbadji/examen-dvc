"""Evaluate the trained model, store the predictions and the metrics.

Outputs:
    data/prediction.csv
    metrics/scores.json
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"


def main() -> None:
    model = joblib.load(MODELS_DIR / "gbr_model.pkl")

    X_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").to_numpy().ravel()

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {"silica_concentrate": y_test, "prediction": y_pred}
    )
    predictions.to_csv(DATA_DIR / "prediction.csv", index=False)

    mse = float(mean_squared_error(y_test, y_pred))
    scores = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "scores.json").open("w", encoding="utf-8") as stream:
        json.dump(scores, stream, indent=4)

    print(f"Evaluation metrics: {scores}")


if __name__ == "__main__":
    main()
