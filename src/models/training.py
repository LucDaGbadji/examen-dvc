"""Train the regression model with the parameters found by the grid search.

Output written to models:
    gbr_model.pkl
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    """Return the parameters stored in params.yaml."""
    with PARAMS_FILE.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    params = load_params()
    best_params = joblib.load(MODELS_DIR / "best_params.pkl")

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").to_numpy().ravel()

    model = GradientBoostingRegressor(
        random_state=params["model"]["random_state"], **best_params
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODELS_DIR / "gbr_model.pkl")

    print(f"Model trained with: {best_params}")
    print(f"Model saved to: {MODELS_DIR / 'gbr_model.pkl'}")


if __name__ == "__main__":
    main()
