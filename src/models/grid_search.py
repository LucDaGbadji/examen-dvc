"""Search the best hyper-parameters of a GradientBoostingRegressor.

Output written to models:
    best_params.pkl
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

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
    grid_params = params["grid_search"]

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").to_numpy().ravel()

    estimator = GradientBoostingRegressor(
        random_state=params["model"]["random_state"]
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid_params["param_grid"],
        cv=grid_params["cv"],
        scoring=grid_params["scoring"],
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_params_, MODELS_DIR / "best_params.pkl")

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validated score: {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
