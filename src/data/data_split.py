"""Split the raw dataset into training and testing sets.

Outputs written to data/processed_data:
    X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw_data" / "raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
TARGET = "silica_concentrate"


def load_params() -> dict:
    """Return the parameters stored in params.yaml."""
    with PARAMS_FILE.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    params = load_params()["split"]

    df = pd.read_csv(RAW_FILE)

    # The raw file carries a non numerical 'date' column, which is a
    # timestamp identifier and not an operational parameter of the process.
    non_numeric = list(df.select_dtypes(exclude="number").columns)
    if non_numeric:
        df = df.drop(columns=non_numeric)

    if df.columns[-1] != TARGET:
        raise ValueError(
            f"Expected '{TARGET}' as the last column, found '{df.columns[-1]}'."
        )

    X = df.drop(columns=[TARGET])
    y = df[[TARGET]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print(f"Training set: {X_train.shape} | Testing set: {X_test.shape}")


if __name__ == "__main__":
    main()
