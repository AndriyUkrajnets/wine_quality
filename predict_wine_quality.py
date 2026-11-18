"""Load the exported wine-quality model and predict quality from wine tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "color",
]
ALLOWED_COLORS = {"red", "white"}
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "wine_quality_random_forest.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a model exported from the notebook's Step 12."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run Step 12 in wine_quality.ipynb first."
        )
    return joblib.load(model_path)


def predict_quality(
    wines: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[float]:
    """Return continuous regression predictions for one or more wines.

    Each row must contain the 11 physicochemical measurements and ``color``
    (``"red"`` or ``"white"``). Extra keys are ignored; missing inputs fail
    with a clear error. Predictions are kept continuous rather than rounded.
    """
    if isinstance(wines, pd.DataFrame):
        frame = wines.copy()
    elif isinstance(wines, dict):
        frame = pd.DataFrame([wines])
    else:
        frame = pd.DataFrame(wines)

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    if frame.empty:
        raise ValueError("Provide at least one wine row to predict.")

    invalid_colors = sorted(set(frame["color"].dropna()) - ALLOWED_COLORS)
    if invalid_colors or frame["color"].isna().any():
        raise ValueError("'color' must be 'red' or 'white' for every wine row.")

    numeric_columns = FEATURE_COLUMNS[:-1]
    try:
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All 11 physicochemical inputs must be numeric.") from exc
    numeric_values = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Physicochemical inputs must not contain missing or infinite values.")

    model = model if model is not None else load_model(model_path)
    predictions = model.predict(frame[FEATURE_COLUMNS])
    return [float(value) for value in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict wine quality from a JSON file containing one wine or a list of wines."
    )
    parser.add_argument("input_json", type=Path, help="Path to JSON object or array of wine inputs")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the joblib model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    with args.input_json.open(encoding="utf-8") as file:
        wines = json.load(file)
    predictions = predict_quality(wines, model_path=args.model)
    print(json.dumps({"predicted_quality": predictions}, indent=2))


if __name__ == "__main__":
    main()
