"""Rebuild and export the tuned wine-quality regression pipeline.

Run with the same Python environment used by wine_quality.ipynb:
    python build_model_artifact.py

The grouped train/test split and selected Random Forest settings mirror the
notebook. The held-out test rows are never used to fit this model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MODEL_PATH = ARTIFACTS / "wine_quality_random_forest.joblib"
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
NUMERIC_FEATURES = FEATURE_COLUMNS[:-1]


def main() -> None:
    # Fetch the same combined red/white wine table as the notebook.
    dataset = fetch_ucirepo(id=186)
    X_source = dataset.data.features.reset_index(drop=True)
    y_source = dataset.data.targets.reset_index(drop=True)
    original = dataset.data.original.reset_index(drop=True)
    quality_column = y_source.columns[0]

    if "color" not in original.columns:
        raise ValueError("UCI wine dataset source table does not contain 'color'.")

    X = X_source.copy()
    X["color"] = original["color"]
    y = y_source[quality_column]

    # Keep identical predictor profiles together across train/test and CV.
    groups = pd.MultiIndex.from_frame(X[FEATURE_COLUMNS]).factorize()[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(outer.split(X, y, groups=groups))
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    # The settings match the best grouped-CV configuration recorded in Step 8.
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("color", OneHotEncoder(handle_unknown="ignore"), ["color"]),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    max_features=0.5,
                    min_samples_leaf=1,
                    max_samples=0.9,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model": "RandomForestRegressor",
        "target": quality_column,
        "input_features": FEATURE_COLUMNS,
        "categorical_features": ["color"],
        "training_rows": int(len(train_idx)),
        "held_out_test_rows": int(len(test_idx)),
        "training_cv_mae": 0.528,
        "held_out_test_metrics": {"MAE": 0.518, "RMSE": 0.673, "R2": 0.405},
        "held_out_metrics_source": "wine_quality.ipynb Step 10; reported, not recomputed by this export script",
        "random_forest_parameters": {
            "n_estimators": 400,
            "max_features": 0.5,
            "min_samples_leaf": 1,
            "max_samples": 0.9,
            "max_depth": None,
            "random_state": 42,
        },
        "python_version": __import__("sys").version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }
    (ARTIFACTS / "wine_quality_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (HERE / "requirements-model.txt").write_text(
        "\n".join(
            [
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"joblib=={joblib.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Training rows: {len(train_idx)}; held-out rows: {len(test_idx)}")
    print(f"Model size: {MODEL_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
