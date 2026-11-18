# Wine quality model deployment

The exported model is a scikit-learn pipeline: it includes both categorical preprocessing for wine color and the tuned Random Forest regressor.

## Create or rebuild the model artifact

For prediction in a fresh environment, install the pinned runtime packages:

```powershell
python -m pip install -r requirements-model.txt
```

To rebuild the artifact, also install `ucimlrepo`, then run the following from this folder using the notebook's Python environment:

```powershell
python build_model_artifact.py
```

The script recreates the grouped train/test split, fits the selected Random Forest on the training partition, and creates:

- `artifacts/wine_quality_random_forest.joblib` — the fitted pipeline.
- `artifacts/wine_quality_model_metadata.json` — input columns, evaluation metrics, and library versions.

The model is trained on the training partition only; the held-out test partition is not used for fitting. Its recorded held-out test MAE was 0.518 and R-squared was 0.405. Scores are continuous regression estimates and may be fractional.

## Predict from Python

With the pinned dependencies installed, use:

```python
from predict_wine_quality import predict_quality

wine = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "color": "red",
}

quality_estimate = predict_quality(wine)[0]
print(quality_estimate)
```

The input must include all 11 physicochemical measurements and `color` (`red` or `white`). The model applies the same preprocessing used during training. Predictions are continuous; round only if the consuming application needs an integer score.

## Predict from the command line

With the example input file in this folder:

```powershell
python predict_wine_quality.py example_wine.json
```

For another deployment environment, copy the model artifact, `predict_wine_quality.py`, and metadata file together. Use the versions in `requirements-model.txt` because joblib model files depend on compatible scikit-learn and Python package versions.
