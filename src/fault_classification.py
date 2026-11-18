"""Fault-window classification.

The selected classifier is HistGradientBoosting. Those weights were not
saved. Logistic regression is loaded only when the caller asks for the
serialized reference by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import WINDOW_FEATURE_NAMES, window_frame
from src.model_loader import WeightsNotPersisted, scale
from src.preprocessing import VOLTAGE_COLUMNS


def refuse_selected_classifier(package: dict) -> None:
    status = package["classifier_status"]
    raise WeightsNotPersisted(
        "The validation-selected classifier is "
        f"{status.get('selected_model')} (window {status.get('window_s')} s, "
        f"class weight {status.get('class_weight')}). "
        "Its weights were not saved (WEIGHTS_NOT_PERSISTED). "
        "Logistic regression is a different model and is not used as a substitute. "
        "Pass --model logistic_reference to run the only serialized classifier."
    )


def ridge_abs_residual(frame: pd.DataFrame, predicted: np.ndarray) -> np.ndarray:
    residual = np.full((len(frame), 4), np.nan, dtype=np.float64)
    scored = frame["pair_status"].to_numpy() == "scored"
    actual = frame.loc[scored, VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
    residual[scored] = np.abs(actual - predicted[scored])
    return residual


def predict_logistic_reference(frame: pd.DataFrame, predicted_voltage: np.ndarray, package: dict) -> pd.DataFrame:
    config = package["inference"]
    windows = window_frame(
        frame,
        ridge_abs_residual(frame, predicted_voltage),
        float(config["classifier_window_s"]),
        float(config["max_gap_s"]),
        float(config["window_min_fraction"]),
        int(config["window_min_samples"]),
    )
    model = package["logistic_reference"]
    columns = [
        "t_s", "run_id", "row_status", "predicted_fault_type",
        "prediction_confidence", "model_name", "selected_model", "model_version",
    ]
    if windows.empty:
        return pd.DataFrame(columns=columns)
    raw = windows[WINDOW_FEATURE_NAMES].to_numpy(dtype=np.float64)
    scaled = scale(raw, model["median"], model["mean"], model["std"])
    scores = scaled @ model["coef"].T + model["intercept"]
    choice = np.argmax(scores, axis=1)
    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability = probability / probability.sum(axis=1, keepdims=True)
    labels = model["classes"][choice]
    return pd.DataFrame({
        "t_s": windows["t_s"].to_numpy(),
        "run_id": windows["run_id"].to_numpy(),
        "row_status": "scored",
        "predicted_fault_type": labels,
        "prediction_confidence": probability[np.arange(len(choice)), choice],
        "model_name": "logistic_reference",
        "selected_model": False,
        "model_version": package["registry"]["model_version"],
    })
