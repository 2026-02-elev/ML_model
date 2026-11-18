"""Run one published model, or the runnable part of function ②."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly_detection import refuse_autoencoder, score_persistence
from src.fault_classification import predict_logistic_reference, refuse_selected_classifier
from src.feature_engineering import annotate_pairs
from src.load_esr import score_load_pairs
from src.model_loader import load_package, scale
from src.preprocessing import VOLTAGE_COLUMNS, load_load_pairs, load_timeseries


def prepare_timeseries(path: str | Path, package: dict | None = None) -> tuple[dict, pd.DataFrame]:
    package = load_package() if package is None else package
    config = package["inference"]
    frame = annotate_pairs(
        load_timeseries(path),
        float(config["pair_dt_min_s"]),
        float(config["pair_dt_max_s"]),
    )
    return package, frame


def predict_ridge(frame: pd.DataFrame, package: dict) -> np.ndarray:
    model = package["ridge"]
    predicted = np.full((len(frame), 4), np.nan, dtype=np.float64)
    scored = frame["pair_status"].to_numpy() == "scored"
    if not scored.any():
        return predicted
    raw = frame.loc[scored, [f"x{i}" for i in range(14)]].to_numpy(dtype=np.float64)
    scaled = scale(raw, model["median"], model["mean"], model["std"])
    predicted[scored] = scaled @ model["coef"].T + model["intercept"]
    return predicted


def regression_frame(frame: pd.DataFrame, predicted: np.ndarray, package: dict) -> pd.DataFrame:
    result = pd.DataFrame({
        "t_s": frame["t"].to_numpy(),
        "run_id": frame["run_id"].to_numpy(),
        "row_status": frame["pair_status"].to_numpy(),
        "model_name": "ridge",
        "model_version": package["registry"]["model_version"],
    })
    if "timestamp" in frame.columns:
        result.insert(0, "timestamp", frame["timestamp"].to_numpy())
    actual = frame[VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
    for index in range(4):
        result[f"predicted_voltage_cell_{index + 1}_v"] = predicted[:, index]
        result[f"ridge_abs_residual_cell_{index + 1}_v"] = np.where(
            np.isfinite(predicted[:, index]),
            np.abs(actual[:, index] - predicted[:, index]),
            np.nan,
        )
    return result


def best_frame(frame: pd.DataFrame, package: dict) -> pd.DataFrame:
    """Ridge voltage plus the selected persistence/temperature detector."""
    predicted = predict_ridge(frame, package)
    voltage = regression_frame(frame, predicted, package)
    alarm = score_persistence(frame, package)
    keep = [
        "ewma_score", "temp_rise_score_c", "anomaly_detected", "anomaly_cell", "detector",
    ]
    for index in range(1, 5):
        keep.append(f"persistence_residual_cell_{index}_v")
        keep.append(f"ewma_cell_{index}")
    merged = pd.concat([voltage.reset_index(drop=True), alarm[keep].reset_index(drop=True)], axis=1)
    merged["tasks"] = "ridge_voltage+persistence_ewma"
    return merged


def run_path(model: str, path: str | Path, package: dict | None = None) -> pd.DataFrame:
    model = model.strip().lower()
    if model in {"classifier", "hgb", "hist_gradient_boosting"}:
        package = load_package() if package is None else package
        refuse_selected_classifier(package)
    if model in {"autoencoder", "anomaly_ae", "ae"}:
        package = load_package() if package is None else package
        refuse_autoencoder(package)
    if model == "load_esr":
        package = load_package() if package is None else package
        return score_load_pairs(load_load_pairs(path), package)
    package, frame = prepare_timeseries(path, package)
    predicted = predict_ridge(frame, package)
    if model == "regression":
        return regression_frame(frame, predicted, package)
    if model == "anomaly":
        return score_persistence(frame, package)
    if model == "logistic_reference":
        return predict_logistic_reference(frame, predicted, package)
    if model == "best":
        return best_frame(frame, package)
    raise ValueError(
        "unknown model. Use best, regression, anomaly, load_esr, logistic_reference. "
        "classifier and autoencoder are refused because their selected weights were not saved."
    )


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path
