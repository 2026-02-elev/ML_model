"""Persistence residual EWMA and temperature-rise alarm.

The validation-selected detector predicts the next voltage as the previous
voltage. It does not use the Ridge residual for the alarm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import temperature_rise
from src.model_loader import WeightsNotPersisted
from src.preprocessing import VOLTAGE_COLUMNS


def time_alpha(alpha_1s: float, dt_s: float) -> float:
    alpha = float(alpha_1s)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha_1s must be in (0, 1]")
    dt = float(dt_s)
    if dt <= 0.0:
        return 0.0
    if alpha == 1.0:
        return 1.0
    return float(1.0 - (1.0 - alpha) ** dt)


def ewma_time(values: np.ndarray, times: np.ndarray, run_ids: np.ndarray,
              alpha_1s: float, max_gap_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(values) != len(times) or len(values) != len(run_ids):
        raise ValueError("values, times, and run_ids must have the same length")
    out = np.empty_like(values, dtype=np.float64)
    prev = None
    prev_t = None
    prev_run = None
    for index in range(len(values)):
        run_id = str(run_ids[index])
        t = float(times[index])
        reset = (
            prev is None
            or run_id != prev_run
            or prev_t is None
            or t <= prev_t
            or (t - prev_t) > float(max_gap_s)
        )
        if reset:
            out[index] = values[index]
        else:
            alpha = time_alpha(alpha_1s, t - prev_t)
            out[index] = alpha * values[index] + (1.0 - alpha) * prev
        prev = out[index]
        prev_t = t
        prev_run = run_id
    return out


def score_persistence(frame: pd.DataFrame, package: dict) -> pd.DataFrame:
    config = package["inference"]
    thresholds = package["anomaly"]
    if thresholds["model"] != "persistence" or thresholds["temp_mode"] != "rise":
        raise RuntimeError("published detector is persistence / rise; the threshold file does not match")
    rise = temperature_rise(frame, float(config["temp_baseline_tau_s"]), float(config["max_gap_s"]))
    scored = frame["pair_status"].to_numpy() == "scored"
    residual = np.full((len(frame), 4), np.nan, dtype=np.float64)
    ewma = np.full((len(frame), 4), np.nan, dtype=np.float64)
    alarm = np.zeros(len(frame), dtype=bool)
    cell = np.full(len(frame), -1, dtype=int)
    if scored.any():
        current = frame.loc[scored, VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
        previous = frame.loc[scored, ["x0", "x1", "x2", "x3"]].to_numpy(dtype=np.float64)
        residual[scored] = np.abs(current - previous)
        ewma_rows = ewma_time(
            residual[scored],
            frame.loc[scored, "t"].to_numpy(dtype=np.float64),
            frame.loc[scored, "run_id"].to_numpy(),
            float(thresholds["alpha"]),
            float(config["max_gap_s"]),
        )
        ewma[scored] = ewma_rows
        temp = rise[scored]
        volt_score = ewma_rows.max(axis=1)
        temp_score = np.nanmax(temp, axis=1)
        volt_alarm = volt_score >= float(thresholds["volt_threshold_v"])
        temp_alarm = temp_score >= float(thresholds["temp_threshold_c"])
        row_alarm = volt_alarm | temp_alarm
        temp_cell = np.nanargmax(np.where(np.isfinite(temp), temp, -np.inf), axis=1) + 1
        volt_cell = np.nanargmax(np.where(np.isfinite(ewma_rows), ewma_rows, -np.inf), axis=1) + 1
        row_cell = np.where(temp_alarm, temp_cell, volt_cell).astype(int)
        alarm[scored] = row_alarm
        cell[scored] = row_cell
    result = pd.DataFrame({
        "t_s": frame["t"].to_numpy(),
        "run_id": frame["run_id"].to_numpy(),
        "row_status": frame["pair_status"].to_numpy(),
        "ewma_score": ewma.max(axis=1),
        "temp_rise_score_c": rise.max(axis=1),
        "anomaly_cell": cell,
        "model_version": package["registry"]["model_version"],
        "detector": "persistence_ewma_temp_rise",
    })
    for index in range(4):
        result[f"persistence_residual_cell_{index + 1}_v"] = residual[:, index]
        result[f"ewma_cell_{index + 1}"] = ewma[:, index]
        result[f"temp_rise_cell_{index + 1}_c"] = rise[:, index]
    detected = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    detected.loc[scored] = alarm[scored]
    result["anomaly_detected"] = detected
    result.loc[~scored, "anomaly_cell"] = pd.NA
    return result


def refuse_autoencoder(package: dict) -> None:
    status = package["autoencoder_status"]
    raise WeightsNotPersisted(
        "Autoencoder weights were not saved (WEIGHTS_NOT_PERSISTED). "
        f"Selected status={status.get('status')}, threshold={status.get('threshold')}. "
        "This package does not retrain the autoencoder and will not invent reconstruction errors."
    )
