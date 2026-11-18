"""Point features, temperature rise, and 30 s window features.

The formulas are the ones used in the measured Phase 4 fit. Residuals used
inside a window are absolute 1-step Ridge errors on this same series. They
are not the training-set out-of-run residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import TEMP_COLUMNS, VOLTAGE_COLUMNS

WINDOW_FEATURE_NAMES = [
    f"c{cell}_{stat}"
    for cell in range(1, 5)
    for stat in (
        "v_mean", "v_std", "v_min", "v_max", "v_slope", "v_delta",
        "temp_mean", "temp_delta", "resid_mean", "resid_max",
    )
] + ["spread_mean", "spread_slope", "spread_last", "slope_range"]


def annotate_pairs(frame: pd.DataFrame, min_dt: float, max_dt: float) -> pd.DataFrame:
    out = frame.copy()
    features = np.full((len(out), 14), np.nan, dtype=np.float64)
    status = np.full(len(out), "insufficient_history", dtype=object)
    for _, group in out.groupby("run_id", sort=False):
        positions = group.index.to_numpy()
        times = group["t"].to_numpy(dtype=np.float64)
        voltage = group[VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
        temp = group[TEMP_COLUMNS].to_numpy(dtype=np.float64)
        spread = group["cell_voltage_spread_v"].to_numpy(dtype=np.float64)
        for local in range(1, len(positions)):
            dt = float(times[local] - times[local - 1])
            target = int(positions[local])
            if not (min_dt < dt <= max_dt):
                status[target] = "gap_outside_pair_window"
                continue
            if local >= 2 and min_dt < float(times[local - 1] - times[local - 2]) <= max_dt:
                delta_v = voltage[local - 1] - voltage[local - 2]
                dt_prev = float(times[local - 1] - times[local - 2])
            else:
                delta_v = np.zeros(4, dtype=np.float64)
                dt_prev = 0.0
            features[target] = np.concatenate([
                voltage[local - 1],
                temp[local - 1],
                [spread[local - 1]],
                delta_v,
                [dt_prev],
            ])
            status[target] = "scored"
    out["pair_status"] = status
    for index in range(14):
        out[f"x{index}"] = features[:, index]
    return out


def temperature_rise(frame: pd.DataFrame, tau_s: float, max_gap_s: float) -> np.ndarray:
    values = frame[TEMP_COLUMNS].to_numpy(dtype=np.float64)
    times = frame["t"].to_numpy(dtype=np.float64)
    runs = frame["run_id"].astype(str).to_numpy()
    baseline = np.empty_like(values, dtype=np.float64)
    state = None
    prev_t = None
    prev_run = None
    for index in range(len(values)):
        run_id = str(runs[index])
        t = float(times[index])
        reset = state is None or run_id != prev_run or prev_t is None or t <= prev_t or (t - prev_t) > max_gap_s
        if reset or not np.isfinite(values[index]).all():
            baseline[index] = values[index]
            state = np.array(values[index], dtype=np.float64, copy=True)
        else:
            baseline[index] = state
            alpha = 1.0 - np.exp(-(t - prev_t) / float(tau_s))
            state = alpha * values[index] + (1.0 - alpha) * state
        prev_t = t
        prev_run = run_id
    return np.clip(values - baseline, 0.0, None)


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return float("nan")
    t = times[finite]
    y = values[finite]
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom <= 0:
        return 0.0
    return float(np.dot(t, y - y.mean()) / denom)


def _nanmean(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _nanmax(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmax(values))


def window_frame(
    frame: pd.DataFrame,
    residual: np.ndarray,
    window_s: float,
    max_gap_s: float,
    min_fraction: float = 0.75,
    min_samples: int = 8,
) -> pd.DataFrame:
    """Windows ending at each sample that has enough contiguous history."""
    meta = []
    features = []
    residual = np.asarray(residual, dtype=np.float64)
    for _, group in frame.groupby("run_id", sort=False):
        indexes = group.index.to_numpy()
        times = group["t"].to_numpy(dtype=np.float64)
        voltage = group[VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
        temp = group[TEMP_COLUMNS].to_numpy(dtype=np.float64)
        spread = group["cell_voltage_spread_v"].to_numpy(dtype=np.float64)
        run_id = group["run_id"].iloc[0]
        local_resid = residual[indexes]
        for end in range(len(indexes)):
            start = end
            while start > 0 and (times[end] - times[start - 1]) <= window_s and (times[start] - times[start - 1]) <= max_gap_s:
                start -= 1
            count = end - start + 1
            span = float(times[end] - times[start])
            if count < min_samples or span < window_s * min_fraction:
                continue
            sl = slice(start, end + 1)
            row = []
            slopes = []
            for cell in range(4):
                volts = voltage[sl, cell]
                temps = temp[sl, cell]
                resid = local_resid[sl, cell]
                slope = _slope(times[sl], volts)
                slopes.append(0.0 if not np.isfinite(slope) else slope)
                row.extend([
                    float(volts.mean()),
                    float(volts.std()),
                    float(volts.min()),
                    float(volts.max()),
                    slope,
                    float(volts[-1] - volts[0]),
                    _nanmean(temps),
                    float(temps[-1] - temps[0]) if np.isfinite(temps[0]) and np.isfinite(temps[-1]) else np.nan,
                    _nanmean(resid),
                    _nanmax(resid),
                ])
            row.extend([
                float(np.nanmean(spread[sl])),
                _slope(times[sl], spread[sl]),
                float(spread[end]),
                float(max(slopes) - min(slopes)),
            ])
            features.append(row)
            meta.append({
                "run_id": run_id,
                "t_s": float(times[end]),
                "target_index": int(indexes[end]),
            })
    if not features:
        empty = pd.DataFrame(columns=["run_id", "t_s", "target_index", *WINDOW_FEATURE_NAMES])
        return empty
    return pd.concat([
        pd.DataFrame(meta),
        pd.DataFrame(features, columns=WINDOW_FEATURE_NAMES),
    ], axis=1)
