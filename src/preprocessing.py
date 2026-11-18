"""Input checks for offline inference.

The pair rule matches the measured pipeline: a row is scored only when the
gap from the previous row in the same run is in (0.5, 2.0] seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VOLTAGE_COLUMNS = [f"cell_{i}_voltage_v" for i in range(1, 5)]
TEMP_COLUMNS = [f"cell_{i}_temperature_c" for i in range(1, 5)]
REST_COLUMNS = [f"rest_cell_{i}_voltage_v" for i in range(1, 5)]
LOAD_COLUMNS = [f"load_cell_{i}_voltage_v" for i in range(1, 5)]
POINT_FEATURE_NAMES = [
    "v1", "v2", "v3", "v4",
    "temp1", "temp2", "temp3", "temp4",
    "spread",
    "dv1", "dv2", "dv3", "dv4",
    "dt_prev",
]


def load_timeseries(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"input file has no rows: {path}")
    missing = [column for column in VOLTAGE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "missing voltage columns: "
            + ", ".join(missing)
            + ". Expected cell_1_voltage_v .. cell_4_voltage_v in cell order H_, H8, H7, H1."
        )
    if "run_id" not in frame.columns:
        raise ValueError("missing column: run_id")
    frame = frame.copy()
    frame["run_id"] = frame["run_id"].astype(str)
    frame["t"] = _time_seconds(frame)
    for column in VOLTAGE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} has a missing or non-numeric value")
        if (frame[column] <= 0).any():
            raise ValueError(
                f"{column} has a non-positive value. The measured pipeline rejected those rows and this package does not impute them."
            )
    for column in TEMP_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "cell_voltage_spread_v" in frame.columns:
        frame["cell_voltage_spread_v"] = pd.to_numeric(frame["cell_voltage_spread_v"], errors="coerce")
    else:
        volts = frame[VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
        frame["cell_voltage_spread_v"] = volts.max(axis=1) - volts.min(axis=1)
    if frame["cell_voltage_spread_v"].isna().any():
        raise ValueError("cell_voltage_spread_v has a missing value")
    frame = frame.sort_values(["run_id", "t"], kind="mergesort").reset_index(drop=True)
    if frame["t"].isna().any():
        raise ValueError("timestamps could not be parsed")
    return frame


def load_load_pairs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"input file has no rows: {path}")
    required = REST_COLUMNS + LOAD_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "load-ESR input is not a 1 Hz CGT file. Missing columns: "
            + ", ".join(missing)
            + ". Provide rest_cell_*_voltage_v and load_cell_*_voltage_v from a paired rest sample and a load sample."
        )
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} has a missing or non-numeric value")
        if (frame[column] <= 0).any():
            raise ValueError(f"{column} has a non-positive value")
    if "run_id" not in frame.columns:
        frame["run_id"] = "unspecified"
    frame["run_id"] = frame["run_id"].astype(str)
    return frame


def _time_seconds(frame: pd.DataFrame) -> np.ndarray:
    if "t_s" in frame.columns:
        values = pd.to_numeric(frame["t_s"], errors="coerce")
        if values.isna().any():
            raise ValueError("t_s has a missing or non-numeric value")
        return values.to_numpy(dtype=np.float64)
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp or t_s is required")
    numeric = pd.to_numeric(frame["timestamp"], errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float64)
    parsed = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601", errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamp must be ISO-8601 or a number (seconds)")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (parsed - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
