from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import load_load_pairs, load_timeseries


def _write_series(path: Path, voltage: str = "0.5") -> None:
    path.write_text(
        "t_s,run_id,cell_1_voltage_v,cell_2_voltage_v,cell_3_voltage_v,cell_4_voltage_v\n"
        f"0,run-a,{voltage},{voltage},{voltage},{voltage}\n"
        f"1,run-a,{voltage},{voltage},{voltage},{voltage}\n",
        encoding="utf-8",
    )


def test_spread_is_max_minus_min(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text(
        "t_s,run_id,cell_1_voltage_v,cell_2_voltage_v,cell_3_voltage_v,cell_4_voltage_v\n"
        "0,run-a,0.40,0.50,0.55,0.45\n",
        encoding="utf-8",
    )
    frame = load_timeseries(path)
    assert frame.loc[0, "cell_voltage_spread_v"] == pytest.approx(0.15)


def test_non_positive_voltage_is_rejected(tmp_path: Path):
    path = tmp_path / "series.csv"
    _write_series(path, "0")
    with pytest.raises(ValueError, match="non-positive"):
        load_timeseries(path)


def test_missing_column_is_rejected(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text("t_s,run_id,cell_1_voltage_v\n0,run-a,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing voltage"):
        load_timeseries(path)


def test_gap_and_pair_status(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text(
        "t_s,run_id,cell_1_voltage_v,cell_2_voltage_v,cell_3_voltage_v,cell_4_voltage_v\n"
        "0,run-a,0.5,0.5,0.5,0.5\n"
        "1,run-a,0.5,0.5,0.5,0.5\n"
        "5,run-a,0.5,0.5,0.5,0.5\n"
        "0,run-b,0.5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )
    from src.feature_engineering import annotate_pairs

    frame = annotate_pairs(load_timeseries(path), 0.5, 2.0)
    status = dict(zip(frame["t"].tolist(), frame["pair_status"].tolist()))
    # run-b is a new run at t=0 after sorting by run_id, so both runs start unscored.
    assert list(frame["pair_status"]) == [
        "insufficient_history",
        "scored",
        "gap_outside_pair_window",
        "insufficient_history",
    ]


def test_cgt_file_is_not_a_load_esr_file(tmp_path: Path):
    path = tmp_path / "series.csv"
    _write_series(path)
    with pytest.raises(ValueError, match="load-ESR"):
        load_load_pairs(path)


def test_iso_timestamp(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text(
        "timestamp,run_id,cell_1_voltage_v,cell_2_voltage_v,cell_3_voltage_v,cell_4_voltage_v\n"
        "2026-11-07T00:00:00Z,run-a,0.5,0.5,0.5,0.5\n"
        "2026-11-07T00:00:01Z,run-a,0.5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )
    frame = load_timeseries(path)
    assert frame.loc[1, "t"] - frame.loc[0, "t"] == pytest.approx(1.0)
