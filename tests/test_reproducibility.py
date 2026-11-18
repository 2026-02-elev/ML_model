from __future__ import annotations

import json

import numpy as np

from src.inference import run_path
from src.model_loader import PACKAGE_ROOT


def test_synthetic_demo_matches_expected_output():
    expected = json.loads((PACKAGE_ROOT / "examples" / "expected_output.json").read_text(encoding="utf-8"))
    assert expected["synthetic"] is True
    assert expected["not_a_measured_evaluation"] is True
    best = run_path("best", PACKAGE_ROOT / "examples" / "example_timeseries.csv")
    esr = run_path("load_esr", PACKAGE_ROOT / "examples" / "example_load_esr.csv")
    logistic = run_path("logistic_reference", PACKAGE_ROOT / "examples" / "example_timeseries.csv")
    scored = best[best["row_status"] == "scored"]
    prediction = scored.iloc[0][[
        "predicted_voltage_cell_1_v",
        "predicted_voltage_cell_2_v",
        "predicted_voltage_cell_3_v",
        "predicted_voltage_cell_4_v",
    ]].to_numpy(dtype=float)
    assert int((best["row_status"] == "scored").sum()) == expected["n_scored"]
    assert int(best["anomaly_detected"].fillna(False).sum()) == expected["n_alarms"]
    assert np.allclose(prediction, np.asarray(expected["first_scored_prediction_v"], dtype=float), atol=1e-8)
    assert float(esr.loc[0, "score"]) == expected["load_esr_score"] or abs(float(esr.loc[0, "score"]) - expected["load_esr_score"]) < 1e-8
    assert int(esr.loc[0, "pred_cell"]) == expected["load_esr_cell"]
    assert bool(esr.loc[0, "alarm"]) is expected["load_esr_alarm"]
    assert len(logistic) == expected["logistic_windows"]
    assert str(logistic.iloc[0]["predicted_fault_type"]) == expected["logistic_first_label"]
