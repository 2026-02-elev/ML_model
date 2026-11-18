from __future__ import annotations

import numpy as np
import pytest

from src.anomaly_detection import refuse_autoencoder
from src.fault_classification import refuse_selected_classifier
from src.inference import prepare_timeseries, run_path
from src.model_loader import PACKAGE_ROOT, WeightsNotPersisted, load_package


SERIES = PACKAGE_ROOT / "examples" / "example_timeseries.csv"
LOADS = PACKAGE_ROOT / "examples" / "example_load_esr.csv"


def test_ridge_and_detector_on_synthetic_series():
    package = load_package()
    best = run_path("best", SERIES, package)
    assert best.loc[0, "row_status"] == "insufficient_history"
    assert int((best["row_status"] == "scored").sum()) == len(best) - 1
    scored = best[best["row_status"] == "scored"]
    assert np.isfinite(scored["predicted_voltage_cell_1_v"]).all()
    alarms = best[best["anomaly_detected"].fillna(False).astype(bool)]
    assert len(alarms) > 0
    assert set(alarms["anomaly_cell"].astype(int)) == {4}


def test_same_input_is_stable():
    first = run_path("best", SERIES)
    second = run_path("best", SERIES)
    left = first.select_dtypes(include="number").to_numpy(dtype=float)
    right = second.select_dtypes(include="number").to_numpy(dtype=float)
    assert np.allclose(left, right, atol=1e-8, equal_nan=True)


def test_load_esr_matches_published_formula():
    package = load_package()
    result = run_path("load_esr", LOADS, package)
    rest = np.array([0.70, 0.71, 0.69, 0.70])
    load = np.array([0.62, 0.63, 0.61, 0.45])
    excess = (rest - load)
    excess = excess - np.median(excess)
    mu = np.asarray(package["load_esr"]["mu_excess_v"])
    sigma = np.asarray(package["load_esr"]["sigma_excess_v"])
    score = float(((excess - mu) / sigma).max())
    assert result.loc[0, "score"] == pytest.approx(score)
    assert int(result.loc[0, "pred_cell"]) == 4


def test_short_series_has_no_logistic_window(tmp_path):
    path = tmp_path / "short.csv"
    rows = ["t_s,run_id,cell_1_voltage_v,cell_2_voltage_v,cell_3_voltage_v,cell_4_voltage_v"]
    for t in range(5):
        rows.append(f"{t},run-a,0.5,0.5,0.5,0.5")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_path("logistic_reference", path)
    assert result.empty


def test_classifier_and_autoencoder_do_not_substitute():
    package = load_package()
    with pytest.raises(WeightsNotPersisted):
        run_path("classifier", SERIES, package)
    with pytest.raises(WeightsNotPersisted):
        run_path("autoencoder", SERIES, package)
    with pytest.raises(WeightsNotPersisted):
        refuse_selected_classifier(package)
    with pytest.raises(WeightsNotPersisted):
        refuse_autoencoder(package)


def test_unknown_model():
    with pytest.raises(ValueError):
        run_path("not-a-model", SERIES)
