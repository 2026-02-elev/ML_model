from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model_loader import PACKAGE_ROOT, WeightsNotPersisted, load_package, verify_checksums
from src.fault_classification import refuse_selected_classifier
from src.anomaly_detection import refuse_autoencoder


def test_checksums_and_shapes():
    digests = verify_checksums(PACKAGE_ROOT)
    assert "models/regression/ridge_numpy.npz" in digests
    package = load_package(PACKAGE_ROOT)
    assert package["ridge"]["coef"].shape == (4, 14)
    assert package["ridge"]["mean"].shape == (14,)
    assert package["anomaly"]["model"] == "persistence"
    assert package["anomaly"]["temp_mode"] == "rise"
    assert len(package["load_esr"]["mu_excess_v"]) == 4
    assert package["registry"]["tasks"]["classifier"]["runnable"] is False
    assert package["registry"]["tasks"]["autoencoder"]["runnable"] is False


def test_selected_weights_are_refused():
    package = load_package(PACKAGE_ROOT)
    with pytest.raises(WeightsNotPersisted):
        refuse_selected_classifier(package)
    with pytest.raises(WeightsNotPersisted):
        refuse_autoencoder(package)


def test_published_files_exist():
    root = PACKAGE_ROOT
    for relative in (
        "models/regression/ridge_numpy.npz",
        "models/classifier/logistic_reference.npz",
        "models/classifier/WEIGHTS_NOT_PERSISTED.json",
        "models/autoencoder/WEIGHTS_NOT_PERSISTED.json",
        "models/anomaly/thresholds.json",
        "models/load_esr/parameters.json",
        "docs/FINAL_REPORT.md",
    ):
        assert (root / relative).is_file(), relative
        assert (root / relative).stat().st_size > 0
