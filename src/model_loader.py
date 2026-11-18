"""Load published NumPy artifacts and reject checksum mismatches.

Joblib and pickle files are not loaded. The published weights are arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class WeightsNotPersisted(RuntimeError):
    """The selected model was evaluated, but its weights were not saved."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checksums(root: Path | None = None) -> dict[str, str]:
    root = PACKAGE_ROOT if root is None else Path(root)
    listing = _read_json(root / "models" / "checksums.json")
    for relative, expected in listing.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"published model file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}. "
                "Refusing to load a file that does not match the published digest."
            )
    return listing


def load_package(root: Path | None = None) -> dict:
    root = PACKAGE_ROOT if root is None else Path(root)
    verify_checksums(root)
    ridge = np.load(root / "models" / "regression" / "ridge_numpy.npz", allow_pickle=False)
    logistic = np.load(root / "models" / "classifier" / "logistic_reference.npz", allow_pickle=False)
    if ridge["coef"].shape != (4, 14):
        raise RuntimeError(f"ridge coef shape {ridge['coef'].shape} is not (4, 14)")
    if logistic["coef"].shape[1] != 44:
        raise RuntimeError(f"logistic reference coef width {logistic['coef'].shape} is not 44")
    return {
        "root": root,
        "registry": _read_json(root / "configs" / "model_registry.json"),
        "inference": yaml.safe_load((root / "configs" / "inference.yaml").read_text(encoding="utf-8")),
        "ridge": {key: ridge[key] for key in ("coef", "intercept", "mean", "std", "median")},
        "anomaly": _read_json(root / "models" / "anomaly" / "thresholds.json"),
        "load_esr": _read_json(root / "models" / "load_esr" / "parameters.json"),
        "logistic_reference": {
            "coef": logistic["coef"],
            "intercept": logistic["intercept"],
            "classes": np.asarray(logistic["classes"]).astype(str),
            "mean": logistic["mean"],
            "std": logistic["std"],
            "median": logistic["median"],
        },
        "classifier_status": _read_json(root / "models" / "classifier" / "WEIGHTS_NOT_PERSISTED.json"),
        "autoencoder_status": _read_json(root / "models" / "autoencoder" / "WEIGHTS_NOT_PERSISTED.json"),
    }


def scale(features: np.ndarray, median: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    filled = np.where(np.isfinite(features), features, median)
    return (filled - mean) / std
