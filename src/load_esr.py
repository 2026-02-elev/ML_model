"""Load-drop ESR score.

excess = sag - median(sag across the four cells)
z = (excess - mu) / sigma
score = max z

This is not an ohmic ESR. Current was not measured. A 1 Hz CGT file alone
is the wrong input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import LOAD_COLUMNS, REST_COLUMNS


def score_load_pairs(frame: pd.DataFrame, package: dict) -> pd.DataFrame:
    params = package["load_esr"]
    mu = np.asarray(params["mu_excess_v"], dtype=np.float64)
    sigma = np.asarray(params["sigma_excess_v"], dtype=np.float64)
    rest = frame[REST_COLUMNS].to_numpy(dtype=np.float64)
    load = frame[LOAD_COLUMNS].to_numpy(dtype=np.float64)
    sag = rest - load
    excess = sag - np.median(sag, axis=1, keepdims=True)
    z = (excess - mu) / sigma
    score = z.max(axis=1)
    cell = z.argmax(axis=1) + 1
    threshold = float(params["threshold"])
    result = pd.DataFrame({
        "run_id": frame["run_id"].to_numpy(),
        "score": score,
        "pred_cell": cell.astype(int),
        "alarm": score >= threshold,
        "threshold": threshold,
        "model_version": package["registry"]["model_version"],
        "label_meaning": params["label_meaning"],
    })
    for index in range(4):
        result[f"sag_cell_{index + 1}_v"] = sag[:, index]
        result[f"excess_cell_{index + 1}_v"] = excess[:, index]
        result[f"z_cell_{index + 1}"] = z[:, index]
    return result
