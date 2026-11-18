"""Score a caller-supplied CSV. This does not reproduce the published test metrics.

The measured evaluation tables are not in this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.inference import prepare_timeseries, predict_ridge, regression_frame, write_frame
from src.preprocessing import VOLTAGE_COLUMNS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a local CSV with the published Ridge model")
    parser.add_argument("--data", default="")
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args(argv)
    if not args.data:
        print("Measured evaluation data is not in this repository.")
        print("The published test MAE, event recalls, and macro F1 cannot be recomputed from the clone alone.")
        return 2
    path = Path(args.data)
    if not path.is_file():
        print(f"Evaluation file not found: {path}")
        return 2
    package, frame = prepare_timeseries(path)
    predicted = predict_ridge(frame, package)
    table = regression_frame(frame, predicted, package)
    scored = frame["pair_status"].to_numpy() == "scored"
    actual = frame.loc[scored, VOLTAGE_COLUMNS].to_numpy(dtype=np.float64)
    pred = predicted[scored]
    summary = {
        "input": str(path),
        "rows": int(len(frame)),
        "scored_rows": int(scored.sum()),
        "published_test_metrics_reproduced": False,
        "reason": "This score uses the file you passed. It is not the frozen measured test split unless that file is the unpublished test table.",
    }
    if scored.any():
        summary["mae_v_on_this_file"] = float(np.mean(np.abs(actual - pred)))
    if "label" in frame.columns:
        normal = scored & (frame["label"].astype(str).to_numpy() == "NORMAL")
        if normal.any():
            summary["mae_v_normal_rows_on_this_file"] = float(np.mean(np.abs(predicted[normal] - frame.loc[normal, VOLTAGE_COLUMNS].to_numpy(dtype=np.float64))))
            summary["normal_rows"] = int(normal.sum())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_frame(table, out / "regression_predictions.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)
