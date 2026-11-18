"""Load the example files and write prediction CSVs. No training."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_path, write_frame
from src.model_loader import PACKAGE_ROOT, load_package


def main() -> int:
    root = PACKAGE_ROOT
    package = load_package(root)
    print("CellGuard Inference Demo")
    print("------------------------")
    print(f"Model version {package['registry']['model_version']} loaded.")
    print("Ridge voltage weights loaded.")
    print("Persistence/EWMA thresholds loaded.")
    print("Load-ESR parameters loaded.")
    print("HistGradientBoosting weights were not saved (WEIGHTS_NOT_PERSISTED).")
    print("Autoencoder weights were not saved (WEIGHTS_NOT_PERSISTED).")
    series = root / "examples" / "example_timeseries.csv"
    loads = root / "examples" / "example_load_esr.csv"
    best = run_path("best", series, package)
    esr = run_path("load_esr", loads, package)
    logistic = run_path("logistic_reference", series, package)
    print("Input validation passed.")
    print("Inference completed. Example rows are synthetic and are not measured scores.")
    outputs = [
        write_frame(best, root / "outputs" / "demo_predictions.csv"),
        write_frame(esr, root / "outputs" / "demo_load_esr.csv"),
        write_frame(logistic, root / "outputs" / "demo_logistic_reference.csv"),
    ]
    alarms = int(best["anomaly_detected"].fillna(False).sum())
    print(f"Scored voltage rows: {int((best['row_status'] == 'scored').sum())} of {len(best)}")
    print(f"Persistence/EWMA alarms on the synthetic series: {alarms}")
    print(f"Load-ESR score: {float(esr['score'].iloc[0]):.6f} cell {int(esr['pred_cell'].iloc[0])} alarm {bool(esr['alarm'].iloc[0])}")
    print(f"Logistic reference windows: {len(logistic)} (not the selected classifier)")
    print("Output saved to:")
    for path in outputs:
        print(path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        raise SystemExit(1)
