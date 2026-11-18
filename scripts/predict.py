"""Score one published model and write a CSV. No training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_path, write_frame
from src.model_loader import WeightsNotPersisted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CellGuard offline inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        frame = run_path(args.model, args.input)
    except WeightsNotPersisted as exc:
        print(exc, file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    path = write_frame(frame, args.output)
    scored = int((frame["row_status"] == "scored").sum()) if "row_status" in frame.columns else len(frame)
    print(f"model={args.model} rows={len(frame)} scored={scored}")
    print(f"output={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
