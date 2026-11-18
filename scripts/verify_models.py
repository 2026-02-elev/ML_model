"""Check that published files match checksums.json."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import PACKAGE_ROOT, load_package, verify_checksums


def main() -> int:
    digests = verify_checksums(PACKAGE_ROOT)
    package = load_package(PACKAGE_ROOT)
    print(f"checksums ok: {len(digests)}")
    for relative in digests:
        print(f"  {relative}")
    tasks = package["registry"]["tasks"]
    for name, task in tasks.items():
        state = "runnable" if task.get("runnable") else task.get("reason")
        print(f"{name}: selected={task.get('selected_model')} {state}")
    print(f"ridge coef {tuple(package['ridge']['coef'].shape)}")
    print(f"logistic reference classes {list(package['logistic_reference']['classes'])} selected=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        raise SystemExit(1)
