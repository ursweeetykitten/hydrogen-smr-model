# -*- coding: utf-8 -*-
"""Run the full 6500 Nm3/h H2 recalculation workflow."""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_catalyst_hydraulics_selection.py",
    "02_operating_optimization.py",
    "03_final_balances.py",
]


def main() -> None:
    for script in SCRIPTS:
        print("\n" + "=" * 80, flush=True)
        print(f"Running {script}", flush=True)
        print("=" * 80, flush=True)
        runpy.run_path(str(HERE / script), run_name="__main__")
    print("\nAll calculations are complete. CSV files are in:", flush=True)
    print(HERE / "results", flush=True)


if __name__ == "__main__":
    main()
