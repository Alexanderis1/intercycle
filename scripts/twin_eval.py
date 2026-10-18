"""EVAL of the press-shop twin runs (PLAN_TWIN §5): primary, gatekeeping H2-H4, outside-gate reports.

    .venv/Scripts/python.exe -m twin.eval_metrics --runs RUNS --margin-file twin/m7_margin.json \
        --invalid results/sealed/invalid.json --out results/sealed/h3_tick_counts.json STREAMS...   # truth, after run
    .venv/Scripts/python.exe scripts/twin_eval.py --runs RUNS --tasks twin/tasks.yaml --split test --engine double \
        --invalid results/sealed/invalid.json --h3-counts results/sealed/h3_tick_counts.json \
        --recall results/sealed/recall_frames.json --r0 R0 --out results/sealed/eval.json

The first step is the only truth read (twin/eval_metrics.py, the allowlisted EVAL slot; it refuses until every run
is closed and logs every opening). This script never opens truth; see intercycle/eval/twin_eval.py for the rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.eval.twin_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
