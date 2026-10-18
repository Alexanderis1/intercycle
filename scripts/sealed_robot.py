"""The robot stage of the sealed run: scripts/sealed_run.py ROBOT_MODULE, imported there BY NAME (importlib), so this
file and what it imports stay outside the real-clip import closure frozen at prereg/FREEZE_REALCLIP (sealed_run robot
check 5): wiring or fixing the robot stage before the robot FREEZE never touches a frozen real-clip file.

sealed_run calls `run_stage(stage, record, repo=, double_runner=)` only after every robot refusal passed and
INTERCYCLE_SEALED_RUN=1 is set; `record` is the attempt's `started` RUNLOG record (attempt number, revealed seeds).

* `double`: the sealed CPU-double batch (USER DECISION Sun ~07:00 CEST, the CUT design: static, learned, reactive x 16
  test worlds x days 9-10 x zones 9 and 14 = 192 missions) through scripts/run_double.py `run_sealed` with the 16
  revealed test seeds (world i = seed_ints[i]), into a FRESH <repo>/out/sealed_double/attempt_<NNN>/ (the twin_eval
  runs root is its runs/). run_double refuses unless the sealed flag is set.
* Isaac, movers and A-lite are descriptive and stay outside sealed_run (USER DECISION): no other stage is wired.

This file is not an orchestrator: it imports no truth module itself (run_double, an ORCHESTRATOR, starts the plant).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
SEALED_ENV = "INTERCYCLE_SEALED_RUN"        # as scripts/sealed_run.py SEALED_ENV
DOUBLE_OUT = "out/sealed_double"            # the CPU-double runs root per attempt (gitignored; the EVAL runner scores it)
WIRED = ("double",)


class StageRefusal(RuntimeError):
    """A robot stage refuses to start."""


def _run_double_module():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_double")   # scripts/run_double.py (an orchestrator)


def ready(stages: list[str]) -> None:
    """Refused (sealed_run.run calls this BEFORE preflight, the drand fetch and the RUNLOG `started` record, so no
    attempt number is used up) unless every requested stage can really run: for `double`, run_double's sealed backend
    is wired (run_double.SEALED_BACKEND_READY; lane double-launcher fix1)."""
    for st in stages:
        if st not in WIRED:
            raise StageRefusal(f"robot stage {st!r} is not wired in scripts/sealed_robot.py (wired: {list(WIRED)})")
    if "double" in stages:
        rd = _run_double_module()
        if not getattr(rd, "SEALED_BACKEND_READY", False):
            raise StageRefusal(f"the double stage's sealed backend ({getattr(rd, 'SEALED_BACKEND', '?')}) is not wired "
                               "in scripts/run_double.py: refusing before any RUNLOG record (no attempt used)")


def run_double_stage(repo: Path, attempt: int, seed_ints: list[int],
                     double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """The sealed CPU-double stage: a fresh runs root per attempt, then run_double.run_sealed (16 test seeds)."""
    out = repo / DOUBLE_OUT / f"attempt_{attempt:03d}"
    if out.exists():
        raise StageRefusal(f"{out} exists; every attempt writes to a fresh one")
    if os.environ.get(SEALED_ENV) != "1":
        raise StageRefusal("the double stage runs only after every refusal passed (sealed flag not set)")
    runner = double_runner
    if runner is None:
        runner = _run_double_module().run_sealed
    rec = runner(out, list(seed_ints))
    return {"out": str(out), "runs_root": rec.get("runs_root"), "invalid": rec.get("invalid"),
            "n_missions": rec.get("n_missions"), "usd": rec.get("usd"), "wall_s": rec.get("wall_s")}


def run_stage(stage: str, record: dict, *, repo: Path = REPO,
              double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """One robot stage of a sealed attempt (sealed_run.run, after its refusals)."""
    if stage not in WIRED:
        raise StageRefusal(f"robot stage {stage!r} is not wired in scripts/sealed_robot.py (wired: {list(WIRED)})")
    return run_double_stage(repo, int(record["attempt"]), [int(s) for s in record["seeds"]["seed_ints"]],
                            double_runner)
