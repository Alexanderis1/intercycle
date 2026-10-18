"""The sealed run's ROBOT stages (scripts/sealed_run.py ROBOT_MODULE, split freeze: USER DECISION Sun ~07:00 CEST).

    imported by name by scripts/sealed_run.py `run --stages double` after its ROBOT preflight passed (prereg/FREEZE,
    TEST_SEED_COMMIT, robot check 5 against prereg/FREEZE_REALCLIP) and INTERCYCLE_SEALED_RUN=1 was set:
        run_stage("double", record) -> dict

It lives outside the real-clip stage's import closure, so wiring the robot stages never touches a file frozen by
prereg/FREEZE_REALCLIP. It reads nothing from the sealed root: the 16 revealed test seeds come in the preflight record
(record["seeds"]["seed_ints"], world i = seed_ints[i]).

Stages:
* `double`: the sealed CPU-double batch (the CUT design: static, learned, reactive x 16 test worlds x days 9-10 x zones
  9, 14 = 192 missions) on the Modal backend of scripts/run_double.py (run_sealed: observer phase per world on Modal,
  then the reactive-chain rounds, each mission the real policy of its condition), into a FRESH
  <repo>/out/sealed_double/attempt_<NNN>/ (twin_eval's runs root is its runs/).
* `isaac`: refused here (Isaac, movers and A-lite are descriptive and run outside sealed_run; USER DECISION).

This file is an orchestrator by delegation: it imports only scripts/run_double.py (twin/truth_guard.py ORCHESTRATORS),
never a truth module, and opens no truth file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
SEALED_ENV = "INTERCYCLE_SEALED_RUN"
DOUBLE_OUT = "out/sealed_double"            # = sealed_run.DOUBLE_OUT
N_TEST_WORLDS = 16


class RobotStageError(RuntimeError):
    """A robot stage refused to start (fail closed)."""


def _run_double():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_double  # scripts/run_double.py (an orchestrator)

    return run_double


def run_stage(stage: str, record: dict, *, repo: Path = REPO,
              double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """One robot stage of a sealed attempt. `record` = sealed_run's `started` RUNLOG record (attempt, seeds, ...)."""
    if os.environ.get(SEALED_ENV) != "1":
        raise RobotStageError("the robot stages run only after sealed_run's robot preflight passed (sealed flag not set)")
    if stage == "isaac":
        raise RobotStageError("isaac is descriptive and runs outside sealed_run (USER DECISION, split freeze)")
    if stage != "double":
        raise RobotStageError(f"unknown robot stage {stage!r}")
    seeds = [int(s) for s in (record.get("seeds") or {}).get("seed_ints") or []]
    if len(seeds) != N_TEST_WORLDS:
        raise RobotStageError(f"the sealed double needs {N_TEST_WORLDS} revealed test seeds, the record has {len(seeds)}")
    attempt = int(record["attempt"])
    out = repo / DOUBLE_OUT / f"attempt_{attempt:03d}"
    if out.exists():
        raise RobotStageError(f"{out} exists; every attempt writes to a fresh one")
    if double_runner is None:
        rd = _run_double()
        if not getattr(rd, "SEALED_BACKEND_READY", False):
            raise RobotStageError(f"run_double's sealed backend ({rd.SEALED_BACKEND}) is not ready")
        double_runner = rd.run_sealed
    rec = double_runner(out, seeds)
    return {"out": str(out), "runs_root": rec.get("runs_root"), "invalid": rec.get("invalid"),
            "n_missions": rec.get("n_missions"), "usd": rec.get("usd"), "wall_s": rec.get("wall_s")}
