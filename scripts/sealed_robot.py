"""The sealed run's ROBOT stages (scripts/sealed_run.py ROBOT_MODULE, split freeze: USER DECISION Sun ~07:00 CEST).

    imported by name (importlib) by scripts/sealed_run.py `run --stages double`:
        ready(stages)              BEFORE preflight, the drand fetch and the RUNLOG `started` record (no attempt used)
        run_stage("double", record) -> dict   after its ROBOT preflight passed (prereg/FREEZE, TEST_SEED_COMMIT, robot
                                   check 5 against prereg/FREEZE_REALCLIP) and INTERCYCLE_SEALED_RUN=1 was set

It lives outside the real-clip stage's import closure, so wiring or fixing the robot stages before the robot FREEZE
never touches a file frozen by prereg/FREEZE_REALCLIP. It reads nothing from the sealed root: the 16 revealed test seeds
come in the preflight record (record["seeds"]["seed_ints"], world i = seed_ints[i]).

Stages:
* `double`: the sealed CPU-double batch (the CUT design: static, learned, reactive x 16 test worlds x days 9-10 x zones
  9, 14 = 192 missions) on the Modal backend of scripts/run_double.py (run_sealed: observer phase per world on Modal,
  then the reactive-chain rounds, each mission the real policy of its condition), into a FRESH
  <repo>/out/sealed_double/attempt_<NNN>/ (twin_eval's runs root is its runs/). run_double refuses unless the sealed
  flag is set.
* `isaac`: refused (Isaac, movers and A-lite are descriptive and run outside sealed_run; USER DECISION). No other stage
  is wired.

This file is an orchestrator by delegation: it imports only scripts/run_double.py (twin/truth_guard.py ORCHESTRATORS),
never a truth module, and opens no truth file.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
SEALED_ENV = "INTERCYCLE_SEALED_RUN"        # as scripts/sealed_run.py SEALED_ENV
DOUBLE_OUT = "out/sealed_double"            # = sealed_run.DOUBLE_OUT (gitignored; the EVAL runner scores it)
N_TEST_WORLDS = 16
WIRED = ("double",)


class RobotStageError(RuntimeError):
    """A robot stage refused to start (fail closed)."""


StageRefusal = RobotStageError              # the name scripts/sealed_run.py catches (one class, two names)


def _run_double_module():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_double")   # scripts/run_double.py (an orchestrator)


def _refuse_unwired(stage: str) -> None:
    if stage == "isaac":
        raise RobotStageError("robot stage 'isaac' is not wired in scripts/sealed_robot.py: isaac is descriptive and "
                              "runs outside sealed_run (USER DECISION, split freeze)")
    if stage not in WIRED:
        raise RobotStageError(f"unknown robot stage {stage!r}: not wired in scripts/sealed_robot.py "
                              f"(wired: {list(WIRED)})")


def ready(stages: list[str]) -> None:
    """Refused (sealed_run.run calls this BEFORE preflight, the drand fetch and the RUNLOG `started` record, so no
    attempt number is used up) unless every requested stage can really run: for `double`, run_double's sealed backend
    is wired (run_double.SEALED_BACKEND_READY)."""
    for st in stages:
        _refuse_unwired(st)
    if "double" in stages:
        rd = _run_double_module()
        if not getattr(rd, "SEALED_BACKEND_READY", False):
            raise RobotStageError(f"the double stage's sealed backend ({getattr(rd, 'SEALED_BACKEND', '?')}) is not "
                                  "wired in scripts/run_double.py: refusing before any RUNLOG record (no attempt used)")


def run_double_stage(repo: Path, attempt: int, seed_ints: list[int],
                     double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """The sealed CPU-double stage: a fresh runs root per attempt, then run_double.run_sealed (16 test seeds)."""
    if os.environ.get(SEALED_ENV) != "1":
        raise RobotStageError("the double stage runs only after sealed_run's robot preflight passed (sealed flag not "
                              "set)")
    if len(seed_ints) != N_TEST_WORLDS:
        raise RobotStageError(f"the sealed double needs {N_TEST_WORLDS} revealed test seeds, the record has "
                              f"{len(seed_ints)}")
    out = repo / DOUBLE_OUT / f"attempt_{attempt:03d}"
    if out.exists():
        raise RobotStageError(f"{out} exists; every attempt writes to a fresh one")
    runner = double_runner
    if runner is None:
        rd = _run_double_module()
        if not getattr(rd, "SEALED_BACKEND_READY", False):
            raise RobotStageError(f"run_double's sealed backend ({getattr(rd, 'SEALED_BACKEND', '?')}) is not ready")
        runner = rd.run_sealed
    rec = runner(out, [int(s) for s in seed_ints])
    return {"out": str(out), "runs_root": rec.get("runs_root"), "invalid": rec.get("invalid"),
            "n_missions": rec.get("n_missions"), "usd": rec.get("usd"), "wall_s": rec.get("wall_s")}


def run_stage(stage: str, record: dict, *, repo: Path = REPO,
              double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """One robot stage of a sealed attempt. `record` = sealed_run's `started` RUNLOG record (attempt, seeds, ...)."""
    if os.environ.get(SEALED_ENV) != "1":
        raise RobotStageError("the robot stages run only after sealed_run's robot preflight passed (sealed flag not "
                              "set)")
    _refuse_unwired(stage)
    seeds = [int(s) for s in (record.get("seeds") or {}).get("seed_ints") or []]
    return run_double_stage(repo, int(record["attempt"]), seeds, double_runner)
