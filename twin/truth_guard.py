"""Truth guard (PLAN_TWIN §2 "Truth guards"): who may open truth, and the runtime access log.

Truth = realised_state*, plant_state, the day event script, per-world dynamics draws, world seeds and defect state.
Only TRUTH_READERS may open it for reading: twin/engine.py, twin/plant.py, twin/camera.py (and its G0 stand-in
twin/camera_v0.py), twin/oracle_feed.py, the EVAL metrics module (after the run only, once log.parquet is closed)
and S7b's offline M-W7 calibration on dev worlds only. TRUTH_PRODUCERS may write it (the day generators, the engine,
the plant). `open_truth` enforces this at run time and appends every opening to results/truth_access.log
(override the path with INTERCYCLE_TRUTH_LOG); tests/test_truth_allowlist.py checks the same lists statically.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import os
from pathlib import Path
from typing import IO, Any

REPO = Path(__file__).resolve().parents[1]

TRUTH_TOKENS = ("realised_state", "plant_state", "day_script", "dynamics_draw", "world_seeds", "defect_state")

# Repo-relative posix paths. twin/eval_metrics.py (S1b EVAL metrics) and twin/margin_calib.py (S7b offline M-W7
# calibration) are the names reserved for the two plan entries that have no file name yet.
TRUTH_READERS = (
    "twin/engine.py",
    "twin/plant.py",
    "twin/camera.py",
    "twin/camera_v0.py",
    "twin/oracle_feed.py",
    "twin/eval_metrics.py",
    "twin/margin_calib.py",
)
TRUTH_PRODUCERS = (
    "twin/daygen.py",
    "twin/daygen_stub.py",
    "twin/engine.py",
    "twin/plant.py",
    "ros2/intercycle_eval/intercycle_eval/plant.py",   # body-shop plant: writes plant_state.json
)
# Files that name truth paths without opening them (contracts, this guard).
CONTRACT_FILES = ("twin/truth_guard.py", "twin/schemas/")
# Process launchers that may import truth modules to start the plant / generator (they never open truth themselves).
ORCHESTRATORS = ("scripts/slice.py", "scripts/sealed_run.py")
# Modules whose import gives access to truth.
TRUTH_MODULES = ("twin.engine", "twin.plant", "twin.camera", "twin.camera_v0", "twin.oracle_feed", "twin.daygen",
                 "twin.daygen_stub", "twin.margin_calib", "twin.eval_metrics")

LOG_HEADER = "# truth access log (PLAN_TWIN §2): utc_iso<TAB>caller<TAB>mode<TAB>path<TAB>pid\n"


def is_truth_path(path: str | os.PathLike) -> bool:
    name = Path(path).name
    return any(tok in name for tok in TRUTH_TOKENS)


def is_truth_module(mod: str) -> bool:
    return any(mod == m or mod.startswith(m + ".") for m in TRUTH_MODULES)


def _rel(filename: str) -> str:
    p = Path(filename).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return f"<outside repo>/{p.name}"


def log_path() -> Path:
    return Path(os.environ.get("INTERCYCLE_TRUTH_LOG") or REPO / "results" / "truth_access.log")


def log_access(caller: str, mode: str, path: str | os.PathLike) -> None:
    lp = log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    new = not lp.exists() or lp.stat().st_size == 0
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")
    with open(lp, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write(LOG_HEADER)
        f.write(f"{stamp}\t{caller}\t{mode}\t{Path(path).as_posix()}\t{os.getpid()}\n")


def open_truth(path: str | os.PathLike, mode: str = "r", **kw: Any) -> IO[Any]:
    """open() for truth files: refuses callers outside the allowlist, then logs the access."""
    if not is_truth_path(path):
        raise ValueError(f"{path} is not a truth path (tokens: {TRUTH_TOKENS})")
    frame = inspect.currentframe()
    caller = _rel(frame.f_back.f_code.co_filename) if frame and frame.f_back else "<unknown>"
    writing = any(c in mode for c in "wax+")
    allowed = TRUTH_READERS + TRUTH_PRODUCERS if writing else TRUTH_READERS
    if caller not in allowed:
        raise PermissionError(f"{caller} may not open truth ({path}, mode {mode!r}); allowlist: {allowed}")
    log_access(caller, mode, path)
    if "b" not in mode:
        kw.setdefault("encoding", "utf-8")
    return open(path, mode, **kw)
