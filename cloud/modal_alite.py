"""A-lite runs on Modal CPUs (PLAN_TWIN §7 item 5; SIMULATED body shop, descriptive only; docs/ALITE_PREREG.md).

    modal run cloud/modal_alite.py --tag alite_v1                      # the pre-registered run (12 episodes + M-W4)
    modal run cloud/modal_alite.py --tag smoke --dry-run --worlds 801 --conditions learned --mw4-worlds ""

One container per episode (world x condition, intercycle/alite/episode.py) and one per M-W4 world
(intercycle/alite/mw4.py, 24 h learning + 30 min after the changes). Writes results/alite/<tag>.json locally
and prints the same tables as scripts/alite_run.py / scripts/alite_mw4.py. App name "s8-alite" (lane S8).
Never part of scripts/sealed_run.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
IGNORE = ["**/__pycache__", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "astropy", "pyyaml", "pydantic>=2", "pynacl", "pandas", "pyarrow")
         .add_local_dir(str(REPO / "intercycle"), "/root/intercycle", ignore=IGNORE)
         .add_local_dir(str(REPO / "twin"), "/root/twin", ignore=IGNORE)
         .add_local_dir(str(REPO / "scenario"), "/root/scenario"))
app = modal.App("s8-alite", image=image)

DRY = dict(learn_s=1800.0, mission_starts_h=(6.75,), mission_s=300.0, mission_day=0)   # = scripts/alite_run.py DRY
MW4_LEARN_H, MW4_AFTER_MIN = 24.0, 30.0


@app.function(cpu=1.0, memory=2048, timeout=2 * 3600, retries=0)
def episode(seed: int, cond: str, dry: bool) -> dict:
    from intercycle.alite import episode as E
    from intercycle.scenario import load
    a = time.time()
    r = E.run(load(), seed, cond, **(DRY if dry else {}))
    r["wall_s"] = round(time.time() - a, 1)
    r["dry_run"] = bool(dry)
    return r


@app.function(cpu=1.0, memory=2048, timeout=2 * 3600, retries=0)
def mw4_world(seed: int, learn_h: float, after_min: float) -> dict:
    from intercycle.alite import mw4
    from intercycle.alite.world import day_bounds
    from intercycle.scenario import load
    scn = load()
    a = time.time()
    r = mw4.run(scn, seed, day_bounds(scn)[1], learn_h * 3600.0, after_min * 60.0)
    r["wall_s"] = round(time.time() - a, 1)
    return r


@app.local_entrypoint()
def main(tag: str = "alite_v1", worlds: str = "811,812,813,814", conditions: str = "learned,reactive,static",
         mw4_worlds: str = "811,812,813,814", dry_run: bool = False):
    sys.path.insert(0, str(REPO / "scripts"))
    import alite_mw4  # pyright: ignore[reportMissingImports]  (scripts/ on sys.path)
    import alite_run  # pyright: ignore[reportMissingImports]
    jobs = [(int(s), c, dry_run) for s in worlds.split(",") if s for c in conditions.split(",") if c]
    mjobs = [(int(s), MW4_LEARN_H, MW4_AFTER_MIN) for s in mw4_worlds.split(",") if s]
    a = time.time()
    calls = [mw4_world.spawn(*j) for j in mjobs]               # the longest jobs start first
    rows = list(episode.starmap(jobs, return_exceptions=True))
    mres = []
    for c in calls:
        try:
            mres.append(c.get())
        except Exception as e:                                  # reported, never dropped silently
            mres.append(e)
    ok = [r for r in rows if isinstance(r, dict)]
    mok = [r for r in mres if isinstance(r, dict)]
    cpu_s = sum(r["wall_s"] for r in ok + mok)
    out = {"kind": "alite_modal", "simulated": True, "tag": tag, "dry_run": dry_run, "rows": ok,
           "mw4": {"learn_h": MW4_LEARN_H, "after_min": MW4_AFTER_MIN, "worlds": mok},
           "errors": [repr(r) for r in rows + mres if not isinstance(r, dict)],
           "wall_total_s": round(time.time() - a, 1), "container_s_sum": round(cpu_s, 1),
           # Modal CPU $0.0000131 / core-s + memory $0.00000222 / GiB-s (docs/probes/H1.md), 1 core + 2 GiB each
           "usd_estimate": round(cpu_s * (0.0000131 + 2 * 0.00000222), 3)}
    dst = REPO / "results" / "alite" / f"{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    if ok:
        print(alite_run.table(ok))
    if mok:
        print(alite_mw4.table(mok))
    print(f"{len(ok)}/{len(jobs)} episodes, {len(mok)}/{len(mjobs)} M-W4 worlds ok; "
          f"~${out['usd_estimate']} -> {dst.relative_to(REPO).as_posix()}")
