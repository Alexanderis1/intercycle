"""Fan the collision-checked simulation out on Modal CPUs.

    modal run cloud/modal_sim.py --seeds 0-19 --t0 7 --t1 13 --tag dev
Writes out/sweeps/<tag>.json locally.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "astropy", "pyyaml", "pydantic>=2", "pynacl", "pandas", "pyarrow")
         .add_local_dir(str(REPO / "intercycle"), "/root/intercycle")
         .add_local_dir(str(REPO / "scenario"), "/root/scenario"))
app = modal.App("intercycle-sim", image=image)


@app.function(cpu=1.0, memory=2048, timeout=3 * 3600, retries=1)
def run_one(strategy: str, seed: int, t0: float, t1: float, fault: str | None = None) -> dict:
    import time
    from intercycle.sim.physics import run
    w = time.time()
    r = run(strategy, seed, t0, t1, fault=fault)
    m = r.metrics()
    m["wall_s"] = round(time.time() - w, 1)
    m["fault"] = fault
    m["inspections"] = r.inspections
    m["events"] = [{k: v for k, v in e.items()} for e in r.events]
    return m


@app.local_entrypoint()
def main(seeds: str = "0-19", t0: float = 7.0, t1: float = 13.0, tag: str = "dev", strategies: str = "A0,A,B,C",
         fault: str = ""):
    a, b = (int(x) for x in seeds.split("-"))
    jobs = [(s, k, t0 * 3600, t1 * 3600, fault or None) for k in range(a, b + 1) for s in strategies.split(",")]
    rows = list(run_one.starmap(jobs, return_exceptions=True))
    ok = [r for r in rows if isinstance(r, dict)]
    out = REPO / "out" / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"jobs": len(jobs), "ok": len(ok), "rows": ok,
                                                  "errors": [repr(r) for r in rows if not isinstance(r, dict)]}, indent=1))
    print(f"{len(ok)}/{len(jobs)} runs ok -> out/sweeps/{tag}.json")
