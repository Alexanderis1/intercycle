"""A-lite run plan on the CPU double (PLAN_TWIN §7 item 5; SIMULATED body shop, descriptive only).

    # the pre-registered run (docs/ALITE_PREREG.md), locally or one process per episode:
    .venv/Scripts/python.exe scripts/alite_run.py --worlds 811,812,813,814 --out results/alite/episodes.json
    # dry run: 1 short episode (30 min of learning on day 0, one 5 min mission right after)
    .venv/Scripts/python.exe scripts/alite_run.py --dry-run --worlds 801 --conditions learned --out out/alite/dry.json

Each (world, condition) is one Episode (intercycle/alite/episode.py): day 0 the observer learns, day 1 the robot runs
its missions. Prints the descriptive table (per condition: sums over worlds and missions; no p-values). Never part of
scripts/sealed_run.py. On Modal: cloud/modal_alite.py runs the same `one()` per episode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from intercycle.alite import episode as E  # noqa: E402
from intercycle.scenario import load  # noqa: E402

DRY = dict(learn_s=1800.0, mission_starts_h=(6.75,), mission_s=300.0, mission_day=0)
KEYS = ("captures_attempted", "captures_passed", "entries_attempted", "entries_clean", "entries_declined", "trips",
        "holds", "lost_s", "Y")


def one(job: tuple[int, str, bool]) -> dict:
    seed, cond, dry = job
    a = time.time()
    r = E.run(load(), seed, cond, **(DRY if dry else {}))
    r["wall_s"] = round(time.time() - a, 1)
    r["dry_run"] = bool(dry)
    return r


def table(rows: list[dict]) -> str:
    out = ["condition | episodes | missions | " + " | ".join(KEYS) + " | Y per mission"]
    for c in E.CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        tot = {k: sum(r["totals"][k] for r in rs) for k in KEYS}
        nm = sum(r["n_missions"] for r in rs)
        out.append(f"{c} | {len(rs)} | {nm} | " + " | ".join(f"{tot[k]:g}" for k in KEYS) +
                   f" | {tot['Y'] / nm if nm else float('nan'):.2f}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A-lite CPU double runs (SIMULATED, descriptive)")
    ap.add_argument("--worlds", default="811,812,813,814")
    ap.add_argument("--conditions", default=",".join(E.CONDITIONS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    jobs = [(int(s), c, a.dry_run) for s in a.worlds.split(",") if s for c in a.conditions.split(",") if c]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, jobs))
    else:
        rows = [one(j) for j in jobs]
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "alite_runs", "simulated": True, "rows": rows}, indent=1, default=str),
                     encoding="utf-8")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
