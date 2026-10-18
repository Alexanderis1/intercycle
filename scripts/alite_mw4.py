"""A-lite M-W4 scenario runner (PLAN_TWIN §7 item 4; SIMULATED body shop, descriptive only).

    .venv/Scripts/python.exe scripts/alite_mw4.py --worlds 801,802 --learn-h 24 --after-min 30 --out results/alite/mw4.json

Per world: the A-lite observer learns the floor for --learn-h hours from 06:05 of day 0, then a program change (S60)
and 4 new arms 0.5-2 m from an existing cell appear (intercycle/alite/mw4.py) and the run goes on for --after-min.
Writes one JSON with every world's result and prints a compact table. Never part of sealed_run.py.
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

from intercycle.alite import mw4  # noqa: E402
from intercycle.alite.world import day_bounds  # noqa: E402
from intercycle.scenario import load  # noqa: E402


def one(args: tuple[int, float, float]) -> dict:
    seed, learn_h, after_min = args
    scn = load()
    t0 = day_bounds(scn)[1]
    a = time.time()
    r = mw4.run(scn, seed, t0, learn_h * 3600.0, after_min * 60.0)
    r["wall_s"] = round(time.time() - a, 1)
    return r


def table(results: list[dict]) -> str:
    rows = ["world | change | t_unknown_s | t_cover_s | unknown share 0-1 / 1-5 / 5-30 min | false alarms before"]
    for r in results:
        for name, c in r["changes"].items():
            rows.append(f"{r['world_seed']} | {name} | {c['t_unknown_s']:.1f} | {c['t_cover_s']:.1f} | "
                        f"{c['unknown_0_1min']:.2f} / {c['unknown_1_5min']:.2f} / {c['unknown_5_30min']:.2f} | "
                        f"{r['false_alarms_before_change']} in {r['detector_hours_before_change']:.1f} h")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A-lite M-W4 scenario runner (SIMULATED, descriptive)")
    ap.add_argument("--worlds", default="801")
    ap.add_argument("--learn-h", type=float, default=24.0)
    ap.add_argument("--after-min", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.worlds.split(",") if s]
    jobs = [(s, a.learn_h, a.after_min) for s in seeds]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, jobs))
    else:
        res = [one(j) for j in jobs]
    out = {"kind": "alite_mw4", "simulated": True, "learn_h": a.learn_h, "after_min": a.after_min, "worlds": res}
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(table(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
