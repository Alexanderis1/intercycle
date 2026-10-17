"""Run every strategy over a seed range, aggregate with 95% CIs, and sweep jitter x andon rate.

    python -m intercycle.eval.run_seeds --seeds 0 19            # development seeds
    python -m intercycle.eval.run_seeds --heldout               # held-out seeds 1000-1199, run once after freeze
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from intercycle.scenario import load
from intercycle.sim.observer_learned import learned_models
from intercycle.sim.run import STRATEGIES, run

OUT = Path(__file__).resolve().parents[2] / "out"
KEYS = ["machine_insp_per_prod_h", "line_time_lost_s", "line_cost_usd", "trips", "holds", "hold_s", "mean_wait_s", "zero_loss_share"]


def _scn(jitter, andon):
    scn = load()
    if jitter is not None or andon is not None:
        line = scn.line.model_copy(update={k: v for k, v in (("jitter", jitter), ("andon_stops_per_h", andon)) if v is not None})
        scn = scn.model_copy(update={"line": line})
    return scn


def _one(args):
    strategy, seed, jitter, andon, t0, t1 = args
    return run(strategy, seed, t0, t1, scn=_scn(jitter, andon)).metrics


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for s in STRATEGIES:
        rs = [r for r in rows if r["strategy"] == s]
        out[s] = {}
        for k in KEYS:
            v = np.array([r[k] for r in rs], dtype=float)
            ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            out[s][k] = {"mean": float(v.mean()), "ci95": float(ci)}
        out[s]["n"] = len(rs)
    a, b, c = (out[s]["line_time_lost_s"]["mean"] for s in ("A", "B", "C"))
    out["gain_retained_line_time"] = (a - b) / (a - c) if a - c > 1e-9 else None
    ib, ic = out["B"]["machine_insp_per_prod_h"]["mean"], out["C"]["machine_insp_per_prod_h"]["mean"]
    out["throughput_vs_oracle"] = ib / ic if ic else None
    return out


def evaluate(seeds, t0=7 * 3600, t1=13 * 3600, jitter=None, andon=None, workers=6) -> dict:
    learned_models(_scn(jitter, andon))           # train the observer once (training day, disk-cached) before fanning out
    jobs = [(s, k, jitter, andon, t0, t1) for k in seeds for s in STRATEGIES]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, jobs))
    return {"seeds": [min(seeds), max(seeds)], "window": [t0, t1], "jitter": jitter, "andon_per_h": andon,
            "aggregate": aggregate(rows), "rows": rows}


def sweep(seeds) -> list[dict]:
    res = []
    for j in (0.005, 0.01, 0.02, 0.05):
        for a in (0.0, 4.0, 12.0):
            e = evaluate(seeds, jitter=j, andon=a)
            ag = e["aggregate"]
            res.append({"jitter": j, "andon_per_h": a, "gain_retained": ag["gain_retained_line_time"],
                        "throughput_vs_oracle": ag["throughput_vs_oracle"],
                        **{f"{s}_{k}": ag[s][k]["mean"] for s in STRATEGIES for k in ("machine_insp_per_prod_h", "line_time_lost_s", "trips")}})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 19])
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    seeds = range(1000, 1200) if args.heldout else range(args.seeds[0], args.seeds[1] + 1)
    tag = "heldout" if args.heldout else f"dev_{seeds[0]}_{seeds[-1]}"
    e = evaluate(seeds)
    (OUT / f"results_{tag}.json").write_text(json.dumps(e, indent=1))
    ag = e["aggregate"]
    print(f"{'robot':6s} {'insp/h':>14s} {'line lost s':>16s} {'trips':>11s} {'holds':>11s} {'wait s':>12s}")
    for s in STRATEGIES:
        f = lambda k, d=1: f"{ag[s][k]['mean']:.{d}f} ±{ag[s][k]['ci95']:.{d}f}"
        print(f"{s:6s} {f('machine_insp_per_prod_h'):>14s} {f('line_time_lost_s',0):>16s} {f('trips'):>11s} {f('holds'):>11s} {f('mean_wait_s'):>12s}")
    print("gain retained (line time):", ag["gain_retained_line_time"], " B/C throughput:", round(ag["throughput_vs_oracle"], 3))
    if args.sweep:
        sw = sweep(range(seeds[0], seeds[0] + 8))
        (OUT / "sweep.json").write_text(json.dumps(sw, indent=1))
        for r in sw:
            print(r["jitter"], r["andon_per_h"], "gain", None if r["gain_retained"] is None else round(r["gain_retained"], 3),
                  "B/C insp", round(r["throughput_vs_oracle"], 3), "A lost", round(r["A_line_time_lost_s"]), "B lost", round(r["B_line_time_lost_s"]))
