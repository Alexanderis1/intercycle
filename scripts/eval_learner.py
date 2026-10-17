"""Evaluate the cycle learner against ground truth on SYNTHETIC camera signals.

For each seed and each of the 10 machines: 45 min of camera signal from 07:00, learn, then nowcast at
issue times from 07:45 and compare the predicted next idle start with the truth. Line stations are
learned jointly (learn_line); cells individually.

    python scripts/eval_learner.py [--seeds 20] [--minutes 45] [--no-pool] [--ttl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intercycle.observer import learn_cycles as LC  # noqa: E402
from intercycle.observer import signals  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.truth import machines as truth  # noqa: E402

T0 = 7 * 3600.0
HORIZONS = (0, 15, 30, 60)
ISSUES = 6          # nowcast issue times: 07:45 + k * 150 s


def run(seeds: int, minutes: float, pool: bool) -> dict:
    scn = load()
    span = minutes * 60
    t_end = T0 + span
    ids = [m.id for m in scn.machines]
    line = [m.id for m in scn.machines if m.on_line]
    out: dict[str, list[dict]] = {i: [] for i in ids}
    for seed in range(seeds):
        sig = {i: signals.machine_signal(scn, i, seed, T0, t_end + ISSUES * 150 + 120) for i in ids}
        learned = {}
        if pool:
            for i, L in zip(line, LC.learn_line([sig[i] for i in line], (T0, t_end), [ids.index(i) for i in line])):
                learned[i] = L
        for i in ids:
            if i not in learned:
                learned[i] = LC.learn(*sig[i], window=(T0, t_end), index=ids.index(i))
        lf = {}                                   # line-aware forecasts per issue time
        if pool:
            mods = [learned[i].model for i in line]
            offs = [learned[i].diag.get("offset_s", 0.0) for i in line]
            for k in range(ISSUES):
                lf[k] = dict(zip(line, LC.line_forecast(mods, [sig[i] for i in line], offs, t_end + 150 * k)))
        for i in ids:
            m, L = scn.machine(i), learned[i]
            md = L.model
            ws = truth.windows(scn, i, seed, T0, t_end + ISSUES * 150 + 400)
            win = [w for w in ws if T0 <= w.bs < t_end]
            cyc = np.array([w.period for w in win if w.kind == "cycle"])
            P_true = scn.line.takt_s if m.on_line else m.period_s
            rec = dict(perr=100 * abs(md.period_s / P_true - 1), berr=100 * abs(md.busy_fraction - m.busy),
                       jit=100 * md.jitter, jit_true=100 * cyc.std(ddof=1) / cyc.mean(), stops=md.stops_per_h,
                       stops_true=sum(w.kind == "andon" for w in win) / (span / 3600), conf=md.confidence,
                       idle_err=abs(L.idle_s - (1 - m.busy) * P_true), nc={h: [] for h in HORIZONS})
            falls = np.array([w.is_ for w in ws])
            for k in range(ISSUES):
                t_now = t_end + 150 * k
                fc = lf[k][i] if i in lf.get(k, {}) else LC.forecast(md, *sig[i], t_now)
                for h in HORIZONS:
                    pred = fc.fall_after(t_now + h)
                    rec["nc"][h].append(pred - falls[np.abs(falls - pred).argmin()])
            out[i].append(rec)
    return out


def table(out: dict) -> None:
    hdr = f"{'id':6}{'Perr%':>7}{'busy pt':>8}{'jit% e/t':>10}{'stop/h e/t':>12}{'conf avg/min':>14}"
    hdr += "".join(f"{'nowcast h=' + str(h):>16}" for h in HORIZONS)
    print(hdr)
    print(" " * 57 + "".join(f"{'med|e|s  <=1s':>16}" for _ in HORIZONS))
    allnc = {h: [] for h in HORIZONS}
    for i, rows in out.items():
        g = lambda k: np.array([r[k] for r in rows])
        s = f"{i:6}{g('perr').mean():7.3f}{g('berr').mean():8.2f}{g('jit').mean():6.2f}/{g('jit_true').mean():.2f}"
        s += f"{g('stops').mean():7.1f}/{g('stops_true').mean():.1f}{g('conf').mean():9.1f}/{g('conf').min():<3d}"
        for h in HORIZONS:
            e = np.abs(np.concatenate([r["nc"][h] for r in rows]))
            allnc[h].append(e)
            s += f"{np.median(e):9.2f} {100 * (e <= 1).mean():4.0f}%"
        print(s)
    g = lambda k: np.concatenate([[r[k] for r in rows] for rows in out.values()])
    s = f"{'ALL':6}{g('perr').mean():7.3f}{g('berr').mean():8.2f}{g('jit').mean():6.2f}/{g('jit_true').mean():.2f}"
    s += f"{g('stops').mean():7.1f}/{g('stops_true').mean():.1f}{g('conf').mean():9.1f}/{int(g('conf').min()):<3d}"
    for h in HORIZONS:
        e = np.concatenate(allnc[h])
        s += f"{np.median(e):9.2f} {100 * (e <= 1).mean():4.0f}%"
    print(s)
    print(f"period err: share <0.3% = {100 * (g('perr') < 0.3).mean():.0f}%  | busy err max = {g('berr').max():.2f} pt | "
          f"conf<4: {int((g('conf') < 4).sum())}/{len(g('conf'))}")


def ttl_mc(period: float, jitter: float, tol_frac: float = 0.025, n: int = 3000, t_max: int = 2400, seed: int = 0) -> float:
    """Monte Carlo of the cycle-boundary random walk (variance rate jitter^2 * period per second): half the
    time until the 95 % |phase error| exceeds tol_frac * period."""
    rng = np.random.default_rng(seed)
    err = np.abs(np.cumsum(rng.standard_normal((n, t_max)) * jitter * np.sqrt(period), axis=1))
    hit = np.flatnonzero(np.quantile(err, 0.95, axis=0) >= tol_frac * period)
    return 0.5 * (hit[0] + 1) if len(hit) else float("inf")


def ttl_table() -> None:
    print("\nTTL rule vs Monte Carlo (period 60 s, no stops; MC = half the time until the 95% phase error reaches 2.5% of the period)")
    print(f"{'jitter':>8}{'MC (s)':>9}{'rule (s)':>10}")
    for j in (0.003, 0.005, 0.01, 0.02, 0.05):
        print(f"{100 * j:7.1f}%{ttl_mc(60, j):9.1f}{LC.ttl_s(60, j, 0):10.1f}")
    print("stops cap: ttl_s(60, 0.005, s/h) =", {s: round(LC.ttl_s(60, 0.005, s), 1) for s in (0, 0.5, 1, 2, 4)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--no-pool", action="store_true", help="learn line stations individually")
    ap.add_argument("--ttl", action="store_true", help="also print the TTL Monte Carlo check")
    a = ap.parse_args()
    table(run(a.seeds, a.minutes, not a.no_pool))
    if a.ttl:
        ttl_table()
