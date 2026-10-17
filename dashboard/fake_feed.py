"""TEST FIXTURE for the dashboard: fake numbers, never a result. Exercises the growing-map axis (1 h -> 1 week), events,
UNKNOWN cells and a new machine, in replay (files) and live (websocket) mode, before M's map_timeline.py exists.

    python dashboard/fake_feed.py [--run _fixture_feed] [--live] [--delay 0.5]

Starts from the real out/layered_map.npz layers and degrades them for short observation times; metrics are made up
(titled "TEST FIXTURE" in the run meta). Writes out/stream/<run>/ and does NOT register the run in dashboard/runs.json.
Open with dashboard/index.html?runpath=../out/stream/_fixture_feed (or ?mode=live while --live runs).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer.stream import StreamWriter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="_fixture_feed")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    m = M.LayeredMap.load(ROOT / "out" / "layered_map.npz")
    s = m.spec["site"]
    grid = dict(x0=s["x0"], z0=s["z0"], res=s["res"], nx=s["nx"], nz=s["nz"])
    ids = m.machine_ids()
    rng = np.random.default_rng(0)
    full = {mid: [(b, *bx) for b in range(M.N_BINS) for bx in m.box_cover(mid, b, max_boxes=6)] for mid in ids}
    new_box = (66.0, 37.5, 72.0, 44.0)                     # fake new machine (north of S70)
    t_new, t_learn = 4 * 86400.0, 4 * 86400.0 + 5 * 3600.0
    sw = StreamWriter(a.run, live=a.live, port=a.port, out_root=ROOT / "out" / "stream",
                      meta=dict(title="TEST FIXTURE (fake numbers, dashboard test only)", axis="t_obs", phase_bins=16, stop_bin=16,
                                period_s={k: float(v) for k, v in zip(ids, m.period)}))
    ts = np.exp(np.linspace(math.log(3600.0), math.log(7 * 86400.0), a.n))
    flags = dict(change=False, spawn=False, named=False)
    for k, t in enumerate(ts):
        conf = min(1.0, math.log(t / 1800.0) / math.log(7 * 86400 / 1800.0))
        tb = M.tod_bin(t)
        p2, h2 = m.traffic(M.VEHICLE, tb, M.RUNNING)
        p3, h3 = m.traffic(M.PERSON, tb, M.RUNNING)
        noise = (1 - conf) * 0.25
        p2 = np.clip(p2 * conf + noise * rng.random(p2.shape) * (rng.random(p2.shape) < 0.15), 0, 1)
        p3 = np.clip(p3 * conf + noise * rng.random(p3.shape) * (rng.random(p3.shape) < 0.15), 0, 1)
        unk = rng.random((s["nz"], s["nx"])) < max(0.0, 0.6 - conf)          # never-observed cells shrink
        l1 = {mid: [r for r in rows if rng.random() < 0.35 + 0.65 * conf] for mid, rows in full.items()}
        if t >= t_new:
            x0, z0, x1, z1 = new_box
            if t < t_learn:
                unk[int(z0):int(z1), int(x0):int(x1)] = True
            else:
                l1["NEW1"] = [(b, x0 + 0.5 + 0.2 * (b % 4), z0 + 0.5, x0 + 3.0 + 0.2 * (b % 4), z1 - 0.5) for b in range(16)]
        if t >= t_new and not flags["change"]:
            sw.event(t_sim=t_new + 600, kind="change_detected", x=69.0, z=40.75, box=list(new_box), caption="Persistent motion where L0 says static: cells set UNKNOWN (= occupied)")
            flags["change"] = True
        if t >= t_learn and not flags["spawn"]:
            sw.event(t_sim=t_learn, kind="machine_spawned", machine="NEW1", x=69.0, z=40.75, caption="NEW1: period converged after 12 stable cycles, L1 model spawned")
            flags["spawn"] = True
        if t >= 2 * 86400 and not flags["named"]:
            sw.event(t_sim=2 * 86400, kind="phase_named", machine="S30", start=0.0, end=0.55, name="weld", caption="Gemma names S30 phase 1 'weld'")
            sw.event(t_sim=2 * 86400 + 60, kind="phase_named", machine="S30", start=0.55, end=0.7, name="tip-dress", caption="Gemma names S30 phase 2 'tip-dress'")
            sw.event(t_sim=2 * 86400 + 120, kind="phase_named", machine="S30", start=0.7, end=1.0, name="transfer", caption="Gemma names S30 phase 3 'transfer'")
            flags["named"] = True
        phase_now = {mid: int(((t % per) / per) * 16) for mid, per in zip(ids, m.period)}
        if "NEW1" in l1:
            phase_now["NEW1"] = int(t / 50.0) % 16
        transition = t_new <= t < t_learn
        sw.snapshot(t_sim=t, t_obs=t, prod_state="running", grid=grid, l1=l1, l2_occ=p2.T, l2_head=h2.T, l3_occ=p3.T, l3_head=h3.T,
                    unknown=unk, phase_now=phase_now,
                    metrics=dict(l1_iou=0.55 + 0.38 * conf, unsafe_free=0.004 * (1 - conf) if not transition else 0.0005, brier=0.08 * (1 - conf) + 0.01,
                                 false_occupied=0.3 * (1 - conf) + 0.06 + (0.05 if transition else 0)))
        if a.delay:
            time.sleep(a.delay)
    sw.close()
    print(f"wrote {a.n} fixture snapshots to out/stream/{a.run}")


if __name__ == "__main__":
    main()
