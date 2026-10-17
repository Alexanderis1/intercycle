"""Synthetic trace.parquet for exercising the render pipelines before the real sim trace exists.

    python render/synth_trace.py --out render/out/trace_synth.parquet [--minutes 4]

Follows the TRACE CONTRACT documented at the top of render/isaac/replay.py (t, entity_id, cls, x, z,
yaw, state). Machines cycle ACTIVE/IDLE from bodyshop.yaml (line stations on one takt, cells on their
own period); the hero AMR drives from the dock to the north light curtain of S30, WAITS until S30 is
idle, ENTERs, EXITs, then inspects a data hall rack. A few walkers, a forklift, an AGV and a tugger move
along the aisles. NOT the real sim: do not use for any metric or claim.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DAY0_T = 10 * 3600.0     # 10:00 sim time (line running)


def build(scn: dict, minutes: float, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = DAY0_T, DAY0_T + minutes * 60
    rows: list[tuple] = []
    takt = scn["line"]["takt_s"]
    s30_idle: list[tuple[float, float]] = []
    for m in scn["machines"]:
        per = takt if m["on_line"] else m["period_s"]
        k = 0
        while t0 + m.get("offset_s", 0.0) + k * per < t1:
            a = t0 + m.get("offset_s", 0.0) + k * per
            i = a + per * m["busy"]
            rows += [(a, m["id"], "machine", m["x"], m["z"], 0.0, "ACTIVE"), (i, m["id"], "machine", m["x"], m["z"], 0.0, "IDLE")]
            if m["id"] == "S30":
                s30_idle.append((i, a + per))
            k += 1
    # hero: dock -> north curtain of S30 -> wait for idle -> enter/exit -> data hall rack
    s30 = next(m for m in scn["machines"] if m["id"] == "S30")
    gate = (s30["x"], s30["z"] + s30["size"][1] / 2 + 1.0)
    inside = (s30["x"], s30["z"] + 2.0)
    dock = tuple(scn["robot"]["dock"])
    arrive = t0 + 40.0
    idle_start = next((a for a, _ in s30_idle if a > arrive + 5), arrive + 20)
    path = [(t0, *dock, "dock"), (t0 + 5, *dock, "move"), (arrive, *gate, "move"), (idle_start, *gate, "wait"),
            (idle_start + 0.3, *gate, "enter"), (idle_start + 4.0, *inside, "enter"), (idle_start + 8.0, *gate, "exit"),
            (idle_start + 20.0, 94.0, 24.0, "move"), (idle_start + 30.0, 100.0, 24.0, "move"), (idle_start + 45.0, 108.0, 21.0, "move"),
            (idle_start + 60.0, 108.0, 21.0, "inspect")]
    for t, x, z, st in path:
        rows.append((t, "robot_B", "wheeled_inspection_amr", x, z, math.nan, st))
    # traffic
    for i in range(6):
        x0, x1 = rng.uniform(5, 40), rng.uniform(55, 90)
        zz = rng.choice([32.0, 16.0, 8.0])
        rows += [(t0, f"person_{i}", "person", x0, zz + rng.uniform(-1, 1), math.nan, "walk"),
                 (t1, f"person_{i}", "person", x1, zz + rng.uniform(-1, 1), math.nan, "walk")]
    rows += [(t0, "forklift_0", "forklift", 4.0, 8.0, math.nan, "drive"), (t1, "forklift_0", "forklift", 90.0, 8.0, math.nan, "drive"),
             (t0, "agv_0", "agv", 90.0, 16.0, math.nan, "drive"), (t1, "agv_0", "agv", 4.0, 16.0, math.nan, "drive"),
             (t0, "tugger_0", "tugger", 4.0, 7.0, math.nan, "drive"), (t1, "tugger_0", "tugger", 90.0, 7.0, math.nan, "drive")]
    return pd.DataFrame(rows, columns=["t", "entity_id", "cls", "x", "z", "yaw", "state"]).sort_values(["entity_id", "t"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenario/bodyshop.yaml")
    ap.add_argument("--out", default="render/out/trace_synth.parquet")
    ap.add_argument("--minutes", type=float, default=4.0)
    a = ap.parse_args()
    scn = yaml.safe_load(open(a.scenario, encoding="utf-8"))
    df = build(scn, a.minutes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"wrote {a.out}: {len(df)} rows, {df.entity_id.nunique()} entities")


if __name__ == "__main__":
    main()
