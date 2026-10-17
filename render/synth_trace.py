"""Synthetic trace.parquet for exercising the render pipelines before the real sim trace exists.

    python render/synth_trace.py --out render/out/trace_synth.parquet [--minutes 4]
    python render/synth_trace.py --movers sim --minutes 15 --out D:/icdata/renders/trace_sim15.parquet
    python render/synth_trace.py --movers sim --machines sim --seed 3 --minutes 45 --out D:/icdata/renders/trace_a1.parquet

`--movers sim` replaces the hand-made walkers/vehicles with the sim's own movers (intercycle/truth/agents.py: AGVs,
tuggers, forklifts, aisle walkers, data-hall techs; break crowds when a break falls in the window), sampled every
MOVER_DT s on their unobstructed paths (the sim's MoverSystem may still hold them for obstacles). Run it with the
repo's Python (needs pydantic). `--machines sim` replaces the takt-based machine states with the sim's own kinematic
timeline (intercycle/sim/world.World + intercycle/truth/kinematics.Kinematics.phase, same seed): one trace row per
state change (ACTIVE / IDLE / HELD / TRIPPED) plus a sidecar `<out>.kin.parquet` (t, entity_id, state, pose_f) every
KIN_DT s, where pose_f is the active fraction that drives the arms (held through HOLD / TRIP, when the machine clock
stops; -1 = home). scene_real.py --kin reads it, so the rendered arms follow the same timeline the phase truth uses
(scripts/phase_discovery.truth_label). The hero AMR stays synthetic.

Follows the TRACE CONTRACT documented at the top of render/isaac/replay.py (t, entity_id, cls, x, z,
yaw, state). Machines cycle ACTIVE/IDLE from bodyshop.yaml (line stations on one takt, cells on their
own period); the hero AMR drives from the dock to the north light curtain of S30, WAITS until S30 is
idle, ENTERs, EXITs, then inspects a data hall rack. A few walkers, a forklift, an AGV and a tugger move
along the aisles. NOT the real sim: do not use for any metric or claim.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DAY0_T = 10 * 3600.0     # 10:00 sim time (line running)
MOVER_DT = 0.25          # s between samples of the sim movers (--movers sim)
KIN_DT = 0.02            # s between samples of the sim's machine kinematics (--machines sim)
STATE = {"active": "ACTIVE", "idle": "IDLE", "hold": "HELD", "trip": "TRIPPED"}


def sim_movers(t0: float, t1: float, seed: int, scenario_path: str) -> list[tuple]:
    """Rows for every intercycle.truth.agents mover present in [t0, t1], one sample every MOVER_DT s while present."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # repo root -> intercycle package
    from intercycle.scenario import load
    from intercycle.truth.agents import agents
    from intercycle.types import MoverClass
    cls = {MoverClass.PERSON: "person", MoverClass.AGV: "agv", MoverClass.TUGGER: "tugger", MoverClass.FORKLIFT: "forklift"}
    rows = []
    ts = np.arange(t0, t1 + 1e-9, MOVER_DT)
    for a in agents(load(scenario_path), seed):
        if a.cls not in cls:
            continue
        state = "walk" if a.cls == MoverClass.PERSON else "drive"
        for t in ts:
            st = a.state(float(t))
            if st is not None:
                rows.append((float(t), a.id, cls[a.cls], st[0], st[1], math.nan, state))
    return rows


def sim_machines(scn_yaml: dict, t0: float, t1: float, seed: int, scenario_path: str) -> tuple[list[tuple], pd.DataFrame]:
    """Machine rows from the sim's kinematics (one per state change) and the KIN_DT sidecar (t, entity_id, state, pose_f)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # repo root -> intercycle package
    from intercycle.scenario import load
    from intercycle.sim.world import World
    from intercycle.truth.kinematics import Kinematics
    scn = load(scenario_path)
    kin = Kinematics(scn, World(scn, seed, t0, t1))
    pos = {m["id"]: (m["x"], m["z"]) for m in scn_yaml["machines"]}
    ts = np.arange(t0, t1 + 1e-9, KIN_DT)
    rows, side = [], []
    for m in scn.machines:
        st = np.empty(len(ts), object)
        pf = np.empty(len(ts), np.float32)
        last, f_pose = None, -1.0
        for k, t in enumerate(ts):
            s_, fr = kin.phase(m.id, float(t))
            if s_ == "active":
                f_pose = fr
            elif s_ == "idle":
                f_pose = -1.0
            st[k], pf[k] = STATE[s_], f_pose                 # hold / trip: the clock stops, the arms keep their pose
            if st[k] != last:
                rows.append((float(t), m.id, "machine", *pos[m.id], 0.0, st[k]))
                last = st[k]
        side.append(pd.DataFrame({"t": ts, "entity_id": m.id, "state": st, "pose_f": pf}))
    return rows, pd.concat(side, ignore_index=True)


def build(scn: dict, minutes: float, seed: int = 3, movers: str = "synth", scenario_path: str = "",
          machines: str = "synth") -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(trace, kinematics sidecar or None)."""
    rng = np.random.default_rng(seed)
    t0, t1 = DAY0_T, DAY0_T + minutes * 60
    rows: list[tuple] = []
    takt = scn["line"]["takt_s"]
    s30_idle: list[tuple[float, float]] = []
    kin = None
    if machines == "sim":
        mrows, kin = sim_machines(scn, t0, t1, seed, scenario_path)
        rows += mrows
        s30 = sorted((t, st) for t, eid, *_, st in mrows if eid == "S30")
        s30_idle = [(a, b) for (a, sa), (b, _) in zip(s30, s30[1:]) if sa == "IDLE"]
    for m in scn["machines"] if machines == "synth" else []:
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
    if movers == "sim":
        rows += sim_movers(t0, t1, seed, scenario_path)
        return pd.DataFrame(rows, columns=["t", "entity_id", "cls", "x", "z", "yaw", "state"]).sort_values(["entity_id", "t"]), kin
    # traffic
    for i in range(6):
        x0, x1 = rng.uniform(5, 40), rng.uniform(55, 90)
        zz = rng.choice([32.0, 16.0, 8.0])
        rows += [(t0, f"person_{i}", "person", x0, zz + rng.uniform(-1, 1), math.nan, "walk"),
                 (t1, f"person_{i}", "person", x1, zz + rng.uniform(-1, 1), math.nan, "walk")]
    rows += [(t0, "forklift_0", "forklift", 4.0, 8.0, math.nan, "drive"), (t1, "forklift_0", "forklift", 90.0, 8.0, math.nan, "drive"),
             (t0, "agv_0", "agv", 90.0, 16.0, math.nan, "drive"), (t1, "agv_0", "agv", 4.0, 16.0, math.nan, "drive"),
             (t0, "tugger_0", "tugger", 4.0, 7.0, math.nan, "drive"), (t1, "tugger_0", "tugger", 90.0, 7.0, math.nan, "drive")]
    return pd.DataFrame(rows, columns=["t", "entity_id", "cls", "x", "z", "yaw", "state"]).sort_values(["entity_id", "t"]), kin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenario/bodyshop.yaml")
    ap.add_argument("--out", default="render/out/trace_synth.parquet")
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--movers", choices=("synth", "sim"), default="synth", help="hand-made traffic or the sim's agents.py movers")
    ap.add_argument("--machines", choices=("synth", "sim"), default="synth",
                    help="takt-based machine states or the sim's kinematic timeline (+ <out>.kin.parquet)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    scn = yaml.safe_load(open(a.scenario, encoding="utf-8"))
    df, kin = build(scn, a.minutes, a.seed, a.movers, a.scenario, a.machines)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"wrote {a.out}: {len(df)} rows, {df.entity_id.nunique()} entities")
    if kin is not None:
        kp = kin_path(a.out)
        kin.astype({"entity_id": "category", "state": "category"}).to_parquet(kp)
        print(f"wrote {kp}: {len(kin)} kinematics samples (seed {a.seed})")


def kin_path(trace: str | Path) -> Path:
    """Sidecar of a trace: trace_a1.parquet -> trace_a1.kin.parquet."""
    p = Path(trace)
    return p.with_name(p.stem + ".kin.parquet")


if __name__ == "__main__":
    main()
