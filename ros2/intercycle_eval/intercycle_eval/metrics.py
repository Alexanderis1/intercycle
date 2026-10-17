"""Per-episode metrics (DIRECTION 5.6, after Francis et al. 2023) from one episode's result + its Volume directory.

    .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/metrics.py out/robot/<batch>   -> metrics.json / .csv

Inputs per episode: <tag>.json (the Modal return: cfg, goal = timing_planner result, sim = runner summary, plant =
plant summary, eval_check) and <tag>/ (truth_log, plant_log.json, plant_events.json). Truth is used here only.

Definitions (stated in ROBOT_RESULTS):
  valid             eval_check passed (camera model fed the gate, no stub, require_camera, no error)
  collision_*       Isaac contact on a non-foot robot body, classified by the nearest body (person / vehicle / machine part
                    / static), OR a truth overlap of the robot footprint (3 discs) with a person / vehicle / machine part
                    (gap < -2 cm); counted once per contiguous event
  collisions_robot_moving  the subset with the robot's true speed > 0.1 m/s at the event (a person walking into a
                    standing robot is counted in collisions, not here)
  safe_success      valid, no collision of any kind, no trip, no fall, planner not FAILED, >= 1 inspection passed (inspections_ok)
                    (episodes are a fixed 600 s: the planner is interrupted at the end, which is not a failure)
  min_dist_person   minimum edge-to-edge distance robot footprint <-> person (m); min_ttc_* constant-velocity disc TTC
  personal_space    entries of a person centre within 0.45 m of the robot footprint edge (Hall intimate distance)
  stops / stalled   while the planner is navigating (to a wait pose, into a zone, out of it), true body speed < 0.05 m/s
                    for >= 1 s counts one stop; stalled_s = total time in such stops (proxy for collision-monitor
                    protective stops: the monitor's own state is not logged)
  SPL               per navigation leg: success * shortest / max(actual, shortest), shortest = 8-connected Dijkstra on
                    the static map at 0.2 m inflated by 0.35 m; mean over legs
  irregularity      total absolute heading change / path length (rad/m)
  windows_hit       hazard targets entered and left with no trip or hold attributed to that zone
  windows_missed    hazard targets waited out (no entry) or skipped, plus entries that tripped
  line_time_lost_s  union of line-group freeze time (trips: 90 s recovery each; holds) inside the episode;
                    cell_time_lost_s the same for HEM1 / SEAL1
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

NAV_V = 0.05
STOP_MIN_S = 1.0


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _truth_log(d: Path):
    pq = d / "truth_log.parquet"
    if pq.exists():
        import pandas as pd

        return pd.read_parquet(pq).to_dict("records")
    return _load(d / "truth_log.json") or []


class Shortest:
    """Static-map shortest paths (0.2 m grid, inflated)."""

    def __init__(self, res: float = 0.2, inflate: float = 0.35):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "intercycle_sim"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scipy import ndimage

        from intercycle_sim.world import build_world_files

        f = build_world_files(empty=False)
        pgm = f["map.pgm"]
        parts = pgm.split(b"\n", 3)
        w, h = (int(v) for v in parts[1].split())
        img = np.frombuffer(parts[3][:w * h], np.uint8).reshape(h, w)[::-1]     # row 0 = y origin
        occ = img < 128
        self.ox, self.oy, r0 = -2.0, -2.0, 0.1
        k = int(round(res / r0))
        H, W = h // k, w // k
        occ = occ[:H * k, :W * k].reshape(H, k, W, k).any(axis=(1, 3))
        occ = ndimage.binary_dilation(occ, iterations=max(1, int(round(inflate / res))))
        self.occ, self.res, self.H, self.W = occ, res, H, W
        self._cache: dict = {}

    def cell(self, x: float, y: float):
        return (int(np.clip((y - self.oy) / self.res, 0, self.H - 1)), int(np.clip((x - self.ox) / self.res, 0, self.W - 1)))

    def dist_from(self, x: float, y: float) -> np.ndarray:
        import heapq

        key = self.cell(x, y)
        if key in self._cache:
            return self._cache[key]
        D = np.full((self.H, self.W), np.inf)
        free = ~self.occ
        free[key] = True
        D[key] = 0.0
        pq = [(0.0, key)]
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > D[i, j]:
                continue
            for di, dj, c in nb:
                a, b = i + di, j + dj
                if 0 <= a < self.H and 0 <= b < self.W and free[a, b]:
                    nd = d + c * self.res
                    if nd < D[a, b]:
                        D[a, b] = nd
                        heapq.heappush(pq, (nd, (a, b)))
        self._cache[key] = D
        return D

    def length(self, a, b) -> float:
        D = self.dist_from(*a)
        i, j = self.cell(*b)
        win = D[max(0, i - 3):i + 4, max(0, j - 3):j + 4]           # goal cell may sit in the inflation: nearest free
        v = float(win.min()) if np.isfinite(win).any() else math.inf
        return v if np.isfinite(v) else math.hypot(a[0] - b[0], a[1] - b[1])


def nav_legs(goal: dict, t0: float) -> list[tuple[float, float, str]]:
    """Navigation intervals (sim time) from the planner's target records (each record = its target_done event, `t` =
    plant time when done; the next target is chosen right after). Times in the records are plant time."""
    out = []
    prev = (goal.get("t_goal_sent") or 0.0) + t0
    for r in goal.get("targets") or []:
        ts = r.get("t_at_wait")
        if ts is not None and ts > prev:
            out.append((prev - t0, ts - t0, "to_wait"))
        if r.get("t_enter") is not None and r.get("t_inside") is not None:
            out.append((r["t_enter"] - t0, r["t_inside"] - t0, "enter"))
        if r.get("t_exit") is not None and r.get("t_inside") is not None:
            out.append((r["t_inside"] - t0 + float(r.get("dwell_s") or 0.0), r["t_exit"] - t0, "exit"))
        prev = r.get("t") or prev
    return out


def target_times(goal: dict, t0: float) -> list[float]:
    out, prev = [], (goal.get("t_goal_sent") or 0.0) + t0
    for r in goal.get("targets") or []:
        if r.get("t") is not None:
            if str(r.get("outcome", "")).startswith("inspected"):
                out.append(r["t"] - prev)
            prev = r["t"]
    return out


def normalize_goal(goal: dict, t0: float) -> dict:
    """An interrupted planner (fixed-length episode) writes its raw events: rebuild the result-shaped record."""
    tg = goal.get("targets") or []
    if not any("event" in r for r in tg):
        return goal
    done = [r for r in tg if r.get("event") == "target_done"]
    chosen = [r for r in tg if r.get("event") == "chosen"]
    g = dict(goal, targets=done, events=tg)
    g["n_inspected"] = sum(1 for r in done if str(r.get("outcome", "")).startswith("inspected"))
    g.setdefault("n_targets", 10)
    if chosen and g.get("t_goal_sent") is None:
        g["t_goal_sent"] = float(chosen[0].get("ros_t", chosen[0]["t"] - t0))
    g["in_progress"] = chosen[-1]["target"] if chosen and (not done or chosen[-1]["t"] >= done[-1]["t"]) else None
    return g


def episode_metrics(res: dict, d: Path, sp: Shortest | None) -> dict:
    cfg = res.get("cfg") or {}
    goal = normalize_goal(res.get("goal") or {}, float((cfg.get("plant") or {}).get("plant_t0", 0.0)))
    sim = res.get("sim") or {}
    pl = res.get("plant") or (sim.get("plant") if sim else None) or {}
    chk = res.get("eval_check") or {}
    t0 = float((cfg.get("plant") or {}).get("plant_t0", 0.0))
    ev = _load(d / "plant_events.json") or {}
    tl = _truth_log(d)
    m = {"tag": res.get("tag") or cfg.get("tag"), "seed": cfg.get("seed"), "condition": cfg.get("condition"),
         "valid": bool(chk.get("valid")), "invalid_reasons": ";".join(chk.get("reasons") or []),
         "error": res.get("error"), "result": goal.get("result"), "timeout": goal.get("result") is None,
         "fixed_length_end": goal.get("result") == "INTERRUPTED",
         "n_targets": goal.get("n_targets"), "n_inspected": goal.get("n_inspected") or 0,
         "rtf": sim.get("rtf"), "sim_s": sim.get("sim_s"), "fell": sim.get("fell"),
         "path_length_m": sim.get("path_length_m"), "time_to_goal_s": goal.get("time_to_goal_s")}
    cb, ob = pl.get("contacts_by") or {}, pl.get("overlaps_by") or {}
    for k in ("person", "vehicle", "machine", "static"):
        m[f"collision_{k}"] = max(int(cb.get(k, 0)), int(ob.get(k, 0)))
    om = pl.get("overlaps_robot_moving_by") or {}
    cm = {}
    for c in (_load(d / "plant_events.json") or {}).get("contacts", []):
        if (c.get("robot_v") or 0) > 0.1:
            cm[c["with"]] = cm.get(c["with"], 0) + 1
    for k in ("person", "vehicle", "machine", "static"):
        m[f"collision_moving_{k}"] = max(int(om.get(k, 0)), int(cm.get(k, 0)))
    m["collisions_robot_moving"] = sum(m[f"collision_moving_{k}"] for k in ("person", "vehicle", "machine", "static"))
    m["person_overlaps_impassive"] = pl.get("person_overlaps_impassive")
    m["collisions"] = sum(m[f"collision_{k}"] for k in ("person", "vehicle", "machine", "static"))
    m["any_collision"] = m["collisions"] > 0
    mg, mt = pl.get("min_gap_m") or {}, pl.get("min_ttc_s") or {}
    m["min_dist_person_m"] = mg.get("person")
    m["min_dist_vehicle_m"] = mg.get("vehicle")
    m["min_ttc_person_s"] = mt.get("person")
    m["min_ttc_vehicle_s"] = mt.get("vehicle")
    m["personal_space_events"] = pl.get("personal_space_events")
    m["personal_space_s"] = pl.get("personal_space_s")
    m["trips"] = pl.get("trips")
    m["any_trip"] = bool(pl.get("trips"))
    m["hold_events"] = pl.get("hold_events")
    m["line_time_lost_s"] = pl.get("line_time_lost_s")
    m["cell_time_lost_s"] = pl.get("cell_time_lost_s")
    m["machine_time_lost_s"] = (pl.get("line_time_lost_s") or 0.0) + (pl.get("cell_time_lost_s") or 0.0) if pl else None
    m["forklift_brakes"] = pl.get("forklift_brakes")
    m["agv_protective_stops"] = pl.get("agv_protective_stops")
    caps = ev.get("captures") or []
    m["captures"] = len(caps)
    m["captures_pass"] = sum(1 for c in caps if c.get("pass"))
    m["capture_pass_rate"] = m["captures_pass"] / len(caps) if caps else None
    m["capture_inspect_level"] = sum(1 for c in caps if c.get("dori_inspect"))
    # inspections: success = planner says inspected AND the capture passed
    recs = goal.get("targets") or []
    cap_by = {}
    for c in caps:
        cap_by.setdefault(c.get("target"), c)
    insp = [r for r in recs if str(r.get("outcome", "")).startswith("inspected")]
    # primary pass (user decision 17 Oct ~19:20): all checks AND DORI Inspect (1000 px/m); Identify kept as secondary
    m["inspections_ok"] = sum(1 for r in insp if (cap_by.get(r.get("target")) or {}).get("pass_inspect"))
    m["inspections_ok_identify"] = sum(1 for r in insp if (cap_by.get(r.get("target")) or {}).get("pass"))
    m["inspection_success_rate"] = m["inspections_ok"] / len(recs) if recs else 0.0
    tpt = target_times(goal, t0)
    m["time_per_target_s"] = float(np.mean(tpt)) if tpt else None
    # windows
    events = ev.get("events") or []
    hit = miss = 0
    for r in recs:
        oc = str(r.get("outcome", ""))
        if oc in ("missed_no_entry", "skipped_no_window"):
            miss += 1
            continue
        if r.get("t_enter") is None:
            continue
        a, b = r["t_enter"] - t0, (r.get("t_exit") or r["t_enter"] + 60) - t0
        z = r.get("machine")
        bad = any(e["type"] in ("trip", "hold") and e.get("zone") == z and a - 1 <= e["t"] <= b + 1 for e in events)
        if bad:
            miss += 1
        else:
            hit += 1
    m["windows_hit"], m["windows_missed"] = hit, miss
    # stops / stalled / irregularity / SPL from the truth log
    if tl:
        t = np.array([r["t"] for r in tl])
        v = np.hypot([r["vx_b"] for r in tl], [r["vy_b"] for r in tl])
        psi = np.unwrap(np.array([r["psi"] for r in tl]))
        xy = np.array([(r["x"], r["y"]) for r in tl])
        L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        m["irregularity_rad_per_m"] = float(np.abs(np.diff(psi)).sum() / L) if L > 1 else None
        legs = nav_legs(goal, t0)
        nav = np.zeros(len(t), bool)
        for a, b, _ in legs:
            nav |= (t >= a + 2.0) & (t <= b)
        slow = nav & (v < NAV_V)
        stops, stalled, k = 0, 0.0, 0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
        while k < len(t):
            if slow[k]:
                j = k
                while j < len(t) and slow[j]:
                    j += 1
                dur = (j - k) * dt
                if dur >= STOP_MIN_S:
                    stops += 1
                    stalled += dur
                k = j
            else:
                k += 1
        m["stops_navigating"], m["stalled_s"] = stops, round(stalled, 1)
        if sp is not None and legs:
            spl = []
            for a, b, kind in legs:
                ia, ib = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
                ia, ib = min(ia, len(t) - 1), min(ib, len(t) - 1)
                if ib <= ia:
                    continue
                act = float(np.linalg.norm(np.diff(xy[ia:ib + 1], axis=0), axis=1).sum())
                sh = sp.length(tuple(xy[ia]), tuple(xy[ib]))
                spl.append(sh / max(act, sh, 1e-6) if act > 0.2 or sh < 0.2 else 0.0)
            m["spl"] = float(np.mean(spl)) if spl else None
    m["safe_success"] = bool(m["valid"] and not m["any_collision"] and not m["any_trip"]
                             and m["result"] not in (None, "FAILED") and not m["fell"] and m["inspections_ok"] >= 1)
    return m


def batch_metrics(batch_dir: Path, spl: bool = True) -> list[dict]:
    sp = Shortest() if spl else None
    rows = []
    for f in sorted(batch_dir.glob("*.json")):
        if f.name in ("batch.json", "metrics.json", "stats.json"):
            continue
        res = _load(f)
        if not res or "cfg" not in res:
            continue
        rows.append(episode_metrics(res, batch_dir / f.stem, sp))
    return rows


def main() -> None:
    d = Path(sys.argv[1])
    rows = batch_metrics(d, spl="--no-spl" not in sys.argv)
    (d / "metrics.json").write_text(json.dumps(rows, indent=1, default=str))
    keys = sorted({k for r in rows for k in r})
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in ("tag", "valid", "result", "n_inspected", "inspections_ok", "trips", "line_time_lost_s",
                                 "collisions", "min_dist_person_m", "spl", "rtf") if k in r})


if __name__ == "__main__":
    main()
