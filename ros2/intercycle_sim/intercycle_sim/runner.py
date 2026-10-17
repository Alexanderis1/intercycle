"""Simulator process: steps the backend in lockstep with the ROS bridge and logs the episode.

    python -u -m intercycle_sim.runner --cfg /tmp/ic/cfg.json          (Isaac python 3.11 for --sim isaac)

cfg keys: sim ("isaac" | "fake"), world_dir, sock, out_dir, seconds (sim-time limit), start [x, y, psi],
max_rtf (pace to <= this real-time factor so Nav2 gets real compute time; 0 = free-running), seed,
movers (scripted list) or mover_source ("pkg.mod:factory" for R3), stop_file (orchestrator ends the episode),
log_hz (true-pose log rate, default 10), safety_scanner (R-W13: fixed ISO 13855 protective field, default false;
summary key "safety_scanner").

Outputs in out_dir: sim_summary.json, truth_log.(parquet|json) with the TRUE pose at log_hz (EVALUATION ONLY: it
never reaches ROS), contacts.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_table(rows: list[dict], path_noext: str) -> str:
    try:
        import pandas as pd

        p = path_noext + ".parquet"
        pd.DataFrame(rows).to_parquet(p)
        return p
    except Exception:  # noqa: BLE001 - pandas/pyarrow may be missing in the Isaac python
        p = path_noext + ".json"
        with open(p, "w") as f:
            json.dump(rows, f)
        return p


ROBOT_R = 0.5          # ANYmal-D body + leg swing as a disc, for the (evaluation-only) clearance metric


def _min_gap(x: float, y: float, prims: dict) -> float:
    """Truth clearance (m) from a robot disc to the nearest mover primitive. EVALUATION ONLY, never sent to ROS."""
    import numpy as np

    g = float("inf")
    c = prims.get("circles")
    if c is not None and len(c):
        g = min(g, float((np.hypot(c[:, 0] - x, c[:, 1] - y) - c[:, 2]).min()) - ROBOT_R)
    b = prims.get("boxes")
    if b is not None and len(b):
        cs, sn = np.cos(b[:, 4]), np.sin(b[:, 4])
        dx, dy = x - b[:, 0], y - b[:, 1]
        lx, ly = np.abs(dx * cs + dy * sn), np.abs(-dx * sn + dy * cs)
        q = np.hypot(np.maximum(lx - b[:, 2], 0), np.maximum(ly - b[:, 3], 0))
        g = min(g, float(q.min()) - ROBOT_R)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    cfg = json.load(open(ap.parse_args().cfg))
    sim = cfg.get("sim", "fake")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    wall_boot = time.time()

    if sim == "isaac":
        from intercycle_sim.isaac_backend import launch_app

        launch_app()
    import numpy as np

    from intercycle_sim import movers as M
    from intercycle_sim.protocol import SimClient
    from intercycle_sim.sensors import SensorSuite, make_static_caster, yaw_of
    from intercycle_sim.world import load_boxes, load_mesh

    wd = cfg["world_dir"]
    boxes = load_boxes(os.path.join(wd, "boxes.json"))
    V, F = load_mesh(os.path.join(wd, "mesh.npz"))
    src = M.load_source(cfg.get("mover_source"), cfg)
    parts = M.parts_for(src.spec())
    # optional R3 plant hooks (intercycle_eval.plant): attach(caster), sensor_prims(t), after_step(t, state, contacts),
    # frame_extras(t) -> extra frame keys for the bridge, on_ros(t, events), finish(out_dir, summary) -> dict
    hook = lambda name: getattr(src, name, None)  # noqa: E731

    if sim == "isaac":
        from intercycle_sim.isaac_backend import IsaacBackend

        be = IsaacBackend(cfg, wd, parts)
        caster = make_static_caster(boxes, V, F, prefer_warp=True, device="cuda:0")
    else:
        from intercycle_sim.fake_sim import FakeBackend

        be = FakeBackend(cfg, boxes)
        caster = make_static_caster(boxes, V, F, prefer_warp=bool(cfg.get("warp_cpu", True)), device="cpu")
    boot_s = time.time() - wall_boot
    caster_check = None
    if caster.name.startswith("warp") and boxes:
        # cross-check the Warp mesh cast against the analytic box cast from the start pose (same rays)
        from intercycle_sim.sensors import BoxCaster, LidarModel

        s0 = be.state()
        args = (float(s0["pos_w"][0]), float(s0["pos_w"][1]), float(s0["pos_w"][2]), yaw_of(s0["quat_wxyz"]), None)
        rw = LidarModel(caster).true_ranges(*args)
        rb = LidarModel(BoxCaster(boxes)).true_ranges(*args)
        both = (rw < 24.99) | (rb < 24.99)
        caster_check = {"rays": int(both.sum()), "max_abs_diff_m": float(np.abs(rw - rb)[both].max()) if both.any()
                        else 0.0, "median_range_m": float(np.median(rw[both])) if both.any() else None}
        print(f"[runner] warp vs box lidar check {caster_check}", flush=True)
    print(f"[runner] backend {be.name} ready in {boot_s:.1f}s, lidar static caster {caster.name}", flush=True)
    if hook("attach"):
        src.attach(caster=caster, dt=be.dt)

    # R-W13: plant-side safety scanner with a fixed ISO 13855 protective field (cfg "safety_scanner": true); it gates
    # the command the locomotion layer executes, from the true pose and the true people / vehicle primitives
    scanner, sc_parts, sc_idx = None, [], np.zeros(0, int)
    if cfg.get("safety_scanner", False):
        from intercycle_sim.safety_scanner import ProtectiveField, scanned_part_index

        scanner = ProtectiveField()
        sc_parts = [parts[k] for k in scanned_part_index(parts)]
        sc_idx = np.array(scanned_part_index(parts), int)
        print(f"[runner] safety scanner: fixed field S = {scanner.spec.s_m:.3f} m over {len(sc_parts)} mover parts",
              flush=True)

    suite = SensorSuite(caster, be.dt, seed=int(cfg.get("seed", 0)), start=tuple(cfg.get("start", (0, 0, 0))))
    mounts = {"laser": list(suite.lidar.mount), "imu": [0.0, 0.0, 0.0], "camera": list(suite.camera.mount)}
    client = SimClient(cfg["sock"], timeout_s=600.0)
    print("[runner] connected to the ROS bridge", flush=True)

    T = float(cfg.get("seconds", 30.0))
    max_rtf = float(cfg.get("max_rtf", 1.0))
    log_every = max(1, int(round(1.0 / (float(cfg.get("log_hz", 10.0)) * be.dt))))
    stop_file = cfg.get("stop_file")
    mover_every = max(1, int(cfg.get("mover_every", 5)))
    prof = {"set_movers": 0.0, "sensors": 0.0, "exchange": 0.0}
    rows, contacts = [], []
    cmd = (0.0, 0.0, 0.0)
    t, k = 0.0, 0
    st = be.state()
    p0 = np.array(st["pos_w"][:2])
    path_len, prev = 0.0, p0.copy()
    fell, end_reason = False, "time_limit"
    min_scan, min_gap, min_gap_t = float("inf"), float("inf"), None
    wall0 = time.time()
    step_wall = 0.0
    try:
        while t < T:
            poses = src.poses(t)
            pp = M.part_poses(parts, poses) if parts else np.zeros((0, 3))
            if parts and k % mover_every == 0:
                w_ = time.time()
                be.set_movers(pp)
                prof["set_movers"] += time.time() - w_            # PhysX kinematic targets at 50 / mover_every Hz (sensors use every step)
            prims = src.sensor_prims(t) if hook("sensor_prims") else (M.primitives(parts, pp) if parts else None)
            cmd_exec = cmd
            if scanner is not None:
                sc_prims = M.primitives(sc_parts, pp[sc_idx]) if len(sc_parts) else None
                cmd_exec = scanner.gate(t, be.dt, float(st["pos_w"][0]), float(st["pos_w"][1]), yaw_of(st["quat_wxyz"]),
                                        cmd, float(np.hypot(st["v_w"][0], st["v_w"][1])), sc_prims)
            w = time.time()
            st = be.step(cmd_exec)
            step_wall += time.time() - w
            t = (k + 1) * be.dt
            w_ = time.time()
            fr = suite.frame(t, st["pos_w"], st["quat_wxyz"], st["v_w"], st["v_b"], st["omega_b"], prims)
            prof["sensors"] += time.time() - w_
            if hasattr(be, "set_estimate"):                  # R-W6: stand mode holds against the robot's own odometry
                be.set_estimate(fr["odom"]["x"], fr["odom"]["y"], fr["odom"]["psi"])
            if k == 0:
                fr["mounts"] = mounts
            if "scan" in fr:
                fin = [v for v in fr["scan"]["ranges"] if v != float("inf")]
                if fin:
                    min_scan = min(min_scan, min(fin))
            if prims is not None:
                gap = _min_gap(st["pos_w"][0], st["pos_w"][1], prims)
                if gap < min_gap:
                    min_gap, min_gap_t = gap, t
            cs = be.contacts() if t >= 0.5 else []          # ignore the spawn settle (first 0.5 s)
            if hook("after_step"):
                src.after_step(t, st, cs)
                if hook("frame_extras"):
                    fr.update(src.frame_extras(t))
            if cs:
                fr["contacts"] = cs
                ev = {"t": round(t, 3), "contacts": cs, "x": st["pos_w"][0], "y": st["pos_w"][1],
                      "nearest_mover": None, "nearest_mover_gap_m": None}
                if parts:
                    gaps = {}
                    for prt, q in zip(parts, pp):
                        one = M.primitives([prt], q[None, :])
                        g = _min_gap(st["pos_w"][0], st["pos_w"][1], one)
                        gaps[prt.mover_id] = min(g, gaps.get(prt.mover_id, float("inf")))
                    mid = min(gaps, key=gaps.get)
                    ev["nearest_mover"], ev["nearest_mover_gap_m"] = mid, round(gaps[mid], 3)
                contacts.append(ev)
            w_ = time.time()
            reply = client.exchange(fr)
            prof["exchange"] += time.time() - w_
            cmd = (reply["vx"], reply["vy"], reply["wz"])
            if reply.get("events") and hook("on_ros"):
                src.on_ros(t, reply["events"])
            p = np.array(st["pos_w"][:2])
            path_len += float(np.linalg.norm(p - prev))
            prev = p
            if k % log_every == 0:
                od = fr["odom"]
                rows.append({"t": round(t, 3), "x": st["pos_w"][0], "y": st["pos_w"][1], "z": st["pos_w"][2],
                             "psi": yaw_of(st["quat_wxyz"]), "vx_b": st["v_b"][0], "vy_b": st["v_b"][1],
                             "wz": st["omega_b"][2], "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_wz": cmd[2],
                             "odom_x": od["x"], "odom_y": od["y"], "odom_psi": od["psi"],
                             "wall": round(time.time() - wall0, 3)})
            if st["pos_w"][2] < 0.30:
                fell, end_reason = True, "fall"
                break
            k += 1
            if stop_file and k % 25 == 0 and os.path.exists(stop_file):
                end_reason = "stop_file"
                break
            if max_rtf > 0:
                ahead = t / max_rtf - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
    except ConnectionError as e:
        end_reason = f"bridge_closed: {e}"
    wall = time.time() - wall0
    client.close()
    p1 = np.array(st["pos_w"][:2])
    summary = {
        "backend": be.name, "policy": getattr(be, "policy_info", {}), "lidar_caster": caster.name, "lidar_caster_check": caster_check,
        "boot_s": round(boot_s, 1), "sim_s": round(t, 2), "wall_s": round(wall, 2),
        "rtf": round(t / wall, 3) if wall > 0 else None,
        "rtf_physics_only": round(t / step_wall, 3) if step_wall > 0 else None,
        "steps": k, "dt": be.dt, "wall_profile_s": {kk: round(v, 1) for kk, v in prof.items()}, "max_rtf": max_rtf, "start": list(p0), "end": list(p1),
        "end_psi": yaw_of(st["quat_wxyz"]), "end_z": st["pos_w"][2],
        "displacement_m": round(float(np.linalg.norm(p1 - p0)), 3), "path_length_m": round(path_len, 3),
        "fell": fell, "end_reason": end_reason,
        "min_scan_m": None if min_scan == float("inf") else round(min_scan, 3),
        "min_mover_gap_m": None if min_gap == float("inf") else round(min_gap, 3), "min_mover_gap_t": min_gap_t, "n_contact_steps": len(contacts),
        "odom_end": rows[-1] if rows else None,
        "safety_scanner": scanner.summary() if scanner is not None else None,
    }
    summary["truth_log"] = _write_table(rows, os.path.join(out_dir, "truth_log"))
    if hook("finish"):
        try:
            summary["plant"] = src.finish(out_dir, summary)
        except Exception as e:  # noqa: BLE001 - never lose the sim summary
            summary["plant_error"] = repr(e)
    with open(os.path.join(out_dir, "contacts.json"), "w") as f:
        json.dump(contacts, f)
    with open(os.path.join(out_dir, "sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[runner] SUMMARY " + json.dumps(summary), flush=True)
    try:
        be.close()
    except Exception as e:  # noqa: BLE001
        print(f"[runner] close: {e!r}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
