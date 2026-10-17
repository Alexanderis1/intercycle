"""Simulator process: steps the backend in lockstep with the ROS bridge and logs the episode.

    python -u -m intercycle_sim.runner --cfg /tmp/ic/cfg.json          (Isaac python 3.11 for --sim isaac)

cfg keys: sim ("isaac" | "fake"), world_dir, sock, out_dir, seconds (sim-time limit), start [x, y, psi],
max_rtf (pace to <= this real-time factor so Nav2 gets real compute time; 0 = free-running), seed,
movers (scripted list) or mover_source ("pkg.mod:factory" for R3), stop_file (orchestrator ends the episode),
log_hz (true-pose log rate, default 10).

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

    suite = SensorSuite(caster, be.dt, seed=int(cfg.get("seed", 0)), start=tuple(cfg.get("start", (0, 0, 0))))
    mounts = {"laser": list(suite.lidar.mount), "imu": [0.0, 0.0, 0.0], "camera": list(suite.camera.mount)}
    client = SimClient(cfg["sock"], timeout_s=600.0)
    print("[runner] connected to the ROS bridge", flush=True)

    T = float(cfg.get("seconds", 30.0))
    max_rtf = float(cfg.get("max_rtf", 1.0))
    log_every = max(1, int(round(1.0 / (float(cfg.get("log_hz", 10.0)) * be.dt))))
    stop_file = cfg.get("stop_file")
    rows, contacts = [], []
    cmd = (0.0, 0.0, 0.0)
    t, k = 0.0, 0
    st = be.state()
    p0 = np.array(st["pos_w"][:2])
    path_len, prev = 0.0, p0.copy()
    fell, end_reason = False, "time_limit"
    wall0 = time.time()
    step_wall = 0.0
    try:
        while t < T:
            poses = src.poses(t)
            pp = M.part_poses(parts, poses) if parts else np.zeros((0, 3))
            if parts:
                be.set_movers(pp)
            prims = M.primitives(parts, pp) if parts else None
            w = time.time()
            st = be.step(cmd)
            step_wall += time.time() - w
            t = (k + 1) * be.dt
            fr = suite.frame(t, st["pos_w"], st["quat_wxyz"], st["v_w"], st["v_b"], st["omega_b"], prims)
            if k == 0:
                fr["mounts"] = mounts
            cs = be.contacts()
            if cs:
                fr["contacts"] = cs
                contacts.append({"t": round(t, 3), "contacts": cs, "x": st["pos_w"][0], "y": st["pos_w"][1]})
            reply = client.exchange(fr)
            cmd = (reply["vx"], reply["vy"], reply["wz"])
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
        "steps": k, "dt": be.dt, "max_rtf": max_rtf, "start": list(p0), "end": list(p1),
        "end_psi": yaw_of(st["quat_wxyz"]), "end_z": st["pos_w"][2],
        "displacement_m": round(float(np.linalg.norm(p1 - p0)), 3), "path_length_m": round(path_len, 3),
        "fell": fell, "end_reason": end_reason, "n_contact_steps": len(contacts),
        "odom_end": rows[-1] if rows else None,
    }
    summary["truth_log"] = _write_table(rows, os.path.join(out_dir, "truth_log"))
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
