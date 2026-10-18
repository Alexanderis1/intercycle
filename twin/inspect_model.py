"""The inspection model of twin/tasks.yaml beyond the 20x zoom capture (lane S2 TASKS; E-W3): the thermal capture and
the keyed defect draw. numpy + the standard library (Isaac-deps rule); the zoom capture itself is
twin/robot_camera.capture.

* `thermal_capture`: the radiometric LWIR core on the same PTZ head (tasks.yaml inspection.thermal_model):
  640 x 512 px, HFOV 24 deg, no zoom, f_th = 320 / tan(12 deg) = 1505.5 px. Pass iff the target is inside the
  640 x 512 frame at the PTZ aim (aim error N(0, AIM_SIGMA) per axis, the same rule as the zoom capture's in_image),
  unoccluded (static caster + movers, robot_camera.occluded), incidence < 60 deg, blur
  b_th = (|v_perp| / R + |omega|) * f_th * 10 ms < 1 px and px_per_m_th = f_th / R * cos(theta) >= 100 px/m.
  ASSUMED (no plant thermal data): the hot-spot contrast is always above NETD.
* `defect_draw(world_seed, day, checkpoint, mission)`: whether that checkpoint carries a defect on that mission,
  probability DEFECT_RATE = 0.25, from the keyed stream (world_seed, day, "defect:<checkpoint>", "mission:<n>")
  (twin/rng.keyed_rng): common random numbers across conditions (the same world / day / checkpoint / mission gives the
  same answer whatever the controller does).

Plant hook (lane S5a; plant.py is not edited here): after `rec.update(RC.capture(...))` in `_capture`, a thermal
checkpoint also needs `IM.thermal_capture(...)` to pass:
    if cp.thermal: rec["thermal"] = IM.thermal_capture(t, cam, xyz, n, v, wz, self.caster, prims, self.cap_rng);
                   rec["pass"] = bool(rec["pass"] and rec["thermal"]["pass"])
"""
from __future__ import annotations

import math

import numpy as np

from twin import robot_camera as RC
from twin.rng import keyed_rng

TH_W, TH_H = 640, 512
TH_HFOV = math.radians(24.0)
F_TH = (TH_W / 2) / math.tan(TH_HFOV / 2)          # 1505.5 px (square pixels: f_y = f_x)
TH_EXPOSURE_S = 0.010
THERMAL_PXM = 100.0
DEFECT_RATE = 0.25
THERMAL_SOURCE = "twin thermal v1 (640x512, HFOV 24 deg, >= 100 px/m, blur < 1 px at 10 ms; NETD ASSUMED met)"


def thermal_capture(t: float, cam: np.ndarray, target_xyz: np.ndarray, normal: np.ndarray, v_world: np.ndarray,
                    wz: float, static, prims: dict | None, rng: np.random.Generator) -> dict:
    """The thermal shot of a thermal checkpoint (the module docstring has the pass rule)."""
    target_xyz = np.asarray(target_xyz, float)
    yaw, pitch = RC.aim(cam, target_xyz)
    yaw_e, pitch_e = yaw + rng.normal(0.0, RC.AIM_SIGMA), pitch + rng.normal(0.0, RC.AIM_SIGMA)
    u, v, zf = RC.project(cam, yaw_e, pitch_e, target_xyz[None, :], F_TH, F_TH)
    inimg = bool((zf[0] > 0.05) and abs(u[0]) <= TH_W / 2 and abs(v[0]) <= TH_H / 2)
    occ = bool(RC.occluded(static, cam, target_xyz[None, :], prims)[0])
    los = target_xyz - cam
    rng_m = float(np.linalg.norm(los))
    ulos = los / max(rng_m, 1e-9)
    n = np.asarray(normal, float) / max(float(np.linalg.norm(normal)), 1e-9)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(-ulos @ n)))))
    px_per_m = F_TH / max(rng_m, 1e-6) * max(math.cos(math.radians(inc)), 0.0)
    vw = np.asarray(v_world, float)
    v_perp = vw - (vw @ ulos) * ulos
    blur = (float(np.linalg.norm(v_perp)) / max(rng_m, 1e-6) + abs(float(wz))) * F_TH * TH_EXPOSURE_S
    checks = {"in_image": inimg, "unoccluded": not occ, "incidence_ok": inc < RC.MAX_INCIDENCE_DEG,
              "pxm_ok": px_per_m >= THERMAL_PXM, "blur_ok": blur < 1.0}
    return {"t": round(t, 3), "range_m": round(rng_m, 3), "incidence_deg": round(inc, 2),
            "px_per_m": round(px_per_m, 1), "blur_px": round(blur, 3), "checks": checks,
            "pass": bool(all(checks.values())), "source": THERMAL_SOURCE}


def defect_draw(world_seed: int, day: int, checkpoint: str, mission: int, rate: float = DEFECT_RATE) -> bool:
    """Keyed (CRN) defect presence of one checkpoint on one mission."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"defect rate {rate} not in [0, 1]")
    return bool(keyed_rng(world_seed, day, f"defect:{checkpoint}", f"mission:{int(mission)}").random() < rate)
