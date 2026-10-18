"""R-W13: a plant-side safety-scanner model with ONE fixed protective field sized by ISO 13855 (S = K * T + C).

Simulator side (it reads the true robot pose and the true mover primitives, like the sensor models); nothing here reaches
ROS. It stands in for the certified safety laser scanner a mobile robot in a plant carries: when a person or a vehicle is
inside the protective field while the robot moves (or is commanded to), the robot is brought to a protective stop (the
command to the locomotion layer is forced to zero) and released automatically once the field has been clear for
`release_s`. Protective stops and stopped time are counted as a metric (sim_summary["safety_scanner"]).

Fixed field (PLAN_TWIN §6: R-W13 reduced to a fixed field; a speed-dependent field is cut), sized for the worst case:
  K = K_person + v_robot_max = 1.6 m/s (ISO 13855, approach speed of the whole body for S > 0.5 m)
      + 0.762 m/s (the robot's own top speed |(0.7, 0.3)| m/s from nav2.yaml: a fixed field must also cover the
      robot driving into a standing person; ASSUMED composition, as in ISO 3691-4 practice for mobile platforms)
  T = t_scanner 0.08 s (ASSUMED: safety laser scanner response, 2 scans at 40 ms)
      + t_control 0.02 s (ASSUMED: safety controller -> locomotion, one 50 Hz step)
      + t_stop 0.50 s (ASSUMED: stopping time from 0.762 m/s at nav2.yaml's max_decel 1.5 m/s^2)
  C = 1200 mm - 0.4 * H (ISO 13855 horizontal approach of an ESPE field) with the detection plane at H = 300 mm
      (ASSUMED mount; ISO 13855 needs H >= 15 * (d - 50) mm, true for a 70 mm resolution) -> C = 1.08 m (>= 0.85 m)
  S = 2.362 * 0.60 + 1.08 = 2.497 m, applied on every side of the robot footprint (+-0.50 x +-0.32 m, nav2.yaml), since
      the base is omnidirectional and backs out of zones.
Only movers at the detection plane are seen (people, vehicles: primitives whose top is above H). Static structure is not
in the field (the field set is assumed to be configured around the fixed plant, as real scanner field sets are), and
machine-part colliders (the "cyl:" pool) are not people: they are not scanned.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

K_PERSON = 1.6                         # m/s, ISO 13855 walking approach (whole body), for S > 0.5 m
V_ROBOT_MAX = math.hypot(0.7, 0.3)     # m/s, nav2.yaml vx_max / vy_max
T_SCANNER = 0.08                       # s, ASSUMED
T_CONTROL = 0.02                       # s, ASSUMED
T_STOP = 0.50                          # s, ASSUMED (0.762 m/s / 1.5 m/s^2 = 0.51 s)
H_PLANE_MM = 300.0                     # detection plane height, ASSUMED
FOOT_HALF = (0.50, 0.32)               # robot footprint half-length / half-width (nav2.yaml)
RELEASE_S = 1.0                        # ASSUMED: automatic restart after the field has been clear this long
SCANNED_CLASSES = ("person", "agv", "tugger", "forklift", "tugger_train", "train", "vehicle")


def iso13855_c_mm(h_mm: float) -> float:
    """ISO 13855 intrusion distance for a horizontal ESPE field at height H: C = 1200 - 0.4 H, never below 850 mm."""
    return max(850.0, 1200.0 - 0.4 * float(h_mm))


def iso13855_s(k_mps: float, t_s: float, c_m: float) -> float:
    """Minimum distance S = K * T + C (metres)."""
    return float(k_mps) * float(t_s) + float(c_m)


@dataclass(frozen=True)
class FieldSpec:
    k_mps: float = K_PERSON + V_ROBOT_MAX
    t_s: float = T_SCANNER + T_CONTROL + T_STOP
    c_m: float = iso13855_c_mm(H_PLANE_MM) / 1000.0
    h_plane_m: float = H_PLANE_MM / 1000.0
    half_len: float = FOOT_HALF[0]
    half_wid: float = FOOT_HALF[1]
    release_s: float = RELEASE_S

    @property
    def s_m(self) -> float:
        return iso13855_s(self.k_mps, self.t_s, self.c_m)

    def record(self) -> dict:
        return {**asdict(self), "s_m": round(self.s_m, 4),
                "formula": "S = K*T + C (ISO 13855); K = 1.6 + v_robot_max, T = t_scan + t_ctrl + t_stop, C = 1200 - 0.4 H",
                "sources": {"K_person": "ISO 13855", "C": "ISO 13855 (H assumed)", "v_robot_max": "nav2.yaml",
                            "t_scan": "ASSUMED", "t_ctrl": "ASSUMED", "t_stop": "ASSUMED", "release_s": "ASSUMED"}}


def _rect_corners(cx, cy, hl, hw, yaw) -> np.ndarray:
    """(..., 4, 2) corners of oriented rectangles."""
    cx, cy, hl, hw, yaw = (np.asarray(v, float) for v in (cx, cy, hl, hw, yaw))
    loc = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    lx, ly = loc[:, 0] * hl[..., None], loc[:, 1] * hw[..., None]
    return np.stack((cx[..., None] + lx * c - ly * s, cy[..., None] + lx * s + ly * c), axis=-1)


def _boxes_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Separating-axis test: a (4, 2) one rectangle, b (N, 4, 2) rectangles -> (N,) overlap."""
    if len(b) == 0:
        return np.zeros(0, bool)
    ok = np.ones(len(b), bool)
    for poly in (a[None, :, :].repeat(len(b), 0), b):
        for k in range(2):                                   # two edge normals per rectangle
            e = poly[:, (k + 1) % 4] - poly[:, k]
            n = np.stack((-e[:, 1], e[:, 0]), axis=1)        # (N, 2)
            pa = np.einsum("nj,ij->ni", n, a)                # (N, 4)
            pb = np.einsum("nj,nij->ni", n, b)
            ok &= (pa.max(1) >= pb.min(1)) & (pb.max(1) >= pa.min(1))
    return ok


class ProtectiveField:
    """The fixed protective field around the robot and its protective-stop state machine."""

    def __init__(self, spec: FieldSpec | None = None):
        self.spec = spec or FieldSpec()
        self.active = False
        self.clear_since: float | None = None
        self.n_stops = 0
        self.stopped_s = 0.0
        self.events: list[dict] = []

    def intruders(self, x: float, y: float, psi: float, prims: dict | None) -> list[tuple[str, float]]:
        """(kind, index) of every scanned primitive inside the field."""
        if not prims:
            return []
        sp = self.spec
        hl, hw = sp.half_len + sp.s_m, sp.half_wid + sp.s_m
        out = []
        c = prims.get("circles")
        if c is not None and len(c):
            c = np.asarray(c, float)
            vis = c[:, 3] >= sp.h_plane_m
            cs, sn = math.cos(psi), math.sin(psi)
            dx, dy = c[:, 0] - x, c[:, 1] - y
            lx, ly = np.abs(dx * cs + dy * sn), np.abs(-dx * sn + dy * cs)
            gap = np.hypot(np.maximum(lx - hl, 0.0), np.maximum(ly - hw, 0.0)) - c[:, 2]
            out += [("circle", int(i)) for i in np.flatnonzero(vis & (gap < 0.0))]
        b = prims.get("boxes")
        if b is not None and len(b):
            b = np.asarray(b, float)
            vis = b[:, 5] >= sp.h_plane_m
            fa = _rect_corners(x, y, hl, hw, psi)
            fb = _rect_corners(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])
            out += [("box", int(i)) for i in np.flatnonzero(vis & _boxes_overlap(fa, fb))]
        return out

    def gate(self, t: float, dt: float, x: float, y: float, psi: float, cmd: tuple[float, float, float],
             speed: float, prims: dict | None) -> tuple[float, float, float]:
        """The command the locomotion layer executes this step: `cmd`, or zero during a protective stop."""
        intr = self.intruders(x, y, psi, prims)
        moving = max(abs(float(v)) for v in cmd) > 1e-3 or float(speed) > 0.05
        if intr and moving and not self.active:
            self.active = True
            self.n_stops += 1
            if len(self.events) < 500:
                self.events.append({"t": round(t, 3), "x": round(x, 3), "y": round(y, 3), "n_intruders": len(intr),
                                    "kinds": sorted({k for k, _ in intr})})
        if self.active:
            if intr:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = t
            if self.clear_since is not None and t - self.clear_since >= self.spec.release_s - 1e-9:
                self.active, self.clear_since = False, None
        if self.active:
            self.stopped_s += dt
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in cmd)  # type: ignore[return-value]

    def summary(self) -> dict:
        return {"field": self.spec.record(), "n_protective_stops": self.n_stops,
                "stopped_s": round(self.stopped_s, 3), "events": self.events}


def scanned_part_index(parts) -> list[int]:
    """Indices of the mover parts the scanner sees (people and vehicles; never the machine-part collider pool)."""
    return [k for k, p in enumerate(parts) if str(p.cls) in SCANNED_CLASSES]
