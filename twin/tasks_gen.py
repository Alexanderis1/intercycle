"""Generate twin/tasks.yaml from twin/site.yaml (lane S2 TASKS; R-W10, E-W3).

`python -m twin.tasks_gen [--check]` writes (or, with --check, compares against) twin/tasks.yaml. Deterministic: the
same site.yaml gives the same file.

Targets (TARGETS below) are anchored on site.yaml press footprints: a vertex or edge midpoint of the traced footprint
polygon facing the visible floor, at an ASSUMED height (ram mid-stroke 2.15 m = plant RAM_TDC_Z - RAM_STROKE_M / 2;
die / bolster 1.0 m; control panel 1.4 m; clutch-brake housing 3.2 m). The face normal is the outward edge normal (or
+x where the press's operator side is not observed: ASSUMED toward the aisle the zone camera sees).

Candidate poses (R-W10, source "generated"): the target normal rotated by a fan of angles, at stand-offs on a 5 cm
grid; a pose is kept if it passes the same clearance checks as twin/compile.py (visible floor, ROBOT_R from unseen
floor and footprints, outside every dilated danger rect for an outside checkpoint; inside its press's danger rect, with
the generated wait pose clear, for an inside_danger checkpoint), the 2-D line of sight to the target crosses no other
footprint, incidence < 60 deg and px/m at the checkpoint's zoom >= 1.5 x DORI_INSPECT. Up to 3 poses per checkpoint
(smallest fan angle first, then smallest stand-off). Commissioning confirms them (source "confirmed").

Checkpoint provenance: the train clip ids of the footprint the target is anchored on (site.yaml), filtered to train
clips of the active split; an ASSUMED footprint (P9b) gives source "assumed" and no clips.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from twin.compile import (ROBOT_R, TWIN, WAIT_OFFSET_M, _EPS, _aabb, _in_rect, _seg_rect_dist, _seg_seg_dist,
                          danger_rect)
from twin.schemas.prereg import load_active_split
from twin.robot_camera import DORI_INSPECT, FX, MAX_INCIDENCE_DEG
from twin.schemas import load
from twin.schemas.site import Press, Site, point_in_polygon
from twin.schemas.tasks import Tasks

H_RAM, H_DIE, H_PANEL, H_CLUTCH = 2.15, 1.0, 1.4, 3.2      # ASSUMED target heights (m)
FAN_DEG = (0, -10, 10, -20, 20, -30, 30, -40, 40, -50, 50)
STANDOFFS = [round(0.6 + 0.05 * k, 2) for k in range(229)]   # 0.6 .. 12.0 m
PXM_MARGIN = 1.5
N_CAND = 3
DWELL_S = 4.0

# id, zone, press, target_kind, anchor (("vertex", i) | ("edge", i) | ("xmax",)), height, normal ("edge" | (nx, nz)),
# inside_danger, zoom, thermal
TARGETS = [
    ("cp_P9a_ram", "z9", "P9a", "press", ("edge", 1), H_RAM, "edge", False, 20.0, False),
    ("cp_P9a_die", "z9", "P9a", "press", ("edge", 1), H_DIE, "edge", True, 4.0, False),
    ("cp_P9a_clutch", "z9", "P9a", "press", ("edge", 1), H_CLUTCH, "edge", False, 12.0, True),
    ("cp_P9a_panel", "z9", "P9a", "panel", ("vertex", 2), H_PANEL, "edge", False, 8.0, False),
    ("cp_P9b_ram", "z9", "P9b", "press", ("xmin",), H_RAM, (-1.0, 0.0), False, 20.0, True),
    ("cp_P14d_die", "z14", "P14d", "press", ("xmax",), H_DIE, (1.0, 0.0), True, 4.0, False),
    ("cp_P14c_ram", "z14", "P14c", "press", ("xmax",), H_RAM, (1.0, 0.0), False, 20.0, True),
    ("cp_P14b_ram", "z14", "P14b", "press", ("xmax",), H_RAM, (1.0, 0.0), False, 20.0, True),
]
DOCKS = {"z9": (0.6, -2.0), "z14": (23.0, 1.1)}
MISSION_ORDER = {"z9": ["cp_P9b_ram", "cp_P9a_panel", "cp_P9a_die", "cp_P9a_ram", "cp_P9a_clutch"],
                 "z14": ["cp_P14b_ram", "cp_P14c_ram", "cp_P14d_die"]}

DETECTABILITY = (
    "A capture passes Inspect iff ALL of: (1) in_image: the target point is inside the zoomed frame after the PTZ aim "
    "(aim error N(0, 2.5 deg) per axis, re-centred by the PTZ before the shot); (2) unoccluded: the 3-D line of sight "
    "from the camera to the target is not cut by the static mesh or a mover; (3) incidence theta < 60 deg (theta = angle "
    "between the reversed line of sight and the target normal); (4) blur b < 1 px with b = (|v_perp| / R + |omega|) * "
    "f0 * Z * t_exp, t_exp = 4 ms; (5) DORI Inspect: px_per_m = f0 * Z / R * cos(theta) >= 1000 px/m (1 px per mm, "
    "Amendment 1), with f0 = 1371.0 px (1920 px, 70 deg HFOV at Z = 1), Z = the checkpoint's optical zoom (1..20), R = "
    "camera-target range (m). A thermal checkpoint also needs the thermal capture to pass (thermal_model). Defects: each "
    "(checkpoint, mission) carries a defect with probability defect_rate = 0.25 (keyed draw world/day/checkpoint, CRN "
    "across conditions); a capture that passes Inspect resolves it (present or absent, no false call: 1 mm class "
    "defects need >= 1 px/mm); a capture that fails, a declined or an unattempted checkpoint resolves nothing and scores "
    "0. No partial credit.")
CAPTURE_MODEL = (
    "Same 20x optical zoom PTZ head in every condition (E-W3): 1920 x 1080 px, HFOV 70 deg / Z (Z = 1..20, f = f0 * Z, "
    "f0 = 1371.0 px), mounted at (0.45, 0, 0.35) m on the body; the shot is taken after the body has been still for "
    "1.0 s (|v| < 0.05 m/s, |omega| < 0.05 rad/s), else at dwell_end (settled = false). A boundary capture fails only "
    "on line of sight, incidence or blur, never on range at Z = 20 inside the zone (R <= 13.7 m keeps 2000 px/m at "
    "theta = 0).")
THERMAL_MODEL = (
    "Radiometric LWIR core on the same head, 640 x 512 px, HFOV 24 deg, no zoom: f_th = 320 / tan(12 deg) = 1505.5 px. "
    "Thermal pass iff in_image (frame 640 x 512 at the same aim), unoccluded, theta < 60 deg, b_th = (|v_perp| / R + "
    "|omega|) * f_th * 10 ms < 1 px, and px_per_m_th = f_th / R * cos(theta) >= 100 px/m (a 10 cm hot spot spans >= 10 "
    "px). ASSUMED (no plant thermal data): the hot-spot contrast is always above NETD.")
WINDOW_DEF = (
    "An inside_danger checkpoint may be entered only inside a press inspection window: (a) a press stop (the press is "
    "idle, not running a stroke, for the whole stay), or (b) a gap between strokes of length >= inside_s + margin_s, "
    "where inside_s = entry_m / v + dwell_s + entry_m / v (entry from the wait pose, dwell, exit; entry_m from "
    "compile.py, v = robot speed) and margin_s = 1.0 s (= compile.SAFETY_MARGIN_S). The window must be open from the "
    "moment the robot crosses into the dilated danger rect until it has left it; a stroke starting with the robot "
    "inside is a robot-caused hold or trip.")


def _edge_normal(poly: list, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint and outward unit normal of edge i (poly[i] -> poly[i+1])."""
    a, b = np.array(poly[i], float), np.array(poly[(i + 1) % len(poly)], float)
    d = b - a
    n = np.array((d[1], -d[0]))
    n /= np.linalg.norm(n)
    c = np.mean(np.array(poly, float), axis=0)
    mid = (a + b) / 2
    if n @ (mid - c) < 0:
        n = -n
    return mid, n


def _anchor(p: Press, anchor: tuple, normal) -> tuple[np.ndarray, np.ndarray]:
    poly = p.footprint.value
    arr = np.array(poly, float)
    kind = anchor[0]
    if kind == "edge":
        pt, n_edge = _edge_normal(poly, anchor[1])
    elif kind == "vertex":
        pt = arr[anchor[1]]
        _, n1 = _edge_normal(poly, anchor[1] - 1)
        _, n2 = _edge_normal(poly, anchor[1])
        n_edge = (n1 + n2) / np.linalg.norm(n1 + n2)
    elif kind == "xmax":
        pt, n_edge = arr[int(np.argmax(arr[:, 0]))], np.array((1.0, 0.0))
    elif kind == "xmin":
        pt, n_edge = arr[int(np.argmin(arr[:, 0]))], np.array((-1.0, 0.0))
    else:
        raise ValueError(anchor)
    n = n_edge if normal == "edge" else np.array(normal, float) / math.hypot(*normal)
    return pt, n


class _Clear:
    """The clearance checks of compile.TwinSiteGraph._clear_pt / _clear_seg for one zone."""

    def __init__(self, floor: list, feet: list, danger: list):
        self.poly = list(floor)
        self.floor = [np.array(q, float) for q in floor]
        self.feet, self.danger = feet, danger

    def pt(self, x: float, z: float, hazards: bool = True) -> bool:
        p = np.array((x, z))
        if not point_in_polygon(x, z, self.poly):
            return False
        nf = len(self.floor)
        if min(_seg_seg_dist(p, p, self.floor[k], self.floor[(k + 1) % nf]) for k in range(nf)) < ROBOT_R - _EPS:
            return False
        if any(_seg_rect_dist(p, p, r) < ROBOT_R - _EPS for r in self.feet):
            return False
        return not (hazards and any(_seg_rect_dist(p, p, r) < ROBOT_R - _EPS for r in self.danger))


def _rot(n: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array((n[0] * math.cos(a) - n[1] * math.sin(a), n[0] * math.sin(a) + n[1] * math.cos(a)))


def _los_clear(pose: np.ndarray, target: np.ndarray, feet: list, own: tuple) -> bool:
    """2-D line of sight (up to 0.15 m before the target) crosses no footprint other than the target's own (the target
    sits on its own footprint's bounding box; the 3-D check against the press body is the plant's capture model)."""
    d = target - pose
    L = float(np.linalg.norm(d))
    end = pose + d * max(L - 0.15, 0.0) / max(L, 1e-9)
    return all(_seg_rect_dist(pose, end, r) > 0.0 for r in feet if r != own)


def candidates(tgt: np.ndarray, h: float, n: np.ndarray, inside: bool, zoom: float, own_danger: tuple,
               own_foot: tuple, clear: _Clear) -> list[dict]:
    out: list[dict] = []
    used_angles: set = set()
    for a in FAN_DEG:
        nr = _rot(n, a)
        for s in STANDOFFS:
            pose = tgt + nr * s
            x, z = float(pose[0]), float(pose[1])
            if inside:
                if not _in_rect(x, z, own_danger) or not clear.pt(x, z, hazards=False):
                    continue
                r = own_danger
                ux, uz = nr
                t_exit = min(((r[2] - x) / ux if ux > 0 else (r[0] - x) / ux) if abs(ux) > _EPS else math.inf,
                             ((r[3] - z) / uz if uz > 0 else (r[1] - z) / uz) if abs(uz) > _EPS else math.inf)
                w = pose + nr * (t_exit + WAIT_OFFSET_M)
                if not clear.pt(float(w[0]), float(w[1])):
                    continue
                if any(_seg_rect_dist(pose, w, f) < ROBOT_R - _EPS for f in clear.feet):
                    continue
            elif not clear.pt(x, z):
                continue
            if not _los_clear(pose, tgt, clear.feet, own_foot):
                continue
            rng = math.hypot(s, h - 0.35 - 0.0)      # camera height ~ body + mount; ASSUMED 0.35 m above base frame
            if abs(a) >= MAX_INCIDENCE_DEG or FX * zoom / rng * math.cos(math.radians(a)) < PXM_MARGIN * DORI_INSPECT:
                continue
            yaw = math.atan2(float(tgt[0] - x), float(tgt[1] - z))
            out.append({"pose": {"x": round(x, 3), "z": round(z, 3), "yaw": round(yaw, 4)}, "source": "generated",
                        "standoff_m": round(s, 2), "_normal": nr})
            used_angles.add(a)
            break
        if len(out) >= N_CAND:
            break
    return out


def generate(site: Site) -> dict:
    split = load_active_split()
    presses = {p.id: p for p in site.presses}
    zones = {z.id: z for z in site.zones}
    cps = []
    for cid, zid, pid, kind, anchor, h, normal, inside, zoom, thermal in TARGETS:
        p = presses[pid]
        zp = [q for q in site.presses if q.zone == zid]
        clear = _Clear(zones[zid].floor_polygon.value, [_aabb(q.footprint.value) for q in zp],
                       [danger_rect(q) for q in zp])
        tgt, n = _anchor(p, anchor, normal)
        cand = candidates(tgt, h, n, inside, zoom, danger_rect(p), _aabb(p.footprint.value), clear)
        if not cand:
            raise SystemExit(f"{cid}: no valid candidate pose")
        nr = cand[0].pop("_normal")
        for c in cand[1:]:
            c.pop("_normal")
        clips = sorted(c for c in p.footprint.clips if split.get(c) is not None and split[c].split == "train")
        src = p.footprint.source if clips else "assumed"
        cps.append({"id": cid, "zone": zid, "press": pid, "target_kind": kind,
                    "target_xzh": [round(float(tgt[0]), 3), round(float(tgt[1]), 3), h],
                    "normal_xz": [round(float(nr[0]), 4), round(float(nr[1]), 4)],
                    "candidate_poses": cand, "inside_danger": inside, "dwell_s": DWELL_S, "zoom": zoom,
                    "thermal": thermal, "source": src, "clips": clips})
    return {"contract": "twin.tasks/1", "site_id": site.site_id,
            "inspection": {"defect_rate": 0.25, "zoom_max": 20.0, "detectability": DETECTABILITY,
                           "capture_model": CAPTURE_MODEL, "thermal_model": THERMAL_MODEL,
                           "window": {"kinds": ["press_stop", "stroke_gap"], "margin_s": 1.0,
                                      "definition": WINDOW_DEF}},
            "zones": [{"zone": z, "dock": {"x": DOCKS[z][0], "z": DOCKS[z][1], "yaw": 0.0},
                       "mission_order": MISSION_ORDER[z]} for z in ("z9", "z14")],
            "checkpoints": cps}


HEADER = ("# twin/tasks.yaml - GENERATED by `python -m twin.tasks_gen` from twin/site.yaml (lane S2 TASKS; R-W10, E-W3).\n"
          "# Do not edit by hand: change twin/tasks_gen.py and regenerate. Contract twin.tasks/1 (twin/schemas/tasks.py).\n"
          "# Target heights, operator-side normals (+x / -x) and the panel position are ASSUMED (see tasks_gen.py).\n")


def render(site: Site) -> str:
    d = generate(site)
    Tasks.model_validate(d)
    return HEADER + yaml.safe_dump(d, sort_keys=False, width=120, allow_unicode=False, default_flow_style=None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(TWIN / "site.yaml"))
    ap.add_argument("--out", default=str(TWIN / "tasks.yaml"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(load(Site, a.site))
    out = Path(a.out)
    if a.check:
        same = out.exists() and out.read_text(encoding="utf-8").replace("\r\n", "\n") == text
        print("tasks.yaml is current" if same else "tasks.yaml is STALE")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
