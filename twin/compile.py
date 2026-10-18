"""Compile the site contracts into what the simulators run on (PLAN_TWIN §2 `twin/compile.py`, §4 S1a; R-W10).

Inputs: `twin/site.yaml` (geometry), `twin/dynamics.json` (press timing priors / fits) and `twin/tasks.yaml`
(checkpoints; the G0 stub `twin/tasks_v0.yaml` is used while tasks.yaml does not exist). Before anything is built,
every source clip id in them must be a TRAIN id of the active split (`twin.provenance.assert_train_only`, which
fails closed: a dev / test / unknown id, or a missing split, raises).

Output, per zone (`CompiledZone`):
* `scenario`: an `intercycle.scenario.Scenario` for the zone. Every press is a machine with `on_line=False` and
  `kind="press"`; its hazard rect (`Machine.rect`) is the axis-aligned danger rect. There is no line, no shift or break
  calendar (the clips carry no timing signal, P2: no time-of-day structure is generated here) and no body-shop
  hall annex, traffic or BLE layout.
* `rects` + `extent`: the static bodies for `intercycle.sim.geometry.StaticMap(None, rects=, extent=)` and
  `world.build_world_files(scenario_path=, rects=, extent=)` (twin/build_worlds.py): one axis-aligned box per press
  footprint (cls "press") and the unseen floor as wall rects (cls "wall"). The extent is the zone's visible-floor
  bounding box snapped out to the 0.1 m grid plus a 2-cell margin, so StaticMap's closing ring lies on unseen floor.
  A cell whose square is not entirely inside the visible floor polygon is wall (fail closed: unseen floor is never
  free, at the cost of at most one cell of visible floor along a slanted edge).
* `danger_rects`: per press, the danger polygon's bounding box grown by `dilation_m` (the exact bounding box of the
  polygon dilated by a disc of that radius). It is a hazard zone, not a static body: it is not in `rects`.
* `graph`: a `TwinSiteGraph` built from tasks.yaml (R-W10) with the `intercycle.robot.sitegraph.SiteGraph`
  interface (`scn`, `pos`, `adj`, `targets`, `inspect_pose`, `mission`). A press checkpoint whose capture pose is
  inside the danger rect becomes a "machine" target: its wait pose is generated, not hand-made, by stepping from the
  capture pose along the target normal out of the danger rect plus WAIT_OFFSET_M. Other checkpoints are "point"
  targets at their capture pose. Transit edges form a visibility graph (dock, wait / point poses, corners of the
  inflated footprints and danger rects) whose segments keep ROBOT_R clear of every footprint, danger rect and the
  visible-floor boundary; every target must be reachable from the dock (CompileError otherwise).
* `cameras`: per framing, the camera placement (`Scenario.cameras` holds its nadir) and the focal length it used:
  K's f is a prior (P6: 1020-1430 px), so `camera_fit` estimates f from H with the principal point of K and fails
  closed outside that range (CompileError).

This module imports only numpy, pydantic, yaml and repo code without scipy, so the Isaac side may import it; the
world-file build (scipy via StaticMap) lives in twin/build_worlds.py.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from intercycle.robot.sitegraph import Target
from intercycle.scenario import Ble, Inspection, Line, Machine, Scenario, Zone
from twin.provenance import ProvenanceError, assert_train_only, source_clips
from twin.schemas import load
from twin.schemas.common import Point
from twin.schemas.dynamics import Dynamics
from twin.schemas.prereg import SplitRow
from twin.schemas.safety import Safety
from twin.schemas.site import Framing, Press, Site, point_in_polygon
from twin.schemas.site import Zone as SiteZone
from twin.schemas.tasks import Checkpoint, Tasks

REPO = Path(__file__).resolve().parents[1]
TWIN = REPO / "twin"
log = logging.getLogger(__name__)

GRID_RES = 0.1                  # must equal intercycle.sim.geometry.RES (tests/test_twin_compile.py pins it)
EXTENT_MARGIN_CELLS = 2         # StaticMap's closing ring is 2 cells: keep it on unseen floor
ROBOT_R = 0.55                  # clearance radius of the robot body (1.08 x 0.84 m envelope, half the length)
WAIT_OFFSET_M = 1.05            # wait pose: robot centre this far outside the danger rect (timing.TimingCore default)
CORNER_PAD_M = 0.05             # visibility-graph corners sit ROBOT_R + this outside each inflated obstacle
KEY_ID = 1                      # broadcast key id of the twin scenarios (the keyring is explicit, C-W5)
# Robot constants, the same as the body-shop scenario (they describe the robot, not the site).
ROBOT_SPEED_MPS = 1.0
SAFETY_MARGIN_S = 1.0
LOCAL_CHECK_LATENCY_S = 0.3
_EPS = 1e-9
_POSE_RANK = {"confirmed": 0, "generated": 1, "prior": 2}


class CompileError(ValueError):
    """The site / tasks / dynamics contracts cannot be compiled (inconsistent ids, unreachable or blocked poses)."""


# ------------------------------------------------------------------------------------------------ inputs


INPUT_FILES = {"site": ("site.yaml", "site_v0.yaml"), "dynamics": ("dynamics.json", "dynamics_v0.json"),
               "tasks": ("tasks.yaml", "tasks_v0.yaml")}       # contract -> (file in force, its G0 v0 stub)


def is_stub(kind: str, path: str | Path) -> bool:
    """True if `path` is the G0 v0 stub (prior placeholder) of contract `kind`."""
    return Path(path).name == INPUT_FILES[kind][1]


def default_inputs(twin_dir: Path = TWIN, *, allow_stubs: bool = False) -> dict[str, Path]:
    """The contract files in force: site.yaml / dynamics.json / tasks.yaml.

    A missing one falls back to its G0 v0 stub only when `allow_stubs` is set (pre-G2 builds, the slice on stubs,
    tests); otherwise it raises CompileError, so a build or run after G2 never runs on placeholder geometry or priors
    unnoticed (e.g. twin/ not mounted). Every choice is logged; a stub choice at WARNING."""
    out: dict[str, Path] = {}
    for kind, (name, stub) in INPUT_FILES.items():
        p = twin_dir / name
        if p.exists():
            log.info("twin input %s: %s", kind, p)
            out[kind] = p
        elif allow_stubs:
            log.warning("twin input %s: %s does not exist; using the G0 v0 STUB %s (allow_stubs)", kind, p,
                        twin_dir / stub)
            out[kind] = twin_dir / stub
        else:
            raise CompileError(f"twin input {kind}: {p} does not exist and stubs are not allowed "
                               f"(pass allow_stubs / --allow-stubs to build on the G0 v0 stub {stub})")
    return out


# ------------------------------------------------------------------------------------------------ geometry helpers

AABB = tuple[float, float, float, float]          # (x0, z0, x1, z1)


def _aabb(poly: list[Point]) -> AABB:
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(zs), max(xs), max(zs)


def _cells(v: float) -> float:
    return round(v / GRID_RES, 6)


def zone_extent(zone: SiteZone) -> AABB:
    """The zone's visible-floor bounding box snapped out to the grid, plus EXTENT_MARGIN_CELLS cells on every side."""
    x0, z0, x1, z1 = _aabb(zone.floor_polygon.value)
    m = EXTENT_MARGIN_CELLS
    i0, j0 = math.floor(_cells(x0)) - m, math.floor(_cells(z0)) - m
    i1, j1 = math.ceil(_cells(x1)) + m, math.ceil(_cells(z1)) + m
    return round(i0 * GRID_RES, 6), round(j0 * GRID_RES, 6), round(i1 * GRID_RES, 6), round(j1 * GRID_RES, 6)


def _rect(id_: str, cls: str, x0: float, z0: float, x1: float, z1: float) -> dict:
    return {"id": id_, "cls": cls, "cx": round((x0 + x1) / 2, 6), "cz": round((z0 + z1) / 2, 6),
            "w": round(x1 - x0, 6), "d": round(z1 - z0, 6), "yaw": 0.0}


def unseen_cells(poly: list[Point], extent: AABB) -> np.ndarray:
    """[nx, nz] bool over the extent's grid: True where the cell's square is not entirely inside the polygon (its
    centre is outside, or a polygon edge crosses its open interior)."""
    x0, z0, x1, z1 = extent
    nx, nz = int(round((x1 - x0) / GRID_RES)), int(round((z1 - z0) / GRID_RES))
    cx = x0 + (np.arange(nx) + 0.5) * GRID_RES
    cz = z0 + (np.arange(nz) + 0.5) * GRID_RES
    X, Z = np.meshgrid(cx, cz, indexing="ij")
    inside = np.zeros((nx, nz), dtype=bool)
    n = len(poly)
    h = GRID_RES / 2
    crossed = np.zeros((nx, nz), dtype=bool)
    for k in range(n):
        (ax, az), (bx, bz) = poly[k], poly[(k + 1) % n]
        straddle = (az > Z) != (bz > Z)                            # even-odd ray cast towards +x
        if az != bz:
            inside ^= straddle & (X < ax + (Z - az) * (bx - ax) / (bz - az))
        # separating-axis test of the segment against each cell's open square: box axes, then the segment normal
        lo_x, hi_x, lo_z, hi_z = min(ax, bx), max(ax, bx), min(az, bz), max(az, bz)
        ex, ez = bx - ax, bz - az
        L = math.hypot(ex, ez)
        if L < _EPS:
            continue
        nxv, nzv = -ez / L, ex / L
        sep_n = np.abs((X - ax) * nxv + (Z - az) * nzv) >= h * (abs(nxv) + abs(nzv)) - _EPS
        sep_x = (X + h <= lo_x + _EPS) | (X - h >= hi_x - _EPS)
        sep_z = (Z + h <= lo_z + _EPS) | (Z - h >= hi_z - _EPS)
        crossed |= ~(sep_n | sep_x | sep_z)
    return ~inside | crossed


def unseen_floor_rects(poly: list[Point], extent: AABB, prefix: str) -> list[dict]:
    """The unseen cells merged into axis-aligned wall rects (runs along x, merged across consecutive z rows)."""
    grid = unseen_cells(poly, extent)
    x0, z0 = extent[0], extent[1]
    nx, nz = grid.shape
    out: list[dict] = []
    open_: dict[tuple[int, int], int] = {}                 # (i0, i1) -> first row j of the growing rect

    def close(i0: int, i1: int, j0: int, j1: int) -> None:
        out.append(_rect(f"{prefix}_{len(out)}", "wall", round(x0 + i0 * GRID_RES, 6), round(z0 + j0 * GRID_RES, 6),
                         round(x0 + i1 * GRID_RES, 6), round(z0 + j1 * GRID_RES, 6)))

    for j in range(nz + 1):
        runs: set[tuple[int, int]] = set()
        if j < nz:
            col = grid[:, j]
            i = 0
            while i < nx:
                if col[i]:
                    k = i
                    while k < nx and col[k]:
                        k += 1
                    runs.add((i, k))
                    i = k
                else:
                    i += 1
        for span in sorted(set(open_) - runs):
            close(span[0], span[1], open_.pop(span), j)
        for span in runs:
            open_.setdefault(span, j)
    return out


def danger_rect(press: Press) -> AABB:
    """Bounding box of the danger polygon dilated by dilation_m (= the polygon's bounding box grown by it)."""
    d = float(press.dilation_m.value)
    x0, z0, x1, z1 = _aabb(press.danger_polygon.value)
    return round(x0 - d, 6), round(z0 - d, 6), round(x1 + d, 6), round(z1 + d, 6)


def footprint_rect(press: Press) -> dict:
    return _rect(f"PRESS_{press.id}", "press", *_aabb(press.footprint.value))


ROT_TOL = 0.05                  # |r1.r2| and | |r1| - |r2| | / mean above this (after the f fit): not a pinhole map
# Pinhole focal range of probe P6 (PLAN_TWIN §3) for the 1920 px wide frame; = intercycle.framing.F_RANGE (pinned by
# tests/test_twin_compile.py; not imported: intercycle.framing has lazy scipy / cv2 imports and compile is Isaac-side).
F_RANGE_PX = (1020.0, 1430.0)
F_RANGE_WIDTH_PX = 1920
# Perspective content of H below which f is not observable from one floor homography (image plane near parallel to
# the floor): sqrt((r1z r2z)^2 + (r1z^2 - r2z^2)^2) at the prior f. Then the prior f is used (f_source "prior").
F_OBSERVABLE_MIN = 0.02


@dataclass(frozen=True)
class CameraFit:
    """The camera of one framing as compile placed it, with the focal length it used."""
    framing: str
    xz: tuple[float, float]           # floor point under the camera centre (nadir)
    height_m: float | None            # camera centre above the floor (None for an affine H)
    f_px: float | None                # focal length used (px; None for an affine H)
    f_source: str                     # "fitted" (from H), "prior" (K's f: not observable from H) or "affine"
    residual: float                   # orthonormality residual of r1, r2 at f_px (0 for an affine H)

    def as_json(self) -> dict:
        return {"framing": self.framing, "xz": [round(v, 6) for v in self.xz],
                "height_m": None if self.height_m is None else round(self.height_m, 6),
                "f_px": None if self.f_px is None else round(self.f_px, 3), "f_source": self.f_source,
                "residual": round(self.residual, 9)}


def _intrinsics(fr: Framing) -> tuple[float, float, float, float]:
    """(fx, aspect fy/fx, cx, cy) of the framing's prior K, which must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""
    K = np.array(fr.K.value, float)
    if not np.all(np.isfinite(K)) or abs(K[2, 2]) < _EPS:
        raise CompileError(f"framing {fr.id}: K is not finite")
    K = K / K[2, 2]
    if max(abs(K[0, 1]), abs(K[1, 0]), abs(K[2, 0]), abs(K[2, 1])) > _EPS or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise CompileError(f"framing {fr.id}: K must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] with fx, fy > 0")
    return float(K[0, 0]), float(K[1, 1] / K[0, 0]), float(K[0, 2]), float(K[1, 2])


def camera_fit(fr: Framing) -> CameraFit:
    """Place the camera of a framing from H_img_to_floor, the principal point and the aspect of K.

    Perspective H: H^-1 = s K [r1 r2 t] and C = -R^T t. K's focal length is a prior (P6: 1020-1430 px), so f is
    estimated from H (Zhang's single-plane constraints r1.r2 = 0 and |r1| = |r2|, linear least squares in 1/f^2 with
    the principal point and aspect of K) when H carries enough perspective (F_OBSERVABLE_MIN), else K's f is used. It
    fails closed (CompileError) when no real f fits, f is outside F_RANGE_PX (scaled to the image width), or r1, r2
    are not orthonormal within ROT_TOL at that f. An affine H (last row 0, 0, 1: the image plane is parallel to the
    floor, as in the site_v0 placeholder) has its nadir at the floor point of the principal point."""
    fx, asp, cx, cy = _intrinsics(fr)
    H = np.array(fr.H_img_to_floor.value, float)
    try:
        G = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise CompileError(f"framing {fr.id}: H_img_to_floor is singular") from None
    if not np.all(np.isfinite(G)):
        raise CompileError(f"framing {fr.id}: H_img_to_floor is singular")
    if abs(H[2, 0]) < _EPS and abs(H[2, 1]) < _EPS:
        x, z, w = H @ np.array((cx, cy, 1.0))
        return CameraFit(fr.id, (float(x / w), float(z / w)), None, None, "affine", 0.0)
    # g_j = K_prior^-1 h_j = (u r_xj, u r_yj, r_zj) * s with u = f / fx; w = 1 / u^2 solves w P + Q = 0
    g = np.stack([(G[0] - cx * G[2]) / fx, (G[1] - cy * G[2]) / (asp * fx), G[2]])
    g = g / np.linalg.norm(g[:, :2])
    g1, g2 = g[:, 0], g[:, 1]
    P = np.array([g1[0] * g2[0] + g1[1] * g2[1], g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    Q = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    scale = float(np.linalg.norm(g1) * np.linalg.norm(g2))
    width = float(fr.image_size_px.value[0])
    lo, hi = (v * width / F_RANGE_WIDTH_PX for v in F_RANGE_PX)
    if scale < _EPS:
        raise CompileError(f"framing {fr.id}: degenerate homography")
    if float(np.linalg.norm(Q)) / scale < F_OBSERVABLE_MIN:
        f, src = fx, "prior"
    else:
        w = -float(P @ Q) / float(P @ P) if float(P @ P) > _EPS else -1.0
        if not w > 0 or not math.isfinite(w):
            raise CompileError(f"framing {fr.id}: no real focal length fits H_img_to_floor "
                               "(not a pinhole homography for the principal point of K)")
        f, src = fx / math.sqrt(w), "fitted"
    if not lo <= f <= hi:
        raise CompileError(f"framing {fr.id}: focal length {f:.1f} px from H_img_to_floor is outside the "
                           f"pre-registered {lo:.0f}-{hi:.0f} px (P6)")
    Kf = np.array([[f, 0.0, cx], [0.0, asp * f, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(Kf) @ G
    n1, n2 = float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A[:, 1]))
    s = (n1 + n2) / 2
    if not math.isfinite(s) or s < _EPS:
        raise CompileError(f"framing {fr.id}: degenerate homography")
    res = max(abs(n1 - n2) / s, abs(float(A[:, 0] @ A[:, 1])) / (n1 * n2))
    if res > ROT_TOL:
        raise CompileError(f"framing {fr.id}: H_img_to_floor is not a pinhole homography for K at f = {f:.1f} px "
                           f"(orthonormality residual {res:.3f} > {ROT_TOL})")
    if A[2, 2] < 0:                                         # the floor is in front of the camera
        s = -s
    r1, r2, t = A[:, 0] / s, A[:, 1] / s, A[:, 2] / s
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    C = -R.T @ t
    if not np.all(np.isfinite(C)):
        raise CompileError(f"framing {fr.id}: camera centre is not finite")
    return CameraFit(fr.id, (float(C[0]), float(C[1])), abs(float(C[2])), float(f), src, float(res))


def camera_floor_xz(fr: Framing) -> tuple[float, float]:
    """Floor point under the camera centre (nadir): `camera_fit(fr).xz`."""
    return camera_fit(fr).xz


def _seg_seg_dist(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between segments pq and ab (0 if they intersect)."""
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    r, s = q - p, b - a
    den = cross(r, s)
    if abs(den) > _EPS:
        t, u = cross(a - p, s) / den, cross(a - p, r) / den
        if -_EPS <= t <= 1 + _EPS and -_EPS <= u <= 1 + _EPS:
            return 0.0

    def pt_seg(x: np.ndarray, a_: np.ndarray, b_: np.ndarray) -> float:
        d = b_ - a_
        L2 = float(d @ d)
        w = 0.0 if L2 < _EPS else min(1.0, max(0.0, float((x - a_) @ d) / L2))
        return float(np.hypot(*(x - (a_ + w * d))))

    return min(pt_seg(p, a, b), pt_seg(q, a, b), pt_seg(a, p, q), pt_seg(b, p, q))


def _seg_rect_dist(p: np.ndarray, q: np.ndarray, r: AABB) -> float:
    x0, z0, x1, z1 = r
    for v in (p, q):
        if x0 <= v[0] <= x1 and z0 <= v[1] <= z1:
            return 0.0
    c = [np.array(v, float) for v in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    return min(_seg_seg_dist(p, q, c[k], c[(k + 1) % 4]) for k in range(4))


def _in_rect(x: float, z: float, r: AABB) -> bool:
    return r[0] < x < r[2] and r[1] < z < r[3]


# ------------------------------------------------------------------------------------------------ site graph


class TwinSiteGraph:
    """Topological site graph of one twin zone, built from tasks.yaml (R-W10); the SiteGraph interface.

    Nodes: DOCK, W_<checkpoint> (machine targets: the generated wait pose), P_<checkpoint> (point targets: the
    capture pose) and C<k>_<corner> visibility corners. Edge weights are seconds at the scenario's robot speed.
    """

    def __init__(self, scn: Scenario, tasks: Tasks, zone: str, floor: list[Point], footprints: list[AABB]):
        self.scn = scn
        self.zone = zone
        self.pos: dict[str, tuple[float, float]] = {}
        self.adj: dict[str, dict[str, float]] = {}
        self.targets: list[Target] = []
        self.inspect: dict[str, tuple[float, float]] = {}    # target id -> capture pose (x, z)
        self.yaw: dict[str, float] = {}                      # target id -> capture yaw (plan yaw)
        self.face: dict[str, tuple[float, float]] = {}       # target id -> the checkpoint's target point (x, z)
        self._floor = [np.array(p, float) for p in floor]
        self._floor_poly = list(floor)
        self._footprints = list(footprints)
        self._danger = [m.rect for m in scn.machines]
        zt = next((z for z in tasks.zones if z.zone == zone), None)
        if zt is None:
            raise CompileError(f"tasks.yaml has no entry for zone {zone}")
        self._order = list(zt.mission_order)
        cps = {c.id: c for c in tasks.checkpoints if c.zone == zone}
        self._node_required("DOCK", zt.dock.x, zt.dock.z)
        for cid in self._order:
            self._add_target(cps[cid])
        self._add_corners()
        self._connect()
        self._check_reachable()

    # ---- clearance
    def _clear_pt(self, x: float, z: float, hazards: bool = True) -> str | None:
        p = np.array((x, z), float)
        if not point_in_polygon(x, z, self._floor_poly):
            return "outside the visible floor"
        nf = len(self._floor)
        if min(_seg_seg_dist(p, p, self._floor[k], self._floor[(k + 1) % nf]) for k in range(nf)) < ROBOT_R - _EPS:
            return f"closer than {ROBOT_R} m to unseen floor"
        for r in self._footprints:
            if _seg_rect_dist(p, p, r) < ROBOT_R - _EPS:
                return f"closer than {ROBOT_R} m to a press footprint {r}"
        if hazards:
            for r in self._danger:
                if _seg_rect_dist(p, p, r) < ROBOT_R - _EPS:
                    return f"closer than {ROBOT_R} m to a danger rect {r}"
        return None

    def _clear_seg(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        p, q = np.array(a, float), np.array(b, float)
        nf = len(self._floor)
        if any(_seg_seg_dist(p, q, self._floor[k], self._floor[(k + 1) % nf]) < ROBOT_R - _EPS for k in range(nf)):
            return False
        return all(_seg_rect_dist(p, q, r) >= ROBOT_R - _EPS for r in self._footprints + self._danger)

    def _node(self, n: str, x: float, z: float) -> None:
        self.pos[n] = (float(x), float(z))
        self.adj.setdefault(n, {})

    def _node_required(self, n: str, x: float, z: float, hazards: bool = True) -> None:
        why = self._clear_pt(x, z, hazards)
        if why:
            raise CompileError(f"zone {self.zone}: node {n} at ({x:.3f}, {z:.3f}) is {why}")
        self._node(n, x, z)

    # ---- targets
    def _add_target(self, cp: Checkpoint) -> None:
        cand = sorted(cp.candidate_poses, key=lambda c: _POSE_RANK[c.source])[0]
        px, pz = cand.pose.x, cand.pose.z
        self.inspect[cp.id], self.yaw[cp.id] = (px, pz), cand.pose.yaw
        self.face[cp.id] = (float(cp.target_xzh[0]), float(cp.target_xzh[1]))
        if cp.inside_danger:
            m = self.scn.machine(cp.press) if cp.press else None
            if m is None:
                raise CompileError(f"checkpoint {cp.id}: inside_danger needs a press")
            r = m.rect
            if not _in_rect(px, pz, r):
                raise CompileError(f"checkpoint {cp.id}: capture pose ({px}, {pz}) is not inside the danger rect {r}")
            why = self._clear_pt(px, pz, hazards=False)
            if why:
                raise CompileError(f"checkpoint {cp.id}: capture pose is {why}")
            nx_, nz_ = cp.normal_xz
            L = math.hypot(nx_, nz_)
            if L < _EPS:
                raise CompileError(f"checkpoint {cp.id}: zero target normal")
            ux, uz = nx_ / L, nz_ / L
            t_exit = min(((r[2] - px) / ux if ux > 0 else (r[0] - px) / ux) if abs(ux) > _EPS else math.inf,
                         ((r[3] - pz) / uz if uz > 0 else (r[1] - pz) / uz) if abs(uz) > _EPS else math.inf)
            t = t_exit + WAIT_OFFSET_M
            node = f"W_{cp.id}"
            self._node_required(node, px + ux * t, pz + uz * t)
            if any(_seg_rect_dist(np.array((px, pz)), np.array(self.pos[node]), f) < ROBOT_R - _EPS
                   for f in self._footprints):
                raise CompileError(f"checkpoint {cp.id}: the entry line from {node} passes a press footprint")
            self.targets.append(Target(cp.id, "machine", node, m.id, cp.dwell_s))
        else:
            node = f"P_{cp.id}"
            self._node_required(node, px, pz)
            self.targets.append(Target(cp.id, "point", node, None, cp.dwell_s))

    def _add_corners(self) -> None:
        d = ROBOT_R + CORNER_PAD_M
        for k, (x0, z0, x1, z1) in enumerate(self._footprints + self._danger):
            for c, (x, z) in enumerate(((x0 - d, z0 - d), (x1 + d, z0 - d), (x1 + d, z1 + d), (x0 - d, z1 + d))):
                if self._clear_pt(x, z) is None:
                    self._node(f"C{k}_{c}", x, z)

    def _connect(self) -> None:
        v = self.scn.inspection.robot_speed_mps
        names = list(self.pos)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if self._clear_seg(self.pos[a], self.pos[b]):
                    d = math.hypot(self.pos[a][0] - self.pos[b][0], self.pos[a][1] - self.pos[b][1]) / v
                    self.adj[a][b] = d
                    self.adj[b][a] = d

    def _check_reachable(self) -> None:
        seen, todo = {"DOCK"}, ["DOCK"]
        while todo:
            for b in self.adj[todo.pop()]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        lost = [t.id for t in self.targets if t.node not in seen]
        if lost:
            raise CompileError(f"zone {self.zone}: targets not reachable from the dock: {lost}")

    # ---- SiteGraph interface
    def inspect_pose(self, target: Target) -> tuple[float, float]:
        """Robot centre while capturing: the chosen candidate pose (inside the danger rect for machine targets)."""
        return self.inspect[target.id]

    def mission(self) -> list[Target]:
        """One inspection round in the zone's fixed mission order (tasks.yaml)."""
        by_id = {t.id: t for t in self.targets}
        return [by_id[i] for i in self._order]


GRAPH_CONTRACT = "twin.sitegraph/1"


def graph_json(g: TwinSiteGraph) -> dict:
    """The zone's TwinSiteGraph as data (sitegraph.json, shipped with the world files): what the robot-side timing
    layer (intercycle_nav.timing.GraphData) rebuilds the SiteGraph interface from, since intercycle_nav never imports
    twin. Wait poses are final (generated along the target normal, R-W10): the robot side must not move them.
    Edge weights are seconds at the scenario's robot speed; `face` is each checkpoint's target point (x, z)."""
    return {"contract": GRAPH_CONTRACT, "zone": g.zone, "wait_poses_final": True,
            "pos": {n: [float(x), float(z)] for n, (x, z) in g.pos.items()},
            "adj": {a: {b: float(w) for b, w in nb.items()} for a, nb in g.adj.items()},
            "targets": [{"id": t.id, "kind": t.kind, "node": t.node, "machine": t.machine, "dwell_s": float(t.dwell_s)}
                        for t in g.targets],
            "inspect": {k: [float(v[0]), float(v[1])] for k, v in g.inspect.items()},
            "face": {k: [float(v[0]), float(v[1])] for k, v in g.face.items()},
            "mission_order": list(g._order)}


# ------------------------------------------------------------------------------------------------ scenario


def _numeric_site_id(site_id: str) -> int:
    """Scenario.site_id is an integer (broadcast header): the first 3 bytes of sha256(site_id)."""
    return int.from_bytes(hashlib.sha256(site_id.encode("utf-8")).digest()[:3], "big")


def _press_param(dyn: Dynamics, press: str, name: str) -> float:
    key = f"press.{press}.{name}"
    try:
        return float(dyn.get(key).value)
    except KeyError:
        raise CompileError(f"dynamics has no {key}") from None


def zone_scenario(site: Site, dyn: Dynamics, tasks: Tasks, zone: SiteZone, extent: AABB,
                  danger: Mapping[str, AABB], fits: Mapping[str, CameraFit] | None = None) -> Scenario:
    fr = {f.id: f for f in site.framings}
    if fits is None:
        fits = {f: camera_fit(fr[f]) for f in zone.framings}
    machines = []
    for p in site.presses:
        if p.zone != zone.id:
            continue
        x0, z0, x1, z1 = danger[p.id]
        machines.append(Machine(id=p.id, kind="press", on_line=False, x=round((x0 + x1) / 2, 6),
                                z=round((z0 + z1) / 2, 6), size=(round(x1 - x0, 6), round(z1 - z0, 6)), busy=_press_param(dyn, p.id, "p_running"),
                                period_s=_press_param(dyn, p.id, "stroke_period_s")))
    zt = next(z for z in tasks.zones if z.zone == zone.id)
    cps = [c for c in tasks.checkpoints if c.zone == zone.id]
    fx0, fz0, fx1, fz1 = _aabb(zone.floor_polygon.value)
    return Scenario(
        name=f"{site.site_id}-{zone.id}", version=1, site_id=_numeric_site_id(site.site_id), key_id=KEY_ID,
        hall={"x": [extent[0], extent[2]], "z": [extent[1], extent[3]]},
        shifts=[], breaks=[], shift_change_stop_min=0.0,      # no shift / break calendar (P2: no timing signal)
        line=Line(takt_s=0.0, jitter=0.0, drift_per_h=0.0, andon_stops_per_h=0.0, andon_stop_s=(0.0, 0.0),
                  recovery_after_trip_s=0.0),                  # no line: every press is off-line
        machines=machines,
        inspection=Inspection(entry_m=0.0, dwell_s=max(c.dwell_s for c in cps), robot_speed_mps=ROBOT_SPEED_MPS,
                              safety_margin_s=SAFETY_MARGIN_S, local_check_latency_s=LOCAL_CHECK_LATENCY_S),
        zones=[Zone(id=zone.id, rect=(fx0, fz0, fx1, fz1))],
        traffic={},                                           # movers come from dynamics.json and the day script
        robot={"dock": [zt.dock.x, zt.dock.z], "dock_yaw": zt.dock.yaw},
        ble=Ble(adv_interval_s=0.5, range_m=25.0, packet_loss=0.10, broadcasters=[]),
        cameras=[fits[f].xz for f in zone.framings],
    )


def scenario_yaml(scn: Scenario) -> str:
    head = ("# GENERATED by twin/compile.py from the site contracts; do not edit (re-run python -m twin.build_worlds).\n"
            "# A press-shop twin zone: presses are off-line machines, no line, no shift / break calendar.\n")
    return head + yaml.safe_dump(scn.model_dump(mode="json"), sort_keys=False)


# ------------------------------------------------------------------------------------------------ compile


@dataclass
class CompiledZone:
    zone: str
    scenario: Scenario
    rects: list[dict]                 # static bodies: press footprints + unseen floor walls
    extent: AABB
    danger_rects: dict[str, AABB]
    graph: TwinSiteGraph
    floor_polygon: list[Point] = field(default_factory=list)
    cameras: dict[str, CameraFit] = field(default_factory=dict)   # framing id -> placement and the f it used


def _check_refs(site: Site, dyn: Dynamics, tasks: Tasks) -> None:
    if not site.site_id == dyn.site_id == tasks.site_id:
        raise CompileError(f"site_id differs: site {site.site_id}, dynamics {dyn.site_id}, tasks {tasks.site_id}")
    zones = {z.id for z in site.zones}
    if {z.zone for z in tasks.zones} != zones:
        raise CompileError(f"tasks zones {sorted(z.zone for z in tasks.zones)} != site zones {sorted(zones)}")
    presses = {p.id: p for p in site.presses}
    # per-press dynamics keys are press.<id>.<param> (press.<param> is a shared parameter)
    dyn_presses = {k.split(".")[1] for k in [*dyn.observed, *dyn.assumed]
                   if k.startswith("press.") and len(k.split(".")) == 3}
    if dyn_presses != set(presses):
        raise CompileError(f"dynamics press ids {sorted(dyn_presses)} != site press ids {sorted(presses)}")
    for c in tasks.checkpoints:
        if c.press is not None and (c.press not in presses or presses[c.press].zone != c.zone):
            raise CompileError(f"checkpoint {c.id}: press {c.press} is not a press of zone {c.zone}")


def _test_split(split: Mapping[str, SplitRow] | None) -> Mapping[str, SplitRow] | None:
    """A substitute split is for tests only: refused (ProvenanceError) in a sealed run or outside pytest."""
    if split is None:
        return None
    if os.environ.get("INTERCYCLE_SEALED_RUN") == "1":
        raise ProvenanceError("a substitute split is refused in a sealed run (INTERCYCLE_SEALED_RUN=1): "
                              "compile checks against the active split only")
    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise ProvenanceError("_split_for_tests is for tests only (no PYTEST_CURRENT_TEST): compile checks against "
                              "the active split")
    return split


def compile_site(site: Site, dynamics: Dynamics, tasks: Tasks, *, safety: Safety | None = None,
                 _split_for_tests: Mapping[str, SplitRow] | None = None) -> dict[str, CompiledZone]:
    """Compile every zone. Provenance first: all source clip ids must be TRAIN ids of the active split
    (prereg/ACTIVE_SPLIT, else prereg/split.csv; fails closed without one). `_split_for_tests` substitutes the split
    under pytest only (`_test_split`)."""
    assert_train_only(source_clips(site=site, dynamics=dynamics, tasks=tasks, safety=safety),
                      _test_split(_split_for_tests))
    _check_refs(site, dynamics, tasks)
    out: dict[str, CompiledZone] = {}
    for zone in site.zones:
        extent = zone_extent(zone)
        presses = [p for p in site.presses if p.zone == zone.id]
        danger = {p.id: danger_rect(p) for p in presses}
        feet = [footprint_rect(p) for p in presses]
        rects = feet + unseen_floor_rects(zone.floor_polygon.value, extent, f"UNSEEN_{zone.id}")
        frs = {f.id: f for f in site.framings}
        fits = {f: camera_fit(frs[f]) for f in zone.framings}
        scn = zone_scenario(site, dynamics, tasks, zone, extent, danger, fits)
        graph = TwinSiteGraph(scn, tasks, zone.id, zone.floor_polygon.value,
                              [_aabb(p.footprint.value) for p in presses])
        machine_targets = [t for t in graph.targets if t.kind == "machine"]
        entry = max((math.dist(graph.pos[t.node], graph.inspect[t.id]) for t in machine_targets), default=0.0)
        scn.inspection.entry_m = round(entry, 6)                         # wait -> capture distance (Inspection.inside_s)
        out[zone.id] = CompiledZone(zone.id, scn, rects, extent, danger, graph, list(zone.floor_polygon.value), fits)
    return out


def compile_paths(site: str | Path, dynamics: str | Path, tasks: str | Path, *,
                  _split_for_tests: Mapping[str, SplitRow] | None = None) -> dict[str, CompiledZone]:
    return compile_site(load(Site, site), load(Dynamics, dynamics), load(Tasks, tasks),
                        _split_for_tests=_split_for_tests)
