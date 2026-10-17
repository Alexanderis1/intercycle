"""The observer's layered map of dynamics (L1 machine envelopes, L2 vehicle flows, L3 people flows).

Never imports ground truth (the camera shim is fed a `SimState` built by the caller). Everything here works
from what ceiling cameras would report after detection / segmentation / projection to the floor:

* `Observation` (2 Hz): body footprints (circles) with a coarse class (person, vehicle, machine part, skid), a
  heading for moving bodies, the machine id a part falls inside, a `moving` flag (frame differencing) and a
  per-camera heartbeat (`cam_ok`, False while a camera region is blacked out).
* `CameraShim.observe` turns a simulated floor into such observations: 0.15 m floor-projection noise, 5 % dropouts,
  random 10 s blackouts per camera region, 0.2 s latency, motion blur (sub-frames) for fast machine parts.
* `ConditionTracker` computes the conditioning variables itself: time-of-day bin (30 min), per-machine cycle phase
  bin (16 bins + one "stopped" bin) from transitions of motion inside each hazard zone, and the production state
  {running, andon, break, shift_change} (line stopped longer than its learned cycle allows = andon; breaks and
  shift changes from the public calendar).
* `LayeredMap` accumulates count grids: L1 per machine [17 phase bins, nx, nz] at 0.25 m over zone + 1 m margin
  (hits / visits = occupancy probability), L2 / L3 site-wide at 1 m [48 tod, 4 prod, nx, nz] (hits, heading
  histogram of 8 directions, per-camera visit counts).

Conventions: x along the line, z across, metres; heading 0 = +z, clockwise towards +x (matches types.Mover).
Grid axes are [x, z]. Time `t` is sim seconds since 00:00 of day 0.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from intercycle.scenario import Scenario

DAY = 86_400.0
PERSON, VEHICLE, PART, SKID = 0, 1, 2, 3
CLASS_NAMES = ("person", "vehicle", "part", "skid")
RUNNING, ANDON, BREAK, SHIFT_CHANGE = 0, 1, 2, 3
PROD_NAMES = ("running", "andon", "break", "shift_change")
N_PHASE = 16                       # cycle-phase bins
STOP_BIN = 16                      # machine overdue / stopped (andon, break, shift change)
N_BINS = N_PHASE + 1
ALL_BINS = tuple(range(N_BINS))
UNKNOWN_ID = "UNKNOWN"             # owner of the UNKNOWN boxes in PredStep.box_machine
TOD_BIN_S = 1800.0
N_TOD = 48
N_PROD = 4
N_HEAD = 8
FRAME_HZ = 2.0
LATENCY_S = 0.2
# Camera-shim parameters set to the perception error MEASURED by workstream V (RF-DETR-M + ByteTrack on PhysicalAI-
# SmartSpaces Warehouse_000, photorealistic synthetic video; user-approved waiver (a), 17 Oct): recall 0.68 -> 32 % miss,
# precision 0.89 -> false positives = 0.11 / 0.89 = 12.4 % of the true detections, floor-position error median ~0.38 m
# with a heavy tail (p90 ~9.7 m: fusion / ID errors). Modelled as: 85 % of detections Gaussian sigma 0.285 m per axis
# (median radial 0.38 m overall) + 15 % gross errors, radial U(5, 20) m in a random direction (P(err > 9.7 m) = 10.3 %);
# false positives are person detections, half next to a true detection (0.5-3 m away), half anywhere on the floor.
# Applied to every class (V measured people only: machine parts and vehicles inheriting it is an assumption).
MEASURED_SHIM = dict(sigma_m=0.285, p_drop=0.32, gross_p=0.15, gross_range_m=(5.0, 20.0), fp_ratio=0.11 / 0.89, fp_near=0.5)
SLOTS_PARTS = 4096            # noise-slot offset separating machine-part circles from skid circles


def tod_bin(t: float) -> int:
    return int((t % DAY) // TOD_BIN_S)


def heading_bin(h: np.ndarray) -> np.ndarray:
    """Compass bin 0..7 (0 = +z, 1 = 45 degrees clockwise towards +x, ...)."""
    return (np.floor(((np.asarray(h) + math.pi / 8) % (2 * math.pi)) / (math.pi / 4)).astype(np.int64)) % N_HEAD


def calendar_state(scn: Scenario, t: float) -> int | None:
    """Public break / shift-change calendar: BREAK, SHIFT_CHANGE or None."""
    tod = t % DAY
    for a, b, kind in scn.line_stops():
        if a <= tod < b:
            return BREAK if kind == "break" else SHIFT_CHANGE
    return None


# ------------------------------------------------------------------------------------------------ rasteriser

def raster(cx, cz, r, x0, z0, nx, nz, res: float, centre: bool = False):
    """Cells hit by circles (cx, cz, r). x0, z0, nx, nz may be scalars or per-circle arrays (local grids).

    centre=False: a cell is hit if the circle intersects it (conservative, used for the 1 m site layers).
    centre=True: a cell is hit if its centre lies inside the circle (used for the 0.25 m envelopes).
    Returns (circle_index, ix, iz) of the valid cells; every circle hits at least the cell under its centre when
    centre=False.
    """
    cx, cz, r = (np.asarray(a, float) for a in (cx, cz, r))
    n = cx.shape[0]
    if n == 0:
        e = np.zeros(0, np.int64)
        return e, e, e
    col = lambda v: v if np.ndim(v) == 0 else np.asarray(v)[:, None]       # per-circle values -> column
    x0c, z0c, nxc, nzc = col(x0), col(z0), col(nx), col(nz)
    m = int(math.ceil(float(r.max()) / res)) + 1
    k = np.arange(-m, m + 1)
    IX = np.floor((cx[:, None] - x0c) / res).astype(np.int64) + k[None, :]
    IZ = np.floor((cz[:, None] - z0c) / res).astype(np.int64) + k[None, :]
    if centre:
        dx = x0c + (IX + 0.5) * res - cx[:, None]
        dz = z0c + (IZ + 0.5) * res - cz[:, None]
    else:
        lo_x = x0c + IX * res
        lo_z = z0c + IZ * res
        dx = np.maximum(np.maximum(lo_x - cx[:, None], cx[:, None] - lo_x - res), 0.0)
        dz = np.maximum(np.maximum(lo_z - cz[:, None], cz[:, None] - lo_z - res), 0.0)
    dx2 = np.where((IX >= 0) & (IX < nxc), dx * dx, 1e12)                   # out-of-grid cells can never be hit
    dz2 = np.where((IZ >= 0) & (IZ < nzc), dz * dz, 1e12)
    hit = dx2[:, :, None] + dz2[:, None, :] <= (r * r)[:, None, None]
    ci, a, b = np.nonzero(hit)
    return ci, IX[ci, a], IZ[ci, b]


def mask_from_circles(circles: np.ndarray, x0: float, z0: float, nx: int, nz: int, res: float) -> np.ndarray:
    """Boolean [nx, nz] grid of the cells whose centre lies inside any circle (ground-truth rasterisation for tests)."""
    out = np.zeros((nx, nz), bool)
    circles = np.asarray(circles, float).reshape(-1, 3)
    if len(circles):
        _, ix, iz = raster(circles[:, 0], circles[:, 1], circles[:, 2], x0, z0, nx, nz, res, centre=True)
        out[ix, iz] = True
    return out


# ------------------------------------------------------------------------------------------------ box cover

def box_cover(mask: np.ndarray, max_boxes: int = 16, cover_frac: float = 0.985) -> np.ndarray:
    """Greedy rectangle cover of a boolean grid [nx, nz]. Every box lies entirely inside the mask (no false area);
    each step takes the box that covers the most still-uncovered cells. Stops at `max_boxes` or when more than
    `cover_frac` of the cells are covered. Returns int array (n, 4) of cell boxes (i0, j0, i1, j1), ends exclusive."""
    G = np.asarray(mask, bool)
    nx, nz = G.shape
    total = int(G.sum())
    if total == 0:
        return np.zeros((0, 4), np.int64)
    H = np.zeros((nx, nz), np.int64)                    # consecutive True cells ending at row i (along x)
    H[0] = G[0]
    for i in range(1, nx):
        H[i] = np.where(G[i], H[i - 1] + 1, 0)
    # h[i, l, w] = height of the tallest all-True rectangle ending at row i, over columns l .. l+w
    cols = np.arange(nz)[:, None] + np.arange(nz)[None, :]
    inside = cols < nz
    h = np.minimum.accumulate(np.where(inside[None], H[:, np.minimum(cols, nz - 1)], 0), axis=2)
    ci, cl, cw = np.nonzero((h > 0) & inside[None])    # candidates: (end row, first column, width - 1)
    ch = h[ci, cl, cw]
    r1, r0 = ci + 1, np.maximum(ci - ch + 1, 0)         # rows [r0, r1)
    c0, c1 = cl, np.minimum(cl + cw + 1, nz)            # columns [c0, c1)
    unc = G.copy()
    boxes: list[tuple[int, int, int, int]] = []
    while len(boxes) < max_boxes and unc.sum() > (1.0 - cover_frac) * total:
        S = np.zeros((nx + 1, nz + 1), np.int64)
        S[1:, 1:] = unc.cumsum(0).cumsum(1)
        wt = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
        k = int(wt.argmax())
        if wt[k] <= 0:
            break
        box = (int(r0[k]), int(c0[k]), int(r1[k]), int(c1[k]))
        boxes.append(box)
        unc[box[0]:box[2], box[1]:box[3]] = False
    return np.array(boxes, np.int64).reshape(-1, 4)


def safe_box_cover(mask: np.ndarray, max_boxes: int = 12) -> np.ndarray:
    """Conservative (outer) rectangle cover: the union of the boxes contains EVERY cell of the mask, at the price of some
    false area. Greedy exact cover first (boxes inside the mask, `box_cover`, aiming at 100 %); whatever is left
    uncovered is added as the bounding boxes of its 8-connected components, and while there are more than `max_boxes`
    boxes the pair whose bounding-box union adds the least new area is merged. Used for the broadcast, where an
    uncovered cell would be a truly-occupied cell sent as free (unsafe-free)."""
    G = np.asarray(mask, bool)
    if not G.any():
        return np.zeros((0, 4), np.int64)
    boxes = [tuple(b) for b in box_cover(G, max_boxes, 1.0)]
    cov = np.zeros_like(G)
    for i0, j0, i1, j1 in boxes:
        cov[i0:i1, j0:j1] = True
    left = G & ~cov
    if left.any():
        lab, n = _label8(left)
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            boxes.append((int(ii.min()), int(jj.min()), int(ii.max()) + 1, int(jj.max()) + 1))
    B = np.array(boxes, np.int64).reshape(-1, 4)
    while len(B) > max_boxes:
        a = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        u0 = np.minimum(B[:, None, :2], B[None, :, :2])
        u1 = np.maximum(B[:, None, 2:], B[None, :, 2:])
        ua = (u1[..., 0] - u0[..., 0]) * (u1[..., 1] - u0[..., 1])
        cost = ua - a[:, None] - a[None, :]                   # new area added by merging (negative when they overlap)
        np.fill_diagonal(cost, np.iinfo(np.int64).max)
        i, j = np.unravel_index(int(cost.argmin()), cost.shape)
        merged = np.r_[u0[i, j], u1[i, j]]
        B = np.vstack([np.delete(B, [i, j], 0), merged[None]])
    return B


def _label8(G: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels of a small boolean grid (no scipy dependency)."""
    lab = np.zeros(G.shape, np.int32)
    n = 0
    nx, nz = G.shape
    for i0, j0 in zip(*np.nonzero(G)):
        if lab[i0, j0]:
            continue
        n += 1
        stack = [(i0, j0)]
        lab[i0, j0] = n
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < nx and 0 <= b < nz and G[a, b] and not lab[a, b]:
                        lab[a, b] = n
                        stack.append((a, b))
    return lab, n


def dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Binary dilation by a disc of `cells` cells radius (inflation margin on a grid)."""
    G = np.asarray(mask, bool)
    if cells <= 0 or not G.any():
        return G.copy()
    out = G.copy()
    nx, nz = G.shape
    for di in range(-cells, cells + 1):
        for dj in range(-cells, cells + 1):
            if di * di + dj * dj > cells * cells + cells:      # disc (rounded outward)
                continue
            a0, a1 = max(0, di), min(nx, nx + di)
            b0, b1 = max(0, dj), min(nz, nz + dj)
            out[a0:a1, b0:b1] |= G[a0 - di:a1 - di, b0 - dj:b1 - dj]
    return out


def iso13855_margin(K: float, T: float, C: float) -> float:
    """Inflation margin S = K * T + C (ISO 13855 style): K = approach speed of the hazard (m/s, here the fastest
    machine part or the robot), T = total delay between what the map says and reality (camera latency + broadcast
    + robot reaction, s), C = position uncertainty (floor-projection noise, cell quantisation, m)."""
    return float(K) * float(T) + float(C)


# ------------------------------------------------------------------------------------------------ observations

@dataclass
class SimState:
    """What the simulated floor looks like at one instant (fed to the camera shim by the caller; may come from ground truth)."""
    t: float
    mover_circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (n, 3) cx, cz, r (one row per body circle)
    mover_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))    # PERSON or VEHICLE
    mover_heading: np.ndarray = field(default_factory=lambda: np.zeros(0))         # rad, 0 = +z clockwise
    mover_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))           # m/s
    parts: dict[str, np.ndarray] = field(default_factory=dict)                      # machine id -> (k, 3) circles
    skids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))            # (s, 3) circles


@dataclass
class Observation:
    """One 2 Hz camera tick as the observer receives it. Positions are noisy floor projections."""
    t: float                    # arrival time (capture + latency)
    t_cap: float                # capture time stamped in the frame
    cls: np.ndarray             # (n,) int8 PERSON / VEHICLE / PART / SKID
    x: np.ndarray
    z: np.ndarray
    r: np.ndarray
    heading: np.ndarray         # rad; NaN = unknown (stationary or not a tracked mover)
    moving: np.ndarray          # (n,) bool, frame-differencing motion flag
    machine: np.ndarray         # (n,) int16 index of the machine zone the detection falls inside, -1 none
    cam_ok: np.ndarray          # (n_cam,) bool heartbeat: False while a camera region is blacked out
    # ---- Observation v2 (None on a v1 observation; `obs_v1_to_v2` fills them, see OBS_V2_FIELDS)
    zone_motion: np.ndarray | None = None    # (M,) float32 motion evidence >= 0 per machine zone; NaN = MISSING
    zone_vis: np.ndarray | None = None       # (M,) float32 in [0, 1]: share of the zone's floor seen by a healthy camera
    cam_floor_ok: np.ndarray | None = None   # (n_cam,) bool per camera polygon: the view of the floor is valid

    def __len__(self) -> int:
        return len(self.cls)

    @property
    def is_v2(self) -> bool:
        return self.zone_motion is not None and self.zone_vis is not None and self.cam_floor_ok is not None


# ------------------------------------------------------------------------------------------------ observation v2
# Contract (PLAN_TWIN §2, lane S4; WEAKNESSES T-W6, M-W10, M-W2). Every v1 field is kept unchanged; v2 adds, per tick:
#   zone_motion  (M,) float32  motion evidence in machine zone m (>= 0, units set by the source: the v1 converter
#                              counts moving PART detections in the zone, stat_v1 / the pixel adapter give ROI motion
#                              energy). NaN = MISSING: the zone was not fully observed this tick. MISSING is never
#                              "no motion" (T-W6: occlusion or a dead camera must not read as an idle machine).
#   zone_vis     (M,) float32  share in [0, 1] of zone m's floor (rect + margin) that a camera with cam_floor_ok sees
#                              this tick (and, once T-W6 lands, not hidden behind a person / vehicle). zone_vis <
#                              ZONE_VIS_MIN forces zone_motion NaN.
#   cam_floor_ok (n_cam,) bool one per camera polygon (the camera's visible floor), same order as cam_ok: the image is
#                              valid for floor observation. Implies cam_ok. The v1 converter copies the heartbeat
#                              (cam_ok); the M-W10 watchdog (frozen / obstructed / dark image) sets it False on its own.
# Zones are indexed like `Observation.machine` (scenario / site machine order); cameras like `cam_ok`.
OBS_V1_FIELDS = ("t", "t_cap", "cls", "x", "z", "r", "heading", "moving", "machine", "cam_ok")
OBS_V2_FIELDS = ("zone_motion", "zone_vis", "cam_floor_ok")
OBS_DTYPES = dict(t=np.float64, t_cap=np.float64, cls=np.int8, x=np.float64, z=np.float64, r=np.float64, heading=np.float64,
                  moving=np.bool_, machine=np.int16, cam_ok=np.bool_, zone_motion=np.float32, zone_vis=np.float32,
                  cam_floor_ok=np.bool_)
_DET_FIELDS = ("cls", "x", "z", "r", "heading", "moving", "machine")
ZONE_VIS_MIN = 1.0      # fail closed: a zone that is not fully seen is MISSING (T-W6 may refine with occlusion geometry)


def zone_camera_share(rects, cams, margin: float = 1.0, res: float = 0.25) -> np.ndarray:
    """(M, n_cam) float64: share of each zone's floor (rect grown by `margin`) inside each camera's polygon. Rows sum
    to 1. The body-shop cameras' polygons are the Voronoi cells of the camera positions (CameraShim.camera_of), sampled
    on a `res` grid of cell centres; the twin passes its own shares from the per-camera visible-floor polygons."""
    R = np.asarray(rects, float).reshape(-1, 4)
    C = np.asarray(cams, float).reshape(-1, 2)
    W = np.zeros((len(R), len(C)))
    for i, (x0, z0, x1, z1) in enumerate(R):
        nx = max(1, int(math.ceil((x1 - x0 + 2 * margin) / res)))
        nz = max(1, int(math.ceil((z1 - z0 + 2 * margin) / res)))
        gx = x0 - margin + (np.arange(nx) + 0.5) * (x1 - x0 + 2 * margin) / nx
        gz = z0 - margin + (np.arange(nz) + 0.5) * (z1 - z0 + 2 * margin) / nz
        px, pz = np.meshgrid(gx, gz, indexing="ij")
        cam = np.hypot(px.reshape(-1, 1) - C[None, :, 0], pz.reshape(-1, 1) - C[None, :, 1]).argmin(1)
        W[i] = np.bincount(cam, minlength=len(C)) / cam.size
    return W


def obs_v1_to_v2(obs: Observation, share: np.ndarray, vis_min: float = ZONE_VIS_MIN) -> Observation:
    """v1 -> v2. `share` = zone_camera_share(...) (M, n_cam). cam_floor_ok = the heartbeat (v1 has no image watchdog);
    zone_vis = share @ cam_floor_ok; zone_motion = moving PART detections per zone, NaN where zone_vis < vis_min.
    The v1 fields are passed through untouched (same arrays)."""
    share = np.asarray(share, float)
    n_zone, n_cam = share.shape
    cam_ok = np.asarray(obs.cam_ok, bool)
    if cam_ok.shape != (n_cam,):
        raise ValueError(f"cam_ok has shape {cam_ok.shape}, share expects ({n_cam},)")
    floor_ok = cam_ok.copy()
    vis = np.clip(share @ floor_ok.astype(float), 0.0, 1.0)
    vis = np.where(vis > 1.0 - 1e-9, 1.0, vis)            # rows of `share` sum to 1 up to rounding
    motion = np.zeros(n_zone)
    sel = (obs.cls == PART) & obs.moving & (obs.machine >= 0) & (obs.machine < n_zone)
    if sel.any():
        np.add.at(motion, obs.machine[sel].astype(np.int64), 1.0)
    motion[vis < vis_min] = np.nan
    return replace(obs, zone_motion=motion.astype(np.float32), zone_vis=vis.astype(np.float32), cam_floor_ok=floor_ok)


def check_obs_v2(obs: Observation, n_zone: int | None = None, n_cam: int | None = None) -> None:
    """Raise ValueError unless `obs` is a well-formed v2 observation (shapes, dtypes, ranges, MISSING rules)."""
    if not obs.is_v2:
        raise ValueError("not a v2 observation (zone_motion / zone_vis / cam_floor_ok missing)")
    zm, zv, cf = obs.zone_motion, obs.zone_vis, obs.cam_floor_ok
    assert zm is not None and zv is not None and cf is not None
    n = len(obs.cls)
    for f in _DET_FIELDS:
        a = getattr(obs, f)
        if a.shape != (n,) or a.dtype != OBS_DTYPES[f]:
            raise ValueError(f"{f}: shape {a.shape} dtype {a.dtype}, want ({n},) {np.dtype(OBS_DTYPES[f])}")
    for f, a in (("zone_motion", zm), ("zone_vis", zv), ("cam_floor_ok", cf), ("cam_ok", obs.cam_ok)):
        if a.ndim != 1 or a.dtype != OBS_DTYPES[f]:
            raise ValueError(f"{f}: shape {a.shape} dtype {a.dtype}, want 1-D {np.dtype(OBS_DTYPES[f])}")
    if zm.shape != zv.shape:
        raise ValueError(f"zone_motion {zm.shape} and zone_vis {zv.shape} differ")
    if cf.shape != obs.cam_ok.shape:
        raise ValueError(f"cam_floor_ok {cf.shape} and cam_ok {obs.cam_ok.shape} differ")
    if n_zone is not None and zm.shape != (n_zone,):
        raise ValueError(f"zone arrays have {zm.shape[0]} zones, want {n_zone}")
    if n_cam is not None and cf.shape != (n_cam,):
        raise ValueError(f"camera arrays have {cf.shape[0]} cameras, want {n_cam}")
    if not (np.isfinite(zv).all() and (zv >= 0).all() and (zv <= 1).all()):
        raise ValueError("zone_vis must be finite and in [0, 1]")
    if (np.isfinite(zm) & (zm < 0)).any() or np.isinf(zm).any():
        raise ValueError("zone_motion must be >= 0 or NaN")
    if np.isfinite(zm[zv == 0]).any():
        raise ValueError("an unseen zone (zone_vis 0) must have zone_motion NaN (MISSING), not a value")
    if (cf & ~obs.cam_ok).any():
        raise ValueError("cam_floor_ok True on a camera whose heartbeat cam_ok is False")


def obs_to_dict(obs: Observation) -> dict:
    """JSON-safe dict (one line of obs_v2.jsonl). NaN -> null; v2 fields are omitted on a v1 observation."""
    def lst(a):
        a = np.asarray(a)
        if a.dtype.kind == "f":
            return [None if not math.isfinite(v) else float(v) for v in a.tolist()]
        return a.tolist()
    d: dict = dict(v=2 if obs.is_v2 else 1, t=float(obs.t), t_cap=float(obs.t_cap))
    for f in OBS_V1_FIELDS[2:] + (OBS_V2_FIELDS if obs.is_v2 else ()):
        d[f] = lst(getattr(obs, f))
    return d


def obs_from_dict(d: dict) -> Observation:
    """Inverse of obs_to_dict (null -> NaN, dtypes per OBS_DTYPES)."""
    def arr(f):
        dt = OBS_DTYPES[f]
        v = d[f]
        if np.dtype(dt).kind == "f":
            v = [math.nan if x is None else x for x in v]
        return np.asarray(v, dt).reshape(-1)
    kw = {f: arr(f) for f in OBS_V1_FIELDS[2:]}
    if all(f in d for f in OBS_V2_FIELDS):
        kw.update({f: arr(f) for f in OBS_V2_FIELDS})
    return Observation(t=float(d["t"]), t_cap=float(d["t_cap"]), **kw)


def save_obs_npz(path, observations, **extra) -> None:
    """A tick sequence as one .npz: per-tick arrays (t, t_cap, cam_ok and, for v2, zone_motion, zone_vis, cam_floor_ok
    stacked to (T, ...)), detections concatenated with `offsets` (T + 1). All ticks must be v1 or all v2. `extra` arrays
    (e.g. the zone_camera_share used) are stored as given."""
    obs = list(observations)
    v2 = [o.is_v2 for o in obs]
    if any(v2) and not all(v2):
        raise ValueError("mixed v1 / v2 observations")
    out: dict = dict(format=np.array("obs_v2" if obs and all(v2) else "obs_v1"),
                     t=np.array([o.t for o in obs], np.float64), t_cap=np.array([o.t_cap for o in obs], np.float64),
                     offsets=np.r_[0, np.cumsum([len(o.cls) for o in obs])].astype(np.int64))
    for f in _DET_FIELDS:
        out[f] = np.concatenate([np.asarray(getattr(o, f), OBS_DTYPES[f]) for o in obs]) if obs else np.zeros(0, OBS_DTYPES[f])
    for f in ("cam_ok",) + (OBS_V2_FIELDS if obs and all(v2) else ()):
        out[f] = np.stack([np.asarray(getattr(o, f), OBS_DTYPES[f]) for o in obs]) if obs else np.zeros((0, 0), OBS_DTYPES[f])
    for k in extra:
        if k in out:
            raise ValueError(f"extra key {k!r} clashes with an observation field")
    np.savez_compressed(path, **out, **extra)


def load_obs_npz(path) -> tuple[list[Observation], dict]:
    """Inverse of save_obs_npz: (observations, extra arrays)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    off = d.pop("offsets")
    fmt = str(d.pop("format"))
    t, t_cap = d.pop("t"), d.pop("t_cap")
    det = {f: d.pop(f) for f in _DET_FIELDS}
    per = {f: d.pop(f) for f in ("cam_ok",) + (OBS_V2_FIELDS if fmt == "obs_v2" else ())}
    obs = [Observation(t=float(t[i]), t_cap=float(t_cap[i]), **{f: a[off[i]:off[i + 1]] for f, a in det.items()},
                       **{f: a[i] for f, a in per.items()}) for i in range(len(t))]
    return obs, d


class CameraShim:
    """Ceiling-camera detection / segmentation / floor-projection model (SYNTHETIC imperfections).

    sigma_m        floor-projection noise on every circle centre (m)
    p_drop         per-circle, per-frame missed detection
    blackout_s     length of a camera-region blackout; `blackouts_per_h` per camera (Poisson)
    latency_s      capture-to-delivery delay
    subframes      machine parts / skids are integrated over the last `exposure_s` seconds (motion blur of the
                   frame differencing), so a fast arm leaves a continuous swept footprint, not dots
    Optional stress terms (all off by default, so map builds are unchanged; used by the change-detection runs):
    clutter_per_h  spurious "unexplained motion" blobs (glare, reflections, swinging cables) anywhere on the floor,
                   Poisson per hour site-wide, each lasting U(clutter_dur_s) seconds, reported as class PART
    persistent     [(x, z, r, t0, t1, duty)]: a fixed spot that flickers (moving detections in a fraction `duty` of the
                   frames) between t0 and t1 - an aperiodic false source a change detector must not turn into a machine
    bumps          [(camera index, dx, dz, t_from)]: from t_from on, every detection in that camera's region is
                   projected (dx, dz) metres off (the camera was knocked; its homography is stale)
    """

    def __init__(self, scn: Scenario, seed: int = 0, sigma_m: float = 0.15, p_drop: float = 0.05, blackout_s: float = 10.0,
                 blackouts_per_h: float = 4.0, latency_s: float = LATENCY_S, subframes: int = 3, exposure_s: float = 0.5,
                 zone_margin_m: float = 1.0, hz: float = FRAME_HZ, clutter_per_h: float = 0.0,
                 clutter_dur_s: tuple[float, float] = (0.5, 3.0), persistent=(), bumps=(), gross_p: float = 0.0,
                 gross_range_m: tuple[float, float] = (5.0, 20.0), fp_ratio: float = 0.0, fp_near: float = 0.5):
        """Defaults = the original assumed shim (0.15 m, 5 % drop, no gross errors, no false positives); the measured
        error is `CameraShim(scn, **MEASURED_SHIM)`. gross_p: share of detections with a gross position error, radial
        U(gross_range_m); fp_ratio: false-positive person detections per true detection (fp_near of them 0.5-3 m from a
        true detection, the rest uniform on the floor)."""
        self.gross_p, self.gross_range, self.fp_ratio, self.fp_near = gross_p, gross_range_m, fp_ratio, fp_near
        self._frng = np.random.default_rng([seed, 79])
        self.clutter_per_h, self.clutter_dur = clutter_per_h, clutter_dur_s
        self.persistent, self.bumps = list(persistent), list(bumps)
        self._clutter: list[tuple[float, float, float, float]] = []      # (x, z, r, t_end)
        self._crng = np.random.default_rng([seed, 78])
        self._next_clutter = None
        hx, hz_ = scn.hall["x"], scn.hall["z"]
        self._floor = (float(hx[0]), float(hz_[0]), float(scn.data_hall["x"][1]), float(hz_[1]))
        self.rng = np.random.default_rng([seed, 77])
        self.sigma, self.p_drop, self.blackout_s = sigma_m, p_drop, blackout_s
        self.latency, self.subframes, self.exposure, self.margin, self.hz = latency_s, max(1, subframes), exposure_s, zone_margin_m, hz
        self.cams = np.array(scn.cameras, float)
        self.mids = [m.id for m in scn.machines]
        self.rects = np.array([m.rect for m in scn.machines], float)
        self.centres = np.array([[m.x, m.z] for m in scn.machines], float)
        n = len(self.cams)
        self._mean_gap = 3600.0 / blackouts_per_h if blackouts_per_h > 0 else math.inf
        self._bo_start = np.full(n, -math.inf)
        self._bo_end = np.full(n, -math.inf)
        self._next_bo = self.rng.exponential(self._mean_gap, n) if math.isfinite(self._mean_gap) else np.full(n, math.inf)
        self._prev_parts: dict[str, tuple] = {}

    # -------------------------------------------------------------------------------- blackouts
    def cam_ok(self, t_cap: float) -> np.ndarray:
        due = t_cap >= self._next_bo
        while due.any():
            s = self._next_bo[due]
            self._bo_start[due], self._bo_end[due] = s, s + self.blackout_s
            self._next_bo[due] = s + self.blackout_s + self.rng.exponential(self._mean_gap, due.sum())
            due = t_cap >= self._next_bo
        return ~((t_cap >= self._bo_start) & (t_cap < self._bo_end))

    def camera_of(self, x, z) -> np.ndarray:
        d = np.hypot(np.asarray(x)[:, None] - self.cams[None, :, 0], np.asarray(z)[:, None] - self.cams[None, :, 1])
        return d.argmin(1)

    def machine_of(self, x, z) -> np.ndarray:
        """Zone (grown by the margin) containing each point; nearest zone centre if several; -1 if none."""
        x, z = np.asarray(x), np.asarray(z)
        R, g = self.rects, self.margin
        inside = (x[:, None] >= R[None, :, 0] - g) & (x[:, None] <= R[None, :, 2] + g) & \
                 (z[:, None] >= R[None, :, 1] - g) & (z[:, None] <= R[None, :, 3] + g)
        d = np.hypot(x[:, None] - self.centres[None, :, 0], z[:, None] - self.centres[None, :, 1])
        d = np.where(inside, d, np.inf)
        k = d.argmin(1)
        return np.where(np.isfinite(d[np.arange(len(x)), k]), k, -1).astype(np.int16)

    def _moved(self, key: str, sizes: tuple, c: np.ndarray) -> np.ndarray:
        """Frame-differencing motion flag per circle: displaced > 2 cm since the previous sub-frame. The first frame, or a
        change in the number of circles of a body group (a carriage appears, a skid is picked), counts as moving - for
        that group only (`sizes` = circles per group, in order); the other groups are differenced as usual."""
        prev = self._prev_parts.get(key)
        self._prev_parts[key] = (sizes, c)
        if prev is None or len(prev[0]) != len(sizes):
            return np.ones(len(c), bool)
        if prev[0] == sizes:
            return np.hypot(c[:, 0] - prev[1][:, 0], c[:, 1] - prev[1][:, 1]) > 0.02
        out = np.ones(len(c), bool)
        a = b = 0
        for n_new, n_old in zip(sizes, prev[0]):
            if n_new == n_old and n_new:
                d = c[a:a + n_new] - prev[1][b:b + n_old]
                out[a:a + n_new] = np.hypot(d[:, 0], d[:, 1]) > 0.02
            a += n_new
            b += n_old
        return out

    def _stress_circles(self, t_cap: float) -> np.ndarray:
        """Spurious motion detections at t_cap (clutter + persistent flicker sources); empty unless enabled."""
        out = []
        if self.clutter_per_h > 0:
            gap = 3600.0 / self.clutter_per_h
            if self._next_clutter is None:
                self._next_clutter = t_cap + self._crng.exponential(gap)
            while t_cap >= self._next_clutter:
                x0, z0, x1, z1 = self._floor
                self._clutter.append((self._crng.uniform(x0, x1), self._crng.uniform(z0, z1), self._crng.uniform(0.3, 0.8),
                                      self._next_clutter + self._crng.uniform(*self.clutter_dur)))
                self._next_clutter += self._crng.exponential(gap)
            self._clutter = [c for c in self._clutter if c[3] > t_cap]
            out += [(x, z, r) for x, z, r, _ in self._clutter]
        for x, z, r, t0, t1, duty in self.persistent:
            if t0 <= t_cap < t1 and self._crng.random() < duty:
                out.append((x + self._crng.normal(0, 0.3), z + self._crng.normal(0, 0.3), r))
        return np.array(out, float).reshape(-1, 3)

    # -------------------------------------------------------------------------------- observe
    def observe(self, state: SimState | Callable[[float], SimState], t: float | None = None) -> Observation:
        """Camera tick. With a callable `state(t) -> SimState` and the tick time `t`, the frame shows the floor at
        t - latency (and machine parts are motion-blurred over the exposure). With a list of SimStates (sub-frames,
        movers only needed in the last) or a plain SimState, the last one is the capture-time state and is delivered
        `latency` later. Only the last sub-frame's movers are used."""
        if callable(state):
            t_now = float(t)
            t_cap = t_now - self.latency
            states = [state(t_cap - self.exposure * (1 - (j + 1) / self.subframes)) for j in range(self.subframes)]
        elif isinstance(state, (list, tuple)):          # sub-frames given by the caller, last one = capture time
            states = list(state)
            t_cap = float(states[-1].t)
            t_now = t_cap + self.latency
        else:
            states = [state]
            t_cap = float(state.t)
            t_now = t_cap + self.latency
        last = states[-1]
        # ---- gather true circles: movers (last sub-frame), parts + skids (all sub-frames; still circles once)
        cir, cls, hd, mv = [last.mover_circles], [last.mover_cls.astype(np.int8)], [last.mover_heading], [last.mover_speed > 0.2]
        nm = len(last.mover_circles)
        slot = [np.arange(nm)]                      # projection-error slot: the same body circle has the same error in every sub-frame
        n_sub = len(states)
        for j, s in enumerate(states):
            keep_still = j == n_sub - 1             # a circle that did not move is the same in every sub-frame: keep it once
            for kind, arrs, key in ((PART, [np.asarray(c, float).reshape(-1, 3) for c in s.parts.values()], "parts"),
                                    (SKID, [np.asarray(s.skids, float).reshape(-1, 3)], "skids")):
                if not arrs:
                    continue
                sizes = tuple(len(c) for c in arrs)
                Cc = np.vstack(arrs) if len(arrs) > 1 else arrs[0]
                moved = self._moved(key, sizes, Cc)
                sel = moved if not keep_still else slice(None)
                if len(Cc):
                    slot.append((nm if kind == PART else nm + SLOTS_PARTS) + np.arange(len(Cc))[sel])
                    cir.append(Cc[sel])
                    cls.append(np.full(len(Cc), kind, np.int8)[sel])
                    hd.append(np.full(len(Cc), np.nan)[sel])
                    mv.append(moved[sel])
        extra = self._stress_circles(t_cap)
        if len(extra):
            slot.append(nm + 2 * SLOTS_PARTS + np.arange(len(extra)))
            cir.append(extra)
            cls.append(np.full(len(extra), PART, np.int8))
            hd.append(np.full(len(extra), np.nan))
            mv.append(np.ones(len(extra), bool))
        C = np.vstack(cir) if cir else np.zeros((0, 3))
        cls_a, hd_a, mv_a = np.concatenate(cls), np.concatenate(hd), np.concatenate(mv)
        ok = self.cam_ok(t_cap)
        cam = self.camera_of(C[:, 0], C[:, 1]) if len(C) else np.zeros(0, int)
        if self.bumps and len(C):
            C = C.copy()
            for ci, dx, dz, t_from in self.bumps:
                if t_cap >= t_from:
                    sel_b = cam == ci
                    C[sel_b, 0] += dx
                    C[sel_b, 1] += dz
        keep = ok[cam] & (self.rng.random(len(C)) >= self.p_drop) if len(C) else np.zeros(0, bool)
        slot_a = np.concatenate(slot)[keep]
        C, cls_a, hd_a, mv_a = C[keep], cls_a[keep], hd_a[keep], mv_a[keep]
        n = len(C)
        err = self.rng.normal(0, self.sigma, (int(slot_a.max()) + 1 if n else 1, 2))
        if self.gross_p > 0:                        # gross errors (one per body-circle slot, shared by its sub-frames)
            g = self._frng.random(len(err)) < self.gross_p
            if g.any():
                rr = self._frng.uniform(*self.gross_range, g.sum())
                aa = self._frng.uniform(0, 2 * math.pi, g.sum())
                err[g] = np.stack([rr * np.sin(aa), rr * np.cos(aa)], 1)
        x = C[:, 0] + err[slot_a, 0]
        z = C[:, 1] + err[slot_a, 1]
        r = np.maximum(C[:, 2] + self.rng.normal(0, 0.03, n), 0.1)
        hd_a = np.where(np.isfinite(hd_a) & mv_a, hd_a + self.rng.normal(0, 0.15, n), np.nan)
        if self.fp_ratio > 0 and n:                 # false-positive person detections
            k = int(self._frng.poisson(self.fp_ratio * float((cls_a == PERSON).sum())))   # precision measured on people
            if k:
                near = self._frng.random(k) < self.fp_near
                fx = np.empty(k)
                fz = np.empty(k)
                src = self._frng.integers(0, n, k)
                d = self._frng.uniform(0.5, 3.0, k)
                a = self._frng.uniform(0, 2 * math.pi, k)
                fx[near] = x[src[near]] + d[near] * np.sin(a[near])
                fz[near] = z[src[near]] + d[near] * np.cos(a[near])
                x0, z0, x1, z1 = self._floor
                fx[~near] = self._frng.uniform(x0, x1, (~near).sum())
                fz[~near] = self._frng.uniform(z0, z1, (~near).sum())
                ok_fp = ok[self.camera_of(fx, fz)]
                fx, fz, kk = fx[ok_fp], fz[ok_fp], int(ok_fp.sum())
                x, z = np.r_[x, fx], np.r_[z, fz]
                r = np.r_[r, np.full(kk, 0.3)]
                cls_a = np.r_[cls_a, np.full(kk, PERSON, np.int8)]
                hd_a = np.r_[hd_a, np.full(kk, np.nan)]
                mv_a = np.r_[mv_a, np.zeros(kk, bool)]
                n = len(x)
        is_m = (cls_a == PART) | (cls_a == SKID)
        mach = np.full(n, -1, np.int16)
        if is_m.any():
            mach[is_m] = self.machine_of(x[is_m], z[is_m])
        return Observation(t_now, t_cap, cls_a, x, z, r, hd_a, mv_a, mach, ok)


# ------------------------------------------------------------------------------------------------ conditioning

@dataclass
class Cond:
    """Conditioning variables the observer computed itself for one tick."""
    t: float
    tod_bin: int
    prod_state: int
    phase_bin: np.ndarray        # (M,) int16: 0..15 phase bin, STOP_BIN overdue/stopped, -1 unknown
    phase: np.ndarray            # (M,) fraction of the learned cycle since the last start (NaN unknown)
    period: np.ndarray           # (M,) learned cycle period (NaN unknown)
    busy: np.ndarray
    jitter: np.ndarray
    line_stopped: bool = False


class ConditionTracker:
    """Online cycle phase and production state computed from the observations alone.

    Motion signals (one boolean per 2 Hz tick):
      * machine m: "a machine part inside zone m is moving" (frame differencing);
      * the LINE row: "a skid is moving", i.e. the line is indexing (all bodies hop to the next station together).
    A rise is confirmed after `rise_confirm` consecutive moving ticks, a fall after `fall_confirm` still ticks;
    camera blackouts (cam_ok False) pause the debouncing, and a rise that only surfaces after a blackout is put at
    the time predicted from the learned period. Period / busy fraction / jitter come from rise-to-rise intervals
    (intervals that are not about one period, i.e. stops, are excluded).
    Phase = time since the last start / period, where the start is the line index for line stations (so every station
    is indexed by the same line phase; dwell pauses inside a station cycle cannot fool it) and the machine's own rise
    for off-line cells; if no line clock has been learned yet a line station falls back to its own rises. Once the
    phase exceeds 1 + tol the machine is overdue: STOP_BIN. Production state: the public calendar gives break and
    shift_change; otherwise the line being overdue (stopped longer than its learned cycle allows) gives andon."""

    def __init__(self, scn: Scenario, hz: float = FRAME_HZ, rise_confirm: int = 2, fall_confirm: int = 4, line_confirm: int = 3,
                 use_learner: bool = True, learn_window_s: float = 1500.0, learn_every_s: float = 900.0, learn_first_s: float = 480.0,
                 line_state: Callable | None = None, calendar: bool = False):
        """Production state. line_state(t) -> RUNNING / ANDON / BREAK / SHIFT_CHANGE or None: the discovered line state
        (workstream P's PhaseModel.line_state, what every script uses). Without it the state is the motion rule only:
        line stopped longer than its cycle allows -> ANDON ("stopped"), else RUNNING; breaks are then not told apart from
        andons. calendar=True re-enables the public break calendar as an EXTERNAL INPUT (unit tests of the layer
        mechanics only; no published number uses it)."""
        self.scn, self.hz = scn, hz
        self.line_state = line_state
        self.calendar = calendar
        M = len(scn.machines)
        self.M, self.L, R = M, M, M + 1
        self.mids = [m.id for m in scn.machines]
        self.on_line = np.array([m.on_line for m in scn.machines])
        cams = np.array(scn.cameras, float)
        cen = np.array([[m.x, m.z] for m in scn.machines], float)
        self.cam_of_machine = np.hypot(cen[:, None, 0] - cams[None, :, 0], cen[:, None, 1] - cams[None, :, 1]).argmin(1)
        self.confirm_rise = np.full(R, rise_confirm)
        self.confirm_rise[M] = line_confirm
        self.fall_confirm = fall_confirm
        self.cur = np.zeros(R, np.int8)
        self.opp = np.zeros(R, np.int64)
        self.opp_t0 = np.zeros(R)
        self.opp_lo = np.full(R, np.nan)
        self.last_valid = np.full(R, np.nan)
        self.last_rise = np.full(R, np.nan)
        self.period = np.full(R, np.nan)
        self.busy = np.full(R, np.nan)
        self.jitter = np.full(R, np.nan)
        self._rises: list[list[float]] = [[] for _ in range(R)]
        self._acts: list[list[float]] = [[] for _ in range(R)]
        self.n_rises = np.zeros(R, np.int64)
        self.t_first: float | None = None
        # optional cross-check / seed from the cycle learner (observer.learn_cycles) on the buffered motion signals
        self.use_learner = use_learner
        self.learn_window_n = int(learn_window_s * hz)
        self.learn_every_s, self.learn_first_s = learn_every_s, learn_first_s
        self._lt: list[float] = []
        self._ly: list[np.ndarray] = []
        self._next_learn: float | None = None
        self.learner_models: dict[int, object] = {}
        self.learner_agreement: dict[int, float] = {}

    def _relearn(self) -> None:
        """Run learn_cycles.learn on every row's buffered motion signal. A confident learner period seeds rows the online
        estimate has not resolved (e.g. a station with long dwell pauses); where both exist the relative disagreement is
        kept in `learner_agreement`."""
        try:
            from intercycle.observer import learn_cycles as LC
        except Exception:                                   # learner not available: online estimates only
            self.use_learner = False
            return
        T, Y = np.array(self._lt), np.array(self._ly, float)
        for r in range(self.M + 1):
            if Y[:, r].std() < 1e-9:
                continue
            try:
                mod = LC.learn(T, Y[:, r], index=r, latency_s=0.0).model
            except Exception:
                continue
            if mod.confidence < LC.MIN_CONF or not 5.0 <= mod.period_s <= 300.0:
                continue
            self.learner_models[r] = mod
            if np.isfinite(self.period[r]):
                self.learner_agreement[r] = abs(mod.period_s / self.period[r] - 1.0)
            else:
                self.period[r] = mod.period_s
                self.jitter[r] = max(float(mod.jitter), 0.002)

    # ---- edge handling
    def _edge_time(self, m: int, rising: bool) -> float:
        t_edge = self.opp_t0[m] - 0.5 / self.hz
        lo, hi = self.opp_lo[m], self.opp_t0[m]
        if np.isfinite(lo) and hi - lo > 1.5 / self.hz and np.isfinite(self.period[m]) and np.isfinite(self.last_rise[m]):
            P = self.period[m]
            pred = self.last_rise[m] + (round((hi - self.last_rise[m]) / P) * P if rising else
                                        (self.busy[m] * P if np.isfinite(self.busy[m]) else math.nan))
            if np.isfinite(pred) and lo - 1.0 <= pred <= hi:
                return float(pred)
        return float(t_edge)

    def _on_rise(self, m: int, t_rise: float) -> None:
        h = self._rises[m]
        h.append(t_rise)
        del h[:-16]
        self.last_rise[m] = t_rise
        self.n_rises[m] += 1
        iv = np.diff(h)
        iv = iv[iv > 3.0]
        if len(iv) < 2:
            return
        if np.isfinite(self.period[m]):
            P0 = self.period[m]
        else:                                           # largest cluster of mutually similar intervals (stops are singletons)
            cnt = (np.abs(iv[:, None] / iv[None, :] - 1.0) < 0.08).sum(1)
            P0 = float(np.median(iv[np.abs(iv / iv[int(cnt.argmax())] - 1.0) < 0.08]))
        g = iv[(iv > 0.88 * P0) & (iv < 1.14 * P0)]
        if len(g) >= 2:
            P = float(np.median(g))
            self.period[m] = P
            self.jitter[m] = max(float(g.std(ddof=1)) / P if len(g) > 2 else 0.01, 0.002)
            a = self._acts[m]
            if a:
                self.busy[m] = float(np.clip(np.median(a) / P, 0.05, 0.98))

    def _on_fall(self, m: int, t_fall: float) -> None:
        if np.isfinite(self.last_rise[m]) and t_fall > self.last_rise[m]:
            a = self._acts[m]
            a.append(t_fall - self.last_rise[m])
            del a[:-16]

    # ---- step
    def step(self, obs: Observation) -> Cond:
        t, M = obs.t_cap, self.M
        y = np.zeros(M + 1, bool)
        sel = (obs.cls == PART) & obs.moving & (obs.machine >= 0)
        if sel.any():
            y[obs.machine[sel]] = True
        y[M] = bool(((obs.cls == SKID) & obs.moving).any())
        valid = np.empty(M + 1, bool)
        valid[:M] = obs.cam_ok[self.cam_of_machine]
        valid[M] = bool(obs.cam_ok.any())
        first = valid & np.isnan(self.last_valid)          # the stream starts mid-cycle: adopt the first state, no edge
        self.cur[first] = y[first]
        diff = valid & (y != (self.cur == 1))
        new_run = diff & (self.opp == 0)
        self.opp_t0[new_run] = t
        self.opp_lo[new_run] = self.last_valid[new_run]
        self.opp = np.where(diff, self.opp + 1, np.where(valid, 0, self.opp))
        thr = np.where(self.cur == 0, self.confirm_rise, self.fall_confirm)
        for m in np.flatnonzero(valid & (self.opp >= thr)):
            rising = self.cur[m] == 0
            te = self._edge_time(int(m), bool(rising))
            self.cur[m] = 1 if rising else 0
            self.opp[m] = 0
            (self._on_rise if rising else self._on_fall)(int(m), te)
        self.last_valid = np.where(valid, t, self.last_valid)
        if self.use_learner:
            self._lt.append(t)
            self._ly.append(np.where(valid, y, self.cur == 1))
            if len(self._lt) > self.learn_window_n:
                del self._lt[0], self._ly[0]
            if self._next_learn is None:
                self._next_learn = t + self.learn_first_s
            elif t >= self._next_learn:
                self._relearn()
                self._next_learn = t + self.learn_every_s
        return self._cond(t)

    def _cond(self, t: float) -> Cond:
        M, L = self.M, self.L
        if self.t_first is None:
            self.t_first = t
        line_seen = self.n_rises[L] > 0                 # without any index event for 10 min, line stations fall back to their own rises
        use_line = self.on_line & (line_seen | (t - self.t_first < 600.0))
        P = np.where(use_line, self.period[L], self.period[:M])
        lr = np.where(use_line, self.last_rise[L], self.last_rise[:M])
        jit = np.where(use_line, self.jitter[L], self.jitter[:M])
        known = np.isfinite(P) & np.isfinite(lr)
        Pk = np.where(known, P, 1.0)
        phase = np.where(known, (t - lr) / Pk, np.nan)
        tol = np.maximum(0.06, 3.0 / Pk)
        over = known & (phase > 1.0 + tol)
        bins = np.where(known, np.minimum(np.floor(np.nan_to_num(phase) * N_PHASE), N_PHASE - 1), -1)
        bins = np.where(over, STOP_BIN, np.maximum(bins, -1)).astype(np.int16)
        kl = known & self.on_line
        line_stopped = bool(kl.sum() >= 3 and (over & kl).sum() >= 0.5 * kl.sum())
        if self.line_state is not None:
            ls = self.line_state(t)
            prod = int(ls) if ls is not None else (ANDON if line_stopped else RUNNING)
        else:
            cal = calendar_state(self.scn, t) if self.calendar else None     # external input only when asked, see __init__
            prod = cal if cal is not None else (ANDON if line_stopped else RUNNING)
        return Cond(t, tod_bin(t), prod, bins, phase, P, self.busy[:M].copy(), jit, line_stopped)


# ------------------------------------------------------------------------------------------------ map views

@dataclass
class PredStep:
    """Predicted dynamic obstacles at one future instant (for the robot's space-time planner)."""
    t: float                    # absolute sim time
    boxes: np.ndarray           # (n, 4) x0, z0, x1, z1 (m): union of machine envelopes expected at t
    box_machine: list           # machine id per box
    circles: np.ndarray         # (m, 3) x, z, r: high-probability traffic cells (vehicles + people)
    circle_p: np.ndarray        # (m,) occupancy probability of that cell
    circle_cls: np.ndarray      # (m,) PERSON / VEHICLE


class MapView:
    """Read API shared by the learned `LayeredMap` and the robot-side decoded map. Subclasses provide
    `spec`, `period`, `jitter` (per machine), `boxes_for(machine_id, bins)` and `traffic(...)`."""

    spec: dict
    cam_offline: set = frozenset()

    def _unknown_static(self) -> np.ndarray:
        raise NotImplementedError

    def machine_ids(self) -> list[str]:
        return [m["id"] for m in self.spec["machines"]]

    def _observed(self, tod: int, prod: int, cls: int | None = None) -> bool:
        raise NotImplementedError

    def _resolve(self, tod: int, prod: int, cls: int | None = None) -> tuple[int, int] | None:
        """Slice to read for (tod, prod): itself when it was observed enough, else the nearest observed tod of the same
        production state, else the same tod while running, else None."""
        if self._observed(tod, prod, cls):
            return tod, prod
        for d in range(1, N_TOD // 2 + 1):
            for tt in ((tod - d) % N_TOD, (tod + d) % N_TOD):
                if self._observed(tt, prod, cls):
                    return tt, prod
        if prod != RUNNING and self._observed(tod, RUNNING, cls):
            return tod, RUNNING
        return None

    @staticmethod
    def _cls(cls) -> int:
        if isinstance(cls, str):
            return {"person": PERSON, "people": PERSON, "vehicle": VEHICLE, "vehicles": VEHICLE}[cls]
        return int(cls)

    def predict_obstacles(self, t_now: float, horizon_s: float, machine_phase_now: dict, prod_state: int, step_s: float = 0.5,
                          k_sigma: float = 2.0, p_hard: float = 0.30, prod_until_s: float = math.inf,
                          andon_valid_s: float = 3.0, with_traffic: bool = True, with_unknown: bool = True,
                          inflate_m: float = 0.0) -> list[PredStep]:
        """Time-indexed obstacles for t_now .. t_now + horizon.

        machine_phase_now: {machine_id: phase fraction 0..1 of its cycle now (from the fast layer)}; None / <0 / beyond
        1 + tol means stopped or unknown. Each machine's phase is advanced with its learned period and widened by
        k_sigma * sqrt(base^2 + jitter^2 * cycles) (the envelope is the union of the neighbouring phase bins). A machine
        in a break / shift change keeps its STOP envelope until `prod_until_s`; in andon only `andon_valid_s` (it can
        restart any moment); after that, and for stopped / unknown machines, the union of all bins is returned.
        UNKNOWN space (`unknown_mask`) is added as boxes owned by "UNKNOWN" (treat as occupied). `inflate_m` grows every
        machine box by that margin on each side (S = K * T + C, see `iso13855_margin`)."""
        ids = self.machine_ids()
        out: list[PredStep] = []
        unk = self.unknown_boxes(tod_bin(t_now), prod_state) if with_unknown else np.zeros((0, 4))
        for dt in np.arange(0.0, horizon_s + 1e-9, step_s):
            boxes, owner = [], []
            for i, mid in enumerate(ids):
                bins = self.predicted_bins(i, machine_phase_now.get(mid), prod_state, float(dt), k_sigma, prod_until_s, andon_valid_s)
                b = self.boxes_for(mid, bins)
                if len(b):
                    if inflate_m > 0:
                        b = b + np.array([-inflate_m, -inflate_m, inflate_m, inflate_m])
                    boxes.append(b)
                    owner += [mid] * len(b)
            if len(unk):                                          # UNKNOWN space is occupied
                boxes.append(unk)
                owner += [UNKNOWN_ID] * len(unk)
            circ, cp, cc = np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int8)
            if with_traffic:
                tb = tod_bin(t_now + dt)
                circ, cp, cc = self._hot_cells(tb, prod_state, p_hard)
            out.append(PredStep(t_now + float(dt), np.vstack(boxes) if boxes else np.zeros((0, 4)), owner, circ, cp, cc))
        return out

    def predicted_bins(self, i: int, ph, prod_state: int, dt: float = 0.0, k_sigma: float = 2.0, prod_until_s: float = math.inf,
                       andon_valid_s: float = 3.0) -> tuple:
        """Phase bins machine i may be in `dt` seconds after a fast-layer phase `ph` (fraction of its cycle; None / < 0 /
        overdue = stopped or unknown).

        The machine's own observed phase comes first: a machine that is mid-cycle keeps cycling whatever the line state
        says (off-line cells run through line andons and breaks, and the line finishes its last cycle when a break
        starts), so its phase is advanced and widened by k_sigma * sqrt(base^2 + jitter^2 * cycles). Only a machine
        that is stopped / overdue gets its STOP bin, and only while the production state says the plant is stopped
        (break / shift change until prod_until_s, andon for andon_valid_s: it may restart any moment); otherwise, and
        whenever the phase or period is unknown, the union of all bins."""
        P = float(self.period[i]) if np.isfinite(self.period[i]) else math.nan
        cycling = ph is not None and np.isfinite(ph) and ph >= 0 and np.isfinite(P) and ph <= 1.0 + max(0.06, 3.0 / P)
        if not cycling:
            stopped = prod_state in (BREAK, SHIFT_CHANGE) and dt < prod_until_s or prod_state == ANDON and dt < andon_valid_s
            return (STOP_BIN,) if stopped else ALL_BINS
        cyc = dt / P
        j = float(self.jitter[i]) if np.isfinite(self.jitter[i]) else 0.02
        sig = math.hypot(0.03, j) if cyc == 0 else math.hypot(0.03, j * math.sqrt(cyc))
        c = int(math.floor(((ph + cyc) % 1.0) * N_PHASE))
        hw = int(math.ceil(k_sigma * sig * N_PHASE))
        return tuple(sorted({(c + d) % N_PHASE for d in range(-hw, hw + 1)}))

    # ---------------------------------------------------------------- UNKNOWN
    def unknown_mask(self, tod: int | None = None, prod: int | None = None) -> np.ndarray:
        """Site grid [nx, nz] of UNKNOWN cells (to be treated as occupied): the static part (`_unknown_static`: never
        observed, new-machine regions, machines with no learned envelope), every cell of an offline camera, and the
        whole site when the traffic slice (tod, prod) cannot be resolved for either class (nothing comparable seen)."""
        U = self._unknown_static().copy()
        for c in self.cam_offline:
            U |= self.cam_of_cell == int(c)
        if tod is not None and prod is not None:
            if any(self._resolve(int(tod), int(prod), c) is None for c in (VEHICLE, PERSON)):
                U[:] = True
        return U

    def unknown_boxes(self, tod: int | None = None, prod: int | None = None, max_boxes: int = 24) -> np.ndarray:
        """World boxes (x0, z0, x1, z1) conservatively covering `unknown_mask` (safe cover: no unknown cell left out)."""
        U = self.unknown_mask(tod, prod)
        if not U.any():
            return np.zeros((0, 4))
        s = self.spec["site"]
        cb = safe_box_cover(U, max_boxes)
        r = s["res"]
        return np.stack([s["x0"] + cb[:, 0] * r, s["z0"] + cb[:, 1] * r, s["x0"] + cb[:, 2] * r, s["z0"] + cb[:, 3] * r], 1).astype(float)

    def rect_cells(self, rect) -> np.ndarray:
        """Site-grid cells [nx, nz] touched by a world rectangle (x0, z0, x1, z1)."""
        s = self.spec["site"]
        x0, z0, x1, z1 = rect
        out = np.zeros((s["nx"], s["nz"]), bool)
        i0, i1 = max(0, int(math.floor((x0 - s["x0"]) / s["res"]))), min(s["nx"], int(math.ceil((x1 - s["x0"]) / s["res"])))
        j0, j1 = max(0, int(math.floor((z0 - s["z0"]) / s["res"]))), min(s["nz"], int(math.ceil((z1 - s["z0"]) / s["res"])))
        out[i0:i1, j0:j1] = True
        return out

    def _hot_cells(self, tb: int, prod_state: int, p_hard: float):
        key = (tb, prod_state, p_hard)
        cache = self.__dict__.setdefault("_hot_cache", {})
        if key not in cache:
            s = self.spec["site"]
            cs, ps, cl = [], [], []
            for cid, name in ((VEHICLE, "vehicle"), (PERSON, "person")):
                p, _ = self.traffic(name, tb, prod_state)
                ix, iz = np.nonzero(p >= p_hard)
                cs.append(np.stack([s["x0"] + (ix + 0.5) * s["res"], s["z0"] + (iz + 0.5) * s["res"],
                                    np.full(len(ix), 0.5 * s["res"] * math.sqrt(2))], 1))
                ps.append(p[ix, iz])
                cl.append(np.full(len(ix), cid, np.int8))
            cache[key] = (np.vstack(cs), np.concatenate(ps), np.concatenate(cl))
            if len(cache) > 256:
                cache.pop(next(iter(cache)))
        return cache[key]


def build_spec(scn: Scenario, env_res: float = 0.25, margin: float = 1.0, site_res: float = 1.0) -> dict:
    hx0, hx1 = scn.hall["x"]
    hz0, hz1 = scn.hall["z"]
    dx1 = scn.data_hall["x"][1]
    machines = []
    for m in scn.machines:
        x0, z0, x1, z1 = m.rect
        gx0, gz0 = math.floor((x0 - margin) / env_res) * env_res, math.floor((z0 - margin) / env_res) * env_res
        machines.append(dict(id=m.id, kind=m.kind, rect=[x0, z0, x1, z1], x0=gx0, z0=gz0,
                             nx=int(math.ceil((x1 + margin - gx0) / env_res)), nz=int(math.ceil((z1 + margin - gz0) / env_res)),
                             on_line=bool(m.on_line), period_nominal=float(scn.line.takt_s if m.on_line else m.period_s)))
    site = dict(x0=float(hx0), z0=float(hz0), res=site_res, nx=int(math.ceil((dx1 - hx0) / site_res)), nz=int(math.ceil((hz1 - hz0) / site_res)))
    return dict(name=scn.name, version=scn.version, site_id=scn.site_id, key_id=scn.key_id, env_res=env_res, margin=margin,
                machines=machines, site=site, cameras=[list(c) for c in scn.cameras], n_phase=N_PHASE, n_tod=N_TOD, n_prod=N_PROD)


# ------------------------------------------------------------------------------------------------ the map

class LayeredMap(MapView):
    """Count grids. L1: hits[machine][phase_bin, nx, nz] / visits[machine, phase_bin]. L2 / L3: hits[slice, nx, nz] and
    heading[slice, 8, nx, nz] with slice = tod_bin * 4 + prod_state, divided by per-camera visit counts."""

    def __init__(self, scn: Scenario | None = None, spec: dict | None = None, min_visits: int = 8, min_ticks: int = 300,
                 shrink_m: float = 0.1):
        self.spec = spec if spec is not None else build_spec(scn)
        sp = self.spec
        M = len(sp["machines"])
        self.M = M
        self.min_visits, self.min_ticks = min_visits, min_ticks
        self.shrink_m = shrink_m           # bias correction for the camera's floor-projection noise (see update)
        mm = sp["machines"]
        self.l1_x0 = np.array([m["x0"] for m in mm])
        self.l1_z0 = np.array([m["z0"] for m in mm])
        self.l1_nx = np.array([m["nx"] for m in mm], np.int64)
        self.l1_nz = np.array([m["nz"] for m in mm], np.int64)
        sizes = N_BINS * self.l1_nx * self.l1_nz
        self.l1_base = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.l1_hits = np.zeros(int(sizes.sum()), np.float32)
        self.l1_visits = np.zeros((M, N_BINS), np.float32)
        self._scratch = np.zeros(int(sizes.sum()), np.int32)
        s = sp["site"]
        self.nx, self.nz = s["nx"], s["nz"]
        cams = np.array(sp["cameras"], float)
        gx = s["x0"] + (np.arange(self.nx) + 0.5) * s["res"]
        gz = s["z0"] + (np.arange(self.nz) + 0.5) * s["res"]
        self.cam_of_cell = np.hypot(gx[:, None, None] - cams[None, None, :, 0], gz[None, :, None] - cams[None, None, :, 1]).argmin(2)
        self.cam_of_machine = np.hypot(np.array([(m["rect"][0] + m["rect"][2]) / 2 for m in mm])[:, None] - cams[None, :, 0],
                                       np.array([(m["rect"][1] + m["rect"][3]) / 2 for m in mm])[:, None] - cams[None, :, 1]).argmin(1)
        S = N_TOD * N_PROD
        self.hits = {c: np.zeros((S, self.nx, self.nz), np.float32) for c in (PERSON, VEHICLE)}
        self.head = {c: np.zeros((S, N_HEAD, self.nx, self.nz), np.float32) for c in (PERSON, VEHICLE)}
        self.visits_cam = np.zeros((S, len(cams)), np.float32)
        self.period = np.full(M, np.nan)
        self.busy = np.full(M, np.nan)
        self.jitter = np.full(M, np.nan)
        self.n_ticks = 0
        self.t_first, self.t_last = math.inf, -math.inf
        self._cover_cache: dict = {}
        self.cam_offline: set[int] = set()                 # cameras currently offline (live state; their cells are UNKNOWN)
        self.unknown_rects: list[tuple[float, float, float, float]] = []   # e.g. a new machine still being learned
        self.min_cam_ticks = min_ticks                      # a camera region seen fewer ticks than this is never-observed

    # ---------------------------------------------------------------- update
    def update(self, obs: Observation, t: float, cond: Cond) -> None:
        sp = self.spec["site"]
        S = cond.tod_bin * N_PROD + cond.prod_state
        self.n_ticks += 1
        self.t_first, self.t_last = min(self.t_first, t), max(self.t_last, t)
        self._cover_cache.clear()
        if self.cam_offline:                    # an offline / bumped camera's detections are not evidence (its cells are UNKNOWN)
            obs = self._drop_offline(obs)
        # ---- L1: machine parts and skids inside a zone, binned by the machine's phase
        sel = ((obs.cls == PART) | (obs.cls == SKID)) & (obs.machine >= 0)
        if sel.any():
            mi = obs.machine[sel].astype(np.int64)
            b = cond.phase_bin[mi].astype(np.int64)
            ok = b >= 0
            if ok.any():
                mi, b = mi[ok], b[ok]
                ci, ix, iz = raster(obs.x[sel][ok], obs.z[sel][ok], np.maximum(obs.r[sel][ok] - self.shrink_m, 0.05), self.l1_x0[mi], self.l1_z0[mi],
                                    self.l1_nx[mi], self.l1_nz[mi], self.spec["env_res"], centre=True)
                mm, bb = mi[ci], b[ci]
                flat = self.l1_base[mm] + (bb * self.l1_nx[mm] + ix) * self.l1_nz[mm] + iz
                if len(flat):
                    pos = np.arange(len(flat), dtype=np.int32)
                    self._scratch[flat] = pos
                    self.l1_hits[flat[self._scratch[flat] == pos]] += 1.0
        okm = (cond.phase_bin >= 0) & obs.cam_ok[self.cam_of_machine[:len(cond.phase_bin)]]   # spawned machines: change.py
        k = np.flatnonzero(okm)
        self.l1_visits[k, cond.phase_bin[k]] += 1.0
        # ---- L2 / L3
        self.visits_cam[S] += obs.cam_ok
        for cid in (PERSON, VEHICLE):
            sel = obs.cls == cid
            if not sel.any():
                continue
            ci, ix, iz = raster(obs.x[sel], obs.z[sel], obs.r[sel], sp["x0"], sp["z0"], self.nx, self.nz, sp["res"], centre=False)
            if not len(ci):
                continue
            cell = np.unique((S * self.nx + ix) * self.nz + iz)
            self.hits[cid].reshape(-1)[cell] += 1.0
            h = obs.heading[sel][ci]
            f = np.isfinite(h)
            if f.any():
                hb = heading_bin(h[f])
                hc = np.unique(((S * N_HEAD + hb) * self.nx + ix[f]) * self.nz + iz[f])
                self.head[cid].reshape(-1)[hc] += 1.0
        for src, dst in ((cond.period, self.period), (cond.busy, self.busy), (cond.jitter, self.jitter)):
            f = np.isfinite(src)
            dst[:len(src)][f] = src[f]

    def _drop_offline(self, obs: Observation) -> Observation:
        s = self.spec["site"]
        ix = np.clip(np.floor((obs.x - s["x0"]) / s["res"]).astype(np.int64), 0, self.nx - 1)
        iz = np.clip(np.floor((obs.z - s["z0"]) / s["res"]).astype(np.int64), 0, self.nz - 1)
        keep = ~np.isin(self.cam_of_cell[ix, iz], list(self.cam_offline))
        ok = obs.cam_ok.copy()
        ok[list(self.cam_offline)] = False
        floor_ok = None if obs.cam_floor_ok is None else obs.cam_floor_ok & ok     # v2 fields kept (cam_floor_ok implies cam_ok)
        return replace(obs, cls=obs.cls[keep], x=obs.x[keep], z=obs.z[keep], r=obs.r[keep], heading=obs.heading[keep],
                       moving=obs.moving[keep], machine=obs.machine[keep], cam_ok=ok, cam_floor_ok=floor_ok)

    def merge(self, other: LayeredMap) -> None:
        """Add another map's counts (maps built on separate windows / processes combine exactly)."""
        self.l1_hits += other.l1_hits
        self.l1_visits += other.l1_visits
        self.visits_cam += other.visits_cam
        for c in (PERSON, VEHICLE):
            self.hits[c] += other.hits[c]
            self.head[c] += other.head[c]
        for a in ("period", "busy", "jitter"):
            mine, theirs = getattr(self, a), getattr(other, a)
            f = np.isfinite(theirs) & ~np.isfinite(mine)
            mine[f] = theirs[f]
        self.n_ticks += other.n_ticks
        self.t_first, self.t_last = min(self.t_first, other.t_first), max(self.t_last, other.t_last)
        self.cam_offline |= set(other.cam_offline)
        self.unknown_rects += [r for r in other.unknown_rects if r not in self.unknown_rects]
        self._cover_cache.clear()

    # ---------------------------------------------------------------- L1 queries
    def _l1(self, i: int) -> np.ndarray:
        n = N_BINS * self.l1_nx[i] * self.l1_nz[i]
        return self.l1_hits[self.l1_base[i]:self.l1_base[i] + n].reshape(N_BINS, self.l1_nx[i], self.l1_nz[i])

    def _mi(self, machine_id: str) -> int:
        return self.machine_ids().index(machine_id)

    def envelope_prob(self, machine_id: str, phase_bin: int) -> np.ndarray:
        i = self._mi(machine_id)
        v = self.l1_visits[i, phase_bin]
        return self._l1(i)[phase_bin] / max(float(v), 1.0)

    def envelope(self, machine_id: str, phase_bin: int, p_min: float = 0.05) -> np.ndarray:
        """Boolean local grid [nx, nz] of the floor cells the machine's moving parts occupy in at least a fraction
        p_min of the observed ticks of that phase bin. A bin with fewer than `min_visits` observed ticks falls back to
        the union of all well-observed bins (conservative)."""
        i = self._mi(machine_id)
        if self.l1_visits[i, phase_bin] >= self.min_visits:
            return self.envelope_prob(machine_id, phase_bin) >= p_min
        out = np.zeros((self.l1_nx[i], self.l1_nz[i]), bool)
        seen = False
        for b in range(N_BINS):
            if self.l1_visits[i, b] >= self.min_visits:
                out |= self.envelope_prob(machine_id, b) >= p_min
                seen = True
        if not seen:                                        # nothing learned yet: the whole zone grid (UNKNOWN = occupied)
            out[:] = True
        return out

    def l1_known(self, machine_id: str) -> bool:
        """True once at least one phase bin of the machine has enough observed ticks to have an envelope."""
        return bool((self.l1_visits[self._mi(machine_id)] >= self.min_visits).any())

    def _unknown_static(self) -> np.ndarray:
        U = self.visits_cam.sum(0)[self.cam_of_cell] < self.min_cam_ticks          # never (or barely) observed
        for r in self.unknown_rects:
            U |= self.rect_cells(r)
        for d in self.spec["machines"]:
            if not self.l1_known(d["id"]):
                U |= self.rect_cells(d["rect"])
        return U

    def add_machine(self, machine_id: str, rect, kind: str = "discovered", period_nominal: float = float("nan")) -> int:
        """Append a machine discovered on the floor (workstream M, observer/change.py) with an empty L1 model over its
        zone rect (+ margin). Returns its index. Its counts are filled by the caller (`l1_view`)."""
        sp = self.spec
        res, margin = sp["env_res"], sp["margin"]
        x0, z0, x1, z1 = (float(v) for v in rect)
        gx0, gz0 = math.floor((x0 - margin) / res) * res, math.floor((z0 - margin) / res) * res
        d = dict(id=machine_id, kind=kind, rect=[x0, z0, x1, z1], x0=gx0, z0=gz0,
                 nx=int(math.ceil((x1 + margin - gx0) / res)), nz=int(math.ceil((z1 + margin - gz0) / res)),
                 on_line=False, period_nominal=period_nominal)
        sp["machines"] = list(sp["machines"]) + [d]
        n = N_BINS * d["nx"] * d["nz"]
        self.l1_base = np.r_[self.l1_base, len(self.l1_hits)].astype(np.int64)
        self.l1_hits = np.r_[self.l1_hits, np.zeros(n, np.float32)]
        self._scratch = np.r_[self._scratch, np.zeros(n, np.int32)]
        self.l1_visits = np.vstack([self.l1_visits, np.zeros((1, N_BINS), np.float32)])
        self.l1_x0, self.l1_z0 = np.r_[self.l1_x0, gx0], np.r_[self.l1_z0, gz0]
        self.l1_nx, self.l1_nz = np.r_[self.l1_nx, d["nx"]].astype(np.int64), np.r_[self.l1_nz, d["nz"]].astype(np.int64)
        cams = np.array(sp["cameras"], float)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        self.cam_of_machine = np.r_[self.cam_of_machine, int(np.hypot(cams[:, 0] - cx, cams[:, 1] - cz).argmin())]
        for a in ("period", "busy", "jitter"):
            setattr(self, a, np.r_[getattr(self, a), np.nan])
        self.M += 1
        self._cover_cache.clear()
        return self.M - 1

    def l1_view(self, i: int) -> np.ndarray:
        """Writable view [N_BINS, nx, nz] of machine i's hit counts (for a spawned model fed by observer/change.py)."""
        return self._l1(i)

    def envelope_union(self, machine_id: str, bins, p_min: float = 0.05) -> np.ndarray:
        out = None
        for b in bins:
            e = self.envelope(machine_id, int(b), p_min)
            out = e if out is None else (out | e)
        return out

    def box_cover(self, machine_id: str, phase_bin: int | tuple | list, p_min: float = 0.05, max_boxes: int = 16,
                  cover_frac: float = 0.985, world: bool = True, safe: bool = True) -> np.ndarray:
        """Axis-aligned boxes covering the envelope. safe=True (default, what is broadcast): conservative outer cover,
        every envelope cell is inside a box (`safe_box_cover`). safe=False: greedy inner cover (no box contains a cell
        below p_min, stops at cover_frac). world=True -> (n, 4) x0, z0, x1, z1 in metres; world=False -> cell boxes
        (i0, j0, i1, j1) in the machine's local grid."""
        bins = (phase_bin,) if isinstance(phase_bin, (int, np.integer)) else tuple(phase_bin)
        key = (machine_id, bins, p_min, max_boxes, cover_frac, world, safe)
        if key not in self._cover_cache:
            i = self._mi(machine_id)
            env = self.envelope_union(machine_id, bins, p_min)
            cb = safe_box_cover(env, max_boxes) if safe else box_cover(env, max_boxes, cover_frac)
            if world:
                res = self.spec["env_res"]
                cb = np.stack([self.l1_x0[i] + cb[:, 0] * res, self.l1_z0[i] + cb[:, 1] * res,
                               self.l1_x0[i] + cb[:, 2] * res, self.l1_z0[i] + cb[:, 3] * res], 1) if len(cb) else np.zeros((0, 4))
            self._cover_cache[key] = cb
        return self._cover_cache[key]

    def boxes_for(self, machine_id: str, bins) -> np.ndarray:
        return self.box_cover(machine_id, tuple(bins))

    # ---------------------------------------------------------------- L2 / L3 queries
    def slice_ticks(self, tod: int, prod: int) -> float:
        return float(self.visits_cam[tod * N_PROD + prod].max())

    def _observed(self, tod: int, prod: int, cls: int | None = None) -> bool:
        return self.slice_ticks(tod, prod) >= self.min_ticks

    def traffic(self, cls, tod_bin: int, prod_state: int, fallback: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(occupancy probability [nx, nz] float32, dominant heading [nx, nz] int8 in 0..7, -1 where unknown)."""
        c = self._cls(cls)
        r = self._resolve(tod_bin, prod_state, c) if fallback else ((tod_bin, prod_state) if self._observed(tod_bin, prod_state, c) else None)
        if r is None:
            return np.zeros((self.nx, self.nz), np.float32), np.full((self.nx, self.nz), -1, np.int8)
        S = r[0] * N_PROD + r[1]
        v = self.visits_cam[S][self.cam_of_cell]
        p = np.where(v > 0, self.hits[c][S] / np.maximum(v, 1.0), 0.0).astype(np.float32)
        hh = self.head[c][S]
        dom = np.where(hh.sum(0) >= 3, hh.argmax(0), -1).astype(np.int8)
        return p, dom

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        meta = json.dumps(dict(spec=self.spec, min_visits=self.min_visits, min_ticks=self.min_ticks, shrink_m=self.shrink_m, n_ticks=self.n_ticks,
                               t_first=self.t_first, t_last=self.t_last, cam_offline=sorted(int(c) for c in self.cam_offline),
                               unknown_rects=[list(map(float, r)) for r in self.unknown_rects]))
        np.savez_compressed(path, meta=np.array(meta), l1_hits=self.l1_hits, l1_visits=self.l1_visits, visits_cam=self.visits_cam,
                            hits_person=self.hits[PERSON], hits_vehicle=self.hits[VEHICLE], head_person=self.head[PERSON],
                            head_vehicle=self.head[VEHICLE], period=self.period, busy=self.busy, jitter=self.jitter)

    @classmethod
    def load(cls, path: str | Path) -> LayeredMap:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        m = cls(spec=meta["spec"], min_visits=meta["min_visits"], min_ticks=meta["min_ticks"], shrink_m=meta.get("shrink_m", 0.0))
        m.l1_hits[:], m.l1_visits[:], m.visits_cam[:] = z["l1_hits"], z["l1_visits"], z["visits_cam"]
        m.hits[PERSON][:], m.hits[VEHICLE][:] = z["hits_person"], z["hits_vehicle"]
        m.head[PERSON][:], m.head[VEHICLE][:] = z["head_person"], z["head_vehicle"]
        m.period[:], m.busy[:], m.jitter[:] = z["period"], z["busy"], z["jitter"]
        m.n_ticks, m.t_first, m.t_last = meta["n_ticks"], meta["t_first"], meta["t_last"]
        m.cam_offline = set(meta.get("cam_offline", []))
        m.unknown_rects = [tuple(r) for r in meta.get("unknown_rects", [])]
        return m


class MapBuilder:
    """Convenience driver: observation -> conditioning -> map.

    phase_model (optional, duck-typed: workstream P's observer/phases.PhaseModel, never imported here): fed every tick
    before the tracker. use_line_state=True (default) makes the discovered line state the production state ("unknown"
    falls back to the motion rule). use_bins=True replaces the tracker's 16 equal-slice bins for L1 by the discovered
    sub-phases (sub-phase k -> bin k, P's "blocked" bin -> STOP_BIN, -1 unknown); off by default (see MAP_RESULTS).
    Without a phase model the production state is the motion rule (calendar=True: public calendar, tests only)."""

    def __init__(self, scn: Scenario, map_: LayeredMap | None = None, tracker: ConditionTracker | None = None,
                 phase_model=None, use_bins: bool = False, use_line_state: bool = True, calendar: bool = False, **map_kw):
        self.scn = scn
        self.map = map_ or LayeredMap(scn, **map_kw)
        self.pm = phase_model
        self.use_bins = phase_model is not None and use_bins
        ls = None
        if phase_model is not None and use_line_state:
            names = {n: i for i, n in enumerate(PROD_NAMES)}
            ls = lambda t: names.get(phase_model.line_state(t))
        self.tracker = tracker or ConditionTracker(scn, line_state=ls, calendar=calendar)
        if tracker is not None and ls is not None:
            tracker.line_state = ls
        self.mids = [m.id for m in scn.machines]

    def condition(self, obs: Observation) -> Cond:
        if self.pm is not None:
            self.pm.update(obs)
        cond = self.tracker.step(obs)
        if self.use_bins:
            b = np.full(len(self.mids), -1, np.int16)
            for i, mid in enumerate(self.mids):
                k = int(self.pm.phase_bin(mid, obs.t_cap))
                n = int(self.pm.n_bins(mid))
                b[i] = -1 if k < 0 else (STOP_BIN if k == n - 1 else min(k, N_PHASE - 1))
            cond.phase_bin = b
        return cond

    def feed(self, obs: Observation) -> Cond:
        cond = self.condition(obs)
        self.map.update(obs, obs.t_cap, cond)
        return cond
