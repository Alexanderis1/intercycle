"""The twin camera model `stat_v1` (lane S3b; PLAN_TWIN §2 twin/camera.py row; WEAKNESSES M-W2, M-W3, T-W6, M-W10).

Realised state (truth, 2 Hz) -> per-tick detections in the plan frame, as the real perception chain would give them.
It runs in the plant process (`twin.plant.make_camera`), is the only reader of realised_state.jsonl besides the
engine / plant / EVAL, and its output becomes runs/<id>/obs_v2.jsonl through `mapping.ObsV2Converter` with
`hold_s = CameraStatV1.hold_s` (sized from the fitted miss bursts, below).

Parameters: `twin/stat_v1_params.json` (scripts/stat_v1_fit.py: DEV labels vs DEV PERC tracks, TRAIN ROI energy) and
`twin/cal/h_draws_v1.json` (scripts/stat_v1_hdraws.py: the 200 CAL v1 joint bootstrap draws). Each item says where
its number comes from: FITTED (dev / train data), SHIM (MEASURED_SHIM, `intercycle.observer.mapping`), CAL (CAL v1) or
ASSUMED (a stated value, no data).

1. **Cameras (CAL).** One world = one CAL v1 draw per camera (index keyed by (world_seed, camera); the framings of
   one camera share it): that draw is the world's TRUE camera (k1, tilt, roll, height, H). The observer keeps the
   NOMINAL calibration (twin/site.yaml). A true floor point is projected to raw pixels through the true camera and
   read back to the floor through the nominal one, so every world carries its own calibration error (scale included;
   `h_mode="fixed_scale"` uses the same draws at the 4 m prior instead).
2. **Frustum and visible floor.** A mover is seen only when it stands on its zone's floor polygon (site.yaml, the
   floor the observer believes visible) and its foot pixel lies inside the image of one of the zone's framings, in
   front of the camera. Zone 9 is the union of framings c9_f0 and c9_f1 (as in site.yaml): a point inside the traced
   c9_f0 floor (twin/cal/site_trace_v1.json) is seen through c9_f0, else through c9_f1. A mover OFF the visible floor
   (zone None, or out of its framings' floor frustum, e.g. a truncated person below the image bottom) within
   OFFFLOOR_MAX_M = 5 m (ASSUMED) of a zone's floor polygon or press footprint, whose box overlaps one of that zone's
   framing images, is an IMAGE-ONLY occluder: it has its own miss chain and visibility, occludes other movers and the
   ram ROIs, but gives no floor detection (the observer models the visible floor only). Real zone-9 ROI occluders
   stand there (twin/stat_v1_roi_occluders.json, TRAIN boxes over the ROI: P9a c9_f0 70 % of the feet in the
   footprint, 23 % on the floor, 43 % cut by the image bottom; P9b c9_f1 70 % in the footprint, 30 % off the floor,
   0 % on it).
3. **Occlusion by footprints and movers (image space).** Each mover is a box from its foot circle to its height
   (person 1.75 m, forklift 2.2 m, robot 0.8 m; ASSUMED), each press a prism of its footprint (PRESS_H_M = 3.0 m,
   ASSUMED). The visible share of a mover's image box = 1 - the share covered by closer objects (raster). Visible
   share >= 0.8 (labels/RULES.md: "occluded" = more than ~20 % hidden): clean recall; 0.2-0.8: the occluded recall
   (FITTED ratio occluded / clean, per camera); < 0.2: hidden. A mover inside a press footprint stands at that
   press's working opening: that press's own prism does not hide it (ASSUMED).
4. **Misses (FITTED, per camera).** A 2 Hz Gilbert chain per mover: stationary miss = 1 - clean recall (DEV labels,
   per camera x class, Jeffreys posterior mean: camera 9 person 27/27, camera 14 person 13/15; the forklift has
   labels on camera 14 only (11/17), used on both cameras, ASSUMED on camera 9; the robot uses the person value,
   ASSUMED); occluded / clean ratio from the camera's own person labels (camera 9 15/55, camera 14 17/22); stay-in-miss
   probability from the DEV track gaps and re-births per camera. Nothing is pooled across cameras. Occlusion is an extra miss on top of the chain (keyed uniform), never a change of the chain.
5. **Foot-point noise (FITTED + SHIM).** Pixel error from DEV labels vs DEV tracks: median bias and robust sigma per
   axis PER CAMERA (camera 9 n = 27: sigma u / v 4.7 / 8.9 px; camera 14 n = 13: 9.6 / 2.6 px), AR(1) in time with rho per camera (DEV tracks). The noisy raw pixel is read through the nominal H; then a
   floor AR(1) term tops the floor covariance S (local Jacobian of the nominal map times the pixel sigma) up in S's
   eigenbasis (`floor_topup`): every eigen-direction with sigma < MEASURED_SHIM's 0.285 m is raised to 0.285 m, so
   sigma >= 0.285 m in every direction (per axis included) at every point.
6. **Gross errors (SHIM + ASSUMED persistence).** MEASURED_SHIM's gross_p (stationary share 0.15) and radial
   U(5, 20) m offset, on every class, as persistent episodes (mean GROSS_MEAN_S = 6 s, M-W2 "2-10 s"; ASSUMED): the
   offset is drawn when an episode starts and kept while it lasts.
7. **False positives (SHIM).** MEASURED_SHIM's fp_ratio per true detection, for EVERY emitted class (person, vehicle,
   other), per zone; fp_near of them 0.5-3 m from a true detection of that class, the rest uniform on the zone floor.
8. **Classes.** Person -> person; forklift -> vehicle on camera 14 (lead decision: RF-DETR "truck" >= 0.5, camera 14
   only); on camera 9 there is no vehicle class, so a forklift there is seen as its driver (class person: the P5
   suppression needs a forklift box; ASSUMED); robot -> other (the real cameras never saw a robot). Class confusion
   person <-> vehicle on camera 14 (FITTED, Jeffreys posterior mean on the DEV matches).
9. **ID switches (FITTED).** Per tick with the DEV re-birth rate per camera and class (the robot uses the person
   rate); a switch, or a miss gap longer than the tracker's lost_track_s (3 s), starts a new track: no motion history.
10. **Moving flag and heading from the noisy positions.** speed = |p_k - p_(k-w)| / (w / 2 Hz) over the same track,
    w = up to MOVING_WIN ticks (2 s; ASSUMED); moving = speed > MOVING_MPS (0.2 m/s, as camera_v0); heading =
    atan2(dx, dz) of that displacement while moving, else NaN. The true speed is never used.
11. **ROI energy (FITTED, TRAIN).** zone_motion = ram-ROI energy / the press's median idle energy, sampled from the
    TRAIN quantiles: "moving" while the press runs unfrozen and its ram moves (ram > 0 or a stroke starts this tick),
    "quiet" while it runs with the ram at rest, "idle" otherwise. Camera-14 presses never run in the footage: their
    moving / quiet pools are the pooled P9a + P9b pools (ASSUMED). The press is seen through the framing of its first
    ram ROI. T-W6: when a DETECTED mover's padded image box (ROI_PAD_FRAC) in the ROI's framing covers the ROI (any
    depth, as the real ingest, which has no depth), the sample is MISSING (NaN) this tick, and stays MISSING for `hold_s` after the last such
    tick, in image space, whatever the occluder's floor position (as `occlusion.roi_missing_series` in the real
    ingest; the observer-side floor hold only arms on detections near the press zone). A blocked ROI outside that
    hold (the occluder missed on its first ticks, before any detection) reads the "idle" pool for a standing mover
    (fail-dangerous, as the real pipeline) or the "moving" pool for a walking one: the arrival leak, counted in
    `stats["roi_blocked_unseen"]`. DISCLOSED GAP (zone 9): the DAYGEN v1 path library has no point at a press (0 of
    13,217 zone-9 points in a footprint, all on the visible floor), so in generated days no zone-9 mover stands where
    the real occluders stand; replaying the library through this camera gives ROI-hit shares (unheld) of 0 % for P9a
    and P9b against 29.3 % / 25.1 % in the TRAIN boxes (machines MISSING 38.5 % / 34.1 %); zone 14: 15-34 % against
    10-19 % (twin/stat_v1_roi_occluders.json). The camera handles at-press movers (image-only occluders) once the
    day generator places operators there (S3a). `MOTION_THR` (FITTED, Youden J) is the matching
    observer_live motion threshold.
12. **hold_s (FITTED).** The smallest hold k (ticks) such that k + 1 consecutive ticks with the occluder missed in
    the image (chain miss, or the occluded-regime extra miss) have probability <= OCC_EPS, at the per-camera
    Jeffreys means, maximised over camera x class (camera 9's occluded recall sets it: person 21 s, the forklift
    driver 34 s); hold_s = k / 2 Hz. `hold_detail` also gives each hold at the one-sided 95 % lower recall bounds
    (camera 9 person 34 s, forklift 71 s; disclosed, not used: the twin simulates the mean model). The same hold
    is the camera's image-space ROI hold (item 11) and the converter's floor hold. Gross floor errors do not count:
    the image-space hold runs on the detection in the image, wherever its floor position lands.

Common random numbers: every draw comes from `twin.rng.keyed_rng(world_seed, day, entity, f"tick:{k}")`, k = the
absolute 2 Hz tick, entity `cam:<mover>` (a fixed number of draws per call, whatever the state),
`cam_fp:<zone>:<class>` or `cam_motion:<press>`. A mover's noise never depends on which other movers exist, so the
robot (whose path differs between paired conditions) changes the people's detections only where it physically
occludes them. Markov states (miss, gross, AR noise, track history) are carried across ticks in time order: call
`observe` on every tick of a mission in order.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from intercycle.observer.mapping import FRAME_HZ, LATENCY_S, MEASURED_SHIM
from intercycle.observer.occlusion import OCC_EPS, ROI_PAD_FRAC
from twin.camera_v0 import Detections, detections_to_json
from twin.rng import keyed_rng
from twin.schemas.runs import RealisedMover, RealisedPress, RealisedTick, parse_realised
from twin.schemas.site import Site, point_in_polygon
from twin.truth_guard import open_truth

__all__ = ["CameraStatV1", "Detections", "detections_to_json", "stream", "load_params", "MOTION_THR",
           "hold_ticks_markov"]

REPO = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO / "twin" / "stat_v1_params.json"
HDRAWS_PATH = REPO / "twin" / "cal" / "h_draws_v1.json"
TRACE_PATH = REPO / "twin" / "cal" / "site_trace_v1.json"
SEALED_ENV = "INTERCYCLE_SEALED_RUN"
IMG_W, IMG_H = 1920, 1080

SIGMA_MIN_M = float(cast(float, MEASURED_SHIM["sigma_m"]))      # SHIM: per-axis floor sigma floor (0.285 m)
GROSS_MEAN_S = 6.0                                   # ASSUMED (M-W2: gross errors persist 2-10 s)
PRESS_H_M = 3.0                                      # ASSUMED press body height (occluder prism)
HEIGHT_M = {"person": 1.75, "forklift": 2.2, "robot": 0.8}   # ASSUMED (plant PERSON_H, FORKLIFT_BOXES, ANYmal)
VIS_CLEAN, VIS_HIDDEN = 0.8, 0.2                     # labels/RULES.md "occluded" = > ~20 % hidden; < 20 % visible = hidden
MOVING_MPS = 0.2                                     # as camera_v0
MOVING_WIN = 4                                       # ticks (2 s) for the noisy-position speed; ASSUMED
LOST_TRACK_S = 3.0                                   # intercycle.perception.tracking.TrackerConfig.lost_track_s
RASTER = (8, 16)                                     # occlusion raster (u, v samples) of a mover's image box
CLASS_OUT = {"person": "person", "forklift": "vehicle", "robot": "other"}
OFFFLOOR_MAX_M = 5.0                                 # ASSUMED: off-floor image occluders within this of a zone


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("contract") != "twin.stat_v1_params/1":
        raise ValueError(f"{path}: not a stat_v1 parameter file")
    return d


_P = load_params()
MOTION_THR = float(_P["energy"]["motion_thr"]["value"])   # FITTED: observer_live motion_thr for stat_v1 zone_motion
MOTION_THR_SOURCE = ("stat_v1 (twin/stat_v1_params.json energy.motion_thr): Youden J of normalised TRAIN ROI energy, "
                     "moving frames of P9a/P9b vs idle frames of every press")


# ------------------------------------------------------------------------------------------------ fitted numbers

@dataclass(frozen=True)
class ClassNoise:
    recall_clean: float
    occ_ratio: float          # recall(occluded) / recall(clean)
    p_stay_miss: float        # Gilbert chain at 2 Hz
    idsw_per_s: float
    rho: float                # AR(1) of the foot-point noise at 2 Hz

    @property
    def pi_miss(self) -> float:
        return 1.0 - self.recall_clean

    @property
    def p_enter_miss(self) -> float:
        pi = self.pi_miss
        return pi * (1.0 - self.p_stay_miss) / (1.0 - pi) if pi < 1.0 else 1.0


def class_noise(params: dict, camera: int, cls: str) -> ClassNoise:
    """Fitted PER CAMERA x class (DEV labels; Jeffreys posterior means `labels.recall_jeffreys`, not the pooled
    recalls: camera 9's occluded person recall is 0.28, camera 14's 0.76). Clean recall: the camera's own class
    labels; the forklift has DEV labels on camera 14 only, so its clean recall is camera 14's on both cameras
    (ASSUMED on camera 9, where it is seen as its driver). Occluded / clean ratio: the camera's own person labels, for
    every class (forklift: 3 occluded labels, too few; ASSUMED). The robot borrows the person values (ASSUMED)."""
    lab, trk = params["labels"], params["tracks"]
    rj = lab["recall_jeffreys"]
    pc, po = rj[f"c{camera}_person_clean"]["mean"], rj[f"c{camera}_person_occluded"]["mean"]
    recall = rj["c14_forklift_clean"]["mean"] if cls == "forklift" else pc
    g = trk["gaps"][str(camera)]
    key = f"c{camera}_{'forklift' if cls == 'forklift' else 'person'}"
    idsw = trk["id_switch"].get(key) or trk["id_switch"][f"c{camera}_person"]
    return ClassNoise(recall_clean=float(recall), occ_ratio=float(min(po / pc, 1.0)),
                      p_stay_miss=float(g["p_stay_miss"]), idsw_per_s=float(idsw["per_s"]),
                      rho=float(trk["ar1"][str(camera)]["rho"]))


def class_noise_bound(params: dict, camera: int, cls: str) -> ClassNoise:
    """`class_noise` at the one-sided 95 % Jeffreys LOWER recall bounds (disclosure of the hold's sensitivity): the
    clean recall and the occluded recall at their `lo95`, so the occluded regime detects with probability lo95."""
    rj = params["labels"]["recall_jeffreys"]
    cn = class_noise(params, camera, cls)
    pc, po = rj[f"c{camera}_person_clean"]["lo95"], rj[f"c{camera}_person_occluded"]["lo95"]
    rc = rj["c14_forklift_clean"]["lo95"] if cls == "forklift" else pc
    return ClassNoise(recall_clean=float(rc), occ_ratio=float(min(po / pc, 1.0)), p_stay_miss=cn.p_stay_miss,
                      idsw_per_s=cn.idsw_per_s, rho=cn.rho)


def hold_ticks_markov(cn: ClassNoise, extra_miss: float, eps: float = OCC_EPS, k_max: int = 2000) -> int:
    """Smallest k >= 0 such that k + 1 consecutive ticks on which the occluder is missed IN THE IMAGE have probability
    <= eps (stationary start): the Gilbert miss chain `cn`, plus, on a tick the chain detects, an iid miss with
    probability `extra_miss` (the occluded regime). Gross floor errors do not count: a detected occluder makes the
    camera's own image-space ROI rule return MISSING wherever its floor position lands (item 11)."""
    pe, ps = cn.p_enter_miss, cn.p_stay_miss
    P = np.array([[1 - pe, pe], [1 - ps, ps]])
    pi = np.array([1 - cn.pi_miss, cn.pi_miss])
    b = np.array([extra_miss, 1.0])
    a = pi * b
    for k in range(k_max):
        if a.sum() <= eps:
            return k
        a = (a @ P) * b
    raise ValueError("hold does not converge (miss process too persistent)")


# ------------------------------------------------------------------------------------------------ geometry

def _undistort(pts: np.ndarray, k1: float, c: np.ndarray) -> np.ndarray:
    """Division model, distorted -> undistorted (as intercycle.framing.undistort; numpy only for the Isaac side)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    d = p - c
    return c + d / (1.0 + k1 * (d ** 2).sum(1, keepdims=True))


def _distort(pts: np.ndarray, k1: float, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted -> distorted (exact inverse of `_undistort` on its valid branch, as intercycle.framing.distort);
    returns (points, ok) with ok False outside the model's valid range."""
    p = np.asarray(pts, float).reshape(-1, 2)
    d = p - c
    r_u = np.sqrt((d ** 2).sum(1))
    out = p.copy()
    ok = np.ones(len(p), bool)
    if abs(k1) < 1e-18:
        return out, ok
    disc = 1.0 - 4.0 * k1 * r_u ** 2
    ok = disc >= 0
    nz = ok & (r_u > 1e-12)
    r_d = 2.0 * r_u[nz] / (1.0 + np.sqrt(disc[nz]))
    out[nz] = c + d[nz] * (r_d / r_u[nz])[:, None]
    out[~ok] = np.nan
    return out, ok


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, float).T
    out = np.full((len(p), 2), np.nan)
    ok = np.abs(q[:, 2]) > 1e-12
    out[ok] = q[ok, :2] / q[ok, 2:3]
    return out


@dataclass
class CamGeom:
    """One framing's pinhole camera, rebuilt from (K, k1, H): H maps undistorted px -> plan floor (x, z).
    K^-1 H^-1 = lambda [r1 r2 t] (the floor plane's two axes and the camera translation); the third axis is the
    floor normal, signed so that the camera centre is above the floor. Plan points are (x, z, height)."""
    fid: str
    K: np.ndarray
    k1: float
    H: np.ndarray
    M: np.ndarray = field(init=False)          # 3x3: camera coords = M @ (x, z, height) + t
    t: np.ndarray = field(init=False)
    nadir: np.ndarray = field(init=False)
    height: float = field(init=False)
    c: np.ndarray = field(init=False)

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        self.H = np.asarray(self.H, float) / self.H[2, 2]
        self.c = self.K[:2, 2].copy()
        A = np.linalg.inv(self.K) @ np.linalg.inv(self.H)
        lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
        if (lam * A[:, 2])[2] < 0:                 # the floor origin must lie in front of the camera
            lam = -lam
        r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
        r3 = np.cross(r1, r2)
        r3 /= np.linalg.norm(r3)
        M = np.column_stack([r1, r2, r3])
        centre = -np.linalg.solve(M, t)
        if centre[2] < 0:                          # the plan frame (x, z, up) is left-handed: flip the normal
            M = np.column_stack([r1, r2, -r3])
            centre = -np.linalg.solve(M, t)
        self.M, self.t = M, t
        self.nadir, self.height = centre[:2].copy(), float(centre[2])

    @classmethod
    def from_rec(cls, fid: str, K, r: dict) -> "CamGeom":
        return cls(fid, np.asarray(K, float), float(r["k1"]), np.asarray(r["H"], float))

    def project(self, xzy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, 3) plan points (x, z, height) -> (raw px (n, 2), ok (n,)): ok = in front of the camera and inside the
        division model's valid range."""
        P = np.asarray(xzy, float).reshape(-1, 3)
        cam = P @ self.M.T + self.t
        ok = cam[:, 2] > 1e-6
        und = np.full((len(P), 2), np.nan)
        if ok.any():
            q = (self.K @ (cam[ok] / cam[ok, 2:3]).T).T
            und[ok] = q[:, :2]
        raw = np.full_like(und, np.nan)
        if ok.any():
            r, okd = _distort(und[ok], self.k1, self.c)
            raw[ok] = r
            ok[np.flatnonzero(ok)[~okd]] = False
        return raw, ok

    def floor_of(self, raw: np.ndarray) -> np.ndarray:
        """Raw px -> plan floor (x, z) through this camera's lens and H."""
        return _apply_h(self.H, _undistort(raw, self.k1, self.c))


def _in_image(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, float).reshape(-1, 2)
    return np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)


def _pip_px(u: float, v: float, poly: np.ndarray) -> bool:
    return point_in_polygon(u, v, [(float(a), float(b)) for a, b in poly])


def _dist_poly(x: float, z: float, poly) -> float:
    """Distance (m) from a plan point to a polygon (0 inside)."""
    if point_in_polygon(x, z, poly):
        return 0.0
    p = np.asarray(poly, float).reshape(-1, 2)
    a, b = p, np.roll(p, -1, axis=0)
    ab = b - a
    q = np.array([x, z], float)
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return float(np.min(np.hypot(*(q - (a + t[:, None] * ab)).T)))


def _bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _circle_prism(x: float, z: float, r: float, hgt: float, n: int = 8) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([x + r * np.sin(a), z + r * np.cos(a)])
    return np.vstack([np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, hgt)])])


def _poly_prism(poly, hgt: float) -> np.ndarray:
    p = np.asarray(poly, float).reshape(-1, 2)
    return np.vstack([np.column_stack([p, np.zeros(len(p))]), np.column_stack([p, np.full(len(p), hgt)])])


def floor_topup(S: np.ndarray, sigma_min: float) -> np.ndarray:
    """Top-up matrix T (2x2) for a floor covariance S, in S's eigenbasis: S = V diag(l) V^T,
    T = V diag(sqrt(max(sigma_min^2 - l_i, 0))) V^T, so S + T T^T = V diag(max(l_i, sigma_min^2)) V^T has
    sigma >= sigma_min in EVERY direction (per axis included). Applied as T @ e with e ~ N(0, I_2)."""
    S = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
    lam, V = np.linalg.eigh(S)
    add = np.sqrt(np.maximum(sigma_min ** 2 - lam, 0.0))
    return V @ np.diag(add) @ V.T


def _quantile_sample(q: list[float], u: float) -> float:
    qq = np.asarray(q, float)
    return float(np.interp(u, np.linspace(0.0, 1.0, len(qq)), qq))


# ------------------------------------------------------------------------------------------------ the camera

@dataclass
class _Track:
    miss: bool
    gross: bool
    off: tuple[float, float]
    e_px: np.ndarray            # AR(1) state of the pixel noise (unit variance, u and v)
    e_fl: np.ndarray            # AR(1) state of the floor top-up (unit variance, x and z)
    k_last: int                 # last tick the chain advanced
    k_seen: int                 # last tick the mover was detected
    hist: deque                 # (k, x, z) noisy detections of the current track


class CameraStatV1:
    """stat_v1: realised tick -> Detections (same container as camera_v0, so the plant's ObsAdapter is unchanged)."""

    def __init__(self, site: Site, world_seed: int, day: int = 1, params: dict | None = None,
                 hdraws: dict | None = None, trace: dict | None = None, h_mode: str = "joint",
                 latency_s: float = LATENCY_S, hz: float = FRAME_HZ, shim: dict[str, Any] | None = None):
        if h_mode not in ("joint", "fixed_scale"):
            raise ValueError(f"h_mode must be 'joint' or 'fixed_scale', got {h_mode!r}")
        self.site, self.world_seed, self.day = site, int(world_seed), int(day)
        self.P = _P if params is None else params
        hd = json.loads(HDRAWS_PATH.read_text(encoding="utf-8")) if hdraws is None else hdraws
        tr = json.loads(TRACE_PATH.read_text(encoding="utf-8")) if trace is None else trace
        s: dict[str, Any] = dict(MEASURED_SHIM if shim is None else shim)
        self.gross_p = float(s["gross_p"])
        self.gross_lo, self.gross_hi = float(s["gross_range_m"][0]), float(s["gross_range_m"][1])
        self.fp_ratio, self.fp_near = float(s["fp_ratio"]), float(s["fp_near"])
        self.latency, self.hz, self.h_mode = latency_s, hz, h_mode
        self.p_exit_gross = 1.0 / (GROSS_MEAN_S * hz)
        self.p_enter_gross = self.gross_p * self.p_exit_gross / (1.0 - self.gross_p)
        foot = self.P["labels"]["foot_px"]["per_camera"]          # FITTED per camera (DEV labels vs DEV tracks)
        self.sig_px = {int(c): np.array([v["sigma_u_px"], v["sigma_v_px"]], float) for c, v in foot.items()}
        self.bias_px = {int(c): np.array([v["bias_u_px"], v["bias_v_px"]], float) for c, v in foot.items()}
        conf = self.P["labels"]["confusion"]
        self.c_pv, self.c_vp = float(conf["person_as_vehicle"]), float(conf["vehicle_as_person"])
        self.zones = {z.id: z for z in site.zones}
        fr_site = {f.id: f for f in site.framings}
        # the observer's (nominal) calibration = the site's framings; the world's true camera = one CAL v1 draw per
        # camera. A site whose framings have no CAL v1 draws (the G0 stub site_v0) gets true = nominal (no calibration
        # error, `cal_source` says so); a sealed run refuses it (fail closed).
        self.nominal: dict[str, CamGeom] = {}
        self.true: dict[str, CamGeom] = {}
        self.draw_index: dict[int, int] = {}
        key = "draws" if h_mode == "joint" else "draws_fixed_scale"
        have = all(fid in hd["framings"] for z in site.zones for fid in z.framings)
        if not have and os.environ.get(SEALED_ENV) == "1":
            raise ValueError("sealed run: every framing needs CAL v1 draws (twin/cal/h_draws_v1.json)")
        self.cal_source = f"cal_v1 {key}" if have else "site framings (no CAL v1 draws: no calibration error)"
        for z in site.zones:
            cam = int(z.camera)
            i = int(keyed_rng(self.world_seed, 0, f"cam_h_draw:{cam}", "cal_v1").integers(int(hd["n"])))
            if have:
                self.draw_index[cam] = i
            for fid in z.framings:
                f = fr_site[fid]
                g = CamGeom(fid, np.asarray(f.K.value, float), float(f.k1.value),
                            np.asarray(f.H_img_to_floor.value, float))
                self.nominal[fid] = g
                if not have:
                    self.true[fid] = g
                    continue
                nom = hd["framings"][fid]["nominal"]
                if abs(float(nom["k1"]) - g.k1) > 1e-15 or not np.allclose(np.asarray(nom["H"], float), g.H,
                                                                           rtol=1e-6, atol=1e-9):
                    raise ValueError(f"{fid}: h_draws_v1.json nominal camera != the site's (re-run stat_v1_hdraws)")
                self.true[fid] = CamGeom.from_rec(fid, f.K.value, hd["framings"][fid][key][i])
        self.floor_px = {fid: np.asarray(tr["floor"][fid], float) for fid in self.nominal if fid in tr["floor"]}
        self.noise = {(int(z.camera), c): class_noise(self.P, int(z.camera), c)
                      for z in site.zones for c in ("person", "forklift", "robot")}
        self.cam_of_zone = {z.id: int(z.camera) for z in site.zones}
        self.cam_of_framing = {fid: int(z.camera) for z in site.zones for fid in z.framings}
        # presses: occluder prisms, ROI framing / box, energy pools
        en = self.P["energy"]["presses"]
        self.presses = {p.id: p for p in site.presses}
        self.roi: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
        for p in site.presses:
            if p.id not in en:
                raise ValueError(f"press {p.id}: no ROI-energy pools in stat_v1_params.json")
            if p.ram_roi.value:                  # no ROI: zone_motion is MISSING (NaN) for that press, always
                r0 = p.ram_roi.value[0]
                self.roi[p.id] = (r0.framing, _bbox(np.asarray(r0.polygon_px, float)))
        self.energy = en
        # zone_vis of an ROI-sourced press = the share of its ROI box inside its framing's image (ObsAdapter roi_vis)
        self.roi_vis: dict[str, dict[int, float]] = {}
        for pid, (fid, (r0, s0, r1, s1)) in self.roi.items():
            area = max(r1 - r0, 0.0) * max(s1 - s0, 0.0)
            inside = max(min(r1, IMG_W) - max(r0, 0.0), 0.0) * max(min(s1, IMG_H) - max(s0, 0.0), 0.0)
            self.roi_vis[pid] = {self.cam_of_framing[fid]: (inside / area) if area > 0 else 0.0}
        self.press_centre = {p.id: np.asarray(p.footprint.value, float).mean(0) for p in site.presses}
        self.press_prism = {p.id: _poly_prism(p.footprint.value, PRESS_H_M) for p in site.presses}
        self.zone_presses = {z.id: [p.id for p in site.presses if p.zone == z.id] for z in site.zones}
        self.tracks: dict[str, _Track] = {}
        self._pboxes: dict[str, list[tuple[tuple, float, str]]] | None = None
        self.hold_s = self._hold_s()
        self.hold_ticks = int(round(self.hold_s * self.hz))
        self.roi_last_hit: dict[str, int] = {}       # press -> last tick a DETECTED occluder covered its ROI (image)
        self.err_log: list[tuple[str, float, float]] | None = None   # tests: (mover, dx, dz) per true detection
        self.stats = {"ticks": 0, "true": 0, "seen": 0, "hidden": 0, "occluded": 0, "missed": 0, "gross": 0,
                      "fp": 0, "idsw": 0, "forklift_as_person_c9": 0, "roi_missing": 0, "roi_held": 0,
                      "roi_blocked_unseen": 0, "image_only": 0}

    # ---------------------------------------------------------------- fitted hold
    def _hold_s(self) -> float:
        """max over camera x class of the hold at the per-camera Jeffreys means (the simulated miss process). The hold
        at the one-sided 95 % lower recall bounds is kept in `hold_detail` for disclosure, not used."""
        self.hold_detail: dict[str, dict[str, float]] = {}
        ks = []
        for (cam, cls), cn in self.noise.items():
            k = hold_ticks_markov(cn, 1.0 - cn.occ_ratio)
            ks.append(k)
            lo = class_noise_bound(self.P, cam, cls)
            self.hold_detail[f"c{cam}_{cls}"] = {"hold_s": k / self.hz,
                                                 "hold_s_at_lo95": hold_ticks_markov(lo, 1.0 - lo.occ_ratio) / self.hz}
        return max(ks) / self.hz

    # ---------------------------------------------------------------- geometry
    def _framing_of(self, zone_id: str, x: float, z: float) -> tuple[str, np.ndarray] | None:
        """(framing, raw foot px through the TRUE camera) of a floor point of the zone, or None if out of frustum."""
        cand = []
        for fid in self.zones[zone_id].framings:
            raw, ok = self.true[fid].project(np.array([[x, z, 0.0]]))
            if ok[0] and _in_image(raw)[0]:
                if fid in self.floor_px and _pip_px(float(raw[0, 0]), float(raw[0, 1]), self.floor_px[fid]):
                    return fid, raw[0]
                cand.append((fid, raw[0]))
        return cand[0] if cand else None

    def _image_only(self, m: RealisedMover) -> tuple[str, str, tuple[float, float, float, float]] | None:
        """(zone, framing, image box) of a mover OFF the visible floor (or out of its framings' floor frustum) that
        stands within OFFFLOOR_MAX_M of a zone's floor polygon or press footprint (the nearest such zone) and whose
        box (through the TRUE camera, every prism point in front) overlaps one of that zone's framing images; else
        None. Such a mover can occlude the ram ROI and other movers in the image, but gives no floor detection."""
        best: tuple[float, str] | None = None
        for zid, zo in self.zones.items():
            d = min([_dist_poly(m.x, m.z, zo.floor_polygon.value)]
                    + [_dist_poly(m.x, m.z, self.presses[p].footprint.value) for p in self.zone_presses[zid]])
            if d <= OFFFLOOR_MAX_M and (best is None or d < best[0]):
                best = (d, zid)
        if best is None:
            return None
        zid = best[1]
        prism = _circle_prism(m.x, m.z, m.r, HEIGHT_M[m.cls])
        for fid in self.zones[zid].framings:
            b = self._box(fid, prism)
            if b is not None and b[2] > 0 and b[0] < IMG_W and b[3] > 0 and b[1] < IMG_H:
                return zid, fid, b
        return None

    def _box(self, fid: str, prism: np.ndarray) -> tuple[float, float, float, float] | None:
        raw, ok = self.true[fid].project(prism)
        if not ok.all():
            return None
        return _bbox(raw)

    def _press_boxes(self) -> dict[str, list[tuple[tuple, float, str]]]:
        """Image boxes, depths and ids of the press prisms per framing (static: computed once per camera)."""
        if self._pboxes is None:
            out: dict[str, list[tuple[tuple, float, str]]] = {}
            for pid, prism in self.press_prism.items():
                for fid in self.zones[self.presses[pid].zone].framings:
                    b = self._box(fid, prism)
                    if b is not None:
                        out.setdefault(fid, []).append((b, self._depth(fid, self.press_centre[pid]), pid))
            self._pboxes = out
        return self._pboxes

    def _depth(self, fid: str, xz: Any) -> float:
        return float(np.hypot(*(np.asarray(xz, float) - self.true[fid].nadir)))

    def _visible_share(self, fid: str, box, depth: float, others: list[tuple[tuple, float]]) -> float:
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return 1.0
        u = np.linspace(x0, x1, RASTER[0])
        v = np.linspace(y0, y1, RASTER[1])
        U, V = np.meshgrid(u, v)
        cov = np.zeros(U.shape, bool)
        for (a0, b0, a1, b1), d in others:
            if d < depth:
                cov |= (U >= a0) & (U <= a1) & (V >= b0) & (V <= b1)
        return 1.0 - float(cov.mean())

    # ---------------------------------------------------------------- per mover
    def _rng(self, entity: str, k: int) -> np.random.Generator:
        return keyed_rng(self.world_seed, self.day, entity, f"tick:{k}")

    def _advance(self, m: RealisedMover, cn: ClassNoise, k: int, g_u, g_n, rr: float, aa: float) -> _Track:
        """Advance the mover's Markov states to tick k (miss chain, gross episode, AR noises)."""
        tk = self.tracks.get(m.id)
        if tk is None or k - tk.k_last != 1:                     # first tick (or a gap in presence): stationary start
            miss = bool(g_u[0] < cn.pi_miss)
            gross = bool(g_u[2] < self.gross_p)
            tk = _Track(miss, gross, (rr * math.sin(aa), rr * math.cos(aa)), g_n[0:2].copy(), g_n[2:4].copy(), k,
                        -10 ** 9, deque(maxlen=MOVING_WIN + 1))
            self.tracks[m.id] = tk
            return tk
        tk.miss = bool(g_u[0] < (cn.p_stay_miss if tk.miss else cn.p_enter_miss))
        if tk.gross:
            tk.gross = not (g_u[2] < self.p_exit_gross)
        elif g_u[2] < self.p_enter_gross:
            tk.gross = True
            tk.off = (rr * math.sin(aa), rr * math.cos(aa))
        r = cn.rho
        s = math.sqrt(max(1.0 - r * r, 0.0))
        tk.e_px = r * tk.e_px + s * g_n[0:2]
        tk.e_fl = r * tk.e_fl + s * g_n[2:4]
        tk.k_last = k
        return tk

    def _floor_cov(self, fid: str, raw: np.ndarray) -> np.ndarray:
        """2x2 floor covariance (m^2, plan x / z) of the pixel noise at this pixel through the nominal map (local
        Jacobian of the nominal lens + H times the camera's pixel sigma)."""
        nom = self.nominal[fid]
        d = 1.0
        p0 = nom.floor_of(raw)[0]
        pu = nom.floor_of(raw + np.array([d, 0.0]))[0]
        pv = nom.floor_of(raw + np.array([0.0, d]))[0]
        J = np.column_stack([(pu - p0) / d, (pv - p0) / d])
        return J @ np.diag(self.sig_px[self.cam_of_framing[fid]] ** 2) @ J.T

    # ---------------------------------------------------------------- one tick
    def observe(self, tick: RealisedTick) -> Detections:
        k = int(round(tick.t * self.hz))
        self.stats["ticks"] += 1
        seen = []                                       # (mover, zone, framing, raw foot | None, box, depth)
        boxes: dict[str, list[tuple[tuple, float, str]]] = {}            # framing -> [(box, depth, id)]
        for m in tick.movers:
            on_floor = (m.zone is not None and m.zone in self.zones
                        and point_in_polygon(m.x, m.z, self.zones[m.zone].floor_polygon.value))
            fo = self._framing_of(str(m.zone), m.x, m.z) if on_floor else None
            if fo is not None:
                fid, raw = fo
                box = self._box(fid, _circle_prism(m.x, m.z, m.r, HEIGHT_M[m.cls]))
                if box is None:
                    continue
                zid: str = str(m.zone)
            else:                                        # off the visible floor: an image-only occluder (item 3)
                io = self._image_only(m)
                if io is None:
                    continue
                zid, fid, box = io
                raw = None
                self.stats["image_only"] += 1
            dep = self._depth(fid, (m.x, m.z))
            seen.append((m, zid, fid, raw, box, dep))
            boxes.setdefault(fid, []).append((box, dep, m.id))
        press_boxes = self._press_boxes()
        cls_l: list[str] = []
        xs: list[float] = []
        zs: list[float] = []
        rs: list[float] = []
        hs: list[float] = []
        mv: list[bool] = []
        zone_l: list[str] = []
        detected: dict[str, bool] = {}
        present = set()
        for m, zid, fid, raw, box, dep in seen:
            present.add(m.id)
            if raw is not None:
                self.stats["true"] += 1
            cam = self.cam_of_zone[zid]
            cn = self.noise[(cam, m.cls)]
            g = self._rng(f"cam:{m.id}", k)
            g_u = g.random(5)                                   # miss, occlusion, gross, class, id switch
            g_n = g.standard_normal(5)                          # pixel u, v; floor x, z; radius
            rr, aa = float(g.uniform(self.gross_lo, self.gross_hi)), float(g.uniform(0.0, 2 * math.pi))
            tk = self._advance(m, cn, k, g_u, g_n, rr, aa)
            # a mover inside a press footprint stands at that press's working opening: the press's own prism box does
            # not hide it (ASSUMED; the real zone-9 ROI occluders' feet project into the footprint and are detected)
            others = [(b, d) for b, d, oid in boxes.get(fid, []) if oid != m.id] + [
                (b, d) for b, d, pid in press_boxes.get(fid, [])
                if not point_in_polygon(m.x, m.z, self.presses[pid].footprint.value)]
            vis = self._visible_share(fid, box, dep, others)
            if vis < VIS_HIDDEN:
                self.stats["hidden"] += 1
                detected[m.id] = False
                continue
            if tk.miss:
                self.stats["missed"] += 1
                detected[m.id] = False
                continue
            if vis < VIS_CLEAN:
                self.stats["occluded"] += 1
                if g_u[1] >= cn.occ_ratio:
                    detected[m.id] = False
                    continue
            detected[m.id] = True
            if raw is None:
                continue                                        # image-only: occludes the ROI, no floor detection
            # class
            if m.cls == "forklift" and cam != 14:
                out_cls = "person"                              # no vehicle class on camera 9: the driver is seen
                self.stats["forklift_as_person_c9"] += 1
            else:
                out_cls = CLASS_OUT[m.cls]
                if cam == 14 and out_cls == "person" and g_u[3] < self.c_pv:
                    out_cls = "vehicle"
                elif cam == 14 and out_cls == "vehicle" and g_u[3] < self.c_vp:
                    out_cls = "person"
            # position: true camera -> noisy raw px -> nominal H; floor top-up to SIGMA_MIN_M
            noisy = raw + self.bias_px[cam] + self.sig_px[cam] * tk.e_px
            if not _in_image(noisy)[0]:
                noisy = np.clip(noisy, [0.0, 0.0], [IMG_W - 1.0, IMG_H - 1.0])
            x, z = self.nominal[fid].floor_of(noisy)[0]
            T = floor_topup(self._floor_cov(fid, noisy), SIGMA_MIN_M)
            dxz = T @ tk.e_fl
            x, z = float(x + dxz[0]), float(z + dxz[1])
            if tk.gross:
                self.stats["gross"] += 1
                x, z = m.x + tk.off[0], m.z + tk.off[1]
            if self.err_log is not None:
                self.err_log.append((m.id, x - m.x, z - m.z))
            # track identity: ID switch or a gap longer than the tracker's buffer starts a new track
            if (k - tk.k_seen) / self.hz > LOST_TRACK_S or g_u[4] < cn.idsw_per_s / self.hz:
                if tk.hist:
                    self.stats["idsw"] += 1
                tk.hist.clear()
            tk.hist.append((k, x, z))
            tk.k_seen = k
            moving, heading = False, math.nan
            if len(tk.hist) >= 2:
                k0, x0, z0 = tk.hist[0]
                dx, dz = x - x0, z - z0
                spd = math.hypot(dx, dz) / ((k - k0) / self.hz)
                if spd > MOVING_MPS:
                    moving, heading = True, math.atan2(dx, dz)
            cls_l.append(out_cls)
            xs.append(x)
            zs.append(z)
            rs.append(max(m.r + 0.03 * float(g_n[4]), 0.1))
            hs.append(heading)
            mv.append(moving)
            zone_l.append(zid)
            self.stats["seen"] += 1
        for mid in list(self.tracks):                                   # movers gone from view: the chain restarts
            if mid not in present and k - self.tracks[mid].k_last > 1:
                del self.tracks[mid]
        n_true = len(cls_l)
        self._false_positives(k, cls_l, xs, zs, rs, hs, mv, zone_l, n_true)
        motion = self._zone_motion(tick, k, seen, detected)
        return Detections(t=tick.t + self.latency, t_cap=tick.t, cls=np.array(cls_l, dtype=str),
                          x=np.array(xs, float), z=np.array(zs, float), r=np.array(rs, float),
                          heading=np.array(hs, float), moving=np.array(mv, bool), zone=np.array(zone_l, dtype=str),
                          zone_motion=motion, cam_ok={zz.camera: True for zz in self.site.zones})

    def _false_positives(self, k, cls_l, xs, zs, rs, hs, mv, zone_l, n_true) -> None:
        for zid, zo in self.zones.items():
            cam = self.cam_of_zone[zid]
            for c in ("person", "vehicle", "other"):
                if c == "vehicle" and cam != 14:
                    continue                                    # no vehicle class on camera 9
                src = [i for i in range(n_true) if cls_l[i] == c and zone_l[i] == zid]
                if not src:
                    continue
                g = self._rng(f"cam_fp:{zid}:{c}", k)
                for _ in range(int(g.poisson(self.fp_ratio * len(src)))):
                    if g.random() < self.fp_near:
                        i = src[int(g.integers(len(src)))]
                        dd, a = g.uniform(0.5, 3.0), g.uniform(0.0, 2 * math.pi)
                        px, pz = xs[i] + dd * math.sin(a), zs[i] + dd * math.cos(a)
                    else:
                        px, pz = self._floor_point(g, zo.floor_polygon.value)
                    cls_l.append(c)
                    xs.append(float(px))
                    zs.append(float(pz))
                    rs.append(0.3)
                    hs.append(math.nan)
                    mv.append(False)
                    zone_l.append(zid)
                    self.stats["fp"] += 1

    @staticmethod
    def _floor_point(g: np.random.Generator, poly) -> tuple[float, float]:
        xs_, zs_ = [p[0] for p in poly], [p[1] for p in poly]
        x0, x1, z0, z1 = min(xs_), max(xs_), min(zs_), max(zs_)
        for _ in range(1000):
            x, z = float(g.uniform(x0, x1)), float(g.uniform(z0, z1))
            if point_in_polygon(x, z, poly):
                return x, z
        return (x0 + x1) / 2, (z0 + z1) / 2

    def _zone_motion(self, tick: RealisedTick, k: int, seen, detected: dict[str, bool]) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in tick.presses:
            if p.id not in self.roi:
                out[p.id] = math.nan
                continue
            fid, (r0, s0, r1, s1) = self.roi[p.id]
            dep_p = self._depth(fid, self.press_centre[p.id])
            u = float(self._rng(f"cam_motion:{p.id}", k).random())
            over, blockers = [], []                     # boxes over the ROI (any depth) / those in front of the press
            pz = self.presses[p.id].zone
            for m, zid, mfid, _raw, box, dep in seen:
                if zid != pz:
                    continue
                if mfid != fid:                          # the mover's box in the ROI's framing (zone 9: two framings)
                    b = self._box(fid, _circle_prism(m.x, m.z, m.r, HEIGHT_M[m.cls]))
                    if b is None:
                        continue
                    box, dep = b, self._depth(fid, (m.x, m.z))
                x0, y0, x1, y1 = box
                pu, pv = ROI_PAD_FRAC * (x1 - x0), ROI_PAD_FRAC * (y1 - y0)
                if x0 - pu <= r1 and x1 + pu >= r0 and y0 - pv <= s1 and y1 + pv >= s0:
                    over.append(m)
                    if dep < dep_p:
                        blockers.append(m)
            pool = self._pool(p)
            if any(detected.get(m.id, False) for m in over):    # a detected box over the ROI, as the real ingest
                out[p.id] = math.nan                              # T-W6: a detected occluder makes the ROI MISSING
                self.roi_last_hit[p.id] = k
                self.stats["roi_missing"] += 1
                continue
            if k - self.roi_last_hit.get(p.id, -10 ** 9) <= self.hold_ticks:
                out[p.id] = math.nan                              # ... held hold_s after the last such tick (image
                self.stats["roi_held"] += 1                       # space, as roi_missing_series in the real ingest)
                continue
            if blockers:
                self.stats["roi_blocked_unseen"] += 1
                pool = "moving" if any(m.speed > MOVING_MPS for m in blockers) else "idle"
            out[p.id] = _quantile_sample(self.energy[p.id][pool]["q"], u)
        return out

    @staticmethod
    def _pool(p: RealisedPress) -> str:
        if p.state != "running" or p.freeze:
            return "idle"
        return "moving" if (p.ram > 1e-9 or p.stroke) else "quiet"

    def run(self, ticks: Iterable[RealisedTick]) -> Iterator[Detections]:
        for tk in ticks:
            yield self.observe(tk)


def stream(path: str | Path, site: Site, world_seed: int, **kw) -> Iterator[Detections]:
    """Read a realised_state.jsonl (truth; logged by the truth guard) and yield stat_v1 detections tick by tick."""
    with open_truth(path, "r") as f:
        header, ticks = parse_realised(f)
        yield from CameraStatV1(site, world_seed, header.day, **kw).run(ticks)
