"""Pixel adapter, spatial only (lane S4 item 5; PLAN_TWIN §4, WEAKNESSES V-W1, M-W10).

Real-clip PERC output -> Observation v2, one framing (= one camera polygon) at a time:

  tracks   $INTERCYCLE_DATA/tracks/<clip>.parquet (foot point u, v in full-res px, PERC esk_ingest)
           -> floor (x, z) through CAL: `esk_ingest.project_tracks` with the framing's k1 and H_img_to_floor from
           twin/site.yaml. A framing without a fitted H gives no floor position and no Observation (fail closed).
  machines $INTERCYCLE_DATA/machines/<press>/<clip>.parquet (ROI motion energy, PERC's image-space occlusion flag)
           -> zone_motion: the ROI energy of the nearest sample, NaN (MISSING) where PERC marked it missing or no
           sample lies within half a tick. Zones = the presses of the framing's site zone, as the plant's ObsAdapter
           (machine zone = danger polygon bbox grown by dilation_m; camera polygon = the zone's floor polygon).
  frames   the clip decoded at 2 Hz -> the M-W10 image watchdog (reference = `reference_from_frames` on the framing's
           TRAIN keyframes) -> floor_ok, ANDed into cam_floor_ok by `mapping.ObsV2Converter` (which also applies the
           T-W6 occlusion hold on the person / vehicle detections).

"Spatial only": positions, machine-zone membership, visibility and MISSING. No phase or map learning from real clips
(PLAN_TWIN §6 cut). Detection radius and heading are not measured from pixels: r is a per-class constant, heading NaN,
moving False.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from intercycle.observer import mapping as OM
from intercycle.observer import watchdog as WD
from intercycle.perception.esk_ingest import project_tracks
from twin.plant import _bbox, floor_share
from twin.schemas.site import Site

TICK_S = 1.0 / OM.FRAME_HZ
WATCH_WIDTH = 960                     # train keyframes are 960 px wide; live frames are resized to it (sharpness scale)
CLASS_CODE = {"person": OM.PERSON, "forklift": OM.VEHICLE, "vehicle": OM.VEHICLE}
CLASS_R_M = {"person": 0.3, "forklift": 1.2, "vehicle": 1.2}   # assumed footprint radii (not measured from pixels)


@dataclass(frozen=True)
class FramingGeom:
    framing: str
    zone: str
    H: np.ndarray | None              # (3, 3) undistorted px -> floor; None = no fitted H
    k1: float
    press_ids: list[str]
    rects: list[tuple[float, float, float, float]]
    share: np.ndarray                 # (M, 1) share of each machine zone's floor inside this camera's floor polygon


def site_geometry(site: Site) -> dict[str, FramingGeom]:
    """{framing id: geometry} from the site contract."""
    fr = {f.id: f for f in site.framings}
    presses = {p.id: p for p in site.presses}
    out: dict[str, FramingGeom] = {}
    for z in site.zones:
        pids = [p.id for p in site.presses if p.zone == z.id]
        rects = [_bbox(presses[p].danger_polygon.value, presses[p].dilation_m.value) for p in pids]
        share = floor_share(rects, [z.floor_polygon.value]) if rects else np.zeros((0, 1))
        for fid in z.framings:
            f = fr[fid]
            H = np.asarray(f.H_img_to_floor.value, float).reshape(3, 3) if f.H_img_to_floor.value is not None else None
            out[fid] = FramingGeom(fid, z.id, H, float(f.k1.value or 0.0), pids, rects, share)
    return out


def project_uv(uv: np.ndarray, geom: FramingGeom) -> np.ndarray:
    """(n, 2) floor (x, z) of full-res pixel points through the framing's k1 + H (project_tracks); NaN without H."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    df = pd.DataFrame({"u": uv[:, 0], "v": uv[:, 1], "framing": geom.framing, "x": np.nan, "y": np.nan})
    H = {} if geom.H is None else {geom.framing: geom.H}
    p = project_tracks(df, H, {geom.framing: geom.k1})
    return p[["x", "y"]].to_numpy(float)


def machine_of(xz: np.ndarray, rects: Sequence[tuple[float, float, float, float]]) -> np.ndarray:
    """(n,) int16 index of the first machine rect containing each floor point, -1 none (NaN points: -1)."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    out = np.full(len(xz), -1, np.int16)
    for j, (x0, z0, x1, z1) in reversed(list(enumerate(rects))):
        inside = (xz[:, 0] >= x0) & (xz[:, 0] <= x1) & (xz[:, 1] >= z0) & (xz[:, 1] <= z1)
        out[inside] = j
    return out


def watch_frame(frame: np.ndarray, width: int = WATCH_WIDTH) -> np.ndarray:
    """Frame resized to the keyframe width (the watchdog's sharpness is resolution-dependent)."""
    import cv2

    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def train_reference(keyframe_paths: Iterable[str | Path], params: WD.WatchdogParams = WD.WatchdogParams()
                    ) -> WD.CamReference | None:
    """Watchdog reference of one framing from its TRAIN keyframes (caller selects train ids); None if none load."""
    import cv2

    frames = []
    for p in keyframe_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            frames.append(watch_frame(im))
    return WD.reference_from_frames(frames, params) if frames else None


def decode_ticks(path: str | Path, t0: float, t1: float, hz: float = OM.FRAME_HZ) -> list[tuple[float, np.ndarray | None]]:
    """The clip's frames at the tick times t0, t0 + 1/hz, ... <= t1 (seek by PTS ms); None where no frame decodes."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    out: list[tuple[float, np.ndarray | None]] = []
    try:
        n = int(math.floor((t1 - t0) * hz + 1e-9)) + 1
        for k in range(max(n, 0)):
            t = t0 + k / hz
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, im = cap.read()
            out.append((t, watch_frame(im) if ok and im is not None else None))
    finally:
        cap.release()
    return out


class PixelAdapter:
    """Observation v2 for ONE framing, n_cam = 1. Stateful (watchdog recovery, T-W6 hold): `reset()` per clip, then
    call `step` on every tick in time order."""

    def __init__(self, geom: FramingGeom, ref: WD.CamReference | None,
                 params: WD.WatchdogParams = WD.WatchdogParams()):
        if geom.H is None:
            raise ValueError(f"framing {geom.framing} has no fitted H: no Observation (PLAN §5)")
        self.geom = geom
        self.wd = WD.CameraWatchdog(params, ref)
        self.conv = OM.ObsV2Converter(geom.share, geom.rects)
        self.last_health: WD.CamHealth | None = None

    def reset(self) -> None:
        self.wd.reset()
        self.conv.reset()

    def step(self, t: float, dets: pd.DataFrame, frame: np.ndarray | None,
             motion: Mapping[str, float]) -> OM.Observation:
        """dets: rows with class, u, v (foot point, full-res px) at this tick; frame: the decoded image (None = none);
        motion: {press id: ROI energy, NaN = MISSING}."""
        g = self.geom
        cls = [str(c) for c in dets["class"]] if len(dets) else []
        keep = [c in CLASS_CODE for c in cls]
        d = dets.loc[keep] if len(dets) else dets
        cls = [c for c, k in zip(cls, keep) if k]
        xz = project_uv(d[["u", "v"]].to_numpy(float), g) if len(d) else np.zeros((0, 2))
        fin = np.isfinite(xz).all(1)
        xz, cls = xz[fin], [c for c, f in zip(cls, fin) if f]
        n = len(cls)
        cam_ok = np.array([frame is not None], bool)
        h = self.wd.update(float(t), frame)
        self.last_health = h
        vis = np.clip(g.share @ cam_ok.astype(float), 0.0, 1.0)
        vis = np.where(vis > 1.0 - 1e-9, 1.0, vis)
        zm = np.array([float(motion.get(p, math.nan)) for p in g.press_ids], np.float64)
        zm[vis < OM.ZONE_VIS_MIN] = np.nan
        obs = OM.Observation(t=float(t), t_cap=float(t), cls=np.array([CLASS_CODE[c] for c in cls], np.int8),
                             x=xz[:, 0].astype(np.float64), z=xz[:, 1].astype(np.float64),
                             r=np.array([CLASS_R_M[c] for c in cls], np.float64), heading=np.full(n, np.nan),
                             moving=np.zeros(n, bool), machine=machine_of(xz, g.rects), cam_ok=cam_ok,
                             zone_motion=zm.astype(np.float32), zone_vis=vis.astype(np.float32),
                             cam_floor_ok=cam_ok.copy())
        return self.conv(obs, floor_ok=np.array([h.ok], bool))


def _nearest_rows(df: pd.DataFrame, t: float, tol: float) -> pd.DataFrame:
    if not len(df):
        return df
    tt = df["t"].to_numpy(float)
    i = int(np.argmin(np.abs(tt - t)))
    if abs(tt[i] - t) > tol:
        return df.iloc[0:0]
    return df.loc[np.isclose(tt, tt[i])]


def motion_at(machines: Mapping[str, pd.DataFrame], t: float, tol: float = TICK_S / 2) -> dict[str, float]:
    """{press: ROI energy of the nearest sample within tol, NaN if missing or none} (fail closed: MISSING)."""
    out: dict[str, float] = {}
    for p, df in machines.items():
        r = _nearest_rows(df, t, tol)
        if not len(r) or bool(r["missing"].iloc[0]) or not np.isfinite(r["roi_energy"].iloc[0]):
            out[p] = math.nan
        else:
            out[p] = float(r["roi_energy"].iloc[0])
    return out


def adapt_clip(adapter: PixelAdapter, tracks: pd.DataFrame, frames: Sequence[tuple[float, np.ndarray | None]],
               machines: Mapping[str, pd.DataFrame], det_tol: float = TICK_S / 2) -> list[OM.Observation]:
    """One clip -> its 2 Hz Observation v2 sequence (ticks at the frame times). Detections = the tracks rows at the
    PERC sample nearest each tick within det_tol (none otherwise)."""
    adapter.reset()
    out = []
    for t, fr in frames:
        out.append(adapter.step(t, _nearest_rows(tracks, t, det_tol), fr, motion_at(machines, t)))
    return out
