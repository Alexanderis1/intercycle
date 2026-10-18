"""Per-clip ingest of the real press-shop clips (PLAN_TWIN §2 tracks / machines contracts, §4 S3a PERC, §5).

ONE function, ``ingest_clip``, turns one clip file into its track table, its per-press machine tables and its raw
detection boxes. The dev batch (``cloud/modal_perc.py``) and the sealed real-clip test (``scripts/sealed_run.py``)
call the SAME function. There is no split logic in this module: ``split`` is only the value written into the
``split`` column; which clips may be read is decided by the caller (the dev batch refuses anything but train/dev).

Pipeline per clip:

1. **Decode** (PyAV) every frame. Times are the container PTS (``frame.time``), shifted so the first frame is 0;
   never ``frame_index / fps``. A frame without a PTS, or PTS going backwards, raises (fail closed).
2. **Sample** the frames nearest a 1/``fps`` grid (default 12 fps) for detection; their own PTS times are kept.
3. **Detect** with RF-DETR-M (COCO), the same model as staging (``cloud/modal_esk.py``), at conf >= ``det_conf``.
4. **Class map** (explicit, ``class_map``): ``person`` -> person; COCO ``truck`` at conf >= 0.5 -> forklift on
   camera 14 only (P5 agent check: precision 1.00, 8 of 13 found). Camera 9 has no vehicle class (P5: "no vehicles
   observed in zone-9 train clips"; its vehicle boxes are the coil and the press). Every other class is dropped.
5. **P5 rule**: a person box whose foot point lies inside a forklift box is suppressed (the driver), before tracking.
6. **Track** with a FRESH ByteTrack (roboflow ``trackers``) per clip and per class, driven by the PTS times.
7. **Occlusion = MISSING** (T-W6): on every press ROI of the clip's framing, a sampled frame is hit when an occluder
   box overlaps the ROI (``intercycle.observer.occlusion.roi_missing_series``, boxes padded 15 %), held ``occ_hold_s``
   after the last hit. Occluders = every confirmed track box plus every person (P5-suppressed drivers included) or
   forklift detection at conf >= ``occ_conf`` (default the tracker's activation confidence 0.4). Boxes at 0.1-0.4
   that no track claims do not occlude: on a dev c9_f1 clip they averaged ~29 per frame and would make the P9b ROI
   MISSING on 98 % of frames.
   A native frame is MISSING when the held flag of the sample at or before it is set, or the sample after it is hit.
8. **Press motion** per press ROI (``press_motion.ingest_clip`` on the native-rate grey frames, with that MISSING
   mask): flow_v / roi_energy NaN and no stroke on MISSING frames.

Tracks follow ``twin.schemas.tables.TrackRow`` (split, t, track_id, class, u, v, x, y, conf, framing): (u, v) is the
foot point (bottom centre) of the tracked detection box in full-resolution px; x, y stay NaN until CAL lands, then
``project_tracks`` fills them. Only confirmed tracks (ByteTrack id >= 0) are written; a row is a detection matched to
a track, so a frame where the detector missed the person has no row (the gaps feed stat_v1's miss-burst model,
``track_gaps``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from intercycle.observer.mapping import OCC_P_MISS
from intercycle.observer.occlusion import OCC_EPS, ROI_PAD_FRAC, hold_ticks, roi_missing_series
from intercycle.perception import press_motion as pm
from intercycle.perception.detector import FORKLIFT, PERSON, Dets
from intercycle.perception.tracking import CameraTracker, TrackerConfig
from intercycle.perception.vehicles import foot_points, suppress_people_on_forklifts

__all__ = ["IngestConfig", "ClipIngest", "TRACK_COLUMNS", "BOX_COLUMNS", "class_map", "apply_class_map",
           "sample_indices", "native_missing", "ingest_clip", "write_outputs", "project_tracks", "track_gaps",
           "load_press_rois"]

TRACK_COLUMNS = ["split", "t", "track_id", "class", "u", "v", "x", "y", "conf", "framing"]
BOX_COLUMNS = ["split", "clip_id", "framing", "frame", "t", "class", "conf", "x1", "y1", "x2", "y2", "suppressed"]
TRUCK = "truck"                     # COCO class name, the raw label the detector is asked to keep
FORKLIFT_CAMERAS = (14,)
DEFAULT_FPS = 12.0

Detector = Callable[[list[np.ndarray]], list[Dets]]


def _default_hold(fps: float) -> float:
    # MEASURED_SHIM's per-frame occluder miss probability (drop or gross error, intercycle.observer.mapping) and the
    # observer's leak target OCC_EPS, counted in detector frames at `fps`: 16 frames = 1.33 s at 12 fps.
    return hold_ticks(OCC_P_MISS, OCC_EPS) / fps


@dataclass(frozen=True)
class IngestConfig:
    fps: float = DEFAULT_FPS               # detection sampling rate on the PTS grid
    det_conf: float = 0.1                  # RF-DETR threshold; every person box >= this is stored and occludes
    forklift_conf: float = 0.5             # COCO truck -> forklift at this confidence (P5 operating point)
    forklift_cameras: tuple[int, ...] = FORKLIFT_CAMERAS
    occ_hold_s: float | None = None        # None = _default_hold(fps)
    occ_pad_frac: float = ROI_PAD_FRAC
    occ_conf: float | None = None          # untracked detections occlude at >= this; None = tracker activation (0.4)
    press_width: int = 640                 # working width of the grey frames for press motion (P4)
    batch: int = 8                         # detector batch
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @property
    def hold_s(self) -> float:
        return _default_hold(self.fps) if self.occ_hold_s is None else float(self.occ_hold_s)

    @property
    def occ_conf_value(self) -> float:
        return float(self.tracker.activation if self.occ_conf is None else self.occ_conf)


@dataclass
class ClipIngest:
    clip_id: str
    tracks: pd.DataFrame                   # TRACK_COLUMNS
    machines: dict[str, pd.DataFrame]      # press id -> press_motion.MACHINE_COLUMNS table
    boxes: pd.DataFrame                    # BOX_COLUMNS: every kept detection on the sampled frames
    stats: dict


# ------------------------------------------------------------------------------------------------ class map

def class_map(camera: int, cfg: IngestConfig | None = None) -> dict[str, tuple[str, float]]:
    """Explicit {raw detector class: (our class, min conf)} for a camera. Anything not listed is dropped."""
    c = cfg or IngestConfig()
    out = {PERSON: (PERSON, c.det_conf)}
    if int(camera) in c.forklift_cameras:
        out[TRUCK] = (FORKLIFT, c.forklift_conf)
    return out


def apply_class_map(d: Dets, camera: int, cfg: IngestConfig | None = None) -> Dets:
    """Relabel raw detections through ``class_map``; drop classes it does not list and boxes under their conf."""
    cm = class_map(camera, cfg)
    if len(d) == 0:
        return Dets.empty()
    lab = np.array([cm[str(x)][0] if str(x) in cm else "" for x in d.label], dtype="<U16")
    thr = np.array([cm[str(x)][1] if str(x) in cm else np.inf for x in d.label], float)
    keep = (lab != "") & (np.asarray(d.score, float) >= thr)
    return Dets(np.asarray(d.xyxy, np.float32)[keep].reshape(-1, 4), np.asarray(d.score, np.float32)[keep], lab[keep])


# ------------------------------------------------------------------------------------------------ timing

def sample_indices(t: np.ndarray, fps: float) -> np.ndarray:
    """Indices of the frames nearest each point of the grid 0, 1/fps, 2/fps, ... (PTS seconds, first frame 0).

    A frame is taken when it is the closest frame to the next grid point (ties go to the earlier frame); the grid
    never runs ahead of the video, so the sampled rate is fps whenever the native rate is above it.
    """
    t = np.asarray(t, float)
    if len(t) == 0:
        return np.zeros(0, int)
    out: list[int] = []
    k = 0
    for i in range(len(t)):
        g = k / fps
        if t[i] + 1e-9 < g:
            continue
        # t[i] >= grid point: take the closer of frame i-1 / i (i-1 if not already taken)
        j = i
        if i > 0 and (not out or out[-1] != i - 1) and abs(t[i - 1] - g) <= abs(t[i] - g):
            j = i - 1
        if not out or out[-1] != j:
            out.append(j)
        while k / fps <= t[i] + 1e-9:
            k += 1
    return np.asarray(out, int)


def native_missing(t_native: np.ndarray, t_samp: np.ndarray, hit: np.ndarray, held: np.ndarray) -> np.ndarray:
    """MISSING on native frames from the sampled-frame flags (fail closed between samples).

    Native frame j is MISSING when the held flag of the last sample at or before it is set, or the first sample at
    or after it is hit. With no samples at all, every frame is MISSING.
    """
    tn = np.asarray(t_native, float)
    ts = np.asarray(t_samp, float)
    if len(ts) == 0:
        return np.ones(len(tn), bool)
    prev = np.searchsorted(ts, tn, side="right") - 1
    nxt = np.searchsorted(ts, tn, side="left")
    a = np.where(prev >= 0, np.asarray(held, bool)[np.clip(prev, 0, len(ts) - 1)], False)
    b = np.where(nxt < len(ts), np.asarray(hit, bool)[np.clip(nxt, 0, len(ts) - 1)], False)
    return a | b


# ------------------------------------------------------------------------------------------------ decode

@dataclass
class _Decoded:
    t: np.ndarray                 # (N,) native PTS seconds, first frame 0
    gray: np.ndarray              # (N, h, w) uint8 at press_width
    samp: np.ndarray              # (S,) native indices of the sampled frames
    dets: list[Dets]              # raw detector output per sampled frame
    size: tuple[int, int]         # full-resolution (W, H)
    decode_s: float
    det_s: float


def _decode_and_detect(path: str | Path, detector: Detector, cfg: IngestConfig) -> _Decoded:
    import av  # pyright: ignore[reportMissingImports]

    t0 = time.perf_counter()
    det_s = 0.0
    ts: list[float] = []
    gray: list[np.ndarray] = []
    samp: list[int] = []
    dets: list[Dets] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal det_s
        if batch:
            a = time.perf_counter()
            out = detector(batch)
            if len(out) != len(batch):
                raise RuntimeError(f"detector returned {len(out)} results for {len(batch)} frames")
            dets.extend(out)
            det_s += time.perf_counter() - a
            batch.clear()

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        W, H = int(s.codec_context.width), int(s.codec_context.height)
        h = int(round(H * cfg.press_width / W / 2)) * 2
        k = 0
        pts0: float | None = None
        prev = None                                   # previous decoded frame (RGB converted only if sampled)
        for f in c.decode(s):
            if f.time is None:
                raise ValueError(f"{path}: frame without PTS")
            tt = float(f.time)
            pts0 = tt if pts0 is None else pts0
            tr = tt - pts0
            if ts and tr <= ts[-1]:
                raise ValueError(f"{path}: PTS not increasing ({tr} after {ts[-1]})")
            # the nearest-to-grid rule of sample_indices, streamed: once frame i reaches the grid point, the closer
            # of frames i-1 / i is taken
            i = len(ts)
            ts.append(tr)
            gray.append(f.to_ndarray(width=cfg.press_width, height=h, format="gray"))
            g = k / cfg.fps
            if tr + 1e-9 >= g:
                take_prev = prev is not None and (not samp or samp[-1] != i - 1) and abs(ts[i - 1] - g) <= abs(tr - g)
                src = prev if take_prev else f
                assert src is not None
                samp.append(i - 1 if take_prev else i)
                batch.append(src.to_ndarray(format="rgb24"))
                while k / cfg.fps <= tr + 1e-9:
                    k += 1
                if len(batch) >= cfg.batch:
                    flush()
            prev = f
    flush()
    t = np.asarray(ts, float)
    samp_a = np.asarray(samp, int)
    if not np.array_equal(samp_a, sample_indices(t, cfg.fps)):
        raise AssertionError("streamed sampling disagrees with sample_indices")
    return _Decoded(t=t, gray=np.stack(gray) if gray else np.zeros((0, 1, 1), np.uint8), samp=samp_a, dets=dets,
                    size=(W, H), decode_s=time.perf_counter() - t0 - det_s, det_s=det_s)


# ------------------------------------------------------------------------------------------------ the ingest

def ingest_clip(path: str | Path, clip_id: str, split: str, camera: int, framing: str,
                press_rois: Mapping[str, Sequence[Sequence[float]]], detector: Detector,
                cfg: IngestConfig | None = None, press_params: pm.PressParams | None = None) -> ClipIngest:
    """One clip -> tracks, per-press machine tables, raw boxes and stats (see the module docstring).

    ``press_rois``: {press id: full-resolution ROI polygon} for THIS clip's framing (``load_press_rois``).
    ``detector(images)``: RGB uint8 frames -> one ``Dets`` per frame with raw labels ``person`` / ``truck``
    (``RFDetrDetector(keep=("person", "truck"), class_map={"truck": "truck"})``).
    """
    cfg = cfg or IngestConfig()
    dec = _decode_and_detect(path, detector, cfg)
    a = time.perf_counter()
    trk = CameraTracker(cfg.fps, (PERSON, FORKLIFT), cfg.tracker)        # fresh per clip
    t_s = dec.t[dec.samp]
    rows: list[tuple] = []
    box_rows: list[tuple] = []
    occ: list[np.ndarray] = []
    ids: dict[tuple[str, int], int] = {}
    n_person = np.zeros(len(t_s), int)
    n_person05 = np.zeros(len(t_s), int)
    n_supp = 0
    for k, (tk, raw) in enumerate(zip(t_s, dec.dets)):
        d = apply_class_map(raw, camera, cfg)
        pmask = d.label == PERSON
        fmask = d.label == FORKLIFT
        keep = suppress_people_on_forklifts(d.xyxy[pmask], d.xyxy[fmask])
        n_supp += int((~keep).sum())
        n_person[k] = int(pmask.sum())
        n_person05[k] = int((d.score[pmask] >= 0.5).sum())
        supp = np.zeros(len(d), bool)
        supp[np.nonzero(pmask)[0][~keep]] = True
        for b, s, lab, sp in zip(d.xyxy, d.score, d.label, supp):
            box_rows.append((split, clip_id, framing, k, float(tk), str(lab), float(s), *map(float, b), bool(sp)))
        tb = trk.update(d.select(~supp), float(tk))
        # occluders (T-W6 "a person / vehicle track overlaps"): every confirmed track box, plus every person (P5-
        # suppressed drivers too) or forklift detection at >= occ_conf, the confidence that spawns a track
        strong = np.asarray(d.score, float) >= cfg.occ_conf_value
        occ.append(np.concatenate([np.asarray(d.xyxy, float).reshape(-1, 4)[strong],
                                   np.asarray(tb.xyxy, float).reshape(-1, 4)]))
        if len(tb):
            fp = foot_points(tb.xyxy)
            for (u, v), s, tid, lab in zip(fp, tb.score, tb.track_id, tb.label):
                key = (str(lab), int(tid))
                if key not in ids:
                    ids[key] = len(ids)
                rows.append((split, float(tk), ids[key], str(lab), float(u), float(v), np.nan, np.nan,
                             float(min(max(s, 0.0), 1.0)), framing))
    track_s = time.perf_counter() - a
    tracks = pd.DataFrame(rows, columns=pd.Index(TRACK_COLUMNS))
    tracks = tracks.astype({"t": "float64", "track_id": "int64", "u": "float64", "v": "float64", "x": "float64",
                            "y": "float64", "conf": "float64"})
    boxes = pd.DataFrame(box_rows, columns=pd.Index(BOX_COLUMNS)).astype(
        {"frame": "int64", "t": "float64", "conf": "float64", "x1": "float64", "y1": "float64", "x2": "float64",
         "y2": "float64", "suppressed": "bool"})

    a = time.perf_counter()
    machines: dict[str, pd.DataFrame] = {}
    press_stats: dict[str, dict] = {}
    if press_rois:
        clip = pm.Clip(t=dec.t, frames=dec.gray, scale=cfg.press_width / dec.size[0])
        for press, roi in press_rois.items():
            held = roi_missing_series(t_s, occ, roi, cfg.hold_s, cfg.occ_pad_frac)
            hit = roi_missing_series(t_s, occ, roi, 0.0, cfg.occ_pad_frac)
            miss = native_missing(dec.t, t_s, hit, held)
            r = pm.ingest_clip(path, clip_id, split, press, framing, roi, clip=clip, params=press_params,
                               missing_mask=miss)
            machines[press] = r.table
            press_stats[press] = {"n_strokes": int(len(r.strokes_t)),
                                  "missing_share": float(np.mean(r.table["missing"].to_numpy(bool))),
                                  "occluded_share": float(miss.mean()), "threshold": r.threshold,
                                  "stab_px": r.stab_px, "beat_hz": r.beat_hz}
    press_s = time.perf_counter() - a

    dur = float(dec.t[-1]) if len(dec.t) else 0.0
    i05 = int(np.argmin(np.abs(t_s - 0.5))) if len(t_s) else -1
    stats = {
        "clip_id": clip_id, "split": split, "camera": int(camera), "framing": framing, "duration_pts": dur,
        "n_native": int(len(dec.t)), "n_samples": int(len(t_s)),
        "fps_native": float((len(dec.t) - 1) / dur) if dur > 0 else float("nan"),
        "fps_sampled": float((len(t_s) - 1) / (t_s[-1] - t_s[0])) if len(t_s) > 1 and t_s[-1] > t_s[0] else float("nan"),
        "decode_s": dec.decode_s, "det_s": dec.det_s, "track_s": track_s, "press_s": press_s,
        "person_boxes": int(n_person.sum()), "person_boxes_05": int(n_person05.sum()),
        "person_at_05s_conf05": int(n_person05[i05]) if i05 >= 0 else 0,
        "forklift_boxes": int((boxes["class"] == FORKLIFT).sum()), "suppressed": n_supp,
        "n_tracks": {c: int(sum(1 for (cc, _) in ids if cc == c)) for c in (PERSON, FORKLIFT)},
        "track_rows": int(len(tracks)), "occ_hold_s": cfg.hold_s, "presses": press_stats,
    }
    return ClipIngest(clip_id=clip_id, tracks=tracks, machines=machines, boxes=boxes, stats=stats)


def write_outputs(res: ClipIngest, root: str | Path) -> list[Path]:
    """Write tracks/<clip>.parquet, machines/<press>/<clip>.parquet and boxes/<clip>.parquet under ``root``."""
    root = Path(root)
    out = []
    p = root / "tracks" / f"{res.clip_id}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    res.tracks.to_parquet(p, index=False)
    out.append(p)
    for press, tab in res.machines.items():
        p = root / "machines" / press / f"{res.clip_id}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_parquet(p, index=False)
        out.append(p)
    p = root / "boxes" / f"{res.clip_id}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    res.boxes.to_parquet(p, index=False)
    out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ helpers

def load_press_rois(path: str | Path) -> dict[str, dict[str, list]]:
    """{framing id: {press id: ROI polygon}} from a P4 ROI file (docs/probes/S3a-P4_rois.json format)."""
    import json

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, list]] = {}
    for pr in d["presses"]:
        rois = pr["rois"] if "rois" in pr else {pr["framing"]: pr["roi_px"]}
        for fr, poly in rois.items():
            out.setdefault(str(fr), {})[str(pr["id"])] = poly
    return out


def project_tracks(tracks: pd.DataFrame, H_by_framing: Mapping[str, object],
                   k1_by_framing: Mapping[str, float] | None = None,
                   center: Sequence[float] | None = None) -> pd.DataFrame:
    """Copy of ``tracks`` with x, y filled: (u, v) undistorted by the framing's division-model k1 (if given), then
    mapped through its H (undistorted px -> floor, ``twin.schemas.site.Framing.H_img_to_floor``). Rows of a
    framing with no H get NaN (PLAN §5: a framing without a fitted H has no floor position)."""
    from intercycle.framing import CENTER, undistort

    out = tracks.copy()
    x = np.full(len(out), np.nan)
    y = np.full(len(out), np.nan)
    fr = out["framing"].astype(str).to_numpy()
    for f in np.unique(fr):
        if f not in H_by_framing or H_by_framing[f] is None:
            continue
        H = np.asarray(H_by_framing[f], float).reshape(3, 3)
        m = fr == f
        uv = out.loc[m, ["u", "v"]].to_numpy(float)
        k1 = float((k1_by_framing or {}).get(f, 0.0) or 0.0)
        if k1 != 0.0:
            uv = undistort(uv, k1, CENTER if center is None else center)
        p = np.c_[uv, np.ones(len(uv))] @ H.T
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = np.abs(p[:, 2]) > 1e-12
            x[m] = np.where(ok, p[:, 0] / p[:, 2], np.nan)
            y[m] = np.where(ok, p[:, 1] / p[:, 2], np.nan)
    out["x"] = x
    out["y"] = y
    return out


def track_gaps(tracks: pd.DataFrame, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Per track: its observed-frame count and every within-track gap (missed sampled frames between two rows).

    Returns one row per (track, run) with columns class, track_id, kind ("obs" run of consecutive observed frames, or
    "gap" run of missed frames), n_frames, seconds. Runs are measured in sampled frames: a step of dt between two
    rows is round(dt * fps) frames, so a gap run has round(dt * fps) - 1 frames.
    """
    rows = []
    for key, g in tracks.groupby(["class", "track_id"], sort=True):
        cls, tid = cast(tuple, key)
        tt = np.sort(g["t"].to_numpy(float))
        steps = np.rint(np.diff(tt) * fps).astype(int)
        run = 1
        for s in steps:
            if s <= 1:
                run += 1
                continue
            rows.append((cls, int(tid), "obs", run, run / fps))
            rows.append((cls, int(tid), "gap", int(s - 1), (s - 1) / fps))
            run = 1
        rows.append((cls, int(tid), "obs", run, run / fps))
    return pd.DataFrame(rows, columns=pd.Index(["class", "track_id", "kind", "n_frames", "seconds"]))
