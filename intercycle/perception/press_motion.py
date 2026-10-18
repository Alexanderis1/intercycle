"""Press-ram motion from fixed-CCTV clips: the per-clip press-motion ingest (PLAN_TWIN §2 machines contract, §3 P4).

One clip and one press ROI in, one ``MachineRow`` table out (``twin/schemas/tables.py``):
split, clip_id, press, framing, t, flow_v, roi_energy, stroke, missing.

Pipeline (the same code runs on dev and, inside sealed_run.py, on test):

1. ``decode_gray``: grey frames at a working width; times are the container PTS (``frame.time``), never index/fps.
2. ``stabilise``: ECC (Euclidean) of every frame onto the first one, with the ram ROI masked out of the fit, so a
   camera shake is not read as ram motion.
3. ``roi_signals``: Farneback flow between consecutive stabilised frames inside the ROI polygon. ``flow_v`` is the
   mean vertical flow in full-resolution px/s (+ = down); ``roi_energy`` the mean absolute frame difference.
   Frames whose ROI overlaps a person box are MISSING (NaN, no stroke).
4. ``detect_strokes``: Schmitt trigger on the down lobe of ``flow_v`` with an adaptive threshold, confirmed by an
   up lobe (the ram returns) within ``max_stroke_s``, debounced by ``min_gap_s``. The emitted stroke time is the
   ram's BOTTOM (BDC: the down-to-up zero crossing of the smoothed ``flow_v`` after the down lobe), the label
   definition of a stroke (labels/RULES.md; USER DECISION Sun ~07:00), not the onset of the down lobe.
5. ``lamp_beat``: dominant frequency of the global (ROI-masked) luma. Mains flicker sampled at ~25 fps aliases to a
   slow beat; a stroke rate equal to that beat is flagged, because it may be light, not a ram.

``av`` is imported lazily (it is not in the main venv); everything after decoding is numpy + OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

__all__ = ["Clip", "PressParams", "decode_gray", "roi_mask", "stabilise", "roi_signals", "detect_strokes",
           "lamp_beat", "intervals", "ingest_clip", "MACHINE_COLUMNS"]

MACHINE_COLUMNS = ["split", "clip_id", "press", "framing", "t", "flow_v", "roi_energy", "stroke", "missing"]
FULL_W = 1920


@dataclass
class Clip:
    t: np.ndarray            # (N,) seconds from container PTS, first frame at 0
    frames: np.ndarray       # (N, H, W) uint8 grey at working width
    scale: float             # working px per full-resolution px


@dataclass
class PressParams:
    k_mad: float = 5.0             # threshold = max(min_thr, k_mad * robust sigma of flow_v)
    min_thr: float = 4.0           # full-res px/s; idle train ROIs (c14, no occluder) have |flow_v| p99 1-3 px/s
    release: float = 0.4           # Schmitt release level as a fraction of the threshold
    max_stroke_s: float = 3.0      # the up lobe must follow the down lobe within this
    min_gap_s: float = 0.3         # debounce between stroke starts
    require_return: bool = True
    smooth_s: float = 0.2          # centred moving-mean window on flow_v before the trigger (s)
    farneback: tuple = (0.5, 3, 15, 3, 5, 1.2, 0)


@dataclass
class ClipResult:
    table: pd.DataFrame
    strokes_t: np.ndarray
    threshold: float
    beat_hz: float
    beat_frac: float
    beat_amp: float
    stab_px: float                 # max ECC translation (full-res px)
    extra: dict = field(default_factory=dict)


def decode_gray(path: str | Path, width: int = 640, t_max: float | None = None) -> Clip:
    """Grey frames at ``width`` px with times from the container PTS (seconds, shifted so the first frame is 0)."""
    import av  # pyright: ignore[reportMissingImports]

    ts: list[float] = []
    fr: list[np.ndarray] = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        h = int(round(s.codec_context.height * width / s.codec_context.width / 2)) * 2
        for f in c.decode(s):
            if f.time is None:
                raise ValueError(f"{path}: frame without PTS")
            ts.append(float(f.time))
            fr.append(f.to_ndarray(width=width, height=h, format="gray"))
            if t_max is not None and ts[-1] - ts[0] >= t_max:
                break
    t = np.asarray(ts, float)
    order = np.argsort(t, kind="stable")      # decode order is presentation order for these clips; keep it safe
    t = t[order] - t[order][0]
    return Clip(t=t, frames=np.stack([fr[i] for i in order]), scale=width / FULL_W)


def roi_mask(poly_px: Sequence[Sequence[float]], shape: tuple[int, int], scale: float) -> np.ndarray:
    """Boolean mask of a full-resolution pixel polygon at the working scale."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(poly_px, float) * scale).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def stabilise(clip: Clip, exclude: np.ndarray | None = None, down: int = 2) -> tuple[np.ndarray, float]:
    """Warp every frame onto frame 0 (ECC, Euclidean, fitted at 1/``down`` of the working size, ROI excluded).

    Returns the stabilised frames and the largest translation found (full-resolution px). A frame where ECC does
    not converge keeps the previous warp.
    """
    n, h, w = clip.frames.shape
    small = lambda a: cv2.resize(a, (w // down, h // down), interpolation=cv2.INTER_AREA)  # noqa: E731
    ref = small(clip.frames[0]).astype(np.float32)
    mask = np.full(ref.shape, 255, np.uint8)
    if exclude is not None:
        keep = ~cv2.dilate(exclude.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        mask = small(keep.astype(np.uint8) * 255)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    warp = np.eye(2, 3, dtype=np.float32)
    out = np.empty_like(clip.frames)
    out[0] = clip.frames[0]
    max_shift = 0.0
    for i in range(1, n):
        try:
            _, warp = cv2.findTransformECC(ref, small(clip.frames[i]).astype(np.float32), warp.copy(),
                                           cv2.MOTION_EUCLIDEAN, crit, mask, 5)
        except cv2.error:
            pass
        full = warp.copy()
        full[:, 2] *= down
        max_shift = max(max_shift, float(np.hypot(full[0, 2], full[1, 2])) / clip.scale)
        out[i] = cv2.warpAffine(clip.frames[i], full, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    return out, max_shift


def _box_hits(mask: np.ndarray, boxes: Sequence[Sequence[float]], scale: float) -> bool:
    h, w = mask.shape
    for b in boxes:
        x1, y1, x2, y2 = (int(np.floor(b[0] * scale)), int(np.floor(b[1] * scale)),
                          int(np.ceil(b[2] * scale)), int(np.ceil(b[3] * scale)))
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 > x1 and y2 > y1 and mask[y1:y2, x1:x2].any():
            return True
    return False


def roi_signals(frames: np.ndarray, t: np.ndarray, mask: np.ndarray, scale: float,
                person_boxes: Sequence[Sequence[Sequence[float]]] | None = None,
                params: PressParams | None = None, missing_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Per frame: t, flow_v (full-res px/s, + = down), roi_energy, missing. Frame 0 has no flow (NaN, missing).

    ``person_boxes[i]`` lists full-resolution (x1, y1, x2, y2) person boxes of frame i; a frame whose ROI touches a
    box is missing (NaN signals). ``missing_mask`` (optional, (N,) bool) is an external MISSING flag (e.g. the T-W6
    occlusion rule ``intercycle.observer.occlusion.roi_missing_series``), OR-ed with the person-box rule.
    """
    p = params or PressParams()
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty ROI")
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    n = len(t)
    flow_v = np.full(n, np.nan)
    energy = np.full(n, np.nan)
    missing = np.zeros(n, bool)
    if person_boxes is not None:
        if len(person_boxes) != n:
            raise ValueError("person_boxes must have one entry per frame")
        missing = np.array([_box_hits(mask, b, scale) for b in person_boxes], bool)
    if missing_mask is not None:
        mm = np.asarray(missing_mask, bool)
        if mm.shape != (n,):
            raise ValueError(f"missing_mask has shape {mm.shape}, expected ({n},)")
        missing = missing | mm
    for i in range(1, n):
        if missing[i]:
            continue
        a = frames[i - 1, y0:y1, x0:x1]
        b = frames[i, y0:y1, x0:x1]
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            continue
        fl = cv2.calcOpticalFlowFarneback(a, b, np.zeros(a.shape + (2,), np.float32), *p.farneback)
        flow_v[i] = float(fl[..., 1][sub].mean()) / scale / dt
        energy[i] = float(np.abs(b.astype(np.int16) - a.astype(np.int16))[sub].mean())
    # no measurement (frame 0, a non-increasing PTS step) is MISSING too: the machines contract reads NaN as missing
    missing = missing | ~np.isfinite(flow_v) | ~np.isfinite(energy)
    return pd.DataFrame({"t": t, "flow_v": flow_v, "roi_energy": energy, "missing": missing})


def _bdc_index(t: np.ndarray, v: np.ndarray, ok: np.ndarray, i: int, max_s: float) -> int:
    """BDC of the down lobe triggered at sample i: of the last finite sample with v > 0 and the first finite sample
    with v <= 0 after it, the one closer to zero; -1 when no finite v <= 0 follows within max_s of t[i]."""
    last_pos = i
    for m in range(i + 1, len(v)):
        if t[m] - t[i] > max_s:
            return -1
        if not ok[m]:
            continue
        if v[m] <= 0:
            return m if abs(v[m]) <= abs(v[last_pos]) else last_pos
        last_pos = m
    return -1


def detect_strokes(t: np.ndarray, flow_v: np.ndarray, params: PressParams | None = None) -> tuple[np.ndarray, float]:
    """Indices of the strokes' BOTTOM samples (BDC) and the threshold used (px/s).

    A stroke is a down lobe (smoothed flow_v above the threshold) confirmed by the return (an up lobe below -threshold
    within ``max_stroke_s``). Its emitted sample is BDC, the label definition of a stroke time: the down-to-up zero
    crossing of the smoothed flow_v after the down lobe, i.e. of the last finite sample with v > 0 and the first
    finite sample with v <= 0 the one closer to zero (both within ``max_stroke_s`` of the trigger). No crossing in
    that span: no stroke (fail closed). ``min_gap_s`` debounces consecutive BDC times.

    NaN samples (missing) are never emitted and reset the trigger.
    """
    p = params or PressParams()
    v = np.asarray(flow_v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.zeros(0, int), float("nan")
    # Noise sigma from the frame-to-frame differences of v (MAD / 0.6745 / sqrt 2 for white noise). A running
    # eccentric press moves on every frame (a smooth near-sinusoid with no dwell), so any estimate taken from the
    # level of v (a percentile of |v|, a MAD of v) is the stroke amplitude itself and the trigger never fires
    # (P4 rerun: train ROIs with a clear 0.5-0.6 Hz ram beat gave thresholds 25-45 px/s above a 10 px/s signal).
    # The smooth stroke changes little between frames, so the differences carry the noise.
    d = np.diff(v[ok])
    sig = float(np.median(np.abs(d - np.median(d)))) / 0.6745 / np.sqrt(2.0) if len(d) else 0.0
    # Low-pass: a centred moving mean over smooth_s (finite samples only; NaN samples stay NaN). Strokes are slower
    # than ~3 Hz, so this keeps the lobes and divides the white-noise sigma by sqrt(samples in the window).
    if p.smooth_s > 0:
        tt = np.asarray(t, float)
        vs = np.full(len(v), np.nan)
        cnt = np.zeros(len(v))
        for i in np.nonzero(ok)[0]:
            w = ok & (np.abs(tt - tt[i]) <= p.smooth_s / 2)
            vs[i] = v[w].mean()
            cnt[i] = w.sum()
        sig = sig / np.sqrt(max(float(np.median(cnt[ok])), 1.0))
        v = vs
    thr = max(p.min_thr, p.k_mad * sig)
    rel = p.release * thr
    starts: list[int] = []
    armed = True
    i, n = 0, len(v)
    while i < n:
        if not ok[i]:
            armed = True
            i += 1
            continue
        if armed and v[i] > thr:
            good = True
            if p.require_return:
                k = i
                while k < n and t[k] - t[i] <= p.max_stroke_s and not (ok[k] and v[k] < -thr):
                    k += 1
                good = k < n and t[k] - t[i] <= p.max_stroke_s
            j0 = _bdc_index(t, v, ok, i, p.max_stroke_s) if good else -1
            if j0 >= 0 and (not starts or t[j0] - t[starts[-1]] >= p.min_gap_s):
                starts.append(j0)
            armed = False
        elif not armed and v[i] < rel:
            armed = True
        i += 1
    return np.asarray(starts, int), float(thr)


def lamp_beat(t: np.ndarray, luma: np.ndarray, f_min: float = 0.1) -> tuple[float, float, float]:
    """Dominant frequency (Hz) of a luma series, the share of non-DC power at that peak (±1 bin), and its amplitude.

    Uses a Lomb-style projection on the PTS times so uneven frame spacing does not bias the peak.
    """
    t = np.asarray(t, float)
    y = np.asarray(luma, float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8 or t[-1] - t[0] <= 0:
        return float("nan"), 0.0, 0.0
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    dur = t[-1] - t[0]
    fs = 1.0 / np.median(np.diff(t))
    freqs = np.arange(max(f_min, 1.0 / dur), fs / 2, 1.0 / (4 * dur))
    if len(freqs) == 0:
        return float("nan"), 0.0, 0.0
    ph = 2 * np.pi * np.outer(freqs, t)
    c, s = np.cos(ph) @ y, np.sin(ph) @ y
    pw = (c * c + s * s) / len(t)
    k = int(np.argmax(pw))
    tot = float(pw.sum())
    peak = float(pw[max(k - 4, 0):k + 5].sum())   # ±1 natural bin = ±4 oversampled bins
    amp = 2 * np.sqrt(pw[k] / len(t))
    return float(freqs[k]), peak / tot if tot > 0 else 0.0, float(amp)


def intervals(strokes_t: np.ndarray, duration: float) -> tuple[np.ndarray, float | None]:
    """In-clip stroke intervals (s) and the right-censored time from the last stroke to the clip end."""
    s = np.sort(np.asarray(strokes_t, float))
    if len(s) == 0:
        return np.zeros(0), None
    return np.diff(s), float(duration - s[-1])


def ingest_clip(path: str | Path, clip_id: str, split: str, press: str, framing: str,
                roi_poly_px: Sequence[Sequence[float]], person_boxes=None, width: int = 640,
                params: PressParams | None = None, clip: Clip | None = None,
                missing_mask: np.ndarray | None = None) -> ClipResult:
    """Full per-clip press ingest: decode -> ECC -> ROI flow -> strokes, plus the lamp-beat check.

    ``person_boxes`` (optional) is one list of full-resolution boxes per decoded frame; ``missing_mask`` (optional) is
    an external per-frame MISSING flag (see ``roi_signals``). A frame without a flow value (frame 0, a repeated PTS) is
    MISSING too: NaN always means MISSING (MachineRow contract, T-W6), never "no motion".
    """
    p = params or PressParams()
    c = clip if clip is not None else decode_gray(path, width=width)
    mask = roi_mask(roi_poly_px, c.frames.shape[1:], c.scale)
    stab, max_shift = stabilise(c, exclude=mask)
    sig = roi_signals(stab, c.t, mask, c.scale, person_boxes, p, missing_mask=missing_mask)
    nan = ~(np.isfinite(sig["flow_v"].to_numpy()) & np.isfinite(sig["roi_energy"].to_numpy()))
    sig["missing"] = sig["missing"].to_numpy() | nan
    sig.loc[sig["missing"], ["flow_v", "roi_energy"]] = np.nan
    idx, thr = detect_strokes(c.t, sig["flow_v"].to_numpy(), p)
    stroke = np.zeros(len(c.t), bool)
    stroke[idx] = True
    stroke &= ~sig["missing"].to_numpy()
    bg = ~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    luma = stab[:, bg].mean(axis=1)
    f, frac, amp = lamp_beat(c.t, luma)
    table = pd.DataFrame({"split": split, "clip_id": clip_id, "press": press, "framing": framing, "t": c.t,
                          "flow_v": sig["flow_v"], "roi_energy": sig["roi_energy"], "stroke": stroke,
                          "missing": sig["missing"]}, columns=pd.Index(MACHINE_COLUMNS))
    roi_luma = stab[:, mask].mean(axis=1)
    rf, rfrac, ramp = lamp_beat(c.t, roi_luma)
    return ClipResult(table=table, strokes_t=c.t[stroke], threshold=thr, beat_hz=f, beat_frac=frac, beat_amp=amp,
                      stab_px=max_shift, extra={"roi_beat_hz": rf, "roi_beat_frac": rfrac, "roi_beat_amp": ramp,
                                                "duration": float(c.t[-1]) if len(c.t) else 0.0})
