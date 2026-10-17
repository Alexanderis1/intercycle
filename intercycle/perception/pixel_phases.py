"""Pixels -> phases (PRODUCTION_BRIEF A1): per-machine feature vectors for `intercycle/observer/phases.py`, computed
from CCTV frames with OpenCV only (Apache-2.0). Never sees ground truth: frames, their capture times and the
commissioning ROIs are the only inputs.

Commissioning (once per camera, a human step): one image box per machine zone, drawn on a frame and stored as JSON
(`load_rois`). Everything else is automatic. Per frame:

1. stabilisation: `cv2.findTransformECC` (Euclidean) of a half-size frame against the first one (camera vibration);
2. luma normalised by the frame median (auto exposure / flicker);
3. motion mask: absolute difference to the previous stabilised frame > DIFF_THR grey levels;
4. morphological opening (3x3) + connected components below MIN_BLOB px dropped (weld sparks, noise);
5. weld flash: a ROI where more than FLASH_FRAC of the pixels change at once gives no sample for this frame;
6. per ROI: motion pixels (motion energy), motion pixels in a 4 x 4 grid over the box, saturated pixels
   (luma >= SAT_THR before normalisation: the weld-on cue). Pixel counts are divided by PX_UNIT so they sit on the
   same scale as the shim's part counts (phases.py takes log1p).

Frames are averaged into the observer's ticks (2 Hz): tick T holds the frames captured in (T - 1/hz, T]. The machine block per tick is
[motion energy, 16 grid cells, weld-on] (N_PIX = 18): `PhaseModel.fit_features` takes any width (standardise + PCA),
so phases.py is used unchanged. Machines without a ROI get NaN rows (no model, as a blacked-out camera).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

GRID = (4, 4)                 # = phases.GRID
N_PIX = 1 + GRID[0] * GRID[1] + 1
DIFF_THR = 18.0               # grey levels (after the median normalisation to 128)
MIN_BLOB = 12                 # px at 640x360: smaller connected motion blobs are sparks / noise
FLASH_FRAC = 0.5              # share of a ROI changing at once = weld flash / exposure jump, not motion
SAT_THR = 250                 # raw luma counted as saturated (arc light, spatter)
PX_UNIT = 20.0                # motion pixels per feature unit
ECC_SCALE = 0.5               # stabilisation runs on a half-size frame


@dataclass(frozen=True)
class Roi:
    mid: str
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 (pixels, x1/y1 exclusive)


def load_rois(path) -> tuple[tuple[int, int], list[Roi]]:
    """{"size": [w, h], "rois": {mid: [x0, y0, x1, y1]}, ...} -> ((w, h), [Roi]); every box must lie inside the frame."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    w, h = int(d["size"][0]), int(d["size"][1])
    rois = []
    for m, (x0, y0, x1, y1) in d["rois"].items():
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"ROI {m} {[x0, y0, x1, y1]} is not inside the {w}x{h} frame")
        rois.append(Roi(m, (int(x0), int(y0), int(x1), int(y1))))
    return (w, h), rois


def _luma(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class PixelFeatures:
    """Stateful per-frame extractor: call `frame(img)` in time order; returns (n_roi, N_PIX) with NaN rows for ROIs
    skipped this frame (flash) and for the first frame (no previous frame yet)."""

    def __init__(self, rois: list[Roi], stabilise: bool = True):
        self.rois, self.stabilise = rois, stabilise
        self.ref: np.ndarray | None = None          # half-size float32 luma of the first frame (ECC template)
        self.ecc_mask: np.ndarray | None = None
        self.prev: np.ndarray | None = None         # previous stabilised, normalised luma (float32)
        self.kernel = np.ones((3, 3), np.uint8)
        self.n_flash = 0
        self.n_ecc_fail = 0

    def _warp(self, g: np.ndarray) -> np.ndarray:
        small = cv2.resize(g, None, fx=ECC_SCALE, fy=ECC_SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
        ref, mask = self.ref, self.ecc_mask
        if ref is None or mask is None:
            self.ref, self.ecc_mask = small, np.ones(small.shape, np.uint8)
            return g
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(ref, small, M, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4), mask, 3)
        except cv2.error:
            self.n_ecc_fail += 1
            return g
        M[:, 2] /= ECC_SCALE
        return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def frame(self, img: np.ndarray) -> np.ndarray:
        raw = _luma(img)
        g = self._warp(raw) if self.stabilise else raw
        gn = g.astype(np.float32) * (128.0 / max(float(np.median(g)), 1.0))
        out = np.full((len(self.rois), N_PIX), np.nan, np.float32)
        prev, self.prev = self.prev, gn
        if prev is None:
            return out
        mask = (np.abs(gn - prev) > DIFF_THR).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_BLOB
        mask = keep[np.asarray(lab, dtype=np.intp)]
        sat = raw >= SAT_THR
        gy, gx = GRID[1], GRID[0]
        for r, roi in enumerate(self.rois):
            x0, y0, x1, y1 = roi.box
            raw_change = np.abs(gn[y0:y1, x0:x1] - prev[y0:y1, x0:x1]) > DIFF_THR
            if raw_change.mean() > FLASH_FRAC:          # most of the zone changed at once: flash, not motion
                self.n_flash += 1
                continue
            m = mask[y0:y1, x0:x1].astype(np.float32)
            h, w = m.shape
            ys = np.linspace(0, h, gy + 1).astype(int)
            xs = np.linspace(0, w, gx + 1).astype(int)
            cells = [m[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].sum() for i in range(gx) for j in range(gy)]   # x-major, as phases
            out[r, 0] = m.sum() / PX_UNIT
            out[r, 1:1 + gx * gy] = np.array(cells, np.float32) / PX_UNIT
            out[r, -1] = sat[y0:y1, x0:x1].sum() / PX_UNIT
        return out


def to_ticks(t_frames: np.ndarray, per_frame: np.ndarray, hz: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Average frame features into observer ticks: tick T = t0 + k / hz averages the frames captured in
    (T - 1/hz, T] (causal: the tick is stamped when its last frame is captured, so a frame is never stamped later
    than it was seen; at fps = hz each tick is exactly one frame). NaN samples are ignored; a tick with no valid
    sample stays NaN. Returns T (n,), X (n, n_roi, N_PIX)."""
    t0 = float(t_frames[0])
    k = np.ceil((t_frames - t0) * hz - 1e-6).astype(np.int64)
    n = int(k.max()) + 1
    s = np.zeros((n, *per_frame.shape[1:]), np.float64)
    c = np.zeros((n, *per_frame.shape[1:]), np.float64)
    ok = ~np.isnan(per_frame)
    np.add.at(s, k, np.where(ok, per_frame, 0.0))
    np.add.at(c, k, ok)
    with np.errstate(invalid="ignore"):
        X = (s / c).astype(np.float32)
    return t0 + np.arange(n) / hz, X


def machine_block(X_roi: np.ndarray, rois: list[Roi], mids: list[str]) -> np.ndarray:
    """(n, n_roi, D) -> (n, M, D) in layout order; machines without a ROI are NaN."""
    out = np.full((X_roi.shape[0], len(mids), X_roi.shape[2]), np.nan, np.float32)
    for r, roi in enumerate(rois):
        out[:, mids.index(roi.mid)] = X_roi[:, r]
    return out


def read_frames_csv(path) -> list[tuple[str, float]]:
    """render/blender/scene_real.py frames.csv: (file, t) in frame order."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["file"], float(row["t"])) for row in csv.DictReader(f)]


def extract_clip(frames_dir, frames: Iterable[tuple[str, float]], rois: list[Roi], size: tuple[int, int],
                 stabilise: bool = True, progress: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-frame features of a clip: (t (n,), features (n, n_roi, N_PIX), stats). Every frame must be `size` (w, h),
    the frame size the ROIs were drawn on."""
    fe = PixelFeatures(rois, stabilise)
    ts, feats = [], []
    for i, (name, t) in enumerate(frames):
        img = cv2.imread(str(Path(frames_dir) / name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(Path(frames_dir) / name)
        if (img.shape[1], img.shape[0]) != tuple(size):
            raise ValueError(f"{name} is {img.shape[1]}x{img.shape[0]}, the ROIs are for {size[0]}x{size[1]}")
        feats.append(fe.frame(img))
        ts.append(t)
        if progress and i % progress == 0:
            print(f"  frame {i} t={t:.1f}", flush=True)
    return np.array(ts), np.stack(feats), dict(frames=len(ts), flash_skips=fe.n_flash, ecc_failures=fe.n_ecc_fail)
