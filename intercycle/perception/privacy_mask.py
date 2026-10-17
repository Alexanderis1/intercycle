"""Track-held privacy mask for real press-shop frames (PLAN_TWIN §3 P8, §2 privacy row, U7; WEAKNESSES G-W1).

Rule (fixed by the plan, whatever the P8 face-size histogram says):
* every person box with detector confidence >= ``conf`` (0.1) is black-boxed WHOLE (filled, not blurred),
  padded by ``pad`` (15 %) of its width / height on each side;
* the mask is HELD over the track: a person's mask persists across the frames of the same track id and through short
  detector gaps (up to ``hold_s`` seconds after the track's last detection). While held, the last box grows by
  ``grow_per_s`` box-heights per second of gap on each side, so a person who keeps walking stays covered.

Tracks: the caller may pass track ids (e.g. from ``CameraTracker``). Boxes without an id (ByteTrack only confirms tracks
above its activation confidence, so a 0.1-conf box usually has none) are associated here by greedy IoU against the held
tracks, so every masked box belongs to a track and can be held. Masking never depends on a track being confirmed: every
box >= ``conf`` is masked in its own frame.

Numpy only, no detector import, so the masker is testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MaskConfig", "TrackHoldMasker", "pad_boxes", "apply_mask"]


@dataclass(frozen=True)
class MaskConfig:
    conf: float = 0.1  # every person box at or above this confidence is masked
    pad: float = 0.15  # padding, fraction of the box width / height, on each side
    hold_s: float = 1.0  # keep masking a track this long after its last detection
    grow_per_s: float = 0.5  # held box grows by this many box-heights per second of gap, on each side
    assoc_iou: float = 0.1  # min IoU (padded+grown held box vs new box) to continue a track without an id


@dataclass
class _Track:
    box: np.ndarray  # (4,) last raw (unpadded) xyxy
    t_last: float


def pad_boxes(xyxy: np.ndarray, pad: float) -> np.ndarray:
    """Grow each (x0, y0, x1, y1) box by ``pad`` x its width / height on each side."""
    b = np.asarray(xyxy, np.float64).reshape(-1, 4)
    w = (b[:, 2] - b[:, 0])[:, None]
    h = (b[:, 3] - b[:, 1])[:, None]
    return b + np.hstack([-pad * w, -pad * h, pad * w, pad * h])


def _grow_abs(xyxy: np.ndarray, d: np.ndarray) -> np.ndarray:
    b = np.asarray(xyxy, np.float64).reshape(-1, 4)
    d = np.asarray(d, np.float64).reshape(-1, 1)
    return b + np.hstack([-d, -d, d, d])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, a (N, 4) x b (M, 4) -> (N, M)."""
    a = a.reshape(-1, 4)[:, None, :]
    b = b.reshape(-1, 4)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


@dataclass
class TrackHoldMasker:
    """Feed frames in time order; ``update`` returns the boxes to black out in that frame (padded, held, clipped)."""

    cfg: MaskConfig = field(default_factory=MaskConfig)
    width: int | None = None  # frame size, for clipping (optional)
    height: int | None = None
    _tracks: dict = field(default_factory=dict, init=False, repr=False)
    _next_auto: int = field(default=-1, init=False, repr=False)  # auto ids are negative, caller ids are >= 0
    _t_prev: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Forget every track (call between clips)."""
        self._tracks.clear()
        self._next_auto = -1
        self._t_prev = None

    def _held_box(self, tr: _Track, t: float) -> np.ndarray:
        gap = max(0.0, t - tr.t_last)
        h = float(tr.box[3] - tr.box[1])
        return _grow_abs(pad_boxes(tr.box, self.cfg.pad), np.array([self.cfg.grow_per_s * h * gap]))[0]

    def update(self, t: float, xyxy, score, track_id=None) -> np.ndarray:
        """Boxes (K, 4) float to mask at time ``t`` (seconds). ``track_id`` entries < 0 or None mean "no id"."""
        if self._t_prev is not None and t < self._t_prev:
            raise ValueError(f"frames must come in time order (t={t} < {self._t_prev}); call reset() between clips")
        self._t_prev = t
        cfg = self.cfg
        b = np.asarray(xyxy, np.float64).reshape(-1, 4)
        s = np.asarray(score, np.float64).reshape(-1)
        ids = (np.full(len(b), -1, np.int64) if track_id is None
               else np.asarray([(-1 if v is None else int(v)) for v in track_id], np.int64).reshape(-1))
        if not (len(b) == len(s) == len(ids)):
            raise ValueError("xyxy, score and track_id must have the same length")
        keep = s >= cfg.conf
        b, ids = b[keep], ids[keep]

        # drop tracks whose hold has expired (before association, so an expired track is never revived)
        for k in [k for k, tr in self._tracks.items() if t - tr.t_last > cfg.hold_s]:
            del self._tracks[k]

        seen: set = set()
        # boxes with a caller id continue (or start) that track
        for i in np.flatnonzero(ids >= 0):
            k = ("id", int(ids[i]))
            self._tracks[k] = _Track(b[i].copy(), t)
            seen.add(k)
        # boxes without an id: greedy IoU against held tracks not yet updated in this frame
        free = np.flatnonzero(ids < 0)
        if len(free):
            cand = [k for k in self._tracks if k not in seen]
            if cand:
                held = np.stack([self._held_box(self._tracks[k], t) for k in cand])
                iou = _iou(b[free], held)
                order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
                used_b, used_t = set(), set()
                for bi, ti in order:
                    if iou[bi, ti] < cfg.assoc_iou:
                        break
                    if bi in used_b or ti in used_t:
                        continue
                    used_b.add(int(bi))
                    used_t.add(int(ti))
                    k = cand[ti]
                    self._tracks[k] = _Track(b[free[bi]].copy(), t)
                    seen.add(k)
                free = np.array([f for j, f in enumerate(free) if j not in used_b], np.int64)
            for i in free:
                k = ("auto", self._next_auto)
                self._next_auto -= 1
                self._tracks[k] = _Track(b[i].copy(), t)
                seen.add(k)

        # every live track: detected now -> its padded box; in a gap -> its held (padded, grown) box
        out = [self._held_box(tr, t) for tr in self._tracks.values()]
        if not out:
            return np.zeros((0, 4), np.float64)
        m = np.stack(out)
        if self.width is not None:
            m[:, [0, 2]] = np.clip(m[:, [0, 2]], 0, self.width)
        if self.height is not None:
            m[:, [1, 3]] = np.clip(m[:, [1, 3]], 0, self.height)
        return m


def apply_mask(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Copy of ``image`` (H, W, C) with every box filled black. Boxes are rounded outwards (floor / ceil)."""
    out = np.array(image, copy=True)
    H, W = out.shape[:2]
    for x0, y0, x1, y1 in np.asarray(boxes, np.float64).reshape(-1, 4):
        c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c1, r1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
        if c1 > c0 and r1 > r0:
            out[r0:r1, c0:c1] = 0
    return out
