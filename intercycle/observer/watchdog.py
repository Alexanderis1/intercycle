"""M-W10: image watchdog per camera (lane S4; PLAN_TWIN §4, WEAKNESSES M-W10). numpy only.

The heartbeat (`Observation.cam_ok`) only says that frames arrive. A camera can keep sending frames that show nothing
useful, and an empty-looking image reads as empty floor (the fail-dangerous direction). This watchdog checks the
image itself and gives one bool per camera, `cam_floor_ok` of Observation v2 (mapping.py):

  FROZEN      the exact same image for >= frozen_s (encoder / pipeline stall): frame hash (BLAKE2b of the pixels)
              unchanged. Live sensor noise changes the hash every frame; a stalled stream repeats it bit for bit.
  BLACK       mean luma < dark_luma (no light, dead sensor, lens cap). A floor the camera cannot light is not seen.
  OBSTRUCTED  the whole image is flat (std < flat_std: covered lens, grey screen), or at least obstruct_share of the
              tiles that had texture in the reference lost it (tile std / image std < tile_ratio x the reference's:
              something in front of part of the lens). Both measures are gain-invariant or absolute-flat, so a
              lighting change does not trip them.
  BLURRED     normalised sharpness var(Laplacian) / var(image) < blur_ratio x the reference's (defocus, fogged or
              smeared lens). Normalised by the image variance, so it does not move with the lighting gain.
  NO_FRAME    no frame this tick (the heartbeat's job, kept so the per-camera state restarts its recovery).

Reference: per camera, given (`reference_from_frames` on train frames of the same framing) or learned from the first
`warmup_n` frames that pass the reference-free checks (median). Until a reference exists the camera is WARMUP and
not ok (fail closed). A reference learned in warm-up trusts the camera's first frames; pass one from train frames
where they exist.
Recovery: after any fault the camera stays not ok (RECOVERING) until every frame for `recover_s` passes, so UNKNOWN is
lifted only once health has returned and held (no flapping on a borderline image).
Known limits (disclosed): a frozen stream reads as healthy for up to frozen_s before it is flagged; a static scene in a
heavily compressed stream can repeat decoded frames bit for bit and is then flagged FROZEN (safe direction:
UNKNOWN); a lens cap at high gain that shows pure noise brighter than dark_luma is not caught; a moved camera
(bump) is not a watchdog fault (homography re-registration is cut, PLAN_TWIN §6).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Checks run in this order and the first that fires is the reason: BLACK, OBSTRUCTED (flat), FROZEN, BLURRED,
# OBSTRUCTED (tiles). The bool is what matters; the reason is for logs.
OK, WARMUP, RECOVERING, NO_FRAME, FROZEN, BLACK, OBSTRUCTED, BLURRED = (
    "ok", "warmup", "recovering", "no_frame", "frozen", "black", "obstructed", "blurred")
FAULTS = (NO_FRAME, FROZEN, BLACK, OBSTRUCTED, BLURRED)


@dataclass(frozen=True)
class WatchdogParams:
    frozen_s: float = 1.5          # identical frame hash for this long -> FROZEN (3 ticks at 2 Hz)
    dark_luma: float = 16.0        # mean luma (0-255) below this -> BLACK
    flat_std: float = 3.0          # image std (0-255) below this -> OBSTRUCTED (nothing in view has texture)
    tiles: tuple[int, int] = (4, 4)   # rows, cols of the tile grid for partial obstruction
    tile_ratio: float = 0.25       # a tile is dead when its relative std < tile_ratio x the reference's
    tile_live_min: float = 0.05    # reference tiles with relative std below this never had texture: not counted
    obstruct_share: float = 0.3    # dead share of the reference's textured tiles at or above this -> OBSTRUCTED
    blur_ratio: float = 0.3        # normalised sharpness below blur_ratio x the reference's -> BLURRED
    recover_s: float = 3.0         # every frame healthy for this long before the camera is ok again
    warmup_n: int = 6              # frames that set the reference when none is given


@dataclass(frozen=True)
class CamReference:
    sharp: float                   # normalised sharpness var(Laplacian) / var(image)
    tile_rel: np.ndarray           # (rows, cols) tile std / image std


@dataclass(frozen=True)
class CamHealth:
    ok: bool                       # cam_floor_ok for this camera this tick
    reason: str                    # OK, WARMUP, RECOVERING or the fault seen this tick
    t: float
    luma: float = math.nan
    std: float = math.nan
    sharp_ratio: float = math.nan  # normalised sharpness / reference
    dead_share: float = math.nan   # share of the reference's textured tiles that went flat


def gray(frame: np.ndarray) -> np.ndarray:
    """float32 luma (H, W) from (H, W) or (H, W, 3|4) uint8 / float frames (BT.601 weights on the first 3 channels)."""
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    elif a.ndim != 2:
        raise ValueError(f"frame must be (H, W) or (H, W, C), got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def frame_hash(frame: np.ndarray) -> bytes:
    a = np.ascontiguousarray(frame)
    h = hashlib.blake2b(digest_size=16)
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.digest()


def laplacian_var(g: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def tile_std(g: np.ndarray, tiles: tuple[int, int]) -> np.ndarray:
    ty, tx = tiles
    hh, ww = (g.shape[0] // ty) * ty, (g.shape[1] // tx) * tx
    if hh == 0 or ww == 0:
        raise ValueError(f"frame {g.shape} smaller than the {tiles} tile grid")
    return g[:hh, :ww].reshape(ty, hh // ty, tx, ww // tx).std(axis=(1, 3))


@dataclass(frozen=True)
class _Stats:
    luma: float
    std: float
    sharp: float
    tile_rel: np.ndarray


def _stats(frame: np.ndarray, tiles: tuple[int, int]) -> _Stats:
    g = gray(frame)
    sd = float(g.std())
    v = max(sd * sd, 1e-6)
    return _Stats(float(g.mean()), sd, laplacian_var(g) / v, tile_std(g, tiles) / max(sd, 1e-3))


def reference_from_frames(frames: Sequence[np.ndarray], params: WatchdogParams = WatchdogParams()) -> CamReference:
    """Reference of one camera / framing from healthy frames (e.g. train keyframes): medians of the image statistics."""
    st = [_stats(f, params.tiles) for f in frames]
    if not st:
        raise ValueError("no frames")
    return CamReference(float(np.median([s.sharp for s in st])), np.median(np.stack([s.tile_rel for s in st]), 0))


class CameraWatchdog:
    """Image health of ONE camera. Call `update(t, frame)` once per tick in time order (frame None = no frame)."""

    def __init__(self, params: WatchdogParams = WatchdogParams(), ref: CamReference | None = None):
        self.p = params
        self.ref0 = ref
        self.reset()

    def reset(self) -> None:
        self.ref = self.ref0
        self._warm: list[_Stats] = []
        self._hash: bytes | None = None
        self._hash_t = math.nan            # time the current hash was first seen
        self._healthy_since = math.nan     # start of the current run of passing frames
        self.last: CamHealth | None = None

    def _fault(self, t: float, frame: np.ndarray) -> tuple[str | None, _Stats]:
        p = self.p
        h = frame_hash(frame)
        if h != self._hash:
            self._hash, self._hash_t = h, t
        s = _stats(frame, p.tiles)
        if s.luma < p.dark_luma:                 # content faults first: a frozen black image reports BLACK
            return BLACK, s
        if s.std < p.flat_std:
            return OBSTRUCTED, s
        if t - self._hash_t >= p.frozen_s - 1e-9:
            return FROZEN, s
        if self.ref is not None:
            if s.sharp < p.blur_ratio * self.ref.sharp:
                return BLURRED, s
            live = self.ref.tile_rel >= p.tile_live_min
            if live.any() and float((s.tile_rel[live] < p.tile_ratio * self.ref.tile_rel[live]).mean()) >= p.obstruct_share:
                return OBSTRUCTED, s
        return None, s

    def update(self, t: float, frame: np.ndarray | None) -> CamHealth:
        p = self.p
        if frame is None:
            self._healthy_since = math.nan
            self._hash = None
            self.last = CamHealth(False, NO_FRAME, t)
            return self.last
        fault, s = self._fault(float(t), frame)
        ratio = s.sharp / self.ref.sharp if self.ref is not None and self.ref.sharp > 0 else math.nan
        dead = math.nan
        if self.ref is not None:
            live = self.ref.tile_rel >= p.tile_live_min
            dead = float((s.tile_rel[live] < p.tile_ratio * self.ref.tile_rel[live]).mean()) if live.any() else 0.0
        if fault is not None:
            self._healthy_since = math.nan
            self.last = CamHealth(False, fault, t, s.luma, s.std, ratio, dead)
            return self.last
        if not math.isfinite(self._healthy_since):
            self._healthy_since = float(t)
        if self.ref is None:
            self._warm.append(s)
            if len(self._warm) >= p.warmup_n:
                self.ref = CamReference(float(np.median([w.sharp for w in self._warm])),
                                        np.median(np.stack([w.tile_rel for w in self._warm]), 0))
                self._warm = []
            self.last = CamHealth(False, WARMUP, t, s.luma, s.std, ratio, dead)
            return self.last
        ok = t - self._healthy_since >= p.recover_s - 1e-9
        self.last = CamHealth(ok, OK if ok else RECOVERING, t, s.luma, s.std, ratio, dead)
        return self.last


class ImageWatchdog:
    """One CameraWatchdog per camera, in `cam_ok` order. `step(t, frames)` -> (n_cam,) bool = cam_floor_ok from the
    images alone (combine with the heartbeat: mapping.ObsV2Converter(obs, floor_ok=...) ANDs it with cam_ok)."""

    def __init__(self, n_cam: int, params: WatchdogParams = WatchdogParams(),
                 refs: Sequence[CamReference | None] | None = None):
        if refs is not None and len(refs) != n_cam:
            raise ValueError(f"{len(refs)} references for {n_cam} cameras")
        self.cams = [CameraWatchdog(params, None if refs is None else refs[i]) for i in range(n_cam)]

    @property
    def status(self) -> list[CamHealth | None]:
        return [c.last for c in self.cams]

    def reset(self) -> None:
        for c in self.cams:
            c.reset()

    def step(self, t: float, frames: Sequence[np.ndarray | None]) -> np.ndarray:
        if len(frames) != len(self.cams):
            raise ValueError(f"{len(frames)} frames for {len(self.cams)} cameras")
        return np.array([c.update(t, f).ok for c, f in zip(self.cams, frames)], bool)
