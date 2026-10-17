"""Per-camera, per-class ByteTrack (roboflow ``trackers``, Apache-2.0) on supervision Detections.

ByteTrack: Zhang et al., arXiv:2110.06864.  One tracker instance per (camera, class) so
a person track can never be matched to a forklift box.  Frames are subsampled, so the
tracker is driven with absolute timestamps (seconds) and its time budgets
(``lost_track_s``) are expressed in seconds, not in frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Dets

__all__ = ["TrackerConfig", "CameraTracker", "TrackedBoxes"]


@dataclass(frozen=True)
class TrackerConfig:
    lost_track_s: float = 3.0  # keep a lost track alive this long (seconds of video time)
    min_consecutive: int = 2  # detections in a row before a track id is issued
    high_conf: float = 0.5  # ByteTrack first-association confidence split
    activation: float = 0.4  # confidence needed to spawn a new track
    min_iou: float = 0.1  # association IoU gate


@dataclass
class TrackedBoxes:
    xyxy: np.ndarray  # (N, 4)
    score: np.ndarray  # (N,)
    track_id: np.ndarray  # (N,) int, unique per (camera, class)
    label: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return int(len(self.score))


class CameraTracker:
    """One camera's trackers (one ByteTrack per class)."""

    def __init__(self, fps: float, classes: tuple[str, ...], cfg: TrackerConfig | None = None):
        from trackers import ByteTrackTracker

        self.cfg = cfg or TrackerConfig()
        self.fps = float(fps)
        self.classes = tuple(classes)
        self._trackers = {
            c: ByteTrackTracker(
                lost_track_buffer=max(1, int(round(self.cfg.lost_track_s * 30.0))),  # unit: 30-fps frames
                frame_rate=self.fps,
                track_activation_threshold=self.cfg.activation,
                minimum_consecutive_frames=self.cfg.min_consecutive,
                minimum_iou_threshold=self.cfg.min_iou,
                high_conf_det_threshold=self.cfg.high_conf,
            )
            for c in self.classes
        }

    def update(self, dets: Dets, t: float) -> TrackedBoxes:
        """Feed one frame's detections (timestamp ``t`` in seconds); return confirmed tracks."""
        import supervision as sv

        xs, ss, ids, ls = [], [], [], []
        for cls, trk in self._trackers.items():
            m = dets.label == cls
            sd = sv.Detections(
                xyxy=dets.xyxy[m].astype(np.float32).reshape(-1, 4),
                confidence=dets.score[m].astype(np.float32),
                class_id=np.zeros(int(m.sum()), dtype=int),
            )
            out = trk.update(sd, timestamp=float(t))
            if out is None or len(out) == 0 or out.tracker_id is None:
                continue
            tid = np.asarray(out.tracker_id, dtype=np.int64)
            ok = tid >= 0  # -1 = not yet confirmed
            if not ok.any():
                continue
            xs.append(np.asarray(out.xyxy, np.float32)[ok])
            conf = out.confidence if out.confidence is not None else np.ones(len(tid), np.float32)
            ss.append(np.asarray(conf, np.float32)[ok])
            ids.append(tid[ok])
            ls.append(np.full(int(ok.sum()), cls, dtype="<U16"))
        if not xs:
            return TrackedBoxes(
                np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), np.array([], dtype="<U16")
            )
        return TrackedBoxes(np.concatenate(xs), np.concatenate(ss), np.concatenate(ids), np.concatenate(ls))
