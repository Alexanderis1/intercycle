"""Write people-masked frames / crops of TRAIN and DEV press-shop clips to ``$INTERCYCLE_DATA/blurred/`` (PLAN_TWIN §3 P8).

Rule (intercycle/perception/privacy_mask.py): every RF-DETR person box at conf >= 0.1 is black-boxed whole, padded 15 %,
and held over its track (same track id, and through detector gaps up to 1 s). Detection runs on every written frame.

    # GPU env (torch + rfdetr + av):  F:/envs/intercycle-gpu/Scripts/python.exe scripts/esk_mask.py --split dev --fps 2
    python scripts/esk_mask.py --split train dev --fps 2 [--clips 0_tr1 3_tr4] [--crop 600,200,1400,900] [--keyframes]

Inputs:  $INTERCYCLE_DATA/inventory.parquet (via intercycle.esk.load, which refuses TEST), clips/<clip>.mp4,
         keyframes/<clip>_k{1,2}.jpg (with --keyframes).
Outputs: blurred/<split>/<clip>/<t_ms>.jpg  (the masked full frame, or the masked crop with --crop),
         blurred/<split>/<clip>/mask.json  (config, frames written, boxes masked, detector),
         blurred/keyframes/<clip>_k{1,2}.jpg  (single-frame mask, no track hold).
Times come from the container PTS (seconds since the first frame), never frame_index / fps.

TEST is never processed: the split list is checked twice (argument parser and every inventory row), and
intercycle.esk.load raises on TEST outside a sealed run. Unmasked frames are never written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle.perception.privacy_mask import MaskConfig, TrackHoldMasker, apply_mask  # noqa: E402

ALLOWED_SPLITS = ("train", "dev")
DetectFn = Callable[[list[np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]  # images -> [(xyxy, score)]


def iter_pts_frames(path: str | Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """(t, RGB frame) at about ``fps`` frames per second; t = PTS seconds since the first frame."""
    import av

    step = 1.0 / fps if fps > 0 else 0.0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        t0 = None
        t_next = 0.0
        for f in c.decode(s):
            if f.pts is None or f.time is None:
                continue
            if t0 is None:
                t0 = float(f.time)
            t = float(f.time) - t0
            if t + 1e-6 < t_next:
                continue
            t_next = (np.floor(t / step + 1e-6) + 1) * step if step > 0 else t  # fixed grid k*step: no drift
            yield t, f.to_ndarray(format="rgb24")


def _write_jpg(path: Path, rgb: np.ndarray, quality: int = 90) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"could not write {path}")


def _crop(rgb: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return rgb
    x0, y0, x1, y1 = crop
    return rgb[y0:y1, x0:x1]


def mask_clip(video: str | Path, out_dir: str | Path, detect: DetectFn, cfg: MaskConfig, fps: float,
              crop: tuple[int, int, int, int] | None = None, batch: int = 8) -> dict:
    """Mask one clip; the frame is masked BEFORE any crop, so a person straddling the crop edge is still covered."""
    out_dir = Path(out_dir)
    masker = TrackHoldMasker(cfg)
    n_frames = n_boxes = 0
    buf: list[tuple[float, np.ndarray]] = []

    def flush() -> None:
        nonlocal n_frames, n_boxes
        if not buf:
            return
        dets = detect([im for _, im in buf])
        if len(dets) != len(buf):
            raise RuntimeError("detector returned a different number of results than frames")
        for (t, im), (xyxy, score) in zip(buf, dets):
            masker.width, masker.height = im.shape[1], im.shape[0]
            boxes = masker.update(t, xyxy, score)
            n_boxes += len(boxes)
            _write_jpg(out_dir / f"{int(round(t * 1000)):06d}.jpg", _crop(apply_mask(im, boxes), crop))
            n_frames += 1
        buf.clear()

    for t, im in iter_pts_frames(video, fps):
        buf.append((t, im))
        if len(buf) >= batch:
            flush()
    flush()
    return {"frames": n_frames, "boxes_masked": n_boxes}


def mask_image(src: str | Path, dst: str | Path, detect: DetectFn, cfg: MaskConfig) -> int:
    """Single still (keyframe): mask every person box >= cfg.conf, padded; no track to hold."""
    import cv2

    bgr = cv2.imread(str(src))
    if bgr is None:
        raise FileNotFoundError(src)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    xyxy, score = detect([rgb])[0]
    boxes = TrackHoldMasker(cfg, width=rgb.shape[1], height=rgb.shape[0]).update(0.0, xyxy, score)
    _write_jpg(Path(dst), apply_mask(rgb, boxes))
    return len(boxes)


def rfdetr_detect_fn(size: str, conf: float) -> DetectFn:
    from intercycle.perception.detector import PERSON, RFDetrDetector

    det = RFDetrDetector(size=size, threshold=conf, keep=(PERSON,))

    def fn(images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(d.xyxy, d.score) for d in det.detect(images)]

    return fn


def main(argv: list[str] | None = None, detect: DetectFn | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--split", nargs="+", default=list(ALLOWED_SPLITS), choices=ALLOWED_SPLITS)
    ap.add_argument("--clips", nargs="*", default=None, help="subset of clip ids (must be in the chosen splits)")
    ap.add_argument("--fps", type=float, default=2.0, help="frames written (and detected) per second")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 in source pixels; masked before cropping")
    ap.add_argument("--keyframes", action="store_true", help="also mask keyframes/<clip>_k{1,2}.jpg")
    ap.add_argument("--size", default="medium", help="RF-DETR size")
    ap.add_argument("--root", default=None, help="data root (default $INTERCYCLE_DATA)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    cfg = MaskConfig()
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    if crop is not None and len(crop) != 4:
        ap.error("--crop needs x0,y0,x1,y1")
    root = esk.data_root(a.root)
    inv = esk.load(a.split, root=root)  # raises on TEST outside a sealed run
    if a.clips:
        unknown = sorted(set(a.clips) - set(inv["clip_id"]))
        if unknown:
            ap.error(f"clips not in {a.split}: {unknown[:5]}")
        inv = inv[inv["clip_id"].isin(a.clips)]
    if detect is None:
        detect = rfdetr_detect_fn(a.size, cfg.conf)

    out_root = root / "blurred"
    done = 0
    for cid, split in zip(inv["clip_id"], inv["split"]):
        if split not in ALLOWED_SPLITS:  # defence in depth
            raise RuntimeError(f"refusing {cid}: split {split!r}")
        video = esk.clip_path(cid, root=root)
        out_dir = out_root / split / cid
        if (out_dir / "mask.json").exists() and not a.overwrite:
            continue
        if not video.exists():
            print(f"missing clip {video}", file=sys.stderr)
            continue
        stats = mask_clip(video, out_dir, detect, cfg, a.fps, crop)
        meta = {"clip_id": cid, "split": split, "fps": a.fps, "crop": crop, "detector": f"rfdetr-{a.size}",
                "rule": {"conf": cfg.conf, "pad": cfg.pad, "hold_s": cfg.hold_s, "grow_per_s": cfg.grow_per_s},
                **stats}
        (out_dir / "mask.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if a.keyframes:
            for k in (1, 2):
                src = root / "keyframes" / f"{cid}_k{k}.jpg"
                if src.exists():
                    mask_image(src, out_root / "keyframes" / src.name, detect, cfg)
        done += 1
        print(f"{cid} ({split}): {stats['frames']} frames, {stats['boxes_masked']} boxes masked")
    print(f"done: {done} clips -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
