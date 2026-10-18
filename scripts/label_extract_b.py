"""Extract the frames for the second, blind DEV labeller (labeller B; lane S4-labels-blind, U14, PLAN_TWIN §5).

Takes the FIRST N (default 20) frame items of labels/dev/sample.json in list order (stroke windows skipped), refuses
the list unless every one of its items is a dev clip of esk.load("dev"), and writes to labeller B's own private store
$INTERCYCLE_DATA/labelling_b/ (gitignored; the images show unmasked people, never commit or publish). Labeller A's
store, labels, crops, detector proposals and results are never read.

* frames/<sample_id>.png   full-res frame at the sampled PTS (decoded PTS must equal t_pts)
* grid/<sample_id>.jpg     a viewing copy at 1280 px with a labelled 100-native-px grid
* crop <sample_id> x0 y0 x1 y1 [--step 20]   an upscaled crop with a fine labelled grid (exact box edges)
* near <sample_id> dt      the frame of the same clip nearest t_pts + dt (border check: step through neighbours),
                           written as near/<sample_id>_<dt>.png and a grid view

    python scripts/label_extract_b.py frames [--n 20]
    python scripts/label_extract_b.py crop <sample_id> x0 y0 x1 y1 [--step 20] [--near DT]
    python scripts/label_extract_b.py near <sample_id> <dt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import label_extract as LE  # noqa: E402  (decode, grid_view: same decoding as labeller A's extractor)

SAMPLE = ROOT / "labels" / "dev" / "sample.json"
N_BLIND = 20


def store() -> Path:
    from intercycle import esk

    p = esk.data_root() / "labelling_b"
    p.mkdir(parents=True, exist_ok=True)
    return p


def blind_items(sample: Path | None = None, n: int = N_BLIND) -> list[dict]:
    """The first n frame items of the sample list, in list order. SystemExit if the list is not all-dev (any item) or
    holds fewer than n frames."""
    items = LE.items(sample or SAMPLE)
    frames = [it for it in items if it["kind"] == "frame"]
    if len(frames) < n:
        raise SystemExit(f"sample list holds only {len(frames)} frame items, need {n}")
    return frames[:n]


def _item(sid: str) -> dict:
    for it in blind_items():
        if it["sample_id"] == sid:
            return it
    raise SystemExit(f"{sid} is not one of labeller B's {N_BLIND} blind frames")


def cmd_frames(n: int) -> None:
    import cv2

    s = store()
    (s / "frames").mkdir(exist_ok=True)
    (s / "grid").mkdir(exist_ok=True)
    for it in blind_items(None, n):
        t = it["t_pts"]
        got = list(LE.decode(it["clip_id"], t, t))
        if len(got) != 1 or abs(got[0][0] - t) > 1e-6:
            raise SystemExit(f"{it['sample_id']}: no decoded frame at PTS {t}")
        img = got[0][1]
        cv2.imwrite(str(s / "frames" / f"{it['sample_id']}.png"), img)
        cv2.imwrite(str(s / "grid" / f"{it['sample_id']}.jpg"), LE.grid_view(img, 1280 / img.shape[1], 100),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(it["sample_id"], it["clip_id"], f"{t:.4f}", img.shape)


def cmd_near(sid: str, dt: float) -> Path:
    """The decoded frame of the same clip nearest t_pts + dt (border check), saved with a grid view."""
    import cv2

    it = _item(sid)
    want = float(it["t_pts"]) + dt
    best = None
    for p, im in LE.decode(it["clip_id"], max(0.0, want - 0.1), want + 0.1):
        if best is None or abs(p - want) < abs(best[0] - want):
            best = (p, im)
    if best is None:
        raise SystemExit(f"{sid}: no frame near t = {want:.3f}")
    s = store() / "near"
    s.mkdir(exist_ok=True)
    out = s / f"{sid}_{dt:+.2f}.png"
    cv2.imwrite(str(out), best[1])
    cv2.imwrite(str(out.with_suffix(".jpg")), LE.grid_view(best[1], 1280 / best[1].shape[1], 100),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{sid} t={best[0]:.4f}")
    return out


def cmd_crop(sid: str, x0: int, y0: int, x1: int, y1: int, step: int, near: float | None = None) -> Path:
    import cv2

    _item(sid)
    s = store()
    src = s / "frames" / f"{sid}.png" if near is None else s / "near" / f"{sid}_{near:+.2f}.png"
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"no extracted frame {src.name}")
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    crop = img[y0:y1, x0:x1]
    scale = min(4.0, 900 / max(crop.shape[:2]))
    view = LE.grid_view(crop, scale, step, origin=(x0, y0))
    tag = "" if near is None else f"_n{near:+.2f}"
    out = s / "crops" / f"{sid}{tag}_{x0}_{y0}_{x1}_{y1}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=["frames", "crop", "near"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=N_BLIND)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--near", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "frames":
        cmd_frames(a.n)
    elif a.cmd == "near":
        cmd_near(a.args[0], float(a.args[1]))
    else:
        sid, *xy = a.args
        print(cmd_crop(sid, *map(int, xy), step=a.step, near=a.near))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
