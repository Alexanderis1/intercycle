"""Extract the sampled DEV frames and stroke windows to the private labelling store (lane S4-labels, U14).

Reads labels/dev/sample.json (refuses any item that is not dev, and any clip that esk.load("dev") does not list) and
writes to $INTERCYCLE_DATA/labelling/dev/ (gitignored; the images show unmasked people, never commit or publish):

* frames/<sample_id>.png       the full-res frame at the sampled PTS (decoded PTS must equal t_pts)
* grid/<sample_id>.jpg         a viewing copy at 1280 px with a labelled 100-native-px grid
* windows/<sample_id>.npz      every frame of a stroke window: ROI crop stack (with a margin) + PTS
* crop <sample_id> x0 y0 x1 y1 an upscaled crop of a frame with a fine labelled grid (for exact box edges)

    python scripts/label_extract.py frames | windows | crop <sample_id> x0 y0 x1 y1 [--step 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "labels" / "dev" / "sample.json"
ROIS = ROOT / "docs" / "probes" / "S3a-P4_rois.json"
ROI_MARGIN = 40


def store() -> Path:
    from intercycle import esk

    p = esk.data_root() / "labelling" / "dev"
    p.mkdir(parents=True, exist_ok=True)
    return p


def items() -> list[dict]:
    from intercycle import esk

    d = json.loads(SAMPLE.read_text(encoding="utf-8"))
    dev = esk.load("dev").set_index("clip_id")
    for it in d["items"]:
        if it["split"] != "dev" or it["clip_id"] not in dev.index:
            raise SystemExit(f"refusing {it['sample_id']}: {it['clip_id']} is not a dev clip")
    return d["items"]


def decode(clip_id: str, t_lo: float, t_hi: float):
    """Yield (pts, bgr) for every frame with t_lo - 1e-6 <= pts <= t_hi + 1e-6."""
    import av  # pyright: ignore[reportMissingImports]
    from intercycle import esk

    with av.open(str(esk.clip_path(clip_id))) as c:
        st = c.streams.video[0]
        tb = st.time_base
        assert tb is not None
        for f in c.decode(st):
            if f.pts is None:
                continue
            t = float(f.pts * tb)
            if t > t_hi + 1e-6:
                break
            if t >= t_lo - 1e-6:
                yield t, f.to_ndarray(format="bgr24")


def grid_view(img: np.ndarray, scale: float, step: int, origin=(0, 0)) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ox, oy = origin
    fs = 0.4 if scale < 1 else 0.45
    for gx in range((ox // step + 1) * step if ox % step else ox, ox + w + 1, step):
        x = int(round((gx - ox) * scale))
        cv2.line(out, (x, 0), (x, out.shape[0] - 1), (0, 255, 255), 1)
        cv2.putText(out, str(gx), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
    for gy in range((oy // step + 1) * step if oy % step else oy, oy + h + 1, step):
        y = int(round((gy - oy) * scale))
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (0, 255, 255), 1)
        cv2.putText(out, str(gy), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
    return out


def cmd_frames() -> None:
    import cv2

    s = store()
    (s / "frames").mkdir(exist_ok=True)
    (s / "grid").mkdir(exist_ok=True)
    for it in items():
        if it["kind"] != "frame":
            continue
        t = it["t_pts"]
        got = [(p, im) for p, im in decode(it["clip_id"], t, t)]
        if len(got) != 1 or abs(got[0][0] - t) > 1e-6:
            raise SystemExit(f"{it['sample_id']}: no decoded frame at PTS {t}")
        img = got[0][1]
        cv2.imwrite(str(s / "frames" / f"{it['sample_id']}.png"), img)
        cv2.imwrite(str(s / "grid" / f"{it['sample_id']}.jpg"), grid_view(img, 1280 / img.shape[1], 100),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(it["sample_id"], it["clip_id"], f"{t:.4f}", img.shape)


def roi_box(press: str, framing: str) -> tuple[int, int, int, int]:
    d = json.loads(ROIS.read_text(encoding="utf-8"))
    poly = np.asarray(next(p for p in d["presses"] if p["id"] == press)["rois"][framing], float)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    return (max(0, int(x0) - ROI_MARGIN), max(0, int(y0) - ROI_MARGIN), min(1920, int(x1) + ROI_MARGIN),
            min(1080, int(y1) + ROI_MARGIN))


def cmd_windows() -> None:
    from intercycle import esk

    s = store()
    (s / "windows").mkdir(exist_ok=True)
    inv = esk.load("dev")
    fr: dict[str, str] = {str(c): str(f) for c, f in zip(inv["clip_id"], inv["framing_id"])}
    for it in items():
        if it["kind"] != "stroke_window":
            continue
        x0, y0, x1, y1 = roi_box(it["press"], fr[it["clip_id"]])
        pts, crops = [], []
        for p, im in decode(it["clip_id"], it["t0"], it["t1"]):
            pts.append(p)
            crops.append(im[y0:y1, x0:x1])
        if not pts or abs(pts[0] - it["t0"]) > 1e-6 or abs(pts[-1] - it["t1"]) > 1e-6:
            raise SystemExit(f"{it['sample_id']}: decoded window does not span [{it['t0']}, {it['t1']}]")
        np.savez_compressed(s / "windows" / f"{it['sample_id']}.npz", pts=np.asarray(pts), crops=np.stack(crops),
                            box=np.asarray([x0, y0, x1, y1]), framing=fr[it["clip_id"]])
        print(it["sample_id"], it["clip_id"], it["press"], fr[it["clip_id"]], len(pts), (x0, y0, x1, y1))


def cmd_crop(sid: str, x0: int, y0: int, x1: int, y1: int, step: int, out: Path | None) -> Path:
    import cv2

    s = store()
    img = cv2.imread(str(s / "frames" / f"{sid}.png"))
    if img is None:
        raise SystemExit(f"no extracted frame {sid}")
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    crop = img[y0:y1, x0:x1]
    scale = min(4.0, 900 / max(crop.shape[:2]))
    view = grid_view(crop, scale, step, origin=(x0, y0))
    out = out or (s / "crops" / f"{sid}_{x0}_{y0}_{x1}_{y1}.jpg")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=["frames", "windows", "crop"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "frames":
        cmd_frames()
    elif a.cmd == "windows":
        cmd_windows()
    else:
        sid, *xy = a.args
        print(cmd_crop(sid, *map(int, xy), step=a.step, out=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
