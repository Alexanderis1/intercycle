"""Extract the sampled DEV frames and stroke windows to the private labelling store (lane S4-labels, U14).

Reads labels/dev/sample.json (refuses any item that is not dev, and any clip that esk.load("dev") does not list) and
writes to $INTERCYCLE_DATA/labelling/dev/ (gitignored; the images show unmasked people, never commit or publish):

* frames/<sample_id>.png       the full-res frame at the sampled PTS (decoded PTS must equal t_pts)
* grid/<sample_id>.jpg         a viewing copy at 1280 px with a labelled 100-native-px grid
* windows/<sample_id>.npz      every frame of a stroke window: ROI crop stack (with a margin) + PTS
* crop <sample_id> x0 y0 x1 y1 an upscaled crop of a frame with a fine labelled grid (for exact box edges)
* kymo <sample_id> x0 x1       space-time strips of a stroke window (column mean of native x0..x1 per frame)

    python scripts/label_extract.py frames | windows [--sample labels/dev/sample_s1.json]
    python scripts/label_extract.py crop <sample_id> x0 y0 x1 y1 [--step 20] | kymo <sample_id> x0 x1 [y0 y1]
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


def items(sample: Path | None = None) -> list[dict]:
    """The sample list's items; refuses (SystemExit) the whole list if any item is not a dev clip of esk.load("dev")."""
    from intercycle import esk

    d = json.loads((sample or SAMPLE).read_text(encoding="utf-8"))
    if d.get("split") != "dev":
        raise SystemExit(f"refusing sample list with split {d.get('split')!r}: dev only")
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


def cmd_frames(sample: Path | None = None) -> None:
    import cv2

    s = store()
    (s / "frames").mkdir(exist_ok=True)
    (s / "grid").mkdir(exist_ok=True)
    for it in items(sample):
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


def cmd_windows(sample: Path | None = None) -> None:
    from intercycle import esk

    s = store()
    (s / "windows").mkdir(exist_ok=True)
    inv = esk.load("dev")
    fr: dict[str, str] = {str(c): str(f) for c, f in zip(inv["clip_id"], inv["framing_id"])}
    for it in items(sample):
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


def cmd_kymo(sid: str, x0: int, x1: int, y_lo: int | None = None, y_hi: int | None = None, frames_per_img: int = 90,
             px_per_frame: int = 8, yscale: int = 4, trace: bool = False) -> list[Path]:
    """Space-time strip of a stroke window: for native columns x0..x1 (inclusive), the column-mean grey of every
    decoded frame of windows/<sid>.npz, one strip column per frame, upscaled; native y on the left (optionally only
    rows y_lo..y_hi), PTS ticks on every frame (labelled every 4th, with a faint vertical line). The labeller reads the
    BDC (lowest image position of the ram band) off it."""
    import cv2

    s = store()
    z = np.load(s / "windows" / f"{sid}.npz")
    pts, crops, box = z["pts"], z["crops"], z["box"]
    bx0, by0 = int(box[0]), int(box[1])
    a, b = x0 - bx0, x1 - bx0 + 1
    if a < 0 or b > crops.shape[2] or a >= b:
        raise SystemExit(f"columns {x0}-{x1} outside the window crop x {bx0}-{bx0 + crops.shape[2] - 1}")
    ra = 0 if y_lo is None else max(0, y_lo - by0)
    rb = crops.shape[1] if y_hi is None else min(crops.shape[1], y_hi - by0 + 1)
    if ra >= rb:
        raise SystemExit(f"rows {y_lo}-{y_hi} outside the window crop")
    by0 += ra
    grey = crops[:, ra:rb, a:b, :].astype(np.float32).mean(axis=(2, 3))     # (N frames, h)
    outs: list[Path] = []
    (s / "kymo").mkdir(exist_ok=True)
    for start in range(0, len(pts), frames_per_img):
        seg = grey[start:start + frames_per_img].T                        # (h, n)
        img = cv2.resize(seg, (seg.shape[1] * px_per_frame, seg.shape[0] * yscale), interpolation=cv2.INTER_NEAREST)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) * (255.0 / max(hi - lo, 1e-6))).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pad_top, pad_left = 34, 40
        canvas = np.zeros((img.shape[0] + pad_top, img.shape[1] + pad_left, 3), np.uint8)
        canvas[pad_top:, pad_left:] = img
        for k in range(seg.shape[1]):
            x = pad_left + k * px_per_frame + px_per_frame // 2
            cv2.line(canvas, (x, pad_top - 6 if k % 4 else pad_top - 14), (x, pad_top - 1), (0, 255, 255), 1)
            if k % 4 == 0:
                cv2.line(canvas, (x, pad_top), (x, canvas.shape[0] - 1), (0, 90, 90), 1)
                cv2.putText(canvas, f"{pts[start + k]:.2f}", (x - 14, 10 + 10 * ((k // 4) % 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
        if trace:      # viewing aid: per frame, the rows of the strongest bright-above (red) / dark-above (green) edge
            prof = cv2.GaussianBlur(seg.astype(np.float32), (1, 5), 0)
            edge = prof[:-2, :] - prof[2:, :]                               # (h-2, n): g(y-1) - g(y+1)
            for k in range(seg.shape[1]):
                cx = pad_left + k * px_per_frame + px_per_frame // 2
                for yy, col in ((int(np.argmax(edge[:, k])) + 1, (0, 0, 255)),
                                (int(np.argmin(edge[:, k])) + 1, (0, 200, 0))):
                    cv2.circle(canvas, (cx, pad_top + yy * yscale + yscale // 2), 2, col, -1)
        for yy in range(0, seg.shape[0], 10):
            y = pad_top + yy * yscale
            cv2.putText(canvas, str(by0 + yy), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1)
            cv2.line(canvas, (pad_left - 4, y), (pad_left - 1, y), (0, 0, 255), 1)
        out = s / "kymo" / f"{sid}_k{x0}_{x1}_y{by0}{'_tr' if trace else ''}_{start}.png"
        cv2.imwrite(str(out), canvas)
        outs.append(out)
    return outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=["frames", "windows", "crop", "kymo"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--sample", default=None, help="sample list (default labels/dev/sample.json)")
    a = ap.parse_args(argv)
    sample = Path(a.sample) if a.sample else None
    if a.cmd == "frames":
        cmd_frames(sample)
    elif a.cmd == "windows":
        cmd_windows(sample)
    elif a.cmd == "kymo":
        sid, xa, xb, *yr = a.args
        ylo, yhi = (int(yr[0]), int(yr[1])) if yr else (None, None)
        for o in cmd_kymo(sid, int(xa), int(xb), ylo, yhi, yscale=8 if yr else 4):
            print(o)
    else:
        sid, *xy = a.args
        print(cmd_crop(sid, *map(int, xy), step=a.step, out=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
