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

SEALED mode (custodian session only): `--sealed` reads the hashed TEST sample list
C:\\intercycle_sealed\\labels\\sample.json (checked against its .sha256; every item must be a test row of the active
split, else the whole list is refused), decodes C:\\intercycle_sealed\\clips\\<clip_id>.mp4 and writes the same outputs
to C:\\intercycle_sealed\\labelling\\ (never into the repo or $INTERCYCLE_DATA). `--sealed-root` moves that root.

    python scripts/label_extract.py --sealed frames | windows
    python scripts/label_extract.py --sealed crop <sample_id> x0 y0 x1 y1 [--step 20] | kymo <sample_id> x0 x1 [y0 y1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "labels" / "dev" / "sample.json"
ROIS = ROOT / "docs" / "probes" / "S3a-P4_rois.json"
ROI_MARGIN = 40
SEALED_ROOT = Path(r"C:\intercycle_sealed")


@dataclass(frozen=True)
class Sealed:
    """Where the sealed TEST mode reads and writes: <root>/labels/sample.json, <root>/clips, <root>/labelling."""
    root: Path = SEALED_ROOT

    @property
    def sample(self) -> Path:
        return self.root / "labels" / "sample.json"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def store(self) -> Path:
        return self.root / "labelling"


def _inside(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sealed_store(sealed: Sealed) -> Path:
    """The sealed labelling store; refuses a location inside the repo or $INTERCYCLE_DATA."""
    p = sealed.store
    data = os.environ.get("INTERCYCLE_DATA")
    if _inside(p, ROOT) or (data and _inside(p, Path(data))):
        raise SystemExit(f"refusing sealed store {p}: inside the repo or $INTERCYCLE_DATA")
    p.mkdir(parents=True, exist_ok=True)
    return p


def sealed_framings(split_csv: Path | None = None) -> dict[str, str]:
    """clip_id -> framing_id of the TEST rows of the active split (train/dev rows are never returned)."""
    import csv
    import io

    from twin.schemas.prereg import active_split_path, parse_split

    path = split_csv or active_split_path()
    text = path.read_text(encoding="utf-8")
    rows = parse_split(text)
    rd = list(csv.DictReader(io.StringIO(text)))
    if not rd or "framing_id" not in rd[0]:
        raise SystemExit(f"{path.name} has no framing_id column (fail closed)")
    return {r["clip_id"]: r["framing_id"] for r in rd if rows[r["clip_id"]].split == "test"}


def sealed_items(sealed: Sealed, split_csv: Path | None = None) -> list[dict]:
    """The sealed TEST list's items. Refuses (SystemExit) the whole list unless it validates as a SampleList of split
    test, its bytes match the sha256 file written next to it, and every item is a test row of the active split."""
    from twin.schemas.labels import SampleList

    path = sealed.sample
    data = path.read_bytes()
    sha_file = path.with_suffix(".sha256")
    if not sha_file.is_file():
        raise SystemExit(f"refusing {path.name}: no {sha_file.name} next to it (the list is hashed before labelling)")
    want = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in sha_file.read_text(encoding="utf-8").splitlines()
            if "  " in ln}
    if want.get(path.name) != hashlib.sha256(data).hexdigest():
        raise SystemExit(f"refusing {path.name}: its bytes do not match {sha_file.name}")
    try:
        sl = SampleList.model_validate(json.loads(data.decode("utf-8")))
    except ValueError as e:
        raise SystemExit(f"refusing {path.name}: not a valid sample list ({type(e).__name__})") from None
    if sl.split != "test":
        raise SystemExit(f"refusing sample list with split {sl.split!r}: test only in --sealed mode")
    test = sealed_framings(split_csv)
    out = []
    for it in sl.items:
        if it.split != "test" or it.clip_id not in test:
            raise SystemExit(f"refusing {it.sample_id}: {it.clip_id} is not a test clip of the active split")
        out.append(it.model_dump(mode="json"))
    return out


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


def decode(clip_id: str, t_lo: float, t_hi: float, path: Path | None = None):
    """Yield (pts, bgr) for every frame with t_lo - 1e-6 <= pts <= t_hi + 1e-6 (of `path`, default the dev clip)."""
    import av  # pyright: ignore[reportMissingImports]
    from intercycle import esk

    with av.open(str(path or esk.clip_path(clip_id))) as c:
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


def _clip_fn(sealed: Sealed | None) -> Callable[[str], Path | None]:
    return (lambda c: sealed.clips / f"{c}.mp4") if sealed else (lambda c: None)


def cmd_frames(sample: Path | None = None, sealed: Sealed | None = None) -> None:
    import cv2

    s = sealed_store(sealed) if sealed else store()
    clip = _clip_fn(sealed)
    (s / "frames").mkdir(exist_ok=True)
    (s / "grid").mkdir(exist_ok=True)
    for it in (sealed_items(sealed) if sealed else items(sample)):
        if it["kind"] != "frame":
            continue
        t = it["t_pts"]
        got = [(p, im) for p, im in decode(it["clip_id"], t, t, clip(it["clip_id"]))]
        if len(got) != 1 or abs(got[0][0] - t) > 1e-6:
            raise SystemExit(f"{it['sample_id']}: no decoded frame at PTS {t}")
        img = got[0][1]
        cv2.imwrite(str(s / "frames" / f"{it['sample_id']}.png"), img)
        cv2.imwrite(str(s / "grid" / f"{it['sample_id']}.jpg"), grid_view(img, 1280 / img.shape[1], 100),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if sealed:
            print(it["sample_id"], f"{t:.4f}", img.shape)
        else:
            print(it["sample_id"], it["clip_id"], f"{t:.4f}", img.shape)


def roi_box(press: str, framing: str) -> tuple[int, int, int, int]:
    d = json.loads(ROIS.read_text(encoding="utf-8"))
    poly = np.asarray(next(p for p in d["presses"] if p["id"] == press)["rois"][framing], float)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    return (max(0, int(x0) - ROI_MARGIN), max(0, int(y0) - ROI_MARGIN), min(1920, int(x1) + ROI_MARGIN),
            min(1080, int(y1) + ROI_MARGIN))


def cmd_windows(sample: Path | None = None, sealed: Sealed | None = None) -> None:
    clip = _clip_fn(sealed)
    if sealed:
        s = sealed_store(sealed)
        its = sealed_items(sealed)
        fr = sealed_framings()
    else:
        from intercycle import esk

        s = store()
        inv = esk.load("dev")
        fr = {str(c): str(f) for c, f in zip(inv["clip_id"], inv["framing_id"])}
        its = items(sample)
    (s / "windows").mkdir(exist_ok=True)
    for it in its:
        if it["kind"] != "stroke_window":
            continue
        x0, y0, x1, y1 = roi_box(it["press"], fr[it["clip_id"]])
        pts, crops = [], []
        for p, im in decode(it["clip_id"], it["t0"], it["t1"], clip(it["clip_id"])):
            pts.append(p)
            crops.append(im[y0:y1, x0:x1])
        if not pts or abs(pts[0] - it["t0"]) > 1e-6 or abs(pts[-1] - it["t1"]) > 1e-6:
            raise SystemExit(f"{it['sample_id']}: decoded window does not span [{it['t0']}, {it['t1']}]")
        np.savez_compressed(s / "windows" / f"{it['sample_id']}.npz", pts=np.asarray(pts), crops=np.stack(crops),
                            box=np.asarray([x0, y0, x1, y1]), framing=fr[it["clip_id"]])
        if sealed:
            print(it["sample_id"], it["press"], len(pts), (x0, y0, x1, y1))
        else:
            print(it["sample_id"], it["clip_id"], it["press"], fr[it["clip_id"]], len(pts), (x0, y0, x1, y1))


def cmd_crop(sid: str, x0: int, y0: int, x1: int, y1: int, step: int, out: Path | None,
             sealed: Sealed | None = None) -> Path:
    import cv2

    s = sealed_store(sealed) if sealed else store()
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
             px_per_frame: int = 8, yscale: int = 4, trace: bool = False, sealed: Sealed | None = None) -> list[Path]:
    """Space-time strip of a stroke window: for native columns x0..x1 (inclusive), the column-mean grey of every
    decoded frame of windows/<sid>.npz, one strip column per frame, upscaled; native y on the left (optionally only
    rows y_lo..y_hi), PTS ticks on every frame (labelled every 4th, with a faint vertical line). The labeller reads the
    BDC (lowest image position of the ram band) off it."""
    import cv2

    s = sealed_store(sealed) if sealed else store()
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
    ap.add_argument("--sealed", action="store_true",
                    help=f"custodian only: the sealed TEST list under {SEALED_ROOT} (labels/sample.json, clips/, "
                         f"labelling/)")
    ap.add_argument("--sealed-root", default=None, help=f"--sealed: another sealed root (default {SEALED_ROOT})")
    a = ap.parse_args(argv)
    if a.sealed and a.sample:
        ap.error("--sealed reads <sealed root>/labels/sample.json only; --sample is the dev list")
    if a.sealed_root and not a.sealed:
        ap.error("--sealed-root needs --sealed")
    sealed = Sealed(Path(a.sealed_root)) if a.sealed_root else (Sealed() if a.sealed else None)
    sample = Path(a.sample) if a.sample else None
    if a.cmd == "frames":
        cmd_frames(sample, sealed)
    elif a.cmd == "windows":
        cmd_windows(sample, sealed)
    elif a.cmd == "kymo":
        sid, xa, xb, *yr = a.args
        ylo, yhi = (int(yr[0]), int(yr[1])) if yr else (None, None)
        for o in cmd_kymo(sid, int(xa), int(xb), ylo, yhi, yscale=8 if yr else 4, sealed=sealed):
            print(o)
    else:
        sid, *xy = a.args
        print(cmd_crop(sid, *map(int, xy), step=a.step, out=None, sealed=sealed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
