"""CCTV wall: four post-processed CCTV sequences (cctv_post output frames, same clock) tiled 2x2 into one frame.

    python -m render.cctv.wall --tiles D:\\icdata\\renders\\wall\\cctv_0\\cctv\\frames D:\\icdata\\renders\\seq_sim1\\cctv_1\\cctv\\frames \\
        D:\\icdata\\renders\\wall\\cctv_5\\cctv\\frames D:\\icdata\\renders\\wall\\cctv_6\\cctv\\frames --out D:\\icdata\\renders\\wall\\out

Tiles are read in order (top-left, top-right, bottom-left, bottom-right), scaled to half the wall size (INTER_AREA)
and keep their own burned-in timestamp / camera / SYNTHETIC text; the wall adds one FILM_TEXT label bottom-centre.
Every tile must have the same frame count (same t0 and fps). Writes <out>/frames/w_00000.png.. and <out>/wall.mp4.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .cctv_post import _COLOR_ENC, _VF_ENC, _ffmpeg, _font, load_rgb, save_rgb

FILM_TEXT = "SYNTHETIC REPLAY OF OUR SIMULATION"   # = scene_real.FILM_TEXT
GUTTER = 4                                          # px, dark seam between tiles


def compose(tiles: list[np.ndarray], w: int, h: int, text: str = FILM_TEXT) -> np.ndarray:
    """2x2 wall of `tiles` (RGB uint8, any size) at w x h with seams and one label."""
    tw, th = w // 2, h // 2
    out = np.zeros((h, w, 3), np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, 2)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
    g = GUTTER // 2
    out[th - g:th + g, :] = 12
    out[:, tw - g:tw + g] = 12
    im = Image.fromarray(out)
    sz = max(12, h // 45)
    ImageDraw.Draw(im).text((w // 2, h - sz), text, font=_font(sz), anchor="md", fill=(255, 255, 255),
                            stroke_width=max(1, sz // 10), stroke_fill=(0, 0, 0))
    return np.asarray(im)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2x2 CCTV wall from four post-processed CCTV frame folders.")
    ap.add_argument("--tiles", nargs=4, type=Path, required=True, help="frame folders: TL TR BL BR")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.split("x"))
    lists = [sorted(d.glob("*.png")) for d in a.tiles]
    counts = {len(fs) for fs in lists}
    if len(counts) != 1 or 0 in counts:
        raise SystemExit(f"tiles need the same, non-zero frame count; got {[len(fs) for fs in lists]}")
    frames = a.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("w_*.png"):        # a shorter re-run must not leave the old tail
        old.unlink()
    for i, paths in enumerate(zip(*lists)):
        save_rgb(frames / f"w_{i:05d}.png", compose([load_rgb(p) for p in paths], w, h))
    mp4 = a.out / "wall.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "w_%05d.png"),
                    "-frames:v", str(len(lists[0])), "-vf", _VF_ENC, "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", *_COLOR_ENC, "-movflags", "+faststart", str(mp4)], check=True)   # same colour tags as cctv_post
    print("wrote", mp4, f"({len(lists[0])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
