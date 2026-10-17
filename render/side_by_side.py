"""Acceptance check (DIRECTION §7 X): a real body-shop photo next to a synthetic render, both labelled.

    python render/side_by_side.py --real D:\\icdata\\assets\\reference\\tesla_biw_line_high_angle_jurvetson.jpg \\
        --render D:\\icdata\\renders\\first_stills\\wide.png --out render/out/compare_wide.jpg

The photo credit comes from D:\\icdata\\assets\\reference\\reference.json (author + licence of that file). Each
panel is centre-cropped to 16:9 and scaled to --panel-w; the render panel is labelled SYNTHETIC RENDER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REF_JSON = Path(r"D:\icdata\assets\reference\reference.json")


def _font(size: int):
    for f in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(path: Path, w: int, h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    r = w / h
    if im.width / im.height > r:                      # too wide: crop the sides
        cw = round(im.height * r)
        im = im.crop(((im.width - cw) // 2, 0, (im.width + cw) // 2, im.height))
    else:                                             # too tall: crop top and bottom
        ch = round(im.width / r)
        im = im.crop((0, (im.height - ch) // 2, im.width, (im.height + ch) // 2))
    return im.resize((w, h), Image.Resampling.LANCZOS)


def _label(im: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(im)
    sz = max(14, im.height // 30)
    d.text((sz // 2, sz // 2), text, fill=(255, 255, 255), font=_font(sz), stroke_width=2, stroke_fill=(0, 0, 0))


def credit(real: Path) -> str:
    refs = json.loads(REF_JSON.read_text(encoding="utf-8")) if REF_JSON.exists() else []
    r = next((x for x in refs if x["file"] == real.name), None)
    if r is None:
        raise SystemExit(f"{real.name} is not in {REF_JSON}: add its author and licence first")
    return f"REAL PHOTO: {r['author']}, {r['licence']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real body-shop photo next to a synthetic render, both labelled.")
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel-w", type=int, default=960)
    a = ap.parse_args(argv)
    w, h = a.panel_w, round(a.panel_w * 9 / 16)
    left, right = _panel(a.real, w, h), _panel(a.render, w, h)
    _label(left, credit(a.real))
    _label(right, "SYNTHETIC RENDER")
    out = Image.new("RGB", (2 * w + 8, h), (20, 20, 20))
    out.paste(left, (0, 0))
    out.paste(right, (w + 8, 0))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out, quality=90)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
