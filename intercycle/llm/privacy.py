"""Privacy filter applied to every image before it is sent to a VLM (PRODUCTION_BRIEF §A6).

* every person box (from the detector, or the frame ground truth for SYNTHETIC frames) is blacked out (filled, not
  blurred), padded by 15 % of the box size on each side;
* the image is downscaled (longest side <= `max_side`) and re-encoded as a fresh JPEG from pixel data only, so no
  EXIF / XMP / PNG text metadata survives.

Production runs the same filter on the edge box before an on-prem model; no video leaves the site.
"""
from __future__ import annotations

import io
from typing import Any, Sequence

from PIL import Image, ImageDraw

PAD = 0.15


def _open(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return Image.open(img).convert("RGB")


def mask_boxes(im: Image.Image, boxes: Sequence[Sequence[float]], pad: float = PAD) -> Image.Image:
    """Black-fill each (x0, y0, x1, y1) box in pixel coordinates, grown by `pad` x its width/height per side."""
    if not boxes:
        return im
    im = im.copy()
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        d.rectangle([x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h], fill=(0, 0, 0))
    return im


def filtered_jpeg(img: Any, person_boxes: Sequence[Sequence[float]] = (), max_side: int = 1280, quality: int = 85) -> bytes:
    """People masked, downscaled, metadata-free JPEG bytes. `person_boxes` are in the input image's pixels."""
    im = mask_boxes(_open(img), person_boxes)
    s = max(im.size)
    if s > max_side:
        k = max_side / s
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    clean = Image.frombytes("RGB", im.size, im.tobytes())          # pixels only: drops info / exif / icc
    b = io.BytesIO()
    clean.save(b, "JPEG", quality=quality)
    return b.getvalue()
