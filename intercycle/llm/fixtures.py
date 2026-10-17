"""Tiny SYNTHETIC labelled frames for the Gemma verify_state harness, plus an offline stand-in model.

Six top-down sketches of a station: a base, a hazard zone (yellow), an arm (blue) either at home or
extended, and a person (red dot) inside the zone, outside it, or absent. Ground truth is known by
construction; in the real pipeline the same labels come for free from the simulator's rendered frames.

Single-frame convention: ``moving`` is True when the arm is away from its home pose (a frame cannot
show motion itself; multi-frame labels may list several images per entry, see ``gemma_eval_verify``).

``FixtureHeuristicTransport`` is NOT Gemma. It reads these sketches with pixel counting so that the
harness, cache and tests run offline. Anything it produces is labelled model "mock-pixel-heuristic".
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from .client import TransportError

SIZE = 192
ZONE = (84, 20, 184, 172)            # x0, y0, x1, y1 of the hazard zone
BASE = (40, 96)
ARM_RGB = (30, 60, 160)
PERSON_RGB = (220, 40, 40)
MOCK_MODEL = "mock-pixel-heuristic"

# (file, machine, arm extended, person position or None)  -> person_in_zone derived from position
_SPEC = [
    ("frame_00.png", "R1", False, None),
    ("frame_01.png", "R1", True, None),
    ("frame_02.png", "R1", False, (150, 150)),
    ("frame_03.png", "R1", True, (118, 148)),
    ("frame_04.png", "R2", False, (28, 160)),      # person visible but outside the zone
    ("frame_05.png", "R2", True, (46, 150)),       # extended arm, person outside the zone
]


def _in_zone(p: tuple[int, int] | None) -> bool:
    return p is not None and ZONE[0] <= p[0] <= ZONE[2] and ZONE[1] <= p[1] <= ZONE[3]


def render(extended: bool, person: tuple[int, int] | None, machine: str):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (SIZE, SIZE), (205, 205, 205))
    d = ImageDraw.Draw(im)
    d.rectangle(ZONE, fill=(255, 236, 150), outline=(230, 180, 0), width=3)       # hazard zone
    d.rectangle((90, 4, 130, 14), fill=(40, 160, 60))                              # static rack (distractor)
    d.ellipse((BASE[0] - 14, BASE[1] - 14, BASE[0] + 14, BASE[1] + 14), fill=(70, 70, 70))
    if extended:
        d.line([BASE, (95, 62), (165, 98)], fill=ARM_RGB, width=8)
        d.ellipse((159, 92, 171, 104), fill=ARM_RGB)
    else:
        d.line([BASE, (56, 86)], fill=ARM_RGB, width=8)
    if person:
        x, y = person
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=PERSON_RGB)
    d.text((6, 4), f"SYNTHETIC {machine}", fill=(0, 0, 0))
    return im


def make_fixture(out_dir: str | Path) -> dict:
    """Write ``out_dir/frames/*.png`` and ``out_dir/labels.json``; returns the labels dict."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    labels = {}
    for name, machine, ext, person in _SPEC:
        render(ext, person, machine).save(out / "frames" / name)
        labels[name] = {"machine": machine, "moving": ext, "person_in_zone": _in_zone(person)}
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


def _decode(url: str):
    from PIL import Image
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _mask(arr, rgb, tol=30):
    import numpy as np
    return (np.abs(arr.astype(int) - np.array(rgb)) <= tol).all(axis=2)


class FixtureHeuristicTransport:
    """Offline stand-in for the model: counts arm-coloured and person-coloured pixels. NOT Gemma."""

    def __call__(self, payload: dict) -> dict:
        sys_text = str(payload["messages"][0].get("content", ""))
        if "vision checker" not in sys_text:
            raise TransportError("mock transport only implements verify_state", status=400)
        urls = [p["image_url"]["url"] for m in payload["messages"] if isinstance(m.get("content"), list)
                for p in m["content"] if p.get("type") == "image_url"]
        if not urls:
            raise TransportError("mock transport needs at least one image", status=400)
        import numpy as np
        moving, person = False, False
        for u in urls:
            arr = np.asarray(_decode(u))
            h, w = arr.shape[:2]
            sx, sy = w / SIZE, h / SIZE
            moving |= _mask(arr, ARM_RGB).sum() / (sx * sy) > 500
            ys, xs = np.nonzero(_mask(arr, PERSON_RGB))
            if len(xs) / (sx * sy) > 40:
                cx, cy = xs.mean() / sx, ys.mean() / sy
                person |= bool(ZONE[0] <= cx <= ZONE[2] and ZONE[1] <= cy <= ZONE[3])
        reply = {"moving": bool(moving), "person_in_zone": person, "confidence": 0.9 if len(urls) > 1 else 0.7,
                 "notes": "mock pixel heuristic on synthetic sketch"}
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(reply)},
                             "finish_reason": "stop"}], "usage": {}}

