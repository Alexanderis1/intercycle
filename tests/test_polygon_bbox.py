"""polygon_bbox (twin/schemas/site.py) is the one truth-free floor-polygon box helper: the plant and the real-clip
pixel adapter share it, so the adapter never imports the truth module twin.plant (PLAN_TWIN §2 truth guards)."""
from __future__ import annotations

import ast
from pathlib import Path

from twin.schemas.site import polygon_bbox

ROOT = Path(__file__).resolve().parents[1]


def test_polygon_bbox_grows_every_side():
    poly = [(1.0, 2.0), (4.0, 2.5), (3.0, 6.0)]
    assert polygon_bbox(poly) == (1.0, 2.0, 4.0, 6.0)
    assert polygon_bbox(poly, 0.5) == (0.5, 1.5, 4.5, 6.5)


def test_pixel_adapter_uses_the_truth_free_helper():
    tree = ast.parse((ROOT / "intercycle" / "observer" / "pixel_adapter.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "twin.plant" not in mods and "twin.schemas.site" in mods
