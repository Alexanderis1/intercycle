"""scripts/cal_click_prep.py: undistorted copy, k1 borrowing, and the committed click plan (train frames only)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cal_click_prep", ROOT / "scripts" / "cal_click_prep.py")
assert spec is not None and spec.loader is not None
ccp = importlib.util.module_from_spec(spec)
sys.modules["cal_click_prep"] = ccp
spec.loader.exec_module(ccp)

from intercycle import esk, framing  # noqa: E402

PLAN = json.loads((ROOT / "twin" / "cal" / "cal_click_plan.json").read_text(encoding="utf-8"))
K1 = json.loads((ROOT / "twin" / "cal" / "k1_v0.json").read_text(encoding="utf-8"))


def test_undistort_image_moves_a_dot_to_its_undistorted_position():
    k1 = K1["c9_f0"]["k1"]
    u = np.array([[1700.0, 900.0]])
    d = framing.distort(u, k1)[0]  # where the dot sits in the raw frame
    raw = np.zeros((1080, 1920, 3), np.uint8)
    raw[int(round(d[1])) - 2:int(round(d[1])) + 3, int(round(d[0])) - 2:int(round(d[0])) + 3] = 255
    out = ccp.undistort_image(raw, k1)
    ys, xs = np.nonzero(out[..., 0] > 100)
    assert len(xs) > 0
    assert abs(xs.mean() - u[0, 0]) < 1.5 and abs(ys.mean() - u[0, 1]) < 1.5


def test_k1_is_borrowed_only_where_a_framing_has_none():
    k, src = ccp.k1_for("c9_f0", K1)
    assert src == "c9_f0" and k == K1["c9_f0"]["k1"]
    k, src = ccp.k1_for("c9_f1", K1)
    assert src == "c9_f0" and k == K1["c9_f0"]["k1"]
    k, src = ccp.k1_for("c14_f0", K1)
    assert src == "c14_f0"


def test_plan_covers_every_framing_with_train_clips_and_uses_train_frames_only():
    with open(esk.active_split_path(), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    train_framings = {r["framing_id"] for r in rows if r["split"] == "train"}
    assert set(PLAN["framings"]) == train_framings
    train_ids = {r["clip_id"] for r in rows if r["split"] == "train"}
    for fid, fr in PLAN["framings"].items():
        clip, k = fr["keyframe"].rsplit("_k", 1)
        assert clip in train_ids and k in ("1", "2")
        assert {r["clip_id"]: r["framing_id"] for r in rows}[clip] == fid


def test_plan_points_are_four_inside_the_frame_with_a_relation_or_metres():
    for fid, fr in PLAN["framings"].items():
        assert [p["id"] for p in fr["points"]] == ["p1", "p2", "p3", "p4"], fid
        for p in fr["points"] + fr.get("extra", []):
            u, v = p["uv"]
            assert 0 <= u < 1920 and 0 <= v < 1080, (fid, p["id"])
            assert p["relation"].strip(), (fid, p["id"])
        # no metre value other than the origin: no length is justified by the pictures
        assert all(p["x_m"] in (None, 0.0) and p["y_m"] in (None, 0.0) for p in fr["points"]), fid
