"""Lane S4 item 5: real-clip scoring (intercycle/eval/realclip.py) and the pixel adapter (observer/pixel_adapter.py)."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from intercycle.eval import realclip as RC
from intercycle.observer import mapping as OM
from intercycle.observer import pixel_adapter as PA
from intercycle.observer import watchdog as WD

ROOT = Path(__file__).resolve().parents[1]


def test_match_boxes_greedy_one_to_one():
    gt = [(0, 0, 10, 10), (20, 0, 30, 10)]
    pred = [(1, 0, 11, 10), (0, 0, 10, 10), (50, 50, 60, 60)]
    m = RC.match_boxes(gt, pred)
    assert [(i, j) for i, j, _ in m] == [(0, 1)]            # pred 0 (IoU 0.82) loses to the exact pred 1; gt 1 unmatched
    assert RC.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert RC.match_boxes(gt, [(0, 0, 10, 10), (math.nan,) * 4]) == [(0, 0, 1.0)]   # NaN box never matches


def test_match_times_tolerance_and_one_to_one():
    m = RC.match_times([1.0, 1.22, 3.0], [1.1, 3.26, 3.24])
    assert sorted(m) == [(0, 0), (2, 2)]                   # 1.22 has no pred left within 0.25 s; 3.26 is outside
    assert RC.match_times([1.0], [1.25]) == [(0, 0)]


def test_clopper_pearson():
    lo, hi = RC.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187, abs=1e-3) and hi == pytest.approx(0.813, abs=1e-3)
    assert RC.clopper_pearson(0, 4)[0] == 0.0 and RC.clopper_pearson(4, 4)[1] == 1.0
    assert all(math.isnan(x) for x in RC.clopper_pearson(0, 0))


def _frec(group, tp, n_gt, camera=9):
    return RC.FrameRec("s", "c", camera, group, True, "c9_f1", True, n_gt, tp, tp, [1.0] * tp, [True] * tp)


def test_small_group_rule_labels_clopper_pearson():
    fr = [_frec(f"g{i}", 1, 2) for i in range(4)]
    b = RC.person_block(fr, 4)
    assert b["recall_ci"]["label"] == "ignores clustering (only 4 groups)"
    assert b["recall_ci"]["ci95"] == RC.clopper_pearson(4, 8)
    assert set(b["per_group_recall"]) == {"g0", "g1", "g2", "g3"} and b["lowest_group_recall"] == 0.5
    b5 = RC.person_block(fr + [_frec("g4", 2, 2)], 5)
    assert b5["recall_ci"]["method"].startswith("bootstrap by group")


def test_no_H_frames_excluded_with_n():
    a = _frec("g0", 2, 2)
    b = _frec("g1", 1, 1)
    b.has_H = False
    blk = RC.person_block([a, b], 2)
    assert blk["foot_point_px"]["n_pairs"] == 2 and blk["foot_point_px"]["n_frames_excluded_no_H"] == 1
    assert blk["zone_accuracy"]["n_pairs"] == 2 and blk["tp"] == 3          # P/R still count the no-H frame


def test_tracked_boxes_recovers_box_and_fails_closed():
    boxes = pd.DataFrame({"t": [1.0, 1.0], "class": ["person", "person"], "conf": [0.9, 0.2],
                          "x1": [0.0, 50.0], "y1": [0.0, 0.0], "x2": [10.0, 60.0], "y2": [20.0, 20.0],
                          "suppressed": [False, False]})
    tracks = pd.DataFrame({"t": [1.0, 1.0], "class": ["person", "person"], "u": [5.0, 200.0], "v": [20.0, 20.0]})
    P, found = RC.tracked_boxes(tracks, boxes, 1.02)
    assert found and P[0] == (0.0, 0.0, 10.0, 20.0) and all(math.isnan(x) for x in P[1])
    P2, found2 = RC.tracked_boxes(tracks, boxes, 2.0)
    assert P2 == [] and not found2
    D, _ = RC.detector_boxes(boxes, 1.0)
    assert D == [(0.0, 0.0, 10.0, 20.0)]                                   # conf 0.2 < 0.4 dropped


def test_score_frame_foot_and_zone():
    lab = SimpleNamespace(sample_id="s", clip_id="c", t_pts=1.0, boxes=[
        SimpleNamespace(cls="person", x0=0.0, y0=0.0, x1=10.0, y1=20.0, foot_u=5.0, foot_v=17.0),
        SimpleNamespace(cls="forklift", x0=0.0, y0=0.0, x1=100.0, y1=100.0, foot_u=None, foot_v=None)])
    meta = {"camera": 9, "group": "g", "unsafe": True, "framing": "f"}
    r = RC.score_frame(lab, [(0.0, 0.0, 10.0, 20.0)], True, meta, lambda uv: np.asarray(uv, float),
                       lambda xz: (np.asarray(xz)[:, 1] > 18).astype(int))
    assert (r.n_gt, r.n_pred, r.tp) == (1, 1, 1)
    assert r.foot_err_px == [pytest.approx(3.0)] and r.zone_same == [False]    # label foot v 17, pred foot v 20


def test_score_window_ignores_spans_and_counts_missing():
    m = pd.DataFrame({"t": np.arange(0, 5.0, 0.5), "stroke": [False, False, True, False, False, False, True, False,
                                                                False, True], "missing": [False] * 6 + [True] * 4})
    win = SimpleNamespace(sample_id="w", clip_id="c", press="P9a", t0=0.0, t1=5.0, strokes=[1.1, 3.0])
    meta = {"camera": 9, "group": "g", "unsafe": False}
    ticks = [(3.0, True), (3.5, True), (4.0, False), (1.0, False)]
    r = RC.score_window(win, m, ignore=[[4.4, 4.6]], hidden=[[3.0, 4.0]], meta=meta, tick_missing=ticks)
    assert (r.n_gt, r.n_pred, r.tp) == (2, 2, 2)                           # 4.5 is inside an ignore span
    assert (r.hidden_ticks, r.hidden_missing_ticks) == (3, 2)
    assert (r.hidden_perc_samples, r.hidden_perc_missing) == (3, 3)
    b = RC.stroke_block([r])
    assert b["f1"] == 1.0 and b["count_error"]["mean_signed"] == 0.0
    assert b["occlusion_to_missing"]["observation_ticks"]["rate"] == pytest.approx(2 / 3)
    assert (r.nonhidden_ticks, r.nonhidden_missing_ticks) == (1, 0)               # base: tick 1.0 outside [3, 4]
    assert b["occlusion_to_missing"]["base_nonhidden_observation_ticks"]["rate"] == 0.0
    r2 = RC.score_window(win, m, ignore=[], hidden=[], meta=meta, tick_missing=ticks)
    assert (r2.hidden_ticks, r2.nonhidden_ticks, r2.nonhidden_missing_ticks) == (0, 4, 2)   # no hidden span: all base


def test_pooled_small_group_label_names_the_camera():
    fr = [_frec(f"g{i}", 1, 2, camera=9) for i in range(4)] + [_frec(f"h{i}", 1, 2, camera=14) for i in range(6)]
    s = RC.summarise(fr, [])
    ci = s["person"]["pooled"]["recall_ci"]
    assert ci["label"] == "ignores clustering (camera 9 has only 4 groups)" and s["person"]["pooled"]["n_groups"] == 10
    assert s["person"]["per_camera"]["c9"]["recall_ci"]["label"] == "ignores clustering (only 4 groups)"
    assert s["person"]["per_camera"]["c14"]["recall_ci"]["method"].startswith("bootstrap by group")


def test_train_references_fail_closed(tmp_path):
    inv = pd.DataFrame({"clip_id": ["a", "b"], "framing_id": ["f", "g"], "split": ["train", "dev"]})
    with pytest.raises(FileNotFoundError, match="no loadable keyframe reference"):
        RC.train_references(inv, tmp_path, ["f"])
    assert RC.train_references(inv, tmp_path, ["g"]) == {"g": None}       # no train clip: no H, no Observation
    import cv2

    (tmp_path / "keyframes").mkdir()
    cv2.imwrite(str(tmp_path / "keyframes" / "a_k1.jpg"), _frame(3))
    assert RC.train_references(inv, tmp_path, ["f"])["f"] is not None


def test_clip_meta_refuses_other_splits():
    inv = pd.DataFrame({"clip_id": ["a", "b"], "camera": [9, 14], "group_id": ["g", "h"], "framing_id": ["f", "f"],
                        "class": ["0", "5"], "split": ["dev", "test"]})
    assert RC.clip_meta(inv, "a", ("dev",))["unsafe"] is True
    with pytest.raises(ValueError, match="fail closed"):
        RC.clip_meta(inv, "b", ("dev",))


# ------------------------------------------------------------------------------------------------ pixel adapter

def _geom(H: np.ndarray | None = np.eye(3)):
    rects = [(10.0, 10.0, 12.0, 12.0)]
    return PA.FramingGeom("f", "z", None if H is None else np.asarray(H, float), 0.0, ["P1"], rects,
                          np.ones((1, 1)))


def test_machine_of_and_project_without_H():
    assert PA.machine_of(np.array([[11.0, 11.0], [0.0, 0.0], [np.nan, 1.0]]), _geom().rects).tolist() == [0, -1, -1]
    assert np.isnan(PA.project_uv(np.array([[1.0, 2.0]]), _geom(None))).all()
    with pytest.raises(ValueError, match="no fitted H"):
        PA.PixelAdapter(_geom(None), None)


def _frame(seed, lum=120.0):
    g = np.random.default_rng(seed)
    return np.clip(lum + 40 * g.standard_normal((90, 160, 3)), 0, 255).astype(np.uint8)


def test_adapter_watchdog_feeds_cam_floor_ok_and_missing():
    ref = WD.reference_from_frames([_frame(i) for i in range(3)])
    ad = PA.PixelAdapter(_geom(), ref)
    ad.reset()
    det = pd.DataFrame({"class": ["person"], "u": [30.0], "v": [30.0]})
    seq = []
    for k in range(10):
        t = k * 0.5
        fr = _frame(100 + k) if k < 8 else np.zeros((90, 160, 3), np.uint8)       # black from t = 4.0
        seq.append(ad.step(t, det, fr, {"P1": 0.7}))
    for o in seq:
        OM.check_obs_v2(o, 1, 1)
    ok = [bool(np.asarray(o.cam_floor_ok)[0]) for o in seq]
    assert ok[:6] == [False] * 6 and ok[6:8] == [True, True] and ok[8:] == [False, False]   # recover_s 3 s, then BLACK
    zm = [np.asarray(o.zone_motion) for o in seq]
    assert zm[7][0] == pytest.approx(0.7) and np.isnan(zm[8][0])
    assert seq[7].x[0] == pytest.approx(30.0) and seq[7].machine[0] == -1
    o = ad.step(5.0, det, None, {"P1": 0.7})
    assert not o.cam_ok[0] and not np.asarray(o.cam_floor_ok)[0] and np.isnan(np.asarray(o.zone_motion)[0])


def test_adapter_occlusion_is_the_image_space_flag():
    """Real clips: MISSING comes from PERC's image-space ram-ROI flag only; a person whose FOOT lies in the machine
    zone does not make the press MISSING (the twin's floor rule is not applied to real clips)."""
    ref = WD.reference_from_frames([_frame(i) for i in range(3)])
    ad = PA.PixelAdapter(_geom(), ref, WD.WatchdogParams(recover_s=0.0))
    det = pd.DataFrame({"class": ["person"], "u": [11.0], "v": [11.0]})
    o = ad.step(0.0, det, _frame(7), {"P1": 0.5})
    assert bool(np.asarray(o.cam_floor_ok)[0]) and o.machine[0] == 0
    assert np.asarray(o.zone_motion)[0] == pytest.approx(0.5)                     # no floor rule
    flagged = PA.motion_at({"P1": pd.DataFrame({"t": [0.5], "roi_energy": [0.5], "missing": [True]})}, 0.5)
    o = ad.step(0.5, det, _frame(8), flagged)
    assert np.isnan(np.asarray(o.zone_motion)[0])                                 # PERC's ROI flag -> MISSING
    assert np.isnan(PA.motion_at({"P1": pd.DataFrame({"t": [0.0], "roi_energy": [1.0], "missing": [True]})}, 0.0)["P1"])
    assert np.isnan(PA.motion_at({"P1": pd.DataFrame({"t": [2.0], "roi_energy": [1.0], "missing": [False]})}, 0.0)["P1"])


def test_site_geometry_from_site_yaml():
    from twin.schemas import load
    from twin.schemas.site import Site

    g = PA.site_geometry(load(Site, ROOT / "twin" / "site.yaml"))
    assert set(g) == {"c9_f0", "c9_f1", "c14_f0"}
    assert g["c9_f1"].press_ids == ["P9a", "P9b"] and g["c9_f1"].share.shape == (2, 1)
    assert all(v.H is not None for v in g.values())
    # zone_vis from the ram ROIs: c9_f0 measures only P9a; c9_f1 both; c14_f0 all four (every ROI fully in frame)
    assert g["c9_f0"].share[:, 0].tolist() == [1.0, 0.0]
    assert g["c9_f1"].share[:, 0].tolist() == [1.0, 1.0]
    assert g["c14_f0"].press_ids == ["P14a", "P14b", "P14c", "P14d"] and g["c14_f0"].share[:, 0].tolist() == [1.0] * 4
    assert PA.roi_in_image([[0, 0], [1920, 1080]], (1920, 1080))
    assert not PA.roi_in_image([[0, 0], [1921, 10]], (1920, 1080)) and not PA.roi_in_image([], (1920, 1080))


def _site_adapter(framing: str):
    from twin.schemas import load
    from twin.schemas.site import Site

    g = PA.site_geometry(load(Site, ROOT / "twin" / "site.yaml"))[framing]
    ref = WD.reference_from_frames([_frame(i) for i in range(3)])
    return g, PA.PixelAdapter(g, ref, WD.WatchdogParams(recover_s=0.0))


NO_DETS = pd.DataFrame({"class": pd.Series([], dtype=str), "u": pd.Series([], dtype=float),
                        "v": pd.Series([], dtype=float)})


@pytest.mark.parametrize("framing", ["c9_f0", "c9_f1", "c14_f0"])
def test_site_adapter_unoccluded_tick_is_finite(framing):
    """Real site geometry: an unoccluded, watchdog-ok tick with a finite PERC ROI energy gives a finite zone_motion
    and zone_vis 1 for every press the framing has a ram ROI for (the always-MISSING bug of the first build)."""
    g, ad = _site_adapter(framing)
    motion = {p: 0.4 + 0.1 * i for i, p in enumerate(g.press_ids)}
    o = ad.step(0.0, NO_DETS, _frame(11), motion)
    OM.check_obs_v2(o, len(g.press_ids), 1)
    assert bool(np.asarray(o.cam_floor_ok)[0])
    zm, zv = np.asarray(o.zone_motion), np.asarray(o.zone_vis)
    seen = g.share[:, 0] == 1.0
    assert seen.any()
    assert np.isfinite(zm[seen]).all() and np.allclose(zm[seen], [motion[p] for p, s in zip(g.press_ids, seen) if s])
    assert (zv[seen] == 1.0).all() and np.isnan(zm[~seen]).all() and (zv[~seen] == 0.0).all()


def test_site_adapter_roi_occluder_is_missing():
    """Real site geometry, image-space rule end to end: a box over P9a's ram ROI on c9_f1 makes PERC's
    roi_missing_series flag True, which makes P9a MISSING in the Observation; a person whose foot lies in P9a's
    machine zone but whose box is not on the ROI leaves it finite (no floor rule on real clips)."""
    from intercycle.observer.occlusion import roi_missing_series
    from twin.schemas import load
    from twin.schemas.site import Site

    site = load(Site, ROOT / "twin" / "site.yaml")
    rois = {p.id: [r.polygon_px for r in p.ram_roi.value if r.framing == "c9_f1"][0]
            for p in site.presses if p.id in ("P9a", "P9b")}
    g, ad = _site_adapter("c9_f1")
    poly = np.asarray(rois["P9a"], float)
    box = [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())]
    t = np.array([0.0, 0.5])
    flags = {p: roi_missing_series(t, [[box], []], rois[p], 0.0) for p in rois}
    assert flags["P9a"].tolist() == [True, False]
    mach = {p: pd.DataFrame({"t": t, "roi_energy": [0.5, 0.5], "missing": flags[p]}) for p in rois}
    o = ad.step(0.0, NO_DETS, _frame(12), PA.motion_at(mach, 0.0))
    zm = np.asarray(o.zone_motion)
    assert np.isnan(zm[0]) and np.isfinite(zm[1]) == (not bool(flags["P9b"][0]))
    uu, vv = np.meshgrid(np.arange(0, 1920, 16.0), np.arange(0, 1080, 16.0))
    uv = np.c_[uu.ravel(), vv.ravel()]
    m = PA.machine_of(PA.project_uv(uv, g), g.rects)
    j = int(np.flatnonzero(m == 0)[0])                        # a pixel whose foot point lands in P9a's machine zone
    det = pd.DataFrame({"class": ["person"], "u": [uv[j, 0]], "v": [uv[j, 1]]})
    o = ad.step(0.5, det, _frame(13), PA.motion_at(mach, 0.5))
    assert bool(np.asarray(o.cam_floor_ok)[0]) and o.machine[0] == 0
    assert np.asarray(o.zone_motion)[0] == pytest.approx(0.5)                     # no floor rule on real clips
