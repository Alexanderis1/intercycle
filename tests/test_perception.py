"""Perception tests: geometry, zones, occupancy, evaluation and DETR post-processing.

CPU-only and torch-free: none of the exercised modules imports torch / av / rfdetr at module level
(asserted below).  The ByteTrack test is skipped unless ``trackers`` + ``supervision`` are installed.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from intercycle.perception import evaluate as ev
from intercycle.perception import geometry as geo
from intercycle.perception import gt as gtmod
from intercycle.perception import occupancy as occ
from intercycle.perception.detector import canonical_label, nms_xyxy, postprocess_detr_numpy

# Real camera: SmartSpaces MTMC_Tracking_2026/train/Warehouse_000, Camera_0008 (cameraMatrix).
CAM8_P = [
    [23.16235223126146, -18.750958405949806, -3.2832739348287623, 441.78440305588526],
    [2.3880370055870075, 0.940510140046732, -27.133889941840884, 247.5367066351158],
    [0.014164480877340182, 0.0055988454731034825, -0.0033638029145834614, 1.0],
]
CAM8_H = [[r[0], r[1], r[3]] for r in CAM8_P]
# A person from ground truth: floor (x, y) in metres and its visible 2D box in this camera.
GT_FLOOR = (-1.30857894, -33.39015552)
GT_BOX = (1307, 268)  # foot point (bottom-centre) of the annotated box


def _calib_dict(with_h: bool = True) -> dict:
    s = {
        "type": "camera",
        "id": "Camera_0008",
        "cameraMatrix": CAM8_P,
        "attributes": [
            {"name": "fps", "value": "30"},
            {"name": "frameWidth", "value": "1920"},
            {"name": "frameHeight", "value": "1080"},
        ],
    }
    if with_h:
        s["homography"] = CAM8_H
    return {"calibrationType": "cartesian", "sensors": [s]}


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize("with_h", [True, False])
def test_calibration_floor_projection_matches_ground_truth(with_h):
    cam = geo.load_calibration(_calib_dict(with_h))["Camera_0008"]
    assert (cam.width, cam.height, cam.fps) == (1920, 1080, 30.0)
    uv = cam.world_to_image(np.array([[GT_FLOOR[0], GT_FLOOR[1], 0.0]]))[0]
    assert np.hypot(uv[0] - GT_BOX[0], uv[1] - GT_BOX[1]) < 2.0  # annotated foot vs projected floor point
    xy = cam.image_to_floor(np.array([GT_BOX], dtype=float))[0]
    assert np.hypot(xy[0] - GT_FLOOR[0], xy[1] - GT_FLOOR[1]) < 0.5  # back-projection within 0.5 m


def test_floor_image_roundtrip():
    cam = geo.load_calibration(_calib_dict())["Camera_0008"]
    pts = np.array([[-10.0, -30.0], [-5.0, -20.0], [-2.0, -33.0]])
    np.testing.assert_allclose(cam.image_to_floor(cam.floor_to_image(pts)), pts, atol=1e-8)


def test_foot_points_are_bottom_centre():
    fp = geo.foot_points(np.array([[10, 20, 30, 60], [0, 0, 4, 8]]))
    np.testing.assert_allclose(fp, [[20, 60], [2, 8]])


def test_point_in_polygon_convex_and_concave():
    sq = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    pts = np.array([[1, 1], [5, 1], [-1, 2], [3.9, 3.9], [4.1, 2]])
    assert geo.points_in_polygon(pts, sq).tolist() == [True, False, False, True, False]
    ell = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], float)  # L shape
    q = np.array([[0.5, 3.0], [3.0, 0.5], [3.0, 3.0], [2.0, 2.0]])
    assert geo.points_in_polygon(q, ell).tolist() == [True, True, False, False]


def test_grid_zones_tile_the_bounds_and_assign():
    zones = geo.grid_zones((0, 0, 30, 20), 3, 2)
    assert [z.name for z in zones] == ["Z00", "Z01", "Z02", "Z03", "Z04", "Z05"]
    idx = geo.assign_zones(np.array([[5, 5], [15, 5], [25, 5], [5, 15], [25, 15], [-1, 5], [31, 5]]), zones)
    assert idx.tolist() == [0, 1, 2, 3, 5, -1, -1]


def test_zones_json_roundtrip(tmp_path):
    zones = geo.grid_zones((0, 0, 2, 2), 2, 1, prefix="A")
    geo.save_zones(zones, tmp_path / "z.json", {"k": 1})
    back = geo.load_zones(tmp_path / "z.json")
    assert [z.name for z in back] == ["A00", "A01"]
    np.testing.assert_allclose(back[1].polygon, zones[1].polygon)


# ------------------------------------------------------------------ occupancy


def test_fuse_merges_across_cameras_but_not_within_one_camera():
    pts = pd.DataFrame(
        {
            "frame": [0, 0, 0, 0, 0],
            "cam": ["a", "b", "a", "a", "b"],
            "cls": ["person"] * 5,
            "wx": [1.0, 1.2, 1.5, 9.0, 9.1],
            "wy": [1.0, 1.0, 1.0, 9.0, 9.0],
            "score": [0.9, 0.8, 0.7, 0.9, 0.6],
        }
    )
    f = occ.fuse_world_points(pts, radius_m=0.8)
    # persons: {a@1.0, b@1.2} merged; a@1.5 is a second person seen only by a (same camera never merges);
    # {a@9.0, b@9.1} merged.
    assert len(f) == 3
    assert sorted(f["n_cams"].tolist()) == [1, 2, 2]


def test_occupancy_steps_bins_with_empty_steps():
    zones = geo.grid_zones((0, 0, 20, 10), 2, 1)  # Z00: x<10, Z01: x>=10
    fps = 30.0
    frames = np.arange(0, 120, 6)  # 20 steps, 4 s of video at 5 Hz
    rows = []
    for f in frames[:10]:  # first half: 2 persons in Z00, 1 in Z01
        rows += [(f, "person", 2.0, 5.0), (f, "person", 3.0, 5.0), (f, "person", 15.0, 5.0)]
    pts = pd.DataFrame(rows, columns=["frame", "cls", "wx", "wy"])
    steps = occ.occupancy_steps(pts, zones, frames, ["person"])
    assert len(steps) == 20 * 2  # dense, incl. empty frames
    z0 = steps[steps["zone"] == "Z00"].sort_values("frame")["count"].tolist()
    assert z0 == [2] * 10 + [0] * 10
    bins = occ.occupancy_bins(steps, fps, bin_seconds=2.0)  # 2 s bins -> frames 0..59 and 60..119
    b = bins[(bins["zone"] == "Z00")].sort_values("bin")
    assert b["mean_count"].tolist() == [2.0, 0.0]
    assert b["max_count"].tolist() == [2, 0]
    assert bins[(bins["zone"] == "Z01")].sort_values("bin")["mean_count"].tolist() == [1.0, 0.0]


def test_auto_bin_seconds():
    assert occ.auto_bin_seconds(60) == 60.0
    assert occ.auto_bin_seconds(300) == 60.0
    assert occ.auto_bin_seconds(3600) == 300.0


# ------------------------------------------------------------------ ground-truth tables


def test_gt_tables_accept_spaced_and_underscored_keys():
    f0 = [
        {"object type": "Person", "object id": 7, "3d location": [1.0, 2.0, 0.9],
         "2d bounding box visible": {"Camera_0001": [10, 20, 30, 80], "Camera_0002": [0, 0, 5, 5]}},
        {"object_type": "PalletTruck", "object_id": 8, "3d_location": [3.0, 4.0, 0.6],
         "2d_bounding_box_visible": {"Camera_0002": [1, 1, 9, 9]}},
        {"object type": "NovaCarter", "object id": 9, "3d location": [0, 0, 0], "2d bounding box visible": {}},
    ]
    boxes, objs = gtmod.build_gt_tables([(0, f0)], ["Camera_0001"])
    assert boxes[["camera", "frame", "object_id", "cls"]].values.tolist() == [["Camera_0001", 0, 7, "person"]]
    assert objs.iloc[0]["max_h"] == 60.0 and objs.iloc[0]["cams"] == "Camera_0001"
    _, objs2 = gtmod.build_gt_tables([(0, f0)], ["Camera_0001", "Camera_0002"])
    assert set(objs2["cls"]) == {"person", "pallet_truck"}


# ------------------------------------------------------------------ evaluation


def _gt_boxes(n_frames=10, tall=60.0):
    rows = []
    for fr in range(n_frames):
        rows.append(("c", fr, 1, "person", 10 + 5 * fr, 100, 30 + 5 * fr, 100 + tall))
        rows.append(("c", fr, 2, "person", 300, 50 + 4 * fr, 330, 50 + 4 * fr + tall))
    return pd.DataFrame(rows, columns=["camera", "frame", "object_id", "cls", "x1", "y1", "x2", "y2"])


def _as_tracks(gt: pd.DataFrame, id_fn=lambda oid, fr: oid) -> pd.DataFrame:
    t = gt.rename(columns={"object_id": "track_id"}).copy()
    t["track_id"] = [id_fn(o, f) for o, f in zip(gt["object_id"], gt["frame"])]
    t["score"] = 0.9
    return t


def test_perfect_prediction_scores_one():
    gt = _gt_boxes()
    frames = np.arange(10)
    dets = gt.drop(columns=["object_id"]).assign(score=0.9)
    d = ev.detection_metrics(dets, gt, frames, ["c"], min_h=20)
    assert (d["tp"], d["fp"], d["fn"]) == (20, 0, 0) and d["precision"] == 1.0 and d["recall"] == 1.0
    t = ev.tracking_metrics(_as_tracks(gt), gt, frames, ["c"], min_h=20, with_motmetrics=False)
    assert t["mota"] == 1.0 and t["idf1"] == 1.0 and t["id_switches"] == 0


def test_id_switch_fp_and_fn_are_counted():
    gt = _gt_boxes()
    frames = np.arange(10)
    # object 1 changes track id at frame 5; one extra false positive at frame 0; object 2 missing at frame 9
    tr = _as_tracks(gt, lambda o, f: (100 if (o == 1 and f >= 5) else o))
    tr = tr[~((tr["track_id"] == 2) & (tr["frame"] == 9))]
    fp = pd.DataFrame(
        [("c", 0, 99, "person", 600, 600, 640, 680, 0.8)],
        columns=["camera", "frame", "track_id", "cls", "x1", "y1", "x2", "y2", "score"],
    )
    tr = pd.concat([tr, fp], ignore_index=True)
    t = ev.tracking_metrics(tr, gt, frames, ["c"], min_h=20, with_motmetrics=False)
    assert t["id_switches"] == 1 and t["fn"] == 1 and t["fp"] == 1
    assert t["mota"] == pytest.approx(1 - 3 / 20)
    # IDF1: GT ids {1: 10 frames, 2: 10 frames}; predicted ids {1: 5, 100: 5, 2: 9, 99: 1} (20 boxes).
    # Best one-to-one identity assignment keeps 5 (obj1) + 9 (obj2) = 14 -> IDFN = IDFP = 6.
    assert t["idf1"] == pytest.approx(2 * 14 / (2 * 14 + 6 + 6))


def test_small_gt_is_ignored_not_penalised():
    gt = _gt_boxes(n_frames=4, tall=10.0)  # all persons are 10 px tall -> ignore regions at min_h=20
    frames = np.arange(4)
    dets = gt.drop(columns=["object_id"]).assign(score=0.9)
    d = ev.detection_metrics(dets, gt, frames, ["c"], min_h=20)
    assert (d["tp"], d["fp"], d["fn"], d["ignored_dets"]) == (0, 0, 0, 8)
    d0 = ev.detection_metrics(dets, gt, frames, ["c"], min_h=0)
    assert (d0["tp"], d0["fp"], d0["fn"]) == (8, 0, 0)
    d_empty = ev.detection_metrics(dets.iloc[0:0], gt, frames, ["c"], min_h=0)
    assert d_empty["recall"] == 0.0 and d_empty["fn"] == 8


def test_occupancy_metrics_perfect_and_biased():
    zones = geo.grid_zones((0, 0, 20, 10), 2, 1)
    frames = np.arange(0, 60, 6)
    gt_rows = []
    for f in frames:
        gt_rows += [(f, 1, "person", 2.0, 5.0, 0.9, "c", 60.0), (f, 2, "person", 15.0, 5.0, 0.9, "c", 60.0)]
    gt = pd.DataFrame(gt_rows, columns=["frame", "object_id", "cls", "wx", "wy", "wz", "cams", "max_h"])
    pts = gt[["frame", "cls", "wx", "wy"]]
    perfect = occ.occupancy_steps(pts, zones, frames, ["person"])
    m = ev.occupancy_metrics(perfect, gt, zones, frames, 30.0, 1.0)
    assert m["step_mae"] == 0 and m["bin_mae"] == 0 and m["total_count_mae"] == 0
    missed = occ.occupancy_steps(pts[pts["wx"] < 10], zones, frames, ["person"])  # lost the Z01 person
    m2 = ev.occupancy_metrics(missed, gt, zones, frames, 30.0, 1.0)
    assert m2["step_mae"] == pytest.approx(0.5) and m2["total_count_mae"] == pytest.approx(1.0)
    z = {r["zone"]: r for r in m2["per_zone"]}
    assert z["Z00"]["mae"] == 0 and z["Z01"]["mae"] == 1


def test_localisation_error_zero_for_exact_foot_points():
    cam = geo.load_calibration(_calib_dict())["Camera_0008"]
    foot = cam.floor_to_image(np.array([GT_FLOOR]))[0]
    box = (foot[0] - 10, foot[1] - 50, foot[0] + 10, foot[1])
    matches = [("Camera_0008", 0, *box, 5)]
    objs = pd.DataFrame([(0, 5, "person", GT_FLOOR[0], GT_FLOOR[1], 0.9, "Camera_0008", 50.0)],
                        columns=["frame", "object_id", "cls", "wx", "wy", "wz", "cams", "max_h"])
    r = ev.localisation_error(matches, objs, {"Camera_0008": cam})
    assert r["n"] == 1 and r["median_m"] < 1e-6


# ------------------------------------------------------------------ detector post-processing (no torch)


def test_nms_suppresses_overlaps_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms_xyxy(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_postprocess_detr_numpy_topk_scaling_and_nms():
    q, c = 4, 5
    logits = np.full((q, c), -10.0, np.float32)
    boxes = np.zeros((q, 4), np.float32)
    logits[0, 1] = 5.0  # person, high
    boxes[0] = [0.5, 0.5, 0.2, 0.4]  # cx cy w h (normalised)
    logits[1, 1] = 3.0  # duplicate of query 0 -> removed by NMS
    boxes[1] = [0.51, 0.5, 0.2, 0.4]
    logits[2, 3] = 2.0  # another class elsewhere
    boxes[2] = [0.1, 0.1, 0.1, 0.1]
    logits[3, 1] = -3.0  # below threshold
    xyxy, score, cls = postprocess_detr_numpy(boxes, logits, (1080, 1920), threshold=0.5, nms_iou=0.7)
    assert len(score) == 2 and cls.tolist() == [1, 3]
    np.testing.assert_allclose(xyxy[0], [0.4 * 1920, 0.3 * 1080, 0.6 * 1920, 0.7 * 1080], rtol=1e-5)
    _, score_nonms, _ = postprocess_detr_numpy(boxes, logits, (1080, 1920), threshold=0.5, nms_iou=None)
    assert len(score_nonms) == 3


def test_canonical_label_mapping():
    assert canonical_label("person") == "person"
    assert canonical_label("Pallet Truck") == "pallet_truck"
    assert canonical_label("PalletTruck") == "pallet_truck"
    assert canonical_label("Forklift") == "forklift"
    assert canonical_label("truck") is None  # COCO truck is NOT silently treated as a forklift
    assert canonical_label("truck", {"truck": "forklift"}) == "forklift"


# ------------------------------------------------------------------ environment guards


def test_geometry_logic_imports_no_heavy_deps():
    code = (
        "import sys; import intercycle.perception.geometry, intercycle.perception.occupancy, "
        "intercycle.perception.evaluate, intercycle.perception.gt, intercycle.perception.detector, "
        "intercycle.perception.video, intercycle.perception.pipeline; "
        "bad=[m for m in ('torch','av','rfdetr','supervision','trackers','onnxruntime') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bytetrack_keeps_identity_on_a_walking_box():
    pytest.importorskip("trackers")
    pytest.importorskip("supervision")
    from intercycle.perception.detector import Dets
    from intercycle.perception.tracking import CameraTracker

    trk = CameraTracker(fps=5.0, classes=("person",))
    ids = []
    for k in range(12):
        x = 100 + 6 * k
        d = Dets(np.array([[x, 200, x + 40, 300]], np.float32), np.array([0.9], np.float32), np.array(["person"]))
        out = trk.update(d, t=k / 5.0)
        ids += out.track_id.tolist()
    assert len(ids) >= 10 and len(set(ids)) == 1


# ------------------------------------------------------------------ stage 2 + evaluation end to end (no model)


def test_aggregate_and_evaluate_run_on_synthetic_tracks(tmp_path):
    from intercycle.perception.pipeline import PipelineConfig, aggregate_run

    frames = [0, 6, 12, 18]
    zones = geo.grid_zones((0, 0, 20, 10), 2, 1)
    geo.save_zones(zones, tmp_path / "zones.json")
    run = tmp_path / "run"
    run.mkdir()
    # two persons walking inside Z00 / Z01; each seen by two cameras with identical floor positions
    trk, gt_boxes, gt_objs, dets = [], [], [], []
    for f in frames:
        for pid, (wx, wy) in {1: (2.0 + 0.1 * f, 5.0), 2: (15.0, 5.0 + 0.05 * f)}.items():
            for cam in ("c1", "c2"):
                box = (100.0 * pid + f, 200.0, 100.0 * pid + f + 40.0, 300.0)
                trk.append((cam, f, f / 30.0, pid, "person", 0.9, *box, box[0] + 20, box[3], wx, wy))
                dets.append((cam, f, "person", 0.9, *box))
                gt_boxes.append((cam, f, pid, "person", *box))
            gt_objs.append((f, pid, "person", wx, wy, 0.9, "c1|c2", 100.0))
    cols = ["camera", "frame", "t", "track_id", "cls", "score", "x1", "y1", "x2", "y2", "foot_u", "foot_v", "wx", "wy"]
    pd.DataFrame(trk, columns=cols).to_parquet(run / "tracks_all.parquet")
    pd.DataFrame(dets, columns=["camera", "frame", "cls", "score", "x1", "y1", "x2", "y2"]).to_parquet(run / "detections_all.parquet")
    (run / "meta_all.json").write_text(json.dumps(
        {"sampled_fps": 5.0, "src_fps": 30.0, "stride": 6, "frame_first": 0, "frame_last": 18, "n_frames": 8,
         "infer_s": 1.0, "wall_s": 1.0, "cameras": ["c1", "c2"]}))
    gdir = tmp_path / "gt"
    gdir.mkdir()
    pd.DataFrame(gt_boxes, columns=["camera", "frame", "object_id", "cls", "x1", "y1", "x2", "y2"]).to_parquet(gdir / "gt_boxes.parquet")
    pd.DataFrame(gt_objs, columns=["frame", "object_id", "cls", "wx", "wy", "wz", "cams", "max_h"]).to_parquet(gdir / "gt_objects.parquet")

    cfg = PipelineConfig(scene_dir=str(tmp_path), out_dir=str(run), zones=str(tmp_path / "zones.json"),
                         gt_dir=str(gdir), bin_seconds=1.0)
    summary = aggregate_run(cfg, log=lambda *a: None)
    assert summary["n_steps"] == 4 and summary["fused_points"] == 8  # 2 cameras merged -> 2 persons x 4 steps
    occ_steps = pd.read_parquet(run / "occupancy_steps.parquet")
    assert occ_steps.groupby("frame")["count"].sum().tolist() == [2, 2, 2, 2]
    for name in ("tracks.parquet", "occupancy.parquet", "world_points.parquet", "zones.json", "meta.json"):
        assert (run / name).exists()

    res = ev.evaluate_run(run, gdir, None, min_h=20.0, log=lambda *a: None)
    m = res["min_h_20"]
    assert m["detection"]["precision"] == 1.0 and m["detection"]["recall"] == 1.0
    assert m["tracking"]["mota"] == 1.0 and m["tracking"]["idf1"] == 1.0
    assert m["occupancy"]["step_mae"] == 0.0 and m["occupancy"]["bin_mae"] == 0.0
