"""Tests for the CCTV post-process. Run from the repo root:

    python -m pytest render/cctv/test_cctv.py -q
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so ``render.cctv`` imports anywhere

from render.cctv import (  # noqa: E402
    GtFrame,
    boxes_from_instances,
    build_geometry,
    camera_params,
    frame_params,
    process_frame,
    process_sequence,
    timestamp_text,
)
from render.cctv.cctv_post import decode_video, main as cli_main  # noqa: E402

BG = 60
SQ = (400, 200, 480, 280)  # x1, y1, x2, y2 (exclusive) in a 960x540 frame


def synth_frame(w=960, h=540, sq=SQ):
    """Dark grey frame with a white square, its instance mask, and a depth map."""
    img = np.full((h, w, 3), BG, np.uint8)
    inst = np.zeros((h, w), np.uint16)
    depth = np.full((h, w), 20.0, np.float32)
    x1, y1, x2, y2 = sq
    img[y1:y2, x1:x2] = 255
    inst[y1:y2, x1:x2] = 7
    depth[y1:y2, x1:x2] = 5.0
    ids = {"7": {"cls": "robot", "entity_id": "robot_07"}}
    return img, GtFrame(inst=inst, depth=depth, ids=ids)


def bright_bbox(img, thr=(BG + 255) / 2):
    """Bounding box (x1, y1, x2 exclusive, y2 exclusive) of the pixels above half level, ignoring the overlay bands."""
    h, w = img.shape[:2]
    y = img.astype(np.float32).mean(axis=2)
    y[: int(h * 0.12)] = 0
    y[int(h * 0.9):] = 0
    ys, xs = np.nonzero(y > thr)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_deterministic_for_a_seed():
    img, gt = synth_frame()
    base = camera_params("CAM03", 5, cam_label="BODY SHOP S30")
    assert base == camera_params("CAM03", 5, cam_label="BODY SHOP S30")
    assert base != camera_params("CAM03", 6, cam_label="BODY SHOP S30")
    assert base != camera_params("CAM04", 5, cam_label="BODY SHOP S30")
    p = frame_params(base, 12, 5)
    assert p == frame_params(base, 12, 5)
    assert p != frame_params(base, 13, 5)
    a, ga = process_frame(img, p, t=0.5, gt=gt)
    b, gb = process_frame(img, frame_params(base, 12, 5), t=0.5, gt=gt)
    assert np.array_equal(a, b)
    assert np.array_equal(ga.inst, gb.inst) and np.array_equal(ga.depth, gb.depth)
    c, _ = process_frame(img, frame_params(base, 13, 5), t=0.5)
    assert not np.array_equal(a, c)  # per-frame randomisation really changes the frame


def test_camera_param_ranges():
    for i in range(40):
        p = camera_params(f"CAM{i:02d}", i)
        assert -0.35 <= p.k1 <= -0.20 and abs(p.k2) <= 0.03
        assert 28 <= p.crf <= 36 and 25 <= p.keyint <= 50
        assert 0.65 <= p.sensor_scale <= 0.85


def test_sensor_scale_softens_fine_detail():
    """A 1-px checkerboard keeps less contrast through a lower effective sensor resolution; 1.0 is a no-op step."""
    import dataclasses
    from render.cctv.cctv_post import _sensor_stage
    h, w = 270, 480
    yy, xx = np.mgrid[:h, :w]
    img = np.where((xx + yy) % 2 == 0, 200, 50).astype(np.uint8)[..., None].repeat(3, axis=2)
    c0 = camera_params("CAM01", 0, size=(w, h))
    geo = build_geometry((w, h), (w, h), 0.0, c0.k1, c0.k2, c0.ca)
    quiet = dict(gain=1e-6, read_noise=0.0, chroma_noise=0.0, sharpen_amount=0.0, soften_sigma=0.0, vignette=0.0)
    p = dataclasses.replace(camera_params("CAM01", 0, size=(w, h)), **quiet)
    sharp = _sensor_stage(img, dataclasses.replace(p, sensor_scale=1.0), geo)
    soft = _sensor_stage(img, dataclasses.replace(p, sensor_scale=0.7), geo)
    core = (slice(20, -20), slice(20, -20))
    assert soft[core].std() < 0.5 * sharp[core].std()


@pytest.mark.parametrize("size,overscan", [((1920, 1080), 0.0), ((1280, 720), 0.0), ((1280, 720), 0.06)])
@pytest.mark.parametrize("sq", [SQ, (700, 300, 780, 380), (60, 340, 140, 420)])
def test_gt_alignment_box_matches_bright_square(size, overscan, sq):
    img, gt = synth_frame(sq=sq)
    if overscan:  # overscanned input: nominal view + margin on every side
        k = 1.0 / (1 - 2 * overscan)
        w, h = round(960 * k), round(540 * k)
        ox, oy = (w - 960) // 2, (h - 540) // 2
        big = np.full((h, w, 3), BG, np.uint8)
        binst = np.zeros((h, w), np.uint16)
        bdepth = np.full((h, w), 20.0, np.float32)
        big[oy:oy + 540, ox:ox + 960] = img
        binst[oy:oy + 540, ox:ox + 960] = gt.inst
        bdepth[oy:oy + 540, ox:ox + 960] = gt.depth
        img, gt = big, GtFrame(inst=binst, depth=bdepth, ids=gt.ids)
    base = camera_params("CAM01", 1, size=size, overscan=overscan)
    out, ogt = process_frame(img, frame_params(base, 3, 1), t=3.0, gt=gt)
    assert out.shape == (size[1], size[0], 3)
    assert ogt.inst.shape == (size[1], size[0]) and ogt.depth.shape == (size[1], size[0])
    assert len(ogt.boxes) == 1
    box = ogt.boxes[0]
    assert box["cls"] == "robot" and box["entity_id"] == "robot_07" and box["instance_id"] == 7
    got = np.array(bright_bbox(out))
    assert np.abs(got - np.array(box["bbox"])).max() <= 2, (got, box["bbox"])
    assert box["pixels"] == int((ogt.inst == 7).sum())
    assert np.all(ogt.depth[ogt.inst == 7] == 5.0)  # depth warped by the same map
    assert set(np.unique(ogt.inst)) == {0, 7}  # nearest neighbour: no new ids


def test_small_boxes_dropped():
    inst = np.zeros((100, 100), np.uint16)
    inst[10:14, 10:14] = 1  # 16 px
    inst[50:70, 50:70] = 2  # 400 px
    boxes = boxes_from_instances(inst, {"2": {"cls": "person", "entity_id": "p2"}}, min_pixels=24)
    assert [b["instance_id"] for b in boxes] == [2]
    assert boxes[0]["bbox"] == [50, 50, 70, 70] and boxes[0]["pixels"] == 400


def test_extra_flag_reaches_boxes_and_coco():
    from render.cctv.gt import build_coco
    inst = np.zeros((100, 100), np.uint16)
    inst[10:40, 10:40] = 1
    inst[50:80, 50:80] = 2
    ids = {"1": {"cls": "person", "entity_id": "W3"}, "2": {"cls": "person", "entity_id": "extra_1", "extra": True}}
    boxes = boxes_from_instances(inst, ids)
    assert ["extra" in b for b in boxes] == [False, True] and boxes[1]["extra"] is True
    anns = build_coco([{"file_name": "f.png", "width": 100, "height": 100, "boxes": boxes}])["annotations"]
    assert [a.get("extra", False) for a in anns] == [False, True]


def test_geometry_has_no_black_corners():
    for i in range(30):
        p = camera_params(f"CAM{i}", i)
        for src in ((960, 540), (1280, 720)):
            g = build_geometry(src, (1920, 1080), 0.0, p.k1, p.k2, p.ca)
            maps = [(g.map_x, g.map_y)] + ([(g.ca_maps[0], g.ca_maps[1]), (g.ca_maps[2], g.ca_maps[3])] if g.ca_maps else [])
            for mx, my in maps:
                assert mx.min() >= 0 and my.min() >= 0
                assert mx.max() <= src[0] - 1 and my.max() <= src[1] - 1
            assert 0.5 < g.zoom <= 1.0


def test_overlay_present():
    img = np.full((540, 960, 3), 90, np.uint8)
    base = camera_params("CAM03", 0, cam_label="BODY SHOP S30", overlay=True)
    p = frame_params(base, 0, 0)
    with_ov, _ = process_frame(img, p, 0.0, compress=False)
    without, _ = process_frame(img, p.__class__(**{**p.__dict__, "overlay": False}), 0.0, compress=False)
    h, w = with_ov.shape[:2]
    corner = (slice(0, int(h * 0.1)), slice(int(w * 0.8), w))  # SYNTHETIC, top-right
    assert np.abs(with_ov[corner].astype(int) - without[corner].astype(int)).max() > 80
    assert (with_ov[corner].mean(axis=2) > 220).sum() > 40  # legible white text pixels
    assert (without[corner].mean(axis=2) > 220).sum() == 0
    for reg in ((slice(0, int(h * 0.1)), slice(0, int(w * 0.3))), (slice(int(h * 0.9), h), slice(int(w * 0.7), w))):
        assert np.abs(with_ov[reg].astype(int) - without[reg].astype(int)).max() > 80  # timestamp, camera name
    assert np.array_equal(with_ov[h // 3: 2 * h // 3, w // 3: 2 * w // 3], without[h // 3: 2 * h // 3, w // 3: 2 * w // 3])


def test_timestamp_advances_with_time():
    p = camera_params("CAM03", 0, t0="2026-10-17 10:00:30")
    assert timestamp_text(p, 0.0) == "2026-10-17 10:00:30"
    assert timestamp_text(p, 1.04) == "2026-10-17 10:00:31"
    assert timestamp_text(p, 30.0) == "2026-10-17 10:01:00"


def test_still_goes_through_real_x264():
    img, _ = synth_frame()
    base = camera_params("CAM01", 2, crf=34)
    p = frame_params(base, 0, 2)
    raw, _ = process_frame(img, p, 0.0, compress=False)
    comp, _ = process_frame(img, p, 0.0, compress=True)
    assert raw.shape == comp.shape == (1080, 1920, 3)
    diff = np.abs(raw.astype(int) - comp.astype(int)).mean()
    assert 0.3 < diff < 20  # changed by the codec, but still the same picture


def test_sequence_video_and_annotations(tmp_path):
    frames, gtd = tmp_path / "in", tmp_path / "gt"
    frames.mkdir()
    gtd.mkdir()
    n = 6
    for i in range(n):
        img, gt = synth_frame(480, 270, sq=(100 + 10 * i, 80, 160 + 10 * i, 140))
        cv2.imwrite(str(frames / f"f_{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gtd / f"f_{i:05d}_inst.png"), gt.inst)
        np.save(gtd / f"f_{i:05d}_depth.npy", gt.depth)
    (gtd / "ids.json").write_text(json.dumps({"7": {"cls": "robot", "entity_id": "robot_07"}}))
    out = tmp_path / "out"
    base = camera_params("CAM03", 0, cam_label="BODY SHOP S30", size=(640, 360))
    r = process_sequence(frames, out, base, fps=25, seed=0, gt_dir=gtd, video=True)
    assert r["n"] == n
    mp4 = Path(r["video"])
    assert mp4.exists() and mp4.stat().st_size > 1000
    assert len(list(decode_video(mp4, 640, 360))) == n  # decodable
    cap = cv2.VideoCapture(str(mp4))
    if cap.isOpened():  # OpenCV's own ffmpeg backend (not guaranteed in every build)
        ok, fr = cap.read()
        assert ok and fr.shape == (360, 640, 3)
    cap.release()
    pngs = sorted((out / "frames").glob("*.png"))
    assert len(pngs) == n
    assert cv2.imread(str(pngs[0])).shape == (360, 640, 3)
    coco = json.loads((out / "annotations_coco.json").read_text())
    assert len(coco["images"]) == n and len(coco["annotations"]) == n
    assert coco["categories"] == [{"id": 1, "name": "robot"}]
    boxes = json.loads((out / "gt" / "f_00005_boxes.json").read_text())["boxes"]
    assert len(boxes) == 1
    out_frame = cv2.cvtColor(cv2.imread(str(pngs[5])), cv2.COLOR_BGR2RGB)
    got = np.array(bright_bbox(out_frame))
    assert np.abs(got - np.array(boxes[0]["bbox"])).max() <= 3  # decoded video frame still matches the GT box
    assert cv2.imread(str(out / "gt" / "f_00005_inst.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint16


def test_cli_single_png(tmp_path):
    img, _ = synth_frame()
    src = tmp_path / "a.png"
    cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert cli_main(["--in", str(src), "--out", str(tmp_path / "o"), "--cam", "CAM03", "--cam-label", "X", "--size", "1280x720"]) == 0
    assert cv2.imread(str(tmp_path / "o" / "a.png")).shape == (720, 1280, 3)


def test_wall_tiles_in_order_with_seams_and_label():
    from render.cctv.wall import GUTTER, compose
    cols = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]
    tiles = [np.full((540, 960, 3), c, np.uint8) for c in cols]
    w = compose(tiles, 640, 360)
    assert w.shape == (360, 640, 3) and w.dtype == np.uint8
    for (r, c), col in zip([(90, 160), (90, 480), (270, 160), (270, 480)], cols):
        assert tuple(w[r, c]) == col                       # TL, TR, BL, BR
    assert (w[180, 50] < 20).all() and (w[50, 320] < 20).all() and GUTTER >= 2
    assert (w[340:356, 300:340] > 230).all(axis=2).any()   # white label text bottom-centre
