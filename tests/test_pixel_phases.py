"""Pixels -> phases (intercycle/perception/pixel_phases.py) on synthetic frames: motion, sparks, flash, stabilisation,
tick averaging, and the unchanged PhaseModel recovering the cycle period from pixel features."""
import ast
from pathlib import Path

import numpy as np

from intercycle.perception import pixel_phases as PX

W, H = 160, 90
ROI = PX.Roi("S30", (40, 20, 120, 70))


def _bg(rng, noise=2.0):
    yy, xx = np.mgrid[0:H, 0:W]
    g = 90 + 30 * np.sin(xx / 7.0) * np.cos(yy / 5.0)          # textured static scene (ECC needs texture)
    return np.clip(g + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)


def _box(img, x, y, v=200, s=10):
    img = img.copy()
    img[y:y + s, x:x + s] = v
    return img


def test_motion_seen_only_while_the_part_moves():
    rng = np.random.default_rng(0)
    fe = PX.PixelFeatures([ROI])
    still = [fe.frame(_box(_bg(rng), 60, 30)) for _ in range(4)]
    moving = [fe.frame(_box(_bg(rng), 60 + 4 * k, 30)) for k in range(1, 5)]
    assert np.isnan(still[0]).all()                              # first frame: no previous frame
    assert all(f[0, 0] == 0 for f in still[1:])
    assert all(f[0, 0] > 0 for f in moving)
    assert moving[-1][0, 1:17].sum() == moving[-1][0, 0]         # the grid splits the same motion pixels


def test_sparks_are_not_motion_but_count_as_weld_on():
    rng = np.random.default_rng(1)
    fe = PX.PixelFeatures([ROI])
    fe.frame(_bg(rng))
    img = _bg(rng)
    for x, y in ((50, 30), (70, 40), (90, 55), (100, 25)):       # 2x2 saturated spatter dots
        img[y:y + 2, x:x + 2] = 255
    f = fe.frame(img)
    assert f[0, 0] == 0 and f[0, -1] > 0


def test_local_flash_gives_no_sample():
    rng = np.random.default_rng(2)
    fe = PX.PixelFeatures([ROI, PX.Roi("S40", (0, 0, 30, 30))])
    fe.frame(_bg(rng))
    img = _bg(rng).astype(np.int16)
    img[20:70, 40:120] += 80                                     # the arc lights the whole cell
    f = fe.frame(np.clip(img, 0, 255).astype(np.uint8))
    assert np.isnan(f[0]).all() and not np.isnan(f[1]).any() and fe.n_flash == 1


def test_stabilisation_cancels_camera_shake():
    rng = np.random.default_rng(3)
    base = _bg(rng, noise=0.0)
    shook = [np.roll(base, (dy, dx), (0, 1)) for dy, dx in ((0, 0), (1, 2), (-1, -1), (2, 0))]
    on, off = PX.PixelFeatures([ROI], stabilise=True), PX.PixelFeatures([ROI], stabilise=False)
    e_on = sum(np.nan_to_num(on.frame(f))[0, 0] for f in shook)
    e_off = sum(np.nan_to_num(off.frame(f))[0, 0] for f in shook)
    assert e_off > 10 and e_on < 0.1 * e_off


def test_ticks_average_frames_and_keep_nan():
    t = np.array([0.0, 0.1, 0.2, 0.5, 0.6, 1.0, 1.4])
    v = np.array([np.nan, 1, 3, 5, 7, np.nan, np.nan], np.float32)[:, None, None]
    T, X = PX.to_ticks(t + 100.0, v, hz=2.0)
    assert np.allclose(T, [100.0, 100.5, 101.0, 101.5])          # tick T holds the frames of (T - 0.5, T]
    assert np.isnan(X[0, 0, 0]) and X[1, 0, 0] == 3.0 and X[2, 0, 0] == 7.0 and np.isnan(X[3, 0, 0])


def test_ticks_at_frame_rate_keep_each_frame_time():
    t = 36000.0 + np.arange(6) / 2.0                              # 2 fps into 2 Hz ticks: no shift
    T, X = PX.to_ticks(t, np.arange(6, dtype=np.float32)[:, None, None], hz=2.0)
    assert np.allclose(T, t) and np.allclose(X[:, 0, 0], np.arange(6))


def test_rois_must_fit_the_frame(tmp_path):
    p = tmp_path / "rois.json"
    p.write_text('{"size": [160, 90], "rois": {"S30": [40, 20, 170, 70]}}', encoding="utf-8")
    try:
        PX.load_rois(p)
    except ValueError:
        pass
    else:
        raise AssertionError("a ROI outside the frame must be rejected")


def test_frames_must_match_the_roi_frame_size(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path / "f_00000.png"), np.zeros((100, 200), np.uint8))
    try:
        PX.extract_clip(tmp_path, [("f_00000.png", 0.0)], [ROI], (W, H))
    except ValueError:
        pass
    else:
        raise AssertionError("a frame of another size must be rejected")


def test_machine_block_in_layout_order():
    X = np.ones((3, 1, PX.N_PIX), np.float32)
    B = PX.machine_block(X, [ROI], ["S10", "S20", "S30"])
    assert B.shape == (3, 3, PX.N_PIX) and np.isnan(B[:, :2]).all() and (B[:, 2] == 1).all()


def test_phase_model_finds_the_cycle_period_from_pixels():
    """A part moves A -> B, holds (sparks), moves to C, holds, returns, then rests: 40 s cycles, 4 fps, 30 min."""
    from intercycle.observer.phases import Layout, PhaseModel
    rng = np.random.default_rng(4)
    bg = _bg(rng, noise=0.0)
    fe = PX.PixelFeatures([ROI])
    fps, per = 4.0, 40.0
    ts = np.arange(0.0, 1800.0, 1 / fps)
    feats = []
    for t in ts:
        u = (t % per) / per
        if u < 0.15:
            x, y = 45 + int(40 * u / 0.15), 25
        elif u < 0.35:
            x, y = 85, 25
        elif u < 0.5:
            x, y = 85, 25 + int(30 * (u - 0.35) / 0.15)
        elif u < 0.65:
            x, y = 85, 55
        elif u < 0.8:
            x, y = 85 - int(40 * (u - 0.65) / 0.15), 55 - int(30 * (u - 0.65) / 0.15)
        else:
            x, y = 45, 25
        img = _box(np.clip(bg + rng.normal(0, 2, bg.shape), 0, 255).astype(np.uint8), x, y)
        if 0.2 < u < 0.3:
            img[y + 4:y + 6, x + 12:x + 14] = 255
        feats.append(fe.frame(img))
    T, Xr = PX.to_ticks(ts + 36000.0, np.stack(feats))
    lay = Layout(["S30"], np.array([[0, 0, 8, 11]], float), np.array([True]), np.array([0]))
    X = PX.machine_block(Xr, [ROI], lay.mids)
    pm = PhaseModel.fit_features(T, X, np.zeros((len(T), 7), np.float32), lay, hz=2.0, fit_line=False)
    mp = pm.machines["S30"]
    assert mp is not None and abs(mp.period_s - per) < 1.0
    bins, *_ = pm.decode_features(T, X, np.zeros((len(T), 7), np.float32))
    assert len(np.unique(bins[:, 0])) >= 3


def test_pixel_path_never_imports_ground_truth():
    src = Path(PX.__file__).read_text(encoding="utf-8")
    mods = [n.module or "" for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if m.startswith("intercycle.truth")]
