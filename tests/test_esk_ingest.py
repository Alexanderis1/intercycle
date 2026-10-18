"""Per-clip ingest (intercycle/perception/esk_ingest.py): PTS sampling, class map, P5 rule, fresh tracker per clip,
occlusion = MISSING on the press ROI, contract tables, re-projection and track gaps."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from intercycle.perception import esk_ingest as ei
from intercycle.perception.detector import Dets

# ------------------------------------------------------------------------------------------------ pure helpers


def test_sample_indices_12fps_from_pts():
    t = np.arange(250) / 24.83                       # native 24.83 fps (the clips' 2483/100)
    idx = ei.sample_indices(t, 12.0)
    assert np.all(np.diff(idx) > 0)
    ts = t[idx]
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    assert abs(rate - 12.0) < 0.15
    # each sample is the native frame nearest its grid point
    grid = np.arange(len(idx)) / 12.0
    assert np.all(np.abs(ts - grid) <= 0.5 / 24.83 + 1e-9)


def test_sample_indices_uneven_pts_uses_times_not_indices():
    t = np.r_[np.arange(10) / 25.0, 0.4 + 1.0 + np.arange(10) / 25.0]   # a 1 s hole in the PTS
    ts = t[ei.sample_indices(t, 12.0)]
    assert not np.any((ts > 0.4) & (ts < 1.4))
    assert len(ts) < len(np.arange(0, ts[-1] + 1e-9, 1 / 12.0))           # no fake frames inside the hole


def _dets(boxes, scores, labels):
    return Dets(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                np.asarray(labels, dtype="<U16"))


def test_class_map_forklift_on_camera_14_only():
    d = _dets([[0, 0, 10, 10], [0, 0, 50, 50], [0, 0, 40, 40], [5, 5, 9, 9], [1, 1, 2, 2]],
              [0.9, 0.6, 0.45, 0.2, 0.05], ["person", "truck", "truck", "car", "person"])
    c14 = ei.apply_class_map(d, 14)
    assert list(c14.label) == ["person", "forklift"]
    c9 = ei.apply_class_map(d, 9)
    assert list(c9.label) == ["person"]                 # no vehicle class on camera 9
    assert ei.class_map(9) == {"person": ("person", 0.1)}
    assert ei.class_map(14)["truck"] == ("forklift", 0.5)


def test_native_missing_fails_closed_between_samples():
    ts = np.array([0.0, 0.1, 0.2, 0.3])
    hit = np.array([False, False, True, False])
    held = np.array([False, False, True, True])
    tn = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    m = ei.native_missing(tn, ts, hit, held)
    # 0.15 precedes the hit sample at 0.2 -> missing; 0.25..0.35 held
    assert m.tolist() == [False, False, False, True, True, True, True, True]
    assert ei.native_missing(tn, np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)).all()


def _tracks(rows):
    return pd.DataFrame(rows, columns=ei.TRACK_COLUMNS)


def test_project_tracks_fills_xy_and_leaves_unknown_framings_nan():
    tr = _tracks([["dev", 0.0, 0, "person", 100.0, 200.0, np.nan, np.nan, 0.9, "c9_f0"],
                  ["dev", 0.0, 1, "person", 10.0, 20.0, np.nan, np.nan, 0.9, "c9_f1"]])
    H = np.diag([0.01, 0.02, 1.0])
    out = ei.project_tracks(tr, {"c9_f0": H})
    assert out.loc[0, "x"] == pytest.approx(1.0) and out.loc[0, "y"] == pytest.approx(4.0)
    assert np.isnan(out.loc[1, "x"]) and np.isnan(out.loc[1, "y"])
    assert np.isnan(tr["x"]).all()                       # input untouched
    # k1 undistortion is applied before H
    from intercycle.framing import undistort
    k1 = -1.6e-7
    out2 = ei.project_tracks(tr, {"c9_f0": H}, {"c9_f0": k1})
    uu = undistort(np.array([[100.0, 200.0]]), k1)
    assert out2.loc[0, "x"] == pytest.approx(uu[0, 0] * 0.01)


def test_track_gaps_runs():
    fps = 12.0
    t = np.r_[0, 1, 2, 5, 6, 10] / fps                   # obs 3, gap 2, obs 2, gap 3, obs 1
    tr = _tracks([["dev", float(x), 0, "person", 0.0, 0.0, np.nan, np.nan, 0.5, "c9_f0"] for x in t])
    g = ei.track_gaps(tr, fps)
    assert g["kind"].tolist() == ["obs", "gap", "obs", "gap", "obs"]
    assert g["n_frames"].tolist() == [3, 2, 2, 3, 1]


def test_load_press_rois_by_framing():
    from pathlib import Path

    r = ei.load_press_rois(Path(__file__).resolve().parents[1] / "docs" / "probes" / "S3a-P4_rois.json")
    assert set(r) == {"c9_f0", "c9_f1", "c14_f0"}
    assert set(r["c9_f1"]) == {"P9a", "P9b"} and set(r["c9_f0"]) == {"P9a"}
    assert set(r["c14_f0"]) == {"P14a", "P14b", "P14c", "P14d"}


def test_module_has_no_split_logic():
    import inspect

    src = inspect.getsource(ei)
    assert "esk.load" not in src and "INTERCYCLE_SEALED_RUN" not in src and "split ==" not in src


# ------------------------------------------------------------------------------------------------ end to end (PyAV)

W, H = 640, 360
RATE = Fraction(2483, 100)
DUR = 4.0
ROI = [[300.0, 100.0], [380.0, 100.0], [380.0, 220.0], [300.0, 220.0]]
PERIOD = 1.0


def _write_clip(path, seed=0):
    av = pytest.importorskip("av")
    import cv2

    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (5, 5), 1.5)
    ram = cv2.GaussianBlur(rng.integers(0, 255, (60, 60)).astype(np.uint8), (5, 5), 1.5)
    n = int(DUR * float(RATE))
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=RATE)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.bit_rate = 8_000_000
        for i in range(n):
            t = i / float(RATE)
            ph = (t % PERIOD) / PERIOD
            y = int(round(110 + 30 * (np.sin(2 * np.pi * ph - np.pi / 2) + 1) / 2))
            f = bg.copy()
            f[y:y + 60, 310:370] = ram
            img = np.dstack([f, f, f])
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


class StubDetector:
    """Per sampled frame k (12 fps): a walker far from the ROI on every frame; a second person over the ROI for
    k in [12, 24) (t ~ 1-2 s); on camera 14 a forklift (truck 0.8) with its driver (foot inside), a weak truck (0.3)
    and a car, which the class map must drop. The walker is missed on frames 30-32 (a 3-frame gap)."""

    def __init__(self):
        self.k = 0

    def __call__(self, images):
        out = []
        for _ in images:
            k = self.k
            self.k += 1
            b, s, lab = [], [], []
            if not 30 <= k <= 32:
                x = 20 + 2 * k
                b.append([x, 250, x + 30, 340]); s.append(0.9); lab.append("person")
            if 12 <= k < 24:
                b.append([320, 150, 360, 260]); s.append(0.8); lab.append("person")
            if 40 <= k < 44:                                                    # weak untracked box on the ROI
                b.append([300, 100, 330, 140]); s.append(0.2); lab.append("person")
            b.append([480, 200, 600, 330]); s.append(0.8); lab.append("truck")
            b.append([520, 210, 550, 300]); s.append(0.7); lab.append("person")     # driver: foot (535, 300) inside
            b.append([100, 20, 150, 60]); s.append(0.3); lab.append("truck")
            b.append([200, 20, 260, 60]); s.append(0.9); lab.append("car")
            out.append(_dets(b, s, lab))
        return out


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    pytest.importorskip("av")
    pytest.importorskip("trackers")
    p = tmp_path_factory.mktemp("clip") / "3_tr1.mp4"
    _write_clip(p)
    return p


def _run(path, camera, split="dev", cfg=None):
    return ei.ingest_clip(path, "3_tr1", split, camera, "c14_f0" if camera == 14 else "c9_f0",
                          {"PX": ROI}, StubDetector(), cfg)


def test_weak_untracked_boxes_do_not_occlude_unless_configured(clip_path):
    t_weak = (40 / 12.0, 44 / 12.0)
    for cfg, want in ((None, False), (ei.IngestConfig(occ_conf=0.1), True)):
        m = _run(clip_path, 9, cfg=cfg).machines["PX"]
        tt = m["t"].to_numpy(float)
        w = (tt > t_weak[0] + 0.05) & (tt < t_weak[1] - 0.05)
        assert w.any() and bool(m["missing"].to_numpy(bool)[w].all()) is want


def test_end_to_end_tables_validate_and_missing(clip_path):
    from twin.schemas.tables import MachineRow, TrackRow, validate_table
    import pyarrow as pa

    r = _run(clip_path, 14)
    st = r.stats
    assert abs(st["fps_sampled"] - 12.0) < 0.3 and abs(st["fps_native"] - float(RATE)) < 0.1
    # tracks contract
    assert list(r.tracks.columns) == ei.TRACK_COLUMNS
    assert validate_table(pa.Table.from_pandas(r.tracks, preserve_index=False), TrackRow) == len(r.tracks)
    assert np.isnan(r.tracks[["x", "y"]].to_numpy(float)).all()
    # forklift tracked, driver suppressed (no person track with a foot point inside the forklift box)
    assert st["n_tracks"]["forklift"] == 1
    assert st["suppressed"] == st["n_samples"]
    per = r.tracks.loc[r.tracks["class"] == "person", ["u", "v"]].to_numpy(float)
    assert not ((per[:, 0] >= 480) & (per[:, 0] <= 600) & (per[:, 1] >= 200) & (per[:, 1] <= 330)).any()
    assert set(r.boxes["class"]) == {"person", "forklift"}            # weak truck and car dropped
    # occlusion = MISSING on the ROI while the second person stands on it (t ~ 1-2 s), held ~1.33 s after
    m = r.machines["PX"]
    assert validate_table(pa.Table.from_pandas(m, preserve_index=False), MachineRow) == len(m)
    tt = m["t"].to_numpy(float)
    miss = m["missing"].to_numpy(bool)
    occ = (tt > 1.05) & (tt < 1.9)
    assert miss[occ].all() and np.isnan(m["flow_v"].to_numpy(float)[occ]).all()
    assert not m["stroke"].to_numpy(bool)[occ].any()
    t_last_hit = 23 / 12.0
    after = tt > t_last_hit + st["occ_hold_s"] + 0.2
    assert after.any() and not miss[after].any()
    assert not miss[1:][tt[1:] < 0.9].any()
    assert miss[0]                                                     # frame 0 has no flow -> missing
    # the ram is still counted outside the occlusion
    assert st["presses"]["PX"]["n_strokes"] >= 2


def test_camera_9_has_no_forklift_and_fresh_tracker_per_clip(clip_path):
    a = _run(clip_path, 9)
    b = _run(clip_path, 9)
    assert a.stats["n_tracks"]["forklift"] == 0 and a.stats["forklift_boxes"] == 0
    assert a.stats["suppressed"] == 0
    # fresh tracker: identical output on a second run, ids start at 0
    pd.testing.assert_frame_equal(a.tracks, b.tracks)
    assert a.tracks["track_id"].min() == 0
    # walker missed on sampled frames 30-32 -> one 3-frame gap (or a new id) in the walker's track
    g = ei.track_gaps(a.tracks, 12.0)
    assert (g.loc[g["kind"] == "gap", "n_frames"] == 3).any() or a.stats["n_tracks"]["person"] >= 3


def test_clip_without_detections_writes_typed_empty_tables(tmp_path, clip_path):
    from twin.schemas.tables import TrackRow, validate_table

    r = ei.ingest_clip(clip_path, "3_tr1", "dev", 9, "c9_f0", {"PX": ROI}, lambda ims: [Dets.empty() for _ in ims])
    assert len(r.tracks) == 0 and len(r.boxes) == 0
    ei.write_outputs(r, tmp_path)
    assert validate_table(tmp_path / "tracks" / "3_tr1.parquet", TrackRow) == 0
    import pyarrow.parquet as pq
    assert str(pq.read_schema(tmp_path / "boxes" / "3_tr1.parquet").field("class").type) != "null"


def test_write_outputs(tmp_path, clip_path):
    r = _run(clip_path, 14)
    paths = ei.write_outputs(r, tmp_path)
    assert (tmp_path / "tracks" / "3_tr1.parquet").exists()
    assert (tmp_path / "machines" / "PX" / "3_tr1.parquet").exists()
    assert (tmp_path / "boxes" / "3_tr1.parquet").exists()
    assert len(paths) == 3
    back = pd.read_parquet(tmp_path / "tracks" / "3_tr1.parquet")
    assert list(back.columns) == ei.TRACK_COLUMNS
