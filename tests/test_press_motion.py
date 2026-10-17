"""Press-motion ingest on synthetic clips: a textured ram moving down and up, camera shake, lamp beat, a person box."""
import numpy as np
import pytest

from intercycle.perception import press_motion as pm

FPS = 25.0
W, H = 640, 360
SCALE = W / 1920
ROI = [[900.0, 150.0], [1140.0, 150.0], [1140.0, 750.0], [900.0, 750.0]]  # full-res px


def _texture(rng, h, w):
    import cv2
    a = rng.integers(0, 255, (h, w)).astype(np.uint8)
    return cv2.GaussianBlur(a, (5, 5), 1.5)


def _ram_y(t, period, down_s=0.4, travel=40.0):
    """Ram offset (working px, + = down): down over down_s, up over down_s, then dwell."""
    ph = np.mod(t, period)
    y = np.where(ph < down_s, ph / down_s, np.where(ph < 2 * down_s, 1 - (ph - down_s) / down_s, 0.0))
    return travel * y


def make_clip(period=1.6, dur=7.0, shake=0.0, beat_hz=0.0, seed=0, jitter=0.002):
    rng = np.random.default_rng(seed)
    bg = _texture(rng, H, W)
    ram = _texture(rng, 120, 60)
    n = int(dur * FPS)
    t = np.arange(n) / FPS + rng.uniform(-jitter, jitter, n)
    t[0] = 0.0
    frames = np.empty((n, H, W), np.uint8)
    x0 = int(1000 * SCALE)
    for i, ti in enumerate(t):
        f = bg.copy()
        y = int(round(60 + _ram_y(np.array([ti]), period)[0]))
        f[y:y + 120, x0:x0 + 60] = ram
        if shake:
            dx, dy = rng.normal(0, shake, 2)
            import cv2
            f = cv2.warpAffine(f, np.array([[1, 0, dx], [0, 1, dy]], np.float32), (W, H), borderMode=cv2.BORDER_REPLICATE)
        if beat_hz:
            f = np.clip(f.astype(float) + 6 * np.sin(2 * np.pi * beat_hz * ti), 0, 255).astype(np.uint8)
        frames[i] = f
    return pm.Clip(t=t, frames=frames, scale=SCALE)


def run(clip, boxes=None):
    return pm.ingest_clip("", "c", "train", "P9a", "c9_f0", ROI, person_boxes=boxes, clip=clip)


def test_strokes_and_period_on_clean_clip():
    r = run(make_clip(period=1.6, dur=7.0))
    assert len(r.strokes_t) == 5                     # strokes start at 0, 1.6, 3.2, 4.8, 6.4 s
    iv, cens = pm.intervals(r.strokes_t, r.extra["duration"])
    assert np.all(np.abs(iv - 1.6) < 0.1)
    assert cens is not None and 0 < cens < 1.6
    assert list(r.table.columns) == pm.MACHINE_COLUMNS


def test_flow_sign_is_down_positive():
    r = run(make_clip(period=1.6, dur=3.0))
    v = r.table["flow_v"].to_numpy()
    t = r.table["t"].to_numpy()
    down = (t > 1.65) & (t < 1.95)
    assert np.nanmean(v[down]) > 0


def test_idle_press_has_no_strokes_under_shake():
    c = make_clip(period=1e9, dur=6.0, shake=1.0)    # ram never moves (period never reached after t=0 dwell)
    c.frames[:] = c.frames[:]                        # ram parked
    r = run(c)
    assert len(r.strokes_t) <= 1                     # at most the initial descent at t=0
    assert r.stab_px > 0


def test_shake_does_not_break_stroke_count():
    r = run(make_clip(period=2.0, dur=8.0, shake=0.8, seed=3))
    assert len(r.strokes_t) == 4


def test_lamp_beat_detected():
    r = run(make_clip(period=1e9, dur=8.0, beat_hz=0.5))
    assert abs(r.beat_hz - 0.5) < 0.06
    assert r.beat_frac > 0.5


def test_person_box_makes_rows_missing():
    c = make_clip(period=1.6, dur=4.0)
    n = len(c.t)
    boxes = [[] for _ in range(n)]
    for i in range(30, 60):
        boxes[i] = [[950.0, 400.0, 1050.0, 900.0]]   # overlaps the ROI
    boxes[10] = [[100.0, 100.0, 200.0, 300.0]]        # far away: not missing
    r = run(c, boxes)
    tab = r.table
    assert tab["missing"].iloc[30:60].all() and not tab["missing"].iloc[10]
    assert tab.loc[tab["missing"], "flow_v"].isna().all()
    assert not (tab["missing"] & tab["stroke"]).any()


def test_rows_satisfy_contract():
    schemas = pytest.importorskip("twin.schemas.tables")
    r = run(make_clip(period=1.6, dur=3.0))
    for rec in r.table.to_dict("records"):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        schemas.MachineRow(**rec)


def test_lamp_beat_flat_series():
    f, frac, amp = pm.lamp_beat(np.arange(100) / 25.0, np.full(100, 100.0))
    assert frac == 0.0 or amp < 1e-6
