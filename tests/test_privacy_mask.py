"""Track-held privacy mask (PLAN_TWIN §3 P8): conf >= 0.1, whole box, 15 % pad, held over the track and short gaps."""
import numpy as np
import pytest

from intercycle.perception.privacy_mask import MaskConfig, TrackHoldMasker, apply_mask, pad_boxes

BOX = [100.0, 200.0, 140.0, 300.0]  # w 40, h 100
PADDED = [94.0, 185.0, 146.0, 315.0]


def _covers(masks: np.ndarray, box) -> bool:
    x0, y0, x1, y1 = box
    return any(m[0] <= x0 and m[1] <= y0 and m[2] >= x1 and m[3] >= y1 for m in masks)


def test_pad_is_15_percent_each_side():
    np.testing.assert_allclose(pad_boxes(np.array([BOX]), 0.15)[0], PADDED)


def test_threshold_is_0_1_and_whole_box():
    m = TrackHoldMasker()
    out = m.update(0.0, [BOX, [500, 500, 520, 560]], [0.1, 0.09])
    assert len(out) == 1  # 0.1 is masked, 0.09 is not
    np.testing.assert_allclose(out[0], PADDED)


def test_unconfirmed_low_conf_box_is_masked_in_its_own_frame():
    # no track id at all (e.g. below ByteTrack's activation): still masked immediately
    out = TrackHoldMasker().update(0.0, [BOX], [0.12], track_id=[None])
    assert _covers(out, BOX)


def test_held_through_short_gap_then_dropped():
    cfg = MaskConfig(hold_s=1.0, grow_per_s=0.5)
    m = TrackHoldMasker(cfg)
    m.update(0.0, [BOX], [0.9], track_id=[7])
    held = m.update(0.5, [], [], track_id=[])  # detector missed the person
    assert len(held) == 1 and _covers(held, PADDED)
    # grown by 0.5 box-heights/s * 0.5 s * 100 px = 25 px on each side
    np.testing.assert_allclose(held[0], [94 - 25, 185 - 25, 146 + 25, 315 + 25])
    assert len(m.update(1.0, [], [])) == 1  # still inside hold_s
    assert len(m.update(1.6, [], [])) == 0  # hold expired


def test_track_id_continues_and_follows_the_box():
    m = TrackHoldMasker()
    m.update(0.0, [BOX], [0.8], track_id=[3])
    moved = [130.0, 200.0, 170.0, 300.0]
    out = m.update(0.25, [moved], [0.8], track_id=[3])
    assert len(out) == 1  # same track: one mask, now on the new position
    assert _covers(out, moved)


def test_box_without_id_continues_a_held_track_by_iou():
    m = TrackHoldMasker()
    m.update(0.0, [BOX], [0.9], track_id=[5])
    m.update(0.25, [], [])  # gap
    near = [104.0, 200.0, 144.0, 300.0]
    out = m.update(0.5, [near], [0.15])  # low-conf re-detection without an id
    assert len(out) == 1 and _covers(out, near)
    # the track's hold restarts from the re-detection
    assert len(m.update(1.4, [], [])) == 1


def test_two_people_kept_apart():
    m = TrackHoldMasker()
    far = [800.0, 200.0, 840.0, 300.0]
    m.update(0.0, [BOX, far], [0.9, 0.3])
    out = m.update(0.5, [far], [0.3])  # first person missed: held; second still detected
    assert len(out) == 2 and _covers(out, BOX) and _covers(out, far)


def test_time_order_and_reset():
    m = TrackHoldMasker()
    m.update(1.0, [BOX], [0.9])
    with pytest.raises(ValueError):
        m.update(0.5, [], [])
    m.reset()
    assert len(m.update(0.0, [], [])) == 0  # no track survives a reset (next clip)


def test_clip_to_frame_and_apply_mask_blacks_whole_box():
    img = np.full((60, 80, 3), 200, np.uint8)
    m = TrackHoldMasker(width=80, height=60)
    out = m.update(0.0, [[60.0, 10.0, 80.0, 50.0]], [0.5])
    assert out[0, 2] == 80 and out[0, 0] == 57  # 60 - 0.15 * 20
    masked = apply_mask(img, out)
    assert (masked[4:56, 57:80] == 0).all()  # whole padded box is black (y 4..56 after 15 % of 40 px)
    assert (masked[:, :57] == 200).all()
    assert (img == 200).all()  # input untouched
