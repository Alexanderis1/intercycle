"""Seeded DEV label sample (lane S4-labels, PLAN_TWIN §5 "Split and labels", U14): scripts/label_sample.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from twin import schemas as S

_SPEC = importlib.util.spec_from_file_location(
    "label_sample", Path(__file__).resolve().parents[1] / "scripts" / "label_sample.py")
assert _SPEC and _SPEC.loader
LS = importlib.util.module_from_spec(_SPEC)
sys.modules["label_sample"] = LS
_SPEC.loader.exec_module(LS)


def _inv() -> pd.DataFrame:
    rows = []
    # camera 9: 3 groups, framings c9_f0 / c9_f1; camera 14: 4 groups
    for g, fr, n, dur in [("c9_a0", "c9_f1", 4, 12.0), ("c9_a1", "c9_f0", 3, 8.0), ("c9_a2", "c9_f1", 2, 6.0)]:
        for k in range(n):
            rows.append(dict(clip_id=f"0_{g}_{k}", camera=9, framing_id=fr, group_id=g, duration=dur, split="dev"))
    for g, n, dur in [("c14_a0", 2, 7.0), ("c14_a1", 1, 4.0), ("c14_a2", 3, 3.0), ("c14_a3", 1, 10.0)]:
        for k in range(n):
            rows.append(dict(clip_id=f"3_{g}_{k}", camera=14, framing_id="c14_f0", group_id=g, duration=dur,
                             split="dev"))
    df = pd.DataFrame(rows)
    df["sha256"] = [f"{i:064x}" for i in range(len(df))]       # all distinct
    return df


def _inv_with_copies() -> pd.DataFrame:
    """_inv() plus byte-identical copies (same sha256, other clip id) of three camera-9 clips."""
    inv = _inv()
    src = pd.DataFrame(inv[inv["clip_id"].isin(["0_c9_a0_0", "0_c9_a1_0", "0_c9_a2_1"])]).copy()
    src["clip_id"] = ["9_copy_" + c for c in src["clip_id"]]
    return pd.DataFrame(pd.concat([inv, src], ignore_index=True))


def _pts(inv: pd.DataFrame):
    dur = dict(zip(inv.clip_id, inv.duration))
    return lambda cid: [round(i / 25.0, 6) for i in range(int(round(dur[cid] * 25)))]   # 25 fps from 0


PRESSES = {"P9a": ["c9_f0", "c9_f1"], "P9b": ["c9_f1"]}


def _build(seed=7, **kw):
    inv = _inv()
    args = dict(n_frames={9: 6, 14: 6}, n_windows={"P9a": 3, "P9b": 2}, n_reserve=1)
    args.update(kw)
    return LS.build_sample_list(inv, _pts(inv), PRESSES, split="dev", seed=seed, **args)


def test_frame_grid_is_one_frame_every_2s_snapped_to_pts():
    pts = [i / 25.0 for i in range(250)]                      # 10 s at 25 fps
    g = LS.frame_grid(pts)
    assert g == [1.0, 3.0, 5.0, 7.0, 9.0]
    assert LS.frame_grid([0.0, 0.04, 0.08]) == []             # a clip shorter than 1 s has no grid frame
    assert LS.frame_grid([0.5 + i / 30.0 for i in range(300)])[0] == pytest.approx(1.5)


def test_sample_list_validates_and_is_deterministic():
    a, b = _build(), _build()
    assert a.items_sha256 == b.items_sha256
    assert _build(seed=8).items_sha256 != a.items_sha256
    S.SampleList.model_validate(a.model_dump(mode="json"))    # the schema hash check passes


def test_frames_are_stratified_by_camera_and_spread_over_groups():
    sl = _build()
    fr = [i for i in sl.items if i.kind == "frame"]
    assert sum(i.camera == 9 for i in fr) == 6 and sum(i.camera == 14 for i in fr) == 6
    assert {i.group_id for i in fr if i.camera == 14} == {"c14_a0", "c14_a1", "c14_a2", "c14_a3"}
    assert {i.group_id for i in fr if i.camera == 9} == {"c9_a0", "c9_a1", "c9_a2"}
    keys = [(i.clip_id, i.t_pts) for i in fr]
    assert len(set(keys)) == len(keys)
    for i in fr:                                              # every frame is on the 2 s grid of its clip
        assert i.t_pts is not None and abs((i.t_pts - 1.0) / 2.0 - round((i.t_pts - 1.0) / 2.0)) < 1e-9


def test_stroke_windows_use_only_the_press_framings_and_are_at_most_10s():
    sl = _build()
    inv = _inv().set_index("clip_id")
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    assert sum(i.press == "P9a" for i in sw) == 3 + 1 and sum(i.press == "P9b" for i in sw) == 2 + 1
    for i in sw:
        assert inv.loc[i.clip_id, "framing_id"] in PRESSES[i.press]
        assert i.t0 is not None and i.t1 is not None and 0 < i.t1 - i.t0 <= 10.0
        dur = float(inv.loc[i.clip_id, "duration"])
        if dur <= 10.0:                                        # whole clip
            assert i.t0 == 0.0 and i.t1 == pytest.approx(dur - 1 / 25.0)
    per_press = {}
    for i in sw:
        per_press.setdefault(i.press, []).append(i.clip_id)
    assert all(len(v) == len(set(v)) for v in per_press.values())   # one window per (press, clip)
    assert [i.sample_id for i in sw if i.press == "P9a"][-1].endswith("r1")


def test_window_start_is_seeded_random_in_long_clips():
    pts = [i / 25.0 for i in range(25 * 20)]                  # 20 s
    import numpy as np
    starts = {LS.window_for(pts, np.random.default_rng(s))[0] for s in range(10)}
    assert len(starts) > 1
    for s in range(10):
        t0, t1 = LS.window_for(pts, np.random.default_rng(s))
        assert t0 in pts and t1 in pts and 9.9 < t1 - t0 <= 10.0


def test_refuses_test_split_and_short_supply():
    inv = _inv()
    with pytest.raises(ValueError, match="dev"):
        LS.build_sample_list(inv, _pts(inv), PRESSES, split="test", seed=1, n_frames={9: 1, 14: 1},
                             n_windows={"P9a": 1}, n_reserve=0)
    with pytest.raises(ValueError, match="only"):
        _build(n_frames={9: 6, 14: 500})


def test_hash_file_lines():
    sl = _build()
    txt = LS.hash_text(sl, b"{}")
    assert txt.splitlines()[0].startswith(sl.items_sha256)


def test_refuses_inventory_rows_outside_dev():
    inv = _inv()
    inv.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="inventory rows outside dev"):
        LS.build_sample_list(inv, _pts(inv), PRESSES, split="dev", seed=1, n_frames={9: 1, 14: 1},
                             n_windows={"P9a": 1}, n_reserve=0)
    base = _build()
    with pytest.raises(ValueError, match="inventory rows outside dev"):
        LS.build_supplement(inv, _pts(inv), PRESSES, base, split="dev", seed=1, n_windows={"P9a": 1},
                            n_reserve={}, n_frames={})


def test_dedupe_keeps_one_clip_per_sha256_and_fails_closed_without_sha():
    inv = _inv_with_copies()
    d = LS.dedupe_by_sha(inv)
    assert len(d) == len(_inv()) and d["sha256"].is_unique
    assert not d["clip_id"].str.startswith("9_copy_").any()   # the smallest clip_id of each set is kept
    with pytest.raises(ValueError, match="sha256"):
        LS.dedupe_by_sha(inv.drop(columns=["sha256"]))


def test_sample_list_never_samples_two_byte_copies():
    inv = _inv_with_copies()
    sl = LS.build_sample_list(inv, _pts(inv), PRESSES, split="dev", seed=3, n_frames={9: 12, 14: 6},
                              n_windows={"P9a": 4, "P9b": 2}, n_reserve=1)
    sha = dict(zip(inv.clip_id, inv.sha256))
    assert not any(i.clip_id.startswith("9_copy_") for i in sl.items)
    for press in ("P9a", "P9b"):
        s = [sha[i.clip_id] for i in sl.items if i.kind == "stroke_window" and i.press == press]
        assert len(s) == len(set(s))
    fr = [(sha[i.clip_id], i.t_pts) for i in sl.items if i.kind == "frame"]
    assert len(fr) == len(set(fr))


def _supplement(inv, base, seed=11, presses=None, **kw):
    args = dict(n_windows={"P9a": 1, "P9b": 1}, n_reserve={"P9a": 1}, n_frames={9: 1})
    args.update(kw)
    return LS.build_supplement(inv, _pts(inv), presses or {"P9a": ["c9_f0"], "P9b": ["c9_f1"]}, base,
                               split="dev", seed=seed, **args)


def test_supplement_excludes_base_sha_per_press_and_restricts_framings():
    inv = _inv_with_copies()
    # a base list drawn the OLD way (no dedupe) may name byte copies; the supplement maps them through sha256
    base = LS.build_sample_list(inv, _pts(inv), PRESSES, split="dev", seed=5, n_frames={9: 6, 14: 2},
                                n_windows={"P9a": 2, "P9b": 2}, n_reserve=0, dedupe=False)
    sup = _supplement(inv, base)
    S.SampleList.model_validate(sup.model_dump(mode="json"))
    assert sup.items_sha256 == _supplement(inv, base).items_sha256          # deterministic
    sha = dict(zip(inv.clip_id, inv.sha256))
    fr_of = dict(zip(inv.clip_id, inv.framing_id))
    for it in sup.items:
        assert not it.clip_id.startswith("9_copy_")
        if it.kind == "stroke_window":
            used = {sha[b.clip_id] for b in base.items if b.kind == "stroke_window" and b.press == it.press}
            assert sha[it.clip_id] not in used
            assert fr_of[it.clip_id] == {"P9a": "c9_f0", "P9b": "c9_f1"}[it.press]
            assert it.sample_id.startswith(f"dev-sws1-{it.press}-")
        else:
            used_f = {(sha[b.clip_id], b.t_pts) for b in base.items if b.kind == "frame"}
            assert (sha[it.clip_id], it.t_pts) not in used_f and it.camera == 9
            assert it.sample_id == "dev-fs1-c9-01"
    ids = [i.sample_id for i in sup.items if i.kind == "stroke_window" and i.press == "P9a"]
    assert ids == ["dev-sws1-P9a-01", "dev-sws1-P9a-r1"]
    assert base.items_sha256 in sup.rule


def test_supplement_refuses_short_supply_and_test_split():
    inv = _inv()
    base = _build()
    with pytest.raises(ValueError, match="only"):
        _supplement(inv, base, n_windows={"P9a": 50}, n_reserve={}, n_frames={})
    with pytest.raises(ValueError, match="dev"):
        LS.build_supplement(inv, _pts(inv), PRESSES, base, split="test", seed=1, n_windows={"P9a": 1},
                            n_reserve={}, n_frames={})
