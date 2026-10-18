"""DAYGEN v1 (twin/daygen.py) and DYN v1 (twin/dynamics.json): schema, keyed RNG (W-W1), seed guard, provenance,
library hash check. Runs without $INTERCYCLE_DATA: a small synthetic library stands in for the real one (its
segments carry real TRAIN clip ids from dynamics.json so the provenance check is the real one)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intercycle import esk
from twin import daygen as G
from twin.provenance import assert_train_only, source_clips
from twin.schemas import Dynamics, Site, load, parse_split
from twin.schemas.dayscript import Segment

REPO = Path(__file__).resolve().parents[1]


def _split_rows():
    return parse_split(esk.active_split_path(REPO).read_text(encoding="utf-8"))


def _library(tables: dict, dyn: Dynamics) -> G.Library:
    units, segs = [], {}
    clips9 = sorted(dyn.get("person.z9.path_library").clips)
    clips14 = sorted(dyn.get("forklift.z14.path_library").clips)
    k = 0
    for zone, cls, clips in (("z9", "person", clips9), ("z14", "person", sorted(dyn.get("person.z14.path_library").clips)),
                             ("z14", "forklift", clips14)):
        for gid, st in sorted(tables["layout_states"][zone].items()):
            for unsafe in (False, True):
                cid = clips[k % len(clips)]
                k += 1
                sid = f"seg:{cid}:{cls}:{k}:0"
                x0 = 1.0 if zone == "z9" else 20.0
                segs[sid] = Segment(id=sid, cls=cls, source="resampled", clip_id=cid, track_id=k,  # type: ignore[arg-type]
                                    points=[(0.0, x0, 0.0), (2.0, x0 + 1.0, 0.5), (4.0, x0 + 2.0, 1.0)])
                units.append(G.Unit(unit_id=f"u{k}", zone=zone, cls=cls, group_id=gid, lum_class=st["lum_class"],
                                    unsafe=unsafe, framing=st["framing"], clip_id=cid, clip_dur=6.0, unit_t0=0.5,
                                    members=[(sid, 0.0)]))
    return G.Library(units=units, segments=segs, sha256="synthetic")


@pytest.fixture(scope="module")
def inp() -> G.Inputs:
    dyn = load(Dynamics, G.DYNAMICS)
    tables = json.loads(G.PRESS_TABLES.read_text(encoding="utf-8"))
    return G.load_inputs(site=load(Site, G.SITE), dynamics=dyn, tables=tables, library=_library(tables, dyn))


def test_dynamics_v1_contract_and_train_only_provenance():
    dyn = load(Dynamics, G.DYNAMICS)
    for name, p in [*dyn.observed.items(), *dyn.assumed.items()]:
        assert p.source in ("fitted", "fitted_censored", "prior", "assumed"), name
        assert p.sweep.lo <= p.value <= p.sweep.hi, name
        if p.source.startswith("fitted"):
            assert p.n > 0 and p.clips, name
    assert all(p.source.startswith("fitted") for n, p in dyn.observed.items() if not n.startswith("press.P14"))
    assert dyn.assumed["site.night_share"].time_of_day
    assert assert_train_only({"dynamics": dyn.train_clips()}, _split_rows()) > 0
    # camera 9 has no vehicle class (P5): no forklift arrivals in zone 9
    assert dyn.get("zone.z9.forklift_rate_per_h").value == 0.0


def test_day_validates_is_deterministic_and_days_differ(inp):
    a = G.generate_day(3, 1, inputs=inp, duration_s=1800.0)
    b = G.generate_day(3, 1, inputs=inp, duration_s=1800.0)
    assert a.model_dump() == b.model_dump()
    assert a.generator == "twin.daygen" and a.duration_s == 1800.0
    assert a.mover_intents and all(s.source == "resampled" for s in a.segments)
    assert G.generate_day(3, 2, inputs=inp, duration_s=1800.0).model_dump() != a.model_dump()
    assert G.generate_day(4, 1, inputs=inp, duration_s=1800.0).model_dump() != a.model_dump()


def test_generated_segments_are_train_clips(inp):
    ds = G.generate_day(7, 1, inputs=inp, duration_s=3600.0)
    refs = source_clips(day_scripts=[ds])
    assert assert_train_only(refs, _split_rows()) > 0


def test_seed_guard(inp, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(PermissionError):
        G.generate_day(1000, 1, inputs=inp, duration_s=60.0)
    G.generate_day(999, 1, inputs=inp, duration_s=60.0)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    G.generate_day(1000, 1, inputs=inp, duration_s=60.0)


def test_entity_draws_do_not_depend_on_order_or_other_entities(inp):
    full = G.generate_day(11, 4, inputs=inp, duration_s=1800.0)
    rev = G.generate_day(11, 4, inputs=inp, duration_s=1800.0,
                         entity_order=["forklifts:z14", "persons:z14", "P14a", "persons:z9", "forklifts:z9", "P9b",
                                       "P9a", "P14b", "P14c", "P14d"])
    assert rev.model_dump() == full.model_dump()
    only = G.generate_day(11, 4, inputs=inp, duration_s=1800.0, entity_order=["P9a", "persons:z14"])
    assert [e for e in only.press_events] == [e for e in full.press_events if e.press == "P9a"]
    z14 = [m for m in full.mover_intents if m.zone == "z14" and m.cls == "person"]
    assert only.mover_intents == z14


def test_press_running_share_tracks_the_world_draw(inp):
    ds = G.generate_day(5, 1, inputs=inp)
    for p in ("P9a", "P9b"):
        ev = [e for e in ds.press_events if e.press == p]
        share = sum(e.t1 - e.t0 for e in ev if e.kind == "running") / ds.duration_s
        assert abs(share - G.world_p_running(5, p, inp.dyn)) < 0.15
        assert all(e.source == "observed" for e in ev if e.kind == "running")
        assert all(e.source == "assumed" and e.cause == "long_stop" for e in ev if e.kind == "stopped")
    assert G.world_p_running(5, "P9a", inp.dyn) == G.world_p_running(5, "P9a", inp.dyn)


def test_night_mapping_is_centred_on_midnight():
    assert G.is_night(0.0, 0.25) and G.is_night(G.DAY_S - 1.0, 0.25)
    assert not G.is_night(G.DAY_S / 2, 0.25)
    assert not G.is_night(4 * 3600.0, 0.25) and G.is_night(2.9 * 3600.0, 0.25)


def test_library_hash_mismatch_fails_closed(tmp_path):
    dyn = load(Dynamics, G.DYNAMICS)
    p = tmp_path / "lib.npz"
    np.savez(p, unit_id=np.array(["u"]), seg_id=np.array(["s"]))
    with pytest.raises(ValueError, match="sha256"):
        G.load_library(dyn, p)


def test_pool_falls_back_when_a_narrowing_is_empty(inp):
    lib = inp.lib
    assert lib.pool("z9", "person", "no_such_state", True)
    assert all(u.unsafe for u in lib.pool("z9", "person", None, True))
    assert lib.pool("z9", "forklift", None, None) == []


def test_validation_sample_shape(inp, monkeypatch):
    tg = np.arange(0, 8.0, 0.04)
    clips = {"P9a": [(tg, np.zeros(len(tg), bool))]}
    res = G.validation_sample(clips, seeds=(901,), windows_per_day=3, inputs=inp)
    assert set(res) >= {"speeds", "intervals", "n_windows", "seeds", "footage_unsafe_share"}
    assert res["intervals"].get("P9a")
    assert all(1.0 < v < 3.0 for v in res["intervals"]["P9a"])
