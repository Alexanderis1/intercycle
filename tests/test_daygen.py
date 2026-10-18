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


# ------------------------------------------------------------------------------------------------ fix round 1

@pytest.mark.parametrize("call", [
    lambda inp, s: G.world_p_running(s, "P9a", inp.dyn),
    lambda inp, s: G.day_states(s, 1, "z9", inp.tables),
    lambda inp, s: G.staffing(s, 1, inp.dyn),
    lambda inp, s: G.press_events(s, 1, "P9a", inp, 600.0),
    lambda inp, s: G.arrivals(s, 1, "z9", "person", inp, 600.0),
], ids=["world_p_running", "day_states", "staffing", "press_events", "arrivals"])
def test_public_draw_helpers_refuse_test_seeds(inp, monkeypatch, call):
    """PLAN §5: test-world truth only inside the sealed run; every public draw helper checks the seed itself."""
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    with pytest.raises(PermissionError):
        call(inp, 5000)
    call(inp, 999)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    call(inp, 5000)


# z14 layout states with uneven unit counts: under a uniform pick inside the day's state, unit "A0" (the only unit of
# the most likely bright state) would get 60 % of the bright arrivals instead of 1/4; state E has no unit at all
SHARE_STATES = {"A": (60.0, "bright", 1), "B": (40.0, "bright", 3), "C": (50.0, "dim", 2), "D": (50.0, "dim", 1),
                "E": (30.0, "dim", 0)}
SHARE_SEEDS = range(0, 1000)


@pytest.fixture(scope="module")
def share_inp(inp) -> G.Inputs:
    tables = {"layout_states": {
        "z14": {g: {"zone": "z14", "clip_seconds": h, "lum_class": lum, "framing": "c14_f0"}
                for g, (h, lum, _) in SHARE_STATES.items()},
        "z9": {"S9": {"zone": "z9", "clip_seconds": 10.0, "lum_class": "bright", "framing": "c9_f0"}}}}
    units, segs = [], {}
    for g, (_, lum, n) in SHARE_STATES.items():
        for i in range(n):
            sid = f"seg:{g}{i}"
            segs[sid] = Segment(id=sid, cls="person", source="resampled", clip_id="x", track_id=i,  # type: ignore[arg-type]
                                points=[(0.0, 20.0, 0.0), (4.0, 22.0, 1.0)])
            units.append(G.Unit(unit_id=f"{g}{i}", zone="z14", cls="person", group_id=g, lum_class=lum,
                                unsafe=bool(i % 2), framing="c14_f0", clip_id="x", clip_dur=6.0, unit_t0=0.5,
                                members=[(sid, 0.0)]))
    dyn = inp.dyn.model_copy(deep=True)
    dyn.assumed["zone.z14.walker_rate_per_h"].value = 2.0
    return G.Inputs(site=inp.site, dyn=dyn, tables=tables, lib=G.Library(units=units, segments=segs, sha256="x"),
                    footage_violation=True)


@pytest.fixture(scope="module")
def share_days(share_inp) -> list[tuple[int, list[G.Arrival]]]:
    return [(s, G.arrivals(s, 1, "z14", "person", share_inp, G.DAY_S)) for s in SHARE_SEEDS]


def test_state_rates_average_to_the_zone_rate(share_inp):
    st = share_inp.tables["layout_states"]["z14"]
    h = sum(s["clip_seconds"] for s in st.values())
    mean = sum(G.state_rate_per_h("z14", "person", g, share_inp) * s["clip_seconds"] / h for g, s in st.items())
    assert mean == pytest.approx(2.0)
    assert G.state_rate_per_h("z14", "person", "E", share_inp) == 0.0
    # per unit, every state has the same rate: units per clip-hour / recall is constant across states
    per_unit = {g: G.state_rate_per_h("z14", "person", g, share_inp) / n * st[g]["clip_seconds"]
                for g, (_, _, n) in SHARE_STATES.items() if n}
    assert max(per_unit.values()) == pytest.approx(min(per_unit.values()))


def test_long_run_unit_shares_match_the_library_within_a_lighting_class(share_inp, share_days):
    """Audit fix 2: within a lighting class every train unit is resampled equally often (1/units of that class)."""
    ns = share_inp.dyn.get("site.night_share").value
    counts: dict[str, dict[str, int]] = {"bright": {}, "dim": {}}
    for _, arr in share_days:
        for a in arr:
            lum = "dim" if G.is_night(a.t, ns) else "bright"
            assert a.unit.group_id in G.lighting_states("z14", lum, share_inp.tables)[0]
            counts[lum][a.unit.unit_id] = counts[lum].get(a.unit.unit_id, 0) + 1
    for lum, want_units in (("bright", {"A0", "B0", "B1", "B2"}), ("dim", {"C0", "C1", "D0"})):
        c = counts[lum]
        assert set(c) == want_units
        tot = sum(c.values())
        for u, k in c.items():
            assert abs(k / tot - 1 / len(want_units)) < 0.04, (lum, u, k / tot)


def test_validation_anchors_weight_every_train_unit_equally(share_inp, share_days):
    """The validation selection keeps anchors proportional to arrivals with the train lighting mix, so each library
    unit anchors equally often (the dataset counts each of its tracks once)."""
    keep = G.anchor_probs(share_inp, 5)
    c: dict[str, int] = {}
    for s, arr in share_days:
        picked, _ = G.anchors(s, 1, "z14", "person", arr, keep, share_inp)
        for a in picked:
            c[a.unit.unit_id] = c.get(a.unit.unit_id, 0) + 1
    tot = sum(c.values())
    assert set(c) == {u.unit_id for u in share_inp.lib.units}
    assert abs(tot / len(share_days) - 5.0) < 0.5
    for u, k in c.items():
        assert abs(k / tot - 1 / 7) < 0.03, (u, k / tot)


def test_unsafe_flag_is_dropped_before_the_state(share_inp):
    lib = share_inp.lib
    assert [u.unit_id for u in lib.pool("z14", "person", "A", True)] == ["A0"]       # A has no unsafe unit
    assert {u.unit_id for u in lib.pool("z14", "person", "B", True)} == {"B1"}


def _site_safety(tmp_path, site: Site) -> Path:
    """twin/safety.yaml re-keyed onto the v1 site (it still names the v0 site and presses P9a, P14a): every v1 press
    gets the P9a guard entry."""
    import yaml

    d = yaml.safe_load((REPO / "twin" / "safety.yaml").read_text(encoding="utf-8"))
    entry = next(p for p in d["presses"] if p["press"] == "P9a")
    d["site_id"] = site.site_id
    d["presses"] = [{**entry, "press": p.id} for p in site.presses]
    p = tmp_path / "safety.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return p


def test_engine_and_plant_build_from_dynamics_v1_and_a_daygen_v1_day(inp, tmp_path, monkeypatch):
    """Audit fix 1: the engine reads the shared press.stroke_impulse_s of twin/dynamics.json (v0 contract key)."""
    from twin import plant as PL
    from twin.engine import Engine

    monkeypatch.setenv("INTERCYCLE_TRUTH_LOG", str(tmp_path / "truth_access.log"))
    dyn = load(Dynamics, G.DYNAMICS)
    p = dyn.get("press.stroke_impulse_s")
    assert p.source == "prior" and p.n == 0 and p.sweep.lo <= p.value <= p.sweep.hi
    ds = G.generate_day(5, 1, inputs=inp, duration_s=600.0)
    eng = Engine(ds, inp.site, dyn)
    assert eng is not None
    sp = G.write_day(ds, tmp_path / "w5_d1.day_script.json")
    cfg = {"seconds": 120.0, "sim": "fake",
           "plant": {"site": str(G.SITE), "safety": str(_site_safety(tmp_path, inp.site)), "dynamics": str(G.DYNAMICS),
                     "day_script": str(sp), "run_id": "dyn_v1", "zone": "z9", "world": 0, "world_split": "dev",
                     "condition": "learned", "code": {"head": "test"}, "plant_t0": ds.t0_s,
                     "run_dir": str(tmp_path / "runs" / "dyn_v1"), "allow_short_mission": True}}
    plant = PL.make(cfg)
    assert plant.engine is not None and plant.world_seed == 5
