"""lane freeze-glue: the fixed S7b margin writer, the slice's stub-margin refusal and grid margin, the pin on the learned
STOPPED TTL (user decision "keep as is, disclose"), and the sealed twin-run observer built through broadcast_live.

Synthetic inputs only: no dev world, no sealed or test clip, no test seed."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intercycle.eval import conformal_margin as CM
from intercycle.eval import twin_eval as TE
from intercycle.margin_inflate import SCHEMA, Margin, inflation_spec, load_margin
from intercycle.observer import learn_cycles as LC
from intercycle.types import MachineNow, State
from tests.test_learn_cycles_press import PRESS, press_stream
from tests.test_observer_live import SPEC, T0, margin_file, stream, write_jsonl
from twin import broadcast_live as BL
from twin import observer_live as OL

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("slice_script_glue", REPO / "scripts" / "slice.py")
assert _spec is not None and _spec.loader is not None
SL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SL)

_wspec = importlib.util.spec_from_file_location("write_fixed_margin_glue", REPO / "scripts" / "write_fixed_margin.py")
assert _wspec is not None and _wspec.loader is not None
WFM = importlib.util.module_from_spec(_wspec)
_wspec.loader.exec_module(WFM)

V_MAX = 1.792664                    # PREREG_TWIN section 7: forklift z14 sweep.hi in twin/dynamics.json
SRC = {"description": "synthetic ages (unit test)"}
VSRC = {"file": "unit test"}


def _result(age_p99_target: float, **kw) -> dict:
    """A fixed-margin result whose age_p99 is exactly age_p99_target (every tick has that age)."""
    ages = np.full(200, age_p99_target)
    return CM.fixed_margin_result(ages, age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC, **kw)


# ------------------------------------------------------------------------------------------------ (a) fixed margin

def test_the_prereg_worked_check_age_0_40_gives_k3_and_0_75_m():
    r = _result(0.40)
    assert r["margin_m"] == 0.75 and r["margin_cells"] == 3 and r["method"]["k"] == 3
    assert r["inflation"]["radius2_cells"] == 9            # k^2: a disc of k cells


@pytest.mark.parametrize("age,k", [(0.0, 0), (0.05, 1), (0.1394, 1), (0.1395, 2), (0.5578, 4), (0.5579, 5)])
def test_k_is_the_ceiling_of_cells(age, k):
    assert CM.fixed_margin_cells(age, V_MAX, 0.25) == k


def test_an_exact_integer_number_of_cells_is_not_rounded_up():
    # age * v / cell = 4.0 exactly: the 1e-9 only absorbs float error, so k = 4 and not 5
    assert CM.fixed_margin_cells(0.25 * 4 / 2.0, 2.0, 0.25) == 4
    assert CM.fixed_margin_cells(0.25 * 4 / 2.0 + 1e-6, 2.0, 0.25) == 5


def test_age_p99_is_the_numpy_linear_percentile_of_the_pooled_ticks():
    ages = np.r_[np.full(990, 0.1), np.linspace(0.2, 0.9, 10)]
    r = CM.fixed_margin_result(ages, age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC, n_ticks_no_fast_layer=7)
    assert r["method"]["age_p99_s"] == pytest.approx(float(np.percentile(ages, 99)))
    inp = r["method"]["inputs"]
    assert inp["age_p99"]["n_ticks"] == 1000 and inp["age_p99"]["n_ticks_no_fast_layer"] == 7
    assert r["fast_layer_age_s"]["p99"] == r["method"]["age_p99_s"]


def test_v_max_is_the_largest_sweep_hi_of_the_three_mover_speeds_in_dynamics_json():
    v, src = CM.v_max_from_dynamics()
    assert v == V_MAX
    assert src["file"] == "dynamics.json" and len(src["sha256"]) == 64
    assert src["sweep_hi_mps"] == {"forklift.z14.speed_mps": 1.792664, "person.z14.speed_mps": 1.527327,
                                   "person.z9.speed_mps": 1.011488}


def test_v_max_refuses_a_dynamics_file_without_a_speed_key(tmp_path):
    p = tmp_path / "dyn.json"
    p.write_text(json.dumps({"observed": {"forklift.z14.speed_mps": {"sweep": {"hi": 1.0}}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="person.z14.speed_mps"):
        CM.v_max_from_dynamics(p)


def test_the_written_file_is_the_frozen_schema_the_robot_and_eval_read(tmp_path):
    r = _result(0.40)
    p = CM.write_fixed_margin_file(r, tmp_path / "m7_margin.json")
    body = json.loads(p.read_text(encoding="utf-8"))
    assert body["schema"] == SCHEMA and body["frozen"] is True and body["status"] == "ok"
    m = body["method"]
    assert m["name"] == CM.FIXED_METHOD and m["certified"] is False and "not certified" in m["disclosure"]
    assert m["rule"] == CM.FIXED_RULE
    assert set(m["inputs"]) == {"age_p99", "v_max", "cell_m"}      # each input with its source
    assert m["inputs"]["v_max"]["value"] == 1.792664
    mg = load_margin(p)                                             # the robot side
    assert (mg.margin_m, mg.cell_m) == (0.75, 0.25)
    margin, ep = TE.load_frozen_margin(p)                           # EVAL
    assert margin == Margin(0.75, 0.25) and ep == 720.0
    assert body["inflation"] == inflation_spec(Margin(0.75, 0.25))


def test_a_frozen_margin_file_is_never_overwritten(tmp_path):
    p = CM.write_fixed_margin_file(_result(0.40), tmp_path / "m7_margin.json")
    first = p.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        CM.write_fixed_margin_file(_result(0.10), p)
    with pytest.raises(FileExistsError):
        CM.write_fixed_margin_file(_result(0.10), p, overwrite=True)
    assert p.read_text(encoding="utf-8") == first


def test_margin_off_the_h3_grid_is_refused_not_clipped():
    # cell 0.07 m: k = 3 gives 0.21 m, not a multiple of 0.05 m
    with pytest.raises(ValueError, match="calibration grid"):
        CM.fixed_margin_result(np.full(10, 0.2), age_source=SRC, v_max_mps=1.0, v_max_source=VSRC, cell_m=0.07)


def test_margin_above_the_grid_margin_is_refused_not_cut(tmp_path):
    ages = np.full(50, 0.60)                       # 0.60 * 1.792664 / 0.25 = 4.3 -> k = 5, 1.25 m
    r = CM.fixed_margin_result(ages, age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC)
    assert r["margin_m"] == 1.25
    with pytest.raises(ValueError, match="never cut to fit"):
        CM.fixed_margin_result(ages, age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC, grid_margin_m=1.0)
    assert not (tmp_path / "m7_margin.json").exists()
    ok = CM.fixed_margin_result(np.full(50, 0.5578), age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC,
                                grid_margin_m=1.0)      # PREREG: age_p99 <= 0.5578 s is the largest that fits
    assert ok["margin_m"] == 1.0


def test_margin_above_4_m_is_refused():
    with pytest.raises(ValueError, match="calibration grid"):
        CM.fixed_margin_result(np.full(10, 2.5), age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC)


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.1, np.nan]), np.array([0.1, -0.2]), np.ones((2, 2))])
def test_bad_ages_are_refused(bad):
    with pytest.raises(ValueError):
        CM.fixed_margin_result(bad, age_source=SRC, v_max_mps=V_MAX, v_max_source=VSRC)


def test_a_tampered_result_is_refused_by_the_writer(tmp_path):
    r = _result(0.40)
    for edit in (lambda d: d.update(margin_m=0.5), lambda d: d["method"].update(k=2),
                 lambda d: d["method"].update(certified=True), lambda d: d.update(inflation={}),
                 lambda d: d.update(schema="other"), lambda d: d["unit"].update(episode_s=0)):
        bad = json.loads(json.dumps(r))
        edit(bad)
        with pytest.raises(ValueError):
            CM.write_fixed_margin_file(bad, tmp_path / "x.json")


def test_the_ltt_writer_refuses_a_fixed_result(tmp_path):
    with pytest.raises(ValueError):
        CM.write_margin_file(_result(0.40), tmp_path / "m7_margin.json")


def _bridge_run(d: Path, ages: list) -> Path:
    d.mkdir(parents=True)
    (d / "observer_bridge.jsonl").write_text(
        "".join(json.dumps({"t": float(i), "fast_age_s": a}) + "\n" for i, a in enumerate(ages)), encoding="utf-8")
    return d


def test_the_cli_pools_the_bridge_ages_and_counts_ticks_without_a_fast_layer(tmp_path, capsys):
    a = _bridge_run(tmp_path / "r1", [0.3, 0.3, None, 0.3])
    b = _bridge_run(tmp_path / "r2", [0.3, None, 0.3])
    out = tmp_path / "m7_margin.json"
    assert WFM.main(["--runs", str(a), str(b), "--out", str(out)]) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    src = body["method"]["inputs"]["age_p99"]
    assert src["n_ticks"] == 5 and src["n_ticks_no_fast_layer"] == 2
    assert [r["n_ticks_no_fast_layer"] for r in src["source"]["runs"]] == [1, 1]
    assert body["method"]["k"] == 3 and body["margin_m"] == 0.75      # 0.3 * 1.792664 / 0.25 = 2.15 -> k = 3


def test_the_cli_refuses_a_run_without_a_bridge_log_and_a_sealed_run(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        WFM.main(["--runs", str(tmp_path), "--out", str(tmp_path / "m.json")])
    r = _bridge_run(tmp_path / "r", [0.3])
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(SystemExit):
        WFM.main(["--runs", str(r), "--out", str(tmp_path / "m.json")])
    assert not (tmp_path / "m.json").exists()


# ------------------------------------------------------------------------------------------------ slice: stub + grid

def test_the_stub_margin_is_refused_in_a_sealed_run_and_for_freeze_and_test_worlds(tmp_path, monkeypatch):
    monkeypatch.setattr(SL.C, "TWIN", tmp_path / "no_twin_dir")           # no frozen margin on disk
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    p, src = SL.margin_path(tmp_path)                                        # a dev slice: the stub, as before
    assert "stub" in src and json.loads(p.read_text(encoding="utf-8"))["margin_m"] == 0.5
    p2, _ = SL.margin_path(tmp_path, world=999)
    assert p2.exists()
    with pytest.raises(SL.SliceError, match="STUB margin is refused"):
        SL.margin_path(tmp_path, freeze=True)
    with pytest.raises(SL.SliceError, match="test seed"):
        SL.margin_path(tmp_path, world=1000)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    with pytest.raises(SL.SliceError, match="INTERCYCLE_SEALED_RUN=1"):
        SL.margin_path(tmp_path)
    with pytest.raises(SL.SliceError, match="INTERCYCLE_SEALED_RUN=1"):
        SL.margin_path(tmp_path, world=0)


def test_a_frozen_margin_is_used_in_a_sealed_run_and_for_a_freeze_world(tmp_path, monkeypatch):
    twin = tmp_path / "twin"
    twin.mkdir()
    CM.write_fixed_margin_file(_result(0.40), twin / "m7_margin.json")
    monkeypatch.setattr(SL.C, "TWIN", twin)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    p, src = SL.margin_path(tmp_path, world=1000, freeze=True)
    assert p == twin / "m7_margin.json" and "frozen" in src


class _Poly:
    def __init__(self, a):
        self.value = a


def _fake_inputs():
    zone = SimpleNamespace(id="z14", floor_polygon=_Poly([[0, 0], [20, 0], [20, 8], [0, 8]]))
    site = SimpleNamespace(zones=[zone])
    adapter = SimpleNamespace(zone_ids=["z14"], press_ids=["P1"], rects=[(2.0, 2.0, 4.0, 4.0)])
    compiled = {"z14": SimpleNamespace(scenario=SimpleNamespace(site_id=7, key_id=1))}
    return site, compiled, adapter


@pytest.mark.parametrize("margin_m,grid", [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.25, 1.25), (2.0, 2.0)])
def test_the_grid_margin_is_at_least_the_frozen_margin_and_never_below_1_m(margin_m, grid):
    site, compiled, adapter = _fake_inputs()
    spec = SL.observer_spec(site, compiled, adapter, margin_m)
    assert spec["margin"] == grid
    assert SL.observer_spec(site, compiled, adapter)["margin"] == 1.0         # the default call is unchanged


def test_observer_live_accepts_a_margin_over_1_m_only_on_the_grown_grid(tmp_path):
    site, compiled, adapter = _fake_inputs()
    mf = margin_file(tmp_path, margin_m=1.25)
    with pytest.raises(ValueError, match="does not fit"):
        OL.ObserverLive(SL.observer_spec(site, compiled, adapter), OL.ObserverLiveConfig(margin_path=mf))
    ob = OL.ObserverLive(SL.observer_spec(site, compiled, adapter, 1.25), OL.ObserverLiveConfig(margin_path=mf))
    assert ob.margin.margin_m == 1.25


# ------------------------------------------------------------------------------------------------ (b) STOPPED TTL pin

STOP_BIN_NEED_S = 8.0       # the learner TTL at which the robot's STOP_BIN lookup opens (twin/observer_live.py tag)


def _stopped_ttl(stops_per_h: float) -> float:
    """The TTL of a confident STOPPED nowcast of a learned press model whose stops_per_h is `stops_per_h`."""
    t, y = press_stream(2.0, seed=4, nan_share=0.0, stop=(1600.0, 1700.0))
    win = t < 1590.0
    L = LC.learn(t[win], y[win], consts=PRESS, latency_s=0.0)
    assert L.model.confidence >= LC.MIN_CONF
    model = dataclasses.replace(L.model, stops_per_h=stops_per_h)
    n = LC.nowcast(model, t, y, 1620.0, consts=PRESS, latency_s=0.0)
    assert n.state == State.STOPPED and n.confidence >= LC.MIN_CONF
    return float(n.ttl_s)


def _robot_lookup(ttl_s: float, lag: float):
    """The twin bin the robot reads for a press whose own confident STOPPED has this raw TTL, `lag` s after issue."""
    t, y = press_stream(2.0, seed=4, nan_share=0.0)
    mdl = LC.learn(t, y, consts=PRESS, latency_s=0.0).model
    now = MachineNow(0, State.STOPPED, 0.0, 0.0, ttl_s, 8, 1)
    return OL.robot_bins(mdl, now, 0.0, lag, 2.0)


@pytest.mark.parametrize("stops_per_h", [1.3, 3.0, 5.0, 20.0, 170.0])
def test_a_learned_stopped_ttl_stays_below_the_8_s_the_stop_bin_envelope_needs(stops_per_h):
    """User decision "keep as is, disclose": the learned arm never enters a stopped press. True stop rate ~3-5 / h
    (10 / stops_per_h = 2-3.3 s), the twin's inflated rate 20-170 / h (the 2 s floor). A later change that lifts the
    STOPPED TTL to 8 s or more would silently open the STOP_BIN envelope at those rates."""
    ttl = _stopped_ttl(stops_per_h)
    assert ttl < STOP_BIN_NEED_S, (stops_per_h, ttl)
    assert ttl <= max(10.0 / stops_per_h, 2.0) + 1e-9             # the cap that makes it so, ttl_min 2 s floor
    for lag in (0.3, 1.0):                  # what the robot reads (lag > 0: the 0.2 s Observation delay): the union
        assert _robot_lookup(ttl, lag) == OL.TWIN_BINS       # the union (0, STOP_BIN), never (STOP_BIN,) alone


def test_the_stop_bin_envelope_needs_exactly_8_s_of_ttl():
    """The other side of the pin: with a raw TTL of 8 s the lookup does open (so 8 s is the threshold the test above
    guards), and a TTL just under it (7.99 s, decoded as 6 s) does not."""
    assert _robot_lookup(8.0, 0.3) == (OL.M.STOP_BIN,)
    assert _robot_lookup(7.99, 0.3) == OL.TWIN_BINS


def test_the_stopped_ttl_has_a_hard_ceiling_of_10_s_even_without_a_learned_stop_rate():
    """stops_per_h == 0 (no stop in the learning window) leaves the 10 / stops_per_h cap off: the STOPPED TTL is then
    only min(ttl_s, 10 s). Pinned so that ceiling cannot grow; it is below the 16.5 s a visit needs, so the learned arm
    still never enters a stopped press (DISCLOSED in OL.STOP_ENTRY_TAG)."""
    ttl = _stopped_ttl(0.0)
    assert ttl <= 10.0
    assert 10.0 < 16.5


# ------------------------------------------------------------------------------------------------ (c) sealed observer

@pytest.fixture()
def small_stat_v1_obs(tmp_path):
    obs = stream(minutes=3, nan_share=0.05, seed=2)
    p = tmp_path / "learn" / "obs_v2.jsonl"
    p.parent.mkdir()
    write_jsonl(p, obs)
    return p, len(obs)


def test_sealed_learned_observer_starts_with_measured_press_constants_and_the_stat_v1_threshold(
        tmp_path, monkeypatch, small_stat_v1_obs):
    monkeypatch.setenv(OL.SEALED_ENV, "1")
    p, n_obs = small_stat_v1_obs
    ob, t_now, n = BL.learned_observer(SPEC, margin_file(tmp_path), p, "stat_v1")
    assert n == n_obs and math.isfinite(t_now) and t_now >= T0
    assert OL.is_sealed(ob.cfg)
    thr, src = OL.stat_v1_motion_thr()
    assert ob.cfg.motion_thr == thr and thr != 0.5 and ob.cfg.motion_thr_source == src
    assert not ob.cfg.motion_thr_source.upper().startswith(LC.PLACEHOLDER)
    press = ob.cfg.consts["press"]
    assert press.placeholder == () and "P4" in press.source and press.p_min == OL.press_consts().p_min
    assert ob.tags["sealed"] is True


def test_sealed_learned_observer_fails_loudly_on_the_placeholder_config(tmp_path, monkeypatch, small_stat_v1_obs):
    monkeypatch.setenv(OL.SEALED_ENV, "1")
    p, _ = small_stat_v1_obs
    mf = margin_file(tmp_path)
    # placeholder cycle constants and threshold (the default config)
    monkeypatch.setattr(OL, "config_for_camera", lambda cm, mp, **kw: OL.ObserverLiveConfig(margin_path=mp))
    with pytest.raises(LC.PlaceholderConstantError):
        BL.learned_observer(SPEC, mf, p, "stat_v1")
    # measured press constants but a placeholder motion threshold
    def only_thr_placeholder(cm, mp, **kw):
        return OL.ObserverLiveConfig(margin_path=mp, consts=dict(LC.KINDS, press=OL.press_consts()))
    monkeypatch.setattr(OL, "config_for_camera", only_thr_placeholder)
    with pytest.raises(LC.PlaceholderConstantError, match="motion_thr"):
        BL.learned_observer(SPEC, mf, p, "stat_v1")


def test_learned_observer_refuses_a_camera_model_other_than_stat_v1(tmp_path, small_stat_v1_obs):
    p, _ = small_stat_v1_obs
    for cm in ("camera_v0", ""):
        with pytest.raises(ValueError, match="stat_v1"):
            BL.learned_observer(SPEC, margin_file(tmp_path), p, cm)
