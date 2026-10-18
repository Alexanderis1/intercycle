"""Off-line machines on the robot side (S6 audit #4 patch, lead decision S6 audit #3 option B), pure logic, no ROS.

* masks.snapshot: a STOPPED votes for a line stop (ANDON) only at confidence >= 4 and only for an on-line machine; a site
  with no on-line machine (the twin) is always RUNNING;
* mapping.MapView.predicted_bins(on_line=False): a stopped / unknown phase never gets the STOP bin from a line state;
* masks.machine_bins: the twin presses (Site.own_stop) get their STOP-bin envelope only from their OWN confident STOPPED
  while it stays inside its TTL over the whole keep-out horizon; UNKNOWN / missing / low confidence / a stop expiring
  inside the horizon -> the union of all bins (fail closed); body shop unchanged (its ANDON vote: see
  test_body_shop_andon_vote_on_its_own_site).

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test/test_twin_presses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))

from intercycle.codec.layout import encode_fast, encode_slow  # noqa: E402
from intercycle.codec.sign import demo_key, sign  # noqa: E402
from intercycle.codec.trust import key_id_of  # noqa: E402
from intercycle.observer import mapping as M  # noqa: E402
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State  # noqa: E402

from intercycle_nav import feed as F  # noqa: E402
from intercycle_nav import masks as K  # noqa: E402

SEED = 7
T0 = 6 * 3600.0
TWIN_Z9 = REPO / "twin" / "z9" / "scenario.yaml"


class _View(M.MapView):
    """The part of a map the bin logic reads: per-machine period and jitter."""

    def __init__(self, n: int, period: float = 2.0):
        self.spec = {"machines": [{"id": f"m{i}"} for i in range(n)]}
        self.period = np.full(n, period)
        self.jitter = np.full(n, 0.01)


def _site(ids, on_line, own_stop=None) -> K.Site:
    rects = {m: (4.0 * i, 0.0, 4.0 * i + 2.0, 2.0) for i, m in enumerate(ids)}
    return K.Site(list(ids), rects, [], 1, SEED, (-2.0, -2.0, 4.0 * len(ids) + 2.0, 4.0),
                  on_line=dict(zip(ids, on_line)), own_stop=dict(zip(ids, own_stop or [False] * len(ids))))


def _state(site: K.Site, nows: list[tuple[State, int]], model_conf: int = 10, ttl_s: float = 10.0,
           t: float = T0) -> K.ObserverState:
    key = demo_key(SEED)
    kid = key_id_of(key.verify_key)
    st = K.ObserverState(site, K.demo_keyring(SEED))
    models = [MachineModel(i, 2.0, 0.4, 0.01, 0.0, 1.0, model_conf) for i in range(len(site.ids))]
    fast = [MachineNow(i, s, 0.0, 0.0, ttl_s, c, 1) for i, (s, c) in enumerate(nows)]
    st.ingest(t, F.BLOB + sign(encode_slow(SlowLayer(site.site_id, kid, DAY0_UNIX + t, [], models)), key))
    st.ingest(t, F.BLOB + sign(encode_fast(FastLayer(site.site_id, kid, DAY0_UNIX + t, 1, fast)), key))
    assert st.stats["ok"] == 2, st.all_stats()
    return st


# ---------------------------------------------------------------- the twin Site
def test_twin_site_has_every_press_off_line_and_own_stop():
    if not TWIN_Z9.exists():
        pytest.skip("twin/z9/scenario.yaml not built")
    site = K.Site.from_scenario(str(TWIN_Z9))
    assert site.ids and all(site.on_line[m] is False for m in site.ids)
    assert all(site.own_stop[m] is True for m in site.ids)
    body = K.Site.from_scenario()
    assert not any(body.own_stop.values())                          # body shop: no machine stops from its own state
    assert sum(body.on_line.values()) >= 2


# ---------------------------------------------------------------- ANDON vote (audit S6 #4 part 1)
def test_all_off_line_site_is_always_running_and_unknown_gets_all_bins():
    site = _site(["A", "B"], [False, False], [True, True])
    st = _state(site, [(State.STOPPED, 8), (State.UNKNOWN, 0)])
    snap = st.snapshot(T0 + 0.1)
    assert snap.prod == M.RUNNING
    p = K.MaskParams()
    view = _View(2)
    b = snap.machines[1]
    assert b.state == "UNKNOWN" and not b.stop_ok
    assert K.machine_bins(view, 1, b, snap.prod, p, site.on_line["B"], site.own_stop["B"]) == M.ALL_BINS


def test_body_shop_line_still_goes_to_andon_on_confident_stops_of_its_on_line_stations():
    ids = ["S1", "S2", "S3", "S4", "C1"]
    site = _site(ids, [True, True, True, True, False])
    st = _state(site, [(State.STOPPED, 8), (State.STOPPED, 8), (State.ACTIVE, 8), (State.ACTIVE, 8), (State.ACTIVE, 8)])
    assert st.snapshot(T0 + 0.1).prod == M.ANDON                     # 2 of 4 on-line stations
    st = _state(site, [(State.STOPPED, 8), (State.ACTIVE, 8), (State.ACTIVE, 8), (State.ACTIVE, 8), (State.STOPPED, 8)])
    assert st.snapshot(T0 + 0.1).prod == M.RUNNING                   # an off-line cell's stop does not vote
    st = _state(site, [(State.STOPPED, 3), (State.STOPPED, 3), (State.ACTIVE, 8), (State.ACTIVE, 8), (State.ACTIVE, 8)])
    assert st.snapshot(T0 + 0.1).prod == M.RUNNING                   # STOPPED below confidence 4 never votes


def test_body_shop_andon_vote_on_its_own_site():
    """Audit S6-optB fix 3: pins the S6 #4 vote change on the body shop's real site (a DELIBERATE effect of the lead's
    patch, disclosed): ANDON needs a confident (>= 4) STOPPED from at least half of the 8 on-line stations (4), where it
    needed a STOPPED of any confidence from half of all 10 machines (5) before. Off-line cells and stops below
    confidence 4 never vote."""
    site = K.Site.from_scenario()
    line = [i for i, m in enumerate(site.ids) if site.on_line[m]]
    off = [i for i, m in enumerate(site.ids) if not site.on_line[m]]
    assert (len(line), len(off)) == (8, 2)

    def prod(stopped: list[int], conf: int = 8) -> int:
        nows = [(State.STOPPED, conf) if i in stopped else (State.ACTIVE, 8) for i in range(len(site.ids))]
        return _state(site, nows).snapshot(T0 + 0.1).prod

    assert prod(line[:4]) == M.ANDON                                  # 4 of 8 on-line (was RUNNING: 4 of 10 < 5)
    assert prod(line[:3]) == M.RUNNING
    assert prod(line[:3] + off) == M.RUNNING                          # 5 of 10 stopped (was ANDON): off-line cells do not vote
    assert prod(line[:4], conf=3) == M.RUNNING                        # low-confidence stops never vote (was: they counted)
    assert prod(line) == M.ANDON


def test_site_without_on_line_ids_keeps_the_body_shop_default():
    site = K.Site(["S1", "S2"], {"S1": (0, 0, 1, 1), "S2": (2, 0, 3, 1)}, [], 1, SEED, (0, 0, 4, 2))
    st = _state(site, [(State.STOPPED, 8), (State.ACTIVE, 8)])
    assert st.snapshot(T0 + 0.1).prod == M.ANDON                     # missing id = on the line


# ---------------------------------------------------------------- predicted_bins (audit S6 #4, S4 patch)
def test_predicted_bins_of_an_off_line_machine_ignore_the_line_state():
    view = _View(3)
    for i in range(3):
        assert view.predicted_bins(i, None, M.ANDON, 0.0, on_line=False) == M.ALL_BINS
        assert view.predicted_bins(i, None, M.ANDON, 0.0) == (M.STOP_BIN,)
        assert view.predicted_bins(i, None, M.BREAK, 0.0, on_line=False) == M.ALL_BINS
        assert view.predicted_bins(i, 0.3, M.RUNNING, 0.0, on_line=False) == view.predicted_bins(i, 0.3, M.RUNNING, 0.0)


# ---------------------------------------------------------------- own confident STOPPED -> STOP bin (option B)
@pytest.mark.parametrize("now,model_conf,stop_bin", [
    ((State.STOPPED, 8), 10, True),        # confident stop: its STOP-bin envelope
    ((State.STOPPED, 3), 10, False),       # low fast confidence
    ((State.STOPPED, 8), 3, False),        # low model confidence
    ((State.UNKNOWN, 0), 10, False),
    ((State.ACTIVE, 8), 10, False),
])
def test_twin_press_gets_its_stop_bin_only_from_its_own_confident_stop(now, model_conf, stop_bin):
    site = _site(["A"], [False], [True])
    st = _state(site, [now], model_conf=model_conf)
    snap = st.snapshot(T0 + 0.1)
    mv = snap.machines[0]
    bins = K.machine_bins(_View(1), 0, mv, snap.prod, K.MaskParams(), site.on_line["A"], site.own_stop["A"])
    if stop_bin:
        assert bins == (M.STOP_BIN,)
    else:
        assert M.STOP_BIN not in bins or bins == M.ALL_BINS
        if now[0] != State.ACTIVE:
            assert bins == M.ALL_BINS


def test_stop_expiring_inside_the_horizon_never_gives_the_stop_bin():
    """Audit S6-optB fix 1: the STOP bin only while the own stop entry stays valid over the WHOLE keep-out horizon
    [t, t + H] (the andon_valid_s pattern); a stop that expires inside it may restart there: the union of all bins."""
    site = _site(["A"], [False], [True])
    p = K.MaskParams()
    assert p.horizon_s == 6.0
    st = _state(site, [(State.STOPPED, 8)], ttl_s=2.0)
    for dt in (0.1, 1.0, 1.9):                                        # stop valid at t, expires at issue + 2 s < t + 6 s
        mv = st.snapshot(T0 + dt).machines[0]
        assert mv.stop_ok and mv.stop_left_s == pytest.approx(2.0 - dt, abs=1e-5)
        assert K.machine_bins(_View(1), 0, mv, M.RUNNING, p, False, True) == M.ALL_BINS
    st = _state(site, [(State.STOPPED, 8)], ttl_s=10.0)
    assert K.machine_bins(_View(1), 0, st.snapshot(T0 + 1.0).machines[0], M.RUNNING, p, False, True) == (M.STOP_BIN,)
    assert K.machine_bins(_View(1), 0, st.snapshot(T0 + 3.9).machines[0], M.RUNNING, p, False, True) == (M.STOP_BIN,)
    assert K.machine_bins(_View(1), 0, st.snapshot(T0 + 4.1).machines[0], M.RUNNING, p, False, True) == M.ALL_BINS
    short = K.MaskParams(horizon_s=1.5)                              # the same rule on a shorter horizon
    st = _state(site, [(State.STOPPED, 8)], ttl_s=2.0)
    assert K.machine_bins(_View(1), 0, st.snapshot(T0 + 0.1).machines[0], M.RUNNING, short, False, True) == (M.STOP_BIN,)
    assert K.machine_bins(_View(1), 0, st.snapshot(T0 + 1.0).machines[0], M.RUNNING, short, False, True) == M.ALL_BINS


def test_expired_stop_falls_back_to_all_bins():
    site = _site(["A"], [False], [True])
    st = _state(site, [(State.STOPPED, 8)], ttl_s=2.0)
    assert st.snapshot(T0 + 1.0).machines[0].stop_ok
    late = st.snapshot(T0 + 3.0).machines[0]
    assert not late.stop_ok
    assert K.machine_bins(_View(1), 0, late, M.RUNNING, K.MaskParams(), False, True) == M.ALL_BINS


def test_own_stop_never_applies_without_the_site_flag_or_on_the_line():
    """Body shop unchanged: an off-line body-shop cell (own_stop False) and an on-line station keep the old rules."""
    site = _site(["C1", "S1"], [False, True])
    st = _state(site, [(State.STOPPED, 8), (State.STOPPED, 8)])
    snap = st.snapshot(T0 + 0.1)
    assert all(m.stop_ok for m in snap.machines)
    p = K.MaskParams()
    view = _View(2)
    assert K.machine_bins(view, 0, snap.machines[0], snap.prod, p, False, False) == M.ALL_BINS
    assert K.machine_bins(view, 1, snap.machines[1], snap.prod, p, True, True) == \
        K.machine_bins(view, 1, snap.machines[1], snap.prod, p)
