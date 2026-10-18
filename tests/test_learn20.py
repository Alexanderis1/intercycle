"""scripts/run_double.py learn20 observer phase (docs/LEARN20_AMENDMENT.md): per learned mission, a fresh observer
learns the robot-free plant stream of the SAME world day over [t0 - 1200 s, t0) (scripts/slice.py LEARN_S), the
default on the sealed and dev paths; the 192 sealed run ids and the refusals are unchanged."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

from intercycle.eval import twin_batch as TB

REPO = Path(__file__).resolve().parents[1]


def _rd():
    sys.path.insert(0, str(REPO / "scripts"))
    import run_double as RD  # pyright: ignore[reportMissingImports]  (scripts/ on sys.path)

    return RD


def _ctx(tmp_path):
    RD = _rd()
    ctx = RD.setup(tmp_path)
    RD.SL.BW.build(ctx["paths"]["site"], ctx["paths"]["dynamics"], ctx["paths"]["tasks"], tmp_path / "worlds")
    return RD, ctx


def test_learn_window_is_the_1200_s_before_t0_of_the_same_day():
    RD = _rd()
    assert RD.LEARN_WINDOW_S == RD.SL.LEARN_S == 1200.0 and RD.DEFAULT_OBSERVER_MODE == RD.OBSERVER_LEARN20
    day0 = TB.day_t0(9)
    assert RD.learn_window(day0 + 5000.0, day0) == (day0 + 3800.0, day0 + 5000.0, False)
    assert RD.learn_window(day0 + 1200.0, day0) == (day0, day0 + 1200.0, False)
    # a slot < 1200 s into its day (SLOT_PAD_S = 600) is clipped at the day start: the day before is never used
    assert RD.learn_window(day0 + 700.0, day0) == (day0, day0 + 700.0, True)


def test_learn20_phase_learns_only_the_window_robot_free_and_fresh_per_slot(tmp_path, monkeypatch):
    RD, ctx = _ctx(tmp_path)
    w = TB.World(0, 0, "dev")
    day_s, learn_s = 3600.0, 60.0                         # a short window: the mechanics, not the 1200 s wall
    scripts = {d: RD.generate_day(ctx, w.seed, d, day_s) for d in (1, 2)}
    extras = []
    tick = RD.Engine.tick

    def spy(self, t, extra=()):
        extras.append(list(extra))
        return tick(self, t, extra)

    monkeypatch.setattr(RD.Engine, "tick", spy)
    t_a, t_b = 1800.0, TB.day_t0(2) + 2000.0               # day 1 and day 2 slots
    snaps, st = RD.learn20_phase(ctx, w, [(2, "z9", t_b), (1, "z14", t_a)], day_s, scripts=scripts, learn_s=learn_s)
    assert st["mode"] == "learn20" and set(snaps) == {(1, "z14"), (2, "z9")}
    assert extras and all(e == [] for e in extras)          # robot-free: the engine never gets a robot mover
    for (d, z), t0 in (((1, "z14"), t_a), ((2, "z9"), t_b)):
        win = st["windows"][f"d{d}_{z}"]
        ob = pickle.loads(snaps[(d, z)])
        assert win["day_start"] == TB.day_t0(d)            # the SAME world day's script
        assert (win["t_from"], win["t_to"], win["clipped"]) == (t0 - learn_s, t0, False)
        assert win["t_cap_first"] == t0 - learn_s           # starts exactly at t0 - learn_s ...
        assert t0 - 5.0 < win["t_cap_last"] < t0            # ... and ends just before t0 (the one-latency gap)
        assert ob.t_last == win["t_cap_last"] and ob.n_ticks == win["ticks"]
        assert win["ticks"] <= int(learn_s * 2)             # 2 Hz captures inside [t0 - learn_s, t0) only
    # fresh per slot: the day-2 snapshot equals learning that slot alone (nothing carried from day 1)
    alone, _ = RD.learn20_phase(ctx, w, [(2, "z9", t_b)], day_s, scripts=scripts, learn_s=learn_s)
    a, b = pickle.loads(alone[(2, "z9")]), pickle.loads(snaps[(2, "z9")])
    assert (a.n_ticks, a.t_last) == (b.n_ticks, b.t_last)
    assert alone[(2, "z9")] == snaps[(2, "z9")]
    # a slot closer to its day start than the window is clipped at the day start (never the previous day)
    s2, st2 = RD.learn20_phase(ctx, w, [(2, "z9", TB.day_t0(2) + 30.0)], day_s, scripts=scripts, learn_s=learn_s)
    win = st2["windows"]["d2_z9"]
    assert win["clipped"] is True and win["t_from"] == TB.day_t0(2) and win["t_cap_first"] == TB.day_t0(2)


def test_sealed_path_runs_learn20_and_refuses_days18(monkeypatch, tmp_path):
    RD = _rd()
    seen = {}
    monkeypatch.setattr(RD, "run_batch", lambda out, worlds, **kw: seen.update(worlds=worlds, **kw) or {})
    monkeypatch.setenv(RD.SEALED_ENV, "1")
    RD.run_sealed(tmp_path, [1000 + i for i in range(16)])
    assert seen["observer_mode"] == "learn20" and seen["backend"] == "modal" and len(seen["worlds"]) == 16
    with pytest.raises(RD.DoubleError, match="refused in a sealed run"):
        RD.check_observer_mode("days18")
    RD.check_observer_mode("learn20")
    monkeypatch.delenv(RD.SEALED_ENV)
    RD.check_observer_mode("days18")                       # dev only, explicit
    with pytest.raises(RD.DoubleError, match="unknown"):
        RD.check_observer_mode("days1-10")


def test_refusals_and_run_ids_unchanged(monkeypatch, tmp_path):
    RD = _rd()
    monkeypatch.delenv(RD.SEALED_ENV, raising=False)
    with pytest.raises(RD.DoubleError, match="sealed_run"):
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)])
    assert RD.main(["--sealed", "--out", str(tmp_path)]) == 2
    assert RD.main(["--observer-days18", "--out", str(tmp_path)]) == 2       # still needs --dev
    monkeypatch.setenv(RD.SEALED_ENV, "1")
    with pytest.raises(RD.DoubleError, match="16 test worlds"):
        RD.run_sealed(tmp_path, [1000 + i for i in range(15)])
    with pytest.raises(RD.DoubleError, match="sealed batch runs only"):
        RD.run_sealed(tmp_path, [1000 + i for i in range(16)], backend="local")
    with pytest.raises(RD.DoubleError, match="refused in a sealed run"):
        RD.run_batch(tmp_path, [TB.World(0, 1000, "test")], backend="modal", observer_mode="days18")
    ms = TB.plan([TB.World(i, 1000 + i, "test") for i in range(16)], ["z9", "z14"])
    want = {f"w{i}_d{d}_{z}_{c}" for i in range(16) for d in (9, 10) for z in ("z9", "z14")
            for c in ("static", "learned", "reactive")}
    assert len(ms) == 192 and {m.run_id for m in ms} == want
