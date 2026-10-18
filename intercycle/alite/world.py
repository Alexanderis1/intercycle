"""W-W1 for A-lite: body-shop truth windows from the shared keyed day-script RNG (SIMULATED plant; truth side).

The check (PLAN_TWIN §7 item 1). The press-shop twin draws every random number of a generated day from
`twin.rng.keyed_rng(world_seed, day, entity, event)`. The body-shop `intercycle.sim.world.World` does NOT: it calls
`intercycle.truth.machines.all_windows(scn, seed, t0 - 3600, t1 + 3600)`, whose line stream is
`np.random.default_rng([seed, 1])` restarted at the start of every query window, and whose takt drift is measured from
that start. So the same hour on day 0 and day 6 of one seed is the same random stream (WEAKNESSES W-W1, verified again
in tests/test_alite_world.py::test_body_shop_world_replays_one_stream_per_window). The body-shop World is imported by
the double's plant, so it is not changed; A-lite uses `KeyedWorld` below instead.

KeyedWorld = the body-shop World (same groups, freezes, holds, trips, planned-stop anchors) with its truth windows
built per PRODUCTION DAY from keyed streams:
- production day p runs from the end of the first shift's shift-change stop (06:05) of calendar day p to the start of
  the same stop (06:00) of day p + 1. A cycle that would run past that stop start ends at the next day's 06:05 (the
  line stops for the shift change anyway), so no draw of day p depends on day p - 1 and the windows of any interval are
  the same whatever query window the World is built for. Times before 06:05 of day 0 belong to production day -1.
  Key day = p + 1 (keyed_rng needs day >= 0).
- line: one stream keyed (world_seed, p + 1, "line", "cycles") for takt jitter and andon stops (same model as
  machines.line_index_times); cells: one stream per cell keyed (world_seed, p + 1, cell id, "cycles"), the first
  cycle starts at the day start (the shift start).
- takt drift (W-W1 "bounded mean-reverting"): a day level d_p follows an AR(1) over production days, phi = 0.5, with
  stationary sd = 8 h x drift_per_h (0.8 % for the body shop), innovations keyed (world_seed, p + 1, "line",
  "drift"); within the day the multiplier moves linearly from 1 + d_p to 1 + d_{p+1}. ASSUMED (simulated) values.
- world seeds follow the twin's seed rule (twin.rng.check_world_seed: 0..999 unless the sealed flag is set); A-lite
  uses dev-range seeds only and never runs inside sealed_run.py.
"""
from __future__ import annotations

import math

import numpy as np

from intercycle.scenario import Machine, Scenario, hhmm
from intercycle.sim.world import Group, World
from intercycle.types import Window
from twin.rng import check_world_seed, keyed_rng

DAY = 86_400.0
DRIFT_PHI = 0.5
DRIFT_SD_HOURS = 8.0                 # stationary sd of the day drift level = 8 h x drift_per_h
DRIFT_LAGS = 30                      # AR(1) truncated after 30 days (phi ** 30 ~ 1e-9)


def day_bounds(scn: Scenario) -> tuple[float, float]:
    """(a0, a1) seconds of day: start and end of the first shift's shift-change stop (06:00, 06:05)."""
    a0 = hhmm(scn.shifts[0].start)
    return a0, a0 + scn.shift_change_stop_min * 60.0


def production_day(scn: Scenario, t: float) -> int:
    _, a1 = day_bounds(scn)
    return int(math.floor((t - a1) / DAY))


def _stop_at(t: float, stops) -> tuple[float, str] | None:
    tod = t % DAY
    base = t - tod
    for a, b, kind in stops:
        if a <= tod < b:
            return base + b, kind
    return None


def drift_level(scn: Scenario, world_seed: int, p: int) -> float:
    """Day drift level d_p (fraction of takt), AR(1) over production days, keyed per day."""
    sd = DRIFT_SD_HOURS * scn.line.drift_per_h
    innov = sd * math.sqrt(1.0 - DRIFT_PHI ** 2)
    out = 0.0
    for k in range(DRIFT_LAGS + 1):
        q = p - k
        if q + 1 < 0:
            break
        eps = float(keyed_rng(world_seed, q + 1, "line", "drift").standard_normal())
        out += DRIFT_PHI ** k * innov * eps
    return out


def line_day(scn: Scenario, world_seed: int, p: int) -> list[tuple[float, float, str]]:
    """Line index events (index_time, cycle_length, kind) of production day p (machines.line_index_times model)."""
    L = scn.line
    stops = scn.line_stops()
    a0, a1 = day_bounds(scn)
    start, end, nxt_start = p * DAY + a1, (p + 1) * DAY + a0, (p + 1) * DAY + a1
    rng = keyed_rng(world_seed, p + 1, "line", "cycles")
    d0, d1 = drift_level(scn, world_seed, p), drift_level(scn, world_seed, p + 1)
    out: list[tuple[float, float, str]] = []
    t = start
    while t < end:
        hit = _stop_at(t, stops)
        if hit is not None:
            t = hit[0]
            continue
        dm = 1.0 + d0 + (d1 - d0) * (t - start) / (nxt_start - start)
        takt = L.takt_s * (1 + L.jitter * rng.standard_normal()) * dm
        extra, kind = 0.0, "cycle"
        if rng.random() < L.andon_stops_per_h * L.takt_s / 3600:
            extra, kind = float(rng.uniform(*L.andon_stop_s)), "andon"
        nxt = t + takt + extra
        if nxt >= end:
            nxt, kind = nxt_start, "shift_change"
        else:
            hit = _stop_at(nxt, stops)
            if hit is not None:
                nxt, kind = hit[0], hit[1]
        out.append((t, nxt - t, kind))
        t = nxt
    return out


def cell_day(m: Machine, scn: Scenario, world_seed: int, p: int) -> list[Window]:
    """Cell windows of production day p (machines._cell_windows model, first cycle at the day start)."""
    stops = scn.line_stops()
    a0, a1 = day_bounds(scn)
    start, end, nxt_start = p * DAY + a1, (p + 1) * DAY + a0, (p + 1) * DAY + a1
    rng = keyed_rng(world_seed, p + 1, m.id, "cycles")
    if m.period_s is None:
        raise ValueError(f"cell {m.id} has no period_s")
    T, j = float(m.period_s), float(m.jitter or 0.0)
    stop_p = (m.stops_per_h or 0.0) * T / 3600
    out: list[Window] = []
    t = start
    while t < end:
        hit = _stop_at(t, stops)
        if hit is not None:
            t = hit[0]
            continue
        b = m.busy * T * (1 + j * rng.standard_normal())
        i = (1 - m.busy) * T * (1 + j * rng.standard_normal())
        kind = "cycle"
        if rng.random() < stop_p:
            i += float(rng.uniform(*(m.stop_s or (5, 30))))
            kind = "andon"
        e = t + b + i
        if e >= end:
            e, kind = nxt_start, "shift_change"
            b = min(b, e - t)
        else:
            hit = _stop_at(e, stops)
            if hit is not None:
                e, kind = hit[0], hit[1]
        out.append(Window(t, t + b, e, kind))
        t = e
    return out


def days_covering(scn: Scenario, t0: float, t1: float) -> range:
    return range(production_day(scn, t0), production_day(scn, t1) + 1)


def keyed_windows(scn: Scenario, world_seed: int, t0: float, t1: float) -> dict[str, list[Window]]:
    """Truth windows of every machine over the WHOLE production days that cover [t0, t1] (a superset of the window:
    a World bisects them, so the result does not depend on the query window). Line stations: the
    machines.windows mapping from line index events to station windows (offset, busy, stop handling)."""
    days = days_covering(scn, t0, t1)
    idx = [e for p in days for e in line_day(scn, world_seed, p)]
    nominal = scn.line.takt_s
    out: dict[str, list[Window]] = {}
    for m in scn.machines:
        if m.on_line:
            ws = []
            for i, (ti, length, kind) in enumerate(idx):
                bs = ti + m.offset_s
                act = m.busy * nominal * (length / nominal if kind == "cycle" else 1.0)
                act = min(act, length - 0.5)
                ie = (idx[i + 1][0] + m.offset_s) if i + 1 < len(idx) else ti + length + m.offset_s
                ws.append(Window(bs, bs + act, ie, kind))
            out[m.id] = ws
        else:
            out[m.id] = [w for p in days for w in cell_day(m, scn, world_seed, p)]
    return out


class KeyedWorld(World):
    """The body-shop World on keyed per-day truth windows (module docstring). Same interface and robot effects."""

    def __init__(self, scn: Scenario, world_seed: int, t0: float, t1: float):
        check_world_seed(world_seed)
        if t0 - 3600 < -DAY:
            raise ValueError("KeyedWorld covers production days >= -1 (t0 >= -86400 + 3600)")
        self.scn = scn
        self.seed = int(world_seed)
        self.W = keyed_windows(scn, world_seed, t0 - 3600, t1 + 3600)
        self.groups: dict[str, Group] = {}
        self.group_of: dict[str, str] = {}
        days = range(int((t0 - 3600) // DAY), int((t1 + 3600) // DAY) + 1)
        anchors = sorted(d * DAY + a for d in days for a, _, _ in scn.line_stops())
        for m in scn.machines:
            g = "line" if m.on_line else m.id
            self.group_of[m.id] = g
            self.groups.setdefault(g, Group(g, anchors=list(anchors)))
        self.events: list[dict] = []


def takt_series(world: World, mid: str) -> np.ndarray:
    """Cycle lengths (ie - bs) of normal cycles of one machine (diagnostics, tests)."""
    return np.array([w.ie - w.bs for w in world.W[mid] if w.kind == "cycle"])
