"""Ground-truth machine cycles. The learner must never import this package (enforced by a test).

Line stations share one takt clock: every cycle the whole line indexes, each station starts its
active phase `offset_s` after the index and stays active for `busy * takt`. Unplanned andon stops
and planned stops (breaks, shift changes) idle the whole line. Off-line cells run their own clock
with their own micro-stops.
"""
from __future__ import annotations

import numpy as np

from intercycle.scenario import Machine, Scenario
from intercycle.types import Window

DAY = 86_400.0


def _in_planned_stop(t: float, stops: list[tuple[float, float, str]]) -> tuple[float, str] | None:
    tod = t % DAY
    base = t - tod
    for a, b, kind in stops:
        if a <= tod < b:
            return base + b, kind
    return None


def line_index_times(scn: Scenario, seed: int, t0: float, t1: float) -> list[tuple[float, float, str]]:
    """Line index events as (index_time, cycle_length, kind of the idle that ends this cycle).

    cycle_length is the time to the next index, including any stop appended after the cycle.
    """
    rng = np.random.default_rng([seed, 1])
    L = scn.line
    stops = scn.line_stops()
    out: list[tuple[float, float, str]] = []
    t = t0
    while t < t1:
        hit = _in_planned_stop(t, stops)
        if hit is not None:  # line is in a planned stop: next index when it ends
            t = hit[0]
            continue
        takt = L.takt_s * (1 + L.jitter * rng.standard_normal()) * (1 + L.drift_per_h * (t - t0) / 3600)
        extra, kind = 0.0, "cycle"
        if rng.random() < L.andon_stops_per_h * L.takt_s / 3600:
            extra, kind = float(rng.uniform(*L.andon_stop_s)), "andon"
        nxt = t + takt + extra
        hit = _in_planned_stop(nxt, stops)
        if hit is not None:
            nxt, kind = hit[0], hit[1]
        out.append((t, nxt - t, kind))
        t = nxt
    return out


def windows(scn: Scenario, machine_id: str, seed: int, t0: float = 0.0, t1: float = DAY) -> list[Window]:
    m = scn.machine(machine_id)
    if m.on_line:
        idx = line_index_times(scn, seed, t0, t1)
        nominal = scn.line.takt_s
        out = []
        for i, (ti, length, kind) in enumerate(idx):
            bs = ti + m.offset_s
            act = m.busy * nominal * (length / nominal if kind == "cycle" else 1.0)
            act = min(act, length - 0.5)
            ie = (idx[i + 1][0] + m.offset_s) if i + 1 < len(idx) else ti + length + m.offset_s
            out.append(Window(bs, bs + act, ie, kind))
        return out
    return _cell_windows(m, seed, t0, t1, scn.line_stops())


def _cell_windows(m: Machine, seed: int, t0: float, t1: float, stops: list[tuple[float, float, str]]) -> list[Window]:
    """Manually loaded cells: own clock and micro-stops; they also stop for breaks and shift changes."""
    rng = np.random.default_rng([seed, 2, sum(map(ord, m.id))])
    T, j = float(m.period_s), float(m.jitter or 0.0)
    stop_p = (m.stops_per_h or 0.0) * T / 3600
    out, t = [], t0 + float(rng.uniform(0, T))
    while t < t1:
        hit = _in_planned_stop(t, stops)
        if hit is not None:
            t = hit[0]
            continue
        b = m.busy * T * (1 + j * rng.standard_normal())
        i = (1 - m.busy) * T * (1 + j * rng.standard_normal())
        kind = "cycle"
        if rng.random() < stop_p:
            i += float(rng.uniform(*(m.stop_s or (5, 30))))
            kind = "andon"
        end = t + b + i
        hit = _in_planned_stop(end, stops)
        if hit is not None:
            end, kind = hit[0], hit[1]
        out.append(Window(t, t + b, end, kind))
        t = end
    return out


def all_windows(scn: Scenario, seed: int, t0: float = 0.0, t1: float = DAY) -> dict[str, list[Window]]:
    return {m.id: windows(scn, m.id, seed, t0, t1) for m in scn.machines}


def state_at(ws: list[Window], t: float) -> tuple[int, Window | None]:
    """Index of the window containing t (bs <= t < ie) and the window, or (-1, None)."""
    lo, hi = 0, len(ws) - 1
    if not ws or t < ws[0].bs:
        return -1, None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ws[mid].bs <= t:
            lo = mid
        else:
            hi = mid - 1
    w = ws[lo]
    return (lo, w) if t < w.ie else (-1, None)


def is_active(ws: list[Window], t: float) -> bool:
    _, w = state_at(ws, t)
    return w is not None and t < w.is_
