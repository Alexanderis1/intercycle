"""Ground-truth hour-of-day traffic per zone (SYNTHETIC design assumptions, consistent with the YAML).

Expected occupancy of a zone = mean number of objects (people, AGVs, tuggers, forklifts, data-hall
technicians) present in it. People follow the shift calendar (overlap and extra walking at the shift
changes, canteen exodus in breaks); AGVs circulate on the south aisle all day; tuggers run the
transport aisle; forklifts work the parts market; technicians are in the data hall mostly 07-19.
The learner must never import this module (it only sees Poisson counts from observer/signals.py).
"""
from __future__ import annotations

import numpy as np

from intercycle.scenario import Scenario, hhmm, load

DAY = 86_400.0
# share of the floor headcount that is in each zone on average (the rest is at machines / elsewhere)
PEOPLE_SHARE = {"Z_AISLE_N": 0.22, "Z_AISLE_S": 0.06, "Z_TRANSPORT": 0.02, "Z_MARKET": 0.08,
                "Z_CLOSURES": 0.24, "Z_DATA_HALL": 0.0, "Z_DOCK": 0.10}
AGV_FLEET = 4               # assumed fleet on the south-aisle loop
AGV_AISLE_S = 80.0          # seconds of each loop spent on the south aisle (96 m at ~1.2 m/s)
TUGGER_DWELL_S = 180.0      # one tugger train is in the transport aisle this long per trip
BREAK_PEOPLE = 0.25         # fraction of people still on the floor during a break
SHIFT_BUMP = 0.6            # extra walking/hand-over after a shift change (relative), centred 10 min past the change
FORKLIFT_DUTY = {"early": 0.70, "late": 0.60, "night": 0.25}
TUGGER_DUTY = {"early": 1.0, "late": 1.0, "night": 0.5}


def _ramp(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _window(tod: np.ndarray, t0: float, t1: float, rise: float, fall: float) -> np.ndarray:
    """Smooth 0..1 box over [t0, t1) with ramps `rise` s before t0 and `fall` s after t1 (wraps midnight)."""
    D = (t1 - t0) % DAY or DAY
    u = ((tod - t0 + DAY / 2) % DAY) - DAY / 2
    return _ramp((u + rise) / rise) * _ramp((D + fall - u) / fall)


def _shift_windows(scn: Scenario, tod: np.ndarray) -> dict[str, np.ndarray]:
    return {s.name: _window(tod, hhmm(s.start), hhmm(s.end), 900.0, 1200.0) for s in scn.shifts}


def staffing(scn: Scenario, tod: np.ndarray) -> np.ndarray:
    """People on the floor: shift headcounts with overlap, shift-change bump, break exodus."""
    tod = np.asarray(tod, float) % DAY
    w = _shift_windows(scn, tod)
    n = sum(scn.traffic["people_per_shift"][k] * v for k, v in w.items())
    bump = sum(np.exp(-0.5 * ((((tod - hhmm(s.start) - 600.0) + DAY / 2) % DAY - DAY / 2) / 900.0) ** 2) for s in scn.shifts)
    brk = np.zeros_like(tod)
    for b in scn.breaks:
        brk = np.maximum(brk, _window(tod, b.t0, b.t1, 120.0, 120.0))
    return n * (1 + SHIFT_BUMP * bump) * (1 - (1 - BREAK_PEOPLE) * brk)


def _current_shift_weight(scn: Scenario, tod: np.ndarray, table: dict[str, float]) -> np.ndarray:
    w = _shift_windows(scn, tod)
    tot = sum(w.values())
    return sum(table[k] * v for k, v in w.items()) / np.maximum(tot, 1e-9)


def expected(scn: Scenario, zone: str, t: np.ndarray | float) -> np.ndarray:
    """Expected occupancy (mean objects present) of `zone` at time(s) t (sim seconds, any day)."""
    tod = np.asarray(t, float) % DAY
    tr = scn.traffic
    lam = staffing(scn, tod) * PEOPLE_SHARE.get(zone, 0.0)
    if zone == "Z_AISLE_S":
        lam = lam + AGV_FLEET * AGV_AISLE_S / float(tr["agv_loop_period_s"])
    elif zone == "Z_TRANSPORT":
        duty = _current_shift_weight(scn, tod, TUGGER_DUTY)
        lam = lam + tr["tuggers_per_h"] * TUGGER_DWELL_S / 3600.0 * duty + 0.10 * tr["forklifts"] * _current_shift_weight(scn, tod, FORKLIFT_DUTY)
    elif zone == "Z_MARKET":
        lam = lam + tr["forklifts"] * 0.8 * _current_shift_weight(scn, tod, FORKLIFT_DUTY)
    elif zone == "Z_DATA_HALL":
        day, night = tr["data_hall_techs"]["day"], tr["data_hall_techs"]["night"]
        lam = lam + night + (day - night) * _window(tod, 7 * 3600.0, 19 * 3600.0, 1200.0, 1200.0)
    return lam


def levels(scn: Scenario | None = None) -> dict[str, list[int]]:
    """Truth hour-of-day levels 0..15 per zone: hourly mean occupancy scaled to the zone's busiest hour."""
    scn = scn or load()
    tod = np.arange(0.0, DAY, 60.0)
    out = {}
    for z in scn.zones:
        hourly = expected(scn, z.id, tod).reshape(24, 60).mean(axis=1)
        out[z.id] = [int(v) for v in np.floor(15 * hourly / max(hourly.max(), 1e-9) + 0.5)]
    return out


def level(zone: str, hour: int, scn: Scenario | None = None) -> int:
    """Reference level 0..15 of `zone` during hour-of-day `hour`."""
    return levels(scn)[zone][int(hour) % 24]
