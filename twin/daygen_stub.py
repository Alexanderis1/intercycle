"""G0 stub day generator (PLAN_TWIN §4 slice step 2): a 30-minute day with the site's presses, Poisson walkers per
zone and one forklift, every draw from `twin.rng.keyed_rng(world_seed, day, entity, event)` (W-W1).

All inputs are v0 priors (twin/site_v0.yaml, twin/dynamics_v0.json); paths are synthetic polylines, not resampled
tracks (DAYGEN v1, lane S3a, replaces this module). Output: a `DayScript` (twin/schemas/dayscript.py), which is TRUTH.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from twin.rng import check_world_seed, keyed_rng
from twin.schemas import load
from twin.schemas.dayscript import RNG_SCHEME, DayScript, MoverIntent, PressEvent, Segment
from twin.schemas.dynamics import Dynamics
from twin.schemas.site import Site, point_in_polygon
from twin.truth_guard import open_truth

HERE = Path(__file__).resolve().parent
SITE_V0 = HERE / "site_v0.yaml"
DYNAMICS_V0 = HERE / "dynamics_v0.json"
GENERATOR, VERSION = "twin.daygen_stub", "0.1"
DAY_S = 86_400.0
FORKLIFT_ZONE_INDEX = 0                      # the stub's forklift drives in the first zone (z9, the vehicle road)


def _lognormal(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(median * math.exp(sigma * rng.standard_normal()))


def _stop_sigma(dyn: Dynamics) -> float:
    return math.log(dyn.get("press.stop_p90_s").value / dyn.get("press.stop_median_s").value) / 1.2815515655446004


def _press_events(world_seed: int, day: int, press: str, dyn: Dynamics, duration: float) -> list[PressEvent]:
    rng = keyed_rng(world_seed, day, f"press:{press}", "schedule")
    period = dyn.get(f"press.{press}.stroke_period_s").value
    running = bool(rng.random() < dyn.get(f"press.{press}.p_running").value)
    run_med = dyn.get(f"press.{press}.run_bout_median_s").value
    stop_med, stop_sig = dyn.get("press.stop_median_s").value, _stop_sigma(dyn)
    out, t = [], 0.0
    while t < duration - 1e-9:
        bout = _lognormal(rng, run_med, 0.8) if running else _lognormal(rng, stop_med, stop_sig)
        t1 = min(duration, t + max(bout, 1.0))
        if running:
            p = period * math.exp(0.05 * rng.standard_normal())
            out.append(PressEvent(press=press, kind="running", t0=t, t1=t1, period_s=p, phase_s=float(rng.uniform(0, p)),
                                  cause="production", source="prior"))
        else:
            out.append(PressEvent(press=press, kind="stopped", t0=t, t1=t1, cause="short_stop", source="assumed"))
        t, running = t1, not running
    return out


def _blocked(x: float, z: float, site: Site) -> bool:
    return any(point_in_polygon(x, z, p.footprint.value) for p in site.presses)


def _clear(a: tuple[float, float], b: tuple[float, float], site: Site, step: float = 0.25) -> bool:
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    return not any(_blocked(a[0] + (b[0] - a[0]) * k / (n - 1), a[1] + (b[1] - a[1]) * k / (n - 1), site) for k in range(n))


def _interior(rng: np.random.Generator, poly, site: Site) -> tuple[float, float]:
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    for _ in range(1000):
        x, z = float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(zs), max(zs)))
        if point_in_polygon(x, z, poly) and not _blocked(x, z, site):
            return x, z
    raise RuntimeError("no free interior point in the zone polygon")


def _boundary(rng: np.random.Generator, poly) -> tuple[float, float]:
    """Uniform point on the polygon boundary, pulled 5 cm inside towards the centroid."""
    n = len(poly)
    lengths = np.array([math.hypot(poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]) for i in range(n)])
    i = int(rng.choice(n, p=lengths / lengths.sum()))
    s = float(rng.random())
    (x0, z0), (x1, z1) = poly[i], poly[(i + 1) % n]
    x, z = x0 + s * (x1 - x0), z0 + s * (z1 - z0)
    cx, cz = sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n
    d = math.hypot(cx - x, cz - z) or 1.0
    return x + 0.05 * (cx - x) / d, z + 0.05 * (cz - z) / d


def _polyline(pts: list[tuple[float, float]], speed: float) -> list[tuple[float, float, float]]:
    out, t = [(0.0, pts[0][0], pts[0][1])], 0.0
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        t += max(math.hypot(xb - xa, zb - za), 1e-3) / speed
        out.append((t, xb, zb))
    return out


def _walker_path(rng: np.random.Generator, poly, site: Site) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for _ in range(50):
        pts = [_boundary(rng, poly), _interior(rng, poly, site), _boundary(rng, poly)]
        if all(_clear(a, b, site) for a, b in zip(pts, pts[1:])):
            break                             # stub: after 50 tries the last path is kept even if it crosses a footprint
    return pts


def generate_day(world_seed: int, day: int, *, site: Site | None = None, dynamics: Dynamics | None = None,
                 duration_s: float = 1800.0, entity_order: list[str] | None = None) -> DayScript:
    """One stub day. `entity_order` (press ids and zone ids, any order, any subset) only exists so the tests can
    show that generation order does not change any entity's events."""
    check_world_seed(world_seed)
    if site is None:
        site = load(Site, SITE_V0)
    dyn: Dynamics = dynamics if dynamics is not None else load(Dynamics, DYNAMICS_V0)
    assert site is not None
    press_ids = [p.id for p in site.presses]
    zone_ids = [z.id for z in site.zones]
    order = entity_order or [*press_ids, *zone_ids, "forklifts"]
    staffing = dyn.get("staff.staffing_factor").value
    events: list[PressEvent] = []
    intents: list[MoverIntent] = []
    segments: list[Segment] = []
    for ent in order:
        if ent in press_ids:
            events += _press_events(world_seed, day, ent, dyn, duration_s)
        elif ent in zone_ids:
            zone = next(z for z in site.zones if z.id == ent)
            rate = dyn.get(f"zone.{ent}.walker_rate_per_h").value * staffing
            ra = keyed_rng(world_seed, day, f"walkers:{ent}", "arrivals")
            n = int(ra.poisson(rate * duration_s / 3600.0))
            starts = np.sort(ra.uniform(0.0, duration_s, n))
            sp = dyn.get("person.speed_mps")
            sigma = sp.dist.params.get("sigma", 0.25) if sp.dist else 0.25
            for k, t0 in enumerate(starts):
                ent_id = f"walker:{ent}:{k:04d}"
                rng = keyed_rng(world_seed, day, ent_id, "path")
                pts = _walker_path(rng, zone.floor_polygon.value, site)
                seg = Segment(id=f"seg:{ent_id}", cls="person", source="synthetic_stub",
                              points=_polyline(pts, _lognormal(rng, sp.value, sigma)))
                segments.append(seg)
                intents.append(MoverIntent(entity=ent_id, cls="person", zone=ent, t_start=float(t0), segment=seg.id,
                                           speed_scale=1.0))
        elif ent == "forklifts":
            zone = site.zones[FORKLIFT_ZONE_INDEX]
            for i in range(int(round(dyn.get("forklift.count").value))):
                ent_id = f"forklift:{i}"
                rng = keyed_rng(world_seed, day, ent_id, "path")
                a = _interior(rng, zone.floor_polygon.value, site)
                b = _interior(rng, zone.floor_polygon.value, site)
                v = _lognormal(rng, dyn.get("forklift.speed_mps").value, 0.15)
                leg = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-3) / v
                pts = [a, b] * (int(duration_s / leg) // 2 + 2)
                seg = Segment(id=f"seg:{ent_id}", cls="forklift", source="synthetic_stub", points=_polyline(pts, v))
                segments.append(seg)
                intents.append(MoverIntent(entity=ent_id, cls="forklift", zone=zone.id, t_start=0.0, segment=seg.id,
                                           speed_scale=1.0))
        else:
            raise ValueError(f"unknown entity group {ent!r}")
    events.sort(key=lambda e: (press_ids.index(e.press), e.t0))
    intents.sort(key=lambda m: m.entity)
    segments.sort(key=lambda s: s.id)
    present = [p for p in press_ids if any(e.press == p for e in events)]
    return DayScript(contract="twin.day_script/1", generator=GENERATOR, generator_version=VERSION, rng_scheme=RNG_SCHEME,
                     world_seed=world_seed, day=day, t0_s=(day - 1) * DAY_S, duration_s=duration_s, zones=zone_ids,
                     presses=present, press_events=events, mover_intents=intents, segments=segments)


def write_day(script: DayScript, path: str | Path) -> Path:
    """Write a day script; the file name must contain `day_script` (it is truth)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open_truth(p, "w") as f:
        json.dump(script.model_dump(mode="json"), f, separators=(",", ":"))
    return p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-seed", type=int, required=True)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--out", required=True, help="output path, e.g. out/w0_d1.day_script.json")
    a = ap.parse_args()
    print(write_day(generate_day(a.world_seed, a.day), a.out))
