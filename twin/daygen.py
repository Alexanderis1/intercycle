"""DAYGEN v1: a generated day by resampling real TRAIN segments (lane S3a, PLAN_TWIN §5 "DYN/DAYGEN").

A day draws, in this order and each from its own keyed stream `twin.rng.keyed_rng(world_seed, day, entity, event)`
(W-W1: an entity's draws never depend on which other entities exist or in which order they are generated):

1. **Lighting** (ASSUMED time of day): night = the `site.night_share` of the 24 h centred on midnight; the share is
   the observed dim-luma share of the train footage, the mapping to hours is assumed (P2: no timing signal).
2. **Layout state** per zone (OBSERVED): one bright and one dim session cluster of the train clips
   (`layout.<zone>.states`, weighted by clip-seconds; a zone without a state of one lighting class uses the other).
   People resampled while it is night come from the dim state, by day from the bright state.
3. **Assumed rates**: staffing factor U(0.8, 1.2) per (world, day) on the walker rates; Poisson arrivals per zone
   and class at `zone.<z>.walker_rate_per_h` / `forklift_rate_per_h`; `person.violation_share` of the person
   arrivals draw from unsafe-behaviour clips.
4. **Resampled segments** (OBSERVED): each arrival replays one resampling unit of the path library whole (a group
   of people that moved together in one train clip, or one forklift pass), at the real speed (speed_scale 1).
5. **Presses**: P(running) per press is drawn once per WORLD from its Beta posterior (key day 0). The day alternates
   run bouts (exponential, mean p/(1-p) x E[stop], so the long-run running share is p) and W-W3 long stops
   (ASSUMED lognormal). A run bout strokes at one period drawn from the press's per-clip train periods (OBSERVED;
   camera-14 presses: the v0 prior, ASSUMED); in-clip pauses on the Kaplan-Meier's observed support split it
   (OBSERVED share, currently 0 for both learned presses: no complete in-clip pause was seen).

The output is a `DayScript` (twin/schemas/dayscript.py), which is TRUTH; this module is a truth producer
(twin/truth_guard.py). Seeds: dev world seeds 0..999 only; any other seed is refused unless INTERCYCLE_SEALED_RUN=1
(`twin.rng.check_world_seed`). The path library lives in $INTERCYCLE_DATA (tracks never enter the repo); its sha256
is checked against twin/dynamics.json before use (fail closed).

DYN validation (PLAN §5): `python -m twin.daygen --validate OUT.json` runs generated days through the validation
camera (`twin.dynfit.PixelCamera`) with the dataset's selection rule and writes the twin side of the statistics
(observed speeds and stroke intervals, no truth); `scripts/dyn_validate.py` compares them with train and dev.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from twin import dynfit as DF
from twin.rng import check_world_seed, keyed_rng
from twin.schemas import load
from twin.schemas.dayscript import RNG_SCHEME, DayScript, MoverIntent, PressEvent, Segment
from twin.schemas.dynamics import Dynamics
from twin.schemas.site import Site
from twin.truth_guard import open_truth

HERE = Path(__file__).resolve().parent
SITE = HERE / "site.yaml"
DYNAMICS = HERE / "dynamics.json"
PRESS_TABLES = HERE / "dyn" / "press_v1.json"
VALCAM = HERE / "dyn" / "valcam_v1.json"
GENERATOR, VERSION = "twin.daygen", "1.0"
DAY_S = 86_400.0
CLASSES = ("person", "forklift")
RATE_NAME = {"person": "walker_rate_per_h", "forklift": "forklift_rate_per_h"}
# dev seeds, disjoint from the dev worlds 0-3 (generated after PREREG v1); 64 worlds, because each day draws only one
# bright and one dim layout state per zone, so few worlds give a noisy state mix
VALIDATION_SEEDS = tuple(range(900, 964))


# ------------------------------------------------------------------------------------------------ inputs

@dataclass
class Unit:
    unit_id: str
    zone: str
    cls: str
    group_id: str
    lum_class: str
    unsafe: bool
    framing: str
    clip_id: str
    clip_dur: float
    unit_t0: float
    members: list[tuple[str, float]] = field(default_factory=list)      # (segment id, start offset in the unit)


@dataclass
class Library:
    units: list[Unit]
    segments: dict[str, Segment]
    sha256: str

    def pool(self, zone: str, cls: str, state: str | None, unsafe: bool | None) -> list[Unit]:
        """Units of (zone, cls), narrowed to the layout state and the safe/unsafe flag; a narrowing that leaves
        nothing is dropped (unsafe flag first, then the state)."""
        base = [u for u in self.units if u.zone == zone and u.cls == cls]
        for st, us in ((state, unsafe), (state, None), (None, unsafe), (None, None)):
            p = [u for u in base if (st is None or u.group_id == st) and (us is None or u.unsafe == us)]
            if p:
                return p
        return []


def _lib_ref_sha(dyn: Dynamics) -> str:
    refs = {p.ref for p in [*dyn.observed.values(), *dyn.assumed.values()] if p.ref and "path_library" in p.ref}
    if len(refs) != 1:
        raise ValueError(f"dynamics names {len(refs)} path libraries, want 1: {sorted(refs)}")
    ref = refs.pop()
    return ref.split("sha256:")[1].strip()


def data_file(name: str) -> Path:
    env = os.environ.get("INTERCYCLE_DATA")
    if not env:
        raise RuntimeError("set INTERCYCLE_DATA (the path library lives in $INTERCYCLE_DATA/dyn/)")
    return Path(env) / "dyn" / name


def library_path() -> Path:
    return data_file("path_library_v1.npz")


def load_library(dyn: Dynamics, path: str | Path | None = None) -> Library:
    """Read the path library (npz, numpy only) and check its sha256 against twin/dynamics.json (fail closed)."""
    p = Path(path) if path is not None else library_path()
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    want = _lib_ref_sha(dyn)
    if sha != want:
        raise ValueError(f"path library {p} sha256 {sha} != dynamics.json {want}")
    with np.load(p, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    seg_rows: dict[str, list[int]] = {}
    for i, sid in enumerate(a["seg_id"].tolist()):
        seg_rows.setdefault(str(sid), []).append(i)
    segs: dict[str, Segment] = {}
    seg_off: dict[str, float] = {}
    for sid in sorted(seg_rows):
        idx = np.asarray(seg_rows[sid])
        idx = idx[np.argsort(a["dt"][idx], kind="stable")]
        i0 = int(idx[0])
        segs[sid] = Segment(id=sid, cls=str(a["cls"][i0]), source="resampled",  # type: ignore[arg-type]
                            clip_id=str(a["clip_id"][i0]), track_id=int(a["track_id"][i0]),
                            points=[(float(a["dt"][i]), float(a["x"][i]), float(a["z"][i])) for i in idx])
        seg_off[sid] = float(a["seg_t_off"][i0])
    unit_rows: dict[str, list[int]] = {}
    for i, uid in enumerate(a["unit_id"].tolist()):
        unit_rows.setdefault(str(uid), []).append(i)
    units = []
    for uid in sorted(unit_rows):
        i0 = unit_rows[uid][0]
        mem = sorted({(str(a["seg_id"][i]), seg_off[str(a["seg_id"][i])]) for i in unit_rows[uid]},
                     key=lambda m: (m[1], m[0]))
        units.append(Unit(unit_id=uid, zone=str(a["zone"][i0]), cls=str(a["cls"][i0]), group_id=str(a["group_id"][i0]),
                          lum_class=str(a["lum_class"][i0]), unsafe=bool(a["unsafe"][i0]),
                          framing=str(a["framing"][i0]), clip_id=str(a["clip_id"][i0]),
                          clip_dur=float(a["clip_dur"][i0]), unit_t0=float(a["unit_t0"][i0]), members=mem))
    return Library(units=units, segments=segs, sha256=sha)


@dataclass
class Inputs:
    site: Site
    dyn: Dynamics
    tables: dict
    lib: Library


def load_inputs(site: Site | None = None, dynamics: Dynamics | None = None, tables: dict | None = None,
                library: Library | None = None) -> Inputs:
    s: Site = site if site is not None else load(Site, SITE)
    dyn: Dynamics = dynamics if dynamics is not None else load(Dynamics, DYNAMICS)
    tab: dict = tables if tables is not None else json.loads(PRESS_TABLES.read_text(encoding="utf-8"))
    lib = library if library is not None else load_library(dyn)
    return Inputs(site=s, dyn=dyn, tables=tab, lib=lib)


# ------------------------------------------------------------------------------------------------ draws

def is_night(t_day: float, night_share: float) -> bool:
    """ASSUMED: night = night_share of the 24 h, centred on midnight (t_day in seconds since midnight)."""
    half = 0.5 * night_share * DAY_S
    t = t_day % DAY_S
    return t < half or t >= DAY_S - half


def world_p_running(world_seed: int, press: str, dyn: Dynamics) -> float:
    """P(running) of a press, drawn once per world from its Beta posterior (key day 0)."""
    p = dyn.get(f"press.{press}.p_running")
    if p.dist is None or p.dist.family != "beta":
        return float(p.value)
    rng = keyed_rng(world_seed, 0, f"press:{press}", "p_running")
    return float(rng.beta(p.dist.params["a"], p.dist.params["b"]))


def day_states(world_seed: int, day: int, zone: str, tables: dict) -> dict[str, str]:
    """{"bright": group id, "dim": group id} of a zone for one day (weighted by train clip-seconds)."""
    st = tables["layout_states"][zone]
    rng = keyed_rng(world_seed, day, f"layout:{zone}", "states")
    u = rng.random(2)
    out = {}
    for k, cls in enumerate(("bright", "dim")):
        ids = sorted(g for g, s in st.items() if s["lum_class"] == cls) or sorted(st)
        w = np.array([st[g]["clip_seconds"] for g in ids], float)
        c = np.cumsum(w / w.sum())
        out[cls] = ids[min(int(np.searchsorted(c, u[k], side="right")), len(ids) - 1)]
    return out


def staffing(world_seed: int, day: int, dyn: Dynamics) -> float:
    p = dyn.get("staff.staffing_factor")
    rng = keyed_rng(world_seed, day, "staff", "factor")
    return float(rng.uniform(p.sweep.lo, p.sweep.hi))


def _stop_draw(rng: np.random.Generator, dyn: Dynamics) -> float:
    d = dyn.get("press.stop_median_s").dist
    assert d is not None
    return float(d.params["median"] * math.exp(d.params["sigma"] * rng.standard_normal()))


def _period_draw(rng: np.random.Generator, press: str, dyn: Dynamics, tables: dict) -> float:
    e = tables["presses"].get(press, {})
    if e.get("periods_s"):
        per = e["periods_s"]
        return float(per[int(rng.integers(len(per)))])
    p = dyn.get(f"press.{press}.stroke_period_s")
    sig = p.dist.params.get("sigma", 0.05) if p.dist else 0.05
    return float(p.value * math.exp(sig * rng.standard_normal()))


def press_events(world_seed: int, day: int, press: str, inp: Inputs, duration: float) -> list[PressEvent]:
    dyn, tables = inp.dyn, inp.tables
    rng = keyed_rng(world_seed, day, f"press:{press}", "schedule")
    p = min(max(world_p_running(world_seed, press, dyn), 1e-6), 1 - 1e-6)
    d = dyn.get("press.stop_median_s").dist
    assert d is not None
    e_stop = d.params["median"] * math.exp(0.5 * d.params["sigma"] ** 2)
    mean_run = p / (1.0 - p) * e_stop
    e = tables["presses"].get(press, {})
    learned = bool(e.get("periods_s"))
    src = "observed" if learned else "prior"
    q = float(e.get("pause_share", 0.0) or 0.0)
    km_t, km_s = np.asarray(e.get("km_t", []), float), np.asarray(e.get("km_s", []), float)
    out: list[PressEvent] = []
    t = 0.0
    running = bool(rng.random() < p)
    while t < duration - 1e-9:
        if not running:
            t1 = min(duration, t + max(_stop_draw(rng, dyn), 1.0))
            out.append(PressEvent(press=press, kind="stopped", t0=t, t1=t1, cause="long_stop", source="assumed"))
            t, running = t1, True
            continue
        bout_end = min(duration, t + max(float(rng.exponential(mean_run)), 1.0))
        period = _period_draw(rng, press, dyn, tables)
        phase = float(rng.uniform(0.0, period))
        while t < bout_end - 1e-9:
            n = int(rng.geometric(q)) if q > 0 else 10 ** 9          # strokes until an in-clip pause
            first = t + phase
            seg_end = min(bout_end, first + (n - 1) * period + period) if n < 10 ** 9 else bout_end
            if seg_end <= t + 1e-9:
                seg_end = min(bout_end, t + period)
            out.append(PressEvent(press=press, kind="running", t0=t, t1=seg_end, period_s=period, phase_s=phase,
                                  cause="production", source=src))
            t = seg_end
            if t >= bout_end - 1e-9:
                break
            gap = DF.km_sample_tail(rng, km_t, km_s, float(e.get("pause_lo_s", 0.0)))
            t1 = min(bout_end, t + max(gap - period, 0.05))
            out.append(PressEvent(press=press, kind="stopped", t0=t, t1=t1, cause="short_stop", source="observed"))
            t, phase = t1, 0.0
        running = False
    return out


@dataclass
class Arrival:
    zone: str
    cls: str
    k: int
    t: float
    unit: Unit


def arrivals(world_seed: int, day: int, zone: str, cls: str, inp: Inputs, duration: float,
             day_t0: float = 0.0) -> list[Arrival]:
    """Poisson arrivals of resampling units of (zone, cls); unit draw per arrival from its own key."""
    name = f"zone.{zone}.{RATE_NAME[cls]}"
    if name not in inp.dyn.assumed:
        return []
    rate = inp.dyn.get(name).value * (staffing(world_seed, day, inp.dyn) if cls == "person" else 1.0)
    if rate <= 0 or not inp.lib.pool(zone, cls, None, None):
        return []
    ra = keyed_rng(world_seed, day, f"arrivals:{zone}:{cls}", "times")
    n = int(ra.poisson(rate * duration / 3600.0))
    times = np.sort(ra.uniform(0.0, duration, n))
    states = day_states(world_seed, day, zone, inp.tables)
    ns = inp.dyn.get("site.night_share").value
    vshare = inp.dyn.get("person.violation_share").value
    out = []
    for k, ta in enumerate(times):
        rng = keyed_rng(world_seed, day, f"{cls}:{zone}:{k:06d}", "unit")
        u_v, u_pick = rng.random(), rng.random()
        state = states["dim" if is_night(day_t0 + float(ta), ns) else "bright"]
        unsafe = bool(u_v < vshare) if cls == "person" else None
        pool = inp.lib.pool(zone, cls, state, unsafe)
        out.append(Arrival(zone=zone, cls=cls, k=k, t=float(ta), unit=pool[min(int(u_pick * len(pool)), len(pool) - 1)]))
    return out


# ------------------------------------------------------------------------------------------------ the day

def _generate(world_seed: int, day: int, inp: Inputs, duration_s: float,
              entity_order: list[str] | None) -> tuple[DayScript, list[Arrival]]:
    check_world_seed(world_seed)
    if day < 1:
        raise ValueError("days start at 1 (day 0 keys the per-world draws)")
    site = inp.site
    press_ids = [p.id for p in site.presses]
    zone_ids = [z.id for z in site.zones]
    groups = [f"{c}s:{z}" for z in zone_ids for c in CLASSES]
    order = entity_order or [*press_ids, *groups]
    events: list[PressEvent] = []
    arr: list[Arrival] = []
    for ent in order:
        if ent in press_ids:
            events += press_events(world_seed, day, ent, inp, duration_s)
        elif ent in groups:
            cls, zone = ent.split("s:", 1)
            arr += arrivals(world_seed, day, zone, cls, inp, duration_s)
        else:
            raise ValueError(f"unknown entity group {ent!r}")
    intents: list[MoverIntent] = []
    used: dict[str, Segment] = {}
    for a in arr:
        for j, (sid, off) in enumerate(a.unit.members):
            ts = a.t + off
            if ts >= duration_s:
                continue
            used[sid] = inp.lib.segments[sid]
            intents.append(MoverIntent(entity=f"{a.cls}:{a.zone}:{a.k:06d}:{j}", cls=a.cls, zone=a.zone,  # type: ignore[arg-type]
                                       t_start=float(ts), segment=sid, speed_scale=1.0))
    events.sort(key=lambda e: (press_ids.index(e.press), e.t0))
    intents.sort(key=lambda m: m.entity)
    present = [p for p in press_ids if any(e.press == p for e in events)]
    ds = DayScript(contract="twin.day_script/1", generator=GENERATOR, generator_version=VERSION, rng_scheme=RNG_SCHEME,
                   world_seed=world_seed, day=day, t0_s=(day - 1) * DAY_S, duration_s=duration_s, zones=zone_ids,
                   presses=present, press_events=events, mover_intents=intents,
                   segments=[used[k] for k in sorted(used)])
    return ds, arr


def generate_day(world_seed: int, day: int, *, inputs: Inputs | None = None, duration_s: float = DAY_S,
                 entity_order: list[str] | None = None) -> DayScript:
    """One generated day (see the module docstring). `entity_order` (press ids and "persons:<zone>" /
    "forklifts:<zone>", any order, any subset) exists so the tests can show order independence (W-W1)."""
    return _generate(world_seed, day, inputs or load_inputs(), duration_s, entity_order)[0]


def write_day(script: DayScript, path: str | Path) -> Path:
    """Write a day script; the file name must contain `day_script` (it is truth)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open_truth(p, "w") as f:
        json.dump(script.model_dump(mode="json"), f, separators=(",", ":"))
    return p


# ------------------------------------------------------------------------------------------------ validation (twin side)

def _positions(seg: Segment, t_start: float, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(seg.points, float)
    tt = t_start + pts[:, 0]
    m = (grid >= tt[0]) & (grid <= tt[-1])
    g = grid[m]
    return g, np.interp(g, tt, pts[:, 1]), np.interp(g, tt, pts[:, 2])


def cameras(site: Site, valcam: dict) -> dict[str, DF.PixelCamera]:
    out = {}
    for f in site.framings:
        fv = valcam["framings"][f.id]
        cv = valcam["cameras"][str(f.camera)]
        out[f.id] = DF.PixelCamera(H=np.asarray(f.H_img_to_floor.value, float), k1=float(f.k1.value),
                                   sigma_u=fv["sigma_u_px"], sigma_v=fv["sigma_v_px"],
                                   p_miss_start=cv["p_miss_start"], p_miss_end=cv["p_miss_end"])
    return out


def validation_sample(press_clips: dict[str, list[tuple[np.ndarray, np.ndarray]]], *, seeds=VALIDATION_SEEDS,
                      day: int = 1, windows_per_day: int = 40, inputs: Inputs | None = None) -> dict:
    """Twin side of the DYN validation (PLAN §5): generated days -> validation camera -> tracks -> statistics.

    Selection rule (the dataset's clips are event-selected short windows): a mover window is anchored at a random
    arrival of the zone and reproduces its source clip (start = arrival - the unit's offset in its clip, length =
    that clip's duration, framing = that clip's framing); every mover of the zone inside the window is filmed and
    the window is kept when it yields at least one observed track row. A press window takes a random TRAIN clip of
    that press (``press_clips``: its frame times and MISSING mask) at a uniform time of the day.
    Returns {"speeds": {"<zone>/<cls>": [...]}, "intervals": {press: [...]}, "n_windows": {...}} (no truth).
    """
    inp = inputs or load_inputs()
    # the dataset's clips are also selected by behaviour class: the validation days draw unsafe-behaviour units at
    # the share the train footage has, not at the ASSUMED violation share of the generated days
    pu = [u for u in inp.lib.units if u.cls == "person"]
    footage_unsafe = float(np.mean([u.unsafe for u in pu])) if pu else 0.0
    dyn = inp.dyn.model_copy(deep=True)
    dyn.assumed["person.violation_share"].value = footage_unsafe
    inp = Inputs(site=inp.site, dyn=dyn, tables=inp.tables, lib=inp.lib)
    valcam = json.loads(VALCAM.read_text(encoding="utf-8"))
    cams = cameras(inp.site, valcam)
    polys = {z.id: z.floor_polygon.value for z in inp.site.zones}
    speeds: dict[str, list[float]] = {}
    ivs: dict[str, list[float]] = {}
    nwin: dict[str, int] = {}
    for ws in seeds:
        ds, arr = _generate(int(ws), day, inp, DAY_S, None)
        segs = {s.id: s for s in ds.segments}
        by_zone: dict[str, list[MoverIntent]] = {}
        for m in ds.mover_intents:
            by_zone.setdefault(m.zone, []).append(m)
        for zone in polys:
            za = [a for a in arr if a.zone == zone]
            if not za:
                continue
            ints = by_zone.get(zone, [])
            starts = np.array([m.t_start for m in ints])
            ends = np.array([m.t_start + segs[m.segment].duration_s for m in ints])
            rng = keyed_rng(int(ws), day, f"validation:{zone}", "windows")
            for _ in range(windows_per_day):
                a = za[int(rng.integers(len(za)))]
                w0 = a.t - a.unit.unit_t0
                w1 = w0 + a.unit.clip_dur
                cam = cams[a.unit.framing]
                grid = w0 + np.arange(0.0, a.unit.clip_dur, 1.0 / cam.fps)
                rows = 0
                found: list[tuple[str, float]] = []
                for i in np.nonzero((starts < w1) & (ends > w0))[0]:
                    m = ints[i]
                    g, x, z = _positions(segs[m.segment], m.t_start, grid)
                    to, xo, zo = cam.observe(rng, g, x, z)
                    rows += len(to)
                    for pc in DF.observed_pieces(to, xo, zo, polys[zone], m.cls, DF.MIN_STAT_S):
                        found.append((m.cls, DF.track_speed(*pc)))
                if rows == 0:
                    continue
                nwin[zone] = nwin.get(zone, 0) + 1
                for cls, v in found:
                    if math.isfinite(v):
                        speeds.setdefault(f"{zone}/{cls}", []).append(v)
        jit = valcam["presses"]
        for press, clips in press_clips.items():
            if press not in jit or not clips:
                continue
            ev = [e for e in ds.press_events if e.press == press and e.kind == "running"]
            rng = keyed_rng(int(ws), day, f"validation:{press}", "windows")
            for _ in range(windows_per_day):
                tg, miss = clips[int(rng.integers(len(clips)))]
                dur = float(tg[-1])
                w0 = float(rng.uniform(0.0, DAY_S - dur))
                st = []
                for e in ev:
                    if e.t1 <= w0 or e.t0 >= w0 + dur:
                        continue
                    assert e.period_s is not None and e.phase_s is not None
                    k0 = max(0, math.ceil((w0 - e.t0 - e.phase_s) / e.period_s))
                    s = e.t0 + e.phase_s + k0 * e.period_s
                    while s < min(e.t1, w0 + dur):
                        st.append(s - w0)
                        s += e.period_s
                stroke = np.zeros(len(tg), bool)
                if st:
                    s_obs = np.asarray(st) + rng.normal(0.0, jit[press]["stroke_jitter_s"], len(st))
                    idx = np.clip(np.searchsorted(tg, s_obs), 0, len(tg) - 1)
                    idx = idx[(s_obs >= 0) & (s_obs <= dur)]
                    stroke[idx] = True
                    stroke &= ~miss
                comp, _ = DF.stroke_intervals(tg, stroke, miss)
                nwin[press] = nwin.get(press, 0) + 1
                ivs.setdefault(press, []).extend(float(v) for v in comp)
    return {"speeds": speeds, "intervals": ivs, "n_windows": nwin, "seeds": [int(s) for s in seeds], "day": day,
            "windows_per_day": windows_per_day, "library_sha256": inp.lib.sha256,
            "footage_unsafe_share": footage_unsafe}


def train_press_clips(path: str | Path | None = None) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Frame times and MISSING masks of the TRAIN machines tables per press (the press windows' camera mask), from
    $INTERCYCLE_DATA/dyn/press_masks_v1.npz (written by scripts/dyn_fit.py; arrays press, start, t, missing)."""
    p = Path(path) if path is not None else data_file("press_masks_v1.npz")
    with np.load(p, allow_pickle=False) as z:
        press, start, t, miss = z["press"], z["start"], z["t"], z["missing"]
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    bounds = np.r_[start, len(t)]
    for k, pid in enumerate(press.tolist()):
        a, b = int(bounds[k]), int(bounds[k + 1])
        out.setdefault(str(pid), []).append((t[a:b].astype(float), miss[a:b].astype(bool)))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-seed", type=int)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--duration-s", type=float, default=DAY_S)
    ap.add_argument("--out", help="output path of one day, e.g. out/w0_d1.day_script.json")
    ap.add_argument("--validate", help="write the twin-side DYN validation statistics to this json")
    ap.add_argument("--windows-per-day", type=int, default=40)
    a = ap.parse_args()
    if a.validate:
        inp = load_inputs()
        res = validation_sample(train_press_clips(), windows_per_day=a.windows_per_day, inputs=inp)
        Path(a.validate).parent.mkdir(parents=True, exist_ok=True)
        Path(a.validate).write_text(json.dumps(res), encoding="utf-8")
        print({k: {kk: len(vv) for kk, vv in v.items()} for k, v in res.items() if k in ("speeds", "intervals")},
              res["n_windows"])
    else:
        if a.world_seed is None or not a.out:
            ap.error("--world-seed and --out are required to write a day")
        print(write_day(generate_day(a.world_seed, a.day, duration_s=a.duration_s), a.out))
