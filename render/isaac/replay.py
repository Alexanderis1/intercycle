#!/usr/bin/env python
"""Path-traced synthetic replay of our simulation (SYNTHETIC) in Isaac Sim 5.1  (DESIGN SKELETON).

    python replay.py --scenario scenario/bodyshop.yaml --trace trace.parquet --out out/ \
        --shots overview,curtain,ble,pov,datahall,ab_cam --quality fast --fps 30
    python replay.py ... --dry-run                 # no Kit: validates the trace contract, prints the plan
    python replay.py compose --left A.mp4 --right B.mp4 --out ab.mp4   # split screen (ffmpeg only)

STATUS. The pure-python half (contract, tracks, shots, layout, plan, ffmpeg) is written to run
anywhere and `--dry-run` was exercised against a synthetic trace. The Kit half (everything under
`# ---- KIT ----`) is UNTESTED against a live Isaac Sim: it is written against the 5.1 API from
the docs, with each fragile step isolated in one small function so it can be fixed in the smoke
container in minutes. Items marked `# VERIFY` are the ones most likely to need a tweak.

THE ONLY INPUT IS THE TRACE. The render tier never simulates: it places assets from
bodyshop.yaml, then replays trace.parquet. Nothing here feeds back into the sim.

TRACE CONTRACT (trace.parquet), one row per (entity, sample), sorted by t within entity
    t          float  sim seconds since 00:00 of day 0 (docs/BRIEF.md "Time convention")
    entity_id  str    "S10".."S80","HEM1","SEAL1" (== scenario machine ids), "robot_B", "agv_0", "person_12", ...
    cls        str    free text; matched case-insensitively by CLASS_RULES below
                      (machine | amr | agv | tugger | forklift | person | body | ble)
    x, z       float  metres, scenario frame (x along the line, z across; +z is NORTH = closures side)
    yaw        float  radians, atan2(dz, dx) i.e. CCW from +x seen from above; NaN/absent => derived from motion
    state      str|int machine: IDLE/ACTIVE/STOPPED/HELD/TRIPPED (or the State IntEnum 0..3);
                       robot: move/enter/exit/dock/wait/...; anything else is carried through
Rows may be sparse: poses are linearly interpolated, states are held (step) until the next row.
Today's sim writes only the robot as (t, x, z, state) at graph nodes; the lead/sim owner must add
machine rows (state transitions only is fine) and, for the best shots, person/AGV/forklift/body rows.

WORLD FRAME. Isaac stage is Z-up, metres: X = sim x, Y = sim z, Z = up  (so +Y is north). yaw maps to
rotateZ with YAW_SIGN/YAW_OFFSET_DEG at the top of the file - check once against the dashboard.

LICENCES. Everything this script references on the NVIDIA bucket is used by URL at render time and
never copied into the repo. Rendered stills/video only. Credits for the assets actually used are
written to <out>/credits.json -> paste into the video description / README (CC-BY-4.0 needs it).
See docs/RENDER.md for the licence reading and the caveats.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

# =====================================================================================
# 0. constants
# =====================================================================================
YAW_SIGN = +1.0          # VERIFY against dashboard: +1 if sim yaw is CCW-from-+x in the (x, z) plane
YAW_OFFSET_DEG = 0.0
EPS = 1e-9
ARM_STANDOFF_M = 3.2     # arm base distance from the line centre-line (body edge ~0.9 m, KR210/R-2000iC reach ~2.7-3.1 m)

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
R51 = f"{BUCKET}/Isaac/5.1/Isaac"
R60 = f"{BUCKET}/Isaac/6.0/Isaac"          # richer + per-asset CC-BY-4.0 LICENSE files; USD is forward/backward
SR60 = f"{R60}/SimReady/Industrial"        # compatible in practice, VERIFY it opens in 5.1 (else bump to isaacsim 6.x / py3.12)
DC = f"{BUCKET}/DigitalTwin/Assets/Datacenter"


@dataclass(frozen=True)
class AssetSpec:
    """One placeable asset. `urls` are tried in order; `fit` rescales to a real-world size so unit
    mismatches (cm vs m) cannot break the layout; `yaw_off` aligns the asset's forward to +x."""
    key: str
    urls: tuple[str, ...]
    fit_len: Optional[float] = None      # target size along the longest horizontal axis, metres
    fit_h: Optional[float] = None        # or target height
    yaw_off: float = 0.0                 # degrees
    licence: str = "CC-BY-4.0"
    credit: str = "NVIDIA Corporation, Isaac Sim assets"


ASSETS: dict[str, AssetSpec] = {a.key: a for a in [
    # robot arms (Fanuc R-2000iC are the spot-welding workhorses of body shops; CC-BY-4.0 in the 6.0 bucket)
    AssetSpec("arm_weld",   (f"{R60}/Robots/Fanuc/r2000ic_210f/r2000ic_210f.usd", f"{R51}/Robots/Kuka/KR210_L150/kr210_l150.usd"), fit_h=2.7),
    AssetSpec("arm_respot", (f"{R60}/Robots/Fanuc/r2000ic_165f/r2000ic_165f.usd", f"{R51}/Robots/Kuka/KR210_L150/kr210_l150.usd"), fit_h=2.6),
    AssetSpec("arm_kuka",   (f"{R51}/Robots/Kuka/KR210_L150/kr210_l150.usd",), fit_h=2.7, licence="Apache-2.0", credit="KUKA KR210 model, ros-industrial/kuka_experimental (Apache-2.0) via NVIDIA Isaac Sim"),
    # mobile robots / vehicles
    AssetSpec("amr",        (f"{R60}/Robots/NVIDIA/NovaCarter/nova_carter.usd", f"{R51}/Robots/NVIDIA/NovaCarter/nova_carter.usd"), fit_len=0.75),
    AssetSpec("agv",        (f"{R60}/Robots/Idealworks/iwhub/iw_hub_static.usd", f"{R51}/Robots/Idealworks/iwhub/iw_hub_static.usd"), fit_len=1.55),
    AssetSpec("forklift",   (f"{R60}/Props/Forklift/forklift.usd", f"{R51}/Props/Forklift/forklift.usd"), fit_len=3.3, yaw_off=0.0),
    AssetSpec("dolly",      (f"{R51}/Props/Dolly/dolly.usd",), fit_len=1.4, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    # people (static pose + slide, or skelanim walk; see PEOPLE_MODE)
    AssetSpec("person_a",   (f"{R51}/People/Characters/original_male_adult_construction_03/male_adult_construction_03.usd",), fit_h=1.78, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    AssetSpec("person_b",   (f"{R51}/People/Characters/original_female_adult_police_02/female_adult_police_02.usd",), fit_h=1.68, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    AssetSpec("person_c",   (f"{R51}/People/Characters/original_male_adult_medical_01/male_adult_medical_01.usd",), fit_h=1.75, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    # data hall
    AssetSpec("rack",       (f"{DC}/Racks/Rack_42U_A/Rack_42U_A_01.usd",), fit_h=2.0, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    AssetSpec("cable_tray", (f"{DC}/Facilities/Cable_Tray/Cable_Tray_61x299x5cm_A01_01.usd",), licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    # fences / guards (SimReady, per-asset CC-BY-4.0)
    AssetSpec("guard",      (f"{SR60}/Warehouse/Barriers/Machine_Guard_A01/sm_machineGuard_a01_01.usd",)),
    AssetSpec("fence",      (f"{SR60}/Warehouse/Barriers/Barrier_Fence_Metal_A02/sm_barrier_fence_metal_a02_01.usd",)),
    AssetSpec("conveyor",   (f"{R60}/Props/Conveyors/ConveyorBelt_A08.usd", f"{R51}/Props/Conveyors/ConveyorBelt_A08.usd"),),
    AssetSpec("pallet",     (f"{R51}/Props/Pallet/pallet.usd",), fit_len=1.2, licence="NVIDIA Omniverse content terms (no per-asset licence file)"),
    AssetSpec("rack_shelf", (f"{SR60}/Warehouse/Racks/Heavy_Duty_Steel_Shelve_A06/sm_shelve_steel_heavyDuty_a06_01.usd",)),
]}

# Optional drop-in for a better body-in-white (converted CC0/CC-BY model, see docs/RENDER.md): --body-usd path_or_url
PEOPLE_MODE = "slide"   # "slide": static pose translated along the trace (robust); "skelanim": bind stand_walk_loop (VERIFY)
WALK_ANIM = f"{R51}/People/Animations/stand_walk_loop.skelanim.usd"

# class aliasing: first match wins. Add the sim's real strings here.
CLASS_RULES: list[tuple[str, str]] = [
    (r"amr|inspection|hero|visitor", "amr"),
    (r"agv", "agv"),
    (r"tugger|tractor|train", "tugger"),
    (r"fork", "forklift"),
    (r"person|people|worker|human|tech|operator|walker", "person"),
    (r"body|biw|skid|car", "body"),
    (r"ble|packet|adv", "ble"),
    (r"machine|weld|frame|framing|respot|lifter|hem|seal|station|robot_arm|cell", "machine"),
]
STATE_ALIASES: list[tuple[str, str]] = [
    (r"^0$|idle|ready|standby", "IDLE"),
    (r"^1$|active|run|busy|moving|cycle", "ACTIVE"),
    (r"^2$|stop|andon|break|fault", "STOPPED"),
    (r"hold|inhibit", "HELD"),
    (r"trip|protective", "TRIPPED"),
]

# =====================================================================================
# 1. contract: scenario, trace, tracks
# =====================================================================================
def load_scenario(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_cls(s: str) -> str:
    s = str(s).lower()
    for pat, name in CLASS_RULES:
        if re.search(pat, s):
            return name
    return "unknown"


def norm_state(v) -> str:
    s = str(v).strip().lower()
    for pat, name in STATE_ALIASES:
        if re.search(pat, s):
            return name
    return str(v)


def _unwrap_to(prev: float, a: float) -> float:
    d = (a - prev + math.pi) % (2 * math.pi) - math.pi
    return prev + d


@dataclass
class Track:
    """Piecewise-linear pose + step state for one entity."""
    entity_id: str
    cls: str
    t: np.ndarray
    x: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    state: list[str]

    def _i(self, t: float) -> tuple[int, int, float]:
        n = len(self.t)
        if n == 1 or t <= self.t[0]:
            return 0, 0, 0.0
        if t >= self.t[-1]:
            return n - 1, n - 1, 0.0
        j = int(np.searchsorted(self.t, t, side="right"))
        i = j - 1
        dt = self.t[j] - self.t[i]
        return i, j, 0.0 if dt < EPS else float((t - self.t[i]) / dt)

    def pose(self, t: float) -> tuple[float, float, float]:
        i, j, a = self._i(t)
        return (float(self.x[i] + a * (self.x[j] - self.x[i])),
                float(self.z[i] + a * (self.z[j] - self.z[i])),
                float(self.yaw[i] + a * (self.yaw[j] - self.yaw[i])))   # yaw is pre-unwrapped

    def state_at(self, t: float) -> str:
        i, _, _ = self._i(t)
        return self.state[i]

    def active(self, t: float) -> bool:
        return self.t[0] - 1e-6 <= t <= self.t[-1] + 1e-6

    def intervals(self, wanted: str) -> list[tuple[float, float]]:
        """[(t_start, t_end)] spans where the held state == wanted."""
        out, start = [], None
        for k, s in enumerate(self.state):
            if s == wanted and start is None:
                start = float(self.t[k])
            elif s != wanted and start is not None:
                out.append((start, float(self.t[k])))
                start = None
        if start is not None:
            out.append((start, float(self.t[-1])))
        return out

    def speed(self, t: float, h: float = 0.5) -> float:
        x0, z0, _ = self.pose(t - h)
        x1, z1, _ = self.pose(t + h)
        return math.hypot(x1 - x0, z1 - z0) / (2 * h)


def derive_yaw(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Heading from motion; holds the last heading while stationary."""
    yaw = np.zeros(len(x))
    last = 0.0
    for i in range(len(x)):
        j = min(i + 1, len(x) - 1)
        dx, dz = x[j] - x[i], z[j] - z[i]
        if i == len(x) - 1 and i > 0:
            dx, dz = x[i] - x[i - 1], z[i] - z[i - 1]
        if math.hypot(dx, dz) > 0.05:
            last = math.atan2(dz, dx)
        yaw[i] = last
    return yaw


def load_trace(path: str | Path) -> dict[str, Track]:
    import pandas as pd
    df = pd.read_parquet(path)
    cols = set(df.columns)
    need = {"t", "entity_id", "x", "z", "state"}
    if not need <= cols:
        raise SystemExit(f"trace missing columns {sorted(need - cols)}; has {sorted(cols)}")
    if "cls" not in cols:
        df["cls"] = "unknown"
    if "yaw" not in cols:
        df["yaw"] = np.nan
    tracks: dict[str, Track] = {}
    for eid, g in df.sort_values(["entity_id", "t"], kind="stable").groupby("entity_id", sort=False):
        t = g["t"].to_numpy(float)
        x = g["x"].to_numpy(float)
        z = g["z"].to_numpy(float)
        yaw = g["yaw"].to_numpy(float, copy=True)
        if np.isnan(yaw).any():
            yaw = np.where(np.isnan(yaw), derive_yaw(x, z), yaw)
        for k in range(1, len(yaw)):                     # unwrap so linear interpolation never spins the long way
            yaw[k] = _unwrap_to(yaw[k - 1], yaw[k])
        cls = norm_cls(g["cls"].iloc[0])
        states = [norm_state(s) if cls == "machine" else str(s) for s in g["state"].tolist()]
        tracks[str(eid)] = Track(str(eid), cls, t, x, z, yaw, states)
    return tracks


def synth_bodies(scn: dict, tracks: dict[str, Track], transfer_s: float = 5.0, delay_s: float = 0.5) -> dict[str, Track]:
    """FALLBACK ONLY (--synth-bodies): if the sim writes no `body` entities, index car bodies through the
    8 stations from S10's ACTIVE/IDLE edges: bodies advance one station in the `transfer_s` after each S10
    IDLE start. This is a visual guess, not a replay - the real fix is for the sim to emit body rows."""
    line = sorted((m for m in scn["machines"] if m.get("on_line")), key=lambda m: m["x"])
    s10 = tracks.get(line[0]["id"])
    if s10 is None:
        return {}
    idle_starts = [a for a, _ in s10.intervals("IDLE")]
    xs = [m["x"] for m in line]
    n = len(xs)
    out: dict[str, Track] = {}
    for b in range(-n, len(idle_starts)):
        rows = []
        for k in range(len(idle_starts)):
            i = k - b                                   # station index this body occupies after cycle k's transfer
            if i < 0 or i >= n + 1:
                continue
            t_a, t_b = idle_starts[k] + delay_s, idle_starts[k] + delay_s + transfer_s
            x_from = xs[0] - 9.0 if i == 0 else xs[i - 1]
            x_to = xs[i] if i < n else xs[-1] + 12.0
            rows += [(t_a, x_from), (t_b, x_to)]
        if len(rows) >= 2:
            t = np.array([r[0] for r in rows]); x = np.array([r[1] for r in rows])
            z = np.full(len(rows), 24.0)
            out[f"body_{b}"] = Track(f"body_{b}", "body", t, x, z, np.zeros(len(rows)), ["move"] * len(rows))
    return out


# =====================================================================================
# 2. layout (pure python): where every asset goes
# =====================================================================================
@dataclass
class Place:
    key: str                 # ASSETS key or "proc:<name>"
    path: str                # prim path under /World
    x: float
    y: float                 # sim z
    z: float = 0.0           # up
    yaw_deg: float = 0.0
    meta: dict = field(default_factory=dict)


def machine_rects(scn: dict) -> dict[str, dict]:
    out = {}
    for m in scn["machines"]:
        w, d = m["size"]
        out[m["id"]] = dict(m, x0=m["x"] - w / 2, x1=m["x"] + w / 2, z0=m["z"] - d / 2, z1=m["z"] + d / 2)
    return out


def arm_layout(m: dict) -> list[Place]:
    """Welding robots 'on both sides' of the line: 2 per side for weld/respot/framing, a tall
    hemming/sealer arm for the closures cells, nothing for the lifter (procedural scissor)."""
    kind, mid, cx, cz = m["kind"], m["id"], m["x"], m["z"]
    w, d = m["size"]
    out: list[Place] = []
    if kind in ("weld_station", "respot", "framing"):
        key = {"weld_station": "arm_weld", "respot": "arm_respot", "framing": "arm_kuka"}[kind]
        for side, sgn in (("N", +1), ("S", -1)):
            for k, dx in enumerate((-w * 0.22, +w * 0.22)):
                out.append(Place(key, f"/World/Line/{mid}/arm_{side}{k}", cx + dx, cz + sgn * ARM_STANDOFF_M, 0.0,
                                 yaw_deg=-90.0 * sgn, meta=dict(machine=mid, side=side, idx=k)))
    elif kind in ("roller_hemming", "sealer"):
        key = "arm_respot" if kind == "roller_hemming" else "arm_kuka"
        for k, dx in enumerate((-w * 0.25, +w * 0.25)):
            out.append(Place(key, f"/World/Closures/{mid}/arm{k}", cx + dx, m["z"] - 1.0, 0.0, yaw_deg=-90.0,
                             meta=dict(machine=mid, side="N", idx=k)))
    return out


def datahall_layout(scn: dict) -> dict:
    """3 rows x 12 racks. Rows run along z; rows 0/1 face each other across a contained cold aisle,
    row 2 faces the east wall CRAHs. 4 CRAH units on the east wall."""
    dh = scn["data_hall"]
    x0, x1 = dh["x"]
    z0, z1 = dh["z"]
    rows = scn["data_hall_targets"]["rows"]
    n = scn["data_hall_targets"]["racks_per_row"]
    pitch = 0.7                                   # 600 mm rack + gap
    span = n * pitch
    zc = (z0 + z1) / 2
    row_x = [x0 + 6.0, x0 + 9.4, x0 + 15.0][:rows]
    racks = []
    for r in range(rows):
        facing = {0: +1, 1: -1, 2: +1}.get(r, +1)   # +1 faces +x
        for i in range(n):
            racks.append(Place("rack", f"/World/DataHall/rack_r{r}_{i:02d}", row_x[r], zc - span / 2 + (i + 0.5) * pitch,
                               0.0, yaw_deg=0.0 if facing > 0 else 180.0, meta=dict(row=r, idx=i)))
    crah = [Place("proc:crah", f"/World/DataHall/crah_{k}", x1 - 1.2, z0 + (k + 0.5) * (z1 - z0) / scn["data_hall_targets"]["crah_units"], 0.0, yaw_deg=180.0,
                  meta=dict(idx=k)) for k in range(scn["data_hall_targets"]["crah_units"])]
    aisle = dict(x=(row_x[0] + row_x[1]) / 2, z0=zc - span / 2 - 0.3, z1=zc + span / 2 + 0.3, w=row_x[1] - row_x[0] - 1.2, h=2.6)
    return dict(racks=racks, crah=crah, cold_aisle=aisle, door=tuple(scn["data_hall"]["door"]))


# =====================================================================================
# 3. choreography helpers (pure python)
# =====================================================================================
HOME_Q = np.radians([0, -20, 25, 0, -40, 0])          # 6-DOF home (VERIFY joint limits per arm model)
WELD_Q = [np.radians(q) for q in (
    [0, -20, 25, 0, -40, 0], [25, 10, 40, 20, -55, 30], [35, 25, 30, -15, -50, 60], [10, 30, 20, 25, -35, -30],
    [-20, 15, 35, -20, -45, 10], [0, -20, 25, 0, -40, 0])]


def arm_q(phase: float, k: int) -> np.ndarray:
    """Joint target for cycle phase in [0,1]; k varies the pattern per arm. Smoothstep between waypoints."""
    n = len(WELD_Q) - 1
    u = min(max(phase, 0.0), 1.0) * n
    i = min(int(u), n - 1)
    a = u - i
    a = a * a * (3 - 2 * a)
    q0, q1 = WELD_Q[i], WELD_Q[i + 1]
    sign = 1.0 if k % 2 == 0 else -1.0
    q = q0 + a * (q1 - q0)
    q = q.copy()
    q[0] *= sign
    q[3] *= sign
    q[5] *= sign
    return q


def phase_of(active: list[tuple[float, float]], t: float) -> tuple[bool, float]:
    """(is_active, phase 0..1 inside the current ACTIVE span)."""
    starts = [a for a, _ in active]
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and t < active[i][1]:
        a0, a1 = active[i]
        return True, (t - a0) / max(a1 - a0, EPS)
    return False, 0.0


def thermal_lut(k: float) -> tuple[float, float, float]:
    """k in [0,1] -> inferno-ish RGB (piecewise linear)."""
    stops = [(0.0, (0.0, 0.0, 0.02)), (0.25, (0.25, 0.04, 0.45)), (0.5, (0.72, 0.17, 0.33)),
             (0.75, (0.97, 0.52, 0.07)), (1.0, (0.99, 0.95, 0.65))]
    k = min(max(k, 0.0), 1.0)
    for (k0, c0), (k1, c1) in zip(stops, stops[1:]):
        if k <= k1:
            a = (k - k0) / (k1 - k0)
            return tuple(c0[i] + a * (c1[i] - c0[i]) for i in range(3))
    return stops[-1][1]


def rack_temp_c(row: int, idx: int, z_frac: float, t: float) -> float:
    """Synthetic exhaust temperature (deg C) for the thermal pass. LABELLED SYNTHETIC in the video."""
    base = 24.0 + 6.0 * z_frac + (4.0 if row == 2 else 0.0)
    hot = 9.0 * math.exp(-((idx - 7.0) ** 2) / 6.0) if row == 1 else 0.0     # one rack running hot
    return base + hot + 0.6 * math.sin(0.05 * t + idx)


def ble_events(scn: dict, hero: Track, t0: float, t1: float, seed: int = 7) -> list[dict]:
    """Visual-only reconstruction of the BLE adverts the hero hears: every adv_interval_s, each
    broadcaster within range_m emits; loss is Bernoulli(packet_loss), seeded. SYNTHETIC visual."""
    ble = scn["ble"]
    rng = np.random.default_rng(seed)
    ev = []
    step = ble["adv_interval_s"]
    t = math.ceil(t0 / step) * step
    while t < t1:
        rx, rz, _ = hero.pose(t)
        for bi, (bx, bz) in enumerate(ble["broadcasters"]):
            d = math.hypot(bx - rx, bz - rz)
            if d <= ble["range_m"]:
                ev.append(dict(t=t, src=bi, bx=bx, bz=bz, ok=bool(rng.random() >= ble["packet_loss"]), d=d))
        t += step
    return ev


# =====================================================================================
# 4. shots (pure python): camera pose as a function of time
# =====================================================================================
@dataclass
class Cam:
    eye: tuple[float, float, float]
    tgt: tuple[float, float, float]
    focal_mm: float = 18.0


def lerp3(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _local(px, py, yaw, fx, fy, fz, pz=0.0):
    """Point at (forward fx, left fy, up fz) of a pose, in world x,y,z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (px + c * fx - s * fy, py + s * fx + c * fy, pz + fz)


@dataclass
class Shot:
    name: str
    title: str
    cam: Callable[[float, float, float, "Ctx"], Cam]      # (t, t_start, t_end, ctx) -> Cam
    window: Callable[["Ctx"], tuple[float, float]]        # default (t0, t1) from the trace
    thermal: bool = False
    show_ble: bool = False
    caption: str = ""


@dataclass
class Ctx:
    scn: dict
    tracks: dict[str, Track]
    hero: Track
    t_min: float
    t_max: float
    dh: dict = field(default_factory=dict)

    def smooth_hero(self, t: float, lag: float = 0.6):
        pts = [self.hero.pose(t - lag * k / 4) for k in range(5)]
        return (sum(p[0] for p in pts) / 5, sum(p[1] for p in pts) / 5, self.hero.pose(t)[2])


def _first(ctx: Ctx, state: str) -> Optional[float]:
    iv = ctx.hero.intervals(state)
    return iv[0][0] if iv else None


def win_overview(c: Ctx):
    return c.t_min, min(c.t_min + 30.0, c.t_max)


def win_curtain(c: Ctx):
    t = _first(c, "enter")
    return (max(c.t_min, t - 28.0), min(c.t_max, t + 12.0)) if t is not None else (c.t_min, min(c.t_min + 40, c.t_max))


def win_ble(c: Ctx):
    t = _first(c, "move")
    t = c.t_min if t is None else t
    return t, min(t + 20.0, c.t_max)


def win_pov(c: Ctx):
    t = _first(c, "enter")
    return (max(c.t_min, t - 12.0), min(c.t_max, t + 4.0)) if t is not None else win_ble(c)


def win_datahall(c: Ctx):
    x_door = c.scn["data_hall"]["door"][0]
    ts = [float(t) for t, x in zip(c.hero.t, c.hero.x) if x >= x_door - 0.5]
    t = ts[0] if ts else c.t_min
    return max(c.t_min, t - 6.0), min(c.t_max, t + 24.0)


def cam_overview(t, a, b, c: Ctx):
    # slow crane move inside the hall, under the trusses (z=9.5), NW corner -> SE, always looking along the line
    u = smooth((t - a) / max(b - a, EPS))
    return Cam(lerp3((2, 44, 9.5), (62, 3, 8.0), u), lerp3((32, 24, 1.5), (62, 24, 1.5), u), 20.0)


def cam_curtain(t, a, b, c: Ctx):
    px, py, yaw = c.smooth_hero(t)
    return Cam(_local(px, py, yaw, -3.6, 1.9, 1.5), _local(px, py, yaw, 3.0, 0.0, 0.9), 22.0)


def cam_ble(t, a, b, c: Ctx):
    px, py, yaw = c.smooth_hero(t)
    return Cam(_local(px, py, yaw, -7.0, -6.0, 6.5), _local(px, py, yaw, 6.0, 0.0, 3.5), 20.0)


def cam_pov(t, a, b, c: Ctx):
    px, py, yaw = c.smooth_hero(t, 0.2)
    return Cam(_local(px, py, yaw, 0.15, 0.0, 1.55), _local(px, py, yaw, 6.0, 0.0, 1.3), 16.0)


def cam_datahall(t, a, b, c: Ctx):
    """Walk down the hot aisle (between rows 1 and 2) at 1.6 m, looking obliquely at the rack backs."""
    dh = c.scn["data_hall"]
    xs = sorted({p.x for p in c.dh["racks"]})
    ax = (xs[1] + xs[2]) / 2 if len(xs) >= 3 else sum(xs) / len(xs)
    zc = sum(dh["z"]) / 2
    u = smooth((t - a) / max(b - a, EPS))
    y = zc - 8.0 + 13.0 * u
    return Cam((ax, y, 1.6), (ax - 2.4, y + 5.0, 1.2), 16.0)


def cam_ab(t, a, b, c: Ctx):
    # fixed side camera on the hazard-zone boundary the hero is working at; same for A and B renders
    ms = machine_rects(c.scn)
    tgt_id = "S40" if "S40" in ms else next(iter(ms))
    m = ms[tgt_id]
    return Cam((m["x"] + 2.5, m["z0"] - 9.0, 3.2), (m["x"], m["z0"] + 0.5, 1.0), 20.0)


SHOTS: dict[str, Shot] = {s.name: s for s in [
    Shot("overview", "Line overview", cam_overview, win_overview, caption="8-station framing line, synthetic replay"),
    Shot("curtain", "Robot waits at the light curtain", cam_curtain, win_curtain,
         caption="Robot holds at the curtain until the zone goes idle"),
    Shot("ble", "BLE broadcast", cam_ble, win_ble, show_ble=True, caption="Signed BLE adverts (one-way), synthetic visual"),
    Shot("pov", "Robot camera: local check", cam_pov, win_pov, caption="Mast camera confirms no motion before crossing"),
    Shot("datahall", "Data hall thermal pass", cam_datahall, win_datahall, thermal=True, caption="Thermal view, synthetic temperatures"),
    Shot("ab_cam", "A vs B, fixed camera", cam_ab, win_curtain, caption="Same seed, same camera"),
]}


# =====================================================================================
# 5. plan / dry-run
# =====================================================================================
def build_ctx(scn: dict, tracks: dict[str, Track], hero_id: Optional[str], synth: bool = False) -> Ctx:
    amrs = [t for t in tracks.values() if t.cls == "amr"]
    if hero_id:
        hero = tracks[hero_id]
    elif amrs:
        hero = amrs[0]
    else:
        raise SystemExit("no entity with cls matching 'amr' in the trace; pass --hero <entity_id>")
    if synth:
        tracks.update(synth_bodies(scn, tracks))
    t_min = min(float(t.t[0]) for t in tracks.values())
    t_max = max(float(t.t[-1]) for t in tracks.values())
    return Ctx(scn, tracks, hero, t_min, t_max, datahall_layout(scn))


def plan(ctx: Ctx, shots: Iterable[str], fps: int, t0: Optional[float], t1: Optional[float], speed: float) -> list[dict]:
    out = []
    for name in shots:
        s = SHOTS[name]
        a, b = s.window(ctx)
        if t0 is not None:
            a = max(a, t0) if name != "overview" else t0
        if t1 is not None:
            b = min(b, t1) if name != "overview" else t1
        n = max(1, int((b - a) * fps / speed))
        out.append(dict(shot=name, title=s.title, t0=a, t1=b, frames=n, seconds=n / fps, thermal=s.thermal))
    return out


def dry_run(scn: dict, tracks: dict[str, Track], ctx: Ctx, shots: list[str], args) -> None:
    from collections import Counter
    print(f"trace: {len(tracks)} entities, t = [{ctx.t_min:.1f}, {ctx.t_max:.1f}] s, hero = {ctx.hero.entity_id}")
    print("  by class:", dict(Counter(t.cls for t in tracks.values())))
    mids = {m["id"] for m in scn["machines"]}
    have = {k for k, t in tracks.items() if t.cls == "machine"}
    if mids - have:
        print(f"  WARNING machines without trace rows (will render IDLE): {sorted(mids - have)}")
    if not any(t.cls == "body" for t in tracks.values()):
        print("  NOTE no 'body' entities: car bodies will be synthesised from S10 ACTIVE/IDLE edges (use --synth-bodies for a visual guess) or omitted")
    for k in sorted(have)[:3]:
        print(f"  {k}: states {sorted(set(tracks[k].state))}, active spans {len(tracks[k].intervals('ACTIVE'))}")
    print("  hero states:", sorted(set(ctx.hero.state)))
    p = plan(ctx, shots, args.fps, args.t0, args.t1, args.speed)
    for r in p:
        print(f"  shot {r['shot']:<9} t=[{r['t0']:.1f}, {r['t1']:.1f}]  {r['frames']} frames = {r['seconds']:.1f} s @ {args.fps} fps  ({r['title']})")
    n_arm = sum(len(arm_layout(m)) for m in scn["machines"])
    print(f"layout: {n_arm} robot arms, {len(ctx.dh['racks'])} racks, {len(ctx.dh['crah'])} CRAH, {len(scn['ble']['broadcasters'])} BLE broadcasters")
    print(f"  total render frames: {sum(r['frames'] for r in p)}   (L40S ~ 1-3 s/frame 'fast' 1080p, ~8-20 s/frame 'final' path-traced)")


# =====================================================================================
# 6. ffmpeg
# =====================================================================================
def ffmpeg_frames_to_mp4(frames: list[Path], out: Path, fps: int, crf: int = 16) -> None:
    lst = out.with_suffix(".frames.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in frames:
            f.write(f"file '{p.as_posix()}'\nduration {1.0 / fps:.6f}\n")
        f.write(f"file '{frames[-1].as_posix()}'\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-vf", f"fps={fps},format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    lst.unlink(missing_ok=True)


def compose_ab(left: Path, right: Path, out: Path, label_l: str, label_r: str, font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") -> None:
    """Side-by-side split screen with labels. Needs two renders with identical camera/window (shot ab_cam)."""
    def dt(txt):
        return (f"drawtext=fontfile={font}:text='{txt}':x=40:y=h-60:fontsize=42:fontcolor=white:"
                f"box=1:boxcolor=black@0.55:boxborderw=12")
    if Path(font).exists():
        flt = f"[0:v]{dt(label_l)}[l];[1:v]{dt(label_r)}[r];[l][r]hstack=inputs=2,scale=1920:-2,format=yuv420p[v]"
    else:
        print(f"[warn] font {font} not found: split screen without labels (apt install fonts-dejavu-core)")
        flt = "[0:v][1:v]hstack=inputs=2,scale=1920:-2,format=yuv420p[v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right), "-filter_complex", flt,
                    "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-movflags", "+faststart", str(out)], check=True)


# =====================================================================================
# ---- KIT ----   everything below needs a running Isaac Sim 5.1 (SimulationApp)
# =====================================================================================
LIGHT_INTENSITY_RECT = 4500.0      # tune on first render (UsdLux, new-style intensity)
LIGHT_INTENSITY_DOME = 250.0


class KitScene:
    """Owns the stage. All Kit/USD imports are local so the module imports without Kit."""

    def __init__(self, app, scn: dict, ctx: Ctx, args):
        self.app, self.scn, self.ctx, self.args = app, scn, ctx, args
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: F401
        import omni.usd
        self.Gf, self.Sdf, self.Usd, self.UsdGeom, self.UsdLux, self.UsdShade = Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
        try:
            from isaacsim.core.utils.stage import create_new_stage   # VERIFY module path in 5.1
            create_new_stage()
        except Exception:  # noqa: BLE001
            omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.ops: dict[str, tuple] = {}          # path -> (translateOp, rotZOp)
        self.mats: dict[str, object] = {}        # path -> UsdShade shader (animatable)
        self.used_assets: dict[str, AssetSpec] = {}
        self.resolved: dict[str, str] = {}      # asset key -> URL that actually loaded
        self.entity_prims: dict[str, str] = {}
        self.sparks: dict[str, tuple] = {}       # machine -> (light prim, [sphere prim paths])
        self.curtains: dict[str, list[str]] = {}
        self.zone_mats: dict[str, object] = {}
        self.arm_ctl: dict[str, list] = {}       # machine -> [(articulation, k)]
        self.skins: list[tuple] = []             # thermal skin (prim, row, idx, zfrac, material)
        self.real_visuals: list[str] = []        # prims hidden in thermal mode
        self.ble_prims: list[dict] = []
        self.world = None

    # ---------- primitives ----------
    def material(self, path: str, rgb, emissive=None, rough=0.5, metal=0.0, opacity=1.0):
        UsdShade, Sdf = self.UsdShade, self.Sdf
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(*(emissive or (0, 0, 0))))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.mats[path] = sh
        return mat

    def set_emissive(self, mat_path: str, rgb, opacity: Optional[float] = None):
        sh = self.mats[mat_path]
        sh.GetInput("emissiveColor").Set(self.Gf.Vec3f(*rgb))
        if opacity is not None:
            sh.GetInput("opacity").Set(float(opacity))

    def box(self, path, size, pos, mat=None, yaw_deg=0.0, animate=False, hidden_in_thermal=True):
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xf = self.UsdGeom.Xformable(cube)
        t = xf.AddTranslateOp()
        t.Set(self.Gf.Vec3d(*pos))
        r = xf.AddRotateZOp()
        r.Set(float(yaw_deg))
        s = xf.AddScaleOp()
        s.Set(self.Gf.Vec3f(*size))
        if mat is not None:
            self.UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
        if animate:
            self.ops[path] = (t, r)
        if hidden_in_thermal:
            self.real_visuals.append(path)
        return cube

    def cyl(self, path, radius, height, pos, mat=None):
        c = self.UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        self.UsdGeom.Xformable(c).AddTranslateOp().Set(self.Gf.Vec3d(*pos))
        if mat is not None:
            self.UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)
        self.real_visuals.append(path)
        return c

    def sphere_light(self, path, pos, radius=0.12, intensity=0.0, rgb=(1.0, 0.85, 0.6)):
        l = self.UsdLux.SphereLight.Define(self.stage, path)
        l.CreateRadiusAttr(radius)
        l.CreateIntensityAttr(float(intensity))
        l.CreateColorAttr(self.Gf.Vec3f(*rgb))
        self.UsdGeom.Xformable(l).AddTranslateOp().Set(self.Gf.Vec3d(*pos))
        return l

    # ---------- asset placement ----------
    def reference(self, spec: AssetSpec, path: str, x, y, z=0.0, yaw_deg=0.0, animate=False) -> Optional[str]:
        """Parent Xform (animated) -> child 'asset' Xform (reference + fit scale + yaw offset).
        The bbox is measured with the parent still at identity, so fitting is independent of yaw.
        Returns the parent path, or None when no URL resolved (caller draws a procedural proxy)."""
        Gf, UsdGeom = self.Gf, self.UsdGeom
        parent = self.stage.DefinePrim(path, "Xform")
        child = self.stage.DefinePrim(path + "/asset", "Xform")
        box = None
        for url in spec.urls:
            try:
                child.GetReferences().ClearReferences()
                child.GetReferences().AddReference(url)
                bb = UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                b = bb.ComputeWorldBound(child).ComputeAlignedBox()
                if max(b.GetSize()) > 1e-6:
                    box, self.resolved[spec.key] = b, url
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[warn] reference failed {url}: {e!r}")
        if box is None:
            print(f"[warn] {spec.key}: no URL resolved -> procedural proxy at {path}")
            self.stage.RemovePrim(path)
            return None
        size, lo, hi = box.GetSize(), box.GetMin(), box.GetMax()
        sc = 1.0
        if spec.fit_len:
            sc = spec.fit_len / max(size[0], size[1], 1e-6)
        elif spec.fit_h:
            sc = spec.fit_h / max(size[2], 1e-6)
        ctr = Gf.Vec3d((lo[0] + hi[0]) / 2 * sc, (lo[1] + hi[1]) / 2 * sc, lo[2] * sc)      # scaled bbox centre / floor
        a = math.radians(spec.yaw_off)
        rot = Gf.Vec3d(math.cos(a) * ctr[0] - math.sin(a) * ctr[1], math.sin(a) * ctr[0] + math.cos(a) * ctr[1], ctr[2])
        cxf = UsdGeom.Xformable(child)
        cxf.AddTranslateOp().Set(-rot)
        cxf.AddRotateZOp().Set(float(spec.yaw_off))
        cxf.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))
        pxf = UsdGeom.Xformable(parent)
        t = pxf.AddTranslateOp()
        t.Set(Gf.Vec3d(x, y, z))
        r = pxf.AddRotateZOp()
        r.Set(float(yaw_deg))
        self.used_assets[spec.key] = spec
        if animate:
            self.ops[path] = (t, r)
        self.real_visuals.append(path)
        return path

    # ---------- world ----------
    def build(self):
        self._lights_and_shell()
        self._line()
        self._closures_and_market()
        self._data_hall()
        self._traffic()
        self._ble_rig()
        self._hero()
        self._materials_for_thermal()

    def _lights_and_shell(self):
        s = self.scn
        X, Z = s["hall"]["x"][1], s["hall"]["z"][1]
        floor = self.material("/World/Looks/floor", (0.42, 0.42, 0.44), rough=0.35)
        wall = self.material("/World/Looks/wall", (0.80, 0.81, 0.83), rough=0.8)
        steel = self.material("/World/Looks/steel", (0.55, 0.57, 0.6), rough=0.4, metal=0.8)
        self.box("/World/Shell/floor", (X + 26, Z + 2, 0.2), ((X + 24) / 2, Z / 2, -0.1), floor)
        h = 11.0
        for name, size, pos in (("wallN", (X, 0.3, h), (X / 2, Z + 0.15, h / 2)), ("wallS", (X, 0.3, h), (X / 2, -0.15, h / 2)),
                                ("wallW", (0.3, Z, h), (-0.15, Z / 2, h / 2))):
            self.box(f"/World/Shell/{name}", size, pos, wall)
        for i in range(0, int(X) + 1, 12):                     # roof trusses
            self.box(f"/World/Shell/truss{i}", (0.4, Z, 0.6), (i, Z / 2, h - 0.3), steel)
        # lighting: soft dome + rect light grid at truss height
        dome = self.UsdLux.DomeLight.Define(self.stage, "/World/Lights/dome")
        dome.CreateIntensityAttr(LIGHT_INTENSITY_DOME)
        self.ceiling_lights = []
        for i in range(8):
            for j in range(4):
                p = f"/World/Lights/rect_{i}_{j}"
                l = self.UsdLux.RectLight.Define(self.stage, p)
                l.CreateWidthAttr(5.0)
                l.CreateHeightAttr(1.0)
                l.CreateIntensityAttr(LIGHT_INTENSITY_RECT)
                l.CreateColorAttr(self.Gf.Vec3f(1.0, 0.97, 0.92))
                xf = self.UsdGeom.Xformable(l)
                xf.AddTranslateOp().Set(self.Gf.Vec3d(6 + i * 12, 6 + j * 12, h - 0.8))
                self.ceiling_lights.append(p)

    def _line(self):
        s = self.scn
        yellow = self.material("/World/Looks/yellow", (0.9, 0.75, 0.05), rough=0.5)
        grey = self.material("/World/Looks/cell", (0.35, 0.37, 0.4), rough=0.5, metal=0.5)
        # skid conveyor rail along the line
        xs = [m["x"] for m in s["machines"] if m.get("on_line")]
        self.box("/World/Line/rail", (max(xs) - min(xs) + 14, 1.6, 0.35), ((max(xs) + min(xs)) / 2, 24, 0.17), grey)
        for m in s["machines"]:
            mid, kind = m["id"], m["kind"]
            w, d = m["size"]
            cx, cz = m["x"], m["z"]
            # hazard-zone floor decal (colour animated by state)
            zm = self.material(f"/World/Looks/zone_{mid}", (0.1, 0.8, 0.2), emissive=(0.0, 0.5, 0.1), opacity=0.35)
            self.box(f"/World/Zones/{mid}", (w, d, 0.02), (cx, cz, 0.012), zm, hidden_in_thermal=True)
            self.zone_mats[mid] = f"/World/Looks/zone_{mid}"
            # gantry frame
            for sx in (-1, 1):
                for sy in (-1, 1):
                    self.box(f"/World/Line/{mid}/post_{sx}_{sy}", (0.3, 0.3, 5.0), (cx + sx * (w / 2 - 0.4), cz + sy * (d / 2 - 0.4), 2.5), grey)
            self.box(f"/World/Line/{mid}/beam_a", (w - 0.2, 0.3, 0.3), (cx, cz + d / 2 - 0.4, 5.0), grey)
            self.box(f"/World/Line/{mid}/beam_b", (w - 0.2, 0.3, 0.3), (cx, cz - d / 2 + 0.4, 5.0), grey)
            # light curtains on both long sides of the hazard zone (N and S edges)
            lc_mat = self.material(f"/World/Looks/lc_{mid}", (0.1, 0.9, 0.2), emissive=(0.0, 1.2, 0.2), opacity=0.18)
            self.curtains[mid] = []
            for side, sgn in (("N", 1), ("S", -1)):
                zc = cz + sgn * (d / 2 + 0.15)
                for ex in (-1, 1):
                    self.box(f"/World/Curtains/{mid}/{side}_post{ex}", (0.12, 0.12, 1.9), (cx + ex * w / 2, zc, 0.95), yellow)
                self.box(f"/World/Curtains/{mid}/{side}_beam", (w, 0.02, 1.8), (cx, zc, 0.95), lc_mat)
                self.curtains[mid].append(f"/World/Looks/lc_{mid}")
            # welding sparks (light + two tiny emissive spheres), only on stations with arms
            if kind in ("weld_station", "respot", "framing", "roller_hemming", "sealer"):
                lp = f"/World/Sparks/{mid}/light"
                self.sphere_light(lp, (cx, cz, 1.3), radius=0.15, intensity=0.0)
                sm = self.material(f"/World/Looks/spark_{mid}", (1, 0.8, 0.3), emissive=(0, 0, 0))
                sp = []
                for k in range(2):
                    p = f"/World/Sparks/{mid}/s{k}"
                    c = self.UsdGeom.Sphere.Define(self.stage, p)
                    c.CreateRadiusAttr(0.04)
                    self.UsdGeom.Xformable(c).AddTranslateOp().Set(self.Gf.Vec3d(cx + (k - 0.5) * 1.0, cz + (0.3 if k else -0.3), 1.2))
                    self.UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(sm)
                    sp.append(p)
                self.sparks[mid] = (lp, f"/World/Looks/spark_{mid}")
            # arms
            self.arm_ctl[mid] = []
            for pl in arm_layout(m):
                path = self.reference(ASSETS[pl.key], pl.path, pl.x, pl.y, 0.0, pl.yaw_deg)
                if path is None:
                    self.box(pl.path + "_proxy", (0.8, 0.8, 2.2), (pl.x, pl.y, 1.1), grey)
                else:
                    self.arm_ctl[mid].append((pl.path, pl.meta["idx"] + (2 if pl.meta["side"] == "S" else 0)))
            if kind == "unload_lifter":
                self.box(f"/World/Line/{mid}/scissor", (w - 2, d - 3, 0.5), (cx, cz, 0.5), yellow, animate=True)
        # fixed perimeter guards around closures cells (SimReady machine guards), VERIFY fit
        for mid in ("HEM1", "SEAL1"):
            m = next((q for q in s["machines"] if q["id"] == mid), None)
            if m:
                self.reference(ASSETS["guard"], f"/World/Closures/{mid}/guard_n", m["x"], m["z"] + m["size"][1] / 2 + 0.6, 0.0, 0.0)

    def _closures_and_market(self):
        shelf = self.material("/World/Looks/shelf", (0.1, 0.25, 0.6), rough=0.5, metal=0.4)
        box = self.material("/World/Looks/cardboard", (0.55, 0.42, 0.25), rough=0.9)
        for i in range(8):                                       # parts supermarket, south wall
            x = 6 + i * 11
            self.box(f"/World/Market/rack_{i}", (8.0, 1.2, 3.0), (x, 2.5, 1.5), shelf)
            for k in range(4):
                self.box(f"/World/Market/box_{i}_{k}", (1.6, 0.9, 0.9), (x - 3 + k * 2, 2.5, 0.5 + (k % 2) * 1.2), box)

    def _data_hall(self):
        dh = self.ctx.dh
        s = self.scn
        x0, x1 = s["data_hall"]["x"]
        z0, z1 = s["data_hall"]["z"]
        conc = self.material("/World/Looks/dh_floor", (0.55, 0.57, 0.6), rough=0.25)
        glass = self.material("/World/Looks/containment", (0.6, 0.8, 1.0), opacity=0.15, rough=0.05)
        crah_m = self.material("/World/Looks/crah", (0.8, 0.82, 0.85), rough=0.4, metal=0.3)
        self.box("/World/DataHall/floor", (x1 - x0, z1 - z0, 0.05), ((x0 + x1) / 2, (z0 + z1) / 2, 0.025), conc)
        self.box("/World/DataHall/wallE", (0.3, z1 - z0, 5.0), (x1 + 0.15, (z0 + z1) / 2, 2.5), crah_m)
        self.box("/World/DataHall/wallN", (x1 - x0, 0.3, 5.0), ((x0 + x1) / 2, z1 + 0.15, 2.5), crah_m)
        self.box("/World/DataHall/wallS", (x1 - x0, 0.3, 5.0), ((x0 + x1) / 2, z0 - 0.15, 2.5), crah_m)
        self.box("/World/DataHall/roof", (x1 - x0, z1 - z0, 0.3), ((x0 + x1) / 2, (z0 + z1) / 2, 5.0), crah_m)
        for k in range(2):
            l = self.UsdLux.RectLight.Define(self.stage, f"/World/Lights/dh_{k}")
            l.CreateWidthAttr(6.0)
            l.CreateHeightAttr(1.0)
            l.CreateIntensityAttr(LIGHT_INTENSITY_RECT * 0.6)
            self.UsdGeom.Xformable(l).AddTranslateOp().Set(self.Gf.Vec3d((x0 + x1) / 2, (z0 + z1) / 2 + (k - 0.5) * 8, 4.8))
            self.ceiling_lights.append(f"/World/Lights/dh_{k}")
        rack_proxy = self.material("/World/Looks/rack", (0.06, 0.06, 0.07), rough=0.35, metal=0.5)
        for pl in dh["racks"]:
            path = self.reference(ASSETS["rack"], pl.path, pl.x, pl.y, 0.0, pl.yaw_deg)
            if path is None:
                self.box(pl.path, (0.6, 1.0, 2.0), (pl.x, pl.y, 1.0), rack_proxy)
            # thermal skins: 3 stacked slabs per rack, hidden unless thermal mode
            for zi in range(3):
                zf = (zi + 0.5) / 3
                mp = f"/World/Looks/skin_{pl.meta['row']}_{pl.meta['idx']}_{zi}"
                m = self.material(mp, (0, 0, 0), emissive=(0.1, 0.1, 0.1))
                self.box(f"/World/Skins/r{pl.meta['row']}_{pl.meta['idx']}_{zi}", (0.62, 1.05, 0.66), (pl.x, pl.y, 0.2 + zi * 0.68 + 0.33), m,
                         hidden_in_thermal=False)
                self.skins.append((f"/World/Skins/r{pl.meta['row']}_{pl.meta['idx']}_{zi}", pl.meta["row"], pl.meta["idx"], zf, mp))
        a = dh["cold_aisle"]
        self.box("/World/DataHall/containment_roof", (a["w"] + 0.1, a["z1"] - a["z0"], 0.05), (a["x"], (a["z0"] + a["z1"]) / 2, 2.2), glass)
        for k, dz in enumerate((a["z0"], a["z1"])):
            self.box(f"/World/DataHall/containment_door{k}", (a["w"], 0.05, 2.2), (a["x"], dz, 1.1), glass)
        for pl in dh["crah"]:
            self.box(pl.path, (1.2, 2.4, 2.4), (pl.x, pl.y, 1.2), crah_m)

    def _traffic(self):
        """Create prims for every non-machine, non-hero entity present in the trace."""
        by = {"agv": "agv", "forklift": "forklift", "person": None, "tugger": None}
        pcount = 0
        for eid, tr in self.ctx.tracks.items():
            if tr.cls in ("agv", "forklift"):
                x, y, yaw = tr.pose(self.ctx.t_min)
                p = self.reference(ASSETS[by[tr.cls]], f"/World/Traffic/{eid}", x, y, 0.0, math.degrees(yaw), animate=True)
                self.entity_prims[eid] = p or ""
            elif tr.cls == "tugger":
                x, y, yaw = tr.pose(self.ctx.t_min)
                m = self.material(f"/World/Looks/tug_{eid}", (0.15, 0.45, 0.2), rough=0.5)
                self.box(f"/World/Traffic/{eid}", (1.8, 0.9, 1.1), (x, y, 0.55), m, animate=True)
                self.entity_prims[eid] = f"/World/Traffic/{eid}"
            elif tr.cls == "person":
                key = ("person_a", "person_b", "person_c")[pcount % 3]
                pcount += 1
                x, y, yaw = tr.pose(self.ctx.t_min)
                p = self.reference(ASSETS[key], f"/World/Traffic/{eid}", x, y, 0.0, math.degrees(yaw), animate=True)
                if p is None:
                    m = self.material(f"/World/Looks/p_{eid}", (0.9, 0.5, 0.1), rough=0.7)
                    self.cyl(f"/World/Traffic/{eid}_proxy", 0.25, 1.7, (x, y, 0.85), m)
                self.entity_prims[eid] = p or ""
            elif tr.cls == "body":
                m = self.material("/World/Looks/biw", (0.62, 0.64, 0.66), rough=0.45, metal=0.7)
                self._biw(f"/World/Bodies/{eid}", m)
                self.entity_prims[eid] = f"/World/Bodies/{eid}"
        # bodies: user-provided USD (converted CC0/CC-BY mesh) replaces the procedural shell
        if getattr(self.args, "body_usd", None):
            print("[info] --body-usd given: swap _biw() for a reference to", self.args.body_usd)  # TODO wire if used

    def _biw(self, path: str, mat):
        """Blocky body-in-white: lower shell + cabin + skid. ~4.6 x 1.85 x 1.45 m, no glass/doors (it IS a BIW)."""
        Xf = self.UsdGeom.Xform.Define(self.stage, path)
        xf = Xf
        t = xf.AddTranslateOp()
        r = xf.AddRotateZOp()
        self.ops[path] = (t, r)
        self.box(path + "/lower", (4.6, 1.85, 0.55), (0, 0, 0.75), mat, hidden_in_thermal=False)
        self.box(path + "/cabin", (2.2, 1.7, 0.55), (-0.2, 0, 1.28), mat, hidden_in_thermal=False)
        self.box(path + "/skid", (4.9, 1.2, 0.2), (0, 0, 0.35), self.material(path + "_skid", (0.2, 0.2, 0.22), rough=0.6), hidden_in_thermal=False)

    def _hero(self):
        h = self.ctx.hero
        x, y, yaw = h.pose(self.ctx.t_min)
        base = self.reference(ASSETS["amr"], f"/World/Hero/{h.entity_id}", x, y, 0.0, math.degrees(yaw), animate=True)
        grey = self.material("/World/Looks/mast", (0.85, 0.86, 0.88), rough=0.35, metal=0.4)
        head = self.material("/World/Looks/mast_head", (0.1, 0.1, 0.12), rough=0.3)
        beacon = self.material("/World/Looks/beacon", (0.2, 0.2, 0.2), emissive=(0.0, 1.5, 0.3))
        parent = f"/World/Hero/{h.entity_id}"
        if base is None:
            self.stage.DefinePrim(parent, "Xform")
            xf = self.UsdGeom.Xformable(self.stage.GetPrimAtPath(parent))
            t = xf.AddTranslateOp()
            r = xf.AddRotateZOp()
            self.ops[parent] = (t, r)
            self.box(parent + "/body", (0.75, 0.55, 0.3), (0, 0, 0.25), grey, hidden_in_thermal=False)
        # sensor mast: pole + head (RGB + thermal + mic) + status beacon
        self.cyl(parent + "/mast", 0.025, 1.4, (-0.1, 0, 1.0), grey)
        self.box(parent + "/head", (0.18, 0.26, 0.16), (-0.1, 0, 1.75), head, hidden_in_thermal=False)
        self.box(parent + "/beacon", (0.08, 0.08, 0.08), (-0.1, 0, 1.88), beacon, hidden_in_thermal=False)
        self.entity_prims[h.entity_id] = parent
        self.beacon_mat = "/World/Looks/beacon"

    def _ble_rig(self):
        """Pre-create pooled packet spheres and pulse rings (animated per frame by apply_ble)."""
        ok = self.material("/World/Looks/ble_ok", (0, 0, 0), emissive=(0.1, 1.4, 2.0))
        bad = self.material("/World/Looks/ble_bad", (0, 0, 0), emissive=(2.0, 0.15, 0.1))
        ring = self.material("/World/Looks/ble_ring", (0, 0, 0), emissive=(0.2, 0.7, 1.0), opacity=0.5)
        for bi, (bx, bz) in enumerate(self.scn["ble"]["broadcasters"]):
            self.box(f"/World/BLE/tx_{bi}", (0.25, 0.25, 0.1), (bx, bz, 9.0), ring, hidden_in_thermal=False)
            self.sphere_light(f"/World/BLE/lamp_{bi}", (bx, bz, 8.9), radius=0.05, intensity=0.0, rgb=(0.3, 0.7, 1.0))
        for k in range(24):                                    # pooled packets
            c = self.UsdGeom.Sphere.Define(self.stage, f"/World/BLE/pkt_{k}")
            c.CreateRadiusAttr(0.14)
            tx = self.UsdGeom.Xformable(c).AddTranslateOp()
            tx.Set(self.Gf.Vec3d(0, 0, -50))
            self.ble_prims.append(dict(path=f"/World/BLE/pkt_{k}", op=tx, mat_ok=ok, mat_bad=bad))

    def _materials_for_thermal(self):
        pass

    # ---------- per-frame application ----------
    def _set_xf(self, path, x, y, yaw, z=0.0):
        t, r = self.ops[path]
        t.Set(self.Gf.Vec3d(x, y, z))
        r.Set(float(YAW_SIGN * math.degrees(yaw) + YAW_OFFSET_DEG))

    def apply(self, t: float, shot: Shot):
        ctx, scn = self.ctx, self.scn
        stage = self.stage
        with self.Sdf.ChangeBlock():
            for eid, prim in self.entity_prims.items():
                tr = ctx.tracks.get(eid)
                if tr is None or not prim or prim not in self.ops:
                    continue
                x, y, yaw = tr.pose(t)
                self._set_xf(prim, x, y, yaw, 0.0 if tr.active(t) else -50.0)   # outside its rows: parked under the floor
        # machines: zone colour, curtains, sparks, arms
        for m in scn["machines"]:
            mid = m["id"]
            tr = ctx.tracks.get(mid)
            st = tr.state_at(t) if tr else "IDLE"
            col, alpha = {"ACTIVE": ((2.0, 0.15, 0.1), 0.45), "IDLE": ((0.0, 0.6, 0.15), 0.25),
                          "HELD": ((2.0, 1.2, 0.0), 0.5), "TRIPPED": ((2.0, 0.0, 0.0), 0.7),
                          "STOPPED": ((0.8, 0.8, 0.0), 0.3)}.get(st, ((0.0, 0.6, 0.15), 0.25))
            self.set_emissive(self.zone_mats[mid], col, alpha)
            for cm in self.curtains.get(mid, []):
                self.set_emissive(cm, col, 0.12 + 0.3 * (st in ("ACTIVE", "TRIPPED")))
            active = tr.intervals("ACTIVE") if tr else []
            on, ph = phase_of(active, t) if active else (False, 0.0)
            if mid in self.sparks:
                lp, sm = self.sparks[mid]
                flick = on and 0.2 < ph < 0.85 and ((int(t * 30) * 2654435761) % 7 < 3)
                self.UsdLux.SphereLight(stage.GetPrimAtPath(lp)).GetIntensityAttr().Set(30000.0 if flick else 0.0)
                self.set_emissive(sm, (30, 20, 8) if flick else (0, 0, 0))
        hs = ctx.hero.state_at(t)
        self.set_emissive(self.beacon_mat, {"enter": (2.0, 0.2, 0.1), "wait": (2.0, 1.2, 0.0), "exit": (2.0, 1.2, 0.0)}.get(hs, (0.0, 1.5, 0.3)))
        self.drive_arms(t)
        self.apply_ble(t, shot)
        if shot.thermal:
            self.apply_thermal(t)

    def drive_arms(self, t: float):
        if self.world is None:
            return
        for mid, arms in self.arm_ctl.items():
            tr = self.ctx.tracks.get(mid)
            active = tr.intervals("ACTIVE") if tr else []
            on, ph = phase_of(active, t) if active else (False, 0.0)
            for art, k in getattr(self, "_arts", {}).get(mid, []):
                q = arm_q(ph, k) if on else HOME_Q
                art.set_joint_positions(q)                       # VERIFY: dof order / count (6) per model
        self.world.step(render=False)                            # one physics tick so link xforms follow

    def init_arms(self):
        """Wrap each arm in an Articulation and teleport its joints per frame. VERIFY on the smoke container:
        (1) the base stays where we placed it (fixed-base articulation under a translated parent), (2) dof == 6.
        Falls back to 'sparks only' (static home pose) when --arms static or init fails."""
        self._arts: dict[str, list] = {}
        if self.args.arms == "static":
            return
        try:
            from isaacsim.core.api import World
            from isaacsim.core.prims import SingleArticulation
            self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 60.0, rendering_dt=1 / 30.0)
            self.world.get_physics_context().set_gravity(0.0)
            self.world.reset()
            for mid, arms in self.arm_ctl.items():
                lst = []
                for path, k in arms:
                    root = self._articulation_root(path)
                    art = SingleArticulation(root)
                    art.initialize()
                    lst.append((art, k))
                self._arts[mid] = lst
            self.app.update()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] articulation init failed ({e!r}); arms stay at home pose, sparks only")
            self.world = None

    def _articulation_root(self, path: str) -> str:
        from pxr import UsdPhysics
        for p in self.Usd.PrimRange(self.stage.GetPrimAtPath(path)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(p.GetPath())
        return path

    def apply_ble(self, t: float, shot: Shot):
        for pk in self.ble_prims:
            pk["op"].Set(self.Gf.Vec3d(0, 0, -50))
        if not shot.show_ble:
            return
        hero = self.ctx.hero
        k = 0
        for ev in ble_events(self.scn, hero, t - 0.4, t + 1e-3):
            age = t - ev["t"]
            if age < 0 or k >= len(self.ble_prims):
                continue
            u = min(age / 0.35, 1.0)
            rx, ry, _ = hero.pose(t)
            p0 = (ev["bx"], ev["bz"], 9.0)
            p1 = (rx - 0.1, ry, 1.75)
            if not ev["ok"]:
                p1 = lerp3(p0, p1, 0.55)                              # lost: dies mid-air
                if u > 0.9:
                    continue
            pk = self.ble_prims[k]
            pk["op"].Set(self.Gf.Vec3d(*lerp3(p0, p1, u)))
            UsdShade = self.UsdShade
            UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(pk["path"])).Bind(pk["mat_ok"] if ev["ok"] else pk["mat_bad"])
            k += 1

    def set_thermal(self, on: bool):
        for p in self.real_visuals:
            pr = self.stage.GetPrimAtPath(p)
            if pr:
                (self.UsdGeom.Imageable(pr).MakeInvisible if on else self.UsdGeom.Imageable(pr).MakeVisible)()
        for sk in self.skins:
            pr = self.stage.GetPrimAtPath(sk[0])
            (self.UsdGeom.Imageable(pr).MakeVisible if on else self.UsdGeom.Imageable(pr).MakeInvisible)()
        for p in self.ceiling_lights:
            self.UsdLux.RectLight(self.stage.GetPrimAtPath(p)).GetIntensityAttr().Set(0.0 if on else LIGHT_INTENSITY_RECT)
        self.UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/Lights/dome")).GetIntensityAttr().Set(0.0 if on else LIGHT_INTENSITY_DOME)

    def apply_thermal(self, t: float):
        for path, row, idx, zf, mp in self.skins:
            c = rack_temp_c(row, idx, zf, t)
            self.set_emissive(mp, tuple(3.0 * v for v in thermal_lut((c - 20.0) / 22.0)))

    # ---------- camera ----------
    def make_camera(self, name: str):
        cam = self.UsdGeom.Camera.Define(self.stage, f"/World/Cams/{name}")
        cam.CreateClippingRangeAttr(self.Gf.Vec2f(0.05, 800.0))
        cam.CreateHorizontalApertureAttr(36.0)
        op = self.UsdGeom.Xformable(cam).AddTransformOp()
        return cam, op

    def point_camera(self, cam, op, c: Cam):
        cam.GetFocalLengthAttr().Set(float(c.focal_mm))
        view = self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*c.eye), self.Gf.Vec3d(*c.tgt), self.Gf.Vec3d(0, 0, 1))
        op.Set(view.GetInverse())


def render_shot(scene: KitScene, app, shot: Shot, a: float, b: float, args, out_dir: Path) -> Path:
    import omni.replicator.core as rep
    W, H = (int(v) for v in args.res.split("x"))
    cam, op = scene.make_camera(shot.name)
    rp = rep.create.render_product(f"/World/Cams/{shot.name}", (W, H))
    frames_dir = out_dir / "frames" / (shot.name + (f"_{args.label}" if args.label else ""))
    shutil.rmtree(frames_dir, ignore_errors=True)
    writer = rep.WriterRegistry.get("BasicWriter")          # VERIFY name in 5.1 (it is the stock PNG writer)
    writer.initialize(output_dir=str(frames_dir), rgb=True)
    writer.attach([rp])
    scene.set_thermal(shot.thermal)
    n = max(1, int((b - a) * args.fps / args.speed))
    for i in range(n):
        t = a + i * args.speed / args.fps
        scene.apply(t, shot)
        scene.point_camera(cam, op, shot.cam(t, a, b, scene.ctx))
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)
        if i % 30 == 0:
            print(f"[{shot.name}] frame {i}/{n}  t={t:.1f}s", flush=True)
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()
    scene.set_thermal(False)
    pngs = sorted(frames_dir.rglob("rgb_*.png"))
    if not pngs:
        raise RuntimeError(f"no frames written under {frames_dir}")
    mp4 = out_dir / (shot.name + (f"_{args.label}" if args.label else "") + ".mp4")
    ffmpeg_frames_to_mp4(pngs, mp4, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return mp4


def set_quality(quality: str) -> None:
    import carb
    st = carb.settings.get_settings()
    if quality == "final":
        st.set("/rtx/rendermode", "PathTracing")             # VERIFY enum spelling in 5.1
        st.set("/rtx/pathtracing/spp", 8)
        st.set("/rtx/pathtracing/totalSpp", 64)
        st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    else:
        st.set("/rtx/rendermode", "RaytracedLighting")


def run_kit(scn: dict, ctx: Ctx, shots: list[str], args, out_dir: Path) -> None:
    from isaacsim import SimulationApp
    W, H = (int(v) for v in args.res.split("x"))
    app = SimulationApp({"headless": True, "width": W, "height": H})
    set_quality(args.quality)
    scene = KitScene(app, scn, ctx, args)
    scene.build()
    scene.init_arms()
    for _ in range(args.warmup):
        app.update()
    outputs = []
    for name, p in zip(shots, plan(ctx, shots, args.fps, args.t0, args.t1, args.speed)):
        outputs.append(str(render_shot(scene, app, SHOTS[name], p["t0"], p["t1"], args, out_dir)))
    credits = sorted({(a.credit, a.licence) for a in scene.used_assets.values()})
    (out_dir / "assets_resolved.json").write_text(json.dumps(scene.resolved, indent=2))
    (out_dir / "credits.json").write_text(json.dumps(
        [dict(credit=c, licence=l) for c, l in credits] +
        [dict(credit="All scenes, schedules and temperatures are SYNTHETIC (InterCycle design assumptions).", licence="n/a")], indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(dict(outputs=outputs, args=vars(args)), indent=2, default=str))
    app.close()


# =====================================================================================
# 7. CLI
# =====================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("compose", help="split-screen two mp4s (ffmpeg only)")
    c.add_argument("--left", required=True); c.add_argument("--right", required=True); c.add_argument("--out", required=True)
    c.add_argument("--label-left", default="A  reactive"); c.add_argument("--label-right", default="B  InterCycle")
    c.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    ap.add_argument("--scenario", default="scenario/bodyshop.yaml")
    ap.add_argument("--trace", default="trace.parquet")
    ap.add_argument("--out", default="render/out")
    ap.add_argument("--shots", default="overview,curtain,ble,pov,datahall,ab_cam")
    ap.add_argument("--hero", default=None, help="entity_id of the hero AMR (default: first cls=amr)")
    ap.add_argument("--label", default="", help="suffix for output names, e.g. A or B for the split screen")
    ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="sim seconds per video second")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--quality", choices=["fast", "final"], default="fast")
    ap.add_argument("--rt-subframes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--arms", choices=["physx", "static"], default="physx")
    ap.add_argument("--body-usd", default=None, help="converted body-in-white USD to use instead of the procedural block body (see docs/RENDER.md)")
    ap.add_argument("--synth-bodies", action="store_true", help="fallback: invent body motion from S10 edges when the trace has no body rows")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "compose":
        compose_ab(Path(args.left), Path(args.right), Path(args.out), args.label_left, args.label_right, args.font)
        return 0
    scn = load_scenario(args.scenario)
    tracks = load_trace(args.trace)
    ctx = build_ctx(scn, tracks, args.hero, args.synth_bodies)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    bad = [s for s in shots if s not in SHOTS]
    if bad:
        raise SystemExit(f"unknown shots {bad}; have {sorted(SHOTS)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run(scn, tracks, ctx, shots, args)
        return 0
    run_kit(scn, ctx, shots, args, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
