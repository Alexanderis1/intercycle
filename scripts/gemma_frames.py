"""Render labelled, camera-like frames of machine hazard zones from OUR simulator (ground truth is free).

    python scripts/gemma_frames.py                  # 160 samples (+ the 0.5 s partner of each) -> out/gemma/frames/
    python scripts/gemma_frames.py --n 20 --out out/gemma/frames_small
    python scripts/gemma_frames.py --preview        # a dozen test renders in out/gemma/preview/

Pipeline
  1. Sim clips.  Two PhysicsRunner runs (strategy "A", the reactive robot: it produces real holds, and a robot in
     the scene) plus one scripted clip (World + Kinematics + MoverSystem stepped directly, with trips and holds
     injected as the same `Freeze` objects the physics creates).  Movers (people, AGVs, tuggers, forklifts) are
     recorded at 10 Hz.  Clips are cached in out/gemma/clips/.
  2. Stratified sampling: per machine (S10..S80, HEM1, SEAL1) 8 ACTIVE samples spread over the cycle fraction
     (weld, tip-dress, return ...), 4 IDLE, 4 STOPPED (hold, trip, andon, break).  Only instants whose state is
     constant over [t - 0.5 s, t + 1.0 s] are used, so a pair of frames 0.5 s apart has one label.
  3. Person scenarios, about 30 % of frames with a person inside the hazard zone.  The simulator's own people walk
     in the aisles (outside every zone), so a person inside a zone is a SYNTHETIC person placed for that sample:
     standing or walking, on free floor (static-map clearance and a gap to every moving-part circle at t-0.3..t+0.8),
     mostly in idle / held / stopped zones.  In a real plant a person in an ACTIVE zone would trip the light curtain,
     so the few "inside + active" frames stand for a curtain fault or bypass.  All of them are marked
     `"source": "synthetic"` in labels.json.  Labels are geometric: person_in_zone = any person body circle
     (r = 0.3 m) overlaps the zone rectangle.
  4. Render: fixed ceiling camera (9 m, 64 deg), pseudo-3D painter's-algorithm scene on a textured floor (slabs,
     stains, aisle lines, lamp pools, hatched zone border), conveyor + car bodies, robot arms from
     Kinematics.poses() (links, joints, shadows), light-curtain posts, hi-vis people, vehicles, the inspection AMR.
     Temporal integration over the exposure (K sub-frames) gives real motion blur / ghosting on moving parts and
     people; lens softness, vignette, signal-dependent noise and a JPEG round trip give CCTV character.
     A burn-in (camera id, sim clock) and a "SYNTHETIC" tag are added.

Outputs in out/gemma/frames/
    fNNN_<machine>.png      frame at t           (label key of labels.json / single mode)
    fNNN_<machine>_b.png    frame at t + 0.5 s   (pair mode: [a, b], oldest first)
    labels.json             {file: {machine, moving, person_in_zone, phase, ...}}   (single mode, label at t)
    labels_pair.json        same, keyed by the _b file with "images": [a]  (label at t + 0.5 s; harness compatible)
    fNNN_<machine>_pair.png side-by-side composites (only for --composites N samples)

CAVEATS: frames are synthetic renders, not real camera images; the exposure default (0.2 s) is a slow-shutter CCTV
setting chosen so that a single frame carries a motion cue (a real 1/30 s exposure blurs a 1 m/s arm by only ~2 px).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.scenario import load  # noqa: E402
from intercycle.sim import geometry as G  # noqa: E402
from intercycle.sim.movers import MoverSystem  # noqa: E402
from intercycle.sim.world import Freeze, World  # noqa: E402
from intercycle.truth import kinematics as KIN  # noqa: E402
from intercycle.truth.agents import agents as make_agents  # noqa: E402
from intercycle.truth.kinematics import Kinematics  # noqa: E402

OUT_ROOT = ROOT / "out" / "gemma"
W, H = 768, 512                 # output size
SS = 2                          # supersampling factor of the painter
FOV_DEG = 64.0
CAM_H = 9.0                     # ceiling cameras are at 9 m (scenario)
DTK = 0.1                       # mover tick
PERSON_R = 0.3
LIGHT = np.array([0.35, 1.0, -0.25])
LIGHT /= np.linalg.norm(LIGHT)
DAY0 = "2026-10-19"
MACHINES = ["S10", "S20", "S30", "S40", "S50", "S60", "S70", "S80", "HEM1", "SEAL1"]


# ====================================================================================================== clips
class Clip:
    """A stretch of simulated time with machine ground truth (World + Kinematics) and recorded movers."""

    def __init__(self, name, scn, seed, t0, t1, freezes, movers, robot, sm):
        self.name, self.scn, self.seed, self.t0, self.t1 = name, scn, seed, t0, t1
        self.freezes_spec = freezes                    # {group: [(t0, t1, kind, machine)]}
        self.world = World(scn, seed, t0, t1)
        for g, fl in freezes.items():
            self.world.groups[g].freezes = [Freeze(*f) for f in fl]
        self.kin = Kinematics(scn, self.world)
        self.mov, self.robot, self.sm = movers, robot, sm

    def _tick(self, t):
        u = (t - self.t0) / DTK
        i = min(max(int(math.floor(u)), 0), len(self.mov) - 2)
        return i, min(max(u - i, 0.0), 1.0)

    def movers_at(self, t):
        """[(id, cls, x, z, yaw, speed)] interpolated between the two surrounding 10 Hz ticks."""
        i, f = self._tick(t)
        a = {m[0]: m for m in self.mov[i]}
        b = {m[0]: m for m in self.mov[i + 1]}
        out = []
        for k, m in a.items():
            n = b.get(k)
            if n is None or math.hypot(n[2] - m[2], n[3] - m[3]) > 1.5:
                out.append((k, m[1], m[2], m[3], m[4], 0.0))
                continue
            sp = math.hypot(n[2] - m[2], n[3] - m[3]) / DTK
            out.append((k, m[1], m[2] + (n[2] - m[2]) * f, m[3] + (n[3] - m[3]) * f, m[4], sp))
        for k, n in b.items():
            if k not in a and f > 0.5:
                out.append((k, n[1], n[2], n[3], n[4], 0.0))
        return out

    def robot_at(self, t):
        if not self.robot:
            return None
        i, _ = self._tick(t)
        return self.robot[i]


def _record_physics(scn, name, seed, t0, t1, strategy):
    from intercycle.sim.physics import PhysicsRunner
    r = PhysicsRunner(scn, strategy, seed, t0, t1, record_hz=1.0 / DTK)
    r.run()
    n = int(round((t1 - t0) / DTK)) + 2
    mov, rob = [None] * n, [None] * n
    for rec in r.trace:
        i = int(round((rec[0] - t0) / DTK))
        if 0 <= i < n:
            rob[i] = (rec[1], rec[2], rec[3])
            mov[i] = [(m[0], m[1], m[2], m[3], m[4]) for m in rec[5]]
    for arr in (mov, rob):                                  # fill any gap with the previous tick
        last = None
        for i in range(n):
            if arr[i] is None:
                arr[i] = last
            else:
                last = arr[i]
        for i in range(n - 1, -1, -1):
            if arr[i] is None:
                arr[i] = arr[i + 1]
    fz = {g: [(f.t0, f.t1, f.kind, f.machine) for f in grp.freezes] for g, grp in r.world.groups.items()}
    info = {"metrics": r.metrics(), "events": r.events}
    return dict(name=name, seed=seed, t0=t0, t1=t1, freezes=fz, movers=mov, robot=rob, info=info)


def _record_scripted(scn, name, seed, t0, t1, script):
    """World + Kinematics + MoverSystem stepped directly; `script` = [(kind, machine, t_request, dur)]."""
    w = World(scn, seed, t0, t1)
    for kind, mid, treq, dur in sorted(script, key=lambda s: s[2]):
        g = w.group(mid)
        if kind == "trip":                                  # a robot crosses into an ACTIVE zone
            t = treq
            while t < t1 and not (w.active(mid, t) and not g.is_frozen(t)):
                t += 0.5
            w.trip(mid, t, "scripted", where=mid)
            g.freezes[-1].t1 = t + dur
        else:                                               # a robot inside an idle zone when the next start is due
            t = w.next_start_after(mid, treq)
            g.freezes.append(Freeze(t, t + dur, "hold", mid))
    fz = {g: [(f.t0, f.t1, f.kind, f.machine) for f in grp.freezes] for g, grp in w.groups.items()}
    kin = Kinematics(scn, w)
    sm = G.StaticMap(scn)
    ms = MoverSystem(make_agents(scn, seed), sm, t0)
    n = int(round((t1 - t0) / DTK)) + 2
    mov = []
    for i in range(n):
        t = t0 + i * DTK
        ms.step(t, DTK, [], kin.all_circles(t))
        mov.append([(m[0], m[1], m[2], m[3], m[4]) for m in ms.snapshot()])
    return dict(name=name, seed=seed, t0=t0, t1=t1, freezes=fz, movers=mov, robot=None, info={"script": script})


CLIP_SPECS = [
    ("P1_morning", "physics", 0, 7 * 3600, 7 * 3600 + 720),
    ("P2_break", "physics", 1, 8 * 3600 + 57 * 60, 9 * 3600 + 9 * 60),
    ("S1_scripted", "scripted", 2, 10 * 3600, 10 * 3600 + 540),
]
SCRIPT = [("trip", "S10", 10 * 3600 + 60, 90), ("trip", "S10", 10 * 3600 + 330, 90),
          ("hold", "S10", 10 * 3600 + 150, 25), ("hold", "S10", 10 * 3600 + 420, 30),
          ("trip", "HEM1", 10 * 3600 + 40, 90), ("trip", "SEAL1", 10 * 3600 + 160, 90),
          ("hold", "HEM1", 10 * 3600 + 220, 30), ("hold", "SEAL1", 10 * 3600 + 280, 30),
          ("trip", "HEM1", 10 * 3600 + 380, 60), ("hold", "SEAL1", 10 * 3600 + 400, 30)]


def load_clips(scn, cache_dir: Path, specs=CLIP_SPECS, refresh=False, log=print):
    cache_dir.mkdir(parents=True, exist_ok=True)
    sm = G.StaticMap(scn)
    clips = []
    for name, kind, seed, t0, t1 in specs:
        p = cache_dir / f"{name}.pkl"
        if p.is_file() and not refresh:
            d = pickle.loads(p.read_bytes())
        else:
            t_ = time.time()
            d = (_record_physics(scn, name, seed, t0, t1, "A") if kind == "physics"
                 else _record_scripted(scn, name, seed, t0, t1, SCRIPT))
            p.write_bytes(pickle.dumps(d))
            log(f"recorded clip {name} ({kind}, {t1 - t0:.0f} s of sim) in {time.time() - t_:.0f} s")
        clips.append(Clip(d["name"], scn, d["seed"], d["t0"], d["t1"], d["freezes"], d["movers"], d["robot"], sm))
    return clips


# ================================================================================================= ground truth
def scan_clip(clip, step=0.5, margin=25.0):
    """Per machine, per 0.5 s tick: (state, window kind, fraction).  Used for stratified sampling."""
    ts = np.arange(clip.t0 + margin, clip.t1 - margin, step)
    out = {}
    for m in clip.scn.machines:
        rows = []
        for t in ts:
            st, fr = clip.kin.phase(m.id, float(t))
            w = clip.world.window(m.id, float(t))
            rows.append((st, w.kind if w else "cycle", fr))
        out[m.id] = (ts, rows)
    return out


def category(state, wkind):
    if state == "active":
        return "active"
    if state in ("hold", "trip"):
        return state
    return "idle" if wkind == "cycle" else wkind          # andon / break / shift_change (extended idle)


def stable_candidates(scan, mid):
    ts, rows = scan[mid]
    out = []
    for i in range(1, len(ts) - 3):
        cats = {category(rows[j][0], rows[j][1]) for j in range(i - 1, i + 3)}
        if len(cats) == 1:
            out.append((float(ts[i]), category(rows[i][0], rows[i][1]), float(rows[i][2])))
    return out


def select_samples(clips, scans, rng, n_total, log=print):
    per = max(n_total // len(MACHINES), 1)
    n_act = round(per * 0.5)
    n_idle = round(per * 0.25)
    n_stop = per - n_act - n_idle
    picks = []
    used = {c.name: [] for c in clips}                      # (clip, t) spacing across machines
    for mid in MACHINES:
        cand = {k: [] for k in ("active", "idle", "hold", "trip", "andon", "break", "shift_change")}
        for ci, c in enumerate(clips):
            for t, cat, fr in stable_candidates(scans[ci], mid):
                cand[cat].append((ci, t, cat, fr))
        chosen = []

        def free(ci, t):
            return (all(abs(t - u) > 8 for (cj, u) in [(x[0], x[1]) for x in chosen] if cj == ci)
                    and all(abs(t - u) > 3 for u in used[clips[ci].name]))

        def take(pool):
            pool = [x for x in pool if free(x[0], x[1])]
            if not pool:
                return None
            x = pool[int(rng.integers(len(pool)))]
            chosen.append(x)
            used[clips[x[0]].name].append(x[1])
            return x

        for b in range(n_act):                              # one per bin of the active fraction
            lo, hi = b / n_act, (b + 1) / n_act
            if not take([x for x in cand["active"] if lo <= x[3] < hi]):
                take(cand["active"])
        for _ in range(n_idle):
            take(cand["idle"])
        order = ["trip", "hold", "andon", "break", "shift_change"]
        k = 0
        for _ in range(n_stop):
            for j in range(len(order)):
                cat = order[(k + j) % len(order)]
                if take(cand[cat]):
                    k = (k + j + 1) % len(order)
                    break
        if len(chosen) < per:
            log(f"  note: only {len(chosen)}/{per} samples found for {mid}")
        picks += [(mid, ci, t, cat, fr) for (ci, t, cat, fr) in chosen]
    return picks


# ======================================================================================================= camera
class Cam:
    def __init__(self, pos, target, fov_deg=FOV_DEG, w=W, h=H, ss=SS):
        self.pos = np.array(pos, float)
        f = np.array(target, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross([0.0, 1.0, 0.0], self.f)
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.f, self.r)
        self.R = np.stack([self.r, self.u, self.f])
        self.w, self.h, self.ss = w * ss, h * ss, ss
        self.fpx = (self.w / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx, self.cy = self.w / 2, self.h / 2

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        c = (P - self.pos) @ self.R.T
        z = np.maximum(c[:, 2], 1e-3)
        return np.stack([self.cx + self.fpx * c[:, 0] / z, self.cy - self.fpx * c[:, 1] / z], 1), z

    def visible(self, x, z, margin=0.12):
        p, d = self.project([[x, 0.8, z]])
        return (-margin * self.w < p[0, 0] < (1 + margin) * self.w and -margin * self.h < p[0, 1] < (1 + margin) * self.h
                and d[0] > 0.5)


def fit_camera(rect, ymax):
    x0, z0, x1, z1 = rect
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    for D in np.arange(6.0, 30.0, 0.25):
        cam = Cam((cx, CAM_H, cz - D), (cx, 0.0, cz + 0.5))
        pts = []
        for (px, pz) in ((x0 - 0.8, z0 - 0.8), (x1 + 0.8, z0 - 0.8), (x1 + 0.8, z1 + 0.8), (x0 - 0.8, z1 + 0.8)):
            pts += [(px, 0.0, pz), (px, ymax, pz)]
        p, _ = cam.project(pts)
        if (p[:, 0].min() > 0.03 * cam.w and p[:, 0].max() < 0.97 * cam.w
                and p[:, 1].min() > 0.04 * cam.h and p[:, 1].max() < 0.96 * cam.h):
            return cam
    return Cam((cx, CAM_H, cz - 30.0), (cx, 0.0, cz + 0.5))


# ======================================================================================================= floor
def _value_noise(rng, X, Z, cell, amp_shape=(64, 64)):
    lat = rng.random(amp_shape).astype(np.float32)
    u = (X / cell) % (amp_shape[0] - 1)
    v = (Z / cell) % (amp_shape[1] - 1)
    return ndimage.map_coordinates(lat, [u.ravel(), v.ravel()], order=1, mode="wrap").reshape(X.shape)


def floor_image(cam: Cam, scn, machine, seed=0):
    """Textured floor as seen by `cam` (SS x resolution). Returns (PIL RGB image, X, Z world coordinates)."""
    ys, xs = np.mgrid[0:cam.h, 0:cam.w].astype(np.float32)
    dx = (xs + 0.5 - cam.cx) / cam.fpx
    dy = (ys + 0.5 - cam.cy) / cam.fpx
    d = cam.f[None, None, :] + dx[..., None] * cam.r + (-dy)[..., None] * cam.u
    assert (d[..., 1] < 0).all(), "camera sees above the horizon"
    t = -cam.pos[1] / d[..., 1]
    X = (cam.pos[0] + t * d[..., 0]).astype(np.float32)
    Z = (cam.pos[2] + t * d[..., 2]).astype(np.float32)
    rng = np.random.default_rng([seed, sum(map(ord, machine))])
    base = np.array([150.0, 152.0, 150.0], np.float32)
    n = (0.55 * _value_noise(rng, X, Z, 0.35) + 0.30 * _value_noise(rng, X, Z, 1.6) + 0.15 * _value_noise(rng, X, Z, 5.0))
    img = base[None, None, :] * (0.88 + 0.22 * n)[..., None]
    # grain
    img *= (0.97 + 0.06 * _value_noise(rng, X, Z, 0.07))[..., None]
    # oil stains / tyre marks
    stain = _value_noise(rng, X + 3, Z - 5, 0.9)
    img *= (1.0 - 0.13 * np.clip((stain - 0.66) * 4, 0, 1))[..., None]
    # slab joints every 6 m
    jx = np.abs(((X + 3.0) % 6.0) - 3.0) < 0.025
    jz = np.abs(((Z + 3.0) % 6.0) - 3.0) < 0.025
    img *= (1.0 - 0.28 * (jx | jz))[..., None]
    # aisle edge lines (yellow, dashed) at z = 14, 18, 30, 34, 6, 10
    for zl in (14.0, 18.0, 30.0, 34.0):
        m = (np.abs(Z - zl) < 0.06) & (((X * 0.5) % 1.0) < 0.7)
        img[m] = (205, 180, 40)
    # pedestrian walkway across the line (green, white edge) at x = 52.5
    wk = (np.abs(X - 52.5) < 1.0) & (Z > 13.0) & (Z < 35.0)
    img[wk] = 0.55 * img[wk] + 0.45 * np.array([40, 120, 70], np.float32)
    # hazard zone border: yellow/black hatch band 0.4 m wide inside the rectangle, plus a thin outer line
    m = scn.machine(machine)
    x0, z0, x1, z1 = m.rect
    inside = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)
    edge = np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])
    band = inside & (edge < 0.4)
    stripe = (((X + Z) / 0.4) % 2.0) < 1.0
    wear = _value_noise(rng, X, Z, 0.25)
    col = np.where(stripe[..., None], np.array([232, 196, 24], np.float32), np.array([30, 30, 30], np.float32))
    col = col * (0.85 + 0.2 * wear)[..., None]
    img[band] = col[band]
    # other zones' hatch (visible neighbours), lighter
    for o in scn.machines:
        if o.id == machine:
            continue
        a0, b0, a1, b1 = o.rect
        ins = (X >= a0) & (X <= a1) & (Z >= b0) & (Z <= b1)
        ed = np.minimum.reduce([X - a0, a1 - X, Z - b0, b1 - Z])
        bd = ins & (ed < 0.4)
        img[bd] = col[bd]
    # ceiling lamp pools
    lamp = np.zeros_like(X)
    for lx in np.arange(0.0, 124.0, 6.0):
        if abs(lx - X.mean()) > 40:
            continue
        for lz in np.arange(3.0, 48.0, 6.0):
            lamp += np.exp(-((X - lx) ** 2 + (Z - lz) ** 2) / (2 * 2.2 ** 2))
    img *= (0.90 + 0.14 * np.clip(lamp, 0, 1.3))[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), X, Z


# ======================================================================================================= scene
def shade(col, f):
    return tuple(int(max(0, min(255, c * f))) for c in col)


def face_shade(n):
    return 0.52 + 0.48 * max(0.0, float(np.dot(n, LIGHT)))


def hull(pts):
    pts = sorted(set((round(float(p[0]), 1), round(float(p[1]), 1)) for p in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


class Scene:
    def __init__(self, cam: Cam, floor: Image.Image, floor_dark: Image.Image):
        self.cam, self.floor, self.floor_dark = cam, floor, floor_dark
        self.prims = []                                     # (layer, depth, fn)
        self.shadows = []                                   # lists of screen points
        self.ls = np.array([-0.5, -0.85])                   # light direction on screen (up-left)

    # ---- helpers
    def depth(self, P):
        return float(np.dot(np.asarray(P, float) - self.cam.pos, self.cam.f))

    def _shadow_pts(self, pts3):
        out = []
        for x, y, z in pts3:
            out.append((x - y * LIGHT[0] / LIGHT[1], 0.0, z - y * LIGHT[2] / LIGHT[1]))
        p, _ = self.cam.project(out)
        return p

    def add_shadow(self, pts3):
        self.shadows.append(hull(self._shadow_pts(pts3)))

    # ---- primitives
    def box(self, cx, cz, w, d, yaw, y0, y1, col, top=None, layer=1, shadow=True, stripes=None):
        s, c = math.sin(yaw), math.cos(yaw)
        fw, lt = np.array([s, c]), np.array([c, -s])
        cor = [np.array([cx, cz]) + a * (w / 2) * lt + b * (d / 2) * fw for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        bot = np.array([[p[0], y0, p[1]] for p in cor])
        topc = np.array([[p[0], y1, p[1]] for p in cor])
        cam = self.cam
        pb, zb = cam.project(bot)
        pt, zt = cam.project(topc)
        if min(zb.min(), zt.min()) < 1.5:                    # too close to / behind the camera plane
            return
        camxz = np.array([cam.pos[0], cam.pos[2]])
        faces = []
        for k in range(4):
            a, b = cor[k], cor[(k + 1) % 4]
            e = b - a
            nrm = np.array([e[1], -e[0]])
            nrm /= (np.linalg.norm(nrm) + 1e-9)
            mid = (a + b) / 2
            if np.dot(nrm, mid - np.array([cx, cz])) < 0:
                nrm = -nrm
            if np.dot(nrm, camxz - mid) > 0:
                faces.append((k, nrm))
        topcol = top or col

        def fn(dr):
            for k, nrm in faces:
                k2 = (k + 1) % 4
                poly = [tuple(pb[k]), tuple(pb[k2]), tuple(pt[k2]), tuple(pt[k])]
                dr.polygon(poly, fill=shade(col, face_shade(np.array([nrm[0], 0.0, nrm[1]]))), outline=shade(col, 0.45))
            dr.polygon([tuple(p) for p in pt], fill=shade(topcol, face_shade(np.array([0.0, 1.0, 0.0]))), outline=shade(topcol, 0.5))
            if stripes:
                stripes(dr, pt, pb)
        self.prims.append((layer, self.depth([cx, (y0 + y1) / 2, cz]), fn))
        if shadow:
            self.add_shadow(np.vstack([bot, topc]))

    def capsule(self, p0, p1, r0, r1, col, layer=1, shadow=True, depth_bias=0.0, joint=True):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        cam, ls = self.cam, self.ls

        if min(float(np.dot(p0 - cam.pos, cam.f)), float(np.dot(p1 - cam.pos, cam.f))) < 1.5:
            return

        def fn(dr):
            P, z = cam.project([p0, p1])
            rp0, rp1 = cam.fpx * r0 / z[0], cam.fpx * r1 / z[1]
            d = P[1] - P[0]
            L = float(np.hypot(*d))
            if L < 1e-3:
                n = np.array([1.0, 0.0])
            else:
                n = np.array([-d[1], d[0]]) / L
            side = 1.0 if np.dot(n, ls) >= 0 else -1.0
            n = n * side                                    # n points toward the light

            def band(a, b, f):
                poly = [P[0] + n * rp0 * a, P[1] + n * rp1 * a, P[1] + n * rp1 * b, P[0] + n * rp0 * b]
                dr.polygon([tuple(q) for q in poly], fill=shade(col, f))
            for rp, Pc in ((rp0, P[0]), (rp1, P[1])):
                dr.ellipse([Pc[0] - rp, Pc[1] - rp, Pc[0] + rp, Pc[1] + rp], fill=shade(col, 0.62))
            band(-1.0, 1.0, 0.62)
            band(-0.55, 0.95, 0.86)
            band(0.05, 0.75, 1.0)
            band(0.30, 0.55, 1.18)
            if joint:
                for rp, Pc in ((rp0, P[0]), (rp1, P[1])):
                    dr.ellipse([Pc[0] - rp * 0.95, Pc[1] - rp * 0.95, Pc[0] + rp * 0.95, Pc[1] + rp * 0.95],
                               fill=shade(col, 0.8), outline=shade(col, 0.45))
                    hx, hy = Pc[0] + n[0] * rp * 0.3, Pc[1] + n[1] * rp * 0.3
                    dr.ellipse([hx - rp * 0.35, hy - rp * 0.35, hx + rp * 0.35, hy + rp * 0.35], fill=shade(col, 1.12))
        self.prims.append((layer, self.depth((p0 + p1) / 2) + depth_bias, fn))
        if shadow:
            pts = []
            for p, r in ((p0, r0), (p1, r1)):
                for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
                    pts.append((p[0] + r * math.cos(a), p[1], p[2] + r * math.sin(a)))
            self.add_shadow(pts)

    def ellipse3(self, c, r, col, layer=1, depth_bias=0.0, outline=None):
        c = np.asarray(c, float)
        cam = self.cam
        if float(np.dot(c - cam.pos, cam.f)) < 1.5:
            return

        def fn(dr):
            P, z = cam.project([c])
            rp = cam.fpx * r / z[0]
            dr.ellipse([P[0, 0] - rp, P[0, 1] - rp, P[0, 0] + rp, P[0, 1] + rp], fill=col, outline=outline)
        self.prims.append((layer, self.depth(c) + depth_bias, fn))

    def poly3(self, pts, col, layer=1, depth_bias=0.0):
        pts = np.asarray(pts, float)
        cam = self.cam

        def fn(dr):
            P, _ = cam.project(pts)
            dr.polygon([tuple(p) for p in P], fill=col)
        self.prims.append((layer, self.depth(pts.mean(0)) + depth_bias, fn))

    # ---- compose
    def render(self):
        cam = self.cam
        mask = Image.new("L", (cam.w, cam.h), 0)
        md = ImageDraw.Draw(mask)
        for poly in self.shadows:
            if len(poly) >= 3:
                md.polygon([tuple(p) for p in poly], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(3 * SS / 2))
        mask = mask.point(lambda v: int(v * 0.62))
        img = Image.composite(self.floor_dark, self.floor, mask)
        dr = ImageDraw.Draw(img)
        for layer, depth, fn in sorted(self.prims, key=lambda p: (p[0], -p[1])):
            fn(dr)
        return img


# ----------------------------------------------------------------------------------------------- entities
VEST = [(255, 118, 8), (255, 122, 10), (205, 238, 20), (255, 140, 0), (210, 240, 40)]
HAT = [(240, 240, 235), (245, 214, 40), (236, 236, 230), (40, 90, 175)]
SKIN = [(224, 172, 140), (190, 140, 105), (140, 98, 72), (236, 190, 160)]
PANTS = [(34, 44, 74), (55, 58, 64), (40, 52, 80)]


def draw_person(sc: Scene, x, z, yaw, speed, t, pid, color_seed=0):
    rng = np.random.default_rng([color_seed, sum(map(ord, str(pid)))])
    vest, hat = VEST[rng.integers(len(VEST))], HAT[rng.integers(len(HAT))]
    skin, pants = SKIN[rng.integers(len(SKIN))], PANTS[rng.integers(len(PANTS))]
    phase0 = float(rng.uniform(0, 6.28))
    fw = np.array([math.sin(yaw), math.cos(yaw)])
    lt = np.array([math.cos(yaw), -math.sin(yaw)])
    walking = speed > 0.15
    sw = 0.26 * math.sin(2 * math.pi * 1.7 * (speed / 1.2) * t + phase0) if walking else 0.0
    sway = 0.0 if walking else 0.015 * math.sin(0.7 * t + phase0)
    base = np.array([x, z]) + sway * lt
    P = lambda off_lat, off_fw, y: np.array([base[0] + lt[0] * off_lat + fw[0] * off_fw, y, base[1] + lt[1] * off_lat + fw[1] * off_fw])  # noqa: E731
    cam = sc.cam
    # whole figure as one painter's item: sub-parts in order of height (legs, torso, arms, head, hat)
    parts = [("cap", P(-0.10, +sw, 0.05), P(-0.10, +sw * 0.3, 0.92), 0.075, 0.095, pants),
             ("cap", P(0.10, -sw, 0.05), P(0.10, -sw * 0.3, 0.92), 0.075, 0.095, pants),
             ("cap", P(0, 0, 0.90), P(0, 0, 1.50), 0.185, 0.17, vest),
             ("cap", P(0, 0, 1.04), P(0, 0, 1.08), 0.19, 0.19, (222, 224, 220)),
             ("cap", P(0, 0, 1.26), P(0, 0, 1.30), 0.185, 0.185, (222, 224, 220)),
             ("cap", P(-0.23, -sw * 0.8, 1.42), P(-0.25, -sw * 1.0, 0.98), 0.05, 0.045, shade(vest, 0.8)),
             ("cap", P(0.23, +sw * 0.8, 1.42), P(0.25, +sw * 1.0, 0.98), 0.05, 0.045, shade(vest, 0.8)),
             ("cap", P(0, 0, 1.52), P(0, 0, 1.58), 0.055, 0.055, skin),
             ("sph", P(0, 0, 1.64), 0.105, skin),
             ("cap", P(0, 0, 1.70), P(0, 0, 1.73), 0.125, 0.115, hat)]
    for kind, *a in parts:
        if kind == "cap":
            sc.capsule(a[0], a[1], a[2], a[3], a[4], layer=1, shadow=False, depth_bias=0.02, joint=False)
        else:
            sc.ellipse3(a[0], a[1], shade(a[2], 0.95), depth_bias=0.03)
    # shadow of the whole figure
    pts = []
    for yy, rr in ((0.0, 0.14), (0.9, 0.2), (1.5, 0.19), (1.74, 0.11)):
        for ang in np.linspace(0, 2 * math.pi, 7)[:-1]:
            pts.append((base[0] + rr * math.cos(ang), yy, base[1] + rr * math.sin(ang)))
    sc.add_shadow(pts)


def draw_vehicle(sc: Scene, cls, x, z, yaw):
    s, c = math.sin(yaw), math.cos(yaw)
    fwd = np.array([s, c])

    def at(off):
        return x + fwd[0] * off, z + fwd[1] * off
    if cls == "agv":
        sc.box(x, z, 1.0, 1.6, yaw, 0.10, 0.42, (38, 92, 170), top=(225, 190, 40))
        bx, bz = at(0.5)
        sc.box(bx, bz, 0.12, 0.12, yaw, 0.42, 0.62, (240, 150, 20))
    elif cls == "tugger":
        sc.box(x, z, 1.0, 1.4, yaw, 0.12, 1.15, (200, 40, 36), top=(170, 36, 32))
        for k in range(3):
            ox, oz = at(-(0.7 + 0.2 + 0.8 + 1.8 * k))
            sc.box(ox, oz, 1.0, 1.6, yaw, 0.22, 0.5, (90, 92, 98))
            sc.box(ox, oz, 0.9, 1.4, yaw, 0.5, 1.05, (170, 130, 80), top=(190, 150, 98))
    elif cls == "forklift":
        sc.box(x, z, 1.1, 1.9, yaw, 0.10, 0.95, (235, 150, 20), top=(210, 130, 18))
        for sx in (-1, 1):
            px, pz = x + math.cos(yaw) * sx * 0.5 + fwd[0] * 0.2, z - math.sin(yaw) * sx * 0.5 + fwd[1] * 0.2
            sc.box(px, pz, 0.07, 0.07, yaw, 0.95, 2.0, (50, 50, 52), shadow=False)
        gx, gz = at(0.1)
        sc.box(gx, gz, 1.15, 1.6, yaw, 2.0, 2.06, (60, 60, 62), shadow=False)
        mx, mz = at(1.05)
        for sx in (-1, 1):
            sc.box(mx + math.cos(yaw) * sx * 0.3, mz - math.sin(yaw) * sx * 0.3, 0.1, 0.1, yaw, 0.1, 2.1, (60, 60, 64))
        for sx in (-1, 1):
            fx, fz = at(1.9)
            sc.box(fx + math.cos(yaw) * sx * 0.3, fz - math.sin(yaw) * sx * 0.3, 0.1, 1.0, yaw, 0.04, 0.1, (80, 80, 84))


def draw_robot(sc: Scene, x, z, yaw):
    sc.box(x, z, 0.8, 1.05, yaw, 0.08, 0.46, (232, 234, 238), top=(245, 246, 248))
    sc.box(x, z, 0.82, 0.25, yaw, 0.2, 0.3, (35, 90, 175), shadow=False)
    sc.capsule((x, 0.46, z), (x, 1.35, z), 0.04, 0.04, (60, 62, 68), joint=False)
    fx, fz = x + math.sin(yaw) * 0.08, z + math.cos(yaw) * 0.08
    sc.box(fx, fz, 0.25, 0.18, yaw, 1.35, 1.52, (40, 44, 52))
    sc.ellipse3((x + math.sin(yaw) * 0.2, 1.43, z + math.cos(yaw) * 0.2), 0.05, (20, 30, 60), depth_bias=-0.05)


ARM_COL = (226, 104, 24)
ARM_DARK = (58, 60, 66)


def draw_arm(sc: Scene, arm, kind, tool_kind):
    bx, bz = arm["base"]
    ex, ez = arm["elbow_xy"]
    tx, tz = arm["tool_xy"]
    sh = np.array([bx, 1.25, bz])
    el = np.array([ex, 1.62, ez])
    tl = np.array([tx, 1.18, tz])
    sc.capsule((bx, 0.9, bz), sh, 0.27, 0.27, ARM_DARK)                       # shoulder column
    sc.capsule(sh, el, 0.21, 0.17, ARM_COL, depth_bias=0.03)                  # upper arm
    sc.capsule(el, tl, 0.16, 0.10, ARM_COL, depth_bias=0.04)                  # forearm
    d = tl - el
    d[1] = 0
    n = d / (np.linalg.norm(d) + 1e-9)
    if tool_kind == "weld":                                                  # C-gun
        a = tl + n * 0.1
        sc.capsule(tl, np.array([a[0], 0.78, a[2]]), 0.07, 0.07, ARM_DARK, depth_bias=0.05)
        sc.capsule(np.array([a[0], 0.78, a[2]]), np.array([a[0] + n[0] * 0.3, 0.74, a[2] + n[2] * 0.3]), 0.05, 0.05, (190, 190, 195), depth_bias=0.05)
    elif tool_kind == "roller":
        sc.capsule(tl, np.array([tx, 0.85, tz]), 0.06, 0.06, ARM_DARK, depth_bias=0.05)
        sc.capsule(np.array([tx - 0.13, 0.82, tz]), np.array([tx + 0.13, 0.82, tz]), 0.09, 0.09, (200, 205, 210), depth_bias=0.06)
    else:                                                                    # sealer nozzle
        sc.capsule(tl, np.array([tx, 0.8, tz]), 0.06, 0.025, (70, 150, 200), depth_bias=0.05)


# ----------------------------------------------------------------------------------------------- the scene
def build_scene(clip: Clip, cam: Cam, floor, floor_dark, scn, machine, t, extra_people, quiet_robot=False):
    sc = Scene(cam, floor, floor_dark)
    kin, poses = clip.kin, clip.kin.poses(t)
    m = scn.machine(machine)
    x0, z0, x1, z1 = m.rect
    # light-curtain posts at the zone corners (all machines in view)
    for o in scn.machines:
        a0, b0, a1, b1 = o.rect
        if not (abs(o.x - m.x) < 18 and abs(o.z - m.z) < 16):
            continue
        for k, (px, pz) in enumerate(((a0, b0), (a1, b0), (a0, b1), (a1, b1))):
            inset = G.POST_INSET_WALK if abs(px - G.WALK_X) < 1.0 else G.POST_INSET
            sx, sz = (1 if px == a0 else -1), (1 if pz == b0 else -1)
            sc.box(px + sx * inset, pz + sz * G.POST_INSET, 0.14, 0.14, 0.0, 0.0, 1.7, (210, 170, 20), top=(60, 60, 60))
            sc.box(px + sx * inset, pz + sz * G.POST_INSET, 0.15, 0.15, 0.0, 1.55, 1.7, (30, 170, 210), shadow=False)
    # conveyor (line stations): bed with a gap at the walkway
    if m.on_line or abs(m.z - 24) < 20:
        lo, hi = max(G.LINE_X[0], m.x - 14), min(G.LINE_X[1], m.x + 14)
        segs = [(lo, min(hi, G.WALK_X - 1.0)), (max(lo, G.WALK_X + 1.0), hi)]
        for a, b in segs:
            if b - a > 0.5:
                def stripes(dr, pt, pb, a=a, b=b):
                    for xx in np.arange(a + 0.3, b, 0.6):
                        P, _ = cam.project([[xx, 0.55, 22.8], [xx, 0.55, 25.2]])
                        dr.line([tuple(P[0]), tuple(P[1])], fill=(34, 40, 52), width=2)
                sc.box((a + b) / 2, 24.0, b - a, 2.4, 0.0, 0.0, 0.55, (52, 62, 84), top=(92, 100, 116), layer=0, stripes=stripes)
    # robot bases + dressers (line stations: diagonal pair, cells: bases + fixture table)
    for o in scn.machines:
        if abs(o.x - m.x) > 18 or abs(o.z - m.z) > 18:
            continue
        if o.kind == "unload_lifter":
            for sx, sz in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                sc.box(o.x + sx * 4.0, o.z + sz * 3.4, 0.4, 0.4, 0.0, 0.0, 6.6, (88, 92, 100))
            for sz in (-1, 1):
                sc.box(o.x, o.z + sz * 3.4, 8.4, 0.35, 0.0, 6.3, 6.6, (88, 92, 100), shadow=False)
        elif o.on_line:
            for sx, sz in ((-1, -1), (1, 1)):
                sc.box(o.x + sx * 1.8, 24 + sz * 3.3, 0.9, 0.9, 0.0, 0.0, 0.9, (74, 76, 84), top=(100, 102, 110))
        else:
            sc.box(o.x, o.z, 3.0, 1.6, 0.0, 0.0, 0.85, (92, 98, 108), top=(128, 134, 144))
            sc.box(o.x, o.z, 2.5, 1.2, 0.0, 0.85, 0.9, (176, 180, 186), top=(196, 200, 206), shadow=False)
            for sx, sz in ((-1, -1), (1, 1)):
                sc.box(o.x + sx * 1.8, o.z + sz * 2.0, 0.9, 0.9, 0.0, 0.0, 0.9, (74, 76, 84), top=(100, 102, 110))
    for mid, dx, dz in kin.dressers:
        o = scn.machine(mid)
        if abs(o.x - m.x) < 18:
            sc.box(dx, dz, 0.5, 0.5, 0.0, 0.0, 0.9, (150, 152, 158), top=(226, 104, 24))
    # car bodies on skids
    for xx in poses["SKIDS"]["x"]:
        if abs(xx - m.x) > 16:
            continue
        sc.box(xx, 24.0, 4.2, 1.9, 0.0, 0.55, 0.66, (60, 66, 76), top=(80, 86, 98))
        sc.box(xx, 24.0, 3.9, 1.7, 0.0, 0.66, 1.2, (168, 174, 182), top=(188, 194, 202))
        sc.box(xx - 0.2, 24.0, 2.1, 1.5, 0.0, 1.2, 1.7, (150, 156, 166), top=(172, 178, 188))
        zf = 24.0 - 0.75 - 0.01                               # side windows (open BIW apertures) on the camera-side face
        for x_a, x_b in ((xx - 1.15, xx - 0.35), (xx - 0.25, xx + 0.55), (xx + 0.65, xx + 0.85)):
            sc.poly3([(x_a, 1.27, zf), (x_b, 1.27, zf), (x_b, 1.62, zf), (x_a, 1.62, zf)], (34, 40, 50), depth_bias=-0.35)
        sc.poly3([(xx - 1.9, 0.72, 24 - 0.86), (xx + 1.9, 0.72, 24 - 0.86), (xx + 1.9, 0.78, 24 - 0.86), (xx - 1.9, 0.78, 24 - 0.86)],
                 (40, 44, 52), depth_bias=-0.35)
    # AGV in a docking slot
    for d in poses["AGVDOCK"]["docked"]:
        if abs(d["xy"][0] - m.x) < 14:
            draw_vehicle(sc, "agv", d["xy"][0] + K_DOCK_DX, d["xy"][1], math.pi / 2)
    # machine moving parts
    for o in scn.machines:
        if abs(o.x - m.x) > 18 or abs(o.z - m.z) > 18:
            continue
        pe = poses[o.id]
        tk = {"weld_station": "weld", "respot": "weld", "framing": "weld", "roller_hemming": "roller", "sealer": "seal"}.get(o.kind, "weld")
        for arm in pe.get("arms", []):
            draw_arm(sc, arm, o.kind, tk)
        if o.kind == "unload_lifter":
            h = pe["lifter_h"]
            sc.box(o.x, o.z, 4.6, 2.6, 0.0, h - 0.35, h, (214, 178, 28), top=(232, 196, 40))
            if pe["carrying"]:
                sc.box(o.x, o.z, 3.9, 1.7, 0.0, h - 1.0, h - 0.35, (168, 174, 182), top=(188, 194, 202))
                sc.box(o.x - 0.2, o.z, 2.1, 1.5, 0.0, h - 0.35 - 0.0, h + 0.0, (150, 156, 166), shadow=False)
    # movers
    for (pid, cls, x, z, yaw, sp) in clip.movers_at(t):
        if not cam.visible(x, z):
            continue
        if cls == "person":
            draw_person(sc, x, z, yaw, sp, t, pid)
        else:
            draw_vehicle(sc, cls, x, z, yaw)
    for p in extra_people:
        px, pz, pyaw, psp = p["fn"](t)
        draw_person(sc, px, pz, pyaw, psp, t, p["id"], color_seed=3)
    rb = clip.robot_at(t)
    if rb is not None and cam.visible(rb[0], rb[1]) and not quiet_robot:
        draw_robot(sc, rb[0], rb[1], rb[2])
    return sc


K_DOCK_DX = KIN.DOCK_DX
_FONT = None


def _font(size):
    global _FONT
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_frame(clip, cam, floor, floor_dark, scn, machine, t, people, exposure, k_sub, rng, cam_id, burn=True):
    """One camera frame at time t: temporal integration over [t - exposure, t], then CCTV character."""
    ts = [t] if exposure <= 0 or k_sub <= 1 else [t - exposure * (1 - i / (k_sub - 1)) for i in range(k_sub)]
    acc = None
    for tt in ts:
        im = build_scene(clip, cam, floor, floor_dark, scn, machine, tt, people).render()
        a = np.asarray(im, np.float32)
        acc = a if acc is None else acc + a
    acc /= len(ts)
    img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(0.55))
    a = np.asarray(img, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1.0 - 0.20 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    a *= vig[..., None]
    a = a * np.array([1.0, 1.0, 0.965], np.float32)                        # slight warm cast
    gray = a.mean(-1, keepdims=True)
    a = gray + 0.88 * (a - gray)                                           # CCTV desaturation
    sig = 2.5 + 0.035 * a
    a += rng.normal(0, 1, a.shape).astype(np.float32) * sig
    a += rng.normal(0, 1, (H, 1, 1)).astype(np.float32) * 0.8              # line noise
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if burn:
        dr = ImageDraw.Draw(img)
        hh = int(t // 3600)
        mm = int((t % 3600) // 60)
        ss = t % 60
        txt = f"{cam_id}  {DAY0}  {hh:02d}:{mm:02d}:{ss:04.1f}"
        f = _font(13)
        dr.text((9, 8), txt, fill=(0, 0, 0), font=f)
        dr.text((8, 7), txt, fill=(235, 235, 235), font=f)
        dr.text((W - 128, H - 20), "SYNTHETIC RENDER", fill=(0, 0, 0), font=f)
        dr.text((W - 129, H - 21), "SYNTHETIC RENDER", fill=(235, 235, 235), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


# ================================================================================================= people
def rect_overlap(rect, x, z, r=PERSON_R):
    x0, z0, x1, z1 = rect
    d = math.hypot(max(x0 - x, 0.0, x - x1), max(z0 - z, 0.0, z - z1))
    return d < r


def signed_margin(rect, x, z):
    """> 0 inside (metres to the nearest boundary), < 0 outside (metres to the rectangle)."""
    x0, z0, x1, z1 = rect
    if x0 <= x <= x1 and z0 <= z <= z1:
        return min(x - x0, x1 - x, z - z0, z1 - z)
    return -math.hypot(max(x0 - x, 0.0, x - x1), max(z0 - z, 0.0, z - z1))


def _free_floor(clip, x, z, ts, scn, machine, gap=0.30):
    """Physically plausible spot: free of static obstacles and moving-part circles over the exposure window."""
    if float(clip.sm.clearance(np.array([[x, z]]))[0]) < PERSON_R + 0.25:
        return False
    for t in ts:
        c = clip.kin.all_circles(t)
        if len(c) and (np.hypot(c[:, 0] - x, c[:, 1] - z) - c[:, 2] - PERSON_R).min() < gap:
            return False
    m = scn.machine(machine)
    if m.on_line and abs(z - 24.0) < 1.2 + PERSON_R + 0.35:                # conveyor strip
        return False
    if (not m.on_line) and abs(x - m.x) < 1.5 + PERSON_R + 0.25 and abs(z - m.z) < 0.8 + PERSON_R + 0.25:   # fixture table
        return False
    return True


def place_synthetic(clip, scn, machine, cam, t, kind, rng, pid, hard=False, tries=600):
    """kind 'inside' or 'outside'. Returns a person dict (fn(t) -> x, z, yaw, speed) or None."""
    m = scn.machine(machine)
    rect = m.rect
    x0, z0, x1, z1 = rect
    ts = [t - 0.3, t, t + 0.5, t + 0.8]
    others = [o.rect for o in scn.machines if o.id != machine]
    for _ in range(tries):
        if kind == "inside":
            lo = 0.35 if hard else 0.9
            hi = 0.85 if hard else 3.0
            edge = rng.uniform(lo, hi)
            if hard:                                            # near the hatch band, not in the middle
                side = rng.integers(4)
                u, v = rng.uniform(0, 1, 2)
                x = (x0 + edge, x1 - edge, x0 + u * (x1 - x0), x0 + u * (x1 - x0))[side]
                z = (z0 + v * (z1 - z0), z0 + v * (z1 - z0), z0 + edge, z1 - edge)[side]
            else:
                x, z = rng.uniform(x0 + lo, x1 - lo), rng.uniform(z0 + lo, z1 - lo)
            if signed_margin(rect, x, z) < 0.3:
                continue
        else:
            d = rng.uniform(0.6, 1.1) if hard else rng.uniform(1.2, 4.5)
            ang_side = rng.integers(4)
            if ang_side == 0:
                x, z = rng.uniform(x0, x1), z0 - d
            elif ang_side == 1:
                x, z = rng.uniform(x0, x1), z1 + d
            elif ang_side == 2:
                x, z = x0 - d, rng.uniform(z0, z1)
            else:
                x, z = x1 + d, rng.uniform(z0, z1)
            if any(rect_overlap(o, x, z, PERSON_R + 0.4) for o in others):
                continue
            if signed_margin(rect, x, z) > -0.55:
                continue
        if not cam.visible(x, z, margin=-0.04):
            continue
        walking = rng.random() < 0.5
        sp = float(rng.uniform(0.5, 1.2)) if walking else 0.0
        if kind == "inside":
            tx, tz = rng.uniform(x0 + 1, x1 - 1), rng.uniform(z0 + 1, z1 - 1)
            hd = math.atan2(tx - x, tz - z)
        else:
            hd = float(rng.uniform(-math.pi, math.pi))
        vx, vz = math.sin(hd) * sp, math.cos(hd) * sp
        ok = True
        for tt in ts:
            px, pz = x + vx * (tt - t), z + vz * (tt - t)
            if not _free_floor(clip, px, pz, [tt], scn, machine):
                ok = False
                break
            sm_ = signed_margin(rect, px, pz)
            if (kind == "inside" and sm_ < 0.3) or (kind == "outside" and sm_ > -0.5):
                ok = False
                break
        if not ok:
            continue
        yaw = hd if walking else float(rng.uniform(-math.pi, math.pi))
        return {"id": pid, "source": "synthetic", "kind": kind, "walking": walking, "speed": sp,
                "fn": (lambda tt, x=x, z=z, vx=vx, vz=vz, yaw=yaw, sp=sp, t=t: (x + vx * (tt - t), z + vz * (tt - t), yaw, sp))}
    return None


def people_truth(clip, rect, people, t, cam):
    """All people drawn at t (natural movers in view + synthetic) and the geometric label."""
    rows = []
    for (pid, cls, x, z, yaw, sp) in clip.movers_at(t):
        if cls == "person" and cam.visible(x, z, 0.0):
            rows.append({"id": pid, "source": "sim", "x": round(x, 2), "z": round(z, 2),
                         "inside": rect_overlap(rect, x, z), "margin_m": round(signed_margin(rect, x, z), 2)})
    for p in people:
        x, z, _, _ = p["fn"](t)
        rows.append({"id": p["id"], "source": "synthetic", "x": round(x, 2), "z": round(z, 2),
                     "inside": rect_overlap(rect, x, z), "margin_m": round(signed_margin(rect, x, z), 2)})
    return rows


def arm_speed(clip, mid, t, dt=0.2):
    """Max tool-tip speed (m/s) of the machine around t, a measure of how visible its motion is."""
    a, b = clip.kin.poses(t - dt / 2), clip.kin.poses(t + dt / 2)
    v = 0.0
    for aa, bb in zip(a[mid].get("arms", []), b[mid].get("arms", [])):
        v = max(v, math.hypot(aa["tool_xy"][0] - bb["tool_xy"][0], aa["tool_xy"][1] - bb["tool_xy"][1]) / dt)
    if "lifter_h" in a[mid]:
        v = max(v, abs(a[mid]["lifter_h"] - b[mid]["lifter_h"]) / dt)
    return v


def skids_moving(clip, t, dt=0.2):
    a = clip.kin.poses(t - dt / 2)["SKIDS"]["x"]
    b = clip.kin.poses(t + dt / 2)["SKIDS"]["x"]
    return len(a) == len(b) and any(abs(p - q) > 0.01 for p, q in zip(a, b))


MACH_DESC = {"weld_station": "spot-welding station", "respot": "re-spot welding station", "framing": "framing station",
             "unload_lifter": "unload lifter", "roller_hemming": "roller-hemming cell", "sealer": "sealer cell"}


def zone_hint(m):
    return (f"{m.size[0]:g} x {m.size[1]:g} m rectangle marked on the floor with a yellow-black hatched border, "
            f"centred in the image; light-curtain posts stand at its corners")


# ======================================================================================================== main
def make_frames(out: Path, n: int, seed: int, exposure: float, k_sub: int, composites: int, refresh: bool,
                specs=CLIP_SPECS, log=print):
    scn = load()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clips = load_clips(scn, OUT_ROOT / "clips", specs, refresh, log)
    scans = [scan_clip(c) for c in clips]
    picks = select_samples(clips, scans, rng, n, log)
    log(f"selected {len(picks)} samples from {len(clips)} clips")
    # person scenarios: 30 % inside (mostly idle/held/stopped), ~35 % outside near, rest none
    by_m = {}
    for i, p in enumerate(picks):
        by_m.setdefault(p[0], []).append(i)
    scenario = ["none"] * len(picks)
    for mid, idxs in by_m.items():
        act = [i for i in idxs if picks[i][3] == "active"]
        non = [i for i in idxs if picks[i][3] != "active"]
        rng.shuffle(act)
        rng.shuffle(non)
        k_in = round(0.31 * len(idxs))
        n_in_act = min(len(act), max(1, round(k_in * 0.2)))
        ins = act[:n_in_act] + non[:max(k_in - n_in_act, 0)]
        rest = [i for i in idxs if i not in ins]
        rng.shuffle(rest)
        for i in ins:
            scenario[i] = "inside"
        for i in rest[:round(0.38 * len(idxs))]:
            scenario[i] = "outside"
    cams, floors = {}, {}
    labels, labels_pair = {}, {}
    t_start = time.time()
    meta = []
    for si, (mid, ci, t, cat, fr) in enumerate(picks):
        clip = clips[ci]
        m = scn.machine(mid)
        if mid not in cams:
            cams[mid] = fit_camera(m.rect, 6.8 if m.kind == "unload_lifter" else 2.6)
            fl, _, _ = floor_image(cams[mid], scn, mid)
            fd = Image.fromarray((np.asarray(fl, np.float32) * 0.45).astype(np.uint8))
            floors[mid] = (fl, fd)
        cam, (fl, fd) = cams[mid], floors[mid]
        prng = np.random.default_rng([seed, si, 17])
        people = []
        sc = scenario[si]
        hard = bool(prng.random() < 0.35)
        if sc == "inside":
            p = place_synthetic(clip, scn, mid, cam, t, "inside", prng, f"SYN{si:03d}", hard=hard)
            if p is None:
                p = place_synthetic(clip, scn, mid, cam, t, "inside", prng, f"SYN{si:03d}", hard=False, tries=1500)
            if p is None:
                log(f"  could not place an inside person for sample {si} ({mid} {cat}); left without")
                sc = "none"
            else:
                people.append(p)
        elif sc == "outside":
            natural_near = any(cls == "person" and cam.visible(x, z, -0.02) and abs(signed_margin(m.rect, x, z)) < 9
                               for (_, cls, x, z, _, _) in clip.movers_at(t))
            if not natural_near or prng.random() < 0.3:
                p = place_synthetic(clip, scn, mid, cam, t, "outside", prng, f"SYN{si:03d}", hard=hard)
                if p is not None:
                    people.append(p)
        stem = f"f{si:03d}_{mid}"
        frng = np.random.default_rng([seed, si, 99])
        cam_id = f"CAM-{mid}"
        ta, tb = t, t + 0.5
        ia = render_frame(clip, cam, fl, fd, scn, mid, ta, people, exposure, k_sub, frng, cam_id)
        ib = render_frame(clip, cam, fl, fd, scn, mid, tb, people, exposure, k_sub, frng, cam_id)
        ia.save(out / f"{stem}.png")
        ib.save(out / f"{stem}_b.png")
        if si < composites:
            comp = Image.new("RGB", (2 * W + 8, H), (20, 20, 20))
            comp.paste(ia, (0, 0))
            comp.paste(ib, (W + 8, 0))
            comp.save(out / f"{stem}_pair.png")
        st_a, fr_a = clip.kin.phase(mid, ta)
        st_b, fr_b = clip.kin.phase(mid, tb)
        pa, pb = people_truth(clip, m.rect, people, ta, cam), people_truth(clip, m.rect, people, tb, cam)
        wk = clip.world.window(mid, ta)
        sp = arm_speed(clip, mid, ta)
        base = {"machine": mid, "machine_type": MACH_DESC.get(m.kind, m.kind), "zone_hint": zone_hint(m),
                "category": cat, "clip": clip.name, "sim_t": round(ta, 2), "state": st_a,
                "window_kind": wk.kind if wk else "cycle", "state_frac": round(fr_a, 3),
                "phase": f"{cat}:{st_a}@{fr_a:.2f}", "arm_speed_mps": round(sp, 2),
                "visible_motion": bool(sp > 0.2 and st_a == "active"), "conveyor_moving": skids_moving(clip, ta),
                "person_scenario": sc, "exposure_s": exposure,
                "robot_in_zone": bool(clip.robot_at(ta) and rect_overlap(m.rect, clip.robot_at(ta)[0], clip.robot_at(ta)[1], 0.45))}
        labels[f"{stem}.png"] = {**base, "moving": st_a == "active", "person_in_zone": any(p["inside"] for p in pa),
                                 "people": pa, "pair": [f"{stem}.png", f"{stem}_b.png"],
                                 "pair_composite": f"{stem}_pair.png" if si < composites else None}
        labels_pair[f"{stem}_b.png"] = {**base, "sim_t": round(tb, 2), "state": st_b, "state_frac": round(fr_b, 3),
                                        "phase": f"{cat}:{st_b}@{fr_b:.2f}", "moving": st_b == "active",
                                        "person_in_zone": any(p["inside"] for p in pb), "people": pb,
                                        "images": [f"{stem}.png"], "frame_dt_s": 0.5}
        if (si + 1) % 10 == 0 or si == len(picks) - 1:
            el = time.time() - t_start
            log(f"  rendered {si + 1}/{len(picks)} samples ({el:.0f} s, {el / (si + 1):.1f} s/sample)")
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    (out / "labels_pair.json").write_text(json.dumps(labels_pair, indent=1), encoding="utf-8")
    summary = summarize(labels)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(json.dumps(summary, indent=1))
    return labels, labels_pair


def summarize(labels):
    L = list(labels.values())
    n = len(L)
    from collections import Counter
    return {"n_samples": n, "n_images": 2 * n,
            "moving_true": sum(l["moving"] for l in L), "person_in_zone_true": sum(l["person_in_zone"] for l in L),
            "person_in_zone_frac": round(sum(l["person_in_zone"] for l in L) / max(n, 1), 3),
            "person_in_zone_active": sum(1 for l in L if l["person_in_zone"] and l["moving"]),
            "synthetic_person_inside": sum(1 for l in L if any(p["inside"] and p["source"] == "synthetic" for p in l["people"])),
            "sim_person_inside": sum(1 for l in L if any(p["inside"] and p["source"] == "sim" for p in l["people"])),
            "any_person_in_view": sum(1 for l in L if l["people"]),
            "by_machine": dict(Counter(l["machine"] for l in L)), "by_category": dict(Counter(l["category"] for l in L)),
            "by_clip": dict(Counter(l["clip"] for l in L)), "visible_motion": sum(l["visible_motion"] for l in L),
            "conveyor_moving": sum(l["conveyor_moving"] for l in L)}


def preview(out: Path, exposure, k_sub, refresh):
    scn = load()
    out.mkdir(parents=True, exist_ok=True)
    clips = load_clips(scn, OUT_ROOT / "clips", CLIP_SPECS, refresh)
    scans = [scan_clip(c) for c in clips]
    rng = np.random.default_rng(1)
    want = [("S30", "active"), ("S30", "idle"), ("S30", "trip"), ("S60", "active"), ("S80", "active"), ("S80", "idle"),
            ("HEM1", "active"), ("HEM1", "hold"), ("SEAL1", "active"), ("S10", "break"), ("S50", "active"), ("S20", "andon")]
    for k, (mid, cat) in enumerate(want):
        cands = [(ci, t) for ci in range(len(clips)) for (t, c, fr) in stable_candidates(scans[ci], mid) if c == cat]
        if not cands:
            print("no candidate for", mid, cat)
            continue
        ci, t = cands[int(rng.integers(len(cands)))]
        clip = clips[ci]
        m = scn.machine(mid)
        cam = fit_camera(m.rect, 6.8 if m.kind == "unload_lifter" else 2.6)
        fl, _, _ = floor_image(cam, scn, mid)
        fd = Image.fromarray((np.asarray(fl, np.float32) * 0.45).astype(np.uint8))
        prng = np.random.default_rng([5, k])
        people = []
        p = place_synthetic(clip, scn, mid, cam, t, "inside" if k % 2 == 0 else "outside", prng, f"SYN{k}", hard=(k % 4 == 0))
        if p:
            people.append(p)
        t_ = time.time()
        img = render_frame(clip, cam, fl, fd, scn, mid, t, people, exposure, k_sub, np.random.default_rng(k), f"CAM-{mid}")
        img.save(out / f"p{k:02d}_{mid}_{cat}.png")
        print(f"preview {mid} {cat} t={t:.1f} clip={clip.name} people={[(q['kind'], q['walking']) for q in people]} "
              f"render {time.time() - t_:.2f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=160, help="number of samples (each gives a frame at t and one at t+0.5 s)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "frames")
    ap.add_argument("--exposure", type=float, default=0.20, help="integration time in s (motion blur / ghosting)")
    ap.add_argument("--k", type=int, default=7, help="sub-frames per exposure")
    ap.add_argument("--composites", type=int, default=8, help="write this many side-by-side pair PNGs")
    ap.add_argument("--refresh", action="store_true", help="re-record the sim clips")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args(argv)
    if a.preview:
        preview(OUT_ROOT / "preview", a.exposure, a.k, a.refresh)
        return 0
    make_frames(a.out, a.n, a.seed, a.exposure, a.k, a.composites, a.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
