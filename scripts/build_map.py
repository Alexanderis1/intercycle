"""Build the observer's layered map of dynamics from simulated camera observations and render it.

    python scripts/build_map.py [--days 3] [--hours 6] [--workers 8] [--seed 0] [--quick]

Floor simulation (no robot): the scenario's movers (truth/agents.py, unobstructed travel) and the moving machine
parts (truth/kinematics.py when present, else a small stand-in) are stepped at 2 Hz; the camera shim
(observer/mapping.py) turns them into noisy observations (0.15 m, 5 % dropouts, 10 s blackouts, 0.2 s latency);
the ConditionTracker computes time-of-day / production state / cycle phase from those observations alone; the
LayeredMap accumulates counts. Windows run in parallel and are merged (counts add).

Outputs: out/layered_map.npz, out/map/*.png, out/map/summary.json (sizes, BLE assembly time).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.scenario import Scenario, load  # noqa: E402
from intercycle.sim import geometry as G  # noqa: E402
from intercycle.sim.world import World  # noqa: E402
from intercycle.truth.agents import PathAgent, agents as make_agents  # noqa: E402
from intercycle.truth.kinematics import Kinematics  # noqa: E402
from intercycle.types import MoverClass  # noqa: E402

OUT = ROOT / "out"
DAY = 86_400.0
HZ = M.FRAME_HZ
CLS = {MoverClass.PERSON: "person", MoverClass.AGV: "agv", MoverClass.TUGGER: "tugger", MoverClass.FORKLIFT: "forklift"}


# ------------------------------------------------------------------------------------------------ movers (vectorised)

def agent_track(a: PathAgent, t: np.ndarray):
    """Vectorised PathAgent.at(tau = t) and presence for an array of times: x, z, heading, speed, present."""
    segs = np.array(a._seg)
    pts = np.array(a._pts)
    dur = segs / a.speed
    starts = np.concatenate([[0.0], np.cumsum(a.pause_s + dur)])
    u = (t + a.phase_s) % a._cycle
    i = np.minimum(np.searchsorted(starts, u, side="right") - 1, len(segs) - 1)
    w = u - starts[i]
    p0, p1 = pts[i], pts[i + 1]
    head = np.arctan2(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
    pausing = w < a.pause_s
    f = np.clip((w - a.pause_s) / np.maximum(dur[i], 1e-9), 0.0, 1.0)
    f = np.where(pausing, 0.0, f)
    x = p0[:, 0] + (p1[:, 0] - p0[:, 0]) * f
    z = p0[:, 1] + (p1[:, 1] - p0[:, 1]) * f
    sp = np.where(pausing, 0.0, a.speed)
    if a.windows:
        tod = t % DAY
        pres = np.zeros(len(t), bool)
        for lo, hi in a.windows:
            pres |= (tod >= lo) & (tod < hi)
    else:
        pres = np.ones(len(t), bool)
    return x, z, head, sp, pres


class MoverBank:
    """Body circles of all movers at arbitrary times, vectorised over a block of times."""

    def __init__(self, scn: Scenario, seed: int):
        self.agents = make_agents(scn, seed)
        off, cls, owner = [], [], []
        for k, a in enumerate(self.agents):
            fp = G.footprint(CLS.get(a.cls, "person"), 0.0, 0.0, 0.0)   # yaw 0: (lat, fwd, r)
            for c in fp:
                off.append((c[1], c[0], c[2]))                           # fwd, lat, r
                cls.append(M.PERSON if a.cls == MoverClass.PERSON else M.VEHICLE)
                owner.append(k)
        self.off = np.array(off)
        self.cls = np.array(cls, np.int8)
        self.owner = np.array(owner)

    def block(self, t: np.ndarray):
        """Arrays for times t (T,): CX, CZ (T, C), HD, SP (T, C), PRES (T, C)."""
        T, A = len(t), len(self.agents)
        X, Z, H, S, P = (np.zeros((T, A)) for _ in range(5))
        P = P.astype(bool)
        for k, a in enumerate(self.agents):
            X[:, k], Z[:, k], H[:, k], S[:, k], P[:, k] = agent_track(a, t)
        o = self.owner
        s, c = np.sin(H[:, o]), np.cos(H[:, o])
        f, l = self.off[:, 0][None, :], self.off[:, 1][None, :]
        return X[:, o] + s * f + c * l, Z[:, o] + c * f - s * l, H[:, o], S[:, o], P[:, o]

    def state(self, t_cap: float, row, parts=None, skids=None, docks=None) -> M.SimState:
        """SimState at t_cap from row k of `block`. `docks`: parked AGV circles (stationary vehicles) to add."""
        CX, CZ, H, S, P = row
        sel = P
        circ = np.stack([CX[sel], CZ[sel], self.off[:, 2][sel]], 1)
        cls, hd, sp = self.cls[sel], H[sel], S[sel]
        if docks is not None and len(docks):
            circ = np.vstack([circ, docks])
            cls = np.concatenate([cls, np.full(len(docks), M.VEHICLE, np.int8)])
            hd = np.concatenate([hd, np.zeros(len(docks))])
            sp = np.concatenate([sp, np.zeros(len(docks))])
        return M.SimState(t_cap, circ, cls, hd, sp, parts or {}, skids if skids is not None else np.zeros((0, 3)))


# ------------------------------------------------------------------------------------------------ one window

def run_window(seed: int, t0: float, t1: float, scn: Scenario | None = None, shim_seed: int | None = None,
               subframes: int = 3, progress: bool = False, record: bool = False, checkpoints=None, on_checkpoint=None,
               tick_hook=None, extra_parts=None, shim_kw: dict | None = None, build: bool = True,
               **map_kw) -> tuple[M.LayeredMap, M.ConditionTracker, dict]:
    """Simulate [t0, t1) at 2 Hz and build a LayeredMap from camera observations only. With record=True the info dict
    also carries the per-tick conditioning (t_cap, prod_state, phase_bin) and the truth objects (world, kinematics).

    checkpoints / on_checkpoint(t, map): called once the simulated time reaches each checkpoint (absolute sim s), used
    by the timeline to save partial maps. tick_hook(k, t_cap, states, obs, cond): called every tick with the true
    sub-frame states (truth, for scoring only) and what the observer saw / computed. extra_parts(t) -> {id: circles}:
    additional moving bodies (e.g. a newly installed machine) handed to the camera as parts. build=False runs the
    tracker only (held-out scoring windows)."""
    scn = scn or load()
    world = World(scn, seed, t0, t1)
    kin = Kinematics(scn, world)
    mids = [m.id for m in scn.machines]
    movers = MoverBank(scn, seed)
    shim = M.CameraShim(scn, seed=seed * 1000 + int(t0 // 3600) if shim_seed is None else shim_seed, subframes=subframes,
                        **(shim_kw or {}))
    builder = M.MapBuilder(scn, **map_kw)
    cps = sorted(float(c) for c in (checkpoints or []) if t0 < c <= t1)
    rec_t, rec_prod, rec_bin = [], [], []
    n = int(round((t1 - t0) * HZ))
    block_n = 1200
    for b0 in range(0, n, block_n):
        ks = np.arange(b0, min(n, b0 + block_n))
        t_now = t0 + ks / HZ
        t_cap = t_now - shim.latency
        blk = movers.block(t_cap)
        for j, k in enumerate(ks):
            tc = float(t_cap[j])
            states = []
            for s in range(subframes):
                ts = tc - shim.exposure * (1 - (s + 1) / subframes)
                bd = kin.bodies(ts)
                parts = {mid: bd[mid] for mid in mids}
                if extra_parts is not None:
                    parts.update(extra_parts(ts))
                if s == subframes - 1:
                    states.append(movers.state(tc, tuple(a[j] for a in blk), parts, bd["SKIDS"], bd["AGVDOCK"]))
                else:
                    states.append(M.SimState(ts, parts=parts, skids=bd["SKIDS"]))
            obs = shim.observe(states)
            if build:
                cond = builder.feed(obs)
            else:
                cond = builder.condition(obs)
            if tick_hook is not None:
                tick_hook(int(k), tc, states, obs, cond)
            while cps and float(t_now[j]) >= cps[0]:
                on_checkpoint(cps.pop(0), builder.map, cond)
            if record:
                rec_t.append(tc)
                rec_prod.append(cond.prod_state)
                rec_bin.append(cond.phase_bin.copy())
        if progress:
            print(f"  seed {seed} t={t_now[-1] / 3600:.2f} h", flush=True)
    while cps and on_checkpoint is not None and cps[0] <= t1 + 1e-6:     # a checkpoint at the window end
        on_checkpoint(cps.pop(0), builder.map, cond)
    info = {"seed": seed, "t0": t0, "t1": t1}
    if record:
        info.update(t=np.array(rec_t), prod=np.array(rec_prod), bins=np.array(rec_bin), world=world, kin=kin)
    return builder.map, builder.tracker, info


def observer_kw(opts: dict | None) -> dict:
    """run_window keywords from a picklable options dict: {"measured": bool (CameraShim = MEASURED_SHIM),
    "pm": path of a fitted PhaseModel (discovered line state; "bins": True also its sub-phases as L1 bins),
    "calendar": bool (public calendar as external input; unit tests only), "shim": extra CameraShim keywords}."""
    opts = opts or {}
    kw: dict = {}
    shim = dict(M.MEASURED_SHIM) if opts.get("measured") else {}
    shim.update(opts.get("shim", {}))
    if shim:
        kw["shim_kw"] = shim
    if opts.get("pm"):
        from intercycle.observer import phases as PH
        PH.MOTIF_ENGINE = "numpy"
        kw["phase_model"] = PH.PhaseModel.load(opts["pm"], load())
        kw["use_bins"] = bool(opts.get("bins", False))
        kw["use_line_state"] = True
    if opts.get("calendar"):
        kw["calendar"] = True
    return kw


def _worker(args):
    """(seed, t0, t1, path[, checkpoints[, opts]]): build one window and save it; with checkpoints also save the partial
    map at each checkpoint as <path stem>_<t:09d>.npz plus the observer's conditioning then (<stem>_<t>.json: production
    state, phase bins, phase), which the timeline's snapshots carry."""
    seed, t0, t1, path = args[:4]
    cps = args[4] if len(args) > 4 else None
    opts = args[5] if len(args) > 5 else None
    t = time.time()
    stem = Path(path).with_suffix("")

    def save(tc, mp, cond):
        mp.save(f"{stem}_{int(round(tc)):09d}.npz")
        Path(f"{stem}_{int(round(tc)):09d}.json").write_text(json.dumps(dict(
            t=tc, prod=int(cond.prod_state), phase_bin=[int(v) for v in cond.phase_bin],
            phase=[None if not np.isfinite(v) else float(v) for v in cond.phase])))
    m, _, _ = run_window(seed, t0, t1, checkpoints=cps, on_checkpoint=save if cps else None, **observer_kw(opts))
    m.save(path)
    return path, time.time() - t


# ------------------------------------------------------------------------------------------------ windows

def plan_windows(days: int, hours: float, seed: int) -> list[tuple[int, float, float]]:
    """`hours`-long windows tiling `days` days (the tiling contains 6 breaks and 3 shift changes per day)."""
    return [(seed, d * DAY + h0 * 3600, d * DAY + (h0 + hours) * 3600) for d in range(days) for h0 in np.arange(0, 24, hours)]


QUICK = [(0, 8.5 * 3600, 9.75 * 3600),      # break 09:00 (15 min)
         (0, 11.0 * 3600, 12.25 * 3600),    # break 11:30 (30 min)
         (0, 13.6 * 3600, 14.4 * 3600)]     # shift change 14:00


def build(windows, workers: int, tmp: Path) -> M.LayeredMap:
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(s, t0, t1, str(tmp / f"w{i:03d}.npz")) for i, (s, t0, t1) in enumerate(windows)]
    t = time.time()
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for path, dt in ex.map(_worker, jobs):
                print(f"  window done {Path(path).name} in {dt:.0f} s", flush=True)
    else:
        for j in jobs:
            print(f"  window done {Path(_worker(j)[0]).name}", flush=True)
    print(f"simulated {sum(t1 - t0 for _, t0, t1 in windows) / 3600:.1f} h in {time.time() - t:.0f} s wall", flush=True)
    m = M.LayeredMap.load(jobs[0][3])
    for j in jobs[1:]:
        m.merge(M.LayeredMap.load(j[3]))
    for j in jobs:
        Path(j[3]).unlink()
    return m


# ------------------------------------------------------------------------------------------------ rendering

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"
SEQ = {"envelope": ("#f2f8f7", "#0b6e69"), "vehicle": ("#fff5e6", "#b45309"), "person": ("#eef4fb", "#1d4f91")}


def _cmap(kind: str):
    from matplotlib.colors import LinearSegmentedColormap
    lo, hi = SEQ[kind]
    return LinearSegmentedColormap.from_list(kind, [lo, hi])


def _backdrop(ax, scn: Scenario, label_machines: bool = True):
    from matplotlib.patches import Rectangle
    h, d = scn.hall, scn.data_hall
    ax.add_patch(Rectangle((h["x"][0], h["z"][0]), h["x"][1] - h["x"][0], h["z"][1] - h["z"][0], fill=False, ec=INK, lw=1.0))
    ax.add_patch(Rectangle((d["x"][0], d["z"][0]), d["x"][1] - d["x"][0], d["z"][1] - d["z"][0], fill=False, ec=INK, lw=1.0))
    ax.add_patch(Rectangle((6, 22.8), 80, 2.4, fill=False, ec=MUTED, lw=0.6, ls=":"))
    for m in scn.machines:
        x0, z0, x1, z1 = m.rect
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=MUTED, lw=0.6, ls="--"))
        if label_machines:
            ax.text(m.x, z1 + 0.3, m.id, ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xlim(-1, 121)
    ax.set_ylim(-1, 49)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=7, color=MUTED)
    ax.set_ylabel("z (m)", fontsize=7, color=MUTED)
    ax.tick_params(labelsize=6, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)


def _heat(ax, m: M.LayeredMap, p: np.ndarray, kind: str, vmax: float = 0.6, heading: np.ndarray | None = None):
    from matplotlib.colors import PowerNorm
    s = m.spec["site"]
    ext = [s["x0"], s["x0"] + s["nx"] * s["res"], s["z0"], s["z0"] + s["nz"] * s["res"]]
    q = np.ma.masked_less(p, 0.02)
    im = ax.imshow(q.T, origin="lower", extent=ext, cmap=_cmap(kind), norm=PowerNorm(0.5, 0.02, vmax), interpolation="nearest", zorder=0)
    if heading is not None:
        ix, iz = np.nonzero((p >= 0.08) & (heading >= 0))
        ang = heading[ix, iz] * math.pi / 4
        ax.quiver(s["x0"] + (ix + 0.5) * s["res"], s["z0"] + (iz + 0.5) * s["res"], np.sin(ang), np.cos(ang), color=INK, alpha=0.55,
                  scale=55, width=0.0016, headwidth=4, headlength=4, zorder=2)
    return im


def aggregate_traffic(m: M.LayeredMap, cls: int, prod: int, tods=range(M.N_TOD)):
    """Occupancy probability pooled over several time-of-day slices of one production state, and dominant heading."""
    hit = np.zeros((m.nx, m.nz))
    vis = np.zeros((m.nx, m.nz))
    head = np.zeros((M.N_HEAD, m.nx, m.nz))
    for t in tods:
        S = t * M.N_PROD + prod
        if m.visits_cam[S].max() < m.min_ticks:
            continue
        hit += m.hits[cls][S]
        vis += m.visits_cam[S][m.cam_of_cell]
        head += m.head[cls][S]
    p = np.where(vis > 0, hit / np.maximum(vis, 1), 0.0)
    return p, np.where(head.sum(0) >= 3, head.argmax(0), -1)


def fig_l1(m: M.LayeredMap, scn: Scenario, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    _backdrop(ax, scn)
    res = m.spec["env_res"]
    for d in m.spec["machines"]:
        e = m.envelope_union(d["id"], range(M.N_BINS))
        ax.imshow(np.ma.masked_where(~e, e).T, origin="lower", extent=[d["x0"], d["x0"] + d["nx"] * res, d["z0"], d["z0"] + d["nz"] * res],
                  cmap=_cmap("envelope"), vmin=-1, vmax=1.2, interpolation="nearest", zorder=1)
        for bx in m.box_cover(d["id"], tuple(range(M.N_BINS)), max_boxes=24, cover_frac=0.995):
            ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=INK, lw=0.25, alpha=0.5, zorder=3))
    ax.set_title("L1 machine envelopes: floor swept by moving parts at any cycle phase (p >= 5 %), with their box cover", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_traffic(m: M.LayeredMap, scn: Scenario, cls: int, path: Path, title: str, kind: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    _backdrop(ax, scn)
    p, h = aggregate_traffic(m, cls, M.RUNNING)
    im = _heat(ax, m, p, kind, 0.5, h)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("occupancy probability (running, all day)", fontsize=7, color=MUTED)
    cb.ax.tick_params(labelsize=6, colors=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_slices(m: M.LayeredMap, scn: Scenario, cls: int, kind: str, path: Path, name: str):
    plt = _plt()
    hh = lambda h: int(h * 2)
    picks = [("running 10:00", hh(10), M.RUNNING), ("running 03:00 (night)", hh(3), M.RUNNING),
             ("break 09:00-09:30", hh(9), M.BREAK), ("break 11:30-12:00", hh(11.5), M.BREAK),
             ("shift change 14:00", hh(14), M.SHIFT_CHANGE), ("andon, 10:00", hh(10), M.ANDON)]
    fig, axs = plt.subplots(3, 2, figsize=(11, 10.5), dpi=130)
    for ax, (label, tod, prod) in zip(axs.flat, picks):
        _backdrop(ax, scn, label_machines=False)
        r = m._resolve(tod, prod, cls)
        p, h = m.traffic(cls, tod, prod)
        _heat(ax, m, p, kind, 0.5, h)
        if r == (tod, prod):
            src = ""
        else:
            src = " (no data; showing " + (f"{M.PROD_NAMES[r[1]]} {r[0] // 2:02d}:{30 * (r[0] % 2):02d}" if r else "nothing") + ")"
        ax.set_title(f"{name}: {label}{src}", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_phase_bins(m: M.LayeredMap, mid: str, path: Path):
    plt = _plt()
    from matplotlib.colors import Normalize
    d = next(x for x in m.spec["machines"] if x["id"] == mid)
    res = m.spec["env_res"]
    ext = [d["x0"], d["x0"] + d["nx"] * res, d["z0"], d["z0"] + d["nz"] * res]
    fig, axs = plt.subplots(3, 6, figsize=(14, 7.2), dpi=120)
    for b, ax in zip(range(M.N_BINS), axs.flat):
        pr = m.envelope_prob(mid, b)
        ax.imshow(np.ma.masked_less(pr, 0.01).T, origin="lower", extent=ext, cmap=_cmap("envelope"), norm=Normalize(0, 1), interpolation="nearest")
        for bx in m.box_cover(mid, b):
            ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=INK, lw=0.6))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        lab = f"phase {b / M.N_PHASE:.2f}-{(b + 1) / M.N_PHASE:.2f}" if b < M.N_PHASE else "stopped / overdue"
        ax.set_title(f"{lab}  (n={int(m.l1_visits[m._mi(mid), b])})", fontsize=7, color=INK)
    for ax in list(axs.flat)[M.N_BINS:]:
        ax.axis("off")
    fig.suptitle(f"{mid}: L1 envelope by cycle-phase bin (occupancy probability, box cover outlined)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_assembly(report: dict, sims: dict, path: Path):
    """Left: what the map costs on the air (bytes by section, chunk count). Right: time to a complete verified map."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=140, gridspec_kw={"width_ratios": [1, 1.1]})
    st = report["XOR parity 1/8"]["stats"]
    secs = [("meta", st["meta"]), ("L1 box covers", st["L1"]), ("L2 vehicles", st["L2_vehicle"]), ("L3 people", st["L3_person"])]
    y = np.arange(len(secs))[::-1]
    a1.barh(y, [v for _, v in secs], color="#6b7785", height=0.55)
    for yy, (n, v) in zip(y, secs):
        a1.text(v + 300, yy, f"{v:,} B", va="center", fontsize=7, color=INK)
    a1.set_yticks(y)
    a1.set_yticklabels([n for n, _ in secs], fontsize=7, color=INK)
    a1.set_xlim(0, max(v for _, v in secs) * 1.25)
    a1.set_xlabel("raw bytes before zlib", fontsize=8, color=MUTED)
    a1.set_title(f"map payload {st['raw_bytes']:,} B raw -> {st['compressed_bytes']:,} B compressed = {st['n_chunks']} chunks of 240 B "
                 f"(+{st['n_parity']} parity)\nsigned manifest {st['manifest_bytes']} B in {st['n_manifest_frames']} frames; "
                 f"{st['payload_on_air_bytes']:,} B on air per carousel loop", fontsize=7.5, color=INK, loc="left")
    cols = {"no parity": "#6b7785", "XOR parity 1/8": "#0b6e69"}
    hi = max(max(x["times"]) for x in sims.values())
    for name, r in sims.items():
        a2.hist(r["times"], bins=np.arange(0, hi + 2, 1.0), alpha=0.65, color=cols[name],
                label=f"{name}: mean {r['mean_s']:.1f} s, p95 {r['p95_s']:.1f} s")
    a2.set_xlabel("seconds to a complete, verified map (20 frames/s, 10 % loss)", fontsize=8, color=MUTED)
    a2.set_ylabel("robots (trials)", fontsize=8, color=MUTED)
    a2.legend(fontsize=7, frameon=False)
    for ax in (a1, a2):
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ble_report(m: M.LayeredMap, scn: Scenario, trials: int = 100) -> tuple[dict, dict]:
    from intercycle.codec import maplayer as ML
    from intercycle.codec.sign import demo_key, keyring_from
    key = demo_key(scn.key_id)
    kr = keyring_from({scn.key_id: key})
    out, sims = {}, {}
    for name, pg in (("no parity", 0), ("XOR parity 1/8", 8)):
        b = ML.pack_map(m, key, version=1, parity_group=pg)
        sim = ML.simulate_assembly(b, kr, scn.site_id, loss=0.10, fps=20.0, trials=trials, seed=1)
        out[name] = dict(stats=b.stats, sim={k: v for k, v in sim.items() if k != "times"})
        sims[name] = sim
    return out, sims


def render_all(m: M.LayeredMap, scn: Scenario, outdir: Path, trials: int = 100) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    fig_l1(m, scn, outdir / "L1_machine_envelopes.png")
    fig_traffic(m, scn, M.VEHICLE, outdir / "L2_vehicle_flows.png", "L2 vehicle flows: AGVs, tuggers, forklifts (running, all day); arrows = dominant heading", "vehicle")
    fig_traffic(m, scn, M.PERSON, outdir / "L3_people_flows.png", "L3 people flows (running, all day); arrows = dominant heading", "person")
    fig_slices(m, scn, M.PERSON, "person", outdir / "L3_people_slices.png", "people")
    fig_slices(m, scn, M.VEHICLE, "vehicle", outdir / "L2_vehicle_slices.png", "vehicles")
    fig_phase_bins(m, "S30", outdir / "S30_phase_bins.png")
    report, sims = ble_report(m, scn, trials)
    fig_assembly(report, sims, outdir / "map_size_and_ble_assembly.png")
    summary = {"map": {"ticks": m.n_ticks, "hours_observed": m.n_ticks / HZ / 3600, "observed_slices": {
        M.CLASS_NAMES[c]: int(sum(m._observed(t, p, c) for t in range(M.N_TOD) for p in range(M.N_PROD))) for c in (M.PERSON, M.VEHICLE)}},
        "ble": report}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--hours", type=float, default=6.0, help="window length")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="three short windows around two breaks and a shift change")
    ap.add_argument("--render-only", action="store_true", help="reuse out/layered_map.npz")
    ap.add_argument("--trials", type=int, default=100, help="BLE assembly Monte Carlo trials")
    a = ap.parse_args()
    scn = load()
    OUT.mkdir(exist_ok=True)
    npz = OUT / "layered_map.npz"
    if a.render_only:
        m = M.LayeredMap.load(npz)
    else:
        windows = QUICK if a.quick else plan_windows(a.days, a.hours, a.seed)
        m = build(windows, a.workers, OUT / "map" / "tmp")
        m.save(npz)
        print(f"saved {npz} ({npz.stat().st_size / 1e6:.1f} MB)")
    s = render_all(m, scn, OUT / "map", a.trials)
    print(json.dumps(s, indent=2, default=float))


if __name__ == "__main__":
    main()
