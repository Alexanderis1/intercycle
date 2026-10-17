"""Map quality on HELD-OUT time: the learned layers scored against ground truth the map was never built from.

Scoring script (may import truth; the observer code it scores never does). Two steps:

1. `record_heldout(outdir)` simulates held-out windows (day 7 of seed 0 by default; the maps are built from days 0-6)
   with the camera shim and the observer's own `ConditionTracker` (it runs warm: each window starts 20 min early and the
   warm-up is not scored). Every tick it stores what the robot would index the map with (time-of-day bin, discovered
   production state, per-machine phase bin and phase) next to the truth:
     * L1, every 2 s: the true cells of every machine's moving parts at that instant on the machine's 0.25 m local grid,
       twice: "centre" (cell centre inside a body circle, the map's own rasterisation) and "overlap" (any part of the
       cell touched, the strict safety definition);
     * L1, every tick: the true swept cells (union of the 3 sub-frames of the last 0.5 s) counted per tracker phase bin,
       i.e. the true swept area of each (machine, phase bin);
     * L2 / L3, every 10 s: the 1 m site cells truly touched by a vehicle / a person.
2. `score(map, records)` scores any `MapView` (LayeredMap or the robot's DecodedMap):
     * L1 per machine x phase bin: IoU / precision / recall of the envelope (p >= 5 %) against the true swept area;
     * L1 per instant: unsafe-free rate = truly occupied cells predicted free / truly occupied cells, and the cost of
       safety = false-occupied area (predicted occupied, truly free), for several predictors: the single tracker bin
       envelope, the broadcast box cover the robot receives (bins widened as `predict_obstacles` does), the old inner
       box cover, and the box cover inflated by 0..k cells (the margin trade-off curve), at horizon 0 and 10 s;
     * L2 / L3 per time-of-day slice x production state: Brier score, AUROC of occupancy, and the climatology
       (one pooled map for all times) Brier for a skill score. UNKNOWN cells count as p = 1.

Everything is SYNTHETIC; the camera shim's noise / dropout / latency are assumptions until workstream V calibrates them.
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from intercycle.observer import mapping as M
from intercycle.scenario import Scenario, load

ROOT = Path(__file__).resolve().parents[2]
H, DAY = 3600.0, 86_400.0
WARM_S = 1200.0
L1_EVERY, L23_EVERY = 4, 20                                   # ticks (2 Hz): 2 s and 10 s
HELDOUT = [(7 * DAY + 8.25 * H, 7 * DAY + 10.25 * H),         # running + break 09:00 (+ andon stops)
           (7 * DAY + 11.0 * H, 7 * DAY + 12.75 * H),         # break 11:30-12:00
           (7 * DAY + 13.4 * H, 7 * DAY + 15.0 * H),          # shift change 14:00
           (7 * DAY + 2.6 * H, 7 * DAY + 4.5 * H)]            # night, break 03:30
NB = 256                                                       # probability histogram bins (AUROC)


def _scripts():
    p = str(ROOT / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import build_map as B
    return B


# ------------------------------------------------------------------------------------------------ grids

class Grids:
    """Stacked local L1 grids (one bin) of the scenario machines + the 1 m site grid."""

    def __init__(self, spec: dict, mids: list[str] | None = None):
        self.spec = spec
        ms = [d for d in spec["machines"] if mids is None or d["id"] in mids]
        self.mids = [d["id"] for d in ms]
        self.x0 = np.array([d["x0"] for d in ms])
        self.z0 = np.array([d["z0"] for d in ms])
        self.nx = np.array([d["nx"] for d in ms], np.int64)
        self.nz = np.array([d["nz"] for d in ms], np.int64)
        self.size = self.nx * self.nz
        self.base = np.r_[0, np.cumsum(self.size)[:-1]].astype(np.int64)
        self.total = int(self.size.sum())
        self.res = spec["env_res"]
        self.site = spec["site"]

    def local_cells(self, circles: np.ndarray, i: int, centre: bool) -> np.ndarray:
        c = np.asarray(circles, float).reshape(-1, 3)
        if not len(c):
            return np.zeros(0, np.int64)
        _, ix, iz = M.raster(c[:, 0], c[:, 1], c[:, 2], self.x0[i], self.z0[i], self.nx[i], self.nz[i], self.res, centre=centre)
        return np.unique(ix * self.nz[i] + iz)

    def site_cells(self, circles: np.ndarray) -> np.ndarray:
        c = np.asarray(circles, float).reshape(-1, 3)
        if not len(c):
            return np.zeros(0, np.int64)
        s = self.site
        _, ix, iz = M.raster(c[:, 0], c[:, 1], c[:, 2], s["x0"], s["z0"], s["nx"], s["nz"], s["res"], centre=False)
        return np.unique(ix * s["nz"] + iz)


def truth_parts(states_last: M.SimState, mids: list[str], shim: M.CameraShim) -> dict[str, np.ndarray]:
    """True moving-part circles per scenario machine at one instant (skids assigned to the zone they are in)."""
    sk = np.asarray(states_last.skids, float).reshape(-1, 3)
    lab = shim.machine_of(sk[:, 0], sk[:, 1]) if len(sk) else np.zeros(0, int)
    out = {}
    for k, mid in enumerate(mids):
        c = np.asarray(states_last.parts.get(mid, np.zeros((0, 3))), float).reshape(-1, 3)
        if len(sk) and (lab == k).any():
            c = np.vstack([c, sk[lab == k]])
        out[mid] = c
    return out


# ------------------------------------------------------------------------------------------------ recorder

class Recorder:
    """tick_hook for build_map.run_window: stores conditioning + truth for scoring (see module doc)."""

    def __init__(self, scn: Scenario, t_rec0: float):
        self.scn, self.t_rec0 = scn, t_rec0
        self.spec = M.build_spec(scn)
        self.g = Grids(self.spec)
        self.mids = self.g.mids
        self.shim = M.CameraShim(scn)
        self.Mn = len(self.mids)
        self.l1_t, self.l1_tod, self.l1_prod, self.l1_bin, self.l1_phase = [], [], [], [], []
        self.e_tick, self.e_mach, self.e_cell, self.e_centre = [], [], [], []
        self.sw_cnt = np.zeros((M.N_BINS, self.g.total), np.int32)
        self.sw_n = np.zeros((self.Mn, M.N_BINS), np.int64)
        self.l23_t, self.l23_tod, self.l23_prod = [], [], []
        self.p_tick, self.p_cell, self.v_tick, self.v_cell = [], [], [], []

    def __call__(self, k: int, tc: float, states, obs, cond) -> None:
        if tc < self.t_rec0:
            return
        g = self.g
        # ---- swept area per tracker bin (every tick, union of sub-frames, centre rasterisation)
        sub = [truth_parts(s, self.mids, self.shim) for s in states]
        for i, mid in enumerate(self.mids):
            b = int(cond.phase_bin[i])
            if b < 0:
                continue
            self.sw_n[i, b] += 1
            cells = np.unique(np.concatenate([g.local_cells(sp[mid], i, True) for sp in sub]))
            if len(cells):
                self.sw_cnt[b, g.base[i] + cells] += 1
        # ---- instants (every 2 s)
        if k % L1_EVERY == 0:
            n = len(self.l1_t)
            self.l1_t.append(tc)
            self.l1_tod.append(cond.tod_bin)
            self.l1_prod.append(cond.prod_state)
            self.l1_bin.append(cond.phase_bin.copy())
            self.l1_phase.append(cond.phase.copy())
            last = sub[-1]
            for i, mid in enumerate(self.mids):
                ov = g.local_cells(last[mid], i, False)
                if not len(ov):
                    continue
                ce = g.local_cells(last[mid], i, True)
                self.e_tick.append(np.full(len(ov), n, np.int32))
                self.e_mach.append(np.full(len(ov), i, np.int8))
                self.e_cell.append(ov.astype(np.int32))
                self.e_centre.append(np.isin(ov, ce))
        # ---- traffic (every 10 s)
        if k % L23_EVERY == 0:
            n = len(self.l23_t)
            self.l23_t.append(tc)
            self.l23_tod.append(cond.tod_bin)
            self.l23_prod.append(cond.prod_state)
            st = states[-1]
            for cls, tl, cl in ((M.PERSON, self.p_tick, self.p_cell), (M.VEHICLE, self.v_tick, self.v_cell)):
                sel = st.mover_cls == cls
                cells = g.site_cells(st.mover_circles[sel])
                tl.append(np.full(len(cells), n, np.int32))
                cl.append(cells.astype(np.int32))

    def save(self, path: Path) -> None:
        cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dt)
        np.savez_compressed(path, l1_t=np.array(self.l1_t), l1_tod=np.array(self.l1_tod, np.int16), l1_prod=np.array(self.l1_prod, np.int8),
                            l1_bin=np.array(self.l1_bin, np.int16).reshape(-1, self.Mn), l1_phase=np.array(self.l1_phase, float).reshape(-1, self.Mn),
                            e_tick=cat(self.e_tick, np.int32), e_mach=cat(self.e_mach, np.int8), e_cell=cat(self.e_cell, np.int32),
                            e_centre=cat(self.e_centre, bool), sw_cnt=self.sw_cnt, sw_n=self.sw_n,
                            l23_t=np.array(self.l23_t), l23_tod=np.array(self.l23_tod, np.int16), l23_prod=np.array(self.l23_prod, np.int8),
                            p_tick=cat(self.p_tick, np.int32), p_cell=cat(self.p_cell, np.int32),
                            v_tick=cat(self.v_tick, np.int32), v_cell=cat(self.v_cell, np.int32),
                            mids=np.array(self.mids))


def _record_worker(args):
    seed, t0, t1, path, scn_path = args
    B = _scripts()
    scn = load(scn_path) if scn_path else load()
    rec = Recorder(scn, t0)
    w0 = time.time()
    B.run_window(seed, t0 - WARM_S, t1, scn=scn, tick_hook=rec, build=False)
    rec.save(Path(path))
    return path, time.time() - w0


def record_heldout(outdir: Path, windows=HELDOUT, seed: int = 0, workers: int = 4, scn_path: str | None = None) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, a, b, str(outdir / f"heldout_{k}.npz"), scn_path) for k, (a, b) in enumerate(windows)]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        for p, dt in ex.map(_record_worker, jobs):
            print(f"  held-out record {Path(p).name} in {dt:.0f} s", flush=True)
    (outdir / "heldout.json").write_text(json.dumps(dict(seed=seed, windows=windows, warm_s=WARM_S,
                                                         l1_every_s=L1_EVERY / M.FRAME_HZ, l23_every_s=L23_EVERY / M.FRAME_HZ)))
    return [Path(j[3]) for j in jobs]


def load_records(outdir: Path) -> list[dict]:
    return [dict(np.load(f, allow_pickle=False)) for f in sorted(Path(outdir).glob("heldout_*.npz"))]


# ------------------------------------------------------------------------------------------------ predictors

class L1Pred:
    """Flat boolean local masks of a map's L1 for bin tuples, per predictor kind, with caching."""

    def __init__(self, mp: M.MapView, g: Grids):
        self.mp, self.g = mp, g
        self.idx = [mp.machine_ids().index(mid) for mid in g.mids]
        self._env: dict = {}
        self._c: dict = {}

    def env(self, i: int, bins: tuple) -> np.ndarray:
        key = ("env", i, bins)
        if key not in self._c:
            mid = self.g.mids[i]
            if hasattr(self.mp, "envelope_union"):
                e = self.mp.envelope_union(mid, bins, 0.05)
            else:                                              # decoded map: only the boxes exist
                e = self.boxes_mask(i, bins)
            self._c[key] = e
        return self._c[key]

    def boxes_mask(self, i: int, bins: tuple, kind: str = "safe") -> np.ndarray:
        key = (kind, i, bins)
        if key not in self._c:
            g, mid = self.g, self.g.mids[i]
            out = np.zeros((g.nx[i], g.nz[i]), bool)
            if hasattr(self.mp, "envelope_union"):
                cb = self.mp.box_cover(mid, bins, 0.05, max_boxes=12, world=False, safe=(kind == "safe"))
                for i0, j0, i1, j1 in cb:
                    out[i0:i1, j0:j1] = True
            else:
                res = g.res
                for x0, z0, x1, z1 in self.mp.boxes_for(mid, bins):
                    i0, i1 = int(round((x0 - g.x0[i]) / res)), int(round((x1 - g.x0[i]) / res))
                    j0, j1 = int(round((z0 - g.z0[i]) / res)), int(round((z1 - g.z0[i]) / res))
                    out[max(i0, 0):i1, max(j0, 0):j1] = True
            self._c[key] = out
        return self._c[key]

    def mask(self, pred: str, i: int, bins: tuple, dil: int = 0) -> np.ndarray:
        key = ("m", pred, i, bins, dil)
        if key not in self._c:
            base = self.env(i, bins) if pred == "env" else self.boxes_mask(i, bins, "safe" if pred == "box" else "inner")
            self._c[key] = M.dilate(base, dil).reshape(-1) if dil else base.reshape(-1)
        return self._c[key]


def _bins_for(mp: M.MapView, mi: int, rec: dict, n: int, i: int, mode: str, horizon: float) -> tuple:
    if mode == "bin":                                          # the tracker's bin as is (raw envelope quality)
        b = int(rec["l1_bin"][n, i])
        return M.ALL_BINS if b < 0 else (b,)
    ph = rec["l1_phase"][n, i]
    b = int(rec["l1_bin"][n, i])
    if b == M.STOP_BIN:
        ph = -1.0
    return mp.predicted_bins(mi, None if not np.isfinite(ph) else float(ph), int(rec["l1_prod"][n]), horizon)


def score_l1(mp: M.MapView, recs: list[dict], preds=(("env", "bin", 0.0, 0), ("box", "robot", 0.0, 0)),
             per_machine: bool = True) -> dict:
    """Unsafe-free / false-occupied for each predictor (pred kind, bin mode, horizon s, dilation cells)."""
    g = Grids(mp.spec, [d for d in recs[0]["mids"].tolist()])
    P = L1Pred(mp, g)
    out = {}
    cell_m2 = g.res ** 2
    for pk, mode, hz, dil in preds:
        name = f"{pk}|{mode}|h{hz:g}|d{dil}"
        tp = np.zeros((len(g.mids), 2))     # [machine, (centre, overlap)]
        fn = np.zeros((len(g.mids), 2))
        fp = np.zeros((len(g.mids), 2))
        n_inst = np.zeros(len(g.mids))
        unknown_ticks = np.zeros(len(g.mids))
        tp_p = np.zeros((M.N_PROD, 2))      # by the observer's production state at the scored instant
        fn_p = np.zeros((M.N_PROD, 2))
        for rec in recs:
            nt = len(rec["l1_t"])
            hstep = int(round(hz * M.FRAME_HZ / L1_EVERY))
            if nt - hstep <= 0:
                continue
            src = np.arange(nt - hstep)                         # prediction made at tick n, truth at n + hstep
            gid = np.zeros((nt, len(g.mids)), np.int64) - 1
            groups: dict = {}
            masks = []
            for n in src:
                for i in range(len(g.mids)):
                    bins = _bins_for(mp, P.idx[i], rec, int(n), i, mode, hz)
                    key = (i, bins)
                    if key not in groups:
                        groups[key] = len(groups)
                        masks.append(P.mask(pk, i, bins, dil))
                    gid[n + hstep, i] = groups[key]
                    if bins == M.ALL_BINS:
                        unknown_ticks[i] += 1
            off = np.r_[0, np.cumsum([len(m) for m in masks])[:-1]].astype(np.int64)
            cat = np.concatenate(masks)
            area = np.array([m.sum() for m in masks], float)
            et, em, ec, ecen = rec["e_tick"], rec["e_mach"].astype(np.int64), rec["e_cell"].astype(np.int64), rec["e_centre"]
            eg = gid[et, em]
            ok = eg >= 0
            hit = np.zeros(len(et), bool)
            hit[ok] = cat[off[eg[ok]] + ec[ok]]
            eprod = rec["l1_prod"][et].astype(np.int64)
            for d, sel in ((0, ok & ecen), (1, ok)):
                tp[:, d] += np.bincount(em[sel & hit], minlength=len(g.mids))
                fn[:, d] += np.bincount(em[sel & ~hit], minlength=len(g.mids))
                tp_p[:, d] += np.bincount(eprod[sel & hit], minlength=M.N_PROD)
                fn_p[:, d] += np.bincount(eprod[sel & ~hit], minlength=M.N_PROD)
            # predicted area per (tick, machine)
            valid = gid >= 0
            a_tm = np.where(valid, area[np.maximum(gid, 0)], 0.0)
            n_inst += valid.sum(0)
            # false occupied = predicted area - true positives (per truth definition)
            for d, sel in ((0, ok & ecen), (1, ok)):
                fp[:, d] += a_tm.sum(0)
                fp[:, d] -= np.bincount(em[sel & hit], minlength=len(g.mids))
        r = {}
        for d, dn in ((0, "centre"), (1, "overlap")):
            T = tp[:, d] + fn[:, d]
            r[f"unsafe_free_{dn}"] = float(fn[:, d].sum() / max(T.sum(), 1))
            r[f"false_occupied_share_{dn}"] = float(fp[:, d].sum() / max((tp[:, d] + fp[:, d]).sum(), 1))
            r[f"false_occupied_m2_per_machine_{dn}"] = float(fp[:, d].sum() * cell_m2 / max(n_inst.sum(), 1))
            if per_machine:
                r[f"per_machine_unsafe_free_{dn}"] = {mid: float(fn[i, d] / max(T[i], 1)) for i, mid in enumerate(g.mids)}
                r[f"by_prod_unsafe_free_{dn}"] = {M.PROD_NAMES[k]: (float(fn_p[k, d] / (tp_p[k, d] + fn_p[k, d])) if tp_p[k, d] + fn_p[k, d] else None)
                                                  for k in range(M.N_PROD)}
                r[f"by_prod_occupied_{dn}"] = {M.PROD_NAMES[k]: int(tp_p[k, d] + fn_p[k, d]) for k in range(M.N_PROD)}
        r["unknown_share"] = float(unknown_ticks.sum() / max(n_inst.sum(), 1))
        r["instants"] = int(n_inst.sum())
        out[name] = r
    return out


def score_bins(mp: M.MapView, recs: list[dict]) -> dict:
    """L1 per machine x phase bin: IoU / precision / recall of the envelope vs the true swept area (tracker bins)."""
    g = Grids(mp.spec, recs[0]["mids"].tolist())
    P = L1Pred(mp, g)
    cnt = sum(r["sw_cnt"] for r in recs)
    nn = sum(r["sw_n"] for r in recs)
    rows = []
    for i, mid in enumerate(g.mids):
        sl = slice(g.base[i], g.base[i] + g.size[i])
        for b in range(M.N_BINS):
            if nn[i, b] < 20:
                continue
            e = P.env(i, (b,)).reshape(-1)
            c = cnt[b, sl]
            for tname, t in (("any", c > 0), ("p05", c >= 0.05 * nn[i, b])):
                inter, uni = (e & t).sum(), (e | t).sum()
                if uni == 0:                                   # nothing on the floor in this bin, learned and true: agreed
                    rows.append(dict(machine=mid, bin=b, truth=tname, n=int(nn[i, b]), empty=True, iou=1.0, precision=1.0,
                                     recall=1.0, area_m2=0.0, truth_m2=0.0))
                    continue
                rows.append(dict(machine=mid, bin=b, truth=tname, n=int(nn[i, b]), iou=float(inter / max(uni, 1)),
                                 precision=float(inter / max(e.sum(), 1)), recall=float(inter / max(t.sum(), 1)),
                                 area_m2=float(e.sum() * g.res ** 2), truth_m2=float(t.sum() * g.res ** 2)))
    agg = {}
    for tname in ("any", "p05"):
        rr = [r for r in rows if r["truth"] == tname and not r.get("empty")]      # empty-empty bins excluded from the means
        if rr:
            agg[tname] = {k: float(np.mean([r[k] for r in rr])) for k in ("iou", "precision", "recall")}
    return dict(rows=rows, mean=agg)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC from histograms of scores (ties count half)."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cn = np.cumsum(neg) - neg                                   # negatives strictly below each bin
    return float(((pos * (cn + 0.5 * neg)).sum()) / (P * N))


def score_traffic(mp: M.MapView, recs: list[dict]) -> dict:
    """L2 / L3 Brier + AUROC per (time-of-day slice, production state) and overall, with a climatology baseline."""
    s = mp.spec["site"]
    ncell = s["nx"] * s["nz"]
    res = {}
    for cname, cid, tk, ck in (("vehicle", M.VEHICLE, "v_tick", "v_cell"), ("person", M.PERSON, "p_tick", "p_cell")):
        # climatology: every observed slice pooled
        if hasattr(mp, "hits"):
            hit = np.zeros((mp.nx, mp.nz))
            vis = np.zeros((mp.nx, mp.nz))
            for S in range(M.N_TOD * M.N_PROD):
                if mp.visits_cam[S].max() >= mp.min_ticks:
                    hit += mp.hits[cid][S]
                    vis += mp.visits_cam[S][mp.cam_of_cell]
            clim = np.where(vis > 0, hit / np.maximum(vis, 1), 0.0).reshape(-1)
        else:
            clim = None
        cache = {}
        acc = {}
        for rec in recs:
            ticks, cells = rec[tk], rec[ck]
            order = np.argsort(ticks, kind="stable")
            ticks, cells = ticks[order], cells[order]
            ptr = np.searchsorted(ticks, np.arange(len(rec["l23_t"]) + 1))
            for n in range(len(rec["l23_t"])):
                key = (int(rec["l23_tod"][n]), int(rec["l23_prod"][n]))
                if key not in cache:
                    p, _ = mp.traffic(cid, key[0], key[1])
                    U = mp.unknown_mask(key[0], key[1])
                    known = ~U.reshape(-1)
                    pk = p.reshape(-1).astype(float)
                    p = np.where(U, 1.0, p).reshape(-1).astype(float)
                    cache[key] = (p, np.histogram(p, NB, (0, 1))[0], float((p * p).sum()), known, float((pk[known] ** 2).sum()), pk)
                p, hall, p2, known, pk2, pk = cache[key]
                y = cells[ptr[n]:ptr[n + 1]]
                a = acc.setdefault(key, dict(n=0, b=0.0, bc=0.0, bk=0.0, nk=0, pos=np.zeros(NB), all=np.zeros(NB), npos=0))
                a["n"] += 1
                a["b"] += p2 + (1 - 2 * p[y]).sum()
                yk = y[known[y]]
                a["bk"] += pk2 + (1 - 2 * pk[yk]).sum()
                a["nk"] += int(known.sum())
                if clim is not None:
                    a["bc"] += float((clim * clim).sum()) + (1 - 2 * clim[y]).sum()
                a["pos"] += np.histogram(p[y], NB, (0, 1))[0]
                a["all"] += hall
                a["npos"] += len(y)
        rows = []
        for (tod, prod), a in sorted(acc.items()):
            N = a["n"] * ncell
            rows.append(dict(tod=tod, prod=M.PROD_NAMES[prod], samples=a["n"], brier=a["b"] / N,
                             brier_known=a["bk"] / max(a["nk"], 1), unknown_share=1 - a["nk"] / N,
                             brier_clim=a["bc"] / N if clim is not None else float("nan"),
                             auroc=_auc(a["pos"], a["all"] - a["pos"]), occupied_share=a["npos"] / N))
        tot = dict(n=sum(a["n"] for a in acc.values()), b=sum(a["b"] for a in acc.values()), bc=sum(a["bc"] for a in acc.values()),
                   bk=sum(a["bk"] for a in acc.values()), nk=sum(a["nk"] for a in acc.values()),
                   pos=sum(a["pos"] for a in acc.values()), all=sum(a["all"] for a in acc.values()))
        N = tot["n"] * ncell
        by_prod = {}
        for pn in M.PROD_NAMES:
            rr = [(k, a) for k, a in acc.items() if M.PROD_NAMES[k[1]] == pn]
            if rr:
                n_ = sum(a["n"] for _, a in rr) * ncell
                pos = sum(a["pos"] for _, a in rr)
                al = sum(a["all"] for _, a in rr)
                by_prod[pn] = dict(brier=sum(a["b"] for _, a in rr) / n_,
                                   brier_known=sum(a["bk"] for _, a in rr) / max(sum(a["nk"] for _, a in rr), 1), brier_clim=sum(a["bc"] for _, a in rr) / n_,
                                   auroc=_auc(pos, al - pos), samples=sum(a["n"] for _, a in rr))
        res[cname] = dict(brier=tot["b"] / max(N, 1), brier_known=tot["bk"] / max(tot["nk"], 1), unknown_share=1 - tot["nk"] / max(N, 1),
                          brier_clim=tot["bc"] / max(N, 1) if clim is not None else float("nan"),
                          auroc=_auc(tot["pos"], tot["all"] - tot["pos"]), by_prod=by_prod, rows=rows)
        if clim is not None and res[cname]["brier_clim"] > 0:
            res[cname]["brier_skill_vs_clim"] = 1 - res[cname]["brier"] / res[cname]["brier_clim"]
    return res


HEADLINE_PREDS = (("env", "bin", 0.0, 0), ("box", "robot", 0.0, 0), ("box", "robot", 10.0, 0))


def score(mp: M.MapView, recs: list[dict], preds=HEADLINE_PREDS, traffic: bool = True, bins: bool = True) -> dict:
    out = {"l1": score_l1(mp, recs, preds)}
    if bins:
        out["l1_bins"] = score_bins(mp, recs)
    if traffic:
        out["traffic"] = score_traffic(mp, recs)
    return out


def headline(sc: dict) -> dict:
    """Small dict of the numbers that go into stream snapshots / convergence curves."""
    r = sc["l1"]
    rob = r.get("box|robot|h0|d0", {})
    h = dict(unsafe_free=rob.get("unsafe_free_overlap", math.nan), unsafe_free_centre=rob.get("unsafe_free_centre", math.nan),
             false_occupied_share=rob.get("false_occupied_share_overlap", math.nan),
             env_unsafe_free=r.get("env|bin|h0|d0", {}).get("unsafe_free_centre", math.nan),
             unknown_share=rob.get("unknown_share", math.nan))
    if "box|robot|h10|d0" in r:
        h["unsafe_free_h10"] = r["box|robot|h10|d0"]["unsafe_free_overlap"]
    if "l1_bins" in sc:
        h["l1_iou"] = sc["l1_bins"]["mean"].get("p05", {}).get("iou", math.nan)
        h["l1_recall"] = sc["l1_bins"]["mean"].get("p05", {}).get("recall", math.nan)
    if "traffic" in sc:
        for c, k in (("vehicle", "l2"), ("person", "l3")):
            h[f"{k}_brier"] = sc["traffic"][c]["brier"]
            h[f"{k}_brier_known"] = sc["traffic"][c]["brier_known"]
            h[f"{k}_unknown_share"] = sc["traffic"][c]["unknown_share"]
            h[f"{k}_auroc"] = sc["traffic"][c]["auroc"]
    return h


# ------------------------------------------------------------------------------------------------ item 2: missed coverage

def missed_analysis(mp: M.MapView, recs: list[dict], max_d: int = 6) -> dict:
    """Where do truly occupied cells predicted free come from? For the raw single-bin envelope (centre and overlap
    truth) every missed cell is classified as
      * timing   - inside the envelope of a neighbouring phase bin (+-1): the phase estimate was a bin off;
      * spatial  - outside the union of ALL bins: the envelope itself is too small there (rare poses, p < 5 % cells,
                   the 0.1 m noise shrink, rasterisation);
      * other    - inside the all-bin union but not the neighbouring bins (e.g. a machine in its STOP bin moving,
                   or a bin with too few observations),
    and its distance (cells) to the nearest predicted cell is histogrammed (how much inflation would close it)."""
    g = Grids(mp.spec, recs[0]["mids"].tolist())
    P = L1Pred(mp, g)
    out = {}
    for dn, use_centre in (("centre", True), ("overlap", False)):
        cls = dict(timing=0, spatial=0, other=0)
        dist = np.zeros(max_d + 2, np.int64)
        total = 0
        for rec in recs:
            et, em, ec, ecen = rec["e_tick"], rec["e_mach"].astype(np.int64), rec["e_cell"].astype(np.int64), rec["e_centre"]
            sel = ecen if use_centre else np.ones(len(et), bool)
            total += int(sel.sum())
            bins = rec["l1_bin"]
            for i in range(len(g.mids)):
                si = sel & (em == i)
                if not si.any():
                    continue
                for b in np.unique(bins[et[si], i]):
                    bb = int(b)
                    sb = si & (bins[et, i] == b)
                    tup = M.ALL_BINS if bb < 0 else (bb,)
                    pm = P.mask("env", i, tup)
                    miss = sb.copy()
                    miss[sb] = ~pm[ec[sb]]
                    if not miss.any():
                        continue
                    cells = ec[miss]
                    allm = P.mask("env", i, M.ALL_BINS)
                    nb = M.ALL_BINS if bb < 0 or bb == M.STOP_BIN else tuple(sorted({(bb - 1) % M.N_PHASE, bb, (bb + 1) % M.N_PHASE}))
                    nbm = P.mask("env", i, nb)
                    t_ = nbm[cells]
                    s_ = ~allm[cells]
                    cls["timing"] += int(t_.sum())
                    cls["spatial"] += int(s_.sum())
                    cls["other"] += int((~t_ & ~s_).sum())
                    # distance to the predicted set
                    base = P.env(i, tup)
                    left = np.ones(len(cells), bool)
                    for d in range(1, max_d + 1):
                        dm = M.dilate(base, d).reshape(-1)
                        hit = left & dm[cells]
                        dist[d] += int(hit.sum())
                        left &= ~hit
                    dist[max_d + 1] += int(left.sum())
        n_miss = sum(cls.values())
        out[dn] = dict(truly_occupied=total, missed=n_miss, missed_rate=n_miss / max(total, 1),
                       by_cause={k: v / max(n_miss, 1) for k, v in cls.items()},
                       distance_cells={("> %d" % max_d if d == max_d + 1 else str(d)): int(dist[d]) for d in range(1, max_d + 2)})
    return out


def part_speeds(scn: Scenario | None = None, seed: int = 0, t0: float = 7 * DAY + 10 * H, dur: float = 600.0, dt: float = 0.05) -> dict:
    """Speed of the fastest moving body circle per machine (truth kinematics, for K in S = K T + C): p50 / p99 / max m/s."""
    from intercycle.sim.world import World
    from intercycle.truth.kinematics import Kinematics
    scn = scn or load()
    world = World(scn, seed, t0, t0 + dur)
    kin = Kinematics(scn, world)
    mids = [m.id for m in scn.machines]
    prev = None
    sp = {mid: [] for mid in mids}
    for t in np.arange(t0, t0 + dur, dt):
        bd = kin.bodies(t)
        cur = {mid: np.asarray(bd[mid]).copy() for mid in mids}
        if prev is not None:
            for mid in mids:
                a, b = prev[mid], cur[mid]
                if len(a) and len(a) == len(b):
                    sp[mid].append(float(np.hypot(*(b[:, :2] - a[:, :2]).T).max() / dt))
        prev = cur
    out = {}
    for mid, v in sp.items():
        v = np.array(v) if v else np.zeros(1)
        mv = v[v > 0.01]
        out[mid] = dict(p50=float(np.median(mv)) if len(mv) else 0.0, p99=float(np.percentile(mv, 99)) if len(mv) else 0.0, max=float(v.max()))
    return out
