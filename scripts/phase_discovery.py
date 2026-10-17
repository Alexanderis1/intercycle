"""Unsupervised phase discovery: simulate floor windows, discover machine sub-phases and line states from camera
observations only (intercycle/observer/phases.py), and score them against ground truth (truth is used HERE only).

    python scripts/phase_discovery.py sim  --name dev --t0 8.5 --t1 10.0 [--seed 0]   # hours of day 0 -> out/phases/<name>.npz
    python scripts/phase_discovery.py eval --train dev_train --test dev_test            # fit on train, score on test

Truth sub-phases (scoring only):
  weld_station / respot / framing (S10-S70): active fraction f of `Kinematics.phase` with T_WELD_END = 0.80,
      T_DRESS_END = 0.92 and the docstring's 0.86 swing/dress split: weld < 0.80 <= swing_out < 0.86 <= dress < 0.92
      <= return; idle otherwise. Coarse set: weld / tip_dress (0.80-0.92) / return / idle.
  unload_lifter (S80): `lift_height` breakpoints: lower 0.10-0.30, down 0.30-0.55, raise 0.55-0.80; the carriage is
      overhead (above H_CLEAR = 2.2 m, no floor footprint) for f < 0.10 and f >= 0.80, merged with idle as "clear".
  roller_hemming / sealer (HEM1, SEAL1): `_timeline_path`: approach (to the first path point, arm spec brk[1]),
      path until 0.92, return 0.92-1.0; idle otherwise.
Truth line state: calendar break / shift_change; andon = the line index overdue (t in [index + takt, next index) of an
andon cycle); running otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer import phases as P  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.sim.world import World  # noqa: E402
from intercycle.truth import kinematics as K  # noqa: E402
from intercycle.truth.machines import line_index_times  # noqa: E402

OUT = ROOT / "out" / "phases"
H = 3600.0
HZ = M.FRAME_HZ

FINE = {
    "weld": ("weld", "swing_out", "dress", "return", "idle"),
    "lift": ("lower", "down", "raise", "clear"),
    "cell": ("approach", "path", "return", "idle"),
}
COARSE_WELD = {"weld": "weld", "swing_out": "tip_dress", "dress": "tip_dress", "return": "return", "idle": "idle"}


def family(kind: str) -> str:
    return "weld" if kind in K.KIND else "lift" if kind == "unload_lifter" else "cell"


def truth_label(fam: str, state: str, f: float, approach_end: float) -> str:
    if state != "active":
        return "clear" if fam == "lift" else "idle"
    if fam == "weld":
        return "weld" if f < K.T_WELD_END else "swing_out" if f < 0.86 else "dress" if f < K.T_DRESS_END else "return"
    if fam == "lift":
        return "lower" if 0.10 <= f < 0.30 else "down" if 0.30 <= f < 0.55 else "raise" if 0.55 <= f < 0.80 else "clear"
    return "approach" if f < approach_end else "path" if f < 0.92 else "return"


def truth_line_states(scn, seed, t, t0, t1) -> np.ndarray:
    lab = np.zeros(len(t), np.int8)                                    # running
    for ti, length, kind in line_index_times(scn, seed, t0 - H, t1 + H):
        if kind == "andon":
            lab[(t >= ti + scn.line.takt_s) & (t < ti + length)] = 1
    tod = t % 86400.0
    for a, b, kind in scn.line_stops():
        lab[(tod >= a) & (tod < b)] = 2 if kind == "break" else 3
    return lab


# ================================================================================================ simulation

def simulate(seed: int, t0: float, t1: float, subframes: int = 3, progress: bool = True, shim_kw: dict | None = None) -> dict:
    """Floor at 2 Hz (as scripts/build_map.run_window) -> camera observations -> phases.Features; plus truth labels."""
    import build_map as B
    scn = load()
    world = World(scn, seed, t0, t1)
    kin = K.Kinematics(scn, world)
    mids = [m.id for m in scn.machines]
    fams = [family(m.kind) for m in scn.machines]
    appr = {}
    for mid in mids:
        b = [s.brk[1] for s in kin.arm_specs if s.mid == mid]
        appr[mid] = float(np.mean(b)) if b else 0.0
    movers = B.MoverBank(scn, seed)
    shim = M.CameraShim(scn, seed=seed * 1000 + int(t0 // 3600), subframes=subframes, **(shim_kw or {}))
    fe = P.Features(P.Layout.from_scenario(scn))
    n = int(round((t1 - t0) * HZ))
    T = np.zeros(n)
    X = np.zeros((n, len(mids), P.N_MFEAT), np.float32)
    F = np.zeros((n, len(P.FLOW_NAMES)), np.float32)
    lab = np.empty((n, len(mids)), object)
    fr = np.zeros((n, len(mids)), np.float32)
    tic = time.time()
    for b0 in range(0, n, 1200):
        ks = np.arange(b0, min(n, b0 + 1200))
        t_cap = t0 + ks / HZ - shim.latency
        blk = movers.block(t_cap)
        for j, k in enumerate(ks):
            tc = float(t_cap[j])
            states = []
            for s in range(subframes):
                ts = tc - shim.exposure * (1 - (s + 1) / subframes)
                bd = kin.bodies(ts)
                parts = {mid: bd[mid] for mid in mids}
                if s == subframes - 1:
                    states.append(movers.state(tc, tuple(a[j] for a in blk), parts, bd["SKIDS"], bd["AGVDOCK"]))
                else:
                    states.append(M.SimState(ts, parts=parts, skids=bd["SKIDS"]))
            T[k], X[k], F[k] = fe.extract(shim.observe(states))
            for i, mid in enumerate(mids):
                st, f = kin.phase(mid, tc)
                lab[k, i] = truth_label(fams[i], st, f, appr[mid])
                fr[k, i] = f if st == "active" else -1.0
        if progress:
            print(f"  t={t_cap[-1] / H:.2f} h  {time.time() - tic:.0f} s", flush=True)
    names = sorted({x for f in FINE.values() for x in f})
    code = {s: i for i, s in enumerate(names)}
    labc = np.vectorize(code.get)(lab).astype(np.int8)
    return dict(T=T, X=X, F=F, lab=labc, lab_names=np.array(names), frac=fr, mids=np.array(mids), fams=np.array(fams),
                line=truth_line_states(scn, seed, T, t0, t1), seed=seed, t0=t0, t1=t1)


def cmd_sim(a):
    OUT.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    d = simulate(a.seed, a.t0 * H, a.t1 * H, shim_kw=M.MEASURED_SHIM if a.shim == "measured" else None)
    np.savez_compressed(OUT / f"{a.name}.npz", **d)
    print(f"wrote {OUT / (a.name + '.npz')} ({len(d['T'])} ticks, {time.time() - tic:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--name", required=True)
    s.add_argument("--t0", type=float, required=True, help="hours since 00:00 of day 0")
    s.add_argument("--t1", type=float, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--shim", default="ideal", choices=("ideal", "measured"), help="camera error model (measured = M.MEASURED_SHIM)")
    e = sub.add_parser("eval")
    e.add_argument("--train", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--engine", default=None, help="motif engine: auto | stumpy | numpy")
    tb = sub.add_parser("table", help="markdown tables from a results json")
    tb.add_argument("results")
    a = ap.parse_args()
    {"sim": cmd_sim, "eval": cmd_eval, "table": cmd_table}[a.cmd](a)


def cmd_table(a):
    r = json.load(open(a.results))
    f = lambda v: "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"
    print("| Machine | truth sub-phases | bins | period (s) | **online seg F1** | online acc | online boundary err median (s) "
          "| online bnd F1 ±1 s | offline DTW seg F1 | offline acc | fine-label online F1 | onset err median (s), resting->working "
          "| onset err median (s), v1: active start -> bin 0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mid, m in r["machines"].items():
        on, off, fi, cy = m["online_hmm/coarse"], m["offline_dtw/coarse"], m["online_hmm/fine"], m["online_hmm/cycle"]
        c1 = m.get("online_hmm/cycle_v1", {"onset_med_s": float("nan")})
        labs = "/".join(sorted(on["per_label_f1"]))
        print(f"| {mid} | {labs} | {m['n_bins']} | {m['period_s']:.2f} | **{f(on['seg_f1'])}** | {f(on['acc'])} | "
              f"{f(on['bnd_med_s'])} | {f(on['bnd_f1_1s'])} | {f(off['seg_f1'])} | {f(off['acc'])} | {f(fi['seg_f1'])} | {f(cy['onset_med_s'])} | {f(c1['onset_med_s'])} |")
    print()
    print("| Machine | per-label online segment F1 |")
    print("|---|---|")
    for mid, m in r["machines"].items():
        print(f"| {mid} | " + ", ".join(f"{k} {v:.2f}" for k, v in m["online_hmm/coarse"]["per_label_f1"].items()) + " |")
    L = r["line"]
    print(f"\nLine: ARI {L['ari']}, purity {L['purity']}, named accuracy {L['named_acc']}, k = {L['diag']['k']}")
    print("\n| true \\ predicted | " + " | ".join(L["confusion"]) + " | recall |")
    print("|---|" + "---|" * (len(L["confusion"]) + 1))
    for s, row in L["confusion"].items():
        n = sum(row.values())
        print(f"| {s} | " + " | ".join(str(v) for v in row.values()) + f" | {row[s] / n:.2f} |" if n else f"| {s} | " + " | ".join("0" for _ in row) + " | – |")


# ================================================================================================ scoring

def runs(lab: np.ndarray) -> list[tuple[int, int, object]]:
    """Contiguous runs (start, end exclusive, label)."""
    if not len(lab):
        return []
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    st = np.r_[0, cut]
    en = np.r_[cut, len(lab)]
    return [(int(x), int(y), lab[x]) for x, y in zip(st, en)]


def majority_map(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Many-to-one: every discovered bin is named by the truth label it overlaps most (scoring only)."""
    out = {}
    for b in np.unique(pred):
        v, c = np.unique(truth[pred == b], return_counts=True)
        out[b] = v[c.argmax()]
    return out


def segment_scores(pred_named: np.ndarray, truth: np.ndarray, hz: float, iou_min: float = 0.5, tol_s: float = 1.0) -> dict:
    """Segment F1 (a true segment is found if a predicted segment with the same name overlaps it with IoU >= 0.5,
    one-to-one), frame accuracy, boundary error (s) from every true boundary to the nearest predicted boundary, and
    boundary F1 at +-tol_s."""
    tr, pr = runs(truth), runs(pred_named)
    pr_by = {}
    for j, (a, b, l) in enumerate(pr):
        pr_by.setdefault(l, []).append((a, b, j))
    used, hit = set(), 0
    for a, b, l in tr:
        best, bj = 0.0, None
        for c, d, j in pr_by.get(l, []):
            if d <= a or c >= b or j in used:
                continue
            iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou > best:
                best, bj = iou, j
        if bj is not None and best >= iou_min:
            used.add(bj)
            hit += 1
    P_ = hit / max(len(pr), 1)
    R_ = hit / max(len(tr), 1)
    tb = np.array([a for a, _, _ in tr[1:]], float)
    pb = np.array([a for a, _, _ in pr[1:]], float)
    if len(tb) and len(pb):
        d = _nearest(pb, tb) / hz
        d2 = _nearest(tb, pb) / hz
        bp, br = float((d2 <= tol_s).mean()), float((d <= tol_s).mean())
        bf1 = 2 * bp * br / max(bp + br, 1e-9)
        bmed, bmean = float(np.median(d)), float(np.mean(np.minimum(d, 30.0)))
    else:
        bf1 = bmed = bmean = float("nan")
    return dict(seg_f1=2 * P_ * R_ / max(P_ + R_, 1e-9), seg_p=P_, seg_r=R_, acc=float((pred_named == truth).mean()),
                bnd_med_s=bmed, bnd_mean_s=bmean, bnd_f1_1s=bf1, n_true_seg=len(tr), n_pred_seg=len(pr))


def _nearest(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """|q - nearest element of sorted ref|."""
    k = np.clip(np.searchsorted(ref, q), 0, len(ref) - 1)
    return np.minimum(np.abs(ref[k] - q), np.abs(ref[np.maximum(k - 1, 0)] - q))


def per_label_f1(pred_named, truth, hz):
    out = {}
    for l in np.unique(truth):
        t = np.where(truth == l, l, "~")
        p = np.where(pred_named == l, l, "~")
        out[str(l)] = round(segment_scores(p, t, hz)["seg_f1"], 3)
    return out


def cycle_onset_error(pred: np.ndarray, truth: np.ndarray, rest_bins: tuple, hz: float) -> dict:
    """Cycle boundaries: true onsets (resting -> moving: idle/clear -> any active sub-phase, i.e. the first frame a floor
    camera can see the machine work) vs predicted onsets (idle/blocked bin -> any other bin)."""
    ta = ~np.isin(truth, ("idle", "clear"))
    pa = ~np.isin(pred, rest_bins)
    to = (np.flatnonzero(ta[1:] & ~ta[:-1]) + 1).astype(float)
    po = (np.flatnonzero(pa[1:] & ~pa[:-1]) + 1).astype(float)
    if not len(to) or not len(po):
        return dict(onset_med_s=float("nan"), onset_p90_s=float("nan"), onset_found=0.0)
    d = _nearest(po, to) / hz
    return dict(onset_med_s=float(np.median(d)), onset_p90_s=float(np.quantile(d, 0.9)), onset_found=float((d <= 2.0).mean()))


def cycle_onset_error_v1(pred: np.ndarray, frac: np.ndarray, hz: float) -> dict:
    """Onset metric as first defined (before the 7185ca6 held-out run): true onset = the arm's / carriage's active start
    (frac >= 0, including the lifter's invisible overhead part), predicted onset = entry into bin 0."""
    act = frac >= 0
    to = (np.flatnonzero(act[1:] & ~act[:-1]) + 1).astype(float)
    po = (np.flatnonzero((pred[1:] == 0) & (pred[:-1] != 0)) + 1).astype(float)
    if not len(to) or not len(po):
        return dict(onset_med_s=float("nan"), onset_p90_s=float("nan"), onset_found=0.0)
    d = _nearest(po, to) / hz
    return dict(onset_med_s=float(np.median(d)), onset_p90_s=float(np.quantile(d, 0.9)), onset_found=float((d <= 2.0).mean()))


def ari(a: np.ndarray, b: np.ndarray) -> float:
    from math import comb
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1), np.int64)
    np.add.at(C, (ai, bi), 1)
    s = lambda x: float(sum(comb(int(v), 2) for v in np.ravel(x)))
    sij, sa, sb = s(C), s(C.sum(1)), s(C.sum(0))
    exp = sa * sb / comb(len(a), 2)
    mx = 0.5 * (sa + sb)
    return (sij - exp) / (mx - exp) if mx != exp else 1.0


def purity(cl: np.ndarray, truth: np.ndarray) -> float:
    return float(sum(np.unique(truth[cl == c], return_counts=True)[1].max() for c in np.unique(cl)) / len(cl))


def coarse(fam: str, lab: np.ndarray, frac: np.ndarray | None = None) -> np.ndarray:
    """Coarse truth: weld / tip_dress / return / idle for the weld family; for the lifter what a floor camera can see at
    all: carriage on the floor (below H_CLEAR; lower / down / raise look identical in plan view) vs clear."""
    if fam == "weld":
        return np.vectorize(COARSE_WELD.get)(lab)
    if fam == "lift" and frac is not None:
        h = np.array([K.lift_height(f) if f >= 0 else K.LIFT_H_TOP for f in frac])
        return np.where(h < K.H_CLEAR, "on_floor", "clear")
    return lab


def cmd_eval(a):
    tr = dict(np.load(OUT / f"{a.train}.npz", allow_pickle=True))
    te = dict(np.load(OUT / f"{a.test}.npz", allow_pickle=True))
    scn = load()
    lay = P.Layout.from_scenario(scn)
    if a.engine:
        P.MOTIF_ENGINE = a.engine
    tic = time.time()
    pm = P.PhaseModel.fit_features(tr["T"], tr["X"], tr["F"], lay, HZ)
    t_fit = time.time() - tic
    tic = time.time()
    bins, states, Fl, cl = pm.decode_features(te["T"], te["X"], te["F"])
    t_dec = time.time() - tic
    names = te["lab_names"]
    res = {"train": a.train, "test": a.test, "fit_s": round(t_fit, 1), "decode_s": round(t_dec, 1),
           "train_h": round(float(tr["T"][-1] - tr["T"][0]) / H, 2), "test_h": round(float(te["T"][-1] - te["T"][0]) / H, 2),
           "motif_engine": P.MOTIF_ENGINE, "machines": {}}
    for i, mid in enumerate(te["mids"]):
        mp = pm.machines[mid]
        fam = str(te["fams"][i])
        truth = names[te["lab"][:, i]]
        r = {"family": fam, "period_s": round(mp.period_s, 2), "n_bins": mp.n_bins, "diag": mp.diag}
        for mode, pred in (("online_hmm", bins[:, i]), ("offline_dtw", mp.label_offline(te["X"][:, i]))):
            for lvl, tl in (("fine", truth), ("coarse", coarse(fam, truth, te["frac"][:, i]))):
                mp_ = majority_map(pred, tl)
                pn = np.vectorize(mp_.get)(pred)
                sc = segment_scores(pn, tl, HZ)
                sc["per_label_f1"] = per_label_f1(pn, tl, HZ)
                sc["bin_names"] = {int(k): str(v) for k, v in mp_.items()}
                r[f"{mode}/{lvl}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}
            r[f"{mode}/cycle"] = {k: round(v, 3) for k, v in cycle_onset_error(
                pred, coarse(fam, truth, te["frac"][:, i]), (mp.idle_seg, mp.blocked_bin), HZ).items()}
            r[f"{mode}/cycle_v1"] = {k: round(v, 3) for k, v in cycle_onset_error_v1(pred, te["frac"][:, i], HZ).items()}
        res["machines"][str(mid)] = r
        c, f, o = r["online_hmm/coarse"], r["online_hmm/fine"], r["offline_dtw/coarse"]
        print(f"{mid:6s} P={mp.period_s:6.2f} bins={mp.n_bins}  online coarse F1 {c['seg_f1']:.3f} acc {c['acc']:.3f} "
              f"bnd {c['bnd_med_s']:.2f}s | fine F1 {f['seg_f1']:.3f} | offline coarse F1 {o['seg_f1']:.3f} acc {o['acc']:.3f} "
              f"| onset {r['online_hmm/cycle']['onset_med_s']:.2f}s  {c['per_label_f1']}", flush=True)
    # ---- line states
    lt = np.array(P.LINE_STATES)[te["line"]]
    res["line"] = dict(ari=round(ari(cl, lt), 3), purity=round(purity(cl, lt), 3),
                       named_acc=round(float((states == lt).mean()), 3),
                       recall={s: round(float((states[lt == s] == s).mean()), 3) for s in P.LINE_STATES if (lt == s).any()},
                       precision={s: round(float((lt[states == s] == s).mean()), 3) for s in P.LINE_STATES if (states == s).any()},
                       clusters=pm.line.names, centres=np.round(pm.line.raw_centres, 2).tolist(), feats=list(P.LINE_FEATS),
                       diag=pm.line.diag, catalogue=pm.line_catalogue(),
                       confusion={s: {u: int(((lt == s) & (states == u)).sum()) for u in P.LINE_STATES} for s in P.LINE_STATES})
    print("line:", json.dumps({k: v for k, v in res["line"].items() if k != "centres"}))
    print("centres:", res["line"]["centres"])
    OUT.mkdir(parents=True, exist_ok=True)
    pm.save(OUT / f"model_{a.train}.pkl")
    with open(OUT / f"results_{a.train}__{a.test}.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"fit {t_fit:.0f} s, decode {t_dec:.0f} s -> {OUT / f'results_{a.train}__{a.test}.json'}")


if __name__ == "__main__":
    main()
