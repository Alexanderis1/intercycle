"""A1 pixels -> phases: features from rendered CCTV frames (intercycle/perception/pixel_phases.py) into the UNCHANGED
PhaseModel, scored against the sim's kinematics SIDE BY SIDE with the camera shim on the same window and seed.

    python scripts/pixel_phases.py extract --frames <cctv_post frames dir> --csv <frames.csv> --rois render/a1/cam02_rois.json \
        --out out/pixels/a1_px.npz
    python scripts/pixel_phases.py score --px out/pixels/a1_px.npz --seed 3 --t0 36000 --t1 38700 --split 1500 \
        --out out/pixels/a1_scores.json

Protocol = scripts/phase_discovery.py cmd_eval: fit on [t0, t0 + split), decode [t0 + split, t1) causally (online HMM,
fresh state) and offline (DTW), name the discovered bins by majority overlap with the test truth, score with the same
functions (segment F1, frame accuracy, boundary error / F1 at +-1 s, cycle onset error). Truth = Kinematics.phase at
each tick time (scoring only). The shim side runs phase_discovery.simulate(seed, t0, t1) (cached next to --out).
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

import phase_discovery as PD  # noqa: E402
from intercycle.observer import phases as P  # noqa: E402
from intercycle.perception import pixel_phases as PX  # noqa: E402
from intercycle.scenario import load  # noqa: E402

HZ = PD.HZ


def cmd_extract(a):
    _, rois = PX.load_rois(a.rois)
    frames = PX.read_frames_csv(a.csv)
    if a.limit:
        frames = frames[:a.limit]
    tic = time.time()
    t, feats, stats = PX.extract_clip(a.frames, frames, rois, stabilise=not a.no_ecc, progress=2000)
    stats["seconds"] = round(time.time() - tic, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, t=t, feats=feats, mids=np.array([r.mid for r in rois]),
                        boxes=np.array([r.box for r in rois]), stats=json.dumps(stats))
    print("extract:", stats)


def truth_at(scn, seed: int, t0: float, t1: float, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fine truth labels (n, M) and active fractions (n, M) at tick times T (as phase_discovery.simulate)."""
    from intercycle.sim.world import World
    from intercycle.truth import kinematics as K
    kin = K.Kinematics(scn, World(scn, seed, t0, t1))
    mids = [m.id for m in scn.machines]
    fams = [PD.family(m.kind) for m in scn.machines]
    appr = {}
    for mid in mids:
        b = [s.brk[1] for s in kin.arm_specs if s.mid == mid]
        appr[mid] = float(np.mean(b)) if b else 0.0
    lab = np.empty((len(T), len(mids)), object)
    fr = np.zeros((len(T), len(mids)), np.float32)
    for k, t in enumerate(T):
        for i, mid in enumerate(mids):
            st, f = kin.phase(mid, float(t))
            lab[k, i] = PD.truth_label(fams[i], st, f, appr[mid])
            fr[k, i] = f if st == "active" else -1.0
    return lab, fr


def true_period(truth: np.ndarray, T: np.ndarray, takt: float) -> float:
    """Median time between true cycle onsets (rest -> active), stops excluded (gaps > 1.5 takt)."""
    act = ~np.isin(truth, ("idle", "clear"))
    on = T[np.flatnonzero(act[1:] & ~act[:-1]) + 1]
    d = np.diff(on)
    d = d[d < 1.5 * takt]
    return float(np.median(d)) if len(d) else float("nan")


def score(name: str, T, X, lab, frac, scn, split_t: float, mids_scored: list[str]) -> dict:
    lay = P.Layout.from_scenario(scn)
    tr, te = T < split_t, T >= split_t
    F = np.zeros((len(T), len(P.FLOW_NAMES)), np.float32)
    tic = time.time()
    pm = P.PhaseModel.fit_features(T[tr], X[tr], F[tr], lay, HZ, fit_line=False)
    fit_s = time.time() - tic
    bins, *_ = pm.decode_features(T[te], X[te], F[te])
    out = {"source": name, "fit_s": round(fit_s, 1), "train_min": round(float(tr.sum()) / HZ / 60, 1),
           "test_min": round(float(te.sum()) / HZ / 60, 1), "machines": {}}
    for mid in mids_scored:
        i = lay.mids.index(mid)
        mp = pm.machines.get(mid)
        fam = PD.family(next(m.kind for m in scn.machines if m.id == mid))
        truth = lab[te, i].astype(str)
        tp = true_period(lab[:, i].astype(str), T, scn.line.takt_s)
        if mp is None:
            out["machines"][mid] = {"model": None, "true_period_s": tp}
            continue
        r = {"period_s": round(mp.period_s, 2), "true_period_s": round(tp, 2),
             "period_err_s": round(abs(mp.period_s - tp), 2), "n_bins": mp.n_bins}
        for mode, pred in (("online_hmm", bins[:, i]), ("offline_dtw", mp.label_offline(X[te][:, i]))):
            for lvl, tl in (("fine", truth), ("coarse", PD.coarse(fam, truth, frac[te][:, i]))):
                mp_ = PD.majority_map(pred, tl)
                pn = np.vectorize(mp_.get)(pred)
                sc = PD.segment_scores(pn, tl, HZ)
                sc["per_label_f1"] = PD.per_label_f1(pn, tl, HZ)
                r[f"{mode}/{lvl}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}
            r[f"{mode}/cycle"] = {k: round(v, 3) for k, v in PD.cycle_onset_error(
                pred, PD.coarse(fam, truth, frac[te][:, i]), (mp.idle_seg, mp.blocked_bin), HZ).items()}
        out["machines"][mid] = r
    return out


def cmd_score(a):
    scn = load()
    lay = P.Layout.from_scenario(scn)
    px = np.load(a.px, allow_pickle=True)
    mids = [str(m) for m in px["mids"]]
    rois = [PX.Roi(m, tuple(int(v) for v in b)) for m, b in zip(mids, px["boxes"])]   # type: ignore[arg-type]
    T_px, X_roi = PX.to_ticks(px["t"], px["feats"], HZ)
    X_px = PX.machine_block(X_roi, rois, lay.mids)
    lab_px, fr_px = truth_at(scn, a.seed, a.t0, a.t1, T_px)
    shim_npz = Path(a.out).with_name(f"shim_seed{a.seed}_{int(a.t0)}_{int(a.t1)}.npz")
    if shim_npz.exists():
        sh = dict(np.load(shim_npz, allow_pickle=True))
    else:
        sh = PD.simulate(a.seed, a.t0, a.t1)
        shim_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shim_npz, **sh)
    lab_sh = sh["lab_names"][sh["lab"]]
    split_t = a.t0 + a.split
    res = {"seed": a.seed, "t0": a.t0, "t1": a.t1, "split_s": a.split, "pixel_stats": json.loads(str(px["stats"])),
           "pixels": score("pixels", T_px, X_px, lab_px, fr_px, scn, split_t, mids),
           "shim": score("shim", sh["T"], sh["X"], lab_sh, sh["frac"], scn, split_t, mids)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(table(res))


def table(res: dict) -> str:
    rows = ["| Machine | source | period (s) / truth | online coarse seg F1 | online coarse acc | online bnd F1 +-1 s "
            "| online bnd err median (s) | online fine acc | offline coarse seg F1 | cycle onset err median (s) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for mid in res["pixels"]["machines"]:
        for src in ("pixels", "shim"):
            m = res[src]["machines"][mid]
            if "online_hmm/coarse" not in m:
                rows.append(f"| {mid} | {src} | no model | | | | | | | |")
                continue
            c, f, o, cy = m["online_hmm/coarse"], m["online_hmm/fine"], m["offline_dtw/coarse"], m["online_hmm/cycle"]
            rows.append(f"| {mid} | {src} | {m['period_s']:.2f} / {m['true_period_s']:.2f} | {c['seg_f1']:.3f} | {c['acc']:.3f} "
                        f"| {c['bnd_f1_1s']:.3f} | {c['bnd_med_s']:.2f} | {f['acc']:.3f} | {o['seg_f1']:.3f} | {cy['onset_med_s']:.2f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description="A1 pixels -> phases: extract features from CCTV frames, score vs the shim.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--frames", required=True)
    e.add_argument("--csv", required=True)
    e.add_argument("--rois", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--no-ecc", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--px", required=True)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--t0", type=float, default=36000.0)
    s.add_argument("--t1", type=float, default=38700.0)
    s.add_argument("--split", type=float, default=1500.0, help="s of training from t0; the rest is scored")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
