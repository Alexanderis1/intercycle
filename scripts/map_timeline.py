"""Map convergence timeline: the observer's map built from 1 h, 6 h, 1 day, 3 days and 1 week of observation.

    python scripts/map_timeline.py build   [--days 7] [--workers 6]     # heavy: simulate + partial maps (~25 min)
    python scripts/map_timeline.py heldout [--workers 4]                # heavy-ish: held-out scoring record (day 7)
    python scripts/map_timeline.py replay  [--live] [--run week1]       # cumulative maps -> metrics, PNGs, stream snapshots

`build` tiles `--days` days with 6 h windows (seed 0) and saves every window's partial map at each checkpoint (every
15 min for the first 2 h, then hourly) under out/timeline/parts/. Counts add, so the cumulative map at time T is the
sum of the windows that ended before T plus the partial map of the window that contains T.
`heldout` records truth + the observer's own conditioning on held-out time (day 7, never used to build the map) via
intercycle/eval/map_metrics.py. `replay` walks the checkpoints in time order, scores every cumulative map on the
held-out record, writes a stream snapshot per checkpoint through intercycle/observer/stream.py (D's contract;
`--live` also pushes to the websocket), PNGs and metrics at 1 h / 6 h / 1 d / 3 d / 1 w, and the convergence curves.

Everything is SYNTHETIC (simulated plant, simulated cameras; the camera shim's noise is assumed until calibrated).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_map as B  # noqa: E402

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.scenario import load  # noqa: E402

OUT = ROOT / "out" / "timeline"
PARTS = OUT / "parts"
H, DAY = 3600.0, 86_400.0
MILESTONES = {"1h": 1 * H, "6h": 6 * H, "1d": DAY, "3d": 3 * DAY, "1w": 7 * DAY}


def checkpoints(days: int) -> list[float]:
    early = [k * 900.0 for k in range(1, 9)]                       # 15 min .. 2 h
    late = [k * H for k in range(3, int(days * 24) + 1)]
    return early + late


def windows(days: int, hours: float = 6.0, seed: int = 0):
    return B.plan_windows(days, hours, seed)


# ------------------------------------------------------------------------------------------------ build

def cmd_build(a) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    cps = checkpoints(a.days)
    jobs = []
    for i, (s, t0, t1) in enumerate(windows(a.days)):
        inside = [c for c in cps if t0 < c < t1]
        jobs.append((s, t0, t1, str(PARTS / f"w{i:03d}.npz"), inside))
    todo = [j for j in jobs if not Path(j[3]).exists()]           # resumable
    print(f"{len(jobs)} windows, {len(todo)} to run, {a.workers} workers", flush=True)
    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for path, dt in ex.map(B._worker, todo):
            print(f"  {Path(path).name} done in {dt:.0f} s  (elapsed {time.time() - t:.0f} s)", flush=True)
    meta = dict(days=a.days, windows=[j[:4] for j in jobs], checkpoints=cps)
    (OUT / "build.json").write_text(json.dumps(meta, indent=1))
    print(f"build done in {time.time() - t:.0f} s", flush=True)


def cumulative_maps(meta: dict):
    """Yield (T, LayeredMap) for every checkpoint and window end in time order (cumulative counts)."""
    wins = meta["windows"]
    base = None
    times = sorted(set(meta["checkpoints"]) | {w[2] for w in wins})
    wi = 0
    for T in times:
        while wi < len(wins) and wins[wi][2] <= T:              # windows complete by T join the base
            m = M.LayeredMap.load(wins[wi][3])
            if base is None:
                base = m
            else:
                base.merge(m)
            wi += 1
        cur = None
        if wi < len(wins) and wins[wi][1] < T:
            stem = Path(wins[wi][3]).with_suffix("")
            f = Path(f"{stem}_{int(round(T)):09d}.npz")
            if f.exists():
                cur = M.LayeredMap.load(f)
                if base is not None:
                    cur.merge(base)
                    # cycle models: the estimate from the completed windows (many cycles), not the young window's tracker
                    for a_ in ("period", "busy", "jitter"):
                        bv = getattr(base, a_)
                        getattr(cur, a_)[np.isfinite(bv)] = bv[np.isfinite(bv)]
        if cur is None:
            if base is None:
                continue
            import copy
            cur = copy.deepcopy(base)
        yield T, cur


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--days", type=int, default=7)
    b.add_argument("--workers", type=int, default=6)
    h = sub.add_parser("heldout")
    h.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("replay")
    r.add_argument("--live", action="store_true")
    r.add_argument("--run", default="week1")
    r.add_argument("--every", type=int, default=1, help="use every n-th checkpoint (milestones always)")
    r.add_argument("--pace", type=float, default=0.5, help="--live: wall seconds between snapshots")
    r.add_argument("--quick-all", action="store_true", help="score every snapshot on all held-out windows (slower)")
    sub.add_parser("report")
    ph = sub.add_parser("phases", help="discovered (workstream P) vs equal-slice L1 bins, discovered vs calendar line state")
    ph.add_argument("--workers", type=int, default=4)
    ph.add_argument("--step", default="all", choices=["fit", "build", "heldout", "score", "all"])
    a = ap.parse_args()
    if a.cmd == "report":
        cmd_report(a)
        return
    if a.cmd == "phases":
        cmd_phases(a)
        return
    if a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "heldout":
        from intercycle.eval import map_metrics as MM
        MM.record_heldout(OUT / "heldout", workers=a.workers)
    else:
        cmd_replay(a)


MARGIN_CELLS = (0, 1, 2, 3, 4)          # inflation margin sweep: 0 .. 1.0 m at 0.25 m
FULL_PREDS = ((("env", "bin", 0.0, 0), ("inner", "robot", 0.0, 0))
              + tuple(("box", "robot", 0.0, d) for d in MARGIN_CELLS)
              + (("box", "robot", 10.0, 0), ("box", "robot", 10.0, 2)))
QUICK_PREDS = (("env", "bin", 0.0, 0), ("box", "robot", 0.0, 0))


def snapshot_layers(m: M.LayeredMap, T: float, scn, max_boxes: int = 8):
    """Stream-contract arrays for the map at observation time T (the slice the robot would read at T)."""
    tod = M.tod_bin(T)
    cal = M.calendar_state(scn, T)
    prod = cal if cal is not None else M.RUNNING
    l2, h2 = m.traffic(M.VEHICLE, tod, prod)
    l3, h3 = m.traffic(M.PERSON, tod, prod)
    unk = m.unknown_mask(tod, prod)
    l1 = {}
    for d in m.spec["machines"]:
        rows = []
        for b in range(M.N_BINS):
            for bx in m.box_cover(d["id"], b, 0.05, max_boxes=max_boxes):
                rows.append((b, *(round(float(v), 2) for v in bx)))
        l1[d["id"]] = rows
    return dict(prod=prod, tod=tod, l2=l2.T, h2=h2.T, l3=l3.T, h3=h3.T, unknown=unk.T, l1=l1)


def fig_milestone(m: M.LayeredMap, scn, label: str, hours: float, metrics: dict, path: Path) -> None:
    plt = B._plt()
    from matplotlib.colors import ListedColormap
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.2), dpi=120)
    res = m.spec["env_res"]
    ax = axs[0, 0]
    B._backdrop(ax, scn)
    for d in m.spec["machines"]:
        e = m.envelope_union(d["id"], range(M.N_BINS))
        ax.imshow(np.ma.masked_where(~e, e).T, origin="lower", extent=[d["x0"], d["x0"] + d["nx"] * res, d["z0"], d["z0"] + d["nz"] * res],
                  cmap=B._cmap("envelope"), vmin=-1, vmax=1.2, interpolation="nearest", zorder=1)
    U = m.unknown_mask(M.tod_bin(10 * H), M.RUNNING)
    s = m.spec["site"]
    if U.any():
        ax.imshow(np.ma.masked_where(~U, U).T, origin="lower", extent=[s["x0"], s["x0"] + s["nx"] * s["res"], s["z0"], s["z0"] + s["nz"] * s["res"]],
                  cmap=ListedColormap(["#9aa5b1"]), alpha=0.45, interpolation="nearest", zorder=2)
    ax.set_title("L1 envelopes (any phase, p >= 5 %); grey = UNKNOWN", fontsize=8, loc="left")
    for ax, cls, prod, kind, title in ((axs[0, 1], M.VEHICLE, M.RUNNING, "vehicle", "L2 vehicles (running, all day)"),
                                       (axs[1, 0], M.PERSON, M.RUNNING, "person", "L3 people (running, all day)"),
                                       (axs[1, 1], M.PERSON, M.BREAK, "person", "L3 people (breaks)")):
        B._backdrop(ax, scn, label_machines=False)
        p, h = B.aggregate_traffic(m, cls, prod)
        B._heat(ax, m, p, kind, 0.5, h)
        ax.set_title(title, fontsize=8, loc="left")
    keys = ("unsafe_free", "l1_iou", "l1_recall", "l2_auroc", "l3_auroc", "unknown_share")
    mt = ", ".join(f"{k} {metrics[k]:.4f}" for k in keys if k in metrics)
    fig.suptitle(f"SYNTHETIC - observer map after {label} of observation ({hours:.1f} h)   held-out: {mt}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(rows: list[dict], path: Path) -> None:
    plt = B._plt()
    hrs = np.array([r["hours"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), dpi=130)
    series = [(axs[0], [("unsafe_free", "unsafe-free (robot boxes, overlap truth)", "#b42318"),
                        ("env_unsafe_free", "unsafe-free (raw envelope, centre truth)", "#6b7785"),
                        ("unknown_share", "machine-instants with unknown phase", "#9aa5b1")], "rate", True),
              (axs[1], [("l1_iou", "L1 IoU (machine x bin, mean)", "#0b6e69"),
                        ("l1_recall", "L1 recall", "#1d4f91"),
                        ("false_occupied_share", "false-occupied share (robot boxes)", "#b45309")], "fraction", False),
              (axs[2], [("l2_auroc", "L2 vehicles AUROC", "#b45309"), ("l3_auroc", "L3 people AUROC", "#1d4f91")], "AUROC", False)]
    for ax, ser, yl, logy in series:
        for k, lab, col in ser:
            y = np.array([r.get(k, np.nan) for r in rows], float)
            ax.plot(hrs, y, color=col, lw=1.4, label=lab)
        for lab, T in MILESTONES.items():
            ax.axvline(T / H, color="#d9dee4", lw=0.8, zorder=0)
            ax.text(T / H, 1.0, lab, transform=ax.get_xaxis_transform(), fontsize=7, color="#6b7785", ha="center", va="bottom")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("hours of observation (log)", fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("SYNTHETIC - map convergence, scored on held-out time (day 7, window 08:15-10:15: running, andon, 09:00 break; "
                 "milestone tables use all 4 held-out windows)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_replay(a) -> None:
    from intercycle.eval import map_metrics as MM
    from intercycle.observer.stream import StreamWriter
    scn = load()
    meta = json.loads((OUT / "build.json").read_text())
    recs = MM.load_records(OUT / "heldout")
    quick_recs = recs if a.quick_all else recs[:1]
    rows, full = [], {}
    sw = StreamWriter(a.run, out_root=ROOT / "out" / "stream", live=a.live,
                      meta=dict(kind="timeline", days=meta["days"], label="SYNTHETIC: simulated plant",
                                prod_state_source="break / shift_change: from public calendar (external input), pending discovered "
                                                  "states (workstream P); andon / running: discovered from line motion",
                                note="metrics on held-out day 7 (never used to build the map); camera-shim noise assumed until "
                                     "workstream V calibrates it"))
    ms_left = dict(sorted(MILESTONES.items(), key=lambda kv: kv[1]))
    t_wall = time.time()
    k = 0
    for T, m in cumulative_maps(meta):
        hours = m.n_ticks / M.FRAME_HZ / H
        label = next((lab for lab, tt in ms_left.items() if abs(tt - T) < 1.0), None)
        k += 1
        if label is None and (k % a.every):
            continue
        if label is not None:
            sc = MM.score(m, recs, preds=FULL_PREDS)
            full[label] = dict(T=T, hours=hours, score=sc)
            ms_left.pop(label)
            m.save(OUT / f"map_{label}.npz")
            hl = MM.headline(sc)
            fig_milestone(m, scn, label, hours, hl, OUT / f"map_{label}.png")
            (OUT / f"score_{label}.json").write_text(json.dumps(sc, indent=1, default=float))
            print(f"[{label}] T={T / H:.1f} h  " + "  ".join(f"{kk}={vv:.4g}" for kk, vv in hl.items()), flush=True)
        else:
            hl = MM.headline(MM.score(m, quick_recs, preds=QUICK_PREDS))
        if label is not None:                                   # the curve uses one scoring set throughout
            hl_all = hl
            hl = MM.headline(MM.score(m, quick_recs, preds=QUICK_PREDS))
            full[label]["headline_quick"] = hl
        row = dict(T=T, hours=hours, milestone=label, **hl)
        rows.append(row)
        L = snapshot_layers(m, T, scn)
        sw.snapshot(t_sim=T, t_obs=hours * H, prod_state=M.PROD_NAMES[L["prod"]], grid=dict(m.spec["site"]), l1=L["l1"],
                    l2_occ=L["l2"], l2_head=L["h2"], l3_occ=L["l3"], l3_head=L["h3"], unknown=L["unknown"],
                    phase_now={d["id"]: None for d in m.spec["machines"]},
                    metrics={kk: float(vv) for kk, vv in row.items() if isinstance(vv, (int, float)) and kk != "T"})
        if a.live and a.pace > 0:
            time.sleep(a.pace)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} snapshots, T={T / H:.0f} h, {time.time() - t_wall:.0f} s", flush=True)
            (OUT / "timeline.json").write_text(json.dumps(dict(rows=rows), indent=1, default=float))
    sw.close()
    ms = {kk: dict(T=v["T"], hours=v["hours"], headline=MM.headline(v["score"])) for kk, v in full.items()}
    (OUT / "timeline.json").write_text(json.dumps(dict(rows=rows, milestones=ms), indent=1, default=float))
    fig_convergence(rows, OUT / "convergence.png")
    print(f"replay done: {len(rows)} snapshots in {time.time() - t_wall:.0f} s", flush=True)



# ------------------------------------------------------------------------------------------------ report (tables for docs/MAP_RESULTS.md)

def _f(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    return str(v)


def cmd_report(a) -> None:
    from intercycle.eval import map_metrics as MM
    recs = MM.load_records(OUT / "heldout")
    L = []
    sc = {lab: json.loads((OUT / f"score_{lab}.json").read_text()) for lab in MILESTONES if (OUT / f"score_{lab}.json").exists()}
    # ---- convergence table
    L.append("| map built from | hours | L1 IoU (bin, p05 truth) | L1 recall | **unsafe-free, robot boxes + 0.25 m (strict overlap)** | "
             "false-occupied share (+0.25 m) | unsafe-free, robot boxes, no margin (strict / centre) | false-occupied share (no margin) | "
             "traffic cells UNKNOWN | L2 Brier (known) | L2 AUROC | L3 Brier (known) | L3 AUROC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, s in sc.items():
        h = MM.headline(s)
        d1 = s["l1"].get("box|robot|h0|d1", {})
        hrs = json.loads((OUT / "timeline.json").read_text())["milestones"][lab]["hours"]
        L.append(f"| {lab} | {hrs:.1f} | {_f(h['l1_iou'], 3)} | {_f(h['l1_recall'], 3)} | **{_f(d1.get('unsafe_free_overlap'))}** | "
                 f"{_f(d1.get('false_occupied_share_overlap'), 3)} | {_f(h['unsafe_free'])} / {_f(h['unsafe_free_centre'])} | "
                 f"{_f(h['false_occupied_share'], 3)} | {_f(h['l2_unknown_share'], 3)} | {_f(h['l2_brier_known'])} | "
                 f"{_f(h['l2_auroc'], 3)} | {_f(h['l3_brier_known'])} | {_f(h['l3_auroc'], 3)} |")
    if "1w" in sc:
        s = sc["1w"]
        # ---- predictors / margin trade-off
        L.append("\n| predictor (1-week map) | horizon | margin | unsafe-free (overlap truth) | unsafe-free (centre truth) | false-occupied share | false-occupied m2 / machine |")
        L.append("|---|---|---|---|---|---|---|")
        for k, v in s["l1"].items():
            pk, mode, hz, d = k.split("|")
            name = {"env": "raw envelope, tracker bin", "box": "broadcast safe boxes, robot bins", "inner": "old inner boxes, robot bins"}[pk]
            L.append(f"| {name} | {hz[1:]} s | {int(d[1:]) * 0.25:.2f} m | {_f(v['unsafe_free_overlap'])} | {_f(v['unsafe_free_centre'])} | "
                     f"{_f(v['false_occupied_share_overlap'], 3)} | {_f(v['false_occupied_m2_per_machine_overlap'], 2)} |")
        # ---- per machine
        rob = s["l1"]["box|robot|h0|d0"]
        env = s["l1"]["env|bin|h0|d0"]
        rows = s["l1_bins"]["rows"]
        L.append("\n| machine | IoU (mean over bins, p05 truth) | precision | recall | unsafe-free robot boxes (overlap) | unsafe-free raw envelope (centre) |")
        L.append("|---|---|---|---|---|---|")
        for mid in rob["per_machine_unsafe_free_overlap"]:
            rr = [r for r in rows if r["machine"] == mid and r["truth"] == "p05" and not r.get("empty")]
            L.append(f"| {mid} | {_f(float(np.mean([r['iou'] for r in rr])), 3)} | {_f(float(np.mean([r['precision'] for r in rr])), 3)} | "
                     f"{_f(float(np.mean([r['recall'] for r in rr])), 3)} | {_f(rob['per_machine_unsafe_free_overlap'][mid])} | "
                     f"{_f(env['per_machine_unsafe_free_centre'][mid])} |")
        # ---- traffic by production state
        L.append("\n| layer | production state (observer) | samples (10 s) | Brier (UNKNOWN = 1) | Brier (known cells) | Brier climatology | AUROC |")
        L.append("|---|---|---|---|---|---|---|")
        for c, nm in (("vehicle", "L2 vehicles"), ("person", "L3 people")):
            t = s["traffic"][c]
            for pn, v in t["by_prod"].items():
                L.append(f"| {nm} | {pn} | {v['samples']} | {_f(v['brier'])} | {_f(v['brier_known'])} | {_f(v['brier_clim'])} | {_f(v['auroc'], 3)} |")
            L.append(f"| {nm} | **all** | | {_f(t['brier'])} | {_f(t['brier_known'])} | {_f(t['brier_clim'])} | {_f(t['auroc'], 3)} |")
        # ---- per tod x prod slices
        L.append("\n| tod | state | L2 Brier | L2 AUROC | L3 Brier | L3 AUROC | samples |")
        L.append("|---|---|---|---|---|---|---|")
        rv = {(r["tod"], r["prod"]): r for r in s["traffic"]["vehicle"]["rows"]}
        for r in s["traffic"]["person"]["rows"]:
            v = rv.get((r["tod"], r["prod"]), {})
            L.append(f"| {r['tod'] // 2:02d}:{30 * (r['tod'] % 2):02d} | {r['prod']} | {_f(v.get('brier_known'))} | {_f(v.get('auroc'), 3)} | "
                     f"{_f(r['brier_known'])} | {_f(r['auroc'], 3)} | {r['samples']} |")
        # ---- missed coverage analysis
        m1 = M.LayeredMap.load(OUT / "map_1w.npz")
        ma = MM.missed_analysis(m1, recs)
        (OUT / "missed_1w.json").write_text(json.dumps(ma, indent=1))
        L.append("\n| truth definition | truly occupied cell-instants | missed by raw bin envelope | rate | timing (+-1 bin) | spatial (outside all-bin union) | other | distance to envelope (cells: count) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for dn, v in ma.items():
            L.append(f"| {dn} | {v['truly_occupied']:,} | {v['missed']:,} | {_f(v['missed_rate'])} | {_f(v['by_cause']['timing'], 3)} | "
                     f"{_f(v['by_cause']['spatial'], 3)} | {_f(v['by_cause']['other'], 3)} | "
                     + ", ".join(f"{k}: {n}" for k, n in v["distance_cells"].items()) + " |")
        sp = MM.part_speeds(dur=300)
        (OUT / "part_speeds.json").write_text(json.dumps(sp, indent=1))
        L.append("\nFastest body circle per machine (truth, m/s, p99 / max): " +
                 ", ".join(f"{k} {v['p99']:.1f} / {v['max']:.1f}" for k, v in sp.items()))
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ discovered phases (P)

PHD = OUT / "phases"
PM_FIT = (6.0 * H, 9.75 * H)               # day 0, 06:00-09:45: running, andon stops and the 09:00 break


def _pm_load():
    from intercycle.observer import phases as PH
    PH.MOTIF_ENGINE = "numpy"
    return PH.PhaseModel.load(PHD / "pm.pkl", load())


def _pm_window(args):
    seed, t0, t1, path = args
    pm = _pm_load()
    t = time.time()
    m, _, _ = B.run_window(seed, t0, t1, phase_model=pm, use_bins=True, use_line_state=True)
    m.save(path)
    return path, time.time() - t


def _pm_record(args):
    seed, t0, t1, path = args
    from intercycle.eval import map_metrics as MM
    pm = _pm_load()
    scn = load()
    rec = MM.Recorder(scn, t0)
    t = time.time()
    B.run_window(seed, t0 - MM.WARM_S, t1, scn=scn, tick_hook=rec, build=False, phase_model=pm, use_bins=True, use_line_state=True)
    rec.save(Path(path))
    return path, time.time() - t


def cmd_phases(a) -> None:
    from intercycle.eval import map_metrics as MM
    PHD.mkdir(parents=True, exist_ok=True)
    if a.step in ("fit", "all") and not (PHD / "pm.pkl").exists():
        from intercycle.observer import phases as PH
        PH.MOTIF_ENGINE = "numpy"
        obs = []
        B.run_window(0, *PM_FIT, tick_hook=lambda k, tc, st, o, c: obs.append(o), build=False)
        t = time.time()
        pm = PH.PhaseModel.fit(obs, load())
        pm.save(PHD / "pm.pkl")
        print(f"PhaseModel fitted on {len(obs)} ticks in {time.time() - t:.0f} s", flush=True)
    if a.step in ("build", "all"):
        jobs = [(s_, t0, t1, str(PHD / f"d{i:02d}.npz")) for i, (s_, t0, t1) in enumerate(windows(1))]
        jobs = [j for j in jobs if not Path(j[3]).exists()]
        with ProcessPoolExecutor(a.workers) as ex:
            for pth, dt in ex.map(_pm_window, jobs):
                print(f"  discovered-bin window {Path(pth).name} in {dt:.0f} s", flush=True)
    if a.step in ("heldout", "all"):
        jobs = [(0, t0, t1, str(PHD / f"heldout_{k}.npz")) for k, (t0, t1) in enumerate(MM.HELDOUT)]
        jobs = [j for j in jobs if not Path(j[3]).exists()]
        with ProcessPoolExecutor(min(a.workers, 4)) as ex:
            for pth, dt in ex.map(_pm_record, jobs):
                print(f"  discovered held-out {Path(pth).name} in {dt:.0f} s", flush=True)
    if a.step in ("score", "all"):
        parts = sorted(PHD.glob("d[0-9][0-9].npz"))
        md = M.LayeredMap.load(parts[0])
        for f in parts[1:]:
            md.merge(M.LayeredMap.load(f))
        md.save(PHD / "map_1d_discovered.npz")
        me = M.LayeredMap.load(OUT / "map_1d.npz")
        rd = [dict(np.load(f)) for f in sorted(PHD.glob("heldout_*.npz"))]
        re_ = MM.load_records(OUT / "heldout")
        preds = (("env", "bin", 0.0, 0), ("env", "bin", 0.0, 1))
        res = {}
        for name, mp, rr in (("equal_slices_calendar", me, re_), ("discovered_P", md, rd)):
            sc = MM.score(mp, rr, preds=preds)
            area = [r["area_m2"] for r in sc["l1_bins"]["rows"] if r["truth"] == "p05"]
            res[name] = dict(l1={k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_")} for k, v in sc["l1"].items()},
                             l1_bins_mean=sc["l1_bins"]["mean"], mean_bin_envelope_m2=float(np.mean(area)) if area else None,
                             n_bins_used=len(area),
                             traffic={c: {k: sc["traffic"][c][k] for k in ("brier", "brier_known", "auroc", "unknown_share")} for c in ("vehicle", "person")},
                             prod_share={M.PROD_NAMES[k]: float(np.mean(np.concatenate([r["l1_prod"] for r in rr]) == k)) for k in range(M.N_PROD)})
        # line-state agreement on held-out time: discovered vs calendar rule (both are the observer's; truth = calendar + andon windows not used here)
        pe = np.concatenate([r["l1_prod"] for r in re_])
        pd_ = np.concatenate([r["l1_prod"] for r in rd])
        n = min(len(pe), len(pd_))
        res["line_state_agreement_with_calendar_rule"] = float((pe[:n] == pd_[:n]).mean())
        (PHD / "compare.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
