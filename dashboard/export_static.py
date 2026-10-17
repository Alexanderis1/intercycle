"""Export the static plant layout and replay runs for the dashboard (SYNTHETIC: simulated plant).

    python dashboard/export_static.py layout                    # dashboard/layout.json (L0 walls, conveyor, machines, cameras)
    python dashboard/export_static.py npz [--npz out/layered_map.npz] [--run build_map_72h] [--step-min 30]
        # replays the FINAL map of scripts/build_map.py across one day of time-of-day slices through observer/stream.py
        # (the map does not grow in this run: it is the skeleton's stand-in until M's map_timeline.py writes real runs)
    python dashboard/export_static.py bundle [--runs build_map_72h,week1] [--out out/stream/dashboard_bundle.html] [--artifact]
        # one self-contained HTML file (CSS, JS, layout and the runs inlined) for an artifact preview
    python dashboard/export_static.py register --run week1 --title "1 week"   # add a run to dashboard/runs.json
    python dashboard/export_static.py phases [--model out/phases/model_final_train.pkl]
        # dashboard/phases.json: workstream P's discovered sub-phases per machine (cycle fractions, names p0.., idle)
    python dashboard/export_static.py publish --run week1 [--max-snaps 80]   # out/stream/week1 -> dashboard/data/week1
    python dashboard/export_static.py gemma     # dashboard/gemma_names.json from out/gemma/phase_names.json (else a FIXTURE)

Light job (loads one .npz, no simulation).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
sys.path.insert(0, str(ROOT))

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer.stream import LABEL, StreamWriter  # noqa: E402

STREAM = ROOT / "out" / "stream"


def cmd_layout(args) -> None:
    from intercycle.scenario import load
    from intercycle.sim import geometry as G

    scn = load(ROOT / "scenario" / "bodyshop.yaml")
    rects, arms = G.build_rects(scn)
    out = dict(schema="intercycle.layout/1", label=LABEL, name=scn.name, hall=dict(x=list(scn.hall["x"]), z=list(scn.hall["z"])),
               data_hall=dict(x=list(scn.data_hall["x"]), z=list(scn.data_hall["z"])),
               line=dict(x=list(G.LINE_X), z=G.LINE_Z, width=G.CONV_W, walk_x=G.WALK_X, walk_gap=G.WALK_GAP),
               rects=[dict(id=r["id"], cls=r["cls"], cx=round(r["cx"], 3), cz=round(r["cz"], 3), w=round(r["w"], 3), d=round(r["d"], 3),
                           yaw=round(r["yaw"], 4)) for r in rects],
               machines=[dict(id=m.id, kind=m.kind, on_line=bool(m.on_line), rect=[float(v) for v in m.rect]) for m in scn.machines],
               arms=[dict(machine=a, x=x, z=z) for a, x, z in arms], cameras=[list(map(float, c)) for c in scn.cameras],
               breaks=[dict(start=a, end=b, kind=k) for a, b, k in scn.line_stops()])
    p = DASH / "layout.json"
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {p} ({p.stat().st_size / 1e3:.1f} kB, {len(rects)} rects)")


def cmd_phases(args) -> None:
    from intercycle.observer import phases as PH
    from intercycle.scenario import load

    pm = PH.PhaseModel.load(ROOT / args.model, load(ROOT / "scenario" / "bodyshop.yaml"))
    out = dict(schema="intercycle.phases/1", label=LABEL, source=str(args.model), machines={})
    for mid, mp in pm.machines.items():
        names = pm.bin_names(mid)
        seg = np.asarray(mp.seg_of_pos)
        m = len(seg)
        segs = []
        for k in range(int(seg.max()) + 1):
            pos = np.flatnonzero(seg == k)
            if len(pos):
                segs.append(dict(bin=k, name=names[k], start=round(pos.min() / m, 4), end=round((pos.max() + 1) / m, 4)))
        out["machines"][mid] = dict(period_s=float(mp.period_s), n_bins=pm.n_bins(mid), bin_names=names, segments=segs)
    p = DASH / "phases.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}: " + ", ".join(f"{k} {len(v['segments'])} sub-phases" for k, v in out["machines"].items()))


def cmd_gemma(args) -> None:
    """dashboard/gemma_names.json from G's out/gemma/phase_names.json; a clearly labelled FIXTURE of the same shape
    (placeholder names, no Gemma call) until that file exists."""
    src = ROOT / args.src
    dst = DASH / "gemma_names.json"
    if src.exists():
        d = json.loads(src.read_text(encoding="utf-8"))
        d["fixture"] = False
        print(f"copied {src} ({sum(len(v) for v in d.get('machines', {}).values())} named bins)")
    else:
        ph = json.loads((DASH / "phases.json").read_text(encoding="utf-8"))
        d = {"label": "FIXTURE: placeholder names, NOT Gemma output (out/gemma/phase_names.json does not exist yet)",
             "fixture": True, "model": None, "provider": "runs on AMD GPUs via Token Factory", "generated_at": None,
             "source": ph.get("source"), "machines": {}, "line_clusters": {}}
        for mid, m in ph["machines"].items():
            for sg in m["segments"]:
                d["machines"].setdefault(mid, {})[str(sg["bin"])] = {"bin_name": sg["name"], "name": f"[fixture] {sg['name']}",
                                                                    "confidence": None, "notes": "placeholder"}
        print(f"{src} missing: wrote a labelled FIXTURE")
    dst.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"wrote {dst}")


def cmd_publish(args) -> None:
    """Copy a finished run from out/stream/<run>/ into dashboard/data/<run>/ (thinned to --max-snaps, evenly in index)
    and point runs.json at it, so the static page and GitHub Pages do not depend on the gitignored out/."""
    import shutil

    src = STREAM / args.run
    idx = json.loads((src / "index.json").read_text(encoding="utf-8"))
    if not idx.get("complete") and not args.allow_incomplete:
        raise SystemExit(f"{src} is not complete (index.json complete=false); pass --allow-incomplete to export anyway")
    L = idx["snapshots"]
    if args.max_snaps and len(L) > args.max_snaps:
        keep = sorted(set(np.linspace(0, len(L) - 1, args.max_snaps).round().astype(int).tolist()))
        idx = dict(idx, snapshots=[L[k] for k in keep], thinned_from=len(L))
    dst = DASH / "data" / args.run
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for s_ in idx["snapshots"]:
        shutil.copyfile(src / s_["file"], dst / s_["file"])
    if (src / "events.jsonl").exists():
        shutil.copyfile(src / "events.jsonl", dst / "events.jsonl")
    (dst / "index.json").write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
    mb = sum(f.stat().st_size for f in dst.iterdir()) / 1e6
    print(f"published {args.run}: {len(idx['snapshots'])} of {len(L)} snapshots, {mb:.2f} MB -> {dst}")
    register(args.run, args.title or idx.get("meta", {}).get("title") or args.run, args.note, path=f"data/{args.run}")


def register(run: str, title: str, note: str = "", path: str | None = None) -> None:
    p = DASH / "runs.json"
    runs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    old = [r for r in runs if r["id"] == run]
    entry = dict(id=run, title=title, path=path or f"../out/stream/{run}", note=note)
    if old:
        runs = [entry if r["id"] == run else r for r in runs]                 # keep its place in the list
    else:
        runs = runs + [entry]
    p.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(f"registered {run} in {p}")


def cmd_npz(args) -> None:
    from intercycle.scenario import load

    scn = load(ROOT / "scenario" / "bodyshop.yaml")
    m = M.LayeredMap.load(ROOT / args.npz)
    ids = m.machine_ids()
    s = m.spec["site"]
    grid = dict(x0=s["x0"], z0=s["z0"], res=s["res"], nx=s["nx"], nz=s["nz"])
    hours = m.n_ticks / M.FRAME_HZ / 3600.0
    # L1 boxes per machine and phase bin (the final map: identical in every snapshot of this run)
    l1 = {}
    area = {}
    for mid in ids:
        rows = []
        area[mid] = []
        for b in range(M.N_BINS):
            for x0, z0, x1, z1 in m.box_cover(mid, b, max_boxes=args.max_boxes):
                rows.append((b, x0, z0, x1, z1))
            area[mid].append(float(m.envelope(mid, b).sum()) * m.spec["env_res"] ** 2)
        l1[mid] = rows
    sw = StreamWriter(args.run, out_root=STREAM, meta=dict(
        title=f"build_map.py final map ({hours:.0f} h observed), replayed over one day",
        source=str(args.npz), hours_observed=round(hours, 2), axis="t_sim", prod_state_source="public calendar (external input)", period_s={k: (None if not math.isfinite(v) else float(v)) for k, v in zip(ids, m.period)},
        phase_bins=M.N_PHASE, stop_bin=M.STOP_BIN, l1_area_m2=area, prod_names=list(M.PROD_NAMES),
        note="Skeleton stand-in: one final map (does not grow); L2/L3 follow the time-of-day slice, production state from the "
             "public calendar, phase-now projected from the learned period. Map statistics only, no accuracy metrics."))
    step = args.step_min * 60.0
    n = int(round(86_400.0 / step))
    min_ticks = m.min_ticks
    for k in range(n):
        t = k * step + 120.0                                    # +2 min lands inside the 5-min shift-change stops
        tb = M.tod_bin(t)
        cal = M.calendar_state(scn, t)
        prod = cal if cal is not None else M.RUNNING
        p2, h2 = m.traffic(M.VEHICLE, tb, prod)
        p3, h3 = m.traffic(M.PERSON, tb, prod)
        S = tb * M.N_PROD + prod
        unk = (m.visits_cam[S][m.cam_of_cell] < min_ticks).T      # this slice under-observed by the cell's camera
        phase_now = {}
        on_line = {mm["id"]: mm["on_line"] for mm in m.spec["machines"]}
        for mid, per in zip(ids, m.period):
            if cal is not None and on_line[mid]:
                phase_now[mid] = M.STOP_BIN                     # the line stops in breaks / shift changes
            else:
                phase_now[mid] = int(((t % per) / per) * M.N_PHASE) % M.N_PHASE if math.isfinite(per) and per > 0 else None
        metrics = dict(slice_ticks=m.slice_ticks(tb, prod), unknown_frac=float(unk.mean()), l2_hot_m2=float((p2 >= 0.3).sum() * s["res"] ** 2),
                       l3_hot_m2=float((p3 >= 0.3).sum() * s["res"] ** 2), l2_mean_occ=float(p2.mean()), l3_mean_occ=float(p3.mean()))
        sw.snapshot(t_sim=t, t_obs=hours * 3600.0, prod_state=M.PROD_NAMES[prod], grid=grid, l1=l1, l2_occ=p2.T, l2_head=h2.T,
                    l3_occ=p3.T, l3_head=h3.T, unknown=unk, phase_now=phase_now, metrics=metrics)
        if k == 0 or cal is not None and M.calendar_state(scn, t - step) is None:
            sw.event(t_sim=t, kind="production_state", state=M.PROD_NAMES[prod], caption=f"line {M.PROD_NAMES[prod]} (public calendar)")
    sw.close()
    idx = json.loads((STREAM / args.run / "index.json").read_text(encoding="utf-8"))
    sz = [s_["bytes"] for s_ in idx["snapshots"]]
    print(f"wrote {len(sz)} snapshots to {STREAM / args.run}: {min(sz) / 1e3:.1f}..{max(sz) / 1e3:.1f} kB each, {sum(sz) / 1e6:.2f} MB total")
    register(args.run, f"Final map, {hours:.0f} h observed (one day replay)", "skeleton stand-in, does not grow")


def cmd_bundle(args) -> None:
    runs_meta = json.loads((DASH / "runs.json").read_text(encoding="utf-8"))
    want = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else [r["id"] for r in runs_meta]
    runs = []
    for r in runs_meta:
        if r["id"] not in want:
            continue
        d = (DASH / r["path"]).resolve() if r.get("path") else STREAM / r["id"]
        if not (d / "index.json").exists():
            print(f"skip {r['id']}: no {d / 'index.json'}")
            continue
        idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
        L = idx["snapshots"]
        if args.max_snaps and len(L) > args.max_snaps:     # thin evenly (the artifact page is capped at 16 MB)
            keep = sorted(set(np.linspace(0, len(L) - 1, args.max_snaps).round().astype(int).tolist()))
            idx = dict(idx, snapshots=[L[k] for k in keep], thinned_from=len(L))
            print(f"{r['id']}: kept {len(keep)} of {len(L)} snapshots")
        snaps = {s["file"]: json.loads((d / s["file"]).read_text(encoding="utf-8")) for s in idx["snapshots"]}
        runs.append(dict(meta=r, index=idx, snaps=snaps))
    html = (DASH / "index.html").read_text(encoding="utf-8")
    css = (DASH / "style.css").read_text(encoding="utf-8")
    js = (DASH / "app.js").read_text(encoding="utf-8")
    layout = (DASH / "layout.json").read_text(encoding="utf-8")
    phases = json.loads((DASH / "phases.json").read_text(encoding="utf-8")) if (DASH / "phases.json").exists() else None
    gemma = json.loads((DASH / "gemma_names.json").read_text(encoding="utf-8")) if (DASH / "gemma_names.json").exists() else None
    bundle = json.dumps(dict(layout=json.loads(layout), phases=phases, gemma=gemma, runs=runs), separators=(",", ":")).replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="app.js"></script>', f"<script>window.__IC_BUNDLE__={bundle};</script>\n<script>\n{js}\n</script>")
    if args.artifact:                     # claude.ai artifact: the publisher adds doctype/head/body itself; <title> first
        html = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", html, flags=re.I)
        html = re.sub(r'<meta charset="utf-8">\s*|<meta name="viewport"[^>]*>\s*', "", html)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, runs: {[r['meta']['id'] for r in runs]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layout")
    a = sub.add_parser("npz")
    a.add_argument("--npz", default="out/layered_map.npz")
    a.add_argument("--run", default="build_map_72h")
    a.add_argument("--step-min", type=float, default=30.0)
    a.add_argument("--max-boxes", type=int, default=8)
    b = sub.add_parser("bundle")
    b.add_argument("--runs", default="")
    b.add_argument("--max-snaps", type=int, default=100, help="per run; 0 = keep all")
    b.add_argument("--out", default="out/stream/dashboard_bundle.html")
    b.add_argument("--artifact", action="store_true", help="strip doctype/html/head/body for a claude.ai artifact publish")
    ph = sub.add_parser("phases")
    ph.add_argument("--model", default="out/phases/model_final_train.pkl")
    pb = sub.add_parser("publish")
    pb.add_argument("--run", required=True)
    pb.add_argument("--title", default="")
    pb.add_argument("--note", default="")
    pb.add_argument("--max-snaps", type=int, default=80)
    pb.add_argument("--allow-incomplete", action="store_true")
    gm = sub.add_parser("gemma")
    gm.add_argument("--src", default="out/gemma/phase_names.json")
    r = sub.add_parser("register")
    r.add_argument("--run", required=True)
    r.add_argument("--title", required=True)
    r.add_argument("--note", default="")
    args = ap.parse_args()
    if args.cmd == "register":
        register(args.run, args.title, args.note)
    else:
        dict(layout=cmd_layout, npz=cmd_npz, bundle=cmd_bundle, phases=cmd_phases, gemma=cmd_gemma, publish=cmd_publish)[args.cmd](args)


if __name__ == "__main__":
    main()
