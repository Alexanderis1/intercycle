"""Keyframes for the Gemma observer tasks G1-G3, rendered with the scripts/gemma_frames.py renderer (SYNTHETIC).

    python scripts/gemma_keyframes.py g1      # one keyframe per machine camera + true zone polygons  -> out/gemma/g1/
    python scripts/gemma_keyframes.py g2      # keyframes of each discovered phase bin (workstream P)  -> out/gemma/g2/
    python scripts/gemma_keyframes.py g3      # change-detection crops: new machine / parked object / camera bump -> out/gemma/g3/

Nothing here calls Gemma; scripts/gemma_tasks.py does. Ground truth is written next to the images and is never shown to
the model. All images carry the "SYNTHETIC RENDER" burn-in; the only people are simulator people (no real faces).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # keep this a single-core job on the shared laptop

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gemma_frames as GF  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.sim import geometry as G  # noqa: E402
from intercycle.sim.world import World  # noqa: E402
from intercycle.truth.kinematics import Kinematics  # noqa: E402

OUT = ROOT / "out" / "gemma"
EXPOSURE, KSUB = 0.20, 5

# ------------------------------------------------------------------------------------------- renderer plumbing
_orig_build_scene = GF.build_scene


def _build_scene(clip, cam, floor, floor_dark, scn, machine, t, extra_people, quiet_robot=False):
    sc = _orig_build_scene(clip, cam, floor, floor_dark, scn, machine, t, extra_people, quiet_robot)
    fn = getattr(clip, "extra_draw", None)
    if fn is not None:
        fn(sc, t)
    return sc


GF.build_scene = _build_scene        # render_frame looks build_scene up in the module globals


class KinClip:
    """Clip-compatible stand-in: machine truth from World + Kinematics, optional parked vehicles, no recorded movers."""

    def __init__(self, scn, seed, t0, t1, name="kin"):
        self.name, self.scn, self.seed, self.t0, self.t1 = name, scn, seed, t0, t1
        self.world = World(scn, seed, t0, t1)
        self.kin = Kinematics(scn, self.world)
        self.parked: list[tuple] = []          # (id, cls, x, z, yaw, 0.0)
        self.extra_draw = None
        self.sm = None

    def movers_at(self, t):
        return list(self.parked)

    def robot_at(self, t):
        return None


def camera_for(m):
    return GF.fit_camera(m.rect, 6.8 if m.kind == "unload_lifter" else 2.6)


def floors(cam, scn, mid, hatch=True):
    fl, _, _ = GF.floor_image(cam, scn, mid, hatch=hatch)
    fd = Image.fromarray((np.asarray(fl, np.float32) * 0.45).astype(np.uint8))
    return fl, fd


def render(clip, cam, fl_fd, scn, mid, t, seed, cam_id):
    fl, fd = fl_fd
    return GF.render_frame(clip, cam, fl, fd, scn, mid, t, [], EXPOSURE, KSUB, np.random.default_rng(seed), cam_id)


def rect_px(cam, rect):
    """Floor rectangle (x0, z0, x1, z1) -> image polygon in output pixels (the renderer supersamples by SS)."""
    x0, z0, x1, z1 = rect
    P, d = cam.project([[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]])
    return [(float(p[0] / GF.SS), float(p[1] / GF.SS)) for p in P], bool((d > 0.5).all())


def visible_frac(poly, w=GF.W, h=GF.H):
    """Fraction of the polygon's area inside the image (raster, 1/4 resolution)."""
    s = 4
    big = Image.new("L", (3 * w // s, 3 * h // s), 0)
    ImageDraw.Draw(big).polygon([((x + w) / s, (y + h) / s) for x, y in poly], fill=1)
    a = np.asarray(big, bool)
    tot = a.sum()
    if not tot:
        return 0.0
    return float(a[h // s:2 * h // s, w // s:2 * w // s].sum() / tot)


def person_boxes(clip, cam, t, exposure=EXPOSURE, r=0.35, h=1.85):
    """Pixel boxes (output resolution) of every person in view over the exposure [t - exposure, t] (frame GT), for
    the privacy mask applied before any VLM call."""
    out = []
    for tt in (t - exposure, t - exposure / 2, t):
        for (pid, cls, x, z, yaw, sp) in clip.movers_at(tt):
            if cls != "person" or not cam.visible(x, z, 0.3):
                continue
            pts = [(x + dx, y, z + dz) for dx in (-r, r) for dz in (-r, r) for y in (0.0, h)]
            P, d = cam.project(pts)
            if (d <= 0.5).any():
                continue
            P = P / GF.SS
            x0, y0, x1, y1 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
            if x1 < 0 or y1 < 0 or x0 > GF.W or y0 > GF.H:
                continue
            out.append([round(float(v), 1) for v in (x0, y0, x1, y1)])
    return out


def g1_boxes(log=print):
    """Add person boxes to out/gemma/g1/labels.json without re-rendering."""
    scn = load()
    labels = json.loads((OUT / "g1" / "labels.json").read_text(encoding="utf-8"))
    clip = {c.name: c for c in GF.load_clips(scn, OUT / "clips", GF.CLIP_SPECS, False, log)}
    for name, lab in labels.items():
        cam = camera_for(scn.machine(lab["primary"]))
        lab["person_boxes"] = person_boxes(clip[lab["clip"]], cam, lab["sim_t"])
        log(f"  {name}: {len(lab['person_boxes'])} person boxes")
    (OUT / "g1" / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")


# ================================================================================================== G1
def make_g1(log=print):
    scn = load()
    out = OUT / "g1"
    out.mkdir(parents=True, exist_ok=True)
    clips = GF.load_clips(scn, OUT / "clips", GF.CLIP_SPECS, False, log)
    clip = clips[0]                                    # P1_morning: running line, simulator people in the aisles
    labels = {}
    for i, m in enumerate(scn.machines):
        cam = camera_for(m)
        t = clip.t0 + 120.0 + 37.0 * i
        cam_id = f"CAM-{i + 1:02d}"           # generic id: the burn-in must not name the machine (audit #5)
        ff = floors(cam, scn, m.id, hatch=False)    # headline set: no painted hazard hatch (it would trace the answer)
        img = render(clip, cam, ff, scn, m.id, t, 1000 + i, cam_id)
        name = f"cam_{m.id}.png"
        img.save(out / name)
        render(clip, cam, floors(cam, scn, m.id), scn, m.id, t, 1000 + i, cam_id).save(out / f"hatch_{name}")
        zones = []
        for o in scn.machines:
            poly, front = rect_px(cam, o.rect)
            vf = visible_frac(poly) if front else 0.0
            if vf >= 0.25:
                zones.append({"machine_id": o.id, "kind": o.kind, "machine_type": GF.MACH_DESC.get(o.kind, o.kind),
                              "polygon_px": [[round(x, 1), round(y, 1)] for x, y in poly], "visible_frac": round(vf, 3),
                              "primary": o.id == m.id})
        labels[name] = {"person_boxes": person_boxes(clip, cam, t), "camera_id": cam_id, "hatched_variant": f"hatch_{name}", "primary": m.id, "sim_t": round(t, 1), "clip": clip.name,
                        "size": [GF.W, GF.H], "zones": zones}
        log(f"  {name}: {len(zones)} zones in view")
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


# ================================================================================================== G2
G2_PER_BIN = 3


def make_g2(log=print):
    from intercycle.observer import phases as P
    scn = load()
    out = OUT / "g2"
    out.mkdir(parents=True, exist_ok=True)
    te = dict(np.load(ROOT / "out" / "phases" / "final_test.npz", allow_pickle=True))
    pm = P.PhaseModel.load(ROOT / "out" / "phases" / "model_final_train.pkl", scn)
    tic = time.time()
    bins, states, _, cl = pm.decode_features(te["T"], te["X"], te["F"])
    log(f"decoded {len(te['T'])} ticks in {time.time() - tic:.0f} s")
    names = te["lab_names"]
    T = te["T"]
    seed, t0, t1 = int(te["seed"]), float(te["t0"]), float(te["t1"])
    clip = KinClip(scn, seed, t0, t1, "final_test")
    rng = np.random.default_rng(23)
    labels = {"bins": {}, "line": {}}
    for i, mid in enumerate(te["mids"]):
        mid = str(mid)
        m = scn.machine(mid)
        fam = str(te["fams"][i])
        mp = pm.machines[mid]
        bn = pm.bin_names(mid)
        truth = names[te["lab"][:, i]]
        coarse = coarse_truth(fam, truth, te["frac"][:, i])
        cam = camera_for(m)
        ff = floors(cam, scn, mid)
        b = bins[:, i]
        for k in range(mp.n_bins):
            idx = np.flatnonzero(b == k)
            if len(idx) == 0:
                continue
            vals, cnt = np.unique(coarse[idx], return_counts=True)
            maj = str(vals[np.argmax(cnt)])
            fv, fc = np.unique(truth[idx], return_counts=True)
            # keyframes: instants from different occurrences of the bin, away from its edges where possible
            run_starts = np.flatnonzero(np.diff(np.r_[-9, idx]) > 1)
            runs = np.split(idx, run_starts[1:]) if len(run_starts) > 1 else [idx]
            runs = [r for r in runs if len(r)]
            pick = rng.choice(len(runs), size=min(G2_PER_BIN, len(runs)), replace=False)
            kf = []
            for j, ri in enumerate(sorted(pick)):
                r = runs[ri]
                tick = int(r[len(r) // 2])
                t = float(T[tick])
                img = render(clip, cam, ff, scn, mid, t, 5000 + 31 * i + 7 * k + j, f"CAM-{mid}")
                fn = f"{mid}_b{k}_{j}.png"
                img.save(out / fn)
                kf.append({"file": fn, "tick": tick, "sim_t": round(t, 2), "truth_coarse": str(coarse[tick]),
                           "truth_fine": str(truth[tick])})
            labels["bins"][f"{mid}:{k}"] = {
                "machine": mid, "family": fam, "machine_type": GF.MACH_DESC.get(m.kind, m.kind), "bin": k,
                "bin_name": bn[k], "n_ticks": int(len(idx)), "truth_majority": maj,
                "truth_purity": round(float(cnt.max() / cnt.sum()), 3),
                "truth_fine_counts": {str(a): int(c) for a, c in zip(fv, fc)},
                "keyframes": kf, "median_dur_s": round(float(np.median([len(r) for r in runs])) / 2.0, 1)}
            log(f"  {mid} bin {k} ({bn[k]}): {len(idx)} ticks, truth {maj} ({cnt.max() / cnt.sum():.2f}), {len(kf)} keyframes")
    # line-state clusters: centroid features (text) + truth majority
    lt = np.array(P.LINE_STATES)[te["line"]]
    for c, nm in enumerate(pm.line.names):
        msk = cl == c
        if not msk.any():
            continue
        vals, cnt = np.unique(lt[msk], return_counts=True)
        labels["line"][str(c)] = {"cluster": c, "observer_rule_name": nm, "n_ticks": int(msk.sum()),
                                  "truth_majority": str(vals[np.argmax(cnt)]),
                                  "truth_purity": round(float(cnt.max() / cnt.sum()), 3),
                                  "centre": dict(zip(P.LINE_FEATS, map(float, np.round(pm.line.raw_centres[c], 3)))),
                                  "median_run_s": _median_run_s(msk)}
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


def _median_run_s(msk):
    d = np.diff(np.r_[0, msk.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return round(float(np.median(e - s)) / 2.0, 1) if len(s) else 0.0


def coarse_truth(fam, truth, frac):
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase_discovery as PD
    return PD.coarse(fam, truth, frac)


# ================================================================================================== G3
NM_T0, NM_T1 = 712800.0, 741600.0


def make_g3(log=print):
    from intercycle.truth.newmachine import NewMachine, load_installs
    scn = load()
    out = OUT / "g3"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_installs(ROOT / "scenario" / "bodyshop_newmachine.yaml")[0]
    t_inst = float(cfg["t_install_s"])
    clip = KinClip(scn, 0, NM_T0, NM_T1, "newmachine_day8")
    rng = np.random.default_rng(31)
    labels = {}
    # free-floor spots north of the line / around the cells where a change blob can appear (same region as NEW1)
    # free floor between / beside the off-line cells (neighbouring zones and robots in view give context); NEW1 first
    spots = [(72.0, 41.0), (31.0, 41.0), (36.0, 42.5), (56.0, 41.0), (60.0, 43.0), (66.0, 42.0), (10.0, 42.0), (29.0, 44.0)]
    # camera-bump crops are centred on structured areas (zone corners, the line at the walkway), as camera 6 / SEAL1
    bump_spots = [(27.0, 40.0), (40.0, 42.0), (52.0, 40.0), (15.0, 42.0), (52.5, 31.0), (30.0, 32.0), (65.0, 32.0), (45.0, 33.0)]

    def cam_for_rect(rect):
        return GF.fit_camera(rect, 2.6)

    def nearest_machine(x, z):
        return min(scn.machines, key=lambda o: math.hypot(o.x - x, o.z - z)).id

    def shot(kind, k, x, z, t_ref, t_after, setup, cam_shift=None):
        rect = (x - 3.5, z - 3.0, x + 3.5, z + 3.0)
        cam = cam_for_rect(rect)
        mid = nearest_machine(x, z)
        ff = floors(cam, scn, mid)
        cam_id = f"CAM-CHG{k:02d}"
        clip.parked, clip.extra_draw = [], None
        ref = render(clip, cam, ff, scn, mid, t_ref, 9000 + k, cam_id)
        setup()
        if cam_shift is not None:
            dx, dz, dyaw = cam_shift
            tgt = cam.pos + cam.f * 10.0
            c, s = math.cos(dyaw), math.sin(dyaw)
            off = tgt - cam.pos
            off = np.array([c * off[0] + s * off[2], off[1], -s * off[0] + c * off[2]])
            cam2 = GF.Cam(cam.pos + np.array([dx, 0.0, dz]), cam.pos + np.array([dx, 0.0, dz]) + off)
            ff2 = floors(cam2, scn, mid)
        else:
            cam2, ff2 = cam, ff
        after = [render(clip, cam2, ff2, scn, mid, ta, 9100 + 10 * k + j, cam_id) for j, ta in enumerate(t_after)]
        files = [f"chg{k:02d}_ref.png"] + [f"chg{k:02d}_after{j}.png" for j in range(len(after))]
        for f, im in zip(files, [ref] + after):
            im.save(out / f)
        labels[f"chg{k:02d}"] = {"truth": kind, "x": x, "z": z, "rect": list(rect), "files": files, "t_ref": t_ref,
                                 "t_after": list(t_after), "camera_shift": list(cam_shift) if cam_shift else None,
                                 "nearest_zone": mid}
        log(f"  chg{k:02d}: {kind} at ({x:.1f}, {z:.1f})")

    k = 0
    # (a) new moving machine: NEW1 itself (the M experiment) at several times, plus the same arm installed elsewhere
    for j in range(8):
        x, z = spots[j]
        c = dict(cfg, x=x, z=z)
        nm = NewMachine(c, seed=j, t_end=NM_T1 + 600)
        ta = t_inst + 1800 + 600 * j + float(rng.uniform(0, 40))
        # pick the two after-frames inside one active cycle so the arm visibly moved between them
        i0 = int(np.searchsorted(nm.starts, ta)) % len(nm.starts)
        s0 = float(nm.starts[i0])
        t_after = (s0 + 0.15 * nm.acts[i0], s0 + 0.55 * nm.acts[i0])

        def setup(nm=nm):
            def draw(sc, t, nm=nm):
                draw_new_machine(sc, nm, t)
            clip.extra_draw = draw
        shot("new_moving_machine", k, x, z, t_inst - 3600.0 - 300 * j, t_after, setup)
        k += 1
    # (b) parked object: a tugger train, forklift or AGV left standing in static space
    for j in range(8):
        x, z = spots[(j + 3) % len(spots)]
        x += float(rng.uniform(-1.0, 1.0))
        cls = ("forklift", "tugger", "agv", "forklift", "agv", "tugger", "forklift", "agv")[j]
        yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2])) + float(rng.uniform(-0.15, 0.15))
        ta = NM_T0 + 3000 + 2500 * j

        def setup(cls=cls, x=x, z=z, yaw=yaw):
            clip.parked = [(f"PARK{j}", cls, x, z, yaw, 0.0)]
        shot("parked_object", k, x, z, ta - 1800.0, (ta, ta + 20.0), setup)
        k += 1
    # (c) camera bump: the camera was knocked (translation 0.5-1.5 m and/or a few degrees of yaw); scene unchanged
    for j in range(8):
        x, z = bump_spots[j]
        sh = (float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1])), float(rng.uniform(-1.0, 1.0)),
              math.radians(float(rng.uniform(-6, 6))))
        ta = NM_T0 + 4000 + 2500 * j
        shot("camera_bump", k, x, z, ta - 1800.0, (ta, ta + 20.0), lambda: None, cam_shift=sh)
        k += 1
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


def make_g3_events(log=print):
    """Crops framed by workstream M's REAL change events (out/newmachine/result.json): the change detector's own rect
    and camera, not the truth. Includes its false alarm (the flicker source: nothing physical changed)."""
    from intercycle.truth.newmachine import NewMachine, load_installs
    scn = load()
    out = OUT / "g3"
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / "out" / "newmachine" / "result.json").read_text(encoding="utf-8"))
    cfg = load_installs(ROOT / "scenario" / "bodyshop_newmachine.yaml")[0]
    nm = NewMachine(cfg, seed=0, t_end=NM_T1 + 600)
    nx0, nz0, nx1, nz1 = nm.rect()
    clip = KinClip(scn, 0, NM_T0, NM_T1, "newmachine_day8")
    clip.extra_draw = lambda sc, t: draw_new_machine(sc, nm, t)
    labels = {}
    for i, e in enumerate(res["events"]):
        if e["kind"] == "change_detected":
            rect = e["rect"]
            over_new = not (rect[2] < nx0 or rect[0] > nx1 or rect[3] < nz0 or rect[1] > nz1) and e["t"] >= nm.t_install
            truth = "new_moving_machine" if over_new else "no_visible_change"
            cam = GF.fit_camera(tuple(rect), 2.6)
            cam2 = cam
        elif e["kind"] == "camera_bump":
            cx, cz = scn.cameras[e["camera"]]
            cam = GF.Cam((cx, GF.CAM_H, cz), (cx - 3.0, 0.0, cz - 8.0))
            b = res["bump"]["offset"]
            cam2 = GF.Cam(cam.pos + np.array([b[0], 0.0, b[1]]), np.array([cx - 3.0, 0.0, cz - 8.0]) + np.array([b[0], 0.0, b[1]]))
            truth, rect = "camera_bump", None
        else:
            continue
        mid = min(scn.machines, key=lambda o: math.hypot(o.x - cam.pos[0], o.z - (cam.pos[2] + 6))).id
        t = float(e["t"])
        key = f"evt{i:02d}"
        cam_id = f"CAM-EVT{i:02d}"
        t_ref = t - 1800.0 if e["kind"] == "change_detected" else res["bump"]["t"] - 600.0
        ref = render(clip, cam, floors(cam, scn, mid), scn, mid, t_ref, 9500 + i, cam_id)
        ff2 = floors(cam2, scn, mid)
        after = [render(clip, cam2, ff2, scn, mid, ta, 9600 + 10 * i + j, cam_id) for j, ta in enumerate((t, t + 10.0))]
        files = [f"{key}_ref.png", f"{key}_after0.png", f"{key}_after1.png"]
        for f, im in zip(files, [ref] + after):
            im.save(out / f)
        labels[key] = {"truth": truth, "event": e["kind"], "event_t": t, "rect": rect, "files": files, "t_ref": t_ref,
                       "t_after": [t, t + 10.0], "framing": "M change event (detector rect / camera), not truth",
                       "camera": e.get("camera")}
        log(f"  {key}: {e['kind']} -> truth {truth}")
    (out / "labels_events.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels


def draw_new_machine(sc, nm, t):
    c = nm.parts(t)
    if len(c) == 0:
        return
    b, el, tl = c[0, :2], c[2, :2], c[4, :2]
    sc.box(b[0], b[1], 0.9, 0.9, 0.0, 0.0, 0.9, (74, 76, 84), top=(100, 102, 110))
    GF.draw_arm(sc, {"base": tuple(b), "elbow_xy": tuple(el), "tool_xy": tuple(tl)}, "pick_place", "roller")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", choices=["g1", "g2", "g3", "g1boxes", "g3events"])
    a = ap.parse_args(argv)
    tic = time.time()
    {"g1": make_g1, "g2": make_g2, "g3": make_g3, "g1boxes": g1_boxes, "g3events": make_g3_events}[a.task]()
    print(f"{a.task} keyframes in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
