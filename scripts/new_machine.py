"""New-machine experiment: a machine is installed on day 8; the observer is not told (SYNTHETIC).

    python scripts/new_machine.py [--hours 8] [--no-bump] [--live]

Plant: scenario/bodyshop.yaml + the overlay scenario/bodyshop_newmachine.yaml (truth only: NEW1, a pick-and-place arm
installed at day 8 08:00, 30 min of irregular commissioning cycles, then a 50 s cycle). The observer starts from the
1-week map (out/timeline/map_1w.npz, else the merged week windows) and keeps learning; observer/change.py watches for
unexplained persistent motion in static space.

Stress on the camera side (all SYNTHETIC assumptions): transient clutter (60 spurious motion blobs / h site-wide,
0.5-3 s each), one aperiodic flicker source for 30 min (06:30-07:00 at x 88, z 44, a swinging cable / reflection),
and camera 6 (54, 46; it sees SEAL1) knocked by (1.5, 1.0) m at 12:00 (its homography is stale from then on).

Scored with truth (this script may import it; change.py never does):
  * detection delay: first UNKNOWN cell over the new machine, and the `change_detected` report, after installation;
  * false alarms / h: change_detected reports not over the new machine (by cause: flicker, camera bump, clutter);
  * time to re-converge: machine_spawned (L1 model created) and machine_converged (envelope learned, UNKNOWN lifted);
    learned period vs the true 50 s; IoU of the learned envelope with the true swept area;
  * unsafe-free rate on the new machine's true cells (0.25 m, any overlap) before detection, during the transition
    (UNKNOWN until converged) and after convergence: predicted occupied = UNKNOWN cells + the spawned machine's
    broadcast boxes at the observer's phase (bins widened as predict_obstacles does);
  * camera bump: whether it was recognised (camera_bump), how long it took, and that no machine was spawned from it.

Outputs: out/newmachine/result.json, out/newmachine/*.png, stream run out/stream/newmachine/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_map as B  # noqa: E402

from intercycle.observer import mapping as M  # noqa: E402
from intercycle.observer.change import ChangeDetector  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.truth.newmachine import NewMachine, load_installs  # noqa: E402

OUT = ROOT / "out" / "newmachine"
H, DAY = 3600.0, 86_400.0
VARIANT = ROOT / "scenario" / "bodyshop_newmachine.yaml"


def prior_map() -> M.LayeredMap:
    f = ROOT / "out" / "timeline" / "map_1w.npz"
    if f.exists():
        return M.LayeredMap.load(f)
    parts = sorted((ROOT / "out" / "timeline" / "parts").glob("w[0-9][0-9][0-9].npz"))
    if not parts:
        raise SystemExit("no prior map: run scripts/map_timeline.py build first")
    m = M.LayeredMap.load(parts[0])
    for p in parts[1:]:
        m.merge(M.LayeredMap.load(p))
    return m


class EvalGrid:
    """0.25 m grid over the new machine's true hazard box (+1 m) for scoring."""

    def __init__(self, nm: NewMachine, res: float = 0.25):
        x0, z0, x1, z1 = nm.rect(pad=1.0)
        self.x0, self.z0, self.res = x0, z0, res
        self.nx, self.nz = int(math.ceil((x1 - x0) / res)), int(math.ceil((z1 - z0) / res))
        gx = x0 + (np.arange(self.nx) + 0.5) * res
        gz = z0 + (np.arange(self.nz) + 0.5) * res
        self.cx, self.cz = np.meshgrid(gx, gz, indexing="ij")

    def truth(self, circles: np.ndarray) -> np.ndarray:
        c = np.asarray(circles, float).reshape(-1, 3)
        out = np.zeros((self.nx, self.nz), bool)
        if len(c):
            _, ix, iz = M.raster(c[:, 0], c[:, 1], c[:, 2], self.x0, self.z0, self.nx, self.nz, self.res, centre=False)
            out[ix, iz] = True
        return out

    def from_site(self, U: np.ndarray, site: dict) -> np.ndarray:
        ix = np.clip(np.floor((self.cx - site["x0"]) / site["res"]).astype(int), 0, site["nx"] - 1)
        iz = np.clip(np.floor((self.cz - site["z0"]) / site["res"]).astype(int), 0, site["nz"] - 1)
        return U[ix, iz]

    def from_boxes(self, boxes: np.ndarray) -> np.ndarray:
        out = np.zeros((self.nx, self.nz), bool)
        for x0, z0, x1, z1 in np.asarray(boxes).reshape(-1, 4):
            out |= (self.cx >= x0) & (self.cx <= x1) & (self.cz >= z0) & (self.cz <= z1)
        return out


def run(a) -> dict:
    scn = load()
    cfg = load_installs(VARIANT)[0]
    t_inst = float(cfg["t_install_s"])
    t0 = t_inst - a.pre_h * H
    t1 = t0 + a.hours * H
    nm = NewMachine(cfg, seed=0, t_end=t1 + 600)
    flicker = (88.0, 44.0, 0.5, t0 + 0.5 * H, t0 + 1.0 * H, 0.3)
    bump_t = t_inst + 4 * H
    bumps = [] if a.no_bump else [(6, 1.5, 1.0, bump_t)]
    shim_kw = dict(clutter_per_h=a.clutter, persistent=[flicker], bumps=bumps)
    mp = prior_map()
    mp.unknown_rects = []
    mp.cam_offline = set()
    det = ChangeDetector(mp)
    eg = EvalGrid(nm)
    site = mp.spec["site"]
    from intercycle.observer.stream import StreamWriter
    OUT.mkdir(parents=True, exist_ok=True)
    sw = StreamWriter(a.run, out_root=ROOT / "out" / "stream", live=a.live,
                      meta=dict(kind="newmachine", label="SYNTHETIC: simulated plant", t_install=t_inst, machine=cfg["id"],
                                note="the observer is not told about the new machine; camera noise / clutter assumed"))
    S = dict(events=[], ticks=[], first_unknown=math.nan)
    acc = {k: dict(tp=0, fn=0, n=0, fo=0) for k in ("pre_detection", "transition", "learned")}
    VARIANTS = (("centre", 0), ("overlap", 1), ("overlap_margin_0.25m", 1), ("overlap_margin_0.5m", 1))
    accv = {k: {v: dict(tp=0, fn=0, fo=0) for v, _ in VARIANTS} for k in acc}
    tl = []
    last_snap = [-math.inf]
    t_wall = time.time()

    def hook(k, tc, states, obs, cond):
        evs = det.step(obs, cond)
        for e in evs:
            S["events"].append(dict(e))
            e = dict(e)
            kind = e.pop("kind")
            tt = e.pop("t")
            sw.event(t_sim=tt, kind=kind, **e)
            print(f"  {tt / H - 8 * 24:7.3f} h (day 8)  {kind}  {e}", flush=True)
        if k % 2:
            return
        tcirc = states[-1].parts.get(cfg["id"], np.zeros((0, 3)))
        truth = eg.truth(tcirc)
        U = eg.from_site(mp.unknown_mask(), site)
        pred = U.copy()
        cd = next((c for c in det.cands if c.spawned), None)
        if cd is not None:
            i = cd.machine_index
            ph = det.phase_now().get(cd.machine_id)
            bins = mp.predicted_bins(i, ph, cond.prod_state, 0.0)
            pred |= eg.from_boxes(mp.box_cover(cd.machine_id, bins, 0.05, max_boxes=12))
        conv = cd is not None and cd.converged
        if tc >= t_inst and truth.any():
            stage = "pre_detection" if not np.isfinite(S["first_unknown"]) else ("learned" if conv else "transition")
            if stage == "pre_detection" and (U & truth).any():
                S["first_unknown"] = tc
                stage = "transition"
            a_ = acc[stage]
            a_["tp"] += int((truth & pred).sum())
            a_["fn"] += int((truth & ~pred).sum())
            a_["fo"] += int((pred & ~truth).sum())
            a_["n"] += 1
            tc_ = M.mask_from_circles(tcirc, eg.x0, eg.z0, eg.nx, eg.nz, eg.res)
            for vn, _ in VARIANTS:
                tt = tc_ if vn == "centre" else truth
                pp = M.dilate(pred, 1) if vn.endswith("0.25m") else M.dilate(pred, 2) if vn.endswith("0.5m") else pred
                b_ = accv[stage][vn]
                b_["tp"] += int((tt & pp).sum())
                b_["fn"] += int((tt & ~pp).sum())
                b_["fo"] += int((pp & ~tt).sum())
        if tc >= t_inst and not np.isfinite(S["first_unknown"]) and U.any() and (U & eg.truth(np.array([[nm.x, nm.z, 3.5]]))).any():
            S["first_unknown"] = tc
        if k % 20 == 0:
            tl.append(dict(t=tc, unknown_m2=float(U.sum() * eg.res ** 2), pred_m2=float(pred.sum() * eg.res ** 2),
                           truth_m2=float(truth.sum() * eg.res ** 2), spawned=cd is not None, converged=conv,
                           n_cands=len(det.cands), bumped=sorted(det.bumped)))
        if tc - last_snap[0] >= a.snap_every_s:
            last_snap[0] = tc
            tod = M.tod_bin(tc)
            l2, h2 = mp.traffic(M.VEHICLE, tod, cond.prod_state)
            l3, h3 = mp.traffic(M.PERSON, tod, cond.prod_state)
            l1 = {}
            for j, d in enumerate(mp.spec["machines"]):            # every phase bin (as week1), the current one is in phase_now
                l1[d["id"]] = [(b, *(round(float(v), 2) for v in bx)) for b in range(M.N_BINS)
                               for bx in mp.box_cover(d["id"], b, 0.05, max_boxes=8)]
            pn = {d["id"]: (int(cond.phase_bin[j]) if j < len(cond.phase_bin) and cond.phase_bin[j] >= 0 else None)
                  for j, d in enumerate(mp.spec["machines"])}
            if cd is not None:
                pb = det.phase_bin(cd, tc)
                pn[cd.machine_id] = pb if pb >= 0 else None
            uf = {kk: (v["fn"] / max(v["tp"] + v["fn"], 1)) for kk, v in acc.items()}
            sw.snapshot(t_sim=tc, t_obs=tc - t0, prod_state=M.PROD_NAMES[cond.prod_state], grid=dict(site), l1=l1,
                        l2_occ=l2.T, l2_head=h2.T, l3_occ=l3.T, l3_head=h3.T, unknown=mp.unknown_mask(tod, cond.prod_state).T,
                        phase_now=pn, metrics={f"unsafe_free_{kk}": float(v) for kk, v in uf.items()})
            print(f"  snapshot day8 {tc / H - 192:.2f} h  cands {len(det.cands)}  spawned {cd is not None}  "
                  f"wall {time.time() - t_wall:.0f} s", flush=True)

    B.run_window(0, t0 - 1200.0, t1, scn=scn, map_=mp, tick_hook=hook, extra_parts=lambda ts: {cfg["id"]: nm.parts(ts)},
                 shim_kw=shim_kw, shim_seed=8001)
    sw.close()
    # ---------------------------------------------------------------- summarise
    ev = S["events"]
    trect = nm.rect(pad=0.5)

    def over_new(e):
        r = e.get("rect")
        return bool(r) and not (r[2] < trect[0] or r[0] > trect[2] or r[3] < trect[1] or r[1] > trect[3])

    det_ev = [e for e in ev if e["kind"] == "change_detected"]
    true_det = [e for e in det_ev if over_new(e) and e["t"] >= t_inst]
    false_det = [e for e in det_ev if not (over_new(e) and e["t"] >= t_inst)]

    def cause(e):
        x, z = e.get("x", 0), e.get("z", 0)
        if math.hypot(x - flicker[0], z - flicker[1]) < 4:
            return "flicker source"
        if bumps and e["t"] >= bump_t and det._cam_of_xy(x, z) == 6:
            return "camera bump"
        return "other / clutter"

    spawn = [e for e in ev if e["kind"] == "machine_spawned"]
    conv = [e for e in ev if e["kind"] == "machine_converged"]
    bump_ev = [e for e in ev if e["kind"] == "camera_bump"]
    res = dict(
        label="SYNTHETIC", install_t=t_inst, run=[t0, t1], hours=a.hours, clutter_per_h=a.clutter, flicker=flicker,
        bump=dict(camera=6, offset=[1.5, 1.0], t=bump_t) if bumps else None,
        detection=dict(first_unknown_delay_s=S["first_unknown"] - t_inst if np.isfinite(S["first_unknown"]) else None,
                       change_detected_delay_s=(true_det[0]["t"] - t_inst) if true_det else None),
        false_alarms=dict(n=len(false_det), per_h=len(false_det) / a.hours, by_cause={c: sum(1 for e in false_det if cause(e) == c)
                                                                                      for c in ("flicker source", "camera bump", "other / clutter")}),
        spawned=[dict(machine=e["machine"], t_after_install_s=e["t"] - t_inst, period_s=e["period_s"], over_new=over_new(e)) for e in spawn],
        false_spawns=sum(1 for e in spawn if not over_new(e)),
        converged=[dict(machine=e["machine"], t_after_install_s=e["t"] - t_inst, period_s=e["period_s"]) for e in conv],
        true_period_s=nm.P,
        camera_bump=dict(recognised=bool(bump_ev), delay_s=(bump_ev[0]["t"] - bump_t) if bump_ev else None,
                         estimated_offset=[bump_ev[0]["dx"], bump_ev[0]["dz"]] if bump_ev else None) if bumps else None,
        unsafe_free={k: dict(rate=(v["fn"] / max(v["tp"] + v["fn"], 1)) if v["n"] else None, instants=v["n"],
                             truly_occupied_cells=v["tp"] + v["fn"], missed_cells=v["fn"],
                             false_occupied_m2_mean=v["fo"] * 0.0625 / max(v["n"], 1)) for k, v in acc.items()},
        unsafe_free_variants={k: {vn: dict(rate=(b["fn"] / max(b["tp"] + b["fn"], 1)) if acc[k]["n"] else None,
                                           false_occupied_m2_mean=b["fo"] * eg.res ** 2 / max(acc[k]["n"], 1))
                                  for vn, b in vv.items()} for k, vv in accv.items()},
        events=ev, timeline=tl)
    # envelope quality of the spawned model vs the true swept area (union over the learned period)
    cd = next((c for c in det.cands if c.spawned), None)
    if cd is not None:
        env = mp.envelope_union(cd.machine_id, M.ALL_BINS)
        i = cd.machine_index
        d = mp.spec["machines"][i]
        sw_t = np.zeros((d["nx"], d["nz"]), bool)
        for tt in np.arange(t1 - 600, t1, 0.25):
            c = nm.parts(tt)
            if len(c):
                sw_t |= M.mask_from_circles(c, d["x0"], d["z0"], d["nx"], d["nz"], mp.spec["env_res"])
        inter = (env & sw_t).sum()
        res["learned_envelope"] = dict(iou=float(inter / max((env | sw_t).sum(), 1)), recall=float(inter / max(sw_t.sum(), 1)),
                                       precision=float(inter / max(env.sum(), 1)), area_m2=float(env.sum() * 0.0625))
    mp.save(OUT / "map_after.npz")
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))
    fig(res, OUT / "new_machine_timeline.png")
    return res


def fig(res: dict, path: Path) -> None:
    plt = B._plt()
    tl = res["timeline"]
    t_inst = res["install_t"]
    t = np.array([r["t"] for r in tl])
    x = (t - t_inst) / 60.0
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=130)
    ax.plot(x, [r["truth_m2"] for r in tl], color="#1f2933", lw=0.6, alpha=0.6, label="new machine: truly occupied area now (m2)")
    ax.plot(x, [r["unknown_m2"] for r in tl], color="#9aa5b1", lw=1.6, label="UNKNOWN area over it (m2)")
    ax.plot(x, [r["pred_m2"] for r in tl], color="#b42318", lw=1.0, ls="--", label="predicted occupied (UNKNOWN + learned boxes)")
    for e in res["events"]:
        if e["kind"] in ("change_detected", "machine_spawned", "machine_converged", "camera_bump", "change_cleared", "change_explained"):
            xx = (e["t"] - t_inst) / 60.0
            ax.axvline(xx, color="#0b6e69" if e["kind"] != "camera_bump" else "#b45309", lw=0.8, alpha=0.7)
            ax.text(xx, 1.0, e["kind"].replace("_", " "), transform=ax.get_xaxis_transform(), rotation=90, fontsize=6,
                    va="top", ha="right", color="#6b7785")
    ax.axvline(0, color="#1f2933", lw=1.2)
    ax.set_xlabel("minutes after installation (truth)", fontsize=8)
    ax.set_ylabel("m2", fontsize=8)
    uf = res["unsafe_free"]
    ax.set_title("SYNTHETIC - new machine: detection -> UNKNOWN -> learned.  unsafe-free: " +
                 ", ".join(f"{k} {v['rate']:.4f}" if v["rate"] is not None else f"{k} n/a" for k, v in uf.items()), fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--pre-h", type=float, default=2.0, help="hours simulated before the installation")
    ap.add_argument("--clutter", type=float, default=60.0, help="spurious motion blobs per hour site-wide")
    ap.add_argument("--no-bump", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--run", default="newmachine")
    ap.add_argument("--snap-every-s", type=float, default=300.0)
    a = ap.parse_args()
    r = run(a)
    print(json.dumps({k: v for k, v in r.items() if k not in ("events", "timeline")}, indent=1, default=float))


if __name__ == "__main__":
    main()
