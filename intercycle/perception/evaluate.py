"""Accuracy of the perception pipeline against SmartSpaces ground truth.

* person detection precision / recall at IoU 0.5 (per camera, per sampled frame)
* floor-localisation error of matched detections (foot point -> floor vs GT 3D location)
* tracking: MOTA (CLEAR-MOT, IoU 0.5) and IDF1 (own implementation; ``motmetrics``, MIT, used as a
  cross-check when installed)
* per-zone occupancy MAE vs GT-derived occupancy (per sampled step and per time bin)

Ground truth is the compact tables produced by ``intercycle.perception.gt``.  GT persons whose
visible 2D box is shorter than ``min_h`` px are treated as *ignore regions* (a detection on them
is neither a true nor a false positive and they are not counted as misses); the headline numbers
are for ``min_h`` = 20 px and the table also reports ``min_h`` = 0 (everything the annotator marked
visible, including heavily occluded and few-pixel persons).

    python -m intercycle.perception.evaluate --run-dir out/perception/demo \
        --gt-dir data/smartspaces/.../gt --scene-dir data/smartspaces/.../Warehouse_000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .geometry import assign_zones, foot_points, load_calibration, load_zones
from .occupancy import occupancy_bins

__all__ = [
    "iou_matrix",
    "detection_metrics",
    "tracking_metrics",
    "occupancy_metrics",
    "evaluate_run",
    "sampled_frames",
    "main",
]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def sampled_frames(metas: list[dict]) -> np.ndarray:
    stride = int(metas[0]["stride"])
    f0 = min(m["frame_first"] for m in metas if m["frame_first"] is not None)
    f1 = max(m["frame_last"] for m in metas if m["frame_last"] is not None)
    return np.arange(f0, f1 + 1, stride)


def _split_gt(g: pd.DataFrame, min_h: float, min_w: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = g["y2"] - g["y1"]
    w = g["x2"] - g["x1"]
    valid = (h >= min_h) & (w >= min_w) & (h > 0)
    return g[valid], g[~valid]


def _hungarian(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    if iou.size == 0:
        return []
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= thr]


def _frame_groups(df: pd.DataFrame, cam_col: str = "camera"):
    return {k: g for k, g in df.groupby([cam_col, "frame"], sort=False)}


def detection_metrics(
    dets: pd.DataFrame,
    gt_boxes: pd.DataFrame,
    frames: np.ndarray,
    cams: list[str],
    cls: str = "person",
    iou_thr: float = 0.5,
    min_h: float = 20.0,
    min_score: float | None = None,
) -> dict:
    """Precision / recall / F1 at ``iou_thr`` plus the list of matches (for localisation error)."""
    d = dets[dets["cls"] == cls]
    if min_score is not None:
        d = d[d["score"] >= min_score]
    g = gt_boxes[(gt_boxes["cls"] == cls) & gt_boxes["frame"].isin(frames) & gt_boxes["camera"].isin(cams)]
    gv, gi = _split_gt(g, min_h)
    dg, gvg, gig = _frame_groups(d), _frame_groups(gv), _frame_groups(gi)
    tp = fp = fn = ign = 0
    per_cam = {c: [0, 0, 0] for c in cams}
    matches: list[tuple] = []  # camera, frame, x1,y1,x2,y2 (det), object_id
    box = ["x1", "y1", "x2", "y2"]
    for cam in cams:
        for fr in frames:
            key = (cam, int(fr))
            D = dg.get(key)
            V = gvg.get(key)
            I = gig.get(key)
            nd = 0 if D is None else len(D)
            nv = 0 if V is None else len(V)
            if nd == 0:
                fn += nv
                per_cam[cam][2] += nv
                continue
            db = D[box].to_numpy()
            if nv:
                iou = iou_matrix(db, V[box].to_numpy())
                pairs = _hungarian(iou, iou_thr)
            else:
                pairs = []
            m_d = {i for i, _ in pairs}
            m_g = {j for _, j in pairs}
            for i, j in pairs:
                matches.append((cam, int(fr), *db[i], int(V.iloc[j]["object_id"])))
            left = [i for i in range(nd) if i not in m_d]
            n_ign = 0
            if left and I is not None and len(I):
                iou2 = iou_matrix(db[left], I[box].to_numpy())
                n_ign = len(_hungarian(iou2, iou_thr))
            t, f_, n_ = len(pairs), len(left) - n_ign, nv - len(m_g)
            tp += t
            fp += f_
            fn += n_
            ign += n_ign
            per_cam[cam][0] += t
            per_cam[cam][1] += f_
            per_cam[cam][2] += n_
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and prec == prec and rec == rec else 0.0
    return {
        "min_h_px": min_h,
        "iou_thr": iou_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_dets": ign,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_gt": tp + fn,
        "per_camera": {
            c: {"tp": v[0], "fp": v[1], "fn": v[2],
                "precision": v[0] / (v[0] + v[1]) if v[0] + v[1] else float("nan"),
                "recall": v[0] / (v[0] + v[2]) if v[0] + v[2] else float("nan")}
            for c, v in per_cam.items()
        },
        "_matches": matches,
    }


def localisation_error(matches: list[tuple], gt_objs: pd.DataFrame, calib: dict) -> dict:
    """Floor error (m) between the projected foot point of matched detections and the GT 3D location."""
    if not matches:
        return {"n": 0}
    m = pd.DataFrame(matches, columns=["camera", "frame", "x1", "y1", "x2", "y2", "object_id"])
    ref = gt_objs.drop_duplicates(["frame", "object_id"])[["frame", "object_id", "wx", "wy"]]
    m = m.merge(ref, on=["frame", "object_id"], how="inner")
    errs = []
    for cam, g in m.groupby("camera"):
        fp = foot_points(g[["x1", "y1", "x2", "y2"]].to_numpy())
        p = calib[cam].image_to_floor(fp)
        errs.append(np.hypot(p[:, 0] - g["wx"].to_numpy(), p[:, 1] - g["wy"].to_numpy()))
    e = np.concatenate(errs) if errs else np.zeros(0)
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "median_m": float(np.median(e)), "mean_m": float(e.mean()),
            "p90_m": float(np.percentile(e, 90)), "within_1m": float((e <= 1.0).mean())}


# --------------------------------------------------------------------------- tracking


def _prep_tracking_frame(P, V, I, box, iou_thr):
    """Return (pred boxes, pred ids, gt boxes, gt ids) after removing predictions that sit on ignore GT."""
    pb = P[box].to_numpy() if P is not None else np.zeros((0, 4))
    pid = P["track_id"].to_numpy() if P is not None else np.zeros(0, int)
    gb = V[box].to_numpy() if V is not None else np.zeros((0, 4))
    gid = V["object_id"].to_numpy() if V is not None else np.zeros(0, int)
    if len(pb) and I is not None and len(I):
        iou = iou_matrix(pb, gb) if len(gb) else np.zeros((len(pb), 0))
        matched = {i for i, _ in _hungarian(iou, iou_thr)}
        left = [i for i in range(len(pb)) if i not in matched]
        if left:
            drop = {left[i] for i, _ in _hungarian(iou_matrix(pb[left], I[box].to_numpy()), iou_thr)}
            keep = [i for i in range(len(pb)) if i not in drop]
            pb, pid = pb[keep], pid[keep]
    return pb, pid, gb, gid


def tracking_metrics(
    tracks: pd.DataFrame,
    gt_boxes: pd.DataFrame,
    frames: np.ndarray,
    cams: list[str],
    cls: str = "person",
    iou_thr: float = 0.5,
    min_h: float = 20.0,
    with_motmetrics: bool = True,
) -> dict:
    """CLEAR-MOT MOTA and IDF1 per camera sequence, micro-averaged over cameras."""
    t = tracks[tracks["cls"] == cls]
    g = gt_boxes[(gt_boxes["cls"] == cls) & gt_boxes["frame"].isin(frames) & gt_boxes["camera"].isin(cams)]
    gv, gi = _split_gt(g, min_h)
    tg, gvg, gig = _frame_groups(t), _frame_groups(gv), _frame_groups(gi)
    box = ["x1", "y1", "x2", "y2"]
    tot = dict(fn=0, fp=0, idsw=0, ngt=0, tp=0, iou_sum=0.0, idtp=0, idfp=0, idfn=0)
    per_cam = {}
    mm_accs = []
    mm = None
    if with_motmetrics:
        try:
            import motmetrics as mm  # type: ignore
        except Exception:
            mm = None
    for cam in cams:
        last_id: dict[int, int] = {}
        prev: dict[int, int] = {}
        c = dict(fn=0, fp=0, idsw=0, ngt=0, tp=0, iou_sum=0.0)
        co: dict[tuple[int, int], int] = {}
        gt_len: dict[int, int] = {}
        pr_len: dict[int, int] = {}
        acc = mm.MOTAccumulator(auto_id=False) if mm is not None else None
        for fr in frames:
            key = (cam, int(fr))
            pb, pid, gb, gid = _prep_tracking_frame(tg.get(key), gvg.get(key), gig.get(key), box, iou_thr)
            c["ngt"] += len(gb)
            for x in gid:
                gt_len[int(x)] = gt_len.get(int(x), 0) + 1
            for x in pid:
                pr_len[int(x)] = pr_len.get(int(x), 0) + 1
            iou = iou_matrix(gb, pb)
            if len(gb) and len(pb):
                ii, jj = np.nonzero(iou >= iou_thr)
                for i, j in zip(ii, jj):
                    k = (int(gid[i]), int(pid[j]))
                    co[k] = co.get(k, 0) + 1
            # CLEAR-MOT: keep previous correspondences first, then Hungarian on the rest
            pairs: list[tuple[int, int]] = []
            gi_of = {int(x): i for i, x in enumerate(gid)}
            pj_of = {int(x): j for j, x in enumerate(pid)}
            used_g, used_p = set(), set()
            for g_id, p_id in prev.items():
                if g_id in gi_of and p_id in pj_of and iou[gi_of[g_id], pj_of[p_id]] >= iou_thr:
                    pairs.append((gi_of[g_id], pj_of[p_id]))
                    used_g.add(gi_of[g_id])
                    used_p.add(pj_of[p_id])
            rg = [i for i in range(len(gb)) if i not in used_g]
            rp = [j for j in range(len(pb)) if j not in used_p]
            if rg and rp:
                for a, b in _hungarian(iou[np.ix_(rg, rp)], iou_thr):
                    pairs.append((rg[a], rp[b]))
            prev = {}
            for i, j in pairs:
                g_id, p_id = int(gid[i]), int(pid[j])
                if g_id in last_id and last_id[g_id] != p_id:
                    c["idsw"] += 1
                last_id[g_id] = p_id
                prev[g_id] = p_id
                c["iou_sum"] += float(iou[i, j])
            c["tp"] += len(pairs)
            c["fn"] += len(gb) - len(pairs)
            c["fp"] += len(pb) - len(pairs)
            if acc is not None:
                dist = 1.0 - iou
                dist[iou < iou_thr] = np.nan
                acc.update(gid.tolist(), pid.tolist(), dist, frameid=int(fr))
        # IDF1: global one-to-one assignment of GT identities to predicted identities
        g_ids = sorted(gt_len)
        p_ids = sorted(pr_len)
        idtp = 0
        if g_ids and p_ids:
            gi_ix = {x: i for i, x in enumerate(g_ids)}
            pi_ix = {x: i for i, x in enumerate(p_ids)}
            M = np.zeros((len(g_ids), len(p_ids)))
            for (a, b), n in co.items():
                M[gi_ix[a], pi_ix[b]] = n
            r, cc = linear_sum_assignment(-M)
            idtp = int(M[r, cc].sum())
        n_g, n_p = sum(gt_len.values()), sum(pr_len.values())
        idfn, idfp = n_g - idtp, n_p - idtp
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else float("nan")
        mota = 1 - (c["fn"] + c["fp"] + c["idsw"]) / c["ngt"] if c["ngt"] else float("nan")
        per_cam[cam] = {**c, "mota": mota, "idf1": idf1, "idtp": idtp, "idfp": idfp, "idfn": idfn}
        for k in ("fn", "fp", "idsw", "ngt", "tp", "iou_sum"):
            tot[k] += c[k]
        tot["idtp"] += idtp
        tot["idfp"] += idfp
        tot["idfn"] += idfn
        if acc is not None:
            mm_accs.append((cam, acc))
    mota = 1 - (tot["fn"] + tot["fp"] + tot["idsw"]) / tot["ngt"] if tot["ngt"] else float("nan")
    denom = 2 * tot["idtp"] + tot["idfp"] + tot["idfn"]
    out = {
        "min_h_px": min_h,
        "iou_thr": iou_thr,
        "mota": mota,
        "idf1": 2 * tot["idtp"] / denom if denom else float("nan"),
        "motp_iou": tot["iou_sum"] / tot["tp"] if tot["tp"] else float("nan"),
        "fn": tot["fn"], "fp": tot["fp"], "id_switches": tot["idsw"], "n_gt": tot["ngt"],
        "per_camera": per_cam,
    }
    if mm is not None and mm_accs:
        try:
            summ = mm.metrics.create().compute_many(
                [a for _, a in mm_accs], metrics=["mota", "idf1", "num_switches"], names=[c for c, _ in mm_accs],
                generate_overall=True,
            )
            row = summ.loc["OVERALL"]
            out["motmetrics_crosscheck"] = {
                "mota": float(row["mota"]), "idf1": float(row["idf1"]), "num_switches": int(row["num_switches"]),
            }
        except Exception as exc:  # pragma: no cover
            out["motmetrics_crosscheck"] = {"error": repr(exc)}
    return out


# --------------------------------------------------------------------------- occupancy


def gt_occupancy_steps(gt_objs: pd.DataFrame, zones, frames, cls: str, min_h: float) -> pd.DataFrame:
    d = gt_objs[(gt_objs["cls"] == cls) & gt_objs["frame"].isin(frames) & (gt_objs["max_h"] >= min_h)]
    zn = [z.name for z in zones]
    full = pd.MultiIndex.from_product([list(frames), zn], names=["frame", "zone"])
    if len(d) == 0:
        return pd.DataFrame({"gt": 0}, index=full).reset_index()
    zi = assign_zones(d[["wx", "wy"]].to_numpy(), zones)
    d = d.assign(zone=np.where(zi >= 0, np.array(zn + [""], dtype=object)[zi], ""))
    d = d[d["zone"] != ""]
    cnt = d.groupby(["frame", "zone"]).size().rename("gt")
    return cnt.reindex(full, fill_value=0).reset_index()


def occupancy_metrics(
    pred_steps: pd.DataFrame,
    gt_objs: pd.DataFrame,
    zones,
    frames: np.ndarray,
    src_fps: float,
    bin_seconds: float,
    cls: str = "person",
    min_h: float = 20.0,
) -> dict:
    gt = gt_occupancy_steps(gt_objs, zones, frames, cls, min_h)
    pr = pred_steps[pred_steps["cls"] == cls][["frame", "zone", "count"]].rename(columns={"count": "pred"})
    m = gt.merge(pr, on=["frame", "zone"], how="left").fillna({"pred": 0})
    err = (m["pred"] - m["gt"]).abs()
    tot = m.groupby("frame")[["pred", "gt"]].sum()
    # binned (time-average) occupancy
    gb = occupancy_bins(m.rename(columns={"gt": "count"}).assign(cls=cls)[["frame", "zone", "cls", "count"]], src_fps, bin_seconds)
    pb = occupancy_bins(m.rename(columns={"pred": "count"}).assign(cls=cls)[["frame", "zone", "cls", "count"]], src_fps, bin_seconds)
    bm = gb.merge(pb, on=["bin", "zone", "cls"], suffixes=("_gt", "_pred"))
    berr = (bm["mean_count_pred"] - bm["mean_count_gt"]).abs()
    per_zone = (
        m.assign(abs_err=err).groupby("zone").agg(gt_mean=("gt", "mean"), pred_mean=("pred", "mean"), mae=("abs_err", "mean"))
        .reset_index()
    )
    gt_mean = float(m["gt"].mean())
    return {
        "cls": cls,
        "min_h_px": min_h,
        "bin_seconds": bin_seconds,
        "step_mae": float(err.mean()),
        "step_mae_rel": float(err.mean() / gt_mean) if gt_mean else float("nan"),
        "bin_mae": float(berr.mean()) if len(berr) else float("nan"),
        "bin_mae_rel": float(berr.mean() / bm["mean_count_gt"].mean()) if len(bm) and bm["mean_count_gt"].mean() else float("nan"),
        "total_count_mae": float((tot["pred"] - tot["gt"]).abs().mean()),
        "total_count_mean_gt": float(tot["gt"].mean()),
        "total_count_mean_pred": float(tot["pred"].mean()),
        "total_count_corr": float(np.corrcoef(tot["pred"], tot["gt"])[0, 1]) if tot["gt"].std() > 0 and tot["pred"].std() > 0 else float("nan"),
        "n_steps": int(len(frames)),
        "per_zone": per_zone.to_dict("records"),
    }


# --------------------------------------------------------------------------- driver


def evaluate_run(run_dir: str | Path, gt_dir: str | Path, scene_dir: str | Path | None = None,
                 min_h: float = 20.0, iou_thr: float = 0.5, log=print) -> dict:
    from .gt import load_gt_tables

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    jobs = meta["jobs"]
    cams = sorted({c for j in jobs for c in j["cameras"]})
    frames = sampled_frames(jobs)
    src_fps = float(jobs[0]["src_fps"])
    dets = pd.read_parquet(run_dir / "detections.parquet")
    tracks = pd.read_parquet(run_dir / "tracks.parquet")
    gt_boxes, gt_objs = load_gt_tables(gt_dir)
    zones = load_zones(run_dir / "zones.json")
    steps = pd.read_parquet(run_dir / "occupancy_steps.parquet")
    bin_s = float(meta["summary"]["bin_seconds"])
    calib = load_calibration(Path(scene_dir) / "calibration.json") if scene_dir else None

    res: dict = {"cameras": cams, "n_frames_per_camera": int(len(frames)), "sampled_fps": jobs[0]["sampled_fps"]}
    for mh in sorted({min_h, 0.0}, reverse=True):
        dm = detection_metrics(dets, gt_boxes, frames, cams, "person", iou_thr, mh)
        matches = dm.pop("_matches")
        entry = {"detection": dm}
        if calib is not None:
            entry["localisation"] = localisation_error(matches, gt_objs, calib)
        entry["tracking"] = tracking_metrics(tracks, gt_boxes, frames, cams, "person", iou_thr, mh)
        entry["occupancy"] = occupancy_metrics(steps, gt_objs, zones, frames, src_fps, bin_s, "person", mh)
        res[f"min_h_{int(mh)}"] = entry
    res["headline_min_h_px"] = min_h
    for k, v in res.items():
        if k.startswith("min_h_"):
            d, tr, oc = v["detection"], v["tracking"], v["occupancy"]
            log(f"[{k}] det P={d['precision']:.3f} R={d['recall']:.3f} (tp={d['tp']} fp={d['fp']} fn={d['fn']}) | "
                f"MOTA={tr['mota']:.3f} IDF1={tr['idf1']:.3f} IDsw={tr['id_switches']} | "
                f"occ step-MAE={oc['step_mae']:.3f} bin-MAE={oc['bin_mae']:.3f} total-count-MAE={oc['total_count_mae']:.2f}")
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m intercycle.perception.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gt-dir", required=True, help="dir with gt_boxes.parquet + gt_objects.parquet")
    ap.add_argument("--scene-dir", default=None, help="for calibration.json (floor-localisation error)")
    ap.add_argument("--min-h", type=float, default=20.0)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args(argv)
    evaluate_run(a.run_dir, a.gt_dir, a.scene_dir, a.min_h, a.iou)
    return 0


if __name__ == "__main__":
    sys.exit(main())
