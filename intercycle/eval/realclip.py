"""Real-clip scoring (lane S4 item 5; PLAN_TWIN §5 "Real-clip test"). Same code for the dev dry run and the sealed test.

Figures, each with its n, pooled and per stratum (camera x safe/unsafe; unsafe = clip class prefix 0-3):
  person P/R at IoU 0.5      greedy one-to-one by IoU on every scored label frame; predictions = PERC person boxes
                             (not P5-suppressed) of the PERC sample nearest the label PTS within FRAME_TOL_S (none
                             there = no detections, fail closed)
  foot-point pixel error     |label foot - predicted foot (box bottom centre)| px on matched pairs
  zone accuracy              matched pairs whose machine zone (danger rect, pixel_adapter.machine_of) is the same for
                             the label's and the prediction's foot point projected through CAL
  stroke F1                  +-0.25 s, greedy one-to-one by |dt|; predictions inside the window's hidden / edge spans
                             are not scored (labels/RULES.md r2)
  stroke-count error         per window n_pred - n_label (mean signed, mean absolute)
  occlusion -> MISSING rate  inside the labelled hidden spans: share of 2 Hz Observation ticks (pixel adapter) whose
                             zone_motion is MISSING for the window's press; PERC's per-sample flag is reported beside it,
                             and the base MISSING rate on the window's non-hidden ticks (the rate means little alone)
A framing with no fitted H (or no train framing) gets no H: its frames are excluded from foot-point error and zone
accuracy, n excluded reported. Recall CIs: bootstrap by group if >= 5 groups for the camera, else per-group recall with
n plus a frame-level Clopper-Pearson interval labelled "ignores clustering (only k groups)".
Every figure here is against independent Claude-agent labels (vision model), not human labels (PLAN U14).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

IOU_MIN = 0.5
STROKE_TOL_S = 0.25
OFFSET_DIAG_S = 1.5               # diagnostic only (event-definition offset between PERC strokes and labelled BDC)
FRAME_TOL_S = 1.0 / 12.0          # PERC samples at 12 fps: the nearest sample is at most half a step + one native frame
MIN_GROUPS_BOOT = 5
N_BOOT = 2000
BOOT_SEED = 20261018
CI_LEVEL = 0.95
UNSAFE_PREFIXES = (0, 1, 2, 3)
LABEL_CAVEAT = "against independent Claude-agent labels (vision model), not human labels"


# ------------------------------------------------------------------------------------------------ primitives

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(gt: Sequence[Sequence[float]], pred: Sequence[Sequence[float]], iou_min: float = IOU_MIN
                ) -> list[tuple[int, int, float]]:
    """Greedy one-to-one: pairs by IoU descending (ties: gt then pred index), each >= iou_min."""
    cand = [(iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)]
    cand = sorted((c for c in cand if c[0] >= iou_min), key=lambda c: (-c[0], c[1], c[2]))
    ug, up, out = set(), set(), []
    for v, i, j in cand:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        out.append((i, j, v))
    return out


def match_times(gt: Sequence[float], pred: Sequence[float], tol: float = STROKE_TOL_S) -> list[tuple[int, int]]:
    """Greedy one-to-one: pairs by |dt| ascending (ties: gt then pred index), each <= tol."""
    cand = sorted(((abs(g - p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred) if abs(g - p) <= tol + 1e-9),
                  key=lambda c: (c[0], c[1], c[2]))
    ug, up, out = set(), set(), []
    for _, i, j in cand:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        out.append((i, j))
    return out


def clopper_pearson(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return (lo, hi)


def _ratio(k: float, n: float) -> float | None:
    return None if n == 0 else float(k) / float(n)


def in_spans(t: float, spans: Iterable[Sequence[float]]) -> bool:
    return any(s[0] - 1e-9 <= t <= s[1] + 1e-9 for s in spans)


# ------------------------------------------------------------------------------------------------ per-item records

@dataclass
class FrameRec:
    sample_id: str
    clip_id: str
    camera: int
    group: str
    unsafe: bool
    framing: str
    has_H: bool
    n_gt: int
    n_pred: int
    tp: int
    foot_err_px: list[float] = field(default_factory=list)
    zone_same: list[bool] = field(default_factory=list)
    pred_frame_found: bool = True


@dataclass
class WindowRec:
    sample_id: str
    clip_id: str
    press: str
    camera: int
    group: str
    unsafe: bool
    n_gt: int
    n_pred: int
    tp: int
    hidden_ticks: int = 0
    hidden_missing_ticks: int = 0
    hidden_perc_samples: int = 0
    hidden_perc_missing: int = 0
    nonhidden_ticks: int = 0          # base: window ticks outside the hidden spans (the rate above means little alone)
    nonhidden_missing_ticks: int = 0
    machines_found: bool = True
    offsets_s: list[float] = field(default_factory=list)   # diagnostic: label - nearest prediction, |dt| <= OFFSET_DIAG_S


TRACK_BOX_PX = 1.0     # a track's foot point equals its detection box's bottom centre (esk_ingest foot_points)
NO_BOX = (math.nan, math.nan, math.nan, math.nan)


def tracked_boxes(tracks: pd.DataFrame | None, boxes: pd.DataFrame | None, t: float, tol: float = FRAME_TOL_S
                  ) -> tuple[list[tuple[float, float, float, float]], bool]:
    """Predicted person boxes = the PERC person TRACKS (the observer's input, tracker operating point) at the sample
    nearest t within tol; each track's box is the stored detection box at that sample whose bottom centre is its foot
    point. A track with no such box keeps a NaN box (it can never match: counted as a false positive, fail closed).
    Returns (boxes, a PERC sample was found within tol)."""
    if tracks is None or not len(tracks):
        return [], boxes is not None and len(boxes) > 0 and bool(np.min(np.abs(boxes["t"].to_numpy(float) - t)) <= tol)
    tt = tracks["t"].to_numpy(float)
    i = int(np.argmin(np.abs(tt - t)))
    if abs(tt[i] - t) > tol + 1e-9:
        found = boxes is not None and len(boxes) > 0 and bool(np.min(np.abs(boxes["t"].to_numpy(float) - t)) <= tol)
        return [], found
    tr = tracks.loc[np.isclose(tt, tt[i]) & (tracks["class"] == "person").to_numpy()]
    bx = boxes.loc[np.isclose(boxes["t"].to_numpy(float), tt[i])] if boxes is not None and len(boxes) else None
    out = []
    for u, v in tr[["u", "v"]].to_numpy(float):
        b = NO_BOX
        if bx is not None and len(bx):
            xy = bx[["x1", "y1", "x2", "y2"]].to_numpy(float)
            d = np.hypot((xy[:, 0] + xy[:, 2]) / 2.0 - u, xy[:, 3] - v)
            j = int(np.argmin(d))
            if d[j] <= TRACK_BOX_PX:
                b = (float(xy[j, 0]), float(xy[j, 1]), float(xy[j, 2]), float(xy[j, 3]))
        out.append(b)
    return out, True


DET_CONF_SECONDARY = 0.4   # tracker activation = esk_ingest occluder threshold; secondary detector-level figure


def detector_boxes(boxes: pd.DataFrame | None, t: float, conf: float = DET_CONF_SECONDARY, tol: float = FRAME_TOL_S
                   ) -> tuple[list[tuple[float, float, float, float]], bool]:
    """Secondary predictions: raw PERC person boxes (not P5-suppressed) with conf >= `conf` at the nearest sample."""
    if boxes is None or not len(boxes):
        return [], False
    tt = boxes["t"].to_numpy(float)
    i = int(np.argmin(np.abs(tt - t)))
    if abs(tt[i] - t) > tol + 1e-9:
        return [], False
    keep = (np.isclose(tt, tt[i]) & (boxes["class"] == "person").to_numpy() & ~boxes["suppressed"].to_numpy(bool)
            & (boxes["conf"].to_numpy(float) >= conf))
    xy = boxes.loc[keep, ["x1", "y1", "x2", "y2"]].to_numpy(float)
    return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in xy], True


def score_frame(label, preds: list[tuple[float, float, float, float]], found: bool, meta: Mapping,
                project: Callable | None, zone_of: Callable | None) -> FrameRec:
    """One label frame against its predicted person boxes (`tracked_boxes`). `project(uv) -> (n, 2)` floor points and
    `zone_of(xz) -> (n,)` machine index, both None when the framing has no H."""
    gt = [b for b in label.boxes if b.cls == "person"]
    G = [(b.x0, b.y0, b.x1, b.y1) for b in gt]
    P = list(preds)
    m = match_boxes(G, P)
    rec = FrameRec(label.sample_id, label.clip_id, int(meta["camera"]), str(meta["group"]), bool(meta["unsafe"]),
                   str(meta["framing"]), project is not None, len(G), len(P), len(m), pred_frame_found=found)
    if project is not None and m:
        lf = np.array([[gt[i].foot_u, gt[i].foot_v] for i, _, _ in m], float)
        pf = np.array([[(P[j][0] + P[j][2]) / 2.0, P[j][3]] for _, j, _ in m], float)
        rec.foot_err_px = [float(e) for e in np.hypot(*(lf - pf).T)]
        if zone_of is not None:
            zl, zp = zone_of(project(lf)), zone_of(project(pf))
            rec.zone_same = [bool(a == b) for a, b in zip(zl, zp)]
    return rec


def score_window(win, machines: pd.DataFrame | None, ignore: Sequence[Sequence[float]],
                 hidden: Sequence[Sequence[float]], meta: Mapping,
                 tick_missing: Sequence[tuple[float, bool]] | None) -> WindowRec:
    """One stroke window. `tick_missing` = [(t, zone_motion MISSING for this press)] from the pixel adapter."""
    pred: list[float] = []
    found = machines is not None
    hs = hm = 0
    if machines is not None and len(machines):
        t = machines["t"].to_numpy(float)
        st = machines["stroke"].to_numpy(bool)
        inw = (t >= win.t0 - 1e-9) & (t <= win.t1 + 1e-9)
        pred = [float(x) for x in t[inw & st] if not in_spans(float(x), ignore)]
        hid = np.array([in_spans(float(x), hidden) for x in t]) & inw if len(hidden) else np.zeros(len(t), bool)
        hs = int(hid.sum())
        hm = int((hid & machines["missing"].to_numpy(bool)).sum())
    gt = list(win.strokes)
    m = match_times(gt, pred)
    rec = WindowRec(win.sample_id, win.clip_id, win.press, int(meta["camera"]), str(meta["group"]),
                    bool(meta["unsafe"]), len(gt), len(pred), len(m), hidden_perc_samples=hs, hidden_perc_missing=hm,
                    machines_found=found)
    if pred:
        for g in gt:
            d = min((g - p for p in pred), key=abs)
            if abs(d) <= OFFSET_DIAG_S:
                rec.offsets_s.append(float(d))
    if tick_missing is not None:
        for t, miss in tick_missing:
            if not (win.t0 - 1e-9 <= t <= win.t1 + 1e-9):
                continue
            if len(hidden) and in_spans(t, hidden):
                rec.hidden_ticks += 1
                rec.hidden_missing_ticks += int(bool(miss))
            else:
                rec.nonhidden_ticks += 1
                rec.nonhidden_missing_ticks += int(bool(miss))
    return rec


# ------------------------------------------------------------------------------------------------ aggregation

def _boot_recall(frames: Sequence[FrameRec], rng: np.random.Generator) -> tuple[float, float]:
    groups = sorted({f.group for f in frames})
    by = {g: (sum(f.tp for f in frames if f.group == g), sum(f.n_gt for f in frames if f.group == g)) for g in groups}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(groups), len(groups), replace=True)
        k = sum(by[groups[i]][0] for i in pick)
        n = sum(by[groups[i]][1] for i in pick)
        if n:
            vals.append(k / n)
    a = (1 - CI_LEVEL) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))) if vals else (math.nan, math.nan)


def person_block(frames: Sequence[FrameRec], n_groups_camera: int | None = None,
                 rule_camera: int | None = None) -> dict:
    """`n_groups_camera` = the group count the small-group rule is applied on; `rule_camera` = the camera that count
    comes from when it is not this block's own (pooled: the camera with the fewest groups), named in the CI label."""
    tp = sum(f.tp for f in frames)
    ng = sum(f.n_gt for f in frames)
    npred = sum(f.n_pred for f in frames)
    groups = sorted({f.group for f in frames})
    k = len(groups) if n_groups_camera is None else n_groups_camera
    rec = {
        "n_frames": len(frames), "n_frames_no_perc_sample": sum(not f.pred_frame_found for f in frames),
        "n_gt_persons": ng, "n_pred_persons": npred, "tp": tp,
        "recall": _ratio(tp, ng), "precision": _ratio(tp, npred),
        "recall_cp95": clopper_pearson(tp, ng), "precision_cp95": clopper_pearson(tp, npred),
        "per_group_recall": {g: {"recall": _ratio(sum(f.tp for f in frames if f.group == g),
                                                  sum(f.n_gt for f in frames if f.group == g)),
                                 "n_gt": sum(f.n_gt for f in frames if f.group == g),
                                 "n_frames": sum(1 for f in frames if f.group == g)} for g in groups},
        "n_groups": len(groups),
    }
    if k >= MIN_GROUPS_BOOT:
        rec["recall_ci"] = {"method": f"bootstrap by group ({N_BOOT}, seed {BOOT_SEED})",
                            "ci95": _boot_recall(frames, np.random.default_rng(BOOT_SEED))}
    else:
        rec["recall_ci"] = {"method": "frame-level Clopper-Pearson",
                            "label": (f"ignores clustering (only {k} groups)" if rule_camera is None else
                                      f"ignores clustering (camera {rule_camera} has only {k} groups)"),
                            "ci95": clopper_pearson(tp, ng)}
    vals = [r["recall"] for r in rec["per_group_recall"].values() if r["recall"] is not None]
    rec["lowest_group_recall"] = min(vals) if vals else None
    withH = [f for f in frames if f.has_H]
    errs = [e for f in withH for e in f.foot_err_px]
    zs = [z for f in withH for z in f.zone_same]
    rec["foot_point_px"] = {"n_pairs": len(errs), "median": float(np.median(errs)) if errs else None,
                            "mean": float(np.mean(errs)) if errs else None,
                            "p90": float(np.quantile(errs, 0.9)) if errs else None,
                            "n_frames_excluded_no_H": len(frames) - len(withH)}
    rec["zone_accuracy"] = {"n_pairs": len(zs), "accuracy": _ratio(sum(zs), len(zs)),
                            "n_frames_excluded_no_H": len(frames) - len(withH)}
    return rec


def stroke_block(wins: Sequence[WindowRec]) -> dict:
    tp = sum(w.tp for w in wins)
    ng = sum(w.n_gt for w in wins)
    npred = sum(w.n_pred for w in wins)
    p, r = _ratio(tp, npred), _ratio(tp, ng)
    f1 = None if not (ng + npred) else 2.0 * tp / (ng + npred)
    d = [w.n_pred - w.n_gt for w in wins]
    ht = sum(w.hidden_ticks for w in wins)
    hm = sum(w.hidden_missing_ticks for w in wins)
    ps = sum(w.hidden_perc_samples for w in wins)
    pm = sum(w.hidden_perc_missing for w in wins)
    bt = sum(w.nonhidden_ticks for w in wins)
    bm = sum(w.nonhidden_missing_ticks for w in wins)
    return {
        "n_windows": len(wins), "n_windows_no_machines_file": sum(not w.machines_found for w in wins),
        "n_gt_strokes": ng, "n_pred_strokes": npred, "tp": tp, "precision": p, "recall": r, "f1": f1,
        "count_error": {"n_windows": len(wins), "mean_signed": float(np.mean(d)) if d else None,
                        "mean_abs": float(np.mean(np.abs(d))) if d else None},
        "occlusion_to_missing": {
            "n_windows_with_hidden": sum(1 for w in wins if w.hidden_ticks or w.hidden_perc_samples),
            "observation_ticks": {"n": ht, "missing": hm, "rate": _ratio(hm, ht)},
            "base_nonhidden_observation_ticks": {
                "n": bt, "missing": bm, "rate": _ratio(bm, bt),
                "note": "window ticks outside the hidden spans; the hidden-span rate is only meaningful against this"},
            "perc_samples": {"n": ps, "missing": pm, "rate": _ratio(pm, ps)}},
        "diagnostic_offset_label_minus_pred_s": {
            "n": len(off := [o for w in wins for o in w.offsets_s]),
            "median": float(np.median(off)) if off else None,
            "iqr": [float(np.quantile(off, 0.25)), float(np.quantile(off, 0.75))] if off else None,
            "note": f"label stroke minus nearest PERC stroke within {OFFSET_DIAG_S} s; diagnostic, not a score"},
    }


def strata(items: Sequence, key: Callable) -> dict[str, list]:
    out: dict[str, list] = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return dict(sorted(out.items()))


def stratum_key(it) -> str:
    return f"c{it.camera}_{'unsafe' if it.unsafe else 'safe'}"


def summarise(frames: Sequence[FrameRec], wins: Sequence[WindowRec]) -> dict:
    ncam = {c: len({f.group for f in frames if f.camera == c}) for c in sorted({f.camera for f in frames})}
    cmin = min(ncam, key=lambda c: (ncam[c], c)) if ncam else None
    out = {"label_caveat": LABEL_CAVEAT,
           "person": {"pooled": person_block(frames, ncam[cmin] if cmin is not None else 0, cmin),
                      "per_camera": {f"c{c}": person_block([f for f in frames if f.camera == c], ncam[c]) for c in ncam},
                      "per_stratum": {k: person_block(v, ncam[v[0].camera]) for k, v in strata(frames, stratum_key).items()}},
           "strokes": {"pooled": stroke_block(wins),
                       "per_press": {k: stroke_block(v) for k, v in strata(wins, lambda w: w.press).items()},
                       "per_stratum": {k: stroke_block(v) for k, v in strata(wins, stratum_key).items()}},
           "groups_per_camera": {f"c{c}": n for c, n in ncam.items()}}
    return out


# ------------------------------------------------------------------------------------------------ dev / sealed driver

def clip_meta(inv: pd.DataFrame, clip_id: str, allowed_splits: Sequence[str]) -> dict:
    r = inv[inv["clip_id"] == clip_id]
    if len(r) != 1:
        raise ValueError(f"clip {clip_id}: {len(r)} inventory rows")
    r = r.iloc[0]
    if str(r["split"]) not in allowed_splits:
        raise ValueError(f"clip {clip_id} is split {r['split']}, not in {list(allowed_splits)} (fail closed)")
    return {"camera": int(r["camera"]), "group": str(r["group_id"]), "framing": str(r["framing_id"]),
            "unsafe": int(str(r["class"])) in UNSAFE_PREFIXES, "split": str(r["split"])}


def _read(p: Path) -> pd.DataFrame | None:
    return pd.read_parquet(p) if p.exists() else None


def train_references(inv: pd.DataFrame, data: Path, framings: Iterable[str]) -> dict[str, object]:
    """{framing: M-W10 watchdog reference from its TRAIN keyframes (k1, k2)}. A framing with train clips but no
    loadable keyframe raises (fail closed: without a reference the watchdog drops its BLURRED / OBSTRUCTED checks);
    a framing with no train clip gets None (it has no H either, so it gets no Observation)."""
    from intercycle.observer import pixel_adapter as PA

    refs: dict[str, object] = {}
    for f in framings:
        ids = sorted(inv.loc[(inv["split"] == "train") & (inv["framing_id"] == f), "clip_id"].astype(str))
        refs[f] = PA.train_reference([data / "keyframes" / f"{c}_k{k}.jpg" for c in ids for k in (1, 2)])
        if ids and refs[f] is None:
            raise FileNotFoundError(f"framing {f}: {len(ids)} train clips but no loadable keyframe reference "
                                    "(fail closed)")
    return refs


def run(labels_dir: Path, data: Path, site_path: Path, allowed_splits: Sequence[str] = ("dev",),
        watchdog: bool = True) -> dict:
    """Score the scored DEV labels (twin.dev_labels.load_scored) against PERC output under `data`, running the pixel
    adapter (with the M-W10 watchdog on decoded frames) on every labelled clip."""
    from intercycle.observer import mapping as OM
    from intercycle.observer import pixel_adapter as PA
    from twin.dev_labels import load_scored
    from twin.schemas import load
    from twin.schemas.site import Site

    sd = load_scored(labels_dir)
    inv = pd.read_parquet(data / "inventory.parquet")
    site = load(Site, site_path)
    geoms = PA.site_geometry(site)
    train_framings = set(inv.loc[inv["split"] == "train", "framing_id"].astype(str))

    def geom_for(framing: str):
        g = geoms.get(framing)
        if g is None or g.H is None or framing not in train_framings:
            return None
        return g

    refs: dict[str, object] = train_references(inv, data, sorted(geoms)) if watchdog else {}

    # pixel adapter over every labelled clip
    clips = sorted({f.clip_id for f in sd.frames} | {w.clip_id for w in sd.windows})
    ticks: dict[str, list] = {}
    wd_stats: dict[str, dict[str, int]] = {}
    for c in clips:
        meta = clip_meta(inv, c, allowed_splits)
        g = geom_for(meta["framing"])
        if g is None:
            continue
        tr = _read(data / "tracks" / f"{c}.parquet")
        if tr is None:
            raise FileNotFoundError(f"no PERC tracks for {c}")
        mach = {p: d for p in g.press_ids if (d := _read(data / "machines" / p / f"{c}.parquet")) is not None}
        dur = float(inv.loc[inv["clip_id"] == c, "duration"].iloc[0])
        frames = PA.decode_ticks(data / "clips" / f"{c}.mp4", 0.0, dur) if watchdog else []
        ad = PA.PixelAdapter(g, refs.get(meta["framing"]))  # type: ignore[arg-type]
        ad.reset()
        seq = []
        for t, fr in frames:
            o = ad.step(t, PA._nearest_rows(tr, t, PA.TICK_S / 2), fr, PA.motion_at(mach, t))
            OM.check_obs_v2(o, len(g.press_ids), 1)
            h = ad.last_health
            st = wd_stats.setdefault(meta["framing"], {})
            st[h.reason if h else "none"] = st.get(h.reason if h else "none", 0) + 1
            seq.append(o)
        ticks[c] = [(o.t, o, g.press_ids) for o in seq]

    frame_recs, det_recs = [], []
    for lab in sd.frames:
        meta = clip_meta(inv, lab.clip_id, allowed_splits)
        g = geom_for(meta["framing"])
        proj = (lambda uv, g=g: PA.project_uv(uv, g)) if g is not None else None
        zf = (lambda xz, g=g: PA.machine_of(xz, g.rects)) if g is not None else None
        bx = _read(data / "boxes" / f"{lab.clip_id}.parquet")
        if bx is None:
            raise FileNotFoundError(f"no PERC boxes for {lab.clip_id}")
        preds, found = tracked_boxes(_read(data / "tracks" / f"{lab.clip_id}.parquet"), bx, lab.t_pts)
        frame_recs.append(score_frame(lab, preds, found, meta, proj, zf))
        dp, dfound = detector_boxes(bx, lab.t_pts)
        det_recs.append(score_frame(lab, dp, dfound, meta, proj, zf))

    win_recs = []
    for w in sd.windows:
        meta = clip_meta(inv, w.clip_id, allowed_splits)
        tm = None
        if w.clip_id in ticks:
            tm = []
            for t, o, pids in ticks[w.clip_id]:
                j = pids.index(w.press) if w.press in pids else -1
                tm.append((t, True if j < 0 else bool(np.isnan(o.zone_motion[j]))))
        win_recs.append(score_window(w, _read(data / "machines" / w.press / f"{w.clip_id}.parquet"),
                                     sd.ignore.get(w.sample_id, []), sd.hidden.get(w.sample_id, []), meta, tm))

    out = summarise(frame_recs, win_recs)
    det = summarise(det_recs, [])["person"]
    out["person_detector_secondary"] = {"note": f"raw PERC person boxes conf >= {DET_CONF_SECONDARY}, not tracked; "
                                                "secondary (the primary is the tracked output the observer receives)",
                                        **det}
    n_ticks = sum(len(v) for v in ticks.values())
    ok_ticks = sum(1 for v in ticks.values() for _, o, _ in v if bool(o.cam_floor_ok[0]))
    out["pixel_adapter"] = {"n_clips": len(ticks), "n_ticks": n_ticks, "cam_floor_ok_ticks": ok_ticks,
                            "cam_floor_ok_share": _ratio(ok_ticks, n_ticks),
                            "watchdog_reasons_by_framing": wd_stats,
                            "watchdog_reference": {f: (r is not None) for f, r in refs.items()},
                            "n_det_ticks": sum(1 for v in ticks.values() for _, o, _ in v if len(o)),
                            "n_person_dets_on_floor": sum(int((o.cls == OM.PERSON).sum())
                                                          for v in ticks.values() for _, o, _ in v),
                            "n_person_dets_in_machine_zone": sum(int(((o.cls == OM.PERSON) & (o.machine >= 0)).sum())
                                                                 for v in ticks.values() for _, o, _ in v)}
    out["n_label_frames"] = len(frame_recs)
    out["n_stroke_windows"] = len(win_recs)
    out["clips"] = clips
    out["frames"] = [vars(f) for f in frame_recs]
    out["windows"] = [vars(w) for w in win_recs]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--labels", default="labels/dev")
    ap.add_argument("--data", default=os.environ.get("INTERCYCLE_DATA", ""))
    ap.add_argument("--site", default="twin/site.yaml")
    ap.add_argument("--out", default="results/dev/realclip_dev.json")
    ap.add_argument("--no-watchdog", action="store_true")
    a = ap.parse_args(argv)
    if not a.data:
        print("INTERCYCLE_DATA not set", file=sys.stderr)
        return 2
    res = run(Path(a.labels), Path(a.data), Path(a.site), ("dev",), watchdog=not a.no_watchdog)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(compact(res))
    return 0


def compact(res: dict) -> str:
    """One line per block: the headline figures with their n."""
    lines = [res["label_caveat"]]

    def pb(name, b):
        fp, za, ci = b["foot_point_px"], b["zone_accuracy"], b["recall_ci"]
        r = "na" if b["recall"] is None else f"{b['recall']:.3f}"
        p = "na" if b["precision"] is None else f"{b['precision']:.3f}"
        lines.append(f"person {name}: frames {b['n_frames']} gt {b['n_gt_persons']} pred {b['n_pred_persons']} "
                     f"tp {b['tp']} R {r} P {p} R-CI {ci['method']}{' (' + ci['label'] + ')' if 'label' in ci else ''} "
                     f"{[round(x, 3) for x in ci['ci95']]} lowest-group R {b['lowest_group_recall']} "
                     f"groups {b['n_groups']} | foot px median {fp['median']} p90 {fp['p90']} n {fp['n_pairs']} "
                     f"(excl no-H {fp['n_frames_excluded_no_H']}) | zone acc {za['accuracy']} n {za['n_pairs']}")

    def sb(name, b):
        o = b["occlusion_to_missing"]
        lines.append(f"  offset diag {b['diagnostic_offset_label_minus_pred_s']}")
        lines.append(f"strokes {name}: windows {b['n_windows']} gt {b['n_gt_strokes']} pred {b['n_pred_strokes']} "
                     f"tp {b['tp']} P {b['precision']} R {b['recall']} F1 {b['f1']} | count err signed "
                     f"{b['count_error']['mean_signed']} abs {b['count_error']['mean_abs']} | occ->MISSING obs "
                     f"{o['observation_ticks']} base non-hidden "
                     f"{ {k: v for k, v in o['base_nonhidden_observation_ticks'].items() if k != 'note'} } "
                     f"perc {o['perc_samples']} (windows w/ hidden {o['n_windows_with_hidden']})")

    pb("pooled", res["person"]["pooled"])
    pb("DETECTOR-secondary pooled", res["person_detector_secondary"]["pooled"])
    for k, v in res["person_detector_secondary"]["per_camera"].items():
        pb("DETECTOR-secondary " + k, v)
    for k, v in res["person"]["per_camera"].items():
        pb(k, v)
    for k, v in res["person"]["per_stratum"].items():
        pb(k, v)
    sb("pooled", res["strokes"]["pooled"])
    for k, v in res["strokes"]["per_press"].items():
        sb(k, v)
    for k, v in res["strokes"]["per_stratum"].items():
        sb(k, v)
    pa = {k: v for k, v in res["pixel_adapter"].items()}
    lines.append(f"pixel adapter: {json.dumps(pa, default=float)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
