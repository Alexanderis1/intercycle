"""Fit the twin camera stat_v1 noise model (lane S3b; PLAN_TWIN §2 twin/camera.py row; WEAKNESSES M-W2, T-W6).

    python scripts/stat_v1_fit.py            # -> twin/stat_v1_params.json (aggregates only, no image data)

Inputs (no TEST split is read: `esk.load` is called for "dev" and "train" only):
  * DEV labels (`labels/dev`, `twin.dev_labels.load_scored`, Claude-agent labels) vs the PERC DEV tracks
    ($INTERCYCLE_DATA/tracks/<clip>.parquet; u, v = foot point = bottom centre of the tracked box, full-res px):
      - recall per camera and class, on clean boxes (not occluded, not truncated) and on occluded boxes;
      - foot-point pixel error (track - label) of matched clean person boxes: robust sigma per axis (1.4826 MAD);
      - class confusion (class-agnostic matching, person <-> forklift), Jeffreys posterior mean;
    matching: per frame, the track rows within HALF_FRAME_S of the label PTS; greedy by foot distance, gate
    GATE_FRAC x the label box height.
  * DEV tracks alone:
      - miss bursts: within-track gaps (the tracker bridges up to lost_track_s = 3 s) plus re-birth gaps (a track ends
        and a new same-class track starts within REBIRTH_S and REBIRTH_PX, away from the image border and the clip
        ends); the 2 Hz Gilbert chain keeps the mean burst length in ticks given that a burst hits a tick;
      - ID switches: re-births per track-second;
      - temporal correlation of the foot-point noise (AR(1) rho at 2 Hz) from the second differences of the 2 Hz
        person tracks, with the label sigma: Var(d2) = 2 sigma^2 (1 - rho)(3 - rho).
  * TRAIN machines tables ($INTERCYCLE_DATA/machines/<press>/<clip>.parquet): ROI-energy quantiles per press for
    "moving" (running clip, |flow_v| >= the P4 trigger threshold), "quiet" (running clip, below it) and "idle"
    (clip with no stroke); running clip = >= 2 strokes (the P4 rule). Energies are divided by the press's median
    idle energy (ROIs differ in size, texture and light). The camera-14 presses never run in the footage (ASSUMED):
    their moving / quiet pools are the pooled normalised pools of the LEARNED presses. The motion threshold (Youden J
    of normalised moving vs idle) is the value for observer_live's `motion_thr`.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle.perception.press_motion import detect_strokes  # noqa: E402
from twin.dev_labels import load_scored  # noqa: E402

OUT = ROOT / "twin" / "stat_v1_params.json"
IMG_W, IMG_H = 1920, 1080
HALF_FRAME_S = 0.5 / 12.0         # PERC samples at ~12 fps
GATE_FRAC = 0.5                   # match gate: foot distance <= GATE_FRAC x label box height
REBIRTH_S, REBIRTH_PX = 6.0, 150.0
BORDER_PX = 40.0
CLIP_EDGE_S = 0.5
HZ = 2.0
N_Q = 101
LABEL_CLS = {"person": "person", "forklift": "forklift"}
LEARNED = ("P9a", "P9b")          # P4: LEARNED on camera 9; the camera-14 presses are idle in all footage (ASSUMED)
UNIT = "roi_energy / the press's median idle train roi_energy (dimensionless)"


def _mad_sigma(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else float("nan")


def _jeffreys(k: int, n: int) -> float:
    return (k + 0.5) / (n + 1.0)


def load_tracks(clip: str) -> pd.DataFrame:
    return pd.read_parquet(esk.data_root() / "tracks" / f"{clip}.parquet")


# ------------------------------------------------------------------------------------------- labels vs tracks

def match_frame(boxes, rows: pd.DataFrame) -> list[tuple[int, int | None]]:
    """Greedy one-to-one match of label boxes to track rows by foot distance (gate GATE_FRAC x box height)."""
    feet = []
    for b in boxes:
        fu = b.foot_u if b.foot_u is not None else (b.x0 + b.x1) / 2
        fv = b.foot_v if b.foot_v is not None else b.y1
        feet.append((fu, fv, max(b.y1 - b.y0, 1.0)))
    uv = rows[["u", "v"]].to_numpy(float)
    pairs = []
    for i, (fu, fv, hgt) in enumerate(feet):
        for j in range(len(uv)):
            d = math.hypot(uv[j, 0] - fu, uv[j, 1] - fv)
            if d <= GATE_FRAC * hgt:
                pairs.append((d, i, j))
    pairs.sort()
    used_i, used_j, out = set(), set(), {}
    for d, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out[i] = j
    return [(i, out.get(i)) for i in range(len(boxes))]


def fit_labels(inv: pd.DataFrame) -> dict:
    sd = load_scored(ROOT / "labels" / "dev")
    cam_of = dict(zip(inv["clip_id"], inv["camera"].astype(int)))
    rec = defaultdict(lambda: [0, 0])            # (cam, cls, kind) -> [hit, n]
    conf = Counter()                              # (label cls, track cls)
    du, dv, du_rel, dv_rel, by_cam = [], [], [], [], defaultdict(list)
    n_frames = 0
    for f in sd.frames:
        if f.clip_id not in cam_of:
            raise SystemExit(f"label clip {f.clip_id} is not a dev clip")
        cam = cam_of[f.clip_id]
        tr = load_tracks(f.clip_id)
        rows = tr.loc[(tr["t"] - f.t_pts).abs() <= HALF_FRAME_S]
        if len(rows):                             # one PERC frame: the nearest sampled time
            tn = rows.loc[(rows["t"] - f.t_pts).abs().idxmin(), "t"]
            rows = rows.loc[rows["t"] == tn]
        n_frames += 1
        for i, j in match_frame(f.boxes, rows.reset_index(drop=True)):
            b = f.boxes[i]
            cls = LABEL_CLS[b.cls]
            kind = "clean" if not (b.occluded or b.truncated) else ("occluded" if b.occluded else "truncated")
            r = rec[(cam, cls, kind)]
            r[1] += 1
            if j is None:
                continue
            r[0] += 1
            tcls = str(rows.iloc[j]["class"])
            conf[(cls, tcls)] += 1
            if cls == "person" and kind == "clean" and b.foot_u is not None and b.foot_v is not None:
                hgt = max(b.y1 - b.y0, 1.0)
                eu, ev = float(rows.iloc[j]["u"]) - b.foot_u, float(rows.iloc[j]["v"]) - b.foot_v
                du.append(eu)
                dv.append(ev)
                du_rel.append(eu / hgt)
                dv_rel.append(ev / hgt)
                by_cam[cam].append((eu, ev))
    recall = {}
    for (cam, cls, kind), (k, n) in sorted(rec.items()):
        recall[f"c{cam}_{cls}_{kind}"] = {"hit": k, "n": n, "recall": round(k / n, 4) if n else None}

    def pooled(cls: str, kinds: tuple[str, ...]) -> dict:
        k = sum(rec[(c, cls, kd)][0] for c in (9, 14) for kd in kinds)
        n = sum(rec[(c, cls, kd)][1] for c in (9, 14) for kd in kinds)
        return {"hit": k, "n": n, "recall": round(k / n, 4) if n else None}

    n_p = sum(v for (a, _b), v in conf.items() if a == "person")
    n_f = sum(v for (a, _b), v in conf.items() if a == "forklift")
    du_a, dv_a = np.asarray(du), np.asarray(dv)
    s_u, s_v = _mad_sigma(du_a), _mad_sigma(dv_a)
    r = np.hypot(du_a, dv_a)
    return {
        "n_frames": n_frames,
        "recall": recall,
        "recall_pooled": {"person_clean": pooled("person", ("clean",)),
                          "person_occluded": pooled("person", ("occluded",)),
                          "person_truncated": pooled("person", ("truncated",)),
                          "forklift_clean": pooled("forklift", ("clean",)),
                          "forklift_all": pooled("forklift", ("clean", "occluded", "truncated"))},
        "confusion_counts": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
        "confusion": {"person_as_vehicle": round(_jeffreys(conf[("person", "forklift")], n_p), 4),
                      "vehicle_as_person": round(_jeffreys(conf[("forklift", "person")], n_f), 4),
                      "n_person": n_p, "n_forklift": n_f, "estimator": "Jeffreys posterior mean (k + 0.5) / (n + 1)"},
        "foot_px": {"n": int(len(du_a)), "sigma_u_px": round(s_u, 2), "sigma_v_px": round(s_v, 2),
                    "bias_u_px": round(float(np.median(du_a)), 2) if len(du_a) else None,
                    "bias_v_px": round(float(np.median(dv_a)), 2) if len(dv_a) else None,
                    "sigma_u_rel_height": round(_mad_sigma(du_rel), 4), "sigma_v_rel_height": round(_mad_sigma(dv_rel), 4),
                    "radial_p50_p90_px": [round(float(np.percentile(r, 50)), 2), round(float(np.percentile(r, 90)), 2)]
                    if len(r) else None,
                    "share_beyond_5_sigma": round(float(np.mean((np.abs(du_a) > 5 * s_u) | (np.abs(dv_a) > 5 * s_v))), 4)
                    if len(du_a) else None,
                    "per_camera_n_sigma_u_v": {str(c): [len(v), round(_mad_sigma([e[0] for e in v]), 2),
                                                        round(_mad_sigma([e[1] for e in v]), 2)] for c, v in by_cam.items()}},
    }


# ------------------------------------------------------------------------------------------- tracks alone

def fit_tracks(inv: pd.DataFrame, sigma_px: float) -> dict:
    gaps = defaultdict(list)                     # cam -> gap durations (s)
    rebirth = Counter()
    track_s = Counter()
    d2 = defaultdict(list)
    for _, row in inv.iterrows():
        clip, cam = row["clip_id"], int(row["camera"])
        tr = load_tracks(clip)
        if tr.empty:
            continue
        t0c, t1c = float(tr["t"].min()), float(tr["t"].max())
        ends, starts = [], []
        for (tid, cls), g in tr.groupby(["track_id", "class"]):
            g = g.sort_values("t")
            t = g["t"].to_numpy(float)
            uv = g[["u", "v"]].to_numpy(float)
            track_s[(cam, cls)] += float(t[-1] - t[0])
            dt = np.diff(t)
            gaps[cam] += [float(x) for x in dt[dt > 1.5 / 12.0] - 1.0 / 12.0]
            ends.append((t[-1], uv[-1], cls, tid))
            starts.append((t[0], uv[0], cls, tid))
            if cls == "person" and len(t) > 12:
                grid = np.arange(t[0], t[-1], 1.0 / HZ)
                iu = np.interp(grid, t, uv[:, 0])
                iv = np.interp(grid, t, uv[:, 1])
                # only grid points with a real sample within half a frame (no interpolation across gaps)
                near = np.min(np.abs(t[None, :] - grid[:, None]), axis=1) <= HALF_FRAME_S
                p = np.column_stack([iu, iv])
                for k in range(1, len(grid) - 1):
                    if near[k - 1] and near[k] and near[k + 1]:
                        d2[cam].append(p[k + 1] - 2 * p[k] + p[k - 1])

        def interior(t, uv):
            return (CLIP_EDGE_S < t - t0c and t1c - t > CLIP_EDGE_S and BORDER_PX < uv[0] < IMG_W - BORDER_PX
                    and uv[1] < IMG_H - BORDER_PX)

        used = set()
        for te, ue, cls, tid in sorted(ends, key=lambda e: e[0]):
            if not interior(te, ue):
                continue
            best = None
            for ts, us, cls2, tid2 in starts:
                if tid2 == tid or cls2 != cls or tid2 in used or not 0 < ts - te <= REBIRTH_S:
                    continue
                d = float(np.hypot(*(us - ue)))
                if d <= REBIRTH_PX and interior(ts, us) and (best is None or ts - te < best[0]):
                    best = (ts - te, tid2)
            if best is not None:
                used.add(best[1])
                rebirth[(cam, cls)] += 1
                gaps[cam].append(best[0])
    out = {"gaps": {}, "id_switch": {}, "ar1": {}}
    for cam, g in sorted(gaps.items()):
        D = np.asarray(g, float)
        hit = np.minimum(1.0, HZ * D)
        mean_b = float((HZ * D).sum() / hit.sum()) if len(D) else 1.0
        mean_b = max(mean_b, 1.0)
        out["gaps"][str(cam)] = {"n": int(len(D)), "p50_p90_s": [round(float(np.percentile(D, 50)), 3),
                                                                  round(float(np.percentile(D, 90)), 3)],
                                 "mean_burst_ticks_2hz": round(mean_b, 3), "p_stay_miss": round(1.0 - 1.0 / mean_b, 4)}
    for (cam, cls), n in sorted(track_s.items()):
        out["id_switch"][f"c{cam}_{cls}"] = {"rebirths": int(rebirth[(cam, cls)]), "track_s": round(n, 1),
                                             "per_s": round(rebirth[(cam, cls)] / n, 5) if n else None}
    for cam, v in sorted(d2.items()):
        a = np.asarray(v, float)
        var = float(np.mean([_mad_sigma(a[:, 0]) ** 2, _mad_sigma(a[:, 1]) ** 2]))
        # 2 s^2 (1 - rho)(3 - rho) = var  ->  rho^2 - 4 rho + 3 - var / (2 s^2) = 0, root in [0, 1]
        c = 3.0 - var / (2.0 * sigma_px ** 2)
        disc = 16.0 - 4.0 * c
        rho = (4.0 - math.sqrt(disc)) / 2.0 if disc >= 0 else 0.0
        out["ar1"][str(cam)] = {"n": int(len(a)), "sigma_d2_px": round(math.sqrt(var), 3),
                                "rho_raw": round(rho, 4), "rho": round(min(max(rho, 0.0), 0.95), 4)}
    return out


# ------------------------------------------------------------------------------------------- ROI energy (train)

def fit_energy() -> dict:
    tr = esk.load("train")
    ids = set(tr["clip_id"])
    root = esk.data_root() / "machines"
    pools: dict[str, dict[str, list[np.ndarray]]] = {}
    clips: dict[str, Counter] = {}
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        press = pdir.name
        pools[press] = {"moving": [], "quiet": [], "idle": []}
        clips[press] = Counter()
        for f in sorted(pdir.glob("*.parquet")):
            if f.stem not in ids:
                continue                           # train only (dev / test never enter the energy pools)
            m = pd.read_parquet(f)
            if set(m["split"]) != {"train"}:
                raise SystemExit(f"{f}: not a train table")
            ok = ~m["missing"].to_numpy(bool) & np.isfinite(m["roi_energy"].to_numpy(float))
            e = m["roi_energy"].to_numpy(float)
            n_st = int(m["stroke"].sum())
            if n_st >= 2:
                _, thr = detect_strokes(m["t"].to_numpy(float), np.where(ok, m["flow_v"].to_numpy(float), np.nan))
                v = np.abs(m["flow_v"].to_numpy(float))
                mv = ok & np.isfinite(v) & (v >= thr)
                pools[press]["moving"].append(e[mv])
                pools[press]["quiet"].append(e[ok & ~mv])
                clips[press]["running"] += 1
            elif n_st == 0:
                pools[press]["idle"].append(e[ok])
                clips[press]["idle"] += 1
            else:
                clips[press]["one_stroke_skipped"] += 1
    qs = np.linspace(0.0, 1.0, N_Q)

    def qrec(a: np.ndarray) -> dict:
        return {"n": int(len(a)), "q": [round(float(x), 4) for x in np.quantile(a, qs)] if len(a) else None}

    out: dict = {"quantile_levels": N_Q, "unit": UNIT, "learned_presses": list(LEARNED), "presses": {}}
    cat = {p: {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in pl.items()} for p, pl in pools.items()}
    base = {}
    for press, pl in cat.items():
        if not len(pl["idle"]):
            raise SystemExit(f"{press}: no idle train frames to normalise by")
        base[press] = float(np.median(pl["idle"]))
    learned_norm = {k: np.concatenate([cat[p][k] / base[p] for p in LEARNED]) for k in ("moving", "quiet")}
    for press, pl in cat.items():
        own = press in LEARNED
        rec = {"clips": dict(clips[press]), "idle_median_raw": round(base[press], 4),
               "raw": {k: qrec(v) for k, v in pl.items()},
               "idle": qrec(pl["idle"] / base[press]),
               "running_source": "fitted (own train clips)" if own else
               "assumed: pooled normalised pools of the P4-LEARNED presses (" + ", ".join(LEARNED) + ")"}
        for k in ("moving", "quiet"):
            rec[k] = qrec(pl[k] / base[press]) if own else qrec(learned_norm[k])
        out["presses"][press] = rec
    mv = learned_norm["moving"]
    idle = np.concatenate([cat[p]["idle"] / base[p] for p in cat])
    cand = np.unique(np.quantile(np.concatenate([mv, idle]), np.linspace(0.01, 0.99, 197)))
    j = [(float(np.mean(mv > c)) - float(np.mean(idle > c)), c) for c in cand]
    best_j, thr = max(j)
    out["motion_thr"] = {"value": round(float(thr), 4), "tpr_moving": round(float(np.mean(mv > thr)), 4),
                         "fpr_idle": round(float(np.mean(idle > thr)), 4), "youden_j": round(best_j, 4),
                         "n_moving": int(len(mv)), "n_idle": int(len(idle)),
                         "rule": "argmax over thresholds of TPR(moving) - FPR(idle) on the normalised energy; moving = "
                                 "the P4-LEARNED presses' train moving frames, idle = every press's idle train frames"}
    return out


def main() -> int:
    dev = esk.load("dev")
    assert set(dev["split"]) == {"dev"}
    lab = fit_labels(dev)
    sigma_px = float(np.hypot(lab["foot_px"]["sigma_u_px"], lab["foot_px"]["sigma_v_px"]) / math.sqrt(2))
    trk = fit_tracks(dev, sigma_px)
    en = fit_energy()
    out = {"_doc": ("twin/camera.py stat_v1 parameters fitted by scripts/stat_v1_fit.py: detection, foot-point, burst, "
                    "ID-switch and class-confusion statistics from DEV labels (Claude-agent labels, not human) vs DEV "
                    "PERC tracks; ROI-energy quantiles and the motion threshold from TRAIN machines tables. Aggregates "
                    "only."),
           "contract": "twin.stat_v1_params/1", "labels": lab, "tracks": trk, "energy": en,
           "inputs": {"label_dir": "labels/dev", "dev_clips": int(len(dev)), "tracks": "$INTERCYCLE_DATA/tracks",
                      "machines": "$INTERCYCLE_DATA/machines (train only)"}}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out["labels"][k] for k in ("recall_pooled", "confusion", "foot_px")}, indent=1))
    print(json.dumps(trk, indent=1))
    print(json.dumps(en["motion_thr"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
