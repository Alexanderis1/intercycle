"""Probe P4 (PLAN_TWIN §3): press strokes on TRAIN clips only.

    python scripts/p4_press_probe.py motion --clips 3_tr1 3_tr2 ...      # vertical-motion maps to locate rams
    python scripts/p4_press_probe.py run --rois docs/probes/S3a-P4_rois.json --per-camera 10 --seed 20261017

`motion` writes, per clip, a grid of the vertical-flow standard deviation (stabilised), so ram candidates can be found
from numbers before the ROI is drawn on a keyframe. `run` picks a seeded sample of train clips per camera, runs
intercycle.perception.press_motion.ingest_clip for each press ROI of that camera's framing, writes
$INTERCYCLE_DATA/machines/<press>/<clip>.parquet and prints/saves a JSON summary (n strokes, in-clip intervals,
censored tails, lamp beat, stabilisation) plus the pass rule: a press is learned if >= 8 train clips show it running
with >= 2 strokes each.

ROI file: {"presses": [{"id": "P9a", "camera": 9, "rois": {"c9_f0": [[x, y], ...], "c9_f1": [...]}}, ...]}
(or the single-framing form "framing": "c9_f0", "roi_px": [...]). Each press samples --per-camera clips from the train
clips of its own framings (same seed per press) and uses the ROI of each clip's framing.
Only train clips are read (esk.load("train")); a clip id outside train raises.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's intercycle
from intercycle import esk  # noqa: E402
from intercycle.perception import press_motion as pm  # noqa: E402

LEARNED_MIN_CLIPS = 8
RUNNING_MIN_STROKES = 2


def _train_ids() -> set[str]:
    return set(esk.load("train")["clip_id"].astype(str))


def motion_map(clip_id: str, cell: int = 16) -> dict:
    import cv2
    c = pm.decode_gray(esk.clip_path(clip_id), width=640)
    stab, shift = pm.stabilise(c)
    n, h, w = stab.shape
    vs = []
    for i in range(1, n):
        dt = float(c.t[i] - c.t[i - 1])
        fl = cv2.calcOpticalFlowFarneback(stab[i - 1], stab[i], np.zeros((h, w, 2), np.float32),
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        vs.append(fl[..., 1] / c.scale / dt)
    v = np.stack(vs)
    gh, gw = h // cell, w // cell
    vv = v[:, :gh * cell, :gw * cell].reshape(len(v), gh, cell, gw, cell).mean(axis=(2, 4))
    sd = vv.std(axis=0)
    top = np.dstack(np.unravel_index(np.argsort(sd, axis=None)[::-1][:8], sd.shape))[0]
    return {"clip_id": clip_id, "cell_full_px": cell / c.scale, "stab_px": shift,
            "top_cells_full_px": [[float((x + 0.5) * cell / c.scale), float((y + 0.5) * cell / c.scale),
                                   float(sd[y, x])] for y, x in top],
            "grid_sd": np.round(sd, 1).tolist()}


def _press_rois(pr: dict) -> dict[str, list]:
    """{framing_id: roi polygon} of one press: the "rois" map, or the single "framing"/"roi_px" pair."""
    if "rois" in pr:
        return {str(k): v for k, v in pr["rois"].items()}
    return {str(pr["framing"]): pr["roi_px"]}


def run(rois: dict, per_camera: int, seed: int, out_json: Path, write_parquet: bool) -> dict:
    inv = esk.load("train")
    train_ids = set(inv["clip_id"].astype(str))
    framing_of = dict(zip(inv["clip_id"].astype(str), inv["framing_id"].astype(str)))
    summary: dict = {"seed": seed, "per_camera": per_camera, "presses": []}
    for pr in rois["presses"]:
        cam = int(pr["camera"])
        proi = _press_rois(pr)
        # the sample is drawn only from the framings where this press has an ROI; a fresh generator per press, so
        # presses with the same framings get the same clips
        rng = np.random.default_rng(seed)
        sel = (inv["camera"] == cam) & inv["framing_id"].astype(str).isin(list(proi))
        pool = sorted(inv.loc[sel, "clip_id"].astype(str))
        pool = [c for c in pool if esk.clip_path(c).exists()]
        pick = list(pr.get("clips") or rng.choice(pool, size=min(per_camera, len(pool)), replace=False))
        rows = []
        for cid in pick:
            if cid not in train_ids:
                raise PermissionError(f"{cid} is not a train clip")
            fr = framing_of[cid]
            if fr not in proi:
                raise ValueError(f"{pr['id']}: no ROI for framing {fr} of {cid}")
            r = pm.ingest_clip(esk.clip_path(cid), cid, "train", pr["id"], fr, proi[fr])
            iv, cens = pm.intervals(r.strokes_t, r.extra["duration"])
            rows.append({"clip_id": cid, "framing": fr, "duration": r.extra["duration"], "n_strokes": int(len(r.strokes_t)),
                         "strokes_t": np.round(r.strokes_t, 3).tolist(), "intervals": np.round(iv, 3).tolist(),
                         "censored_tail": cens, "threshold": r.threshold, "stab_px": r.stab_px,
                         "beat_hz": r.beat_hz, "beat_frac": r.beat_frac, "beat_amp": r.beat_amp,
                         "roi_beat_hz": r.extra["roi_beat_hz"], "roi_beat_frac": r.extra["roi_beat_frac"],
                         "flow_v_p99": float(np.nanpercentile(np.abs(r.table["flow_v"]), 99)),
                         "missing_share": float(r.table["missing"].mean())})
            if write_parquet:
                d = esk.data_root() / "machines" / pr["id"]
                d.mkdir(parents=True, exist_ok=True)
                r.table.to_parquet(d / f"{cid}.parquet", index=False)
            print(f"{pr['id']} {cid}: {len(r.strokes_t)} strokes, iv={rows[-1]['intervals']}, "
                  f"beat {r.beat_hz:.2f} Hz ({r.beat_frac:.2f}), stab {r.stab_px:.1f}px", flush=True)
        running = [x for x in rows if x["n_strokes"] >= RUNNING_MIN_STROKES]
        iv_all = np.concatenate([np.asarray(x["intervals"]) for x in rows]) if rows else np.zeros(0)
        summary["presses"].append({
            "id": pr["id"], "camera": cam, "rois": proi,
            "n_clips": len(rows), "n_running": len(running), "learned": len(running) >= LEARNED_MIN_CLIPS,
            "running_fraction": len(running) / len(rows) if rows else 0.0,
            "n_intervals": int(len(iv_all)),
            "period_median_s": float(np.median(iv_all)) if len(iv_all) else None,
            "period_p10_p90_s": [float(np.percentile(iv_all, 10)), float(np.percentile(iv_all, 90))]
            if len(iv_all) else None,
            "clips": rows})
    summary["n_learned"] = sum(p["learned"] for p in summary["presses"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("motion")
    m.add_argument("--clips", nargs="+", required=True)
    m.add_argument("--out", type=Path, default=Path("out/p4/motion.json"))
    r = sub.add_parser("run")
    r.add_argument("--rois", type=Path, required=True)
    r.add_argument("--per-camera", type=int, default=10)
    r.add_argument("--seed", type=int, default=20261017)
    r.add_argument("--out", type=Path, default=Path("out/p4/summary.json"))
    r.add_argument("--no-parquet", action="store_true")
    a = ap.parse_args()
    if a.cmd == "motion":
        train = _train_ids()
        bad = [c for c in a.clips if c not in train]
        if bad:
            raise PermissionError(f"not train clips: {bad}")
        res = [motion_map(c) for c in a.clips]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res), encoding="utf-8")
        for x in res:
            print(x["clip_id"], "stab", round(x["stab_px"], 1), "top", x["top_cells_full_px"][:4])
    else:
        s = run(json.loads(a.rois.read_text(encoding="utf-8")), a.per_camera, a.seed, a.out, not a.no_parquet)
        print(json.dumps({p["id"]: {k: p[k] for k in ("n_clips", "n_running", "learned", "period_median_s",
                                                       "period_p10_p90_s")} for p in s["presses"]}))


if __name__ == "__main__":
    main()
