"""Summarise and check the PERC batch outputs (lane S3a item 2) under $INTERCYCLE_DATA, train + dev only.

    python scripts/perc_summary.py [--sha-out docs/perc_sha256.txt] [--json-out $INTERCYCLE_DATA/perc/summary.json]

* checks every tracks/<clip>.parquet against twin.schemas.tables.TrackRow and every machines/<press>/<clip>.parquet
  against MachineRow (test rows refused), and that every train/dev clip has its tracks file and one machines file per
  press ROI of its framing;
* aggregates per split x camera: sampled fps, person boxes per frame (conf >= 0.5, on all 12 fps frames and on the
  frame nearest t = 0.5 s, the staging definition), track counts, track-gap distribution (stat_v1 input);
* writes the sha256 list of every output file (relative path, LF line endings) for the repo.

Reads only esk.load(["train", "dev"]); a file whose clip id is not train/dev raises.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intercycle import esk  # noqa: E402
from intercycle.perception.esk_ingest import load_press_rois, track_gaps  # noqa: E402
from twin.schemas.tables import MachineRow, TrackRow, validate_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# staging aggregates (docs/probes/G0_PROBES.md, P3 under A1+): RF-DETR-M on the t = 0.5 s frame, person conf >= 0.5
STAGING_PPF = {("train", 9): 2.38, ("train", 14): 1.44, ("dev", 9): 2.32, ("dev", 14): 1.21}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _q(a, qs=(0.5, 0.9, 0.99)) -> list[float]:
    a = np.asarray(a, float)
    return [round(float(np.quantile(a, q)), 3) for q in qs] if len(a) else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha-out", type=Path, default=ROOT / "docs" / "perc_sha256.txt")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--stats", default="stats.jsonl")
    ap.add_argument("--rois", type=Path, default=ROOT / "docs" / "probes" / "S3a-P4_rois.json")
    a = ap.parse_args(argv)
    root = esk.data_root()
    inv = esk.load(["train", "dev"])
    meta = {str(r.clip_id): (str(r.split), int(r.camera), str(r.framing_id))  # type: ignore[attr-defined]
            for r in inv.itertuples()}
    rois = load_press_rois(a.rois)

    # files: every one must belong to a train/dev clip
    files = sorted([*(root / "tracks").glob("*.parquet"), *(root / "machines").glob("*/*.parquet"),
                    *(root / "boxes").glob("*.parquet")])
    for p in files:
        if p.stem not in meta:
            raise PermissionError(f"{p.relative_to(root)}: not a train/dev clip")
    missing = []
    for cid, (_, _, fr) in meta.items():
        if not (root / "tracks" / f"{cid}.parquet").exists():
            missing.append(f"tracks/{cid}")
        for press in rois.get(fr, {}):
            if not (root / "machines" / press / f"{cid}.parquet").exists():
                missing.append(f"machines/{press}/{cid}")
    n_rows = {"tracks": 0, "machines": 0}
    tracks_by_clip: dict[str, pd.DataFrame] = {}
    mach: list[dict] = []
    for p in files:
        kind = p.relative_to(root).parts[0]
        if kind == "tracks":
            n_rows["tracks"] += validate_table(p, TrackRow)
            tracks_by_clip[p.stem] = pd.read_parquet(p)
        elif kind == "machines":
            n_rows["machines"] += validate_table(p, MachineRow)
            m = pd.read_parquet(p)
            mach.append({"press": p.parent.name, "clip_id": p.stem, "split": meta[p.stem][0], "n": len(m),
                         "missing": int(m["missing"].to_numpy(bool).sum()),
                         "strokes": int(m["stroke"].to_numpy(bool).sum())})
    last: dict[str, dict] = {}
    sp = root / "perc" / a.stats
    if sp.exists():
        for line in sp.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            last[d["clip_id"]] = d
    groups: dict[tuple[str, int], list[dict]] = {}
    for cid, d in last.items():
        if cid in meta:
            groups.setdefault((meta[cid][0], meta[cid][1]), []).append(d)

    out: dict = {"n_clips": len(meta), "n_files": len(files), "missing_outputs": missing[:20],
                 "n_missing_outputs": len(missing), "rows": n_rows, "by_split_camera": {}, "presses": {}}
    for (split, cam), g in sorted(groups.items()):
        s = lambda k: float(sum(float(d[k]) for d in g))  # noqa: E731
        cls_out = {}
        for cls in ("person", "forklift"):
            gl_l, ob_l, spans = [], [], []
            n_rows_c = 0
            for d in g:
                t = tracks_by_clip.get(d["clip_id"])
                if t is None or not len(t):
                    continue
                tc = t.loc[t["class"] == cls]
                n_rows_c += len(tc)
                if not len(tc):
                    continue
                x = track_gaps(tc)
                gl_l.append(x.loc[x["kind"] == "gap", "n_frames"].to_numpy(int))
                ob_l.append(x.loc[x["kind"] == "obs", "n_frames"].to_numpy(int))
                tid = tc["track_id"].to_numpy(int)
                tt = tc["t"].to_numpy(float)
                spans += [float(tt[tid == k].max() - tt[tid == k].min()) for k in np.unique(tid)]
            gl = np.concatenate(gl_l) if gl_l else np.zeros(0, int)
            ob = np.concatenate(ob_l) if ob_l else np.zeros(0, int)
            n_tracks = int(sum(int(d["n_tracks"].get(cls, 0)) for d in g))
            cls_out[cls] = {
                "n_tracks": n_tracks, "n_rows": n_rows_c, "track_span_s_p50_p90": _q(spans, (0.5, 0.9)),
                "n_gaps": int(len(gl)), "n_obs_runs": int(len(ob)),
                "gap_frames_hist": {str(k): int(v) for k, v in sorted(Counter(np.minimum(gl, 36).tolist()).items())},
                "gap_frames_mean": round(float(gl.mean()), 3) if len(gl) else None,
                "gap_frames_p50_p90_p99": _q(gl), "obs_run_frames_mean": round(float(ob.mean()), 3) if len(ob) else None,
                "obs_run_frames_p50_p90": _q(ob, (0.5, 0.9)),
                # two-state Markov view on the 12 fps frames inside a track's life: P(observed -> missed) is gaps over
                # observed frames that have a next frame; P(missed -> observed) is gaps over missed frames
                "p_enter_gap": round(len(gl) / max(int(ob.sum()) - n_tracks, 1), 4) if n_tracks else None,
                "p_leave_gap": round(len(gl) / max(int(gl.sum()), 1), 4) if len(gl) else None,
            }
        ns = s("n_samples")
        out["by_split_camera"][f"{split}_{cam}"] = {
            "clips": len(g), "video_s": round(s("duration_pts"), 1), "frames_sampled": int(ns),
            "frames_native": int(s("n_native")),
            "fps_sampled_mean_min": [round(s("fps_sampled") / len(g), 3), round(min(float(d["fps_sampled"]) for d in g), 3)],
            "fps_native_mean": round(s("fps_native") / len(g), 3),
            "det_fps_per_container": round(ns / s("det_s"), 1),
            "clip_wall_s_sum": round(s("wall_s"), 1),
            "person_per_frame_conf05_all_frames": round(s("person_boxes_05") / ns, 3),
            "person_per_frame_conf05_t05": round(s("person_at_05s_conf05") / len(g), 3),
            "staging_person_per_frame_t05": STAGING_PPF.get((split, cam)),
            "person_boxes_per_frame_conf01": round(s("person_boxes") / ns, 3),
            "forklift_boxes": int(s("forklift_boxes")), "suppressed_person_boxes": int(s("suppressed")),
            "classes": cls_out,
        }
    pg: dict[str, list[dict]] = {}
    for r in mach:
        pg.setdefault(f"{r['press']}_{r['split']}", []).append(r)
    for key, g in sorted(pg.items()):
        n = sum(r["n"] for r in g)
        out["presses"][key] = {"clips": len(g), "frames": n, "missing_share": round(sum(r["missing"] for r in g) / n, 3),
                               "strokes": sum(r["strokes"] for r in g),
                               "clips_ge2_strokes": sum(1 for r in g if r["strokes"] >= 2)}
    lines = [f"{_sha(p)}  {p.relative_to(root).as_posix()}" for p in files]
    a.sha_out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    js = a.json_out or (root / "perc" / "summary.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
