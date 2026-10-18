"""Fit DYN v1 from TRAIN clips (lane S3a, PLAN_TWIN §5 DYN/DAYGEN): twin/dynamics.json, twin/dyn/press_v1.json,
twin/dyn/valcam_v1.json, the path library $INTERCYCLE_DATA/dyn/path_library_v1.npz (gitignored; its sha256 is in
dynamics.json) and the train press MISSING masks $INTERCYCLE_DATA/dyn/press_masks_v1.npz (DYN validation only).
Both .npz files are plain numpy arrays: the generator (twin/daygen.py) is Isaac-side and may not import pandas.

Inputs: PERC tracks and machines tables of the A1+ TRAIN split (`esk.load("train")`), projected to the floor with
CAL v1 (`twin/site.yaml` H and k1 per framing, `project_tracks`), and the train keyframes' mean grey level (luma).
The arrival-rate correction uses person recall measured on the DEV labels (labels/dev, or the `--labels-ref` git
ref when labels/dev is not in the tree) against the DEV tracks; it only enters ASSUMED parameters (their notes say
so). The labels' scored.json must have the pinned sha256 (`SCORED_SHA256`, or `--labels-sha256`): other labels fail
closed instead of silently changing the rates; its full sha256 and source (tree or ref + commit) are recorded.
Fail-closed order: everything is computed in memory first; `twin.provenance.assert_train_only` then checks every
clip id set the outputs come from (fitted parameters, library, press clips / judged / running, press masks, layout
states, the jitter and gap clips of the validation camera) and only then is any file written. An .npz whose arrays
are unchanged is not rewritten (np.savez stamps the zip with the time, so a rewrite would change the library sha256
pinned in dynamics.json for the same content).

    python scripts/dyn_fit.py [--labels-ref lane/S4-labels-fix2] [--labels-sha256 <sha>]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle.perception.esk_ingest import project_tracks, track_gaps  # noqa: E402
from twin import dynfit as DF  # noqa: E402
from twin.provenance import assert_train_only  # noqa: E402
from twin.schemas import Dynamics, Site, load, parse_split  # noqa: E402

SITE = ROOT / "twin" / "site.yaml"
OUT_DYN = ROOT / "twin" / "dynamics.json"
OUT_PRESS = ROOT / "twin" / "dyn" / "press_v1.json"
OUT_CAM = ROOT / "twin" / "dyn" / "valcam_v1.json"
LIB_NAME = "dyn/path_library_v1.npz"
MASKS_NAME = "dyn/press_masks_v1.npz"    # train press clips' frame times + MISSING masks (DYN validation windows)
IMPULSE_HALF_S = 0.8               # +-0.8 s around a stroke (under half of the shorter learned period, 1.69 s)
FORKLIFT_RECALL = (8, 13)          # P5 agent check on camera 14 (lead decision): 8 of 13 forklifts found, precision 1.00
UNSAFE_PREFIXES = (0, 1, 2, 3)     # scripts/esk_split.py: class prefixes 0-3 are unsafe behaviour clips
LABELS_REF = "lane/S4-labels-fix2"  # DEV labels (S4) used for the recall correction; not merged into main yet
SCORED_SHA256 = "9e68d146bd0807b91b8a1aa119d83325f684fafc24b43e504b40cf90b9d78380"   # its labels/dev/scored.json (LF)
SHARED_IMPULSE_PRIOR_S = 0.6       # dynamics_v0 `press.stroke_impulse_s` (the engine's one ram impulse for all presses)
LIBRARY_COLUMNS = ["unit_id", "seg_id", "zone", "cls", "clip_id", "track_id", "piece", "group_id", "framing",
                   "lum_class", "unsafe", "clip_dur", "unit_t0", "seg_t_off", "dt", "x", "z"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plain_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Plain (non-object) dtypes, so np.load(allow_pickle=False) reads them."""
    out = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.dtype == object:
            a = a.astype(str)
        out[k] = a
    return out


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **plain_arrays(arrays))  # type: ignore[arg-type]
    return buf.getvalue()


def same_npz(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    """True when `path` holds exactly these arrays (same names, dtypes, shapes and values)."""
    if not path.exists():
        return False
    want = plain_arrays(arrays)
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(want):
            return False
        return all(z[k].dtype == want[k].dtype and z[k].shape == want[k].shape and np.array_equal(z[k], want[k])
                   for k in want)


def put_npz(path: Path, arrays: dict[str, np.ndarray], raw: bytes) -> str:
    """Write the npz unless the file already holds the same arrays; returns the sha256 of the file on disk."""
    if not same_npz(path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return sha256_file(path)


def npz_sha(path: Path, arrays: dict[str, np.ndarray], raw: bytes) -> str:
    """sha256 the file will have after `put_npz` (the existing file's when its arrays are unchanged)."""
    return sha256_file(path) if same_npz(path, arrays) else hashlib.sha256(raw).hexdigest()


def sha256_lf(p: Path) -> str:
    """sha256 of a text file with CRLF normalised to LF (= the git blob under core.autocrlf)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def r(x: float | np.floating, n: int = 4) -> float:
    return float(round(float(x), n))


def param(value, unit, source, n, lo, hi, *, dist=None, ref=None, clips=(), note=None, time_of_day=False) -> dict:
    lo, hi = min(lo, value), max(hi, value)
    d = {"value": r(value, 6), "unit": unit, "source": source, "n": int(n), "sweep": {"lo": r(lo, 6), "hi": r(hi, 6)}}
    if dist:
        d["dist"] = dist
    if ref:
        d["ref"] = ref
    if clips:
        d["clips"] = sorted(set(clips))
    if time_of_day:
        d["time_of_day"] = True
    if note:
        d["note"] = note
    return d


# ------------------------------------------------------------------------------------------------ luma

def clip_luma(data: Path, clip_ids) -> dict[str, float]:
    import cv2

    out = {}
    for cid in clip_ids:
        v = []
        for k in (1, 2):
            im = cv2.imread(str(data / "keyframes" / f"{cid}_k{k}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                v.append(float(im.mean()))
        if not v:
            raise FileNotFoundError(f"no keyframe for train clip {cid}")
        out[cid] = float(np.mean(v))
    return out


def two_means_split(x: np.ndarray) -> float:
    """1-D 2-means threshold (Lloyd from the min / max)."""
    a, b = float(np.min(x)), float(np.max(x))
    for _ in range(100):
        thr = (a + b) / 2
        lo, hi = x[x < thr], x[x >= thr]
        na, nb = (float(lo.mean()) if len(lo) else a), (float(hi.mean()) if len(hi) else b)
        if abs(na - a) < 1e-9 and abs(nb - b) < 1e-9:
            break
        a, b = na, nb
    return (a + b) / 2


# ------------------------------------------------------------------------------------------------ recall (dev)

def labels_source(ref: str | None) -> str:
    """Where load_labels reads labels/dev from: the tree, else `<ref> @ <commit>`."""
    if (ROOT / "labels" / "dev" / "scored.json").exists():
        return "labels/dev (tree)"
    if not ref:
        raise FileNotFoundError("labels/dev missing; pass --labels-ref <git ref holding labels/dev>")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], check=True,
                            capture_output=True, text=True).stdout.strip()
    return f"{ref} @ {commit}"


def load_labels(ref: str | None, want_sha256: str = SCORED_SHA256) -> tuple[list[dict], str]:
    """Scored DEV label frames (labels/dev/scored.json statuses) and the sha256 (LF) of scored.json, which must be
    `want_sha256` (fail closed: other labels would change the ASSUMED rates without a deliberate re-pin)."""
    def read(name: str) -> bytes:
        p = ROOT / "labels" / "dev" / name
        if p.exists():
            return p.read_bytes()
        if not ref:
            raise FileNotFoundError(f"{p} missing; pass --labels-ref <git ref holding labels/dev>")
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:labels/dev/{name}"], check=True,
                              capture_output=True).stdout

    scored_b = read("scored.json")
    sha = hashlib.sha256(scored_b.replace(b"\r\n", b"\n")).hexdigest()
    if sha != want_sha256:
        raise ValueError(f"labels/dev/scored.json sha256 {sha} != pinned {want_sha256}: re-pin deliberately "
                         "(--labels-sha256) to fit on other labels")
    scored = json.loads(scored_b)
    sets: dict[str, dict] = {}
    frames = []
    for f in scored["frames"]:
        if f["status"] != "scored":
            continue
        ls = f["label_set"]
        if ls not in sets:
            sets[ls] = {fr["sample_id"]: fr for fr in json.loads(read(ls))["frames"]}
        frames.append({**sets[ls][f["sample_id"]], "camera": int(f["camera"])})
    return frames, sha


def person_recall(frames: list[dict], data: Path) -> dict[int, tuple[int, int]]:
    """Per camera: (labelled persons matched by a DEV person track row, labelled persons). A labelled box is matched
    by a track row of its clip within half a 12 fps frame of the label time whose foot point lies in the box padded
    by 10 %; each row matches at most one box."""
    dev = set(esk.load("dev")["clip_id"])
    out: dict[int, list[int]] = {}
    for fr in frames:
        if fr["clip_id"] not in dev:
            raise ValueError(f"label frame {fr['sample_id']} is not a dev clip")
        tr_all = pd.read_parquet(data / "tracks" / f"{fr['clip_id']}.parquet")
        sel = (tr_all["class"] == "person") & ((tr_all["t"] - fr["t_pts"]).abs() <= 0.5 / 12 + 1e-6)
        tr = cast(pd.DataFrame, tr_all.loc[sel])
        used: set[int] = set()
        k = out.setdefault(fr["camera"], [0, 0])
        for b in fr["boxes"]:
            if b["cls"] != "person":
                continue
            k[1] += 1
            pw, ph = 0.1 * (b["x1"] - b["x0"]), 0.1 * (b["y1"] - b["y0"])
            for i, (u, v) in zip(tr.index, tr[["u", "v"]].to_numpy(float)):
                if i in used:
                    continue
                if b["x0"] - pw <= u <= b["x1"] + pw and b["y0"] - ph <= v <= b["y1"] + ph:
                    used.add(i)
                    k[0] += 1
                    break
    return {c: (v[0], v[1]) for c, v in out.items()}


# ------------------------------------------------------------------------------------------------ main fit

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-ref", default=LABELS_REF, help="git ref to read labels/dev from when not in the tree")
    ap.add_argument("--labels-sha256", default=SCORED_SHA256, help="required sha256 (LF) of labels/dev/scored.json")
    a = ap.parse_args()

    data = esk.data_root()
    site = load(Site, SITE)
    H = {f.id: f.H_img_to_floor.value for f in site.framings}
    K1 = {f.id: float(f.k1.value) for f in site.framings}
    zone_of_cam = {z.camera: z for z in site.zones}
    train = esk.load("train")
    tr_ids = set(train["clip_id"])
    inv = train.set_index("clip_id")

    # ---- luma -> lighting class per clip and per layout state (session cluster)
    lum = clip_luma(data, train["clip_id"])
    lum_class: dict[str, str] = {}
    lum_info: dict[str, dict] = {}
    for cam in (9, 14):
        ids = list(train.loc[train["camera"] == cam, "clip_id"])
        x = np.array([lum[c] for c in ids])
        thr = two_means_split(x)
        for c in ids:
            lum_class[c] = "dim" if lum[c] < thr else "bright"
        dur = inv.loc[ids, "duration"].to_numpy(float)
        dim = np.array([lum_class[c] == "dim" for c in ids])
        lum_info[str(cam)] = {"threshold": r(thr, 2), "luma_min": r(x.min(), 1), "luma_max": r(x.max(), 1),
                              "dim_clip_seconds_share": r(dur[dim].sum() / dur.sum()), "n_clips": len(ids)}
    states: dict[str, dict] = {}
    for gid, g in train.groupby("group_id"):
        secs = g["duration"].to_numpy(float)
        dim_s = float(secs[[lum_class[c] == "dim" for c in g["clip_id"]]].sum())
        fr_counts = g.groupby("framing_id")["duration"].sum()
        states[str(gid)] = {"zone": zone_of_cam[int(g["camera"].iloc[0])].id, "clip_seconds": r(secs.sum(), 3),
                            "lum_class": "dim" if dim_s > secs.sum() / 2 else "bright",
                            "mean_luma": r(np.mean([lum[c] for c in g["clip_id"]]), 2),
                            "framing": str(fr_counts.idxmax()), "n_clips": int(len(g))}

    # ---- tracks -> pieces -> units (groups) -> library
    lib_rows: list[tuple] = []
    units: list[dict] = []
    speeds: dict[tuple[str, str], list[float]] = {}
    speed_clips: dict[tuple[str, str], set[str]] = {}
    group_sizes: list[int] = []
    jitter_tracks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    gap_tabs: dict[int, list[pd.DataFrame]] = {}
    valcam_clips: set[str] = set()                  # clips whose tracks fit the validation camera (jitter, gaps)
    for cid in sorted(tr_ids):
        row = inv.loc[cid]
        cam, fr = int(row["camera"]), str(row["framing_id"])
        zone = zone_of_cam[cam]
        poly = zone.floor_polygon.value
        raw = pd.read_parquet(data / "tracks" / f"{cid}.parquet")
        if len(raw) == 0:
            continue
        for _, g in raw.groupby(["class", "track_id"], sort=False):
            g = g.sort_values("t")
            jitter_tracks.setdefault(fr, []).append((g["u"].to_numpy(float), g["v"].to_numpy(float)))
        gap_tabs.setdefault(cam, []).append(track_gaps(cast(pd.DataFrame, raw.loc[raw["class"] == "person"])))
        valcam_clips.add(cid)
        tr = project_tracks(raw, H, K1)
        pieces: dict[str, list[tuple]] = {"person": [], "forklift": []}
        for key, g in tr.groupby(["class", "track_id"], sort=True):
            cls, tid = str(cast(tuple, key)[0]), int(cast(tuple, key)[1])
            if cls not in pieces:
                continue
            ps = DF.observed_pieces(g["t"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy(), poly, cls, DF.MIN_SEG_S)
            for k, p in enumerate(ps):
                pieces[cls].append((tid, k, p))
                if p[0][-1] - p[0][0] >= DF.MIN_STAT_S:
                    speeds.setdefault((zone.id, cls), []).append(DF.track_speed(*p))
                    speed_clips.setdefault((zone.id, cls), set()).add(cid)
        for cls, plist in pieces.items():
            if not plist:
                continue
            lab = DF.link_groups([p for _, _, p in plist]) if cls == "person" else np.arange(len(plist))
            for g_id in np.unique(lab):
                members = [plist[i] for i in np.nonzero(lab == g_id)[0]]
                t0 = min(p[0][0] for _, _, p in members)
                uid = f"{cid}:{cls}:{int(g_id)}"
                if cls == "person":
                    group_sizes.append(len(members))
                units.append({"unit_id": uid, "zone": zone.id, "cls": cls, "clip_id": cid, "n": len(members),
                              "group_id": str(row["group_id"]), "framing": fr})
                for tid, k, (t, x, z) in members:
                    keep = DF.decimate(t)
                    sid = f"seg:{cid}:{cls}:{tid}:{k}"
                    for i in keep:
                        lib_rows.append((uid, sid, zone.id, cls, cid, tid, k, str(row["group_id"]), fr,
                                         lum_class[cid], int(cid.split("_")[0]) in UNSAFE_PREFIXES,
                                         float(row["duration"]), float(t0), float(t[0] - t0), float(t[i] - t[0]),
                                         float(x[i]), float(z[i])))
    lib = pd.DataFrame(lib_rows, columns=pd.Index(LIBRARY_COLUMNS))
    lib_path = data / LIB_NAME
    lib_arrays = {c: lib[c].to_numpy() for c in LIBRARY_COLUMNS}               # numpy-readable (Isaac-side generator)
    lib_raw = npz_bytes(lib_arrays)
    lib_sha = npz_sha(lib_path, lib_arrays, lib_raw)                        # written only after the provenance check
    udf = pd.DataFrame(units)
    lib_clips = sorted(set(lib["clip_id"]))

    # ---- validation-camera parameters (train tracks)
    valcam: dict = {"contract": "intercycle.dyn_valcam/1", "framings": {}, "cameras": {}, "presses": {}}
    for fr, tabs in jitter_tracks.items():
        su, sv, n = DF.fit_pixel_jitter(tabs)
        valcam["framings"][fr] = {"sigma_u_px": r(su, 3), "sigma_v_px": r(sv, 3), "n_rows": n}
    for cam, tabs in gap_tabs.items():
        gt = pd.concat(tabs)
        n = int(gt.loc[gt["kind"] == "obs", "n_frames"].sum())
        gap = gt[gt["kind"] == "gap"]
        p0, p1 = DF.fit_miss_chain(n, len(gap), int(gap["n_frames"].sum()))
        valcam["cameras"][str(cam)] = {"p_miss_start": r(p0, 5), "p_miss_end": r(p1, 5), "n_frames": n}

    # ---- presses
    press_tab: dict = {"contract": "intercycle.dyn_press/1", "presses": {}}
    press_params: dict[str, dict] = {}
    masks: list[tuple[str, np.ndarray, np.ndarray]] = []
    mask_clips: set[str] = set()
    for p in site.presses:
        mdir = data / "machines" / p.id
        cids = sorted(c for c in (f.stem for f in mdir.glob("*.parquet")) if c in tr_ids)
        comp_all, cens_all, periods, judged, running, prof, jit = [], [], [], [], [], [], []
        dt_frame = math.nan
        for cid in cids:
            m = pd.read_parquet(mdir / f"{cid}.parquet")
            t = m["t"].to_numpy(float)
            st, miss = m["stroke"].to_numpy(bool), m["missing"].to_numpy(bool)
            if len(t) > 2:
                masks.append((p.id, t, miss))
                mask_clips.add(cid)
            if len(t) > 2:
                dt_frame = float(np.median(np.diff(t)))
            obs_s = float((~miss).sum() * (np.median(np.diff(t)) if len(t) > 2 else 0.0))
            if obs_s >= DF.JUDGE_S:
                judged.append(cid)
                if int(st.sum()) >= DF.RUN_MIN_STROKES:
                    running.append(cid)
            comp, cens = DF.stroke_intervals(t, st, miss)
            comp_all.append(comp)
            cens_all.append(cens)
            prof += DF.impulse_fwhm(t, m["flow_v"].to_numpy(float), st, miss, half_s=IMPULSE_HALF_S)
        comp = np.concatenate(comp_all) if comp_all else np.zeros(0)
        cens = np.concatenate(cens_all) if cens_all else np.zeros(0)
        a_post, b_post = DF.beta_posterior(len(running), len(judged))
        entry: dict = {"clips": cids, "judged": len(judged), "running": len(running),
                       "beta": [a_post, b_post], "n_complete": int(len(comp)), "n_censored": int(len(cens))}
        learned = len(comp) >= 10
        if learned:
            med_all = float(np.median(comp))
            for cid, cc in zip(cids, comp_all):
                mm = cc[(cc >= DF.MODE_LO * med_all) & (cc <= DF.MODE_HI * med_all)]
                if len(mm) >= 1:
                    periods.append(float(np.median(mm)))
                    jit += list(mm - np.median(mm))
            period = float(np.median(periods))
            km_t, km_s = DF.kaplan_meier(comp, cens)
            pause_lo = DF.MODE_HI * period
            # pauses with an observed duration: KM mass between pause_lo and the last complete interval; the mass
            # beyond the last complete interval has no observed duration (it goes to the long-stop process)
            q_beyond = DF.km_survival_at(km_t, km_s, pause_lo)
            q = (q_beyond - float(km_s[-1])) if len(km_t) and km_t[-1] > pause_lo else 0.0
            mean_prof = np.mean(prof, axis=0) if prof else np.zeros(0)
            impulse = DF.fwhm_s(mean_prof, dt_frame) if len(mean_prof) else math.nan
            jit_a = np.asarray(jit)
            jitter_sd = float(1.4826 * np.median(np.abs(jit_a)) / math.sqrt(2)) if len(jit_a) else 0.0
            entry.update({"period_median_s": r(period), "periods_s": [r(x) for x in periods],
                          "km_t": [r(x) for x in km_t], "km_s": [r(x, 6) for x in km_s],
                          "pause_lo_s": r(pause_lo), "pause_share": r(q, 6), "km_s_at_pause_lo": r(q_beyond, 6), "impulse_fwhm_s": r(impulse),
                          "n_impulse": len(prof), "km_last_s": r(km_t[-1]) if len(km_t) else None,
                          "km_s_last": r(km_s[-1], 6) if len(km_s) else None})
            valcam["presses"][p.id] = {"stroke_jitter_s": r(jitter_sd, 5), "n": int(len(jit_a)),
                                       "frame_dt_s": r(dt_frame, 5)}
        press_tab["presses"][p.id] = entry
        press_params[p.id] = {**entry, "learned": learned, "judged": judged, "running": running, "cids": cids}

    starts = np.cumsum([0] + [len(t) for _, t, _ in masks[:-1]])
    mask_arrays = {"press": np.array([p for p, _, _ in masks]), "start": starts.astype(np.int64),
                   "t": np.concatenate([t for _, t, _ in masks]), "missing": np.concatenate([m for _, _, m in masks])}
    mask_raw = npz_bytes(mask_arrays)

    # ---- recall (dev labels vs dev tracks) for the ASSUMED arrival-rate correction
    labels_src = labels_source(a.labels_ref)
    frames, scored_sha = load_labels(a.labels_ref, a.labels_sha256)
    rec = person_recall(frames, data)

    # ---- press tables (in memory; written after the provenance check)
    zone_states = {zid: {g: s for g, s in states.items() if s["zone"] == zid}
                   for zid in sorted({s["zone"] for s in states.values()})}
    press_tab["layout_states"] = zone_states
    press_tab["luma"] = lum_info
    press_tab["library"] = {"path": f"$INTERCYCLE_DATA/{LIB_NAME}", "sha256": lib_sha, "rows": int(len(lib)),
                            "units": int(len(udf)), "segments": int(lib["seg_id"].nunique())}
    press_txt = json.dumps(press_tab, indent=1) + "\n"
    cam_txt = json.dumps(valcam, indent=1) + "\n"
    press_sha = hashlib.sha256(press_txt.encode("utf-8")).hexdigest()           # = sha256_lf of the LF file

    # ---- dynamics.json
    hours = {cam: float(train.loc[train["camera"] == cam, "duration"].sum()) / 3600.0 for cam in (9, 14)}
    obs: dict[str, dict] = {}
    asm: dict[str, dict] = {}
    lib_ref = f"$INTERCYCLE_DATA/{LIB_NAME} sha256:{lib_sha}"
    press_ref = f"twin/dyn/press_v1.json sha256(LF):{press_sha}"
    for (zid, cls), v in sorted(speeds.items()):
        v_a = np.asarray(v)
        v_a = v_a[np.isfinite(v_a)]
        p10, p50, p90 = np.percentile(v_a, [10, 50, 90])
        name = f"{cls}.{zid}.speed_mps"
        obs[name] = param(p50, "m/s", "fitted", len(v_a), p10, p90, clips=speed_clips[(zid, cls)],
                          dist={"family": "empirical", "params": {"p10": r(p10), "p50": r(p50), "p90": r(p90),
                                                                  "mean": r(v_a.mean())}},
                          ref=lib_ref, note="per-track median floor speed over 1 s of the smoothed track (twin.dynfit."
                          "track_speed), pieces >= 2 s on the zone floor; resampled as is (speed_scale 1)")
    gs = np.asarray(group_sizes)
    obs["person.group_size"] = param(gs.mean(), "people", "fitted", len(gs), 1.0, float(gs.max()),
                                     clips=sorted(set(udf.loc[udf["cls"] == "person", "clip_id"])),
                                     dist={"family": "empirical", "params": {
                                         "share_1": r((gs == 1).mean()), "share_2": r((gs == 2).mean()),
                                         "share_3plus": r((gs >= 3).mean())}},
                                     note=f"pieces overlapping >= {DF.GROUP_OVERLAP_S} s at median distance <= "
                                          f"{DF.GROUP_DIST_M} m are one group (twin.dynfit.link_groups); a group is "
                                          "resampled whole")
    for zid in sorted(udf["zone"].unique()):
        for cls in ("person", "forklift"):
            u = udf[(udf["zone"] == zid) & (udf["cls"] == cls)]
            if len(u) == 0:
                continue
            obs[f"{cls}.{zid}.path_library"] = param(
                1.0, "scale", "fitted", len(u), 1.0, 1.0, ref=lib_ref, clips=sorted(set(u["clip_id"])),
                note=f"{len(u)} resampling units ({int(u['n'].sum())} smoothed track pieces, rolling median "
                     f"{DF.SMOOTH_N} samples, one point per {DF.DECIM_S} s) from train tracks projected with CAL v1")
    for zid, st in zone_states.items():
        cl = sorted(train.loc[train["group_id"].isin(list(st)), "clip_id"])
        tot = sum(s["clip_seconds"] for s in st.values())
        obs[f"layout.{zid}.states"] = param(
            float(len(st)), "states", "fitted", len(cl), float(len(st)), float(len(st)), clips=cl, ref=press_ref,
            dist={"family": "empirical", "params": {g: r(s["clip_seconds"] / tot, 5) for g, s in sorted(st.items())}},
            note="layout states = the A1+ session clusters (group_id) of the train clips, weighted by clip-seconds; "
                 "a day draws one bright and one dim state per zone (states and their luma class in "
                 "twin/dyn/press_v1.json 'layout_states')")
    dim_secs = sum(float(inv.loc[c, "duration"]) for c in tr_ids if lum_class[c] == "dim")
    dim_share = dim_secs / float(train["duration"].sum())
    obs["luma.dim_clip_share"] = param(
        dim_share, "fraction", "fitted", len(tr_ids), 0.0, 1.0, clips=sorted(tr_ids),
        dist={"family": "empirical", "params": {f"c{c}_threshold": v["threshold"] for c, v in lum_info.items()}},
        note="share of train clip-seconds in the dimmer of two luma classes (2-means split of keyframe mean grey per "
             "camera; per-camera ranges in twin/dyn/press_v1.json 'luma'). Footage is event-selected; what the dim "
             "class means (night, weather, lamps) is not observed")
    for pid, e in press_params.items():
        a_post, b_post = e["beta"]
        mean = a_post / (a_post + b_post)
        lo, hi = _beta_q(a_post, b_post, 0.05), _beta_q(a_post, b_post, 0.95)
        obs[f"press.{pid}.p_running"] = param(
            mean, "probability", "fitted", len(e["judged"]), lo, hi, clips=e["judged"],
            dist={"family": "beta", "params": {"a": a_post, "b": b_post}},
            note=f"Beta(1,1) prior + {len(e['running'])} running of {len(e['judged'])} judged train clips (ROI observed >= "
                 f"{DF.JUDGE_S} s; running = >= {DF.RUN_MIN_STROKES} strokes, P4 rule); drawn once per world")
        if e["learned"]:
            per = np.asarray(e["periods_s"])
            obs[f"press.{pid}.stroke_period_s"] = param(
                e["period_median_s"], "s", "fitted", len(per), float(per.min()), float(per.max()), clips=e["cids"],
                ref=press_ref, dist={"family": "empirical", "params": {
                    "p10": r(np.percentile(per, 10)), "p50": r(np.percentile(per, 50)), "p90": r(np.percentile(per, 90))}},
                note="per-clip median of the main-mode in-clip intervals; a run bout draws one of them")
            obs[f"press.{pid}.interval_km"] = param(
                e["pause_share"], "probability", "fitted_censored", e["n_complete"] + e["n_censored"], 0.0, 1.0,
                clips=e["cids"], ref=press_ref,
                dist={"family": "kaplan_meier", "params": {"n_complete": e["n_complete"], "n_censored": e["n_censored"],
                                                           "pause_lo_s": e["pause_lo_s"]}},
                note=f"value = KM share of strokes followed by an in-clip pause (> {DF.MODE_HI} x period) on the KM's "
                     "observed support (complete intervals; censored at the clip end and at MISSING). The KM mass "
                     f"beyond the last complete interval (P(T > pause_lo) = {e['km_s_at_pause_lo']}) has no observed "
                     "duration and goes to the ASSUMED long-stop process (W-W3); long intervals include detector misses")
            obs[f"press.{pid}.stroke_impulse_s"] = param(
                e["impulse_fwhm_s"], "s", "fitted", e["n_impulse"], 0.5 * e["impulse_fwhm_s"],
                1.5 * e["impulse_fwhm_s"], clips=e["cids"],
                note=f"full width at half maximum of the mean |flow_v| profile in +-{IMPULSE_HALF_S} s around the "
                     "train strokes (ROI-mean flow, not the ram travel itself)"
                     + ("; NOT RESOLVED: the width reaches the window (the ROI moves through the whole cycle), so "
                        "this is a lower bound of a near-continuous motion, not an impulse"
                        if e["impulse_fwhm_s"] >= 0.9 * 2 * IMPULSE_HALF_S else ""))
        else:
            asm[f"press.{pid}.stroke_period_s"] = param(
                4.0, "s", "prior", 0, 2.0, 8.0, dist={"family": "lognormal", "params": {"median": 4.0, "sigma": 0.05}},
                note="ASSUMED (grey): P4 found no running train clip, so the period is the v0 prior")
            asm[f"press.{pid}.stroke_impulse_s"] = param(0.6, "s", "prior", 0, 0.3, 1.0, note="v0 prior")
            asm[f"press.{pid}.interval_km"] = param(0.0, "probability", "prior", 0, 0.0, 0.2,
                                                    note="no in-clip pauses without strokes; prior 0")
    # the engine (twin/engine.py) draws one ram impulse for every press from the shared key press.stroke_impulse_s
    # (the frozen v0 contract key): the per-press fits are not resolved impulses, so the shared value stays the prior
    fwhm = {pid: e["impulse_fwhm_s"] for pid, e in press_params.items() if e["learned"]}
    asm["press.stroke_impulse_s"] = param(
        SHARED_IMPULSE_PRIOR_S, "s", "prior", 0, 0.3, max([1.0, *fwhm.values()]),
        note="ram down-and-up duration of one stroke, shared by all presses (read by twin/engine.py): the v0 prior, "
             "ASSUMED. Not fitted: the per-press values press.<id>.stroke_impulse_s ("
             + ", ".join(f"{p} {v:.2f} s" for p, v in sorted(fwhm.items()))
             + ") are the FWHM of the ROI-mean |flow_v| and are NOT RESOLVED (the width reaches the "
             f"+-{IMPULSE_HALF_S} s window), so they are a lower bound of a near-continuous ROI motion, not the ram "
             "impulse; the sweep reaches the largest of them")
    # ---- ASSUMED block
    rec_note = ", ".join(f"camera {c}: {k}/{n}" for c, (k, n) in sorted(rec.items()))
    for zid, cam in (("z9", 9), ("z14", 14)):
        u = udf[(udf["zone"] == zid) & (udf["cls"] == "person")]
        k, n = rec.get(cam, (0, 0))
        recall = k / n if n else 1.0
        obs_rate = len(u) / hours[cam]
        rate = obs_rate / max(recall, 1e-6)
        asm[f"zone.{zid}.walker_rate_per_h"] = param(
            rate, "1/h", "assumed", 0, 0.1 * obs_rate, 1.2 * rate, dist={"family": "poisson", "params": {}},
            note=f"ASSUMED: observed {len(u)} person units in {hours[cam]:.3f} h of train footage ({obs_rate:.1f}/h) "
                 f"divided by DEV person recall {recall:.3f} (dev labels vs dev tracks, {rec_note}; labels/dev/"
                 f"scored.json sha256(LF) {scored_sha} from {labels_src}). The footage is event-selected, so this is "
                 "the crowding the clips show, not an average hour; sweep down to 0.1x the observed rate. DAYGEN "
                 "spreads it over the layout states (twin.daygen.state_rate_per_h: each state's train units per "
                 "clip-hour / recall)")
    fl = udf[(udf["zone"] == "z14") & (udf["cls"] == "forklift")]
    fr_rec = FORKLIFT_RECALL[0] / FORKLIFT_RECALL[1]
    obs_fl = len(fl) / hours[14]
    asm["zone.z14.forklift_rate_per_h"] = param(
        obs_fl / fr_rec, "1/h", "assumed", 0, 0.1 * obs_fl, 1.2 * obs_fl / fr_rec,
        dist={"family": "poisson", "params": {}},
        note=f"ASSUMED: {len(fl)} forklift units in {hours[14]:.3f} h of camera-14 train footage ({obs_fl:.1f}/h) "
             f"divided by the P5 agent-check recall {FORKLIFT_RECALL[0]}/{FORKLIFT_RECALL[1]}; event-selected footage")
    asm["zone.z9.forklift_rate_per_h"] = param(
        0.0, "1/h", "assumed", 0, 0.0, 0.0,
        note="no vehicles observed in zone-9 train clips (P5; camera 9 has no vehicle class)")
    asm["staff.staffing_factor"] = param(1.0, "factor", "assumed", 0, 0.8, 1.2,
                                         dist={"family": "uniform", "params": {"lo": 0.8, "hi": 1.2}},
                                         note="staffing +-20 %: one draw per (world, day) scales the walker rates")
    sig = math.log(30.0) / 2.3263478740408408
    asm["press.stop_median_s"] = param(60.0, "s", "assumed", 0, 30.0, 300.0,
                                       dist={"family": "lognormal", "params": {"median": 60.0, "sigma": r(sig, 6)}},
                                       note="W-W3 long stops: lognormal, median 60 s, p99 30 min; run bouts are "
                                            "exponential with mean p/(1-p) x E[stop] so the long-run running share "
                                            "equals the world's P(running) draw")
    asm["press.stop_p90_s"] = param(60.0 * math.exp(1.2815515655446004 * sig), "s", "assumed", 0, 120.0, 1800.0,
                                    note="p90 of the W-W3 stop lognormal (p99 = 1800 s)")
    asm["person.violation_share"] = param(0.05, "fraction", "assumed", 0, 0.0, 0.2,
                                          note="share of person arrivals drawn from unsafe-behaviour clips (class "
                                               "prefixes 0-3); the footage share is event-selected "
                                               f"({r(float(udf.loc[udf['cls'] == 'person', 'clip_id'].map(lambda c: int(c.split('_')[0]) in UNSAFE_PREFIXES).mean()), 3)} of units)")
    asm["social.A_n"] = param(2.1, "N", "prior", 0, 1.0, 5.0, note="published social-force default (fitting is cut)")
    asm["social.B_m"] = param(0.3, "m", "prior", 0, 0.1, 1.0)
    asm["social.tau_s"] = param(0.5, "s", "prior", 0, 0.2, 1.0)
    asm["site.night_share"] = param(
        dim_share, "fraction", "assumed", 0, 0.0, 0.5, time_of_day=True,
        note="ASSUMED time of day: the dim luma share (luma.dim_clip_share) is mapped to night hours centred on "
             "midnight; the clips carry no hour-of-day or shift signal (P2: no timing information)")
    dyn = {"contract": "twin.dynamics/1", "site_id": site.site_id, "version": "v1-train-resampled",
           "observed": obs, "assumed": asm}
    model = Dynamics.model_validate(dyn)
    # the split rows are read through esk's pointer rule (prereg/<name>), as scripts/cal_v1.py does: the bare
    # twin.schemas.prereg.load_active_split resolves the pointer from the repo root and fails closed on it
    split_rows = parse_split(esk.active_split_path(ROOT).read_text(encoding="utf-8"))
    state_clips = sorted(train.loc[train["group_id"].isin([g for st in zone_states.values() for g in st]), "clip_id"])
    id_sets = {"dynamics": model.train_clips(), "library": lib_clips, "press_masks": sorted(mask_clips),
               "layout_states": state_clips, "valcam_jitter_gaps": sorted(valcam_clips)}
    for pid, e in press_params.items():
        id_sets.update({f"press:{pid}:clips": e["cids"], f"press:{pid}:judged": e["judged"],
                        f"press:{pid}:running": e["running"]})
    n = assert_train_only(id_sets, split_rows)          # fail closed BEFORE any output is written
    # ---- write (only after the provenance check)
    dyn_txt = json.dumps(dyn, indent=1) + "\n"
    Dynamics.model_validate_json(dyn_txt)
    lib_sha_disk = put_npz(lib_path, lib_arrays, lib_raw)
    if lib_sha_disk != lib_sha:
        raise RuntimeError(f"path library sha256 {lib_sha_disk} != the {lib_sha} pinned in dynamics.json")
    put_npz(data / MASKS_NAME, mask_arrays, mask_raw)
    OUT_PRESS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PRESS.write_text(press_txt, encoding="utf-8", newline="\n")
    OUT_CAM.write_text(cam_txt, encoding="utf-8", newline="\n")
    OUT_DYN.write_text(dyn_txt, encoding="utf-8", newline="\n")
    print(json.dumps({"train_ids_checked": n, "library": press_tab["library"], "recall": rec,
                      "labels": {"scored_sha256": scored_sha, "source": labels_src},
                      "dynamics_sha256_lf": sha256_lf(OUT_DYN), "press_sha256_lf": sha256_lf(OUT_PRESS)}, indent=1))


def _beta_q(a: float, b: float, q: float) -> float:
    from scipy.stats import beta

    return float(beta.ppf(q, a, b))


if __name__ == "__main__":
    main()
