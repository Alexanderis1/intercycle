"""Staging of the press-shop clips on Modal (lane S3a; PLAN_TWIN §3 steps 1-3, probes P1/P2/P3).

    modal run cloud/modal_esk.py --action pass1      # CPU: stream all 691 clips from HF, store features only
    modal run cloud/modal_esk.py --action split      # CPU: run scripts/esk_split.py ONCE on the features
    modal run cloud/modal_esk.py --action detect     # L4: RF-DETR people on the t=0.5 s frame of every clip
    modal run cloud/modal_esk.py --action report     # CPU: P1/P2/P3 aggregates (test clips: aggregates only)
    modal run cloud/modal_esk.py --action purge_test # CPU: delete every image/feature file of TEST clips
    modal run cloud/modal_esk.py --action check      # CPU: assert no TEST clip_id has image data on the volume
    modal run cloud/modal_esk.py --action pass2      # CPU: fetch ONLY train+dev ids of prereg/split.csv + keyframes

Volume ``intercycle-esk`` (shared store for the press-shop data)::

    /esk/meta/<clip>.json       sha256, ffprobe summary, container-PTS fps/duration       (deleted for TEST)
    /esk/feat/<clip>.npz        phash, bg (36x64x3), luma                                  (deleted for TEST)
    /esk/frame05/<clip>.jpg     one full-res frame at t = 0.5 s                            (deleted for TEST)
    /esk/det/<clip>.json        RF-DETR boxes on that frame                                (deleted for TEST)
    /esk/split/split.csv, summary.json, public.json, report.json, test_manifest.json
    /esk/clips/<clip>.mp4       Pass 2, train + dev only
    /esk/keyframes/<clip>_k{1,2}.jpg   Pass 2, 960 px wide, train + dev only

SEALED: nothing per-clip about a TEST clip is returned to the caller except the §2 inventory columns (clip_id, camera,
class, group, split, sha256, duration). Only aggregates of test clips leave Modal.

Dataset: HF Voxel51/Safe_and_Unsafe_Behaviours (CC BY 4.0), pinned to revision REV. No HF token is used.
Licences: PyAV BSD-3, OpenCV Apache-2.0, numpy/scipy BSD, RF-DETR Apache-2.0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "s3a-stage-esk"
VOLUME_NAME = "intercycle-esk"
VOL = "/vol"
ESK = f"{VOL}/esk"
REPO_ID = "Voxel51/Safe_and_Unsafe_Behaviours"
REV = "7369ff1830b5f6c00f7d7383eeeaa90704bfc5b5"
N_EXPECTED = 691
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
PER_CLIP_DIRS = ("meta", "feat", "frame05", "det", "clips", "keyframes")
KEYFRAME_W = 960
PERSON_CONF = 0.5  # people-per-frame threshold for the balance report
OCC_IOA = 0.10  # a person box is "occluded" if >= 10 % of its area is covered by another detected box

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("av==19.0.0", "numpy==2.2.6", "scipy==1.16.2", "opencv-python-headless==4.12.0.88")
    .add_local_file(str(ROOT / "scripts" / "esk_split.py"), "/root/esk_split.py")
    .add_local_file(str(ROOT / "intercycle" / "framing.py"), "/root/framing.py")
)
SPLIT_A1 = "split_a1.csv"
# P6 framing references (docs/probes/S3a-P6.md, twin/cal/k1_v0.json ref_clip_id), in P6's framing-index order
P6_REFS = ("0_te21", "0_te1", "3_te1")  # c9_f0, c9_f1, c14_f0
P6_SMALL = (960, 540)
P6_MIN_INLIERS = 60
P6_MAX_DISP_SMALL_PX = 12.0


def _prefetch_weights() -> None:
    import rfdetr

    rfdetr.RFDETRMedium(device="cpu")


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("rfdetr==1.11.1", "supervision==0.30.6", "transformers==5.18.0", "numpy", "pillow")
    .run_function(_prefetch_weights)
    .apt_install("libgl1", "libglib2.0-0")
)


# --------------------------------------------------------------------------- helpers (run in the container)


def _cid_meta(cid: str) -> tuple[int, int]:
    cls = int(cid.split("_")[0])
    return cls, CAMERA_OF_PREFIX[cls]


def _download(cid: str, dst: str) -> None:
    import time
    import urllib.request

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REV}/data/{cid}.mp4"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            return
        except Exception:  # 429 / transient network errors: back off
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ffprobe(path: str) -> dict:
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    fmt = j.get("format", {})
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    tags = dict(fmt.get("tags", {}))
    for s in streams:
        for k, val in (s.get("tags") or {}).items():
            tags.setdefault(f"stream{s.get('index')}:{k}", val)
    return {
        "format_name": fmt.get("format_name"),
        "format_duration": float(fmt["duration"]) if fmt.get("duration") else None,
        "creation_time": (fmt.get("tags") or {}).get("creation_time"),
        "stream_creation_times": sorted({(s.get("tags") or {}).get("creation_time") for s in streams} - {None}),
        "n_audio": sum(s.get("codec_type") == "audio" for s in streams),
        "n_streams": len(streams),
        "codec": v.get("codec_name"),
        "r_frame_rate": v.get("r_frame_rate"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "nb_frames": v.get("nb_frames"),
        "width": v.get("width"),
        "height": v.get("height"),
        "tag_keys": sorted(tags),
        "encoder": tags.get("encoder"),
    }


def _phash(gray32) -> int:
    import cv2
    import numpy as np

    d = cv2.dct(gray32.astype(np.float32))[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _decode_features(path: str) -> dict:
    """One pass over the video: container-PTS timing, sampled frames -> pHash, 64x36 thumbs, luma, t=0.5 s frame."""
    import av
    import cv2
    import numpy as np

    sys.path.insert(0, "/root")
    import esk_split as S

    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur_guess = float(st.duration * st.time_base) if st.duration else None
        fps = S.PHASH_FPS_SHORT if (dur_guess is not None and dur_guess < S.SHORT_CLIP_S) else S.PHASH_FPS
        times: list[float] = []
        phash: list[int] = []
        thumbs = []
        luma = []
        frame05 = None
        next_t = 0.5 / fps
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base) if fr.pts is not None else (times[-1] + 0.04 if times else 0.0)
            times.append(t)
            t_rel = t - times[0]
            want05 = frame05 is None and t_rel >= 0.5
            if t_rel >= next_t or want05:
                rgb = fr.to_ndarray(format="rgb24")
                if want05:
                    frame05 = rgb
                if t_rel >= next_t:
                    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    phash.append(_phash(cv2.resize(y, (32, 32), interpolation=cv2.INTER_AREA)))
                    thumbs.append(cv2.resize(rgb, S.BG_SIZE, interpolation=cv2.INTER_AREA))
                    luma.append(cv2.resize(y, (160, 90), interpolation=cv2.INTER_AREA))
                    next_t += 1.0 / fps
    n = len(times)
    span = times[-1] - times[0] if n > 1 else 0.0
    fps_pts = (n - 1) / span if span > 0 else None
    dur_pts = n / fps_pts if fps_pts else None
    # a clip that turned out shorter than 3 s but was sampled at 1 fps is re-sampled at 4 fps in a second pass
    if dur_pts is not None and dur_pts < S.SHORT_CLIP_S and fps != S.PHASH_FPS_SHORT:
        return _decode_features_fixed(path, S.PHASH_FPS_SHORT, times, fps_pts, dur_pts)
    if not thumbs:  # clip shorter than one sampling step: use the frame at t=0.5 s or the last decoded frame
        raise RuntimeError(f"no sampled frames in {path}")
    ly = np.stack(luma).astype(np.float32)
    return {
        "phash": np.array(phash, dtype=np.uint64),
        "bg": np.median(np.stack(thumbs), axis=0).astype(np.uint8),
        "luma": np.array([ly.mean(), ly.std(), np.percentile(ly, 10), np.percentile(ly, 90)], np.float32),
        "frame05": frame05,
        "n_frames": n,
        "fps_pts": fps_pts,
        "duration_pts": dur_pts,
        "phash_fps": fps,
        "n_sampled": len(phash),
    }


def _decode_features_fixed(path, fps, times, fps_pts, dur_pts) -> dict:
    import av
    import cv2
    import numpy as np

    phash, thumbs, luma, frame05 = [], [], [], None
    next_t = 0.5 / fps
    t0 = times[0]
    with av.open(path) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            t_rel = float(fr.pts * st.time_base) - t0
            want05 = frame05 is None and t_rel >= 0.5
            if t_rel >= next_t or want05:
                rgb = fr.to_ndarray(format="rgb24")
                if want05:
                    frame05 = rgb
                if t_rel >= next_t:
                    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    phash.append(_phash(cv2.resize(y, (32, 32), interpolation=cv2.INTER_AREA)))
                    thumbs.append(cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA))
                    luma.append(cv2.resize(y, (160, 90), interpolation=cv2.INTER_AREA))
                    next_t += 1.0 / fps
    ly = np.stack(luma).astype(np.float32)
    return {"phash": np.array(phash, dtype=np.uint64), "bg": np.median(np.stack(thumbs), axis=0).astype(np.uint8),
            "luma": np.array([ly.mean(), ly.std(), np.percentile(ly, 10), np.percentile(ly, 90)], np.float32),
            "frame05": frame05, "n_frames": len(times), "fps_pts": fps_pts, "duration_pts": dur_pts,
            "phash_fps": fps, "n_sampled": len(phash)}


def _read_split(name: str = "split.csv") -> dict[str, dict]:
    import csv

    if name not in ("split.csv", SPLIT_A1):
        raise ValueError(f"unknown split file {name!r}")
    with open(f"{ESK}/split/{name}", newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


# --------------------------------------------------------------------------- Pass 1


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=3072, timeout=900, max_containers=24, retries=2)
def pass1_clip(cid: str) -> dict:
    """Features of one clip. Returns only {clip_id, ok, seconds}; everything else stays on the volume."""
    import os
    import time

    import cv2
    import numpy as np

    t0 = time.time()
    for d in ("meta", "feat", "frame05"):
        os.makedirs(f"{ESK}/{d}", exist_ok=True)
    tmp = f"/tmp/{cid}.mp4"
    _download(cid, tmp)
    sha = _sha256(tmp)
    probe = _ffprobe(tmp)
    f = _decode_features(tmp)
    cls, cam = _cid_meta(cid)
    np.savez(f"{ESK}/feat/{cid}.npz", phash=f["phash"], bg=f["bg"], luma=f["luma"])
    if f["frame05"] is not None:
        cv2.imwrite(f"{ESK}/frame05/{cid}.jpg", cv2.cvtColor(f["frame05"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta = {"clip_id": cid, "camera": cam, "cls": cls, "sha256": sha, "size_bytes": os.path.getsize(tmp),
            "n_frames": f["n_frames"], "fps_pts": f["fps_pts"], "duration_pts": f["duration_pts"],
            "phash_fps": f["phash_fps"], "n_sampled": f["n_sampled"], "has_frame05": f["frame05"] is not None,
            "ffprobe": probe}
    with open(f"{ESK}/meta/{cid}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.remove(tmp)
    vol.commit()
    return {"clip_id": cid, "ok": True, "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- split (run ONCE)


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=1800)
def run_split() -> dict:
    """Run scripts/esk_split.py once over the Pass 1 features. Refuses to run twice."""
    import glob
    import os
    import subprocess

    vol.reload()
    out_dir = f"{ESK}/split"
    if os.path.exists(f"{out_dir}/split.csv"):
        raise RuntimeError("split.csv already exists on the volume: the split script runs only once")
    os.makedirs(out_dir, exist_ok=True)
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{ESK}/meta/*.json"))]
    if len(metas) != N_EXPECTED:
        raise RuntimeError(f"expected {N_EXPECTED} clips with features, found {len(metas)}")
    slim = [{k: m[k] for k in ("clip_id", "camera", "cls", "sha256")} for m in metas]
    with open(f"{out_dir}/meta_in.json", "w", encoding="utf-8") as fh:
        json.dump(slim, fh)
    r = subprocess.run([sys.executable, "/root/esk_split.py", "--features", f"{ESK}/feat", "--meta",
                        f"{out_dir}/meta_in.json", "--out", f"{out_dir}/split.csv", "--summary",
                        f"{out_dir}/summary.json"], capture_output=True, text=True, check=True)
    # public per-clip metadata for inventory.parquet (§2 columns only)
    public = [{"clip_id": m["clip_id"], "camera": m["camera"], "cls": m["cls"], "sha256": m["sha256"],
               "duration": m["duration_pts"]} for m in metas]
    with open(f"{out_dir}/public.json", "w", encoding="utf-8") as fh:
        json.dump(public, fh)
    vol.commit()
    summary = json.load(open(f"{out_dir}/summary.json", encoding="utf-8"))
    return {"stdout": r.stdout[-4000:], "split_csv": open(f"{out_dir}/split.csv", encoding="utf-8").read(),
            "public": public, "groups": summary["groups"], "cameras": summary["cameras"]}


def _register_framings(ids: list[str]) -> dict[str, str]:
    """framing_id of every clip: SIFT/RANSAC registration of its t=0.5 s frame to the P6 framing references.

    Same matcher and thresholds as cloud/modal_p6.py (SIFT 3000, ratio 0.75, RANSAC 3 px, >= 60 inliers, corner
    displacement <= 12 px at 960x540). The P6 references come first, in P6's index order, so c9_f0 / c9_f1 / c14_f0 keep
    their P6 meaning; a clip that matches no reference founds a new framing (c<cam>_f<k>, k >= the P6 count).
    """
    import cv2
    import numpy as np

    sys.path.insert(0, "/root")
    import framing as fr

    sift = cv2.SIFT_create(nfeatures=3000)
    feats: dict[str, tuple] = {}
    for cid in ids:
        g = cv2.imread(f"{ESK}/frame05/{cid}.jpg", cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"no frame05 for {cid}")
        kp, d = sift.detectAndCompute(cv2.resize(g, P6_SMALL, interpolation=cv2.INTER_AREA), None)
        feats[cid] = (np.float32([k.pt for k in kp]), d)
    bf = cv2.BFMatcher()

    def match(a: str, b: str) -> tuple[int, float]:
        pa, da = feats[a]
        pb, db = feats[b]
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return 0, float("inf")
        good = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            return len(good), float("inf")
        Hm, mask = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], cv2.RANSAC, 3.0)
        if Hm is None:
            return 0, float("inf")
        return int(mask.sum()), fr.corner_displacement(Hm, *P6_SMALL)

    order = list(P6_REFS) + sorted(c for c in ids if c not in P6_REFS)
    asg = fr.assign_framings(order, lambda c: _cid_meta(c)[1], lambda c: True, match, P6_MIN_INLIERS,
                             P6_MAX_DISP_SMALL_PX)
    out = {c: v["framing_id"] for c, v in asg["clips"].items()}
    assert (out["0_te21"], out["0_te1"], out["3_te1"]) == ("c9_f0", "c9_f1", "c14_f0"), "P6 reference order changed"
    return out


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=1800)
def run_split_a1(viewed: list[str], p6_framing: dict[str, str]) -> dict:
    """Amendment A1+ (prereg/AMENDMENTS.md): framing registration of every clip + scripts/esk_split.py --rule a1plus.

    Runs ONCE (refuses if split_a1.csv exists). Returns the split rows (clip_id, group_id, split, sha256, framing_id),
    group-level counts and how many of the P6 framing ids (train+dev of the original split) it reproduces. Nothing else
    per clip leaves Modal.
    """
    import csv
    import glob
    import os
    import subprocess

    vol.reload()
    out_dir = f"{ESK}/split"
    if os.path.exists(f"{out_dir}/{SPLIT_A1}"):
        raise RuntimeError(f"{SPLIT_A1} already exists on the volume: the A1+ split runs only once")
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{ESK}/meta/*.json"))]
    if len(metas) != N_EXPECTED:
        raise RuntimeError(f"expected {N_EXPECTED} clips with features, found {len(metas)}")
    ids = [m["clip_id"] for m in metas]
    framing_of = _register_framings(ids)
    p6_same = sum(1 for c, f in p6_framing.items() if framing_of.get(c) == f)
    with open(f"{out_dir}/framing_a1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "framing_id"])
        w.writerows(sorted(framing_of.items()))
    with open(f"{out_dir}/viewed_a1.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(viewed)) + "\n")
    r = subprocess.run([sys.executable, "/root/esk_split.py", "--features", f"{ESK}/feat", "--meta",
                        f"{out_dir}/meta_in.json", "--out", f"{out_dir}/{SPLIT_A1}", "--summary",
                        f"{out_dir}/summary_a1.json", "--rule", "a1plus", "--viewed", f"{out_dir}/viewed_a1.txt",
                        "--framing", f"{out_dir}/framing_a1.csv"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    vol.commit()
    summary = json.load(open(f"{out_dir}/summary_a1.json", encoding="utf-8"))
    return {"stdout": r.stdout[-4000:], "split_csv": open(f"{out_dir}/{SPLIT_A1}", encoding="utf-8").read(),
            "groups": summary["groups"], "cameras": summary["cameras"],
            "p6_framing_reproduced": [p6_same, len(p6_framing)]}


# --------------------------------------------------------------------------- RF-DETR on the t=0.5 s frames


@app.function(image=gpu_image, gpu="L4", volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def detect_people(threshold: float = 0.25) -> dict:
    """RF-DETR-M (COCO) on every /esk/frame05 image; boxes go to /esk/det. Returns counts only."""
    import glob
    import os

    import numpy as np
    import rfdetr
    from PIL import Image

    vol.reload()
    os.makedirs(f"{ESK}/det", exist_ok=True)
    model = rfdetr.RFDETRMedium(device="cuda")
    paths = sorted(glob.glob(f"{ESK}/frame05/*.jpg"))
    n_done = 0
    for p in paths:
        cid = Path(p).stem
        d = model.predict(Image.open(p).convert("RGB"), threshold=threshold)
        names = d.data.get("class_name") if getattr(d, "data", None) else None
        cls_ids = np.asarray(d.class_id).astype(int).tolist()
        if names is None:
            names = ["person" if c == 1 else str(c) for c in cls_ids]
        rec = {"xyxy": np.asarray(d.xyxy, float).round(1).tolist(), "conf": np.asarray(d.confidence, float)
               .round(4).tolist(), "class_id": cls_ids, "class_name": [str(n) for n in names]}
        with open(f"{ESK}/det/{cid}.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        n_done += 1
    vol.commit()
    return {"frames": len(paths), "done": n_done}


# --------------------------------------------------------------------------- aggregates (P1, P2, P3 balance)


def _occluded_share(det: dict) -> tuple[int, int]:
    """(occluded person boxes, person boxes) at PERSON_CONF; occluded = >= OCC_IOA of its area under another box."""
    import numpy as np

    boxes = np.array(det["xyxy"], float).reshape(-1, 4)
    conf = np.array(det["conf"], float)
    keep = conf >= PERSON_CONF
    boxes, names = boxes[keep], [n for n, k in zip(det["class_name"], keep) if k]
    pidx = [i for i, n in enumerate(names) if n == "person"]
    occ = 0
    for i in pidx:
        a = boxes[i]
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        for j in range(len(boxes)):
            if j == i:
                continue
            b = boxes[j]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if iw * ih / area >= OCC_IOA:
                occ += 1
                break
    return occ, len(pidx)


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=1800)
def report(split_name: str = "split.csv") -> dict:
    """P1/P2/P3 aggregates per split x camera. Only aggregates are returned for TEST clips."""
    import glob
    import os
    from collections import defaultdict

    import numpy as np

    vol.reload()
    split = _read_split(split_name)
    metas = {Path(p).stem: json.load(open(p, encoding="utf-8")) for p in glob.glob(f"{ESK}/meta/*.json")}
    acc: dict = defaultdict(lambda: defaultdict(list))
    p2: dict = defaultdict(lambda: defaultdict(int))
    p2_values: dict = defaultdict(set)
    for cid, m in metas.items():
        s = split[cid]["split"]
        cam = m["camera"]
        key = f"{s}|{cam}"
        a = acc[key]
        a["duration"].append(m["duration_pts"])
        a["fps_pts"].append(m["fps_pts"])
        z = np.load(f"{ESK}/feat/{cid}.npz")
        a["luma_mean"].append(float(z["luma"][0]))
        a["luma_std"].append(float(z["luma"][1]))
        dp = f"{ESK}/det/{cid}.json"
        if os.path.exists(dp):
            det = json.load(open(dp, encoding="utf-8"))
            ppl = sum(1 for n, c in zip(det["class_name"], det["conf"]) if n == "person" and c >= PERSON_CONF)
            occ, npers = _occluded_share(det)
            a["people"].append(ppl)
            a["occ"].append(occ)
            a["persons"].append(npers)
            a["vehicles"].append(sum(1 for n, c in zip(det["class_name"], det["conf"])
                                     if n in ("car", "truck", "bus", "motorcycle") and c >= PERSON_CONF))
        pr = m["ffprobe"]
        p2[s]["clips"] += 1
        p2[s]["with_creation_time"] += int(bool(pr.get("creation_time")))
        p2[s]["with_stream_creation_time"] += int(bool(pr.get("stream_creation_times")))
        p2[s]["with_audio"] += int(pr.get("n_audio", 0) > 0)
        p2[s]["with_encoder_tag"] += int(bool(pr.get("encoder")))
        if pr.get("creation_time"):
            p2_values[s].add(pr["creation_time"])
        for k in pr.get("tag_keys", []):
            p2[s][f"tag:{k}"] += 1
        p2[s]["fps_tag_vs_pts_gt_1pct"] += int(_fps_mismatch(pr.get("avg_frame_rate"), m["fps_pts"]))
    table = {}
    for key, a in sorted(acc.items()):
        def q(v):
            v = np.array([x for x in v if x is not None], float)
            return None if v.size == 0 else {"n": int(v.size), "mean": round(float(v.mean()), 3),
                                              "p10": round(float(np.percentile(v, 10)), 3),
                                              "p50": round(float(np.percentile(v, 50)), 3),
                                              "p90": round(float(np.percentile(v, 90)), 3)}
        persons = int(np.sum(a["persons"])) if a["persons"] else 0
        table[key] = {"clips": len(a["duration"]), "clip_seconds": round(float(np.nansum(a["duration"])), 1),
                      "duration": q(a["duration"]), "fps_pts": q(a["fps_pts"]), "luma_mean": q(a["luma_mean"]),
                      "luma_std": q(a["luma_std"]), "people_per_frame": q(a["people"]),
                      "frames_with_people": int(np.sum(np.array(a["people"]) > 0)) if a["people"] else 0,
                      "person_boxes": persons, "occluded_person_boxes": int(np.sum(a["occ"])) if a["occ"] else 0,
                      "occlusion_share": round(float(np.sum(a["occ"])) / persons, 3) if persons else None,
                      "frames_with_vehicle_box": int(np.sum(np.array(a["vehicles"]) > 0)) if a["vehicles"] else 0}
    # authors' te/tr overlap: groups that hold both an authors' train and an authors' test clip
    groups: dict = defaultdict(set)
    for cid, r in split.items():
        groups[r["group_id"]].add("te" if "_te" in cid else "tr")
    mixed = [g for g, v in groups.items() if len(v) == 2]
    te_ids = [c for c in split if "_te" in c]
    te_in_mixed = sum(1 for c in te_ids if split[c]["group_id"] in mixed)
    group_counts: dict = defaultdict(int)
    seen = set()
    for cid, r in split.items():
        g = r["group_id"]
        if g not in seen:
            seen.add(g)
            group_counts[f"{r['split']}|{_cid_meta(cid)[1]}"] += 1
    out = {"table": table, "p2": {s: dict(v) for s, v in p2.items()},
           "p2_distinct_creation_times": {s: len(v) for s, v in p2_values.items()},
           "p2_creation_time_examples_train": sorted(p2_values.get("train", set()))[:5],
           "authors_overlap": {"groups": len(groups), "groups_with_both_tr_and_te": len(mixed),
                               "authors_test_clips": len(te_ids), "authors_test_clips_in_mixed_groups": te_in_mixed},
           "group_counts": dict(group_counts), "person_conf": PERSON_CONF, "occ_ioa": OCC_IOA, "split_file": split_name}
    rep = "report.json" if split_name == "split.csv" else "report_a1.json"
    with open(f"{ESK}/split/{rep}", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    vol.commit()
    return out


def _fps_mismatch(rate: str | None, fps_pts: float | None) -> bool:
    if not rate or not fps_pts or "/" not in rate:
        return False
    a, b = rate.split("/")
    if float(b) == 0:
        return False
    return abs(float(a) / float(b) - fps_pts) / fps_pts > 0.01


# --------------------------------------------------------------------------- purge + check


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=1, memory=2048, timeout=900)
def purge_test(split_name: str = "split.csv") -> dict:
    """Delete meta/feat (pHash, median background, luma)/frame05/det/clips/keyframes of every TEST clip.

    Keeps sha256 + camera/class counts in test_manifest[_a1].json (and the §2 inventory columns in public.json).
    """
    import os
    from collections import Counter

    vol.reload()
    split = _read_split(split_name)
    test = sorted(c for c, r in split.items() if r["split"] == "test")
    manifest = {"split_file": split_name, "n": len(test), "sha256": sorted(split[c]["sha256"] for c in test),
                "by_camera": dict(Counter(str(_cid_meta(c)[1]) for c in test)),
                "by_class": dict(Counter(str(_cid_meta(c)[0]) for c in test))}
    name = "test_manifest.json" if split_name == "split.csv" else "test_manifest_a1.json"
    with open(f"{ESK}/split/{name}", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    removed = 0
    for c in test:
        for d, ext in (("meta", "json"), ("feat", "npz"), ("frame05", "jpg"), ("det", "json"),
                       ("clips", "mp4"), ("keyframes", "k1.jpg"), ("keyframes", "k2.jpg")):
            p = f"{ESK}/{d}/{c}.{ext}" if d != "keyframes" else f"{ESK}/{d}/{c}_{ext}"
            if os.path.exists(p):
                os.remove(p)
                removed += 1
    # public.json held test durations/sha256 only (§2 inventory columns), it is kept; summary.json holds group sizes
    vol.commit()
    return {"test_clips": len(test), "files_removed": removed, "by_camera": manifest["by_camera"],
            "by_class": manifest["by_class"]}


def _stem_ids(name: str) -> set[str]:
    import re

    stem = name.rsplit(".", 1)[0]
    return {stem, stem.rsplit("_k", 1)[0]} | set(re.findall(r"(?<![0-9A-Za-z])\d_t[re]\d+(?![0-9])", name))


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=1, memory=2048, timeout=900)
def check_no_test_images(split_name: str = "split.csv") -> dict:
    """Assert that no file anywhere on the volume (except the id/hash tables in esk/split/) belongs to a TEST clip."""
    import os

    vol.reload()
    split = _read_split(split_name)
    test = {c for c, r in split.items() if r["split"] == "test"}
    bad, n_files = [], 0
    for dp, _, files in os.walk(ESK):
        if os.path.normpath(dp).startswith(os.path.normpath(f"{ESK}/split")):
            continue
        for f in files:
            n_files += 1
            if _stem_ids(f) & test:
                bad.append(f"{dp}/{f}")
    if bad:
        raise AssertionError(f"{len(bad)} TEST files on the volume")
    return {"ok": True, "files_checked": n_files, "test_ids": len(test), "split_file": split_name}


def check_local_no_test(test: set[str], roots: list[Path]) -> dict:
    """Desktop half of the check: no file under ``roots`` is named after a TEST clip, and no image-derived per-clip
    table in the data root (p5/detections.json, keyframes/*.json) carries a TEST clip id. inventory.parquet and the
    framing table hold only the §2 columns of test clips and are allowed."""
    bad: list[str] = []
    n = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            n += 1
            if _stem_ids(p.name) & test:
                bad.append(str(p))
            if p.suffix == ".json" and p.parent.name in ("p5", "keyframes"):
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, list):
                    ids = {r.get("clip_id") if isinstance(r, dict) else str(r) for r in d}
                else:
                    ids = set(d) if isinstance(d, dict) else set()
                if ids & test:
                    bad.append(f"{p} (content)")
    if bad:
        raise AssertionError(f"{len(bad)} local TEST traces (names withheld; rerun with care)")
    return {"ok": True, "local_files_checked": n, "roots": [str(r) for r in roots]}


# --------------------------------------------------------------------------- Pass 2 (train + dev only)


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=1, memory=2048, timeout=900, max_containers=16, retries=2)
def pass2_clip(cid: str, expected_sha: str, expected_split: str, split_name: str = "split.csv") -> dict:
    """Fetch one TRAIN/DEV clip, verify its sha256, store it and its 2 keyframes (960 px) on the volume."""
    import os

    import av
    import cv2

    if expected_split not in ("train", "dev"):
        raise ValueError(f"pass2 refuses split={expected_split!r} for {cid}")
    split = _read_split(split_name)
    if split[cid]["split"] != expected_split or split[cid]["split"] not in ("train", "dev"):
        raise ValueError(f"{cid} is not train/dev on the volume split")
    for d in ("clips", "keyframes"):
        os.makedirs(f"{ESK}/{d}", exist_ok=True)
    tmp = f"/tmp/{cid}.mp4"
    _download(cid, tmp)
    sha = _sha256(tmp)
    if sha != expected_sha:
        raise ValueError(f"sha256 mismatch for {cid}")
    with av.open(tmp) as c:
        st = c.streams.video[0]
        frames = []
        for fr in c.decode(st):
            frames.append(float(fr.pts * st.time_base))
        t0, t1 = frames[0], frames[-1]
    targets = [t0 + (t1 - t0) / 3.0, t0 + 2.0 * (t1 - t0) / 3.0]
    k = 0
    with av.open(tmp) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            if k < 2 and float(fr.pts * st.time_base) >= targets[k]:
                rgb = fr.to_ndarray(format="rgb24")
                h = round(rgb.shape[0] * KEYFRAME_W / rgb.shape[1])
                img = cv2.resize(rgb, (KEYFRAME_W, h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{ESK}/keyframes/{cid}_k{k + 1}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                k += 1
    import shutil

    shutil.copyfile(tmp, f"{ESK}/clips/{cid}.mp4")
    os.remove(tmp)
    vol.commit()
    return {"clip_id": cid, "keyframes": k}


# --------------------------------------------------------------------------- P7: cross-camera overlap

p7_image = gpu_image.pip_install("kornia==0.8.1", "opencv-python-headless==4.12.0.88")


@app.function(image=p7_image, gpu="L4", volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def p7_overlap(pairs: list[list[str]], n_features: int = 2048) -> list[dict]:
    """DISK + LightGlue (kornia) and SIFT + ratio 0.75 between /esk/frame05 images; RANSAC fundamental-matrix inliers.

    Returns numbers only (matches and inliers per pair), never images.
    """
    import cv2
    import kornia.feature as KF
    import numpy as np
    import torch

    vol.reload()
    dev = torch.device("cuda")
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlueMatcher("disk").eval().to(dev)
    sift = cv2.SIFT_create(nfeatures=4000)

    def load(cid: str):
        bgr = cv2.imread(f"{ESK}/frame05/{cid}.jpg")
        bgr = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
        return bgr

    def inliers(p1, p2) -> int:
        if len(p1) < 8:
            return 0
        _, mask = cv2.findFundamentalMat(np.float32(p1), np.float32(p2), cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
        return 0 if mask is None else int(mask.sum())

    out = []
    for a, b in pairs:
        ia, ib = load(a), load(b)
        # LightGlue
        ta = torch.from_numpy(cv2.cvtColor(ia, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(dev)
        tb = torch.from_numpy(cv2.cvtColor(ib, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(dev)
        with torch.inference_mode():
            fa, fb = disk(torch.cat([ta, tb]), n_features, pad_if_not_divisible=True)
            ka, kb = fa.keypoints, fb.keypoints
            la = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=dev))
            lb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=dev))
            hw = torch.tensor(ta.shape[2:], device=dev)
            _, idx = lg(fa.descriptors, fb.descriptors, la, lb, hw1=hw, hw2=hw)
        idx = idx.cpu().numpy()
        p1, p2 = ka.cpu().numpy()[idx[:, 0]], kb.cpu().numpy()[idx[:, 1]]
        lg_matches, lg_inl = int(len(idx)), inliers(p1, p2)
        # SIFT + ratio test
        ga, gb = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
        k1, d1 = sift.detectAndCompute(ga, None)
        k2, d2 = sift.detectAndCompute(gb, None)
        good = []
        if d1 is not None and d2 is not None and len(k2) >= 2:
            for m in cv2.BFMatcher().knnMatch(d1, d2, k=2):
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    good.append(m[0])
        s1 = [k1[m.queryIdx].pt for m in good]
        s2 = [k2[m.trainIdx].pt for m in good]
        out.append({"a_cam": _cid_meta(a)[1], "b_cam": _cid_meta(b)[1], "lg_matches": lg_matches,
                    "lg_inliers": lg_inl, "sift_ratio_matches": len(good), "sift_inliers": inliers(s1, s2)})
    return out


# --------------------------------------------------------------------------- local entrypoint


def _all_ids() -> list[str]:
    from huggingface_hub import HfApi  # only used locally, listing (no media)

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=REV)
    ids = sorted(f[len("data/"):-len(".mp4")] for f in files if f.startswith("data/") and f.endswith(".mp4"))
    assert len(ids) == N_EXPECTED, len(ids)
    return ids


def _local_roots(scratch: str) -> list[Path]:
    import os

    roots = [Path(os.environ["INTERCYCLE_DATA"])] if os.environ.get("INTERCYCLE_DATA") else []
    return roots + [Path(s) for s in scratch.split(";") if s]


def _local_split(split_name: str) -> list[dict]:
    import csv

    with open(ROOT / "prereg" / split_name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@app.local_entrypoint()
def main(action: str = "pass1", ids_file: str = "", out: str = "", split_name: str = "split.csv",
         scratch: str = ""):
    import csv
    import time

    t0 = time.time()
    if action == "split_a1":
        import os

        viewed = [ln.strip() for ln in (ROOT / "prereg" / "viewed_clips.txt").read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")]
        p6 = {}
        p6_csv = Path(os.environ["INTERCYCLE_DATA"]) / "p6_framing.csv"
        with open(p6_csv, newline="", encoding="utf-8") as fh:
            p6 = {r["clip_id"]: r["framing_id"] for r in csv.DictReader(fh)}
        res = run_split_a1.remote(viewed, p6)
        (ROOT / "prereg" / SPLIT_A1).write_text(res["split_csv"], encoding="utf-8", newline="\n")
        Path(out).write_text(json.dumps({k: v for k, v in res.items() if k != "split_csv"}, indent=1),
                             encoding="utf-8")
        print(res["stdout"])
        print(json.dumps({"p6_framing_reproduced": res["p6_framing_reproduced"]}))
        return
    if action in ("report", "purge_test", "check", "pass2") and split_name not in ("split.csv", SPLIT_A1):
        raise SystemExit(f"unknown split file {split_name}")
    if action == "pass1":
        ids = Path(ids_file).read_text(encoding="utf-8").split() if ids_file else _all_ids()
        ok, fail = 0, []
        for r in pass1_clip.map(ids, return_exceptions=True):
            if isinstance(r, dict) and r.get("ok"):
                ok += 1
            else:
                fail.append(repr(r)[:200])
        print(json.dumps({"ok": ok, "failed": len(fail), "fail_examples": fail[:5], "wall_s": round(time.time() - t0)}))
    elif action == "split":
        res = run_split.remote()
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(res["stdout"])
    elif action == "detect":
        print(json.dumps(detect_people.remote()))
    elif action == "report":
        res = report.remote(split_name)
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res["group_counts"]))
    elif action == "p7":
        import random

        with open(ROOT / "prereg" / "split.csv", newline="", encoding="utf-8") as fh:
            train = [r["clip_id"] for r in csv.DictReader(fh) if r["split"] == "train"]
        rnd = random.Random(7)
        c9 = rnd.sample(sorted(c for c in train if _cid_meta(c)[1] == 9), 8)
        c14 = rnd.sample(sorted(c for c in train if _cid_meta(c)[1] == 14), 8)
        pairs = [[a, b] for a in c9 for b in c14]
        pairs += [[c9[i], c9[i + 1]] for i in range(7)] + [[c14[i], c14[i + 1]] for i in range(7)]
        res = p7_overlap.remote(pairs)
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        for kind in ((9, 14), (9, 9), (14, 14)):
            sel = [r for r in res if (r["a_cam"], r["b_cam"]) == kind]
            print(kind, len(sel), {k: [min(r[k] for r in sel), sorted(r[k] for r in sel)[len(sel) // 2],
                                       max(r[k] for r in sel)] for k in ("lg_inliers", "sift_inliers")})
    elif action == "purge_test":
        print(json.dumps(purge_test.remote(split_name)))
    elif action == "check":
        print(json.dumps(check_no_test_images.remote(split_name)))
        test = {r["clip_id"] for r in _local_split(split_name) if r["split"] == "test"}
        print(json.dumps(check_local_no_test(test, _local_roots(scratch))))
    elif action == "pass2":
        rows = [r for r in _local_split(split_name) if r["split"] in ("train", "dev")]
        args = [(r["clip_id"], r["sha256"], r["split"], split_name) for r in rows]
        ok, fail = 0, []
        for r in pass2_clip.starmap(args, return_exceptions=True):
            if isinstance(r, dict) and r.get("keyframes") == 2:
                ok += 1
            else:
                fail.append(repr(r)[:200])
        print(json.dumps({"ok": ok, "failed": len(fail), "fail_examples": fail[:5], "wall_s": round(time.time() - t0)}))
    else:
        raise SystemExit(f"unknown action {action}")
