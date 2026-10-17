"""Probe P6 (lane S3a, PLAN_TWIN §3): framings, CAL v0 k1 per framing and a person-height scale check, on Modal.

    modal run cloud/modal_p6.py --out <scratch>/p6.json

Inputs on the shared volume ``intercycle-esk`` (written by cloud/modal_esk.py): /esk/frame05/<clip>.jpg (full-res frame
at t = 0.5 s) and /esk/det/<clip>.json (RF-DETR-M COCO boxes on that frame). Only TRAIN and DEV clip ids of
prereg/split.csv are passed in; TEST ids are never read. Only numbers come back. Two derived images per framing (a
median background of train frames, which removes most people, and the chain overlay) stay on the volume under
/esk/p6/ for a later visual check.

Licences: OpenCV Apache-2.0, numpy/scipy BSD.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "s3a-p6-esk"
VOL = "/vol"
ESK = f"{VOL}/esk"
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
SMALL = (960, 540)
MIN_INLIERS = 60
MAX_DISP_SMALL_PX = 12.0  # corner displacement at 960x540 (= 24 px at full res) to call two frames one framing
N_MEDIAN = 25
N_SINGLE = 6
PERSON_CONF = 0.5
SEED = 20261017

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("intercycle-esk")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "scipy==1.16.2", "opencv-python-headless==4.12.0.88")
    .add_local_file(str(ROOT / "intercycle" / "framing.py"), "/root/framing.py")
)


def _cam(cid: str) -> int:
    return CAMERA_OF_PREFIX[int(cid.split("_")[0])]


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3600)
def p6(train_ids: list[str], dev_ids: list[str], k1_override: dict | None = None) -> dict:
    import math
    import os
    import random

    import cv2
    import numpy as np

    import framing as fr

    vol.reload()
    allowed = set(train_ids) | set(dev_ids)
    split = {c: "train" for c in train_ids} | {c: "dev" for c in dev_ids}
    sift = cv2.SIFT_create(nfeatures=3000)
    feats: dict[str, tuple] = {}

    def load_small(cid: str):
        assert cid in allowed
        g = cv2.imread(f"{ESK}/frame05/{cid}.jpg", cv2.IMREAD_GRAYSCALE)
        return cv2.resize(g, SMALL, interpolation=cv2.INTER_AREA)

    for cid in sorted(allowed):
        if not os.path.exists(f"{ESK}/frame05/{cid}.jpg"):
            continue
        kp, d = sift.detectAndCompute(load_small(cid), None)
        feats[cid] = (np.float32([k.pt for k in kp]), d)
    bf = cv2.BFMatcher()
    hom: dict[tuple[str, str], np.ndarray] = {}

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
        hom[(a, b)] = Hm
        return int(mask.sum()), fr.corner_displacement(Hm, *SMALL)

    ids = sorted(c for c in train_ids if c in feats) + sorted(c for c in dev_ids if c in feats)
    # train clips come first, so train framings get the low indices; a dev clip that matches no train framing founds a
    # dev-only framing (registration only: no k1, no CAL, n_train = 0), so dev clips are grouped too
    asg = fr.assign_framings(ids, _cam, lambda c: True, match, MIN_INLIERS, MAX_DISP_SMALL_PX)
    clips = asg["clips"]
    # diagnostic: displacement / inliers of every clip against its camera's first reference
    diag = {}
    for cam, refs in asg["refs"].items():
        r0 = refs[0]
        vals = [match(c, r0) for c in ids if _cam(c) == int(cam) and c != r0]
        d = np.array([v[1] for v in vals], float)
        n = np.array([v[0] for v in vals], float)
        diag[cam] = {"n": len(vals), "disp_small_px_p50_p90_max": [float(np.nanpercentile(d[np.isfinite(d)], q))
                                                                     for q in (50, 90, 100)] if np.isfinite(d).any() else None,
                     "inliers_min_p10_p50": [float(np.percentile(n, q)) for q in (0, 10, 50)],
                     "n_disp_gt_thr": int((~np.isfinite(d) | (d > MAX_DISP_SMALL_PX)).sum())}

    os.makedirs(f"{ESK}/p6", exist_ok=True)
    S = np.diag([2.0, 2.0, 1.0])
    Si = np.linalg.inv(S)
    rnd = random.Random(SEED)
    per_framing = {}
    fids = sorted({v["framing_id"] for v in clips.values() if not v["framing_id"].endswith("unmatched")})
    for fid in fids:
        members = [c for c, v in clips.items() if v["framing_id"] == fid]
        tr = sorted(c for c in members if split[c] == "train")
        ref = clips[tr[0]]["ref"] if tr else None
        rec: dict = {"n_train": len(tr), "n_dev": len(members) - len(tr), "ref": ref}
        if not tr:
            per_framing[fid] = rec
            continue

        def full_warp(c: str):
            img = cv2.imread(f"{ESK}/frame05/{c}.jpg")
            if c == ref:
                return img
            Hm = hom.get((c, ref))
            if Hm is None:
                match(c, ref)
                Hm = hom.get((c, ref))
            return cv2.warpPerspective(img, S @ Hm @ Si, (fr.W, fr.H_IMG), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)

        pick = sorted(rnd.sample(tr, min(N_MEDIAN, len(tr))))
        stack = np.stack([full_warp(c) for c in pick])
        bg = np.median(stack, axis=0).astype(np.uint8)
        del stack
        cv2.imwrite(f"{ESK}/p6/{fid}_bg.jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        chains = fr.edge_chains(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
        rec["median_of"] = pick
        rec["n_chains"] = len(chains)
        if len(chains) >= 2:
            rob = fr.fit_k1_robust(chains)
            rec["k1"] = {
                "top3_k1_px": rob["top_fit"]["k1"], "top3_k1_norm": rob["top_fit"]["k1_norm"],
                "top3_at_bound": rob["top_fit"]["at_bound"],
                "top3_rms_before_after_px": [rob["top_fit"]["rms_before_px"], rob["top_fit"]["rms_after_px"]],
                "all_k1_norm": rob["all_fit"]["k1_norm"], "n_consistent": len(rob["consistent"]),
                "loo_k1_norm_min_max": [min(rob["loo_k1_norm"]), max(rob["loo_k1_norm"])] if rob["loo_k1_norm"] else None,
                "top_chains": [{"len_px": fr.chain_length(chains[i]),
                                "mid_xy": [float(x) for x in chains[i].mean(0)],
                                "end_xy": [[float(v) for v in chains[i][np.argmin(chains[i][:, 0] + chains[i][:, 1])]],
                                           [float(v) for v in chains[i][np.argmax(chains[i][:, 0] + chains[i][:, 1])]]]}
                               for i in rob["top"]],
            }
            ov = bg.copy()
            for j, i in enumerate(rob["consistent"]):
                col = (0, 0, 255) if i in rob["top"] else (0, 255, 255)
                for x, y in chains[i].astype(int):
                    cv2.circle(ov, (int(x), int(y)), 2, col, -1)
            cv2.imwrite(f"{ESK}/p6/{fid}_chains.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # spread: k1 from single train frames (people present), same robust rule
            singles = []
            for c in sorted(rnd.sample(tr, min(N_SINGLE, len(tr)))):
                g = cv2.cvtColor(full_warp(c), cv2.COLOR_BGR2GRAY)
                ch = fr.edge_chains(g)
                if len(ch) >= 2:
                    singles.append(fr.fit_k1_robust(ch)["top_fit"]["k1_norm"])
            rec["k1"]["single_frame_top3_k1_norm"] = singles
            k1 = rob["top_fit"]["k1"]
        else:
            k1 = 0.0
        if k1_override and fid in k1_override:  # CAL v0 k1 from the hinted lines (scripts/esk_p6_k1.py)
            k1 = float(k1_override[fid])
        rec["k1_used_for_scale"] = k1
        rec["opencv_equiv_f1330"] = fr.opencv_coeffs(k1)

        # scale: person boxes on the train frames of this framing (det boxes are on the clip's own frame05)
        feet, heads, owners = [], [], []
        for c in tr:
            p = f"{ESK}/det/{c}.json"
            if not os.path.exists(p):
                continue
            det = json.load(open(p, encoding="utf-8"))
            b = np.array(det["xyxy"], float).reshape(-1, 4)
            conf = np.array(det["conf"], float)
            person = np.array([n == "person" for n in det["class_name"]], dtype=bool) & (conf >= PERSON_CONF)
            others = b[conf >= PERSON_CONF]
            for bb in b[person]:
                x0, y0, x1, y1 = bb
                if y1 - y0 < 40 or x0 < 5 or y0 < 5 or x1 > fr.W - 6 or y1 > fr.H_IMG - 6:
                    continue
                area = (x1 - x0) * (y1 - y0)
                occl = False
                for ob in others:
                    if np.allclose(ob, bb):
                        continue
                    iw = max(0.0, min(x1, ob[2]) - max(x0, ob[0]))
                    ih = max(0.0, min(y1, ob[3]) - max(y0, ob[1]))
                    if iw * ih / area >= 0.10:
                        occl = True
                        break
                if occl or (y1 - y0) < 1.2 * (x1 - x0):  # occluded, or not upright (crouching / sitting / lying)
                    continue
                feet.append([(x0 + x1) / 2, y1])
                heads.append([(x0 + x1) / 2, y0])
                owners.append(c)
        sc: dict = {"n_person_boxes": len(feet), "n_clips": len(set(owners))}
        if len(feet) >= 15:
            F = np.array(feet)
            Hh = np.array(heads)
            fu, hu = fr.undistort(F, k1), fr.undistort(Hh, k1)
            for f in (fr.F_RANGE[0], fr.F_PRIOR, fr.F_RANGE[1]):
                sc[f"tilt_fit_f{int(f)}"] = fr.fit_height_tilt(fu, hu[:, 1], f)
            sc["tilt_fit_f1330_no_k1"] = fr.fit_height_tilt(F, Hh[:, 1], fr.F_PRIOR)
            sc["linear_horizon_fit"] = fr.fit_height_linear(fu[:, 1], fu[:, 1] - hu[:, 1])
            # clip-bootstrap of h at f = 1330
            uo = sorted(set(owners))
            idx = {c: [i for i, o in enumerate(owners) if o == c] for c in uo}
            br = np.random.default_rng(SEED)
            hs = []
            for _ in range(100):
                sel = np.concatenate([idx[uo[j]] for j in br.integers(0, len(uo), len(uo))])
                hs.append(fr.fit_height_tilt(fu[sel], hu[sel, 1], fr.F_PRIOR)["h_m"])
            sc["h_m_boot_p5_p50_p95_f1330"] = [float(np.percentile(hs, q)) for q in (5, 50, 95)]
            th = math.radians(sc["tilt_fit_f1330"]["tilt_deg"])
            h_fit = sc["tilt_fit_f1330"]["h_m"]
            sc["m_per_px_at_rows_f1330_hfit"] = {str(v): fr.ground_m_per_px(v, fr.F_PRIOR, h_fit, th)
                                                 for v in (600, 800, 1000)}
        rec["scale_people"] = sc
        per_framing[fid] = rec
    vol.commit()
    counts = {}
    for c, v in clips.items():
        counts.setdefault(v["framing_id"], {"train": 0, "dev": 0})[split[c]] += 1
    return {"framing_of": {c: v["framing_id"] for c, v in clips.items()},
            "match_of": {c: [v["inliers"], v["disp_px"]] for c, v in clips.items()},
            "refs": asg["refs"], "counts": counts, "diag_vs_first_ref": diag, "per_framing": per_framing,
            "missing_frame05": sorted(allowed - set(feats)),
            "params": {"min_inliers": MIN_INLIERS, "max_disp_small_px": MAX_DISP_SMALL_PX, "n_median": N_MEDIAN,
                       "person_conf": PERSON_CONF, "seed": SEED}}


@app.function(image=image, volumes={VOL: vol}, cpu=1, memory=2048, timeout=600)
def fetch_p6_images(names: list[str]) -> dict[str, bytes]:
    """Derived P6 images (median backgrounds / overlays of TRAIN frames) from /esk/p6, for a visual check."""
    vol.reload()
    out = {}
    for n in names:
        assert "/" not in n and n.endswith(".jpg")
        out[n] = Path(f"{ESK}/p6/{n}").read_bytes()
    return out


@app.local_entrypoint()
def main(out: str, fetch: str = "", fetch_dir: str = ""):
    if fetch:
        imgs = fetch_p6_images.remote(fetch.split(","))
        for n, b in imgs.items():
            Path(fetch_dir, n).write_bytes(b)
        print("fetched", sorted(imgs))
        return
    with open(ROOT / "prereg" / "split.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    train = [r["clip_id"] for r in rows if r["split"] == "train"]
    dev = [r["clip_id"] for r in rows if r["split"] == "dev"]
    k1_file = ROOT / "twin" / "cal" / "k1_v0.json"
    k1_override = None
    if k1_file.exists():
        k1_v0 = json.loads(k1_file.read_text(encoding="utf-8"))
        k1_override = {k: v["k1"] for k, v in k1_v0.items() if isinstance(v, dict) and v.get("k1") is not None}
    res = p6.remote(train, dev, k1_override)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"counts": res["counts"], "refs": res["refs"], "diag": res["diag_vs_first_ref"]}, indent=1))
