"""Prepare the CAL v0 click images (lane S2): one people-masked TRAIN keyframe per framing, plus an undistorted copy.

Two steps (GPU env, torch + rfdetr + cv2):

    # 1. rank every TRAIN keyframe of each framing (persons, sharpness); writes ranking.json and masked contact sheets
    F:/envs/intercycle-gpu/Scripts/python.exe scripts/cal_click_prep.py rank --out <scratch dir>

    # 2. export the chosen keyframe of each framing (masked, full resolution) and its undistorted copy
    F:/envs/intercycle-gpu/Scripts/python.exe scripts/cal_click_prep.py export \
        --pick c9_f0=2_tr26_k2 --pick c9_f1=4_tr31_k1 --pick c14_f0=7_tr19_k2 --out F:/intercycle/out/cal_click

    # 3. guide images and README.txt from twin/cal/cal_click_plan.json
    python scripts/cal_click_prep.py guide --out F:/intercycle/out/cal_click

Rules:
* Only rows with ``split == train`` in the active split (prereg/ACTIVE_SPLIT) are read. TEST is never loaded.
* Every person box at conf >= 0.1 is black-boxed, padded 15 % (intercycle.perception.privacy_mask), BEFORE anything is
  written. Unmasked pixels never leave memory. The contact sheets and the exported images are masked.
* The undistorted copy applies the CAL v0 division-model k1 of ``twin/cal/k1_v0.json`` (same 1920x1080 canvas, the
  distortion centre stays at the image centre, so the pixels of the outer ring of the raw frame fall outside it).
  A framing without its own k1 (c9_f1 has no train clip in the original split) borrows the k1 of another framing of
  the same camera and says so in ``cal_click_manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle import esk, framing  # noqa: E402
from intercycle.perception.privacy_mask import MaskConfig, TrackHoldMasker, apply_mask  # noqa: E402

K1_JSON = ROOT / "twin" / "cal" / "k1_v0.json"
PLAN_JSON = ROOT / "twin" / "cal" / "cal_click_plan.json"
LABELLER = ROOT / "tools" / "labeller" / "index.html"
BORROW_K1_FROM = {"c9_f1": "c9_f0"}  # same physical camera 9 (fixed 4.0 mm lens); no k1 of its own yet


def train_keyframes(root: Path) -> list[tuple[str, str, Path]]:
    """(framing_id, keyframe name, path) for every TRAIN keyframe of the active split."""
    inv = esk.load("train", root=root)
    assert set(inv["split"]) == {"train"}
    out = []
    for cid, fid in zip(inv["clip_id"], inv["framing_id"]):
        for k in (1, 2):
            p = root / "keyframes" / f"{cid}_k{k}.jpg"
            if p.exists():
                out.append((str(fid), f"{cid}_k{k}", p))
    return out


def k1_for(framing_id: str, k1_table: dict) -> tuple[float, str]:
    """(k1 in px^-2, source framing). Borrowed k1 is reported by the second element."""
    src = framing_id if framing_id in k1_table else BORROW_K1_FROM[framing_id]
    return float(k1_table[src]["k1"]), src


def undistort_image(img: np.ndarray, k1: float) -> np.ndarray:
    """Resample ``img`` so that straight world lines become straight (division model, same canvas size)."""
    import cv2

    h, w = img.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts_u = np.stack([uu.ravel(), vv.ravel()], 1)
    pts_d = framing.distort(pts_u, k1)  # where each undistorted pixel sits in the raw frame
    mx = pts_d[:, 0].reshape(h, w).astype(np.float32)
    my = pts_d[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def full_res_keyframe(clip: Path, k: int) -> np.ndarray:
    """Full 1920x1080 BGR frame that the 960 px keyframe ``k`` (1 or 2) was cut from (cloud/modal_esk.py pass2_clip:
    the first frame with PTS >= t0 + k (t1 - t0) / 3, PTS from the container)."""
    import av

    with av.open(str(clip)) as c:
        st = c.streams.video[0]
        pts = [float(f.pts * st.time_base) for f in c.decode(st)]
    t0, t1 = pts[0], pts[-1]
    target = t0 + k * (t1 - t0) / 3.0
    with av.open(str(clip)) as c:
        st = c.streams.video[0]
        for f in c.decode(st):
            if float(f.pts * st.time_base) >= target:
                return f.to_ndarray(format="bgr24")
    raise RuntimeError(f"no frame at {target} in {clip}")


def _detector(size: str, conf: float):
    from intercycle.perception.detector import PERSON, RFDetrDetector

    return RFDetrDetector(size=size, threshold=conf, keep=(PERSON,))


def _masked(bgr: np.ndarray, det, cfg: MaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """(masked BGR image, padded boxes)."""
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    d = det.detect([rgb])[0]
    boxes = TrackHoldMasker(cfg, width=rgb.shape[1], height=rgb.shape[0]).update(0.0, d.xyxy, d.score)
    return apply_mask(bgr, boxes), boxes


def sharpness(masked_bgr: np.ndarray, boxes: np.ndarray) -> float:
    """Variance of the Laplacian at 960x540 over the pixels outside the black boxes."""
    import cv2

    g = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (960, 540), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    keep = np.ones(g.shape, bool)
    for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4) / 2.0:
        keep[max(0, int(y0) - 3):int(y1) + 4, max(0, int(x0) - 3):int(x1) + 4] = False
    return float(lap[keep].var()) if keep.any() else 0.0


def cmd_rank(a) -> int:
    import cv2

    root = esk.data_root(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = MaskConfig()
    det = _detector(a.size, cfg.conf)
    rows: dict[str, list[dict]] = {}
    for fid, name, path in train_keyframes(root):
        bgr = cv2.imread(str(path))
        m, boxes = _masked(bgr, det, cfg)
        area = 0.0
        if len(boxes):
            cover = np.zeros(m.shape[:2], bool)
            for x0, y0, x1, y1 in boxes:
                cover[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
            area = float(cover.mean())
        rows.setdefault(fid, []).append({"name": name, "n_masked": int(len(boxes)), "masked_frac": round(area, 4),
                                         "sharp": round(sharpness(m, boxes), 2)})
        if len(rows[fid]) % 50 == 0:
            print(fid, len(rows[fid]), flush=True)
    sheets = {}
    for fid, lst in rows.items():
        lst.sort(key=lambda r: (r["n_masked"] > 0, r["masked_frac"] > 0.02, -r["sharp"]))
        top = lst[: a.top]
        tiles = []
        for r in top:
            bgr = cv2.imread(str(root / "keyframes" / f"{r['name']}.jpg"))
            m, _ = _masked(bgr, det, cfg)
            t = cv2.resize(m, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{r['name']} n={r['n_masked']} s={r['sharp']:.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        sp = out / f"sheet_{fid}.jpg"
        cv2.imwrite(str(sp), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheets[fid] = str(sp)
    (out / "ranking.json").write_text(json.dumps({"rows": rows, "sheets": sheets}, indent=1), encoding="utf-8")
    print("framings:", {k: len(v) for k, v in rows.items()}, "sheets:", sheets)
    return 0


def cmd_export(a) -> int:
    import cv2

    root = esk.data_root(a.root)
    out = Path(a.out)
    (out / "click_these").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    cfg = MaskConfig()
    det = _detector(a.size, cfg.conf)
    k1_table = json.loads(K1_JSON.read_text(encoding="utf-8"))
    allowed = {name: fid for fid, name, _ in train_keyframes(root)}  # TRAIN keyframes only
    manifest = {}
    for item in a.pick:
        fid, name = item.split("=")
        if allowed.get(name) != fid:
            raise SystemExit(f"{name} is not a TRAIN keyframe of framing {fid}")
        cid, k = name.rsplit("_k", 1)
        bgr = full_res_keyframe(esk.clip_path(cid, root=root), int(k))
        small = cv2.imread(str(root / "keyframes" / f"{name}.jpg"))
        diff = float(np.abs(cv2.resize(bgr, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA).astype(float)
                            - small).mean())
        if diff > 4.0:  # JPEG noise is about 1; a different frame is far above this
            raise SystemExit(f"full-res frame of {name} does not match its 960 px keyframe (mean abs diff {diff:.1f})")
        m, boxes = _masked(bgr, det, cfg)
        k1, k1_src = k1_for(fid, k1_table)
        raw_p = out / "click_these" / f"{fid}.jpg"  # the file name is the default framing_id in the labeller
        und_p = out / "reference" / f"{fid}_undistorted.jpg"
        cv2.imwrite(str(raw_p), m, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # resampling can make a person-like patch appear that the raw pass did not flag: mask the copy again (same rule)
        und, und_boxes = _masked(undistort_image(m, k1), det, cfg)
        cv2.imwrite(str(und_p), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest[fid] = {"keyframe": name, "raw": f"click_these/{raw_p.name}",
                         "undistorted": f"reference/{und_p.name}", "frame_diff_vs_keyframe": round(diff, 2),
                         "n_person_boxes_masked": int(len(boxes)),
                         "n_extra_boxes_masked_on_undistorted_copy": int(len(und_boxes)), "k1": k1, "k1_norm": k1 * 1330.0**2,
                         "k1_source_framing": k1_src, "k1_borrowed": k1_src != fid,
                         "image_size": [int(m.shape[1]), int(m.shape[0])]}
    (out / "cal_click_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


def draw_guides(out: Path, plan: dict) -> None:
    """reference/<framing>_GUIDE.jpg (numbered markers on the masked raw frame) and a zoomed crop of the click area."""
    import cv2

    for fid, fr in plan["framings"].items():
        img = cv2.imread(str(out / "click_these" / f"{fid}.jpg"))
        pts = [(p["id"], p["uv"]) for p in fr["points"]] + [(p["id"], p["uv"]) for p in fr.get("extra", [])]
        vis = img.copy()
        for pid, (u, v) in pts:
            cv2.circle(vis, (int(u), int(v)), 16, (0, 0, 0), 5)
            cv2.circle(vis, (int(u), int(v)), 16, (0, 255, 255), 2)
            cv2.putText(vis, pid, (int(u) + 20, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(vis, pid, (int(u) + 20, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(out / "reference" / f"{fid}_GUIDE.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        us = [p[1][0] for p in pts]
        vs = [p[1][1] for p in pts]
        x0, x1 = max(0, int(min(us)) - 120), min(img.shape[1], int(max(us)) + 120)
        y0, y1 = max(0, int(min(vs)) - 120), min(img.shape[0], int(max(vs)) + 120)
        cv2.imwrite(str(out / "reference" / f"{fid}_GUIDE_zoom.jpg"), vis[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_readme(out: Path, plan: dict, manifest: dict) -> None:
    """README.txt in plain words, and a copy of the labeller page next to it (one static file, no network)."""
    import shutil

    (out / "labeller").mkdir(exist_ok=True)
    shutil.copyfile(LABELLER, out / "labeller" / "index.html")
    o = str(out).replace("/", "\\")
    L = [
        "CAL v0 CLICKS - what to do (about 10 minutes)",
        "",
        "1. Open this page in Chrome or Edge (double-click it, or paste the address):",
        f"   {o}\\labeller\\index.html",
        "   (It is a copy of tools/labeller/index.html of the repo. It needs no internet and sends nothing anywhere.)",
        "2. Press the key 1, or the button '1 CAL points'.",
        "3. Click 'Open images' and select all files in this folder:",
        f"   {o}\\click_these",
        f"   You get {len(manifest)} pictures: " + ", ".join(f"{f}.jpg" for f in manifest) + ".",
        "   Each file name is the framing name. Leave the 'framing_id' box as it is.",
        "   The picture is the RAW camera frame (the lens curves straight lines a little). Click on THIS picture, not on the",
        "   undistorted copy: the pipeline straightens your clicks itself.",
        "   People in the pictures are covered by black boxes (privacy rule). The black boxes are not floor: do not click them.",
        "4. For each picture, click the floor points listed below, in the order p1, p2, p3, p4. Left click = new point,",
        "   drag a point to move it, mouse wheel = zoom, right-drag = pan, Ctrl+Z = undo.",
        f"   Each picture has a numbered guide: {o}\\reference\\<framing>_GUIDE.jpg (whole picture) and",
        "   <framing>_GUIDE_zoom.jpg (close-up of the area). The yellow circles show where each point is, to about 8 pixels.",
        "   Click the exact feature, not the circle. Zoom in first.",
        "5. After each point, type in the right-hand panel what is written under 'Type' below: 'x m' and 'y m' are the two",
        "   small boxes in the table, 'relation' is the text box under it. Leave a box EMPTY when it says 'blank'.",
        "   Never type a length that is not given: none of those lengths can be justified from the pictures.",
        "6. Press Ctrl+S. The page downloads a file named cal_clicks_<date>.json. Save it (or move it from your",
        f"   Downloads folder) to: {o}",
        "",
        "Floor frame used in the typed numbers: p1 is the origin (0, 0); x runs along the short arm of the painted line,",
        "y runs along the long painted line. 'blank' = the metres are not known.",
        "The only length used anywhere is the painted-stripe width of 0.10 m. That is the assumption of probe P6",
        "(docs/probes/S3a-P6.md), not a measurement. The scale of the floor map comes from the 4 m camera-height prior.",
        "",
    ]
    for fid, fr in plan["framings"].items():
        m = manifest[fid]
        L += ["=" * 78, f"PICTURE {fid}.jpg   (clip frame {m['keyframe']}, a train clip, 1920 x 1080)",
              f"Why this frame: {fr['why']}", f"Where to look: {fr['region']}",
              f"Lens (undistorted copy only): {fr['k1_note']}", ""]
        for p in fr["points"]:
            xm = "blank" if p["x_m"] is None else f"{p['x_m']:g}"
            ym = "blank" if p["y_m"] is None else f"{p['y_m']:g}"
            L += [f"  {p['id']}  near pixel ({p['uv'][0]}, {p['uv'][1]})", f"      Click: {p['what']}",
                  f"      Type:  x m = {xm},  y m = {ym}", f"      relation: {p['relation']}"]
        for p in fr.get("extra", []):
            L += [f"  {p['id']} (optional)  near pixel ({p['uv'][0]}, {p['uv'][1]})", f"      Click: {p['what']}",
                  f"      relation: {p['relation']}"]
        if fr.get("also_seen"):
            L += ["", f"  Also seen: {fr['also_seen']}"]
        L += ["", f"  Framing note (type once for this picture): '{fid}, frame {m['keyframe']}, clicked on the raw frame'.", ""]
    L += ["=" * 78, "FILES",
          f"  {o}\\click_these\\*.jpg            the pictures to click (raw frame, people black-boxed)",
          f"  {o}\\reference\\*_undistorted.jpg  same frame with the CAL v0 lens correction (straight lines look straight)",
          f"  {o}\\reference\\*_GUIDE*.jpg       numbered markers",
          f"  {o}\\cal_click_manifest.json     which clip frame, which k1, where the k1 came from", "",
          "If a point cannot be found or you are not sure, skip it and write 'skipped, why' in the framing note: a skipped",
          "point is better than a wrong one. Camera 14 (c14_f0) is the weak picture: it has only one clear painted line."]
    (out / "README.txt").write_text("\r\n".join(L) + "\r\n", encoding="utf-8")


def cmd_guide(a) -> int:
    out = Path(a.out)
    plan = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    manifest = json.loads((out / "cal_click_manifest.json").read_text(encoding="utf-8"))
    for fid, fr in plan["framings"].items():
        if manifest[fid]["keyframe"] != fr["keyframe"]:
            raise SystemExit(f"{fid}: exported keyframe {manifest[fid]['keyframe']} != plan keyframe {fr['keyframe']}")
    draw_guides(out, plan)
    write_readme(out, plan, manifest)
    print("guides and README written to", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("guide")
    pg.add_argument("--out", required=True)
    for n in ("rank", "export"):
        p = sub.add_parser(n)
        p.add_argument("--out", required=True)
        p.add_argument("--root", default=None)
        p.add_argument("--size", default="medium")
        if n == "rank":
            p.add_argument("--top", type=int, default=8)
        else:
            p.add_argument("--pick", action="append", required=True, help="framing_id=<clip>_k1|k2")
    a = ap.parse_args(argv)
    return {"rank": cmd_rank, "export": cmd_export, "guide": cmd_guide}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
