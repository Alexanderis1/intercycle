"""Where do the real ram-ROI occluders stand? (lane S3b fix1; WEAKNESSES T-W6; audit finding "zone-9 ROI occlusion
can never happen in the twin").

    python scripts/stat_v1_roi_occluders.py      # -> twin/stat_v1_roi_occluders.json (aggregates only, no image data)
    python scripts/stat_v1_roi_occluders.py --replay-only   # re-run the twin replays only, keep the TRAIN/DEV block
    python -m twin.camera --replay-held      # then: the held replay of the stat_v1 camera itself (key HELD_KEY)

This script imports no truth module (tests/test_truth_allowlist.py): the held replay runs the stat_v1 camera
(twin/camera.py, a truth reader) and therefore lives there, as `replay_library_held`; it reads no truth either.

TRAIN and DEV only (`esk.load("train")` / `esk.load("dev")`; no TEST clip is read). For every clip, every press ROI of
the clip's framing (twin/site.yaml ram_roi) and every sampled frame of the PERC boxes table
($INTERCYCLE_DATA/boxes/<clip>.parquet), the occluders are rebuilt as in the ingest (intercycle/perception/
esk_ingest.py step 7): every detection at conf >= OCC_CONF (0.4, the tracker activation) plus every detection whose
foot point is a confirmed track's foot point in that frame (tracks table, within TRACK_MATCH_PX). A frame is HIT when an
occluder box, padded ROI_PAD_FRAC, overlaps the ROI (`occlusion.box_hits_roi`). For each hitting box, its foot point
(bottom centre) is read to the plan floor through the NOMINAL camera of its framing (site.yaml K, k1, H) and
classified: on the zone's visible-floor polygon, inside the press footprint, off the floor; the distance to the
footprint; whether the foot is cut by the image bottom; in front of the press (closer to the camera nadir than the
footprint centre) or behind it.

The machines tables' MISSING share (ingest: hit held 1.33 s, frame 0 / non-increasing PTS frames MISSING too) is
reported next to the unheld hit share for the same clips, as a check that the rebuilt occluders match the ingest.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle.observer.occlusion import ROI_PAD_FRAC, box_hits_roi  # noqa: E402
from twin.cam_geom import (HEIGHT_M, PRESS_H_M, CamGeom, box_share_in_image, circle_prism,  # noqa: E402
                           press_anchor, roi_box)
from twin.schemas import load  # noqa: E402
from twin.schemas.site import Site, point_in_polygon  # noqa: E402

OUT = ROOT / "twin" / "stat_v1_roi_occluders.json"
HELD_KEY = "twin_replay_held_stat_v1_camera"   # written by twin/camera.py --replay-held (the camera is truth-side)
DISCLOSURE = (
    "fix2 attribution: the fix1 camera kept each ram ROI at its drawn pixels while the press prism and the movers "
    "were projected through the world's TRUE camera (one CAL v1 draw), so the ROI no longer sat on its press in a "
    "drawn world (fixed_roi_*: P9a 0 hits in most draws). That was a camera-geometry error, not a DAYGEN library gap: "
    "with the ROI anchored on the press (stat_v1 since fix2) P9a's library hit share is close to the nominal camera's "
    "and to TRAIN's in every draw. Only P9b's 0 % is a library gap (0 % even through the nominal camera: its real "
    "occluders stand in or at the footprint, where the library has no point); it needs at-press operators from the "
    "day generator (S3a). Draws whose anchored ROI leaves the image make that press MISSING for the whole world.")
OCC_CONF = 0.4
TRACK_MATCH_PX = 2.0
IMG_H = 1080


def _dist_poly(x: float, z: float, poly) -> float:
    if point_in_polygon(x, z, poly):
        return 0.0
    p = np.asarray(poly, float)
    best = math.inf
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        ab = b - a
        t = float(np.clip(np.dot([x, z] - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
        best = min(best, float(np.hypot(*(np.array([x, z]) - (a + t * ab)))))
    return best


def _q(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [round(float(np.percentile(a, q)), 2) for q in qs] if len(a) else None


def twin_replay(site: Site, presses: list[str] | None = None, draw_step: int = 1) -> dict:
    """The twin camera's geometry on the DAYGEN v1 path library ($INTERCYCLE_DATA/dyn/path_library_v1.npz: the
    TRAIN tracks the generated days resample, projected to the plan floor): per press, the share of (clip, 0.5 s)
    stamps at which a library person / forklift, placed exactly where its track says, has a box (TWIN heights, the
    stat_v1 prisms) over the press's first ROI (the one stat_v1 reads) in that ROI's framing. Through the NOMINAL
    camera and through each of the 200 CAL v1 joint draws (twin/cal/h_draws_v1.json; a world's TRUE camera is one
    draw): quantiles over the draws. The ROI is the one stat_v1 uses since fix2: its 3-D anchor on the press's
    visible surface (`cam_geom.press_anchor`) projected through the same camera as the movers (`roi_mode`). The fix1 camera's fixed
    drawn ROI is kept as `fixed_roi_*` (the audit's geometry error: the press and the movers move with the draw, the
    ROI did not). Plus the number of library points inside a press footprint (where the real zone-9 occluders
    stand) and the draws whose anchored ROI is not fully inside the image (stat_v1: that press is MISSING on every
    tick of such a world, zone_vis < 1). `presses` / `draw_step` (every k-th draw) restrict the replay (tests)."""
    lib = np.load(esk.data_root() / "dyn" / "path_library_v1.npz")
    hd = json.loads((ROOT / "twin" / "cal" / "h_draws_v1.json").read_text(encoding="utf-8"))
    fr = {f.id: f for f in site.framings}
    t = np.round((lib["unit_t0"] + lib["seg_t_off"] + lib["dt"]) * 2.0) / 2.0
    out: dict = {"library_points": int(len(t)), "n_draws": len(range(0, int(hd["n"]), draw_step)),
                 "roi_mode": ("anchored on the press's visible surface (cam_geom.press_anchor: the plane facing the "
                              "camera where the ROI-centre ray enters the press prism), projected through the same "
                              "camera as the movers"),
                 "presses": {}}
    for p in site.presses:
        if not p.ram_roi.value or (presses is not None and p.id not in presses):
            continue
        r = p.ram_roi.value[0]
        fid = r.framing
        poly = np.asarray(r.polygon_px, float)
        drawn = (float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max()))
        sel = np.flatnonzero(lib["zone"] == p.zone)
        keys = np.array([f"{c}@{tt}" for c, tt in zip(lib["clip_id"][sel], t[sel])])
        uk, inv = np.unique(keys, return_inverse=True)
        prisms = np.stack([circle_prism(float(lib["x"][i]), float(lib["z"][i]),
                                        0.3 if str(lib["cls"][i]) == "person" else 1.0,
                                        HEIGHT_M.get(str(lib["cls"][i]), 1.75)) for i in sel])     # (n, 16, 3)
        n, m = prisms.shape[0], prisms.shape[1]
        in_fp = int(sum(any(point_in_polygon(float(lib["x"][i]), float(lib["z"][i]), q.footprint.value)
                            for q in site.presses if q.zone == p.zone) for i in sel))

        def share(g: CamGeom, roi) -> float:
            if roi is None:
                return 0.0
            r0, s0, r1, s1 = roi
            raw, ok = g.project(prisms.reshape(-1, 3))
            raw, ok = raw.reshape(n, m, 2), ok.reshape(n, m).all(1)
            u0, v0, u1, v1 = raw[..., 0].min(1), raw[..., 1].min(1), raw[..., 0].max(1), raw[..., 1].max(1)
            pu, pv = ROI_PAD_FRAC * (u1 - u0), ROI_PAD_FRAC * (v1 - v0)
            hit = ok & (u0 - pu <= r1) & (u1 + pu >= r0) & (v0 - pv <= s1) & (v1 + pv >= s0)
            hk = np.zeros(len(uk), bool)
            np.logical_or.at(hk, inv, hit)
            return float(hk.mean()) if len(uk) else float("nan")

        K = np.asarray(fr[fid].K.value, float)
        gn = CamGeom(fid, K, float(fr[fid].k1.value), np.asarray(fr[fid].H_img_to_floor.value, float))
        anc = press_anchor(gn, poly, p.footprint.value, PRESS_H_M)
        nom = share(gn, roi_box(gn, anc))
        gts = [CamGeom.from_rec(fid, K, rec) for rec in hd["framings"][fid]["draws"][::draw_step]]
        tbs = [roi_box(g, anc) for g in gts]
        draws = np.array([share(g, tb) for g, tb in zip(gts, tbs)])
        fixed = np.array([share(g, drawn) for g in gts])
        shift = np.array([[(tb[0] + tb[2] - drawn[0] - drawn[2]) / 2, (tb[1] + tb[3] - drawn[1] - drawn[3]) / 2]
                          if tb is not None else [np.nan, np.nan] for tb in tbs])
        cut = int(sum(box_share_in_image(tb) < 1.0 for tb in tbs))
        out["presses"][p.id] = {"framing": fid, "stamps": int(len(uk)),
                                "roi_w_h_px": [round(drawn[2] - drawn[0], 1), round(drawn[3] - drawn[1], 1)],
                                "twin_hit_share_unheld_nominal": round(nom, 4),
                                "twin_hit_share_unheld_draws_q5_50_95": _q(draws, (5, 50, 95)),
                                "draws_with_zero_hits": int((draws == 0).sum()),
                                "draws_roi_not_fully_in_image": cut,
                                "roi_abs_shift_px_u_q50_90": _q(np.abs(shift[:, 0]), (50, 90)),
                                "roi_abs_shift_px_v_q50_90": _q(np.abs(shift[:, 1]), (50, 90)),
                                "fixed_roi_hit_share_unheld_draws_q5_50_95": _q(fixed, (5, 50, 95)),
                                "fixed_roi_draws_with_zero_hits": int((fixed == 0).sum()),
                                "library_points_in_a_press_footprint": in_fp}
    return out


def real_occluders(site: Site) -> dict:
    """The TRAIN / DEV block (module doc): where the real ROI occluders stand, per split x press x framing."""
    zones = {z.id: z for z in site.zones}
    geom = {f.id: CamGeom(f.id, np.asarray(f.K.value, float), float(f.k1.value), np.asarray(f.H_img_to_floor.value, float))
            for f in site.framings}
    rois = defaultdict(list)                       # framing -> [(press, roi polygon)]
    for p in site.presses:
        for r in p.ram_roi.value:
            rois[r.framing].append((p, np.asarray(r.polygon_px, float)))
    root = esk.data_root()
    acc: dict = defaultdict(lambda: defaultdict(list))
    cnt: dict = defaultdict(lambda: defaultdict(int))
    for split in ("train", "dev"):
        inv = esk.load(split)
        assert set(inv["split"]) == {split}
        for clip, fid in zip(inv["clip_id"].astype(str), inv["framing_id"].astype(str)):
            if fid not in rois:
                continue
            bx = pd.read_parquet(root / "boxes" / f"{clip}.parquet")
            tr = pd.read_parquet(root / "tracks" / f"{clip}.parquet")
            tfeet = defaultdict(list)
            for t, u, v in zip(tr["t"].to_numpy(float), tr["u"].to_numpy(float), tr["v"].to_numpy(float)):
                tfeet[round(t, 6)].append((u, v))
            frames = bx.groupby("frame") if len(bx) else []
            n_frames = len(np.unique(bx["frame"].to_numpy())) if len(bx) else 0
            hit_frames = defaultdict(int)
            for _, g in frames:
                t = round(float(g["t"].iloc[0]), 6)
                xy = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
                fu, fv = (xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]
                strong = g["conf"].to_numpy(float) >= OCC_CONF
                tracked = np.zeros(len(g), bool)
                for (u, v) in tfeet.get(t, []):
                    tracked |= np.hypot(fu - u, fv - v) <= TRACK_MATCH_PX
                occ = strong | tracked
                for press, roi in rois[fid]:
                    key = f"{split}:{press.id}:{fid}"
                    hit_any = False
                    for i in np.flatnonzero(occ):
                        if not box_hits_roi(xy[i], roi, ROI_PAD_FRAC):
                            continue
                        hit_any = True
                        fl = geom[fid].floor_of(np.array([[fu[i], fv[i]]]))[0]
                        x, z = float(fl[0]), float(fl[1])
                        poly = zones[press.zone].floor_polygon.value
                        on_floor = point_in_polygon(x, z, poly)
                        in_fp = point_in_polygon(x, z, press.footprint.value)
                        d_fp = _dist_poly(x, z, press.footprint.value)
                        nad = geom[fid].nadir
                        front = float(np.hypot(x - nad[0], z - nad[1])) < float(
                            np.hypot(*(np.asarray(press.footprint.value, float).mean(0) - nad)))
                        cnt[key]["boxes"] += 1
                        cnt[key]["on_floor"] += on_floor
                        cnt[key]["in_footprint"] += in_fp
                        cnt[key]["off_floor"] += (not on_floor) and (not in_fp)
                        cnt[key]["foot_cut_by_image_bottom"] += fv[i] >= IMG_H - 3
                        cnt[key]["in_front_of_press"] += front
                        acc[key]["d_footprint_m"].append(d_fp)
                        acc[key]["box_h_px"].append(xy[i, 3] - xy[i, 1])
                        acc[key]["x"].append(x)
                        acc[key]["z"].append(z)
                    hit_frames[press.id] += hit_any
            for press, _roi in rois[fid]:
                key = f"{split}:{press.id}:{fid}"
                cnt[key]["clips"] += 1
                cnt[key]["frames"] += n_frames
                cnt[key]["hit_frames"] += hit_frames[press.id]
                mp = root / "machines" / press.id / f"{clip}.parquet"
                if mp.exists():
                    m = pd.read_parquet(mp)
                    cnt[key]["machine_frames"] += len(m)
                    cnt[key]["machine_missing"] += int(m["missing"].to_numpy(bool).sum())
    out: dict = {"_doc": (__doc__ or "").split("\n\n")[0].strip(), "occ_conf": OCC_CONF, "roi_pad_frac": ROI_PAD_FRAC,
                 "camera": "nominal (twin/site.yaml)", "by_split_press_framing": {}}
    for key in sorted(cnt):
        c, a = cnt[key], acc[key]
        nb = max(c["boxes"], 1)
        out["by_split_press_framing"][key] = {
            "clips": c["clips"], "sampled_frames": c["frames"],
            "hit_share_unheld": round(c["hit_frames"] / c["frames"], 4) if c["frames"] else None,
            "machines_missing_share": round(c["machine_missing"] / c["machine_frames"], 4) if c["machine_frames"] else None,
            "hit_boxes": c["boxes"],
            "share_foot_on_zone_floor": round(c["on_floor"] / nb, 4),
            "share_foot_in_press_footprint": round(c["in_footprint"] / nb, 4),
            "share_foot_off_floor_and_footprint": round(c["off_floor"] / nb, 4),
            "share_foot_cut_by_image_bottom": round(c["foot_cut_by_image_bottom"] / nb, 4),
            "share_in_front_of_press": round(c["in_front_of_press"] / nb, 4),
            "d_footprint_m_q5_25_50_75_95": _q(a["d_footprint_m"]),
            "box_h_px_q5_25_50_75_95": _q(a["box_h_px"]),
            "foot_x_q5_50_95": _q(a["x"], (5, 50, 95)), "foot_z_q5_50_95": _q(a["z"], (5, 50, 95)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    site = load(Site, ROOT / "twin" / "site.yaml")
    if "--replay-only" in argv:                    # keep the TRAIN/DEV block, re-run the twin replays
        out = json.loads(OUT.read_text(encoding="utf-8"))
    else:
        out = real_occluders(site)
    out["twin_replay_daygen_v1_library"] = twin_replay(site)
    out["twin_replay_daygen_v1_library"]["disclosure"] = DISCLOSURE
    # the held replay runs the stat_v1 CAMERA itself, a truth module (twin/truth_guard.py): it lives in twin/camera.py
    # (`python -m twin.camera --replay-held`, which writes HELD_KEY of this file) and is kept here as last written
    if OUT.exists() and HELD_KEY in (prev := json.loads(OUT.read_text(encoding="utf-8"))):
        out[HELD_KEY] = prev[HELD_KEY]
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["by_split_press_framing"], indent=1))
    print(json.dumps(out["twin_replay_daygen_v1_library"], indent=1))
    print(f"{HELD_KEY}: re-run with  python -m twin.camera --replay-held  (after this script)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
