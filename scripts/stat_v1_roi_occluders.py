"""Where do the real ram-ROI occluders stand? (lane S3b fix1; WEAKNESSES T-W6; audit finding "zone-9 ROI occlusion
can never happen in the twin").

    python scripts/stat_v1_roi_occluders.py      # -> twin/stat_v1_roi_occluders.json (aggregates only, no image data)

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
from twin.camera import CamGeom  # noqa: E402
from twin.schemas import load  # noqa: E402
from twin.schemas.site import Site, point_in_polygon  # noqa: E402

OUT = ROOT / "twin" / "stat_v1_roi_occluders.json"
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


def twin_replay(site: Site, world: int = 7) -> dict:
    """The twin camera's geometry on the DAYGEN v1 path library ($INTERCYCLE_DATA/dyn/path_library_v1.npz: the
    TRAIN tracks the generated days resample, projected to the plan floor): per press, the share of (clip, 0.5 s)
    stamps at which a library person / forklift, placed exactly where its track says, has a box (true camera of
    `world`, TWIN heights) over the press's ROI in the ROI's framing, plus the share of library points on the visible
    floor and inside a press footprint (where the real zone-9 occluders stand)."""
    from twin.camera import HEIGHT_M, CameraStatV1, _circle_prism

    lib = np.load(esk.data_root() / "dyn" / "path_library_v1.npz")
    cam = CameraStatV1(site, world, 1)
    t = np.round((lib["unit_t0"] + lib["seg_t_off"] + lib["dt"]) * 2.0) / 2.0
    out: dict = {"world": world, "library_points": int(len(t)), "presses": {}}
    for p in site.presses:
        if p.id not in cam.roi:
            continue
        fid, (r0, s0, r1, s1) = cam.roi[p.id]
        sel = np.flatnonzero(lib["zone"] == p.zone)
        stamps, hit = set(), set()
        in_fp = 0
        for i in sel:
            x, z, cls = float(lib["x"][i]), float(lib["z"][i]), str(lib["cls"][i])
            key = (str(lib["clip_id"][i]), float(t[i]))
            stamps.add(key)
            in_fp += any(point_in_polygon(x, z, q.footprint.value) for q in site.presses if q.zone == p.zone)
            b = cam._box(fid, _circle_prism(x, z, 0.3 if cls == "person" else 1.0, HEIGHT_M.get(cls, 1.75)))
            if b is None:
                continue
            pu, pv = ROI_PAD_FRAC * (b[2] - b[0]), ROI_PAD_FRAC * (b[3] - b[1])
            if b[0] - pu <= r1 and b[2] + pu >= r0 and b[1] - pv <= s1 and b[3] + pv >= s0:
                hit.add(key)
        out["presses"][p.id] = {"framing": fid, "stamps": len(stamps),
                                "twin_hit_share_unheld": round(len(hit) / max(len(stamps), 1), 4),
                                "library_points_in_a_press_footprint": int(in_fp)}
    return out


def main() -> int:
    site = load(Site, ROOT / "twin" / "site.yaml")
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
    out["twin_replay_daygen_v1_library"] = twin_replay(site)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["by_split_press_framing"], indent=1))
    print(json.dumps(out["twin_replay_daygen_v1_library"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
