"""CAL v1 (lane S2): k1 refit, agent-placed floor points -> H, scale interval, bootstrap, site.yaml.

Every image input is an A1+ TRAIN clip (``esk.load("train")``); TEST is never loaded. Images stay in
``$INTERCYCLE_DATA/cal/`` (gitignored); only numbers are committed.

    # 1. GPU env, under the heavy lock: full-res keyframes of every train clip -> RF-DETR person boxes (conf >= 0.5)
    #    and a per-framing median background
    F:/envs/intercycle-gpu/Scripts/python.exe scripts/cal_v1.py gather
    # 2. k1 per framing from the straight lines of twin/cal/k1_hints_v1.json traced on the median backgrounds
    python scripts/cal_v1.py k1
    # 3. H per framing from twin/cal/cal_points_v1.json, scale checks, 200 bootstrap draws -> twin/cal/cal_v1.json
    python scripts/cal_v1.py fit
    # 4. twin/site.yaml from twin/cal/site_trace_v1.json (pixel tracings) through the CAL v1 H
    python scripts/cal_v1.py site
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle import framing as fr  # noqa: E402

CAL_DIR = ROOT / "twin" / "cal"
FRAMINGS = ("c9_f0", "c9_f1", "c14_f0")
CAMERA = {"c9_f0": 9, "c9_f1": 9, "c14_f0": 14}
CAM_FRAMINGS = {9: ("c9_f0", "c9_f1"), 14: ("c14_f0",)}
PERSON_CONF = 0.5
MEDIAN_MAX = 61
SEED = 20261018
# c14_f0's k1 rests on one floor line plus a short column edge (no leave-one-out): the bootstrap draws its k1
# uniformly between its own fit and the camera-9 fit of the same lens model (c9_f0, the one farthest from it)
K1_ALT = {"c14_f0": "c9_f0"}
# stripe estimates left out of the scale median (still reported): (framing, line) -> reason
STRIPE_FLAGGED = {("c9_f0", "B_green_walkway_left_yellow_line"):
                  "line B borders the green walkway band: one of its two traced edges is the green edge (artefact)"}


def data_cal() -> Path:
    d = esk.data_root() / "cal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_rows():
    inv = esk.load("train")
    assert set(inv["split"]) == {"train"}, "only train rows may be read"
    return inv


def keyframes_full(clip: Path) -> list[np.ndarray]:
    """Full-res BGR frames at 1/3 and 2/3 of the clip (the positions of keyframes k1 and k2)."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in (1, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(k * (n - 1) / 3.0))))
        ok, im = cap.read()
        if ok and im is not None and im.shape[:2] == (fr.H_IMG, fr.W):
            out.append(im)
    cap.release()
    return out


# --------------------------------------------------------------------------- 1. gather (GPU)


def cmd_gather(a) -> int:
    import cv2

    inv = train_rows()
    det = None
    if not a.bg_only:
        from intercycle.perception.detector import PERSON, RFDetrDetector

        det = RFDetrDetector(size="medium", threshold=PERSON_CONF, keep=(PERSON,))
    rng = np.random.default_rng(SEED)
    persons: dict = {"_doc": f"RF-DETR-M person boxes (conf >= {PERSON_CONF}) on full-res frames at 1/3 and 2/3 of "
                             "every A1+ TRAIN clip (keyframe positions). xyxy in px of 1920x1080, score.",
                     "frames": []}
    stacks: dict[str, list[np.ndarray]] = {f: [] for f in FRAMINGS}
    per_f = {f: sorted(inv.loc[inv["framing_id"] == f, "clip_id"]) for f in FRAMINGS}
    med_pick = {f: set(rng.choice(ids, size=min(MEDIAN_MAX, len(ids)), replace=False)) for f, ids in per_f.items()}
    for f, ids in per_f.items():
        for cid in ids:
            if det is None and cid not in med_pick[f]:
                continue
            frames = keyframes_full(esk.clip_path(cid))
            for k, im in enumerate(frames, start=1):
                if det is None:
                    break
                d = det.detect([cv2.cvtColor(im, cv2.COLOR_BGR2RGB)])[0]
                persons["frames"].append({"clip_id": cid, "k": k, "framing": f,
                                          "xyxy": np.round(d.xyxy, 1).tolist(), "score": np.round(d.score, 3).tolist()})
            if cid in med_pick[f] and frames:
                stacks[f].append(frames[0])
        print(f, len(ids), "clips;", len(stacks[f]), "frames in the median", flush=True)
    out = data_cal()
    if det is not None:
        (out / "persons_train.json").write_text(json.dumps(persons), encoding="utf-8")
        if not a.write_bg:
            print("wrote", out / "persons_train.json")
            return 0
    src = {}
    for f, st in stacks.items():
        bg = np.median(np.stack(st), axis=0).astype(np.uint8)
        cv2.imwrite(str(out / f"{f}_bg.png"), bg)
        src[f] = sorted(med_pick[f])
    (out / "bg_sources.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


# --------------------------------------------------------------------------- 2. k1


def load_bg(fid: str) -> np.ndarray:
    import cv2

    p = data_cal() / f"{fid}_bg.png"
    im = cv2.imread(str(p))
    if im is None:
        raise SystemExit(f"missing {p}: run `scripts/cal_v1.py gather` first")
    return im


MIN_TRACE_PTS = 15


def hinted_lines(hints: dict) -> set[str]:
    """World line names of a hint dict ('#n' pieces of one line count once)."""
    return {n.split("#")[0] for n in hints if not n.startswith("_")}


def trace_lines(fid: str, hints: dict, dropped: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """{world line name: raw-pixel chain}; '#n' pieces of one line are joined.

    A hint (or a '#n' piece) that does not trace, or a line with fewer than MIN_TRACE_PTS points, is left out of the
    result and recorded in ``dropped`` (name -> reason); a line with a dropped piece is dropped whole. Callers that
    need every hinted line fail closed on a non-empty ``dropped`` (``require_all``)."""
    import cv2

    gray = cv2.cvtColor(load_bg(fid), cv2.COLOR_BGR2GRAY)
    out: dict[str, list[np.ndarray]] = {}
    lost: dict[str, str] = {}
    for n, h in hints.items():
        if n.startswith("_"):
            continue
        try:
            c = fr.trace_edge(gray, h[0], h[1], **(h[2] if len(h) > 2 else {}))
        except (TypeError, ValueError) as e:      # no consistent edge along the hint
            print(f"{fid} {n}: trace failed ({e}); line dropped")
            lost[n.split("#")[0]] = f"{n}: trace failed ({e})"
            continue
        print(f"{fid} {n}: {len(c)} pts")
        out.setdefault(n.split("#")[0], []).append(c)
    res = {}
    for n, cs in out.items():
        if n in lost:
            continue
        if sum(len(c) for c in cs) < MIN_TRACE_PTS:
            lost[n] = f"{n}: fewer than {MIN_TRACE_PTS} edge points"
            continue
        res[n] = np.vstack(cs)
    if dropped is not None:
        dropped.update(lost)
    return res


def require_all(fid: str, what: str, hints: dict, chains: dict, dropped: dict[str, str], min_lines: int = 2) -> None:
    """Fail closed: every hinted line traced, and at least ``min_lines`` of them."""
    missing = sorted(hinted_lines(hints) - set(chains))
    if dropped or missing or len(chains) < min_lines:
        raise SystemExit(f"{fid} {what}: {len(chains)} of {len(hinted_lines(hints))} hinted lines traced "
                         f"(need all, and >= {min_lines}); dropped {dropped or missing}")


def bg_sources() -> dict[str, list[str]]:
    return json.loads((data_cal() / "bg_sources.json").read_text(encoding="utf-8"))


def cmd_k1(a) -> int:
    hints = json.loads((CAL_DIR / "k1_hints_v1.json").read_text(encoding="utf-8"))
    src = bg_sources()
    out: dict = {
        "_doc": "CAL v1 lens distortion (lane S2), one division-model k1 per framing, refit on A1+ TRAIN clips only. "
                "x_u = c + (x_d - c) / (1 + k1 r_d^2), r_d in px of the 1920x1080 frame, c = image centre; "
                "k1_norm = k1 * 1330^2. Lines: twin/cal/k1_hints_v1.json traced on the per-framing median of one "
                "full-res frame from each of up to 61 seeded train clips (source_clip_ids). opencv: K and the rational "
                "dist vector for cv2.undistortPoints(P=K). Supersedes k1_v0.json (INVALID: fitted before A1+).",
        "supersedes": "k1_v0.json", "center": list(fr.CENTER), "f_norm": fr.F_PRIOR}
    for fid in FRAMINGS:
        dropped: dict[str, str] = {}
        chains = trace_lines(fid, hints[fid], dropped)
        require_all(fid, "k1", hints[fid], chains, dropped)
        rec: dict = {"lines": {n: {"n_pts": int(len(c)), "len_px": round(fr.chain_length(c), 1),
                                   "rms_raw_px": round(fr.line_rms(c), 3)} for n, c in chains.items()}}
        fit = fr.fit_k1(list(chains.values()))
        rec.update({"k1": fit["k1"], "k1_norm": fit["k1_norm"], "at_bound": fit["at_bound"],
                    "rms_after_px": dict(zip(chains, [round(x, 3) for x in fit["rms_after_px"]]))})
        rec["single_line_k1_norm"] = {n: round(fr.fit_k1([c])["k1_norm"], 4) for n, c in chains.items()}
        if len(chains) >= 3:
            rec["leave_one_out_k1_norm"] = {
                n: round(fr.fit_k1([c for m, c in chains.items() if m != n])["k1_norm"], 4) for n in chains}
        if fid in K1_ALT:
            rec["weak_fit_note"] = (f"only {len(chains)} lines (no leave-one-out): the CAL v1 bootstrap draws this "
                                    f"framing's k1 uniformly between this fit and the {K1_ALT[fid]} fit (same lens "
                                    "model); the shift it causes is in twin/cal/cal_v1.json c14_f0_k1_alt_check")
        rec["opencv"] = fr.opencv_coeffs(fit["k1"])
        rec["source_clip_ids"] = src[fid]
        out[fid] = rec
        print(fid, json.dumps({k: v for k, v in rec.items() if k not in ("opencv", "source_clip_ids")}))
    (CAL_DIR / "k1_v1.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", CAL_DIR / "k1_v1.json")
    return 0


# --------------------------------------------------------------------------- 3. fit


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def framing_cal(fid: str, pts_doc: dict, k1_doc: dict, hints: dict, h_prior: float = fr.MOUNT_H_PRIOR_M):
    from intercycle import calib as C

    d = pts_doc[fid]
    k1 = float(k1_doc[fid]["k1"])
    dropped: dict[str, str] = {}
    chains = trace_lines(fid, hints[fid], dropped)
    chains.update(trace_lines(fid, d.get("line_hints", {}), dropped))   # short stripe arms for the relations
    used = {n for r in d["relations"] for n in r[1:]} | {d["axis"]}
    lines = {n: fr.undistort(c, k1) for n, c in chains.items() if n in used}
    missing = sorted(n for n in used if n not in lines and n not in d["segments"])
    if missing or any(n in used for n in dropped):
        raise SystemExit(f"{fid}: relation lines that did not trace (whole): {missing}; dropped {dropped}")
    alt = K1_ALT.get(fid)
    cal = C.FramingCal(
        framing=fid, k1=k1,
        points_raw={k: np.asarray(v["uv"], float) for k, v in d["points"].items()},
        sigma_px={k: float(v["sigma_px"]) for k, v in d["points"].items()},
        lines_undist=lines, segments={k: (v[0], v[1]) for k, v in d["segments"].items()},
        relations=[C.Relation(r[0], r[1], r[2] if len(r) > 2 else "") for r in d["relations"]],
        origin=d["origin"], axis=d["axis"],
        lines_raw={n: c for n, c in chains.items() if n in used},
        k1_alt=None if alt is None else float(k1_doc[alt]["k1"]))
    return cal, chains


def stripe_edges(fid: str, hints: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    """Both edges (opposite polarity) of each painted stripe hint, RAW pixels; and the stripe hints that did not give
    two edges (name -> reason), which callers record."""
    import cv2

    gray = cv2.cvtColor(load_bg(fid), cv2.COLOR_BGR2GRAY)
    out, dropped = {}, {}
    for n, h in hints.items():
        if n.startswith("_") or "#" in n or not any(s in n for s in ("stripe", "yellow")):
            continue
        try:
            ea = fr.trace_edge(gray, h[0], h[1], half_width=12.0, polarity=1.0)
            eb = fr.trace_edge(gray, h[0], h[1], half_width=12.0, polarity=-1.0)
        except (TypeError, ValueError) as e:
            dropped[n] = f"edge trace failed ({e})"
            continue
        if len(ea) >= MIN_TRACE_PTS and len(eb) >= MIN_TRACE_PTS:
            out[n] = (ea, eb)
        else:
            dropped[n] = f"edges of {len(ea)} / {len(eb)} points (< {MIN_TRACE_PTS})"
    if dropped:
        print(f"{fid} stripes dropped: {dropped}")
    return out, dropped


def person_boxes(fid: str, persons: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RAW foot and head-top pixels and clip ids of the upright, unclipped person boxes of this framing."""
    foot, head, clip = [], [], []
    for fr_rec in persons["frames"]:
        if fr_rec["framing"] != fid:
            continue
        for (x0, y0, x1, y1), s in zip(fr_rec["xyxy"], fr_rec["score"]):
            w, hgt = x1 - x0, y1 - y0
            if s < PERSON_CONF or x0 < 5 or y0 < 5 or x1 > fr.W - 5 or y1 > fr.H_IMG - 5:
                continue
            if hgt < 80 or hgt < 2.0 * w:          # small or not upright (sitting, bending, occluded)
                continue
            cx = 0.5 * (x0 + x1)
            foot.append([cx, y1])
            head.append([cx, y0])
            clip.append(fr_rec["clip_id"])
    return np.asarray(foot, float).reshape(-1, 2), np.asarray(head, float).reshape(-1, 2), np.asarray(clip, str)


def person_heights(fid: str, k1: float, theta: float, phi: float, persons: dict) -> dict:
    """Camera height from upright, unclipped RF-DETR person boxes of TRAIN frames of this framing (1.72 m walker),
    at ONE orientation (the nominal fit); the joint spread over the bootstrap orientations is ``joint_bootstrap``'s."""
    from intercycle import calib as C

    foot, head, clip = person_boxes(fid, persons)
    if not len(foot):
        return {"n_boxes": 0}
    hs = C.person_height_h(fr.undistort(foot, k1), fr.undistort(head, k1), theta, phi)
    ok = np.isfinite(hs)
    hs, clip_a = hs[ok], clip[ok]
    ids = sorted(set(clip_a))
    rng = np.random.default_rng(SEED)
    by = {c: hs[clip_a == c] for c in ids}
    boots = [float(np.median(np.concatenate([by[c] for c in rng.choice(ids, len(ids))]))) for _ in range(1000)]
    return {"n_boxes": int(len(hs)), "n_clips": len(ids), "h_median_m": float(np.median(hs)),
            "h_boot90_m": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "clips": ids}


def cmd_fit(a) -> int:
    from intercycle import calib as C

    pts_doc = load_json(CAL_DIR / "cal_points_v1.json")
    k1_doc = load_json(CAL_DIR / "k1_v1.json")
    hints = load_json(CAL_DIR / "k1_hints_v1.json")
    persons = load_json(data_cal() / "persons_train.json")
    res: dict = {}
    for fid in FRAMINGS:
        cal, _ = framing_cal(fid, pts_doc, k1_doc, hints)
        b = cal.build()
        ori = b["orientation"]
        th, ph = math.radians(ori["theta_deg"]), math.radians(ori["phi_deg"])
        und = b["undist"]
        ids = list(und)
        uv = np.array([und[i] for i in ids])
        xz = C.apply_h(b["H"], uv)
        H_dlt = C.homography_from_points(uv, xz)
        dlt_err = float(np.nanmax(np.linalg.norm(C.apply_h(H_dlt, uv) - xz, axis=1)))
        if dlt_err > 1e-3:
            raise SystemExit(f"{fid}: findHomography disagrees with the model H by {dlt_err} m")
        edges, st_dropped = stripe_edges(fid, hints[fid])
        stripes = {n: C.stripe_h(fr.undistort(ea, cal.k1), fr.undistort(eb, cal.k1), th, ph)
                   for n, (ea, eb) in edges.items()}
        people = person_heights(fid, cal.k1, th, ph, persons)
        res[fid] = {"k1": cal.k1, "theta_deg": ori["theta_deg"], "phi_deg": ori["phi_deg"],
                    "relation_resid_deg": ori["resid_deg"], "H_at_4m": b["H"].tolist(),
                    "points_undist": {i: und[i].round(2).tolist() for i in ids},
                    "points_floor_xz_at_4m": {i: xz[k].round(3).tolist() for k, i in enumerate(ids)},
                    "stripe_h_m": {n: round(v, 3) for n, v in stripes.items()}, "stripes_dropped": st_dropped,
                    "people": people}
        print(fid, json.dumps({k: v for k, v in res[fid].items() if k not in ("H_at_4m", "people")}),
              json.dumps({k: v for k, v in people.items() if k != "clips"}))
    # cross-framing check (zone 9): the recess corner R2 in both c9 framings' zone frames
    r2 = {f: res[f]["points_floor_xz_at_4m"][pts_doc[f]["checks"]["R2"]] for f in ("c9_f0", "c9_f1")}
    res["check_R2_c9"] = {"c9_f0": r2["c9_f0"], "c9_f1": r2["c9_f1"],
                          "dist_m": float(np.hypot(*(np.subtract(r2["c9_f0"], r2["c9_f1"]))))}
    print("R2 check", res["check_R2_c9"])
    (data_cal() / "fit_v1.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- 4. site

N_BOOT = 200
RASTER_M = 0.05
ZONE_OF = {"c9_f0": "z9", "c9_f1": "z9", "c14_f0": "z14"}
ZONE_GAP_M = 10.0


def _pct(a, q) -> float:
    return round(float(np.nanpercentile(np.asarray(a, float), q)), 3)


def joint_bootstrap(cals: dict, persons: dict, hints: dict, fit: dict, n: int = N_BOOT) -> tuple[dict, dict, dict]:
    """CAL v1 bootstrap with the camera height drawn JOINTLY with each draw's lens and orientation.

    Per camera and draw i (the framings of one camera share draw i and one height, one mount):
    1. every framing: ``calib.draw_orientations`` (points jittered by their sigma, lines resampled, k1 drawn where
       K1_ALT says so, tilt / roll refitted);
    2. at THAT draw's k1, tilt and roll: people height = median over a clip-resampled set of the framing's upright
       person boxes (1.72 m walker); stripe height = median over the camera's 10 cm stripes (edges resampled),
       STRIPE_FLAGGED left out;
    3. h_i ~ U(span of {4 m prior, each framing's people_i, stripes_i});
    4. H_i = the draw rebuilt at h_i.
    Published camera interval: [min(prior, p5 of each estimate), max(prior, p95 of each estimate)] over the draws.

    Returns ({framing: draws at h_i}, {framing: the same draws at the 4 m prior}, {camera: scale record}).
    """
    from intercycle import calib as C

    prior = fr.MOUNT_H_PRIOR_M
    draws_h, draws_4, scale = {}, {}, {}
    for cam, fids in CAM_FRAMINGS.items():
        d4 = {f: C.draw_orientations(cals[f], n, SEED + FRAMINGS.index(f)) for f in fids}
        rng = np.random.default_rng(SEED + 1000 + cam)
        boxes = {f: person_boxes(f, persons) for f in fids}
        by_clip = {f: {c: np.flatnonzero(boxes[f][2] == c) for c in sorted(set(boxes[f][2]))} for f in fids}
        stripes = {}
        flagged, dropped = {}, {}
        for f in fids:
            e, dr = stripe_edges(f, hints[f])
            dropped.update({f"{f}/{k}": v for k, v in dr.items()})
            for k, v in e.items():
                if (f, k) in STRIPE_FLAGGED:
                    flagged[f"{f}/{k}"] = STRIPE_FLAGGED[(f, k)]
                else:
                    stripes[(f, k)] = v
        if not stripes:
            raise SystemExit(f"camera {cam}: no stripe traced for the scale cross-check")
        ppl: dict[str, list[float]] = {f: [] for f in fids}
        st_i, h_i, span_i = [], [], []
        draws_h.update({f: [] for f in fids})
        for i in range(n):
            for f in fids:
                d = d4[f][i]
                th, ph = math.radians(d["orientation"]["theta_deg"]), math.radians(d["orientation"]["phi_deg"])
                foot, head, _ = boxes[f]
                hb = C.person_height_h(fr.undistort(foot, d["k1"]), fr.undistort(head, d["k1"]), th, ph)
                ids = list(by_clip[f])
                pick = rng.choice(len(ids), len(ids))
                ppl[f].append(float(np.nanmedian(np.concatenate([hb[by_clip[f][ids[j]]] for j in pick]))))
            sv = []
            for (f, _k), (ea, eb) in stripes.items():
                d = d4[f][i]
                th, ph = math.radians(d["orientation"]["theta_deg"]), math.radians(d["orientation"]["phi_deg"])
                ia = np.sort(rng.integers(0, len(ea), len(ea)))
                ib = np.sort(rng.integers(0, len(eb), len(eb)))
                sv.append(C.stripe_h(fr.undistort(ea[ia], d["k1"]), fr.undistort(eb[ib], d["k1"]), th, ph))
            st_i.append(float(np.nanmedian(sv)))
            cand = [prior, st_i[-1]] + [ppl[f][-1] for f in fids]
            cand = [c for c in cand if np.isfinite(c)]
            lo, hi = min(cand), max(cand)
            span_i.append((lo, hi))
            h = float(rng.uniform(lo, hi))
            h_i.append(h)
            for f in fids:
                draws_h[f].append(C.at_height(cals[f], d4[f][i], h))
        draws_4.update(d4)
        lo_all = min([prior, _pct(st_i, 5)] + [_pct(ppl[f], 5) for f in fids])
        hi_all = max([prior, _pct(st_i, 95)] + [_pct(ppl[f], 95) for f in fids])
        # nominal (single-orientation) estimates, for the P6 disagreement rule and the record
        p_nom = float(np.median([fit[f]["people"]["h_median_m"] for f in fids]))
        s_nom_all = {f"{f}/{k}": v for f in fids for k, v in fit[f]["stripe_h_m"].items()}
        s_nom = float(np.median([v for k, v in s_nom_all.items() if k not in flagged]))
        s_nom_incl = float(np.median(list(s_nom_all.values())))
        est = [prior, p_nom, s_nom]
        tilt = {f: np.array([d["orientation"]["theta_deg"] for d in d4[f]]) for f in fids}
        rec = {
            "h_interval_m": [round(lo_all, 3), round(hi_all, 3)],
            "factor": [round(lo_all / prior, 4), round(hi_all / prior, 4)], "prior_h_m": prior,
            "joint_people_h_p5_p50_p95_m": {f: [_pct(ppl[f], 5), _pct(ppl[f], 50), _pct(ppl[f], 95)] for f in fids},
            "joint_stripes_h_p5_p50_p95_m": [_pct(st_i, 5), _pct(st_i, 50), _pct(st_i, 95)],
            "draw_h_p5_p50_p95_m": [_pct(h_i, 5), _pct(h_i, 50), _pct(h_i, 95)],
            "draw_span_lo_p5_hi_p95_m": [_pct([s[0] for s in span_i], 5), _pct([s[1] for s in span_i], 95)],
            "corr_tilt_people_h": {f: round(float(np.corrcoef(tilt[f], ppl[f])[0, 1]), 3) for f in fids},
            "nominal_people_h_boot90_m": {f: [round(v, 3) for v in fit[f]["people"]["h_boot90_m"]] for f in fids},
            "nominal_people_h_median_m": round(p_nom, 3),
            "nominal_stripes_h_m": {k: round(v, 3) for k, v in s_nom_all.items()},
            "nominal_stripes_median_h_m": round(s_nom, 3),
            "nominal_stripes_median_incl_flagged_h_m": round(s_nom_incl, 3),
            "stripes_flagged_out": flagged, "stripes_dropped": dropped,
            "max_disagreement": round((max(est) - min(est)) / min(est), 3),
            "people_n_boxes": int(sum(fit[f]["people"]["n_boxes"] for f in fids)),
            "people_n_clips": int(sum(fit[f]["people"]["n_clips"] for f in fids))}
        rec["verdict"] = ("FALLBACK (> 20 %): scale interval, robot at both ends"
                          if rec["max_disagreement"] > 0.20 else "pass (<= 20 %)")
        scale[cam] = rec
        print(f"camera {cam} scale", json.dumps(rec))
    return draws_h, draws_4, scale


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(p: np.ndarray) -> np.ndarray:
    p = np.unique(np.round(np.asarray(p, float), 9), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h: list = []
        for q in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    return float(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def buffer_convex(poly: np.ndarray, r: float, n: int = 16) -> np.ndarray:
    """Convex polygon grown by r (Minkowski sum with a disc, n points per vertex), counter-clockwise."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.stack([np.cos(ang), np.sin(ang)], 1) * r
    return convex_hull((np.asarray(poly, float)[:, None, :] + circ[None]).reshape(-1, 2))


def union_polygon(polys: list[np.ndarray], disc: tuple[np.ndarray, float] | None) -> np.ndarray:
    """Outer outline of the union of plan polygons (5 cm raster), intersected with a disc (far cut)."""
    import cv2

    allp = np.vstack(polys)
    lo = allp.min(0) - 1.0
    hi = allp.max(0) + 1.0
    w, h = (np.ceil((hi - lo) / RASTER_M)).astype(int) + 1
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round((p - lo) / RASTER_M).astype(np.int32)], 1)
    if disc is not None:
        d = np.zeros_like(m)
        c = np.round((disc[0] - lo) / RASTER_M).astype(int)
        cv2.circle(d, (int(c[0]), int(c[1])), int(disc[1] / RASTER_M), 1, -1)
        m &= d
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cs, key=cv2.contourArea)
    big = cv2.approxPolyDP(big, 0.10 / RASTER_M, True).reshape(-1, 2)
    return big * RASTER_M + lo


def tag(value, source: str, clips: list[str] | None = None, note: str | None = None) -> dict:
    d: dict = {"value": value, "source": source}
    if clips:
        d["clips"] = sorted(set(clips))
    if note:
        d["note"] = note
    return d


def rnd(a, k: int = 3):
    return [[round(float(x), k) for x in row] for row in np.asarray(a, float)]


def cmd_site(a) -> int:
    import yaml

    from intercycle import calib as C
    from twin import provenance as PV
    from twin.schemas import site as SS

    pts_doc = load_json(CAL_DIR / "cal_points_v1.json")
    k1_doc = load_json(CAL_DIR / "k1_v1.json")
    hints = load_json(CAL_DIR / "k1_hints_v1.json")
    trace = load_json(CAL_DIR / "site_trace_v1.json")
    fit = load_json(data_cal() / "fit_v1.json")
    rois = load_json(ROOT / "docs" / "probes" / "S3a-P4_rois.json")
    src = bg_sources()
    persons = load_json(data_cal() / "persons_train.json")
    people_clips = sorted({c for f in FRAMINGS for c in fit[f]["people"]["clips"]})
    click_clips = sorted(pts_doc["_click_sheet_clips"].values())
    p4_clips = {k: v for k, v in trace["p4_clips"].items() if not k.startswith("_")}

    # nominal calibrations; zone 14 is shifted to the right of zone 9 (no shared floor, P7)
    cals = {f: framing_cal(f, pts_doc, k1_doc, hints)[0] for f in FRAMINGS}

    def plan_poly(fid: str, raw_px) -> np.ndarray:
        b = cals[fid].build()
        return C.apply_h(b["H"], C.undistort_cv2(np.asarray(raw_px, float), cals[fid].k1))

    def nadir(fid: str) -> np.ndarray:
        return np.asarray(cals[fid].build()["nadir_xz"], float)

    far = float(trace["far_cut_m"])
    z9 = union_polygon([plan_poly(f, trace["floor"][f]) for f in ("c9_f0", "c9_f1")],
                       (nadir("c9_f0"), far))
    z14_local = union_polygon([plan_poly("c14_f0", trace["floor"]["c14_f0"])], (nadir("c14_f0"), far))
    off = float(math.ceil(z9[:, 0].max() + ZONE_GAP_M - z14_local[:, 0].min()))
    cals["c14_f0"].offset_xz = (off, 0.0)
    z14 = z14_local + np.array([off, 0.0])

    # CAL v1 bootstrap: per camera, h drawn jointly with each draw's k1 / tilt / roll (joint_bootstrap)
    boots, boots4, scale = joint_bootstrap(cals, persons, hints, fit)
    nominal = {f: cals[f].build() for f in FRAMINGS}

    # c14_f0 lens check: the nominal fit redone with the camera-9 k1 (the other end of the k1 draw)
    c14, alt_f = cals["c14_f0"], K1_ALT["c14_f0"]
    assert c14.lines_raw is not None and c14.k1_alt is not None
    c14_alt = c14.build(lines={k: fr.undistort(v, c14.k1_alt) for k, v in c14.lines_raw.items()}, k1=c14.k1_alt)

    presses_out, press_meta = [], {}
    roi_by = {p["id"]: p["rois"] for p in rois["presses"]}
    clearance = float(trace["danger_clearance_m"])
    for pid, pr in trace["presses"].items():
        fid = pr["framing"]
        if "footprint_px" in pr:
            fp = plan_poly(fid, pr["footprint_px"])
            fp_tag = tag(rnd(fp), "traced_v1", src[fid], f"press {pr['status']}; " + pr["note"])
        else:
            c = plan_poly(fid, [pr["centre_px"]])[0]
            sx, sz = pr["size_m"]
            fp = np.array([[c[0] - sx / 2, c[1] - sz / 2], [c[0] + sx / 2, c[1] - sz / 2],
                           [c[0] + sx / 2, c[1] + sz / 2], [c[0] - sx / 2, c[1] + sz / 2]])
            fp_tag = tag(rnd(fp), "assumed", None, f"press {pr['status']}; " + pr["note"])
        danger = buffer_convex(convex_hull(fp), clearance)
        # dilation: p95 over draws and danger vertices of the floor displacement (RAW pixels, each draw with its own
        # k1), in the framing the footprint was placed in and every framing with a P4 ROI of the press
        fset = sorted({fid} | set(roi_by[pid]))
        rng_c = np.random.default_rng(SEED + 2000 + list(trace["presses"]).index(pid))
        disp, disp4, disp_local = [], [], []
        for f in fset:
            Hn = nominal[f]["H"]
            raw = fr.distort(C.apply_h(np.linalg.inv(Hn), danger), cals[f].k1)   # RAW pixels of the danger vertices
            if f == fid and "centre_px" in pr:
                # centre-placed (rigid metric square): every vertex moves with the centre pixel, which also carries
                # the stated placement sigma
                c_raw = np.asarray(pr["centre_px"], float)
                x0 = C.floor_of_raw(nominal[f], c_raw)[0]
                jit = rng_c.normal(0.0, float(pr["centre_sigma_px"]), (N_BOOT, 2))
                for dset, out in ((boots[f], disp), (boots4[f], disp4)):
                    sh = np.array([np.linalg.norm(C.floor_of_raw(b, c_raw + jit[k])[0] - x0)
                                   for k, b in enumerate(dset)])
                    out.append(np.repeat(sh, len(danger)))
                disp_local.append(np.zeros(N_BOOT * len(danger)))
                continue
            disp.append(C.displacement_raw_m(nominal[f], boots[f], raw).ravel())
            disp4.append(C.displacement_raw_m(nominal[f], boots4[f], raw).ravel())
            x0 = C.floor_of_raw(nominal[f], raw)
            for b in boots[f]:                                   # shape/size error only (press centroid pinned)
                xb = C.floor_of_raw(b, raw)
                disp_local.append(np.linalg.norm((xb - xb.mean(0)) - (x0 - x0.mean(0)), axis=1))
        d, d4 = np.concatenate(disp), np.concatenate(disp4)
        dil = float(np.percentile(d, 95))
        dil4 = float(np.percentile(d4, 95))
        press_meta[pid] = {"dilation_m": round(dil, 3), "disp_p50_m": round(float(np.median(d)), 3),
                           "dilation_fixed_scale_p95_m": round(dil4, 3),
                           "dilation_local_p95_m": round(float(np.percentile(np.concatenate(disp_local), 95)), 3),
                           "framings": fset, "footprint_area_m2": round(polygon_area(fp), 2),
                           "footprint_side_m": [round(float(np.linalg.norm(fp[(i + 1) % len(fp)] - fp[i])), 2)
                                                for i in range(len(fp))]}
        if "centre_px" in pr:
            press_meta[pid]["centre_sigma_px"] = float(pr["centre_sigma_px"])
        if fid == "c14_f0":
            ca = C.floor_of_raw(c14_alt, fr.distort(C.apply_h(np.linalg.inv(nominal[fid]["H"]), fp), c14.k1))
            press_meta[pid]["centroid_shift_with_c9_k1_m"] = round(float(np.linalg.norm(ca.mean(0) - fp.mean(0))), 3)
        presses_out.append({
            "id": pid, "zone": pr["zone"], "footprint": fp_tag,
            "ram_roi": tag([{"framing": f, "polygon_px": [[float(x), float(y)] for x, y in poly]}
                            for f, poly in roi_by[pid].items()], "measured", p4_clips[pid],
                           "probe P4 ROIs (docs/probes/S3a-P4_rois.json), full-res RAW px, unchanged. clips = the 10 "
                           "train clips probe P4 measured this press on (P4 run, seed 20261017); P4 drew the ROIs on "
                           "train keyframes and motion maps of 15-16 train clips per framing whose ids it did not record"),
            "danger_polygon": tag(rnd(danger), "traced_v1" if "footprint_px" in pr else "assumed",
                                  src[fid] if "footprint_px" in pr else None,
                                  f"footprint grown by {clearance:g} m ASSUMED clearance (no guard geometry observed); "
                                  "consumers grow it further by dilation_m (see its note)"),
            "dilation_m": tag(round(dil, 3), "cal_v1", sorted({c for f in fset for c in src[f]}),
                              f"p95 floor displacement of the danger-polygon vertices over {N_BOOT} bootstrap draws of "
                              "H (agent click sigma, traced-line resampling, k1 draw for c14_f0, and the camera height "
                              "drawn jointly with each draw's tilt / roll)"
                              + (f", plus the centre placement sigma {pr['centre_sigma_px']:g} px"
                                 if "centre_px" in pr else "")
                              + ", framings " + ", ".join(fset) + ". CONSUMER NOTE: scale-inclusive. A consumer that "
                              "applies the scale interval separately (robot run at both scale ends) counts scale twice "
                              f"with this value; use dilation_fixed_scale_p95_m = {dil4:.3f} m instead (same draws at "
                              "the 4 m prior; twin/cal/cal_v1.json presses)")})

    framings_out = []
    for f in FRAMINGS:
        b = nominal[f]
        o = b["orientation"]
        lo, hi = scale[CAMERA[f]]["h_interval_m"]
        lens_note = ""
        if f in K1_ALT:
            sh = [press_meta[p]["centroid_shift_with_c9_k1_m"] for p in press_meta if "centroid_shift_with_c9_k1_m"
                  in press_meta[p]]
            lens_note = (f"; k1 is weak (one floor line + a short column edge): the bootstrap draws k1 uniformly between "
                         f"this fit (k1_norm {k1_doc[f]['k1_norm']:.3f}) and the {K1_ALT[f]} fit "
                         f"({k1_doc[K1_ALT[f]]['k1_norm']:.3f}); with the {K1_ALT[f]} k1 the roll moves from "
                         f"{o['phi_deg']:.2f} to {c14_alt['orientation']['phi_deg']:.2f} deg and the press footprint "
                         f"centroids move by {min(sh):.2f}-{max(sh):.2f} m (included in dilation_m)")
        framings_out.append({
            "id": f, "camera": CAMERA[f],
            "image_size_px": tag([fr.W, fr.H_IMG], "measured", src[f]),
            "K": tag(rnd(C.K_matrix(), 4), "prior", None, "f = 1330 px prior (vendor VFOV; range 1020-1430 not "
                     "propagated), principal point = image centre (959.5, 539.5)"),
            "k1": tag(float(k1_doc[f]["k1"]), "cal_v1", src[f], f"division model, k1_norm "
                      f"{k1_doc[f]['k1_norm']:.4f}; twin/cal/k1_v1.json"),
            "H_img_to_floor": tag(rnd(b["H"], 10), "cal_v1", src[f],
                                  f"undistorted px -> zone frame (x, z) m at h = 4.0 m; tilt {o['theta_deg']:.2f} deg, "
                                  f"roll {o['phi_deg']:.2f} deg from agent-placed points (twin/cal/cal_points_v1.json), "
                                  "CAL points placed by a Claude agent from train keyframes (vision), not by a human"
                                  + lens_note),
            "mount_height_m": tag(fr.MOUNT_H_PRIOR_M, "prior", None,
                                  f"P6 prior; camera-{CAMERA[f]} interval {lo}-{hi} m (scale_v1.json)")})

    lo_all = min(s["factor"][0] for s in scale.values())
    hi_all = max(s["factor"][1] for s in scale.values())
    site = {
        "contract": "twin.site/1", "site_id": "esk_press_v1", "frame": "plan_xz_m",
        "scale": tag({"lo": lo_all, "mid": 1.0, "hi": hi_all}, "measured", sorted(set(people_clips) |
                     {c for f in FRAMINGS for c in src[f]}),
                     "factor on metric coordinates (H is built at the 4 m prior, mid = 1). Union of the per-camera "
                     f"intervals: camera 9 x{scale[9]['factor'][0]}-{scale[9]['factor'][1]}, camera 14 "
                     f"x{scale[14]['factor'][0]}-{scale[14]['factor'][1]} (twin/cal/scale_v1.json); run the robot at "
                     "both ends"),
        "cal_sources": sorted(set(people_clips) | {c for f in FRAMINGS for c in src[f]} | set(click_clips)
                              | {c for v in p4_clips.values() for c in v}),
        "framings": framings_out,
        "zones": [
            {"id": "z9", "camera": 9, "framings": ["c9_f0", "c9_f1"],
             "floor_polygon": tag(rnd(z9), "traced_v1", src["c9_f0"] + src["c9_f1"],
                                  f"union of the c9_f0 and c9_f1 visible floor (twin/cal/site_trace_v1.json), cut at "
                                  f"{far:g} m from the camera nadir; origin = corner of the yellow L by the walkway")},
            {"id": "z14", "camera": 14, "framings": ["c14_f0"],
             "floor_polygon": tag(rnd(z14), "traced_v1", src["c14_f0"],
                                  f"c14_f0 visible floor, cut at {far:g} m from the camera nadir; shifted by x + {off:g} m "
                                  "so the two zones do not overlap (no shared floor, P7)")}],
        "presses": presses_out}
    model = SS.Site.model_validate(site)
    # twin.schemas.prereg.active_split_path resolves the ACTIVE_SPLIT entry against the repo root, but the pointer
    # holds a name inside prereg/ (as intercycle.esk reads it); pass the split explicitly (reported to the lead)
    from twin.schemas.prereg import parse_split

    split_rows = parse_split(esk.active_split_path(ROOT).read_text(encoding="utf-8"))
    n = PV.assert_train_only(PV.source_clips(site=model), split_rows)
    header = ("# twin/site.yaml - CAL v1 site geometry (lane S2, contract twin.site/1, twin/schemas/site.py).\n"
              "# Written by scripts/cal_v1.py site; method, points and residuals in docs/CAL.md.\n"
              "# CAL points placed by a Claude agent from train keyframes (vision), not by a human.\n"
              "# Every source clip is an A1+ TRAIN id (twin/provenance.py checked "
              f"{n} ids). Plan frame: x, z in metres; pixels in the native 1920x1080 RAW image.\n")
    (ROOT / "twin" / "site.yaml").write_text(header + yaml.safe_dump(site, sort_keys=False, width=120,
                                                                     default_flow_style=None), encoding="utf-8")

    # numbers for docs/CAL.md and other lanes
    summary = {"_doc": "CAL v1 results (lane S2): per-framing orientation, H at 4 m, bootstrap spread, per-press "
                       "dilation_m. Inputs: cal_points_v1.json, k1_v1.json, site_trace_v1.json; train clips only.",
               "n_boot": N_BOOT, "seed": SEED, "zone14_offset_x_m": off, "framings": {}, "presses": press_meta,
               "check_R2_c9": fit["check_R2_c9"],
               "c14_f0_k1_alt_check": {"k1_norm_fit": k1_doc["c14_f0"]["k1_norm"],
                                       "k1_norm_alt": k1_doc[alt_f]["k1_norm"], "alt_from": alt_f,
                                       "theta_deg": [round(nominal["c14_f0"]["orientation"]["theta_deg"], 3),
                                                     round(c14_alt["orientation"]["theta_deg"], 3)],
                                       "phi_deg": [round(nominal["c14_f0"]["orientation"]["phi_deg"], 3),
                                                   round(c14_alt["orientation"]["phi_deg"], 3)],
                                       "press_centroid_shift_m": {p: m["centroid_shift_with_c9_k1_m"]
                                                                  for p, m in press_meta.items()
                                                                  if "centroid_shift_with_c9_k1_m" in m}},
               "cal_sources_extra": {"click_sheet_clips": pts_doc["_click_sheet_clips"], "p4_clips": p4_clips}}
    for f in FRAMINGS:
        b = nominal[f]
        th = np.array([x["orientation"]["theta_deg"] for x in boots[f]])
        ph = np.array([x["orientation"]["phi_deg"] for x in boots[f]])
        poly = np.asarray(trace["floor"][f], float)
        dd = C.displacement_raw_m(b, boots[f], poly)
        hh = np.array([x["h"] for x in boots[f]])
        summary["framings"][f] = {
            "theta_deg": round(b["orientation"]["theta_deg"], 3), "phi_deg": round(b["orientation"]["phi_deg"], 3),
            "relation_resid_deg": [round(v, 3) for v in b["orientation"]["resid_deg"]],
            "theta_boot90_deg": [round(float(np.percentile(th, 5)), 2), round(float(np.percentile(th, 95)), 2)],
            "phi_boot90_deg": [round(float(np.percentile(ph, 5)), 2), round(float(np.percentile(ph, 95)), 2)],
            "floor_vertex_disp_p95_m": round(float(np.percentile(dd, 95)), 3),
            "k1_boot_range": [float(min(x["k1"] for x in boots[f])), float(max(x["k1"] for x in boots[f]))],
            "h_boot_p5_p50_p95_m": [_pct(hh, 5), _pct(hh, 50), _pct(hh, 95)],
            "corr_tilt_h": round(float(np.corrcoef(th, hh)[0, 1]), 3),
            "H_img_to_floor": rnd(b["H"], 10),
            "points_floor_xz_m": {k: [round(float(v), 3) for v in C.apply_h(b["H"], b["undist"][k][None])[0]]
                                  for k in b["undist"]},
            "relations": pts_doc[f]["relations"]}
    (CAL_DIR / "cal_v1.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    sc = {"_doc": "CAL v1 scale (lane S2), expressed as camera height h (ground scale is proportional to h; factor = "
                  "h / 4.0 m prior; H is built at 4.0 m). Orientation (tilt, roll) is fixed by the agent-placed floor "
                  "relations (cal_points_v1.json); h from (a) RF-DETR-M person boxes (conf >= 0.5, upright: height "
                  ">= 80 px and >= 2x width, not at the border) on full-res frames at 1/3 and 2/3 of every A1+ TRAIN "
                  "clip, walker 1.72 m, per-box solve, median; (b) painted stripes ASSUMED 10 cm wide (both edges "
                  "traced on the train median background; stripes_flagged_out left out of the median). JOINT "
                  "bootstrap (scripts/cal_v1.py joint_bootstrap, 200 draws): in every draw the people and stripe "
                  "heights are re-solved at THAT draw's k1, tilt and roll (people: clip-resampled; stripes: edge "
                  "points resampled), and the draw's h is uniform over the span of {4 m prior, people, stripes}. "
                  "h_interval_m = [min(prior, p5 of each estimate), max(prior, p95 of each estimate)]. The nominal_* "
                  "fields are the single-orientation estimates (the P6 > 20 % disagreement rule is read on them). "
                  "Rule (PLAN_TWIN §3 P6): estimates disagree by > 20 % -> publish the interval and run the robot at "
                  "both ends. Supersedes scale_v0.json (INVALID: fitted before A1+; its camera-14 4.0-5.93 m used "
                  "13 clips that are now TEST, so it is not reused).",
          "supersedes": "scale_v0.json", "camera_9": scale[9], "camera_14": scale[14],
          "people_clips": people_clips}
    (CAL_DIR / "scale_v1.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"scale": {k: {kk: vv for kk, vv in v.items() if kk != "nominal_stripes_h_m"}
                                for k, v in scale.items()},
                      "presses": press_meta, "zone14_offset": off,
                      "z9_extent": [z9.min(0).round(1).tolist(), z9.max(0).round(1).tolist()],
                      "z14_extent": [z14.min(0).round(1).tolist(), z14.max(0).round(1).tolist()],
                      "framings": {f: {k: v for k, v in s.items() if k not in ("H_img_to_floor", "relations")}
                                   for f, s in summary["framings"].items()}}, indent=1))
    print("provenance: train-only ids", n)
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gather")
    g.add_argument("--bg-only", action="store_true", help="median backgrounds only (CPU, no detector)")
    g.add_argument("--write-bg", action="store_true", help="with the detector: also rewrite the backgrounds")
    sub.add_parser("k1")
    sub.add_parser("fit")
    sub.add_parser("site")
    a = ap.parse_args(argv)
    return {"gather": cmd_gather, "k1": cmd_k1, "fit": cmd_fit, "site": cmd_site}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
