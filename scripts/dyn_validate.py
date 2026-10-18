"""DYN validation report (lane S3a, PLAN_TWIN §5): twin vs real DEV tracks, speed and stroke-interval KS only.

Twin side: `python -m twin.daygen --validate <json>` (generated days -> validation camera -> tracks, with the
dataset's selection rule; see `twin.daygen.validation_sample`). This script reads that json (it never imports the
generator: twin.daygen is a truth module), computes the same statistics on the real TRAIN and DEV tables, and writes
docs/DYN_VALIDATION.md.

Rule (PLAN §5): a row passes when KS(twin, dev) <= 1.5 x the train-vs-dev noise floor KS(train, dev) AND
KS(twin, dev) <= the 95th percentile of KS(bootstrap of train tracks, dev) (train resampled by clip, B draws).

    python -m twin.daygen --validate out/dyn_validation/twin_stats.json
    python scripts/dyn_validate.py --twin out/dyn_validation/twin_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle import esk  # noqa: E402
from intercycle.perception.esk_ingest import project_tracks  # noqa: E402
from twin import dynfit as DF  # noqa: E402
from twin.schemas import Site, load  # noqa: E402

REPORT = ROOT / "docs" / "DYN_VALIDATION.md"
FLOOR_FACTOR = 1.5
B_BOOT = 500
BOOT_SEED = 20261018


def sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


COVERAGE: dict[str, dict[str, float]] = {}       # split -> "<zone>/<cls>" -> share of track rows on the traced floor


def real_speeds(split: str, site: Site) -> dict[str, list[tuple[str, float]]]:
    """{"<zone>/<cls>": [(clip id, per-piece speed)]} with the twin's track statistic. Also records in COVERAGE the
    share of projected track rows that fall on the zone's traced floor polygon."""
    cov: dict[str, list[int]] = {}
    data = esk.data_root()
    H = {f.id: f.H_img_to_floor.value for f in site.framings}
    K1 = {f.id: float(f.k1.value) for f in site.framings}
    zone_of_cam = {z.camera: z for z in site.zones}
    out: dict[str, list[tuple[str, float]]] = {}
    for cid, cam in zip(*[esk.load(split)[c] for c in ("clip_id", "camera")]):
        zone = zone_of_cam[int(cam)]
        tr = pd.read_parquet(data / "tracks" / f"{cid}.parquet")
        if len(tr) == 0:
            continue
        tr = project_tracks(tr, H, K1)
        for key, g in tr.groupby(["class", "track_id"]):
            cls = str(cast(tuple, key)[0])
            c = cov.setdefault(f"{zone.id}/{cls}", [0, 0])
            c[0] += int(DF.inside(g["x"].to_numpy(float), g["y"].to_numpy(float), zone.floor_polygon.value).sum())
            c[1] += len(g)
            for pc in DF.observed_pieces(g["t"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy(),
                                         zone.floor_polygon.value, cls, DF.MIN_STAT_S):
                v = DF.track_speed(*pc)
                if math.isfinite(v):
                    out.setdefault(f"{zone.id}/{cls}", []).append((str(cid), v))
    COVERAGE[split] = {k: v[0] / v[1] for k, v in sorted(cov.items()) if v[1]}
    return out


def real_intervals(split: str, site: Site) -> dict[str, list[tuple[str, float]]]:
    data = esk.data_root()
    ids = set(esk.load(split)["clip_id"])
    out: dict[str, list[tuple[str, float]]] = {}
    for p in site.presses:
        for f in sorted((data / "machines" / p.id).glob("*.parquet")):
            if f.stem not in ids:
                continue
            m = pd.read_parquet(f, columns=["t", "stroke", "missing"])
            comp, _ = DF.stroke_intervals(m["t"].to_numpy(float), m["stroke"].to_numpy(bool),
                                          m["missing"].to_numpy(bool))
            out.setdefault(p.id, []).extend((f.stem, float(v)) for v in comp)
    return out


def boot_ks(train: list[tuple[str, float]], dev: np.ndarray, rng: np.random.Generator, b: int = B_BOOT) -> np.ndarray:
    """KS(bootstrap of train resampled by clip, dev), b draws."""
    by: dict[str, list[float]] = {}
    for c, v in train:
        by.setdefault(c, []).append(v)
    clips = sorted(by)
    out = np.empty(b)
    for i in range(b):
        pick = rng.integers(len(clips), size=len(clips))
        out[i] = DF.ks_2samp(np.concatenate([by[clips[k]] for k in pick]), dev)
    return out


def evaluate(twin: dict[str, list[float]], train: dict[str, list[tuple[str, float]]],
             dev: dict[str, list[tuple[str, float]]], rng: np.random.Generator) -> list[dict]:
    rows = []
    for key in sorted(set(twin) | set(train) | set(dev)):
        tw = np.asarray(twin.get(key, []), float)
        tr = train.get(key, [])
        dv = np.asarray([v for _, v in dev.get(key, [])], float)
        row = {"stat": key, "n_twin": len(tw), "n_train": len(tr), "n_dev": len(dv),
               "n_dev_clips": len({c for c, _ in dev.get(key, [])}), "n_train_clips": len({c for c, _ in tr})}
        if len(tw) == 0 or len(tr) == 0 or len(dv) == 0:
            row.update({"verdict": "n/a (no data on one side)"})
            rows.append(row)
            continue
        trv = np.asarray([v for _, v in tr], float)
        ks_twin, ks_floor = DF.ks_2samp(tw, dv), DF.ks_2samp(trv, dv)
        bk = boot_ks(tr, dv, rng)
        q95 = float(np.percentile(bk, 95))
        p1 = ks_twin <= FLOOR_FACTOR * ks_floor
        p2 = ks_twin <= q95
        row.update({"ks_twin_dev": ks_twin, "ks_train_dev": ks_floor, "limit_floor": FLOOR_FACTOR * ks_floor,
                    "ks_boot_q95": q95, "ks_boot_median": float(np.median(bk)), "pass_floor": p1, "pass_boot": p2,
                    "verdict": "PASS" if p1 and p2 else "FAIL",
                    "median_twin": float(np.median(tw)), "median_train": float(np.median(trv)),
                    "median_dev": float(np.median(dv))})
        rows.append(row)
    return rows


def fmt(x, n=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", required=True, help="twin-side json from `python -m twin.daygen --validate`")
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args()
    twin = json.loads(Path(a.twin).read_text(encoding="utf-8"))
    site = load(Site, ROOT / "twin" / "site.yaml")
    rng = np.random.default_rng(BOOT_SEED)
    sp_rows = evaluate(twin["speeds"], real_speeds("train", site), real_speeds("dev", site), rng)
    iv_rows = evaluate(twin["intervals"], real_intervals("train", site), real_intervals("dev", site), rng)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    hashes = {p: sha_lf(ROOT / p) for p in ("twin/daygen.py", "twin/dynfit.py", "twin/rng.py", "twin/dynamics.json",
                                            "twin/dyn/press_v1.json", "twin/dyn/valcam_v1.json")}
    res = {"speeds": sp_rows, "intervals": iv_rows, "hashes": hashes, "library_sha256": twin["library_sha256"],
           "twin": {k: twin[k] for k in ("seeds", "day", "windows_per_day", "n_windows", "footage_unsafe_share")},
           "head": head, "floor_coverage": COVERAGE,
           "dynamics": json.loads((ROOT / "twin" / "dynamics.json").read_text(encoding="utf-8"))}
    Path(a.out).write_text(render(res), encoding="utf-8", newline="\n")
    res.pop("dynamics")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8",
                                                newline="\n")
    for r in sp_rows + iv_rows:
        print(r["stat"], r["verdict"], fmt(r.get("ks_twin_dev")), fmt(r.get("limit_floor")), fmt(r.get("ks_boot_q95")))


def limits(res: dict) -> list[str]:
    dyn = res["dynamics"]
    o, s = dyn["observed"], dyn["assumed"]
    press = json.loads((ROOT / "twin" / "dyn" / "press_v1.json").read_text(encoding="utf-8"))["presses"]
    cov = res["floor_coverage"].get("train", {})
    rows = {r["stat"]: r for r in res["speeds"] + res["intervals"]}
    m = re.search(r"scored\.json sha256\(LF\) ([0-9a-f]{64}) from (.+?)\)\.", s["zone.z9.walker_rate_per_h"]["note"])
    if not m:
        raise ValueError("dynamics.json walker_rate note does not record the DEV labels' sha256 and source")
    labels = {"sha": m.group(1), "src": m.group(2)}
    out = [
        "- **Floor coverage.** Only track rows on the zone's traced floor polygon (twin/site.yaml) enter the library "
        "and the statistics (the twin models only that floor; the rest is a wall). Train share of rows on it: "
        + ", ".join(f"{k} {v:.0%}" for k, v in cov.items())
        + ". Operators standing beside the presses mostly fall outside the traced camera-9 floor, and on camera 14 "
        "the floor polygon keeps very few person rows; S2 owns the polygon.",
        f"- **Zone-14 people.** {rows.get('z14/person', {}).get('n_train', 0)} train and "
        f"{rows.get('z14/person', {}).get('n_dev', 0)} dev person pieces from "
        f"{o.get('person.z14.path_library', {}).get('n', 0)} train resampling units: the row has little power at "
        "this n. The FAIL of the first report (KS 0.921, twin median 1.78 m/s) came from the generator, not from "
        "the data or the floor polygon: DAYGEN v1 picked a unit uniformly inside the day's layout state, so a unit "
        "weighed P(state) / units in the state; unit 3_tr6:person:0 (1.81 m/s), the only zone-14 person unit of the "
        "dominant bright state, carried about 61 % of the zone-14 person arrivals (effective 2.3 of 9 units; the "
        "audit's simulation over seeds 900-963: 64 %, 2.1). Fix round 1: state-specific rates (each state's train "
        "units per clip-hour / recall), so every unit of a lighting class is resampled equally often, and the "
        "validation anchors (proportional to arrivals, train lighting mix, anchor unit only) weight every train "
        "unit equally.",
        "- **Generated days vs validation days.** Inside a lighting class every unit is equally likely, but the "
        "lighting periods of a generated day follow the ASSUMED site-wide night share "
        f"({s['site.night_share']['value']:.3f}), not each zone's train dim share: in zone 14, 78 % of the person "
        "units come from dim states, so the 2 bright zone-14 person units carry about 62 % of a generated day's "
        "zone-14 person arrivals (effective 4.8 of 9 units; zone-14 forklifts 27 of 48, zone-9 persons 625 of 643). "
        "The validation removes this time-of-day weighting by drawing its anchors with the train lighting mix.",
        "- **Camera-14 presses** never ran in a judged train clip (P4): P(running) is the Beta posterior of 0 running "
        "clips (mean " + ", ".join(f"{p} {o[f'press.{p}.p_running']['value']:.3f}" for p in ("P14a", "P14b", "P14c", "P14d"))
        + "), the stroke period is the v0 prior (ASSUMED, grey); no dev strokes exist to compare.",
        "- **In-clip pauses.** No complete in-clip interval longer than 1.5 x the period was seen on P9a or P9b, so the "
        "generator draws no observed micro-stops. The Kaplan-Meier mass beyond the last complete interval ("
        + ", ".join(f"{p} P(T > {press[p]['pause_lo_s']} s) = {press[p]['km_s_at_pause_lo']:.3f}" for p in ("P9a", "P9b"))
        + ") is censored at the clip end or at MISSING; it is left to the ASSUMED W-W3 long-stop process.",
        "- **Stroke impulse** (FWHM of the ROI |flow_v| around a stroke): "
        + ", ".join(f"{p} {o[f'press.{p}.stroke_impulse_s']['value']:.2f} s" for p in ("P9a", "P9b"))
        + "; it reaches the +-0.8 s window, so the ROI moves through the whole cycle and this is not a resolved impulse. "
        "The engine's shared ram impulse `press.stroke_impulse_s` (the v0 contract key, read by twin/engine.py) "
        f"therefore stays the v0 prior {s['press.stroke_impulse_s']['value']} s (ASSUMED, sweep "
        f"{s['press.stroke_impulse_s']['sweep']['lo']}-{s['press.stroke_impulse_s']['sweep']['hi']} s).",
        "- **Arrival rates are ASSUMED** and follow the plan's rule (train footage rate / recall): persons z9 "
        f"{s['zone.z9.walker_rate_per_h']['value']:.0f}/h, z14 {s['zone.z14.walker_rate_per_h']['value']:.0f}/h, "
        f"forklifts z14 {s['zone.z14.forklift_rate_per_h']['value']:.0f}/h, z9 0/h (no vehicles observed in zone-9 "
        "train clips). The clips are event-selected, so these reproduce the crowding the clips show, not an average "
        "hour. Person recall comes from the DEV labels (independent Claude-agent labels) against the DEV tracks "
        f"(labels/dev/scored.json sha256 `{labels['sha']}` from `{labels['src']}`, pinned in scripts/dyn_fit.py; "
        "the fit fails on other labels); forklift recall from the P5 agent check (8/13). Each day spreads the zone "
        "rate over the layout states (each state's train units per clip-hour / recall).",
        f"- **Selection rule and the violation share.** The generated days draw unsafe-behaviour units at the ASSUMED "
        f"share {s['person.violation_share']['value']}; the validation days draw them at the train footage share "
        f"({res['twin'].get('footage_unsafe_share', float('nan')):.3f}), because the dataset's clips are selected by "
        "behaviour class. The first trial run (16 worlds, assumed share) failed the zone-9 speed row; the share fix and "
        "64 worlds (each day uses only one bright and one dim layout state per zone) were the changes before the first "
        "report. Fix round 1 (audit) changed the generator's unit weighting and the validation's anchor rule (above) "
        "and nothing else; no dynamics parameter was tuned on dev in either round.",
        "- **Not tested here:** arrival rates, staffing, stops, lighting / time of day and layout-state frequencies "
        "(all ASSUMED or not comparable on short clips), and paths beyond speed. Time of day is ASSUMED: night = the "
        f"dim-luma share ({s['site.night_share']['value']:.3f}) of the 24 h, centred on midnight (P2: no timing signal).",
    ]
    return out


def render(res: dict) -> str:
    def table(rows: list[dict], unit: str) -> list[str]:
        out = [f"| statistic | n twin / train / dev (dev clips) | median twin / train / dev ({unit}) | KS twin-dev | "
               "KS train-dev (floor) | 1.5 x floor | KS boot q95 | verdict |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r['stat']} | {r['n_twin']} / {r['n_train']} / {r['n_dev']} ({r['n_dev_clips']}) | "
                       f"{fmt(r.get('median_twin'), 2)} / {fmt(r.get('median_train'), 2)} / {fmt(r.get('median_dev'), 2)} | "
                       f"{fmt(r.get('ks_twin_dev'))} | {fmt(r.get('ks_train_dev'))} | {fmt(r.get('limit_floor'))} | "
                       f"{fmt(r.get('ks_boot_q95'))} | **{r['verdict']}** |")
        return out

    tw = res["twin"]
    L = ["# DYN validation (DAYGEN v1 vs real DEV tracks)", "",
         "Lane S3a, PLAN_TWIN §5 \"DYN/DAYGEN\". Generated by `scripts/dyn_validate.py` from "
         "`python -m twin.daygen --validate`; numbers below are copied from `docs/DYN_VALIDATION.json`.", "",
         "## Rule", "",
         "- Twin output goes through a camera model into tracks, the dataset's selection rule is applied, and the "
         "result is compared with the real DEV tracks on two statistics only: per-track floor speed and in-clip "
         "stroke interval (two-sample KS).",
         f"- A row passes when KS(twin, dev) <= {FLOOR_FACTOR} x KS(train, dev) (the train-vs-dev noise floor) AND "
         f"KS(twin, dev) <= the 95th percentile of KS(bootstrap of train, dev) ({B_BOOT} bootstrap draws of the train "
         f"clips with replacement, seed {BOOT_SEED}).",
         "- Speed statistic (`twin.dynfit.observed_pieces` + `track_speed`, identical for twin, train and dev): track "
         "projected to the floor with CAL v1, rolling median of 7 samples, cut at the zone's traced floor polygon, at "
         "detection gaps > 1 s and at ID-switch jumps, pieces >= 2 s; speed = median over the piece of the floor "
         "displacement over 1 s.",
         "- Stroke-interval statistic (`twin.dynfit.stroke_intervals`): consecutive strokes with no MISSING frame "
         "between them.", "",
         "## Twin side", "",
         f"- World seeds {tw['seeds'][0]}-{tw['seeds'][-1]} (dev range; NOT the dev worlds 0-3, which PLAN §4 generates "
         f"only after PREREG v1), day {tw['day']}, full 24 h days, on average {tw['windows_per_day']} windows per "
         f"(zone, class) and exactly {tw['windows_per_day']} per press per day; windows kept: {tw['n_windows']}.",
         "- Selection rule (event-selected clips, each of their tracks counted once): a mover window is anchored at "
         "an arrival and reproduces its source clip (start = arrival - the unit's offset in its clip, length = that "
         "clip's duration, framing = that clip's framing); the anchor's unit (a group whole) is filmed and the window "
         "is kept when the camera returns at least one track row. Anchors are an independent thinning of each day's "
         "arrivals per (zone, class): an arrival in lighting class L is kept with probability windows x share_L / "
         "E[arrivals in L per day], share_L = the train units' share in L, so anchors are proportional to arrivals "
         "and every train unit anchors equally often. Other movers inside the window are not counted (counting them "
         "would weight the units of dense layout states by their co-presence). The validation days pick units "
         "uniformly inside the day's state (the train footage unsafe share). A press window takes a random TRAIN "
         "clip of the press (its frame grid and MISSING mask) at a uniform time of the generated day.",
         "- Camera: **validation camera `twin.dynfit.PixelCamera`, not stat_v1** (twin/camera.py, lane S3b, was not "
         "merged when this ran). Floor -> pixels through the inverse CAL v1 H and k1 -> foot-point jitter and a "
         "two-state miss chain fitted on TRAIN tracks (`twin/dyn/valcam_v1.json`) -> back to the floor through the "
         "same CAL chain as the real tracks. Strokes: timing jitter fitted on train intervals, snapped to the clip's "
         "frame grid, dropped on MISSING frames. Re-run this report once stat_v1 lands.", "",
         "## Speed (m/s)", "", *table(res["speeds"], "m/s"), "",
         "## Stroke interval (s)", "", *table(res["intervals"], "s"), "",
         "## What this does and does not show", "", *limits(res), "",
         "## Hashes (sha256 of the LF-normalised file = the git blob)", "",
         f"- HEAD when the report ran: `{res['head']}` (the files below are committed with this report)",
         *[f"- `{p}`: `{h}`" for p, h in res["hashes"].items()],
         f"- path library `$INTERCYCLE_DATA/dyn/path_library_v1.npz` (gitignored, raw bytes): "
         f"`{res['library_sha256']}`",
         "- generator = `twin/daygen.py` + `twin/dynfit.py` + `twin/rng.py`; dynamics = `twin/dynamics.json` + "
         "`twin/dyn/press_v1.json` + the path library.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
