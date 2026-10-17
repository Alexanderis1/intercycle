"""CAL v0 k1 per framing (probe P6): trace the hinted straight lines on a framing's TRAIN median background, fit k1.

    python scripts/esk_p6_k1.py --bg-dir <dir with <framing>_bg.jpg> [--overlay-dir <dir>] [--write]

Hints: twin/cal/k1_hints.json. Output (with --write): twin/cal/k1_v0.json, one division-model k1 per framing with the
source train clip ids (the clips the median background was built from, from the P6 Modal result --p6-json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intercycle import framing as fr  # noqa: E402


def fit_framing(gray: np.ndarray, hints: dict) -> dict:
    chains = {n: fr.trace_edge(gray, h[0], h[1], **(h[2] if len(h) > 2 else {})) for n, h in hints.items()
              if not n.startswith("_")}
    used = {n: c for n, c in chains.items() if len(c) >= 15}
    rec: dict = {"lines": {n: {"n_pts": int(len(c)), "len_px": round(fr.chain_length(c), 1) if len(c) else 0.0,
                               "rms_raw_px": round(fr.line_rms(c), 3) if len(c) > 2 else None}
                           for n, c in chains.items()}}
    if len(used) < 2:
        rec["k1"] = None
        return rec
    fit = fr.fit_k1(list(used.values()))
    rec.update({"k1": fit["k1"], "k1_norm": fit["k1_norm"], "at_bound": fit["at_bound"],
                "rms_after_px": dict(zip(used, [round(x, 3) for x in fit["rms_after_px"]]))})
    rec["single_line_k1_norm"] = {n: round(fr.fit_k1([c])["k1_norm"], 4) for n, c in used.items()}
    if len(used) >= 3:
        rec["leave_one_out_k1_norm"] = {n: round(fr.fit_k1([c for m, c in used.items() if m != n])["k1_norm"], 4)
                                        for n in used}
    rec["_chains"] = used
    return rec


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-dir", type=Path, required=True)
    ap.add_argument("--hints", type=Path, default=ROOT / "twin" / "cal" / "k1_hints.json")
    ap.add_argument("--p6-json", type=Path, default=None, help="cloud/modal_p6.py result (source clip ids)")
    ap.add_argument("--overlay-dir", type=Path, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    hints = json.loads(a.hints.read_text(encoding="utf-8"))
    p6 = json.loads(a.p6_json.read_text(encoding="utf-8")) if a.p6_json else None
    out: dict = {"_doc": "CAL v0 lens distortion, one division-model k1 per framing (probe P6, PLAN_TWIN §3). "
                         "x_u = c + (x_d - c) / (1 + k1 r_d^2), r_d in px of the 1920x1080 frame, c = image centre; "
                         "k1_norm = k1 * 1330^2. opencv: K and the rational dist vector for cv2.undistortPoints(P=K). "
                         "Sources are TRAIN clip ids only.",
                 "center": list(fr.CENTER), "f_norm": fr.F_PRIOR}
    for fid, h in hints.items():
        if fid.startswith("_"):
            continue
        bg = cv2.imread(str(a.bg_dir / f"{fid}_bg.jpg"))
        if bg is None:
            raise SystemExit(f"missing {fid}_bg.jpg in {a.bg_dir}")
        rec = fit_framing(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), h)
        chains = rec.pop("_chains", {})
        if a.overlay_dir is not None:
            ov = bg.copy()
            for c in chains.values():
                for x, y in c.astype(int):
                    cv2.circle(ov, (int(x), int(y)), 3, (255, 0, 255), -1)
            cv2.imwrite(str(a.overlay_dir / f"{fid}_k1lines.jpg"), ov)
        if rec.get("k1") is not None:
            rec["opencv"] = fr.opencv_coeffs(rec["k1"])
        if p6 is not None:
            rec["source_clip_ids"] = p6["per_framing"][fid]["median_of"]
            rec["ref_clip_id"] = p6["per_framing"][fid]["ref"]
        out[fid] = rec
        print(fid, json.dumps({k: v for k, v in rec.items() if k not in ("opencv", "source_clip_ids")}))
    if a.write:
        dst = ROOT / "twin" / "cal" / "k1_v0.json"
        dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", dst)


if __name__ == "__main__":
    main()
