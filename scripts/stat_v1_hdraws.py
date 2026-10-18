"""Export the CAL v1 joint-bootstrap draws of H for the twin camera stat_v1 (lane S3b; PLAN_TWIN §2 twin/camera.py
"projected through one H bootstrap draw per world").

    python scripts/stat_v1_hdraws.py            # -> twin/cal/h_draws_v1.json

`scripts/cal_v1.py site` builds 200 joint draws per camera (k1, tilt, roll and camera height drawn together; the
framings of one camera share draw i) but keeps only their summaries. This script re-runs exactly that bootstrap
(`cal_v1.joint_bootstrap`, same seeds, same TRAIN-only inputs in $INTERCYCLE_DATA/cal/) and writes, per framing, the
nominal camera and every draw as (k1, tilt, roll, h, H). It fails closed unless:
  * the nominal H of every framing equals the one in twin/site.yaml (to 1e-6 relative), and
  * the re-run scale record equals twin/cal/scale_v1.json (h interval and draw percentiles),
so the exported draws are the published CAL v1 draws, not a new bootstrap.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "twin" / "cal" / "h_draws_v1.json"


def _cal_v1():
    spec = importlib.util.spec_from_file_location("cal_v1", ROOT / "scripts" / "cal_v1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(b: dict) -> dict:
    H = np.asarray(b["H"], float)
    return {"k1": float(b["k1"]), "theta_deg": float(b["orientation"]["theta_deg"]),
            "phi_deg": float(b["orientation"]["phi_deg"]), "h_m": float(b["h"]),
            "H": [[float(v) for v in row] for row in (H / H[2, 2])]}


def main() -> int:
    import yaml

    cv = _cal_v1()
    pts_doc = cv.load_json(cv.CAL_DIR / "cal_points_v1.json")
    k1_doc = cv.load_json(cv.CAL_DIR / "k1_v1.json")
    hints = cv.load_json(cv.CAL_DIR / "k1_hints_v1.json")
    cal_doc = cv.load_json(cv.CAL_DIR / "cal_v1.json")
    fit = cv.load_json(cv.data_cal() / "fit_v1.json")
    persons = cv.load_json(cv.data_cal() / "persons_train.json")
    site = yaml.safe_load((ROOT / "twin" / "site.yaml").read_text(encoding="utf-8"))
    site_H = {f["id"]: np.asarray(f["H_img_to_floor"]["value"], float) for f in site["framings"]}

    cals = {f: cv.framing_cal(f, pts_doc, k1_doc, hints)[0] for f in cv.FRAMINGS}
    cals["c14_f0"].offset_xz = (float(cal_doc["zone14_offset_x_m"]), 0.0)
    nominal = {f: cals[f].build() for f in cv.FRAMINGS}
    for f, b in nominal.items():
        H = np.asarray(b["H"], float) / b["H"][2, 2]
        Hs = site_H[f] / site_H[f][2, 2]
        err = float(np.max(np.abs(H - Hs)) / np.max(np.abs(Hs)))
        if err > 1e-6:
            raise SystemExit(f"{f}: nominal H differs from twin/site.yaml (rel {err:.2e}): not the published CAL v1")
    draws_h, draws_4, scale = cv.joint_bootstrap(cals, persons, hints, fit)
    pub = cv.load_json(cv.CAL_DIR / "scale_v1.json")
    for cam, rec in scale.items():
        want = pub[f"camera_{cam}"]
        for k in ("h_interval_m", "draw_h_p5_p50_p95_m"):
            if not np.allclose(rec[k], want[k], atol=1e-3):
                raise SystemExit(f"camera {cam} {k}: re-run {rec[k]} != published {want[k]}: not the CAL v1 draws")
    out = {
        "_doc": ("CAL v1 joint-bootstrap draws (scripts/cal_v1.py joint_bootstrap re-run by scripts/stat_v1_hdraws.py; "
                 "same seeds, TRAIN-only inputs; checked against twin/site.yaml H and twin/cal/scale_v1.json). Per "
                 "framing: the nominal camera (h = 4 m prior) and n draws (k1 px^-2 division model, tilt / roll deg, "
                 "camera height h m, H = undistorted px -> plan floor (x, z) m); `draws_fixed_scale` = the same draws at the 4 m "
                 "prior (scale left to the site scale interval). The framings of one camera share draw "
                 "i (one mount). Used by twin/camera.py stat_v1: one draw per world is that world's true camera."),
        "source": "cal_v1", "n": len(next(iter(draws_h.values()))),
        "cameras": {str(c): list(fs) for c, fs in cv.CAM_FRAMINGS.items()},
        "framings": {f: {"nominal": _rec(nominal[f]), "draws": [_rec(d) for d in draws_h[f]],
                          "draws_fixed_scale": [_rec(d) for d in draws_4[f]]} for f in cv.FRAMINGS},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({out['n']} draws x {len(cv.FRAMINGS)} framings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
