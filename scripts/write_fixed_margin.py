"""Write twin/m7_margin.json: the fixed latency-p99 margin (PREREG_TWIN section 7, the fallback of PLAN_TWIN section 6).

    .venv/Scripts/python.exe scripts/write_fixed_margin.py --runs RUN_DIR [RUN_DIR ...] --out twin/m7_margin.json

margin: k = ceil(age_p99 * v_max / cell_m - 1e-9), margin_m = round(k * cell_m, 2), with
  * age_p99: the 99th percentile (numpy linear) of `fast_age_s` over every tick of every RUN_DIR's observer_bridge.jsonl
    (the fast-layer age the robot holds, robot side). PREREG: every tick of every learned-arm mission on dev worlds
    0-3, days 2-8, zones 9 and 14. A tick with no fast layer (fast_age_s null) is left out and counted;
  * v_max: the largest sweep.hi of the three mover speeds of twin/dynamics.json (--dynamics);
  * cell_m: 0.25 m, the observer's env_res.
Fail closed, no clipping: the margin must be on the H3 grid (0-4 m, 5 cm steps); with --grid-margin-m G it must also be
<= G (PREREG section 7 (b) names 1.0). The writer refuses to overwrite a frozen file and never runs inside a sealed run.
It reads dev run logs only: choosing the RUN_DIRs (the dev runs of the PREREG scope) is the caller's job and is recorded
in the file's method.inputs.age_p99.source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from intercycle.eval import conformal_margin as CM  # noqa: E402

BRIDGE_FILE = "observer_bridge.jsonl"


def read_ages(run_dirs: list[Path]) -> tuple[np.ndarray, int, dict]:
    """(ages of every tick with a fast layer, number of ticks without one, source block) over the runs' bridge logs."""
    ages: list[float] = []
    n_none = 0
    runs = []
    for d in run_dirs:
        p = Path(d) / BRIDGE_FILE
        if not p.is_file():
            raise FileNotFoundError(f"{p}: no {BRIDGE_FILE} (is this a learned-arm mission directory?)")
        raw = p.read_bytes()
        n0, nn0 = len(ages), n_none
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            v = json.loads(line).get("fast_age_s")
            if v is None:
                n_none += 1
            else:
                ages.append(float(v))
        runs.append(dict(dir=str(d), file=BRIDGE_FILE, sha256=hashlib.sha256(raw).hexdigest(),
                         n_ticks=len(ages) - n0, n_ticks_no_fast_layer=n_none - nn0))
    src = dict(description="fast_age_s of observer_bridge.jsonl, pooled over every tick of the named learned-arm runs",
               scope_required="learned-arm missions on dev worlds 0-3, days 2-8, zones 9 and 14 (PREREG_TWIN section 7)",
               runs=runs)
    return np.asarray(ages, float), n_none, src


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="learned-arm dev mission directories")
    ap.add_argument("--out", type=Path, default=ROOT / CM.DEFAULT_OUT)
    ap.add_argument("--dynamics", type=Path, default=CM.DYNAMICS_JSON)
    ap.add_argument("--cell-m", type=float, default=CM.FIXED_CELL_M)
    ap.add_argument("--grid-margin-m", type=float, default=None,
                    help="refuse a margin above this observer grid margin (PREREG section 7 (b): 1.0); default: none")
    a = ap.parse_args(argv)
    if os.environ.get("INTERCYCLE_SEALED_RUN") == "1":
        raise SystemExit("the fixed margin is computed from dev runs only, never inside a sealed run")
    ages, n_none, src = read_ages(a.runs)
    v_max, v_src = CM.v_max_from_dynamics(a.dynamics)
    res = CM.fixed_margin_result(ages, age_source=src, v_max_mps=v_max, v_max_source=v_src, cell_m=a.cell_m,
                                 n_ticks_no_fast_layer=n_none, grid_margin_m=a.grid_margin_m)
    p = CM.write_fixed_margin_file(res, a.out)
    m = res["method"]
    print(f"wrote {p}: margin_m={res['margin_m']} (k={m['k']}, age_p99={m['age_p99_s']:.4f} s over "
          f"{len(ages)} ticks, {n_none} without a fast layer, v_max={v_max} m/s, cell_m={a.cell_m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
