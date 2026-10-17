"""Phase-convention check (R2's request): does the phase the robot believes (observer_bridge.jsonl, from the verified
broadcast) match the true cycle phase, with map phase 0 = start of motion in the zone?

    .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/phasecheck.py out/robot/<batch>/<tag>

Truth: the episode World rebuilt locally with the freezes the plant logged (plant_summary.json), then
Kinematics.cycle_phase(mid, t) = (machine time - window start) / window length, window start = motion start.
Reports per machine the circular phase error in map bins (16 per cycle): median |error| and the share within +-1 bin
(the bridge pads +-1 bin), plus the envelope coverage of the true moving parts by the predicted bins' boxes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def check(d: Path, map_path: Path | None = None) -> dict:
    from intercycle import scenario as S
    from intercycle.observer import mapping as M
    from intercycle.sim.world import Freeze, World
    from intercycle.truth.kinematics import Kinematics

    summ = json.loads((d / "plant_summary.json").read_text())
    scn = S.load()
    t0 = float(summ["plant_t0"])
    world = World(scn, int(summ["seed"]), t0, t0 + float(summ["sim_s"]) + 120)
    for g, lst in (summ.get("freezes") or {}).items():
        for a, b, kind, mid in lst:
            world.groups[g].freezes.append(Freeze(a, b, kind, mid))
    kin = Kinematics(scn, world)
    lm = M.LayeredMap.load(map_path or REPO / "out" / "layered_map.npz")
    rows = [json.loads(line) for line in (d / "observer_bridge.jsonl").read_text().splitlines() if line.strip()]
    nb = M.N_PHASE
    err: dict[str, list[float]] = {}
    cov: dict[str, list[float]] = {}
    ids = lm.machine_ids()
    for r in rows[::4]:
        t = float(r["t"])
        for mid, mv in (r.get("machines") or {}).items():
            ph = mv.get("phase")
            if ph is None:
                continue
            kind, _ = kin.phase(mid, t)
            if kind in ("hold", "trip"):
                continue
            tp = kin.cycle_phase(mid, t)
            e = ((ph - tp + 0.5) % 1.0 - 0.5) * nb
            err.setdefault(mid, []).append(float(e))
            if mid in ids:
                b = int(ph * nb) % nb
                bins = tuple(sorted({(b + k) % nb for k in (-1, 0, 1)}))
                boxes = lm.boxes_for(mid, bins)
                c = kin.bodies(t).get(mid)
                if c is not None and len(c) and len(boxes):
                    inside = np.zeros(len(c), bool)
                    for x0, z0, x1, z1 in boxes:
                        inside |= (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= z0) & (c[:, 1] <= z1)
                    cov.setdefault(mid, []).append(float(inside.mean()))
    out = {}
    for mid, e in err.items():
        a = np.abs(np.array(e))
        out[mid] = {"n": len(e), "median_abs_bins": round(float(np.median(a)), 2),
                    "within_1_bin": round(float((a <= 1.0).mean()), 3), "mean_signed_bins": round(float(np.mean(e)), 2),
                    "part_centres_covered": round(float(np.mean(cov[mid])), 3) if mid in cov else None}
    return out


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None), indent=1))
