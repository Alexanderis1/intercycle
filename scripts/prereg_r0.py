"""r0 for gate H4 (PREREG v1, prereg/PREREG_TWIN.md §4): set from the 20 blind DEV frames only.

    python scripts/prereg_r0.py [--out prereg/r0_v1.json]

Rule (frozen in PREREG v1; nothing here reads test data):
* reference = labeller B's blind set `labels/dev/labels_blind_B.json` (the 20 blind dev frames, U14 rule 1), on the
  frames that `labels/dev/scored.json` marks `scored` (19: dev-f-c9-14 is a byte-and-PTS duplicate of dev-f-c9-03);
  every `person` box counts (occluded and truncated included), as on the test labels;
* detections = the dev PERC ingest's stored detector boxes `$INTERCYCLE_DATA/boxes/<clip>.parquet`
  (`intercycle.perception.esk_ingest`, the same per-clip code the sealed run uses), class `person`, not `suppressed`,
  conf >= DET_CONF = 0.4 (= `TrackerConfig.activation`, the confidence at which a detection starts a track), on the
  sampled frame nearest to the label's PTS; if that frame is more than MAX_DT_S away, the frame has no detection
  (fail closed: all its labelled people are misses);
* matching = `scripts/label_agreement.match`: greedy one-to-one by IoU, highest first, IoU >= 0.5 (PLAN_TWIN §5);
* recall = matched / labelled person boxes (pooled over the frames);
* r0 = the one-sided 95 % Clopper-Pearson lower bound of that recall = Beta.ppf(0.05, k, n - k + 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from label_agreement import match  # noqa: E402

from twin.schemas.labels import LabelSet  # noqa: E402

DEV = ROOT / "labels" / "dev"
DET_CONF = 0.4
MAX_DT_S = 0.05
LEVEL = 0.95
CONTRACT = "intercycle.prereg_r0/1"


@dataclass(frozen=True)
class Det:
    """A detector box in the shape `label_agreement.match` reads (cls, x0, y0, x1, y1)."""
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float


def cp_lower_one_sided(k: int, n: int, level: float = LEVEL) -> float:
    from scipy.stats import beta

    if n <= 0:
        raise ValueError("no labelled person")
    return 0.0 if k == 0 else float(beta.ppf(1 - level, k, n - k + 1))


def detections(boxes_dir: Path, clip_id: str, t_pts: float) -> tuple[list[Det], float]:
    import pyarrow.parquet as pq

    p = boxes_dir / f"{clip_id}.parquet"
    if not p.is_file():
        raise FileNotFoundError(f"{p}: no PERC boxes for dev clip {clip_id}")
    rows = pq.read_table(p).to_pylist()
    if not rows or any(r["split"] != "dev" for r in rows):
        raise ValueError(f"{p}: not a dev-only box table")
    ts = sorted({float(r["t"]) for r in rows})
    t_near = min(ts, key=lambda t: abs(t - t_pts))
    dt = abs(t_near - t_pts)
    if dt > MAX_DT_S:
        return [], dt
    out = [Det("person", float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
           for r in rows if float(r["t"]) == t_near and r["class"] == "person" and not r["suppressed"]
           and float(r["conf"]) >= DET_CONF]
    return out, dt


def run(boxes_dir: Path, dev: Path = DEV) -> dict:
    b_bytes = (dev / "labels_blind_B.json").read_bytes()
    s_bytes = (dev / "scored.json").read_bytes()
    lb = LabelSet.model_validate(json.loads(b_bytes))
    if not lb.blind or lb.proposals_shown:
        raise ValueError("labels_blind_B.json must be blind and without proposals")
    status = {r["sample_id"]: r["status"] for r in json.loads(s_bytes)["frames"] if r["label_set"] == "labels.json"}
    per, k_all, n_all, k_s, n_s = [], 0, 0, 0, 0
    for f in lb.frames:
        gt = [b for b in f.boxes if b.cls == "person"]
        det, dt = detections(boxes_dir, f.clip_id, f.t_pts)
        m = match(gt, det)  # type: ignore[arg-type]  # Det has the fields match reads
        st = status.get(f.sample_id)
        if st not in ("scored", "duplicate"):
            raise ValueError(f"{f.sample_id}: status {st!r} in scored.json")
        per.append(dict(sample_id=f.sample_id, clip_id=f.clip_id, status=st, n_gt=len(gt), n_det=len(det),
                        tp=len(m), dt_s=dt))
        k_all, n_all = k_all + len(m), n_all + len(gt)
        if st == "scored":
            k_s, n_s = k_s + len(m), n_s + len(gt)
    r0 = cp_lower_one_sided(k_s, n_s)
    return dict(contract=CONTRACT, rule=(__doc__ or "").split("Rule", 1)[-1].strip(), det_conf=DET_CONF, max_dt_s=MAX_DT_S,
                level=LEVEL, labels_blind_B_sha256=hashlib.sha256(b_bytes).hexdigest(),
                scored_sha256=hashlib.sha256(s_bytes).hexdigest(),
                scored=dict(frames=sum(p["status"] == "scored" for p in per), tp=k_s, n_gt=n_s,
                            recall=k_s / n_s, r0_cp_lower_one_sided_95=r0),
                all_20=dict(frames=len(per), tp=k_all, n_gt=n_all, recall=k_all / n_all,
                            cp_lower_one_sided_95=cp_lower_one_sided(k_all, n_all)),
                r0=round(r0, 4), per_frame=per)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", default=str(Path(os.environ.get("INTERCYCLE_DATA", "data/esk")) / "boxes"))
    ap.add_argument("--out", default=str(ROOT / "prereg" / "r0_v1.json"))
    a = ap.parse_args(argv)
    rep = run(Path(a.boxes))
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rep[k] for k in ("scored", "all_20", "r0")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
