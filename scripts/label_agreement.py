"""Inter-labeller agreement on the blind DEV frames (lane S4-labels-blind, U14, PLAN_TWIN §5).

Compares labeller B's blind set (labels/dev/labels_blind_B.json) with labeller A's DEV labels (labels/dev/labels.json,
read only after B's set was committed) on the frames B labelled:

* boxes of the same class are matched one-to-one, greedily by IoU (highest first), a match needs IoU >= 0.5
  (the real-clip test's rule, PLAN_TWIN §5);
* recall of B against A = matched / A boxes; precision of B against A = matched / B boxes;
* foot-point pixel error = Euclidean distance between the two foot points of each matched person pair;
* frame-level Clopper-Pearson 95 % intervals ignore clustering (frames from few clips and groups);
* the same figures for A boxes that are neither occluded nor truncated, and per frame.

A's set is opened through `twin.dev_labels.load_scored`, which checks the hashes of labels.json, the sample lists and
scored.json and fails closed; frame statuses (scored / duplicate) come from that scored set. Nothing is decided here:
the numbers feed r0 (PLAN_TWIN §5), which is chosen elsewhere.

    python scripts/label_agreement.py [--out labels/dev/agreement_blind_B.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.schemas.labels import Box, FrameLabel, LabelSet, SampleList, check_complete  # noqa: E402

DEV = ROOT / "labels" / "dev"
IOU_MIN = 0.5
CONTRACT = "intercycle.label_agreement/1"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(b: Box) -> tuple[float, float, float, float]:
    return (b.x0, b.y0, b.x1, b.y1)


def match(a: list[Box], b: list[Box], thr: float = IOU_MIN) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of same-class boxes by IoU, highest first; pairs (i_a, i_b, iou) with iou >= thr."""
    cand = sorted(((iou(_xyxy(x), _xyxy(y)), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                   if x.cls == y.cls), key=lambda t: (-t[0], t[1], t[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    out = []
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, v))
    return out


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    if n <= 0:
        return (math.nan, math.nan)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return (lo, hi)


def _foot(b: Box) -> tuple[float, float]:
    assert b.foot_u is not None and b.foot_v is not None
    return (b.foot_u, b.foot_v)


def agreement(frames_a: list[FrameLabel], frames_b: list[FrameLabel], thr: float = IOU_MIN,
              keep_a=lambda box: True) -> dict:
    """Pooled and per-frame agreement of B against A. `keep_a` restricts the A boxes counted (B boxes matched to a
    dropped A box are dropped too, so precision is only reported for the full set)."""
    ida = [f.sample_id for f in frames_a]
    idb = [f.sample_id for f in frames_b]
    if ida != idb:
        raise ValueError(f"agreement needs the same frames in the same order: {ida} vs {idb}")
    n_a = n_b = tp = 0
    foot: list[float] = []
    ious: list[float] = []
    per: dict[str, dict] = {}
    for fa, fb in zip(frames_a, frames_b):
        pairs = match(fa.boxes, fb.boxes, thr)
        keep = [i for i, x in enumerate(fa.boxes) if keep_a(x)]
        k_tp = 0
        for i, j, v in pairs:
            if i not in keep:
                continue
            k_tp += 1
            ious.append(v)
            if fa.boxes[i].cls == "person":
                (ua, va), (ub, vb) = _foot(fa.boxes[i]), _foot(fb.boxes[j])
                foot.append(math.hypot(ua - ub, va - vb))
        per[fa.sample_id] = dict(n_a=len(keep), n_b=len(fb.boxes), tp=k_tp)
        n_a += len(keep)
        n_b += len(fb.boxes)
        tp += k_tp
    return dict(n_frames=len(frames_a), n_a=n_a, n_b=n_b, tp=tp,
                recall_b_vs_a=tp / n_a if n_a else math.nan, precision_b_vs_a=tp / n_b if n_b else math.nan,
                recall_ci95=clopper_pearson(tp, n_a), precision_ci95=clopper_pearson(tp, n_b),
                foot_px=foot, iou=ious, per_frame=per)


def _summary(xs: list[float]) -> dict:
    import numpy as np

    if not xs:
        return dict(n=0)
    v = np.asarray(xs, float)
    return dict(n=int(v.size), median=round(float(np.median(v)), 2), mean=round(float(v.mean()), 2),
                p90=round(float(np.percentile(v, 90)), 2), max=round(float(v.max()), 2))


def run(dev: Path = DEV) -> dict:
    from twin.dev_labels import load_scored, sha256_bytes

    scored = load_scored(dev)                                  # hash checks on A's set; fails closed
    status = {r["sample_id"]: r for r in scored.doc["frames"] if r["label_set"] == "labels.json"}
    b_bytes = (dev / "labels_blind_B.json").read_bytes()
    a_bytes = (dev / "labels.json").read_bytes()
    lb = LabelSet.model_validate(json.loads(b_bytes))
    la = LabelSet.model_validate(json.loads(a_bytes))
    samples = SampleList.model_validate(json.loads((dev / "sample.json").read_text(encoding="utf-8")))
    check_complete(lb, samples, require_all=False)
    check_complete(la, samples, require_all=True)
    if not lb.blind or lb.proposals_shown or lb.labeller_id == la.labeller_id:
        raise ValueError("labeller B's set must be blind, without proposals, from a different labeller")
    amap = {f.sample_id: f for f in la.frames}
    fb = list(lb.frames)
    fa = [amap[f.sample_id] for f in fb]
    for x, y in zip(fa, fb):
        if x.clip_id != y.clip_id or abs(x.t_pts - y.t_pts) > 1e-6:
            raise ValueError(f"{x.sample_id}: A and B label different frames")
    sids_scored = [f.sample_id for f in fb if status[f.sample_id]["status"] == "scored"]
    fa_s = [amap[s] for s in sids_scored]
    fb_s = [f for f in fb if f.sample_id in set(sids_scored)]

    def pack(r: dict) -> dict:
        r = dict(r)
        r["foot_px_summary"] = _summary(r.pop("foot_px"))
        r["iou_summary"] = _summary(r.pop("iou"))
        for k in ("recall_b_vs_a", "precision_b_vs_a"):
            r[k] = round(r[k], 4)
        r["recall_ci95"] = [round(x, 4) for x in r["recall_ci95"]]
        r["precision_ci95"] = [round(x, 4) for x in r["precision_ci95"]]
        return r

    clean = lambda b: not b.occluded and not b.truncated     # noqa: E731
    out = dict(
        contract=CONTRACT, iou_min=IOU_MIN, matching="greedy one-to-one by IoU, same class",
        labeller_a=la.labeller_id, labeller_b=lb.labeller_id,
        files={"labels.json": sha256_bytes(a_bytes), "labels_blind_B.json": sha256_bytes(b_bytes),
               "scored.json": sha256_bytes((dev / "scored.json").read_bytes())},
        frames=[f.sample_id for f in fb],
        duplicates={s: status[s]["dup_of"] for s in (f.sample_id for f in fb) if status[s]["status"] == "duplicate"},
        all_frames=pack(agreement(fa, fb)),
        scored_frames=pack(agreement(fa_s, fb_s)),
        scored_frames_a_clean={k: v for k, v in pack(agreement(fa_s, fb_s, keep_a=clean)).items()
                               if k not in ("n_b", "precision_b_vs_a", "precision_ci95")},   # recall only
        a_box_flags=dict(occluded=sum(b.occluded for f in fa_s for b in f.boxes),
                         truncated=sum(b.truncated for f in fa_s for b in f.boxes)),
        b_box_flags=dict(occluded=sum(b.occluded for f in fb_s for b in f.boxes),
                         truncated=sum(b.truncated for f in fb_s for b in f.boxes)),
    )
    # A-only and B-only boxes on scored frames, by position (for the write-up; no image content)
    misses = []
    for x, y in zip(fa_s, fb_s):
        pairs = match(x.boxes, y.boxes)
        ma = {i for i, _, _ in pairs}
        mb = {j for _, j, _ in pairs}
        best = lambda box, others: max((iou(_xyxy(box), _xyxy(o)) for o in others), default=0.0)  # noqa: E731
        misses += [dict(frame=x.sample_id, who="A_only", box=[round(v) for v in _xyxy(b)], occluded=b.occluded,
                        truncated=b.truncated, best_iou=round(best(b, y.boxes), 3))
                   for i, b in enumerate(x.boxes) if i not in ma]
        misses += [dict(frame=y.sample_id, who="B_only", box=[round(v) for v in _xyxy(b)], occluded=b.occluded,
                        truncated=b.truncated, best_iou=round(best(b, x.boxes), 3))
                   for j, b in enumerate(y.boxes) if j not in mb]
    out["unmatched_scored"] = misses
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=str(DEV / "agreement_blind_B.json"))
    a = ap.parse_args(argv)
    res = run()
    Path(a.out).write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: res[k] for k in ("all_frames", "scored_frames", "scored_frames_a_clean")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
