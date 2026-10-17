"""Build $INTERCYCLE_DATA/inventory.parquet (PLAN_TWIN §2) from the active split + the public per-clip metadata.

    python scripts/esk_inventory.py --public public.json [--split prereg/split_a1.csv]

The split defaults to the active one (prereg/ACTIVE_SPLIT, amendment A1+). public.json comes from cloud/modal_esk.py
run_split (clip_id, camera, cls, sha256, duration from container PTS). framing_id comes from the split csv when it has
that column (split_a1.csv); otherwise it is the provisional "c<cam>_f0". Extra columns ``dup_of`` / ``dup_classes`` mark
byte-identical files (same sha256) and their class labels when those disagree. The result is checked against the split
csv with intercycle.esk.check_against_split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's intercycle, not an installed copy
from intercycle import esk  # noqa: E402


def build(split_csv: Path, public: list[dict]) -> pd.DataFrame:
    with open(split_csv, newline="", encoding="utf-8") as fh:
        split = {r["clip_id"]: r for r in csv.DictReader(fh)}
    pub = {p["clip_id"]: p for p in public}
    if set(pub) != set(split):
        raise ValueError("public metadata and the split csv list different clips")
    same: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(split):
        same[split[cid]["sha256"]].append(cid)
    rows = []
    for cid in sorted(split):
        p, s = pub[cid], split[cid]
        if p["sha256"] != s["sha256"] or int(p["camera"]) != esk.camera_of(cid):
            raise ValueError(f"metadata mismatch for {cid}")
        twins = same[s["sha256"]]
        classes = sorted({int(c.split("_")[0]) for c in twins})
        rows.append({"clip_id": cid, "camera": int(p["camera"]),
                     "framing_id": s.get("framing_id") or f"c{int(p['camera'])}_f0",
                     "group_id": s["group_id"], "class": int(p["cls"]), "duration": float(p["duration"]),
                     "sha256": s["sha256"], "split": s["split"],
                     "dup_of": ",".join(c for c in twins if c != cid),
                     "dup_classes": ",".join(str(c) for c in classes) if len(classes) > 1 else ""})
    return pd.DataFrame(rows, columns=esk.COLUMNS + esk.EXTRA_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=None, help="default: the active split (prereg/ACTIVE_SPLIT)")
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    split = a.split or esk.active_split_path()
    inv = build(split, json.loads(a.public.read_text(encoding="utf-8")))
    out = a.out or esk.data_root() / "inventory.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_parquet(out, index=False)
    esk.check_against_split(pd.read_parquet(out), split)
    print(f"{out}: {len(inv)} rows, equals {split}")


if __name__ == "__main__":
    main()
