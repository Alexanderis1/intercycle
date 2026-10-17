"""Build $INTERCYCLE_DATA/inventory.parquet (PLAN_TWIN §2) from prereg/split.csv + the public per-clip metadata.

    python scripts/esk_inventory.py --split prereg/split.csv --public public.json

public.json comes from cloud/modal_esk.py run_split (clip_id, camera, cls, sha256, duration from container PTS).
framing_id is provisional (one framing per camera, "c<cam>_f0") until probe P6 fills it.
The result is checked against split.csv with intercycle.esk.check_against_split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's intercycle, not an installed copy
from intercycle import esk  # noqa: E402


def build(split_csv: Path, public: list[dict]) -> pd.DataFrame:
    with open(split_csv, newline="", encoding="utf-8") as fh:
        split = {r["clip_id"]: r for r in csv.DictReader(fh)}
    pub = {p["clip_id"]: p for p in public}
    if set(pub) != set(split):
        raise ValueError("public metadata and split.csv list different clips")
    rows = []
    for cid in sorted(split):
        p, s = pub[cid], split[cid]
        if p["sha256"] != s["sha256"] or int(p["camera"]) != esk.camera_of(cid):
            raise ValueError(f"metadata mismatch for {cid}")
        rows.append({"clip_id": cid, "camera": int(p["camera"]), "framing_id": f"c{int(p['camera'])}_f0",
                     "group_id": s["group_id"], "class": int(p["cls"]), "duration": float(p["duration"]),
                     "sha256": s["sha256"], "split": s["split"]})
    return pd.DataFrame(rows, columns=esk.COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path("prereg/split.csv"))
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    inv = build(a.split, json.loads(a.public.read_text(encoding="utf-8")))
    out = a.out or esk.data_root() / "inventory.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_parquet(out, index=False)
    esk.check_against_split(pd.read_parquet(out), a.split)
    print(f"{out}: {len(inv)} rows, equals {a.split}")


if __name__ == "__main__":
    main()
