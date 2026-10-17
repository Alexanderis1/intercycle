"""Write probe P6 framing ids (train + dev only) into $INTERCYCLE_DATA/inventory.parquet.

    python scripts/esk_p6_apply.py --p6-json <cloud/modal_p6.py result>

The framing map is also saved as $INTERCYCLE_DATA/p6_framing.csv (clip_id, framing_id; gitignored data). If
inventory.parquet does not exist yet, only the csv is written and the script says so; run it again after the inventory
is built. TEST rows are never touched (intercycle.framing.merge_framings raises on a test id).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intercycle import esk  # noqa: E402
from intercycle import framing as fr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p6-json", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=ROOT / "prereg" / "split.csv")
    a = ap.parse_args()
    framing = json.loads(a.p6_json.read_text(encoding="utf-8"))["framing_of"]
    with open(a.split, newline="", encoding="utf-8") as fh:
        split = {r["clip_id"]: r["split"] for r in csv.DictReader(fh)}
    bad = [c for c in framing if split.get(c) not in ("train", "dev")]
    if bad:
        raise SystemExit(f"framing map holds non train/dev ids: {bad[:5]}")
    root = esk.data_root()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "p6_framing.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "framing_id"])
        w.writerows(sorted(framing.items()))
    inv_path = root / "inventory.parquet"
    if not inv_path.exists():
        print(f"wrote {root / 'p6_framing.csv'} ({len(framing)} clips); {inv_path} does not exist yet: rerun after it is built")
        return
    inv = fr.merge_framings(pd.read_parquet(inv_path), framing)
    esk.check_against_split(inv, a.split)
    inv.to_parquet(inv_path, index=False)
    print(f"framing_id set for {len(framing)} train/dev clips in {inv_path}")
    print(inv.groupby(["split", "framing_id"]).size().to_string())


if __name__ == "__main__":
    main()
