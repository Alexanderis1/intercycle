"""Amendment A1+ step 5: delete desktop copies of TEST-clip image data (ids only are printed).

    python scripts/esk_purge_local.py [--split prereg/split_a1.csv] [--dry-run] [--root DIR ...]

* every file under the roots whose name carries a TEST clip id (e.g. $INTERCYCLE_DATA/p5/frames/<id>.jpg) is deleted;
* $INTERCYCLE_DATA/p5/detections.json (per-frame boxes of the P5 run) is rewritten without TEST rows.
Prints a JSON log: deleted file names relative to their root, and the ids removed from detections.json. File contents
are never opened except detections.json, which is read as JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"(?<![0-9A-Za-z])\d_t[re]\d+(?![0-9])")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=ROOT / "prereg" / "split_a1.csv")
    ap.add_argument("--root", type=Path, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.split, newline="", encoding="utf-8") as fh:
        test = {r["clip_id"] for r in csv.DictReader(fh) if r["split"] == "test"}
    data = Path(os.environ["INTERCYCLE_DATA"])
    roots = [data] + a.root
    log: dict = {"deleted": [], "detections_rows_removed": []}
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                if set(ID_RE.findall(f)) & test:
                    p = Path(dp) / f
                    log["deleted"].append(str(p.relative_to(root)).replace("\\", "/"))
                    if not a.dry_run:
                        p.unlink()
    det = data / "p5" / "detections.json"
    if det.exists():
        rows = json.loads(det.read_text(encoding="utf-8"))
        keep = [r for r in rows if r["clip_id"] not in test]
        log["detections_rows_removed"] = sorted(r["clip_id"] for r in rows if r["clip_id"] in test)
        if not a.dry_run and len(keep) != len(rows):
            det.write_text(json.dumps(keep), encoding="utf-8")
    log["deleted"].sort()
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
