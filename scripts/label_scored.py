"""Write labels/dev/scored.json + scored.sha256 from the DEV label sets (lane S4-labels fix round 1).

    python scripts/label_scored.py          (reads esk.load("dev") for the clip sha256; dev only)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.dev_labels import Part, build_scored, dump, sha256_bytes  # noqa: E402

DIR = ROOT / "labels" / "dev"
PARTS = [("labels.json", "sample.json", False), ("labels_s1.json", "sample_s1.json", True)]


def main() -> int:
    from intercycle import esk

    inv = esk.load("dev")
    sha_of = {str(c): str(s) for c, s in zip(inv["clip_id"], inv["sha256"])}
    parts = [Part(lab, (DIR / lab).read_bytes(), smp, (DIR / smp).read_bytes(), supplement=sup)
             for lab, smp, sup in PARTS]
    doc = build_scored(parts, sha_of)
    raw = dump(doc)
    (DIR / "scored.json").write_bytes(raw)
    (DIR / "scored.sha256").write_text(f"{sha256_bytes(raw)}  scored.json\n", encoding="utf-8", newline="\n")
    print(doc["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
