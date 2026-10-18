"""Rebuild prereg/viewed_clips.txt: clips an agent LOOKED AT (vision) before amendment A1+ (prereg/AMENDMENTS.md).

    python scripts/esk_viewed_clips.py [--out prereg/viewed_clips.txt]

Sources (all committed before A1+; every id is a TRAIN id of the original prereg/split.csv):
* P5 (docs/probes/S3a-P5.md, "Degenerate split" bullet): the 40 agent-checked frames and the 22 other frames the agent
  viewed (vision).
* P6 (twin/cal/k1_v0.json, source_clip_ids): the 25 train frames per framing whose per-pixel median background the
  agent viewed to pick the k1 lines and look for guard hardware.

Processed automatically only (no agent looked at the pixels); they stay ELIGIBLE for test, and any local image copy of
one that lands in test is deleted (A1+ step 5):
* P5: all 398 train t=0.5 s frames (detectors on the desktop GPU; copies in $INTERCYCLE_DATA/p5/frames).
* P6: SIFT registration and person-box scale fit over all train+dev frames, on Modal.
* P7: 8 + 8 train t=0.5 s frames, matcher counts only (cloud/modal_esk.py p7, random.Random(7)).
* P8: 100 train t=0.5 s frames, RF-DETR + YuNet on Modal, aggregates only (cloud/modal_p8_faces.py, seed 20261025).
* Staging (Pass 1, RF-DETR on all 691 frames): automatic.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
ID_RE = re.compile(r"\b\d_t[re]\d+\b")


def _cam(c: str) -> int:
    return CAMERA_OF_PREFIX[int(c.split("_")[0])]


def p5_viewed() -> tuple[list[str], list[str]]:
    txt = (ROOT / "docs" / "probes" / "S3a-P5.md").read_text(encoding="utf-8")
    a = txt.index("The 40 checked frames:")
    b = txt.index("The other frames viewed:")
    c = txt.index("The `te`/`tr` in these ids")
    checked, other = ID_RE.findall(txt[a:b]), ID_RE.findall(txt[b:c])
    assert len(checked) == 40 and len(set(checked)) == 40, len(checked)
    return checked, other


def p6_viewed() -> list[str]:
    k = json.loads((ROOT / "twin" / "cal" / "k1_v0.json").read_text(encoding="utf-8"))
    out: list[str] = []
    for fid in ("c9_f0", "c14_f0"):
        out += list(k[fid]["source_clip_ids"])
    return out


def _orig_train() -> list[str]:
    with open(ROOT / "prereg" / "split.csv", newline="", encoding="utf-8") as fh:
        return [r["clip_id"] for r in csv.DictReader(fh) if r["split"] == "train"]


def p7_auto() -> list[str]:
    train = _orig_train()  # same draw as cloud/modal_esk.py action p7
    rnd = random.Random(7)
    c9 = rnd.sample(sorted(c for c in train if _cam(c) == 9), 8)
    c14 = rnd.sample(sorted(c for c in train if _cam(c) == 14), 8)
    return c9 + c14


def p8_auto() -> list[str]:
    train = _orig_train()  # same draw as cloud/modal_p8_faces.py
    by = {c: sorted(x for x in train if _cam(x) == c) for c in (9, 14)}
    rng = np.random.default_rng(20261017 + 8)
    want = {c: int(round(100 * len(v) / len(train))) for c, v in by.items()}
    want[9] = 100 - want[14]
    ch = {c: sorted(rng.choice(by[c], size=want[c], replace=False).tolist()) for c in (9, 14)}
    return ch[9] + ch[14]


def build() -> dict[str, list[str]]:
    checked, other = p5_viewed()
    return {"viewed": sorted(set(checked) | set(other) | set(p6_viewed())),
            "p5_checked": checked, "p5_other": other, "p6": p6_viewed(), "p7_auto": p7_auto(), "p8_auto": p8_auto()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "prereg" / "viewed_clips.txt")
    a = ap.parse_args()
    s = build()
    lines = ["# Clips an agent LOOKED AT (vision) before amendment A1+; their groups are ineligible for TEST.",
             "# Built by scripts/esk_viewed_clips.py. One id per line; '#' lines are comments.",
             f"# P5 agent-checked frames ({len(s['p5_checked'])}) + other P5 frames viewed ({len(s['p5_other'])}),"
             f" P6 median-background sources ({len(s['p6'])}); union {len(s['viewed'])}.", *s["viewed"]]
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    auto = sorted((set(s["p7_auto"]) | set(s["p8_auto"])) - set(s["viewed"]))
    print(json.dumps({"viewed": len(s["viewed"]), "p5": len(set(s["p5_checked"]) | set(s["p5_other"])),
                      "p6": len(set(s["p6"])), "p7_auto_not_viewed": len(set(s["p7_auto"]) - set(s["viewed"])),
                      "p8_auto_not_viewed": len(set(s["p8_auto"]) - set(s["viewed"])), "p7_p8_auto_only": len(auto)}))


if __name__ == "__main__":
    main()
