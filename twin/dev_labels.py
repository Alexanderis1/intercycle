"""The scored DEV label set, machine-readable (lane S4-labels fix round 1; labels/RULES.md r1 "The scored set").

`labels/dev/scored.json` gives every sampled item of the DEV label sets one status, derived from the labels by fixed
rules (never typed by hand):

* frames: `scored` | `duplicate` (same clip sha256 and PTS as an earlier labelled frame; `dup_of` names it);
* stroke windows, in list order (base list, then supplement): `not_needed` (an unlabelled supplementary reserve),
  `duplicate` (same press, clip sha256 and window as an earlier labelled window), `ambiguous` (note starts with
  "NOT SCORABLE:", strokes empty), `not_running` (strokes empty, note starts with "press not running"), else `scored`.
  Any other shape fails closed. `hidden` lists the spans whose BDCs the labeller could not see (note text
  "hidden a-b s"); a scorer ignores detections inside them.

Scorers call `load_scored(dir)`: it checks the sha256 of scored.json (scored.sha256) and of every label file and sample
list it names, and returns only the scored frames and windows.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from twin.schemas.labels import FrameLabel, LabelSet, SampleList, StrokeWindow, check_complete

CONTRACT = "intercycle.dev_label_scoring/1"
NOT_SCORABLE = "NOT SCORABLE:"
NOT_RUNNING = "press not running"
HIDDEN_RE = re.compile(r"hidden (\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) s")
TARGET_PER_PRESS = 10
RULE = ("Statuses derived by twin/dev_labels.py build_scored from the label notes and strokes (labels/RULES.md r1): "
        "frames scored | duplicate (same clip sha256 and PTS); stroke windows in list order: not_needed (unlabelled "
        "supplementary reserve), duplicate (same press, clip sha256, t0, t1), ambiguous ('NOT SCORABLE:' note), "
        "not_running ('press not running' note, no strokes), scored (at least one BDC). Detections inside 'hidden' "
        "spans are not scored. Scored windows: 10 per press with distinct clip sha256.")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hidden_spans(note: str | None) -> list[list[float]]:
    out = [[float(a), float(b)] for a, b in HIDDEN_RE.findall(note or "")]
    if any(b <= a for a, b in out):
        raise ValueError(f"empty hidden span in note: {note!r}")
    return out


@dataclass
class Part:
    """One label file and the sample list it answers. `supplement` lists may leave reserves unlabelled."""
    labels_name: str
    labels_bytes: bytes
    samples_name: str
    samples_bytes: bytes
    supplement: bool = False
    labels: LabelSet = field(init=False)
    samples: SampleList = field(init=False)

    def __post_init__(self):
        self.labels = LabelSet.model_validate(json.loads(self.labels_bytes))
        self.samples = SampleList.model_validate(json.loads(self.samples_bytes))
        if self.labels.split != "dev" or self.samples.split != "dev":
            raise ValueError(f"{self.labels_name}: dev label sets only")
        check_complete(self.labels, self.samples, require_all=not self.supplement)


def build_scored(parts: list[Part], sha_of: dict[str, str]) -> dict:
    """The scored.json document for `parts` (base first). `sha_of` maps clip_id -> sha256 (dev inventory)."""
    def csha(cid: str) -> str:
        if cid not in sha_of:
            raise ValueError(f"clip {cid} is not in the dev inventory")
        return sha_of[cid]

    frames_out, wins_out = [], []
    seen_f: dict[tuple, str] = {}
    seen_w: dict[tuple, str] = {}
    for part in parts:
        fl = {f.sample_id: f for f in part.labels.frames}
        wl = {w.sample_id: w for w in part.labels.stroke_windows}
        for it in part.samples.items:
            if it.kind == "frame":
                f = fl.get(it.sample_id)
                if f is None:
                    raise ValueError(f"frame {it.sample_id} is not labelled (every sampled frame is labelled)")
                key = (csha(it.clip_id), round(float(it.t_pts or 0.0), 6))
                dup = seen_f.get(key)
                seen_f.setdefault(key, it.sample_id)
                frames_out.append(dict(sample_id=it.sample_id, label_set=part.labels_name, clip_id=it.clip_id,
                                       clip_sha256=key[0], camera=it.camera, n_boxes=len(f.boxes),
                                       status="duplicate" if dup else "scored", dup_of=dup))
                continue
            w = wl.get(it.sample_id)
            row = dict(sample_id=it.sample_id, label_set=part.labels_name, press=it.press, clip_id=it.clip_id,
                       clip_sha256=csha(it.clip_id), t0=it.t0, t1=it.t1, dup_of=None, n_strokes=0, hidden=[])
            if w is None:
                if not (part.supplement and it.sample_id.rsplit("-", 1)[-1].startswith("r")):
                    raise ValueError(f"stroke window {it.sample_id} is not labelled")
                row["status"] = "not_needed"
                wins_out.append(row)
                continue
            note = w.note or ""
            key = (it.press, row["clip_sha256"], round(float(it.t0 or 0.0), 6), round(float(it.t1 or 0.0), 6))
            dup = seen_w.get(key)
            seen_w.setdefault(key, it.sample_id)
            row.update(n_strokes=len(w.strokes), hidden=hidden_spans(note))
            if dup:
                row.update(status="duplicate", dup_of=dup)
            elif note.startswith(NOT_SCORABLE):
                if w.strokes:
                    raise ValueError(f"{it.sample_id}: a NOT SCORABLE window has clicked strokes")
                row["status"] = "ambiguous"
            elif not w.strokes:
                if not note.startswith(NOT_RUNNING):
                    raise ValueError(f"{it.sample_id}: no strokes but neither 'NOT SCORABLE:' nor "
                                     f"'{NOT_RUNNING}' (fail closed)")
                row["status"] = "not_running"
            else:
                row["status"] = "scored"
            wins_out.append(row)
    _check_stopping(wins_out)
    files = {}
    for part in parts:
        files[part.labels_name] = sha256_bytes(part.labels_bytes)
        files[part.samples_name] = sha256_bytes(part.samples_bytes)
    counts = {
        "frames_scored": sum(f["status"] == "scored" for f in frames_out),
        "frames_duplicate": sum(f["status"] == "duplicate" for f in frames_out),
        "windows": {p: {s: sum(w["press"] == p and w["status"] == s for w in wins_out)
                        for s in ("scored", "not_running", "ambiguous", "duplicate", "not_needed")}
                    for p in sorted({w["press"] for w in wins_out})},
        "scored_window_seconds": round(sum(w["t1"] - w["t0"] for w in wins_out if w["status"] == "scored"), 3),
        "scored_hidden_seconds": round(sum(b - a for w in wins_out if w["status"] == "scored"
                                           for a, b in w["hidden"]), 3),
        "scored_strokes": sum(w["n_strokes"] for w in wins_out if w["status"] == "scored"),
    }
    return dict(contract=CONTRACT, rule=RULE, files=files, frames=frames_out, stroke_windows=wins_out, counts=counts)


def _check_stopping(wins: list[dict]) -> None:
    """Supplement stopping rule: per press, a reserve is left unlabelled only once the press has its target; scored
    windows of one press have distinct clip sha256."""
    for press in sorted({w["press"] for w in wins}):
        n = 0
        shas = set()
        for w in (x for x in wins if x["press"] == press):
            if w["status"] == "not_needed" and n < TARGET_PER_PRESS:
                raise ValueError(f"{w['sample_id']} left unlabelled while {press} has only {n} scored windows")
            if w["status"] == "scored":
                if w["clip_sha256"] in shas:
                    raise ValueError(f"{w['sample_id']}: a second scored window on the same clip bytes")
                shas.add(w["clip_sha256"])
                n += 1


def dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class ScoredDev:
    frames: list[FrameLabel]
    windows: list[StrokeWindow]
    hidden: dict[str, list[list[float]]]      # sample_id -> hidden spans of a scored window
    doc: dict


def load_scored(labels_dir: Path) -> ScoredDev:
    """The scored DEV frames and stroke windows. Fails closed (ValueError) on any hash mismatch."""
    labels_dir = Path(labels_dir)
    raw = (labels_dir / "scored.json").read_bytes()
    want = (labels_dir / "scored.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_bytes(raw) != want:
        raise ValueError("scored.json does not match scored.sha256")
    doc = json.loads(raw)
    if doc.get("contract") != CONTRACT:
        raise ValueError(f"unknown scored contract {doc.get('contract')!r}")
    sets: dict[str, LabelSet] = {}
    for name, h in doc["files"].items():
        b = (labels_dir / name).read_bytes()
        if sha256_bytes(b) != h:
            raise ValueError(f"{name} changed after scored.json was made")
        if name.startswith("labels"):
            sets[name] = LabelSet.model_validate(json.loads(b))
    fmap = {(n, f.sample_id): f for n, s in sets.items() for f in s.frames}
    wmap = {(n, w.sample_id): w for n, s in sets.items() for w in s.stroke_windows}
    frames = [fmap[(r["label_set"], r["sample_id"])] for r in doc["frames"] if r["status"] == "scored"]
    rows = [r for r in doc["stroke_windows"] if r["status"] == "scored"]
    windows = [wmap[(r["label_set"], r["sample_id"])] for r in rows]
    return ScoredDev(frames=frames, windows=windows, hidden={r["sample_id"]: r["hidden"] for r in rows}, doc=doc)
