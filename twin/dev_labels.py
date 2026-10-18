"""The scored DEV label set, machine-readable (lane S4-labels fix round 1; labels/RULES.md r1 "The scored set").

`labels/dev/scored.json` gives every sampled item of the DEV label sets one status, derived from the labels by fixed
rules (never typed by hand):

* frames: `scored` | `duplicate` (same clip sha256 and PTS as an earlier labelled frame; `dup_of` names it);
* stroke windows, in list order (base list, then supplement): `not_needed` (an unlabelled supplementary reserve),
  `duplicate` (same press, clip sha256 and window as an earlier labelled window), `ambiguous` (note starts with
  "NOT SCORABLE:", strokes empty), `not_running` (strokes empty, note starts with "press not running"), else `scored`.
  Any other shape fails closed. `hidden` lists the spans whose BDCs the labeller could not see (note text
  "hidden a-b s"); `edge` lists the spans at a window edge where a possible BDC was left unclicked (note text
  "edge a-b s", required whenever the note says "edge, not clicked"). A scorer ignores detections inside both
  (`ScoredDev.ignore`). A span must overlap its window, and a bound within SNAP_S of a window edge must reach it.
* Stopping rule (fix round 2): each press ends with exactly TARGET_PER_PRESS scored windows, a supplementary reserve
  is left unlabelled only once its press has the target, and is labelled only while its press is below the target.

Scorers call `load_scored(dir)`: it checks the sha256 of scored.json (scored.sha256) and of every label file and sample
list it names, rebuilds the statuses from those files and requires them to equal scored.json, and returns only the
scored frames and windows.
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
EDGE_RE = re.compile(r"edge (\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) s")
EDGE_NOT_CLICKED = "edge, not clicked"
RESERVE_RE = re.compile(r"-r\d+$")
SNAP_S = 0.1          # a span bound this close to a window edge must reach it (rounding never leaves a frame out)
TARGET_PER_PRESS = 10
RULE = ("Statuses derived by twin/dev_labels.py build_scored from the label notes and strokes (labels/RULES.md r2): "
        "frames scored | duplicate (same clip sha256 and PTS); stroke windows in list order: not_needed (unlabelled "
        "supplementary reserve), duplicate (same press, clip sha256, t0, t1), ambiguous ('NOT SCORABLE:' note), "
        "not_running ('press not running' note, no strokes), scored (at least one BDC). Detections inside 'hidden' "
        "and 'edge' spans are not scored. Scored windows: exactly 10 per press with distinct clip sha256; a "
        "supplementary reserve is labelled only while its press has fewer than 10.")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _spans(rx: re.Pattern, what: str, note: str | None) -> list[list[float]]:
    out = [[float(a), float(b)] for a, b in rx.findall(note or "")]
    if any(b <= a for a, b in out):
        raise ValueError(f"empty {what} span in note: {note!r}")
    return out


def hidden_spans(note: str | None) -> list[list[float]]:
    return _spans(HIDDEN_RE, "hidden", note)


def edge_spans(note: str | None) -> list[list[float]]:
    return _spans(EDGE_RE, "edge", note)


def _check_spans(sid: str, spans: list[list[float]], t0: float, t1: float) -> None:
    for a, b in spans:
        if b <= t0 or a >= t1:
            raise ValueError(f"{sid}: span {a}-{b} s lies outside the window [{t0}, {t1}]")
        if 0 < t1 - b < SNAP_S or 0 < a - t0 < SNAP_S:
            raise ValueError(f"{sid}: span {a}-{b} s stops short of the window edge [{t0}, {t1}] "
                             f"(write a bound that reaches it)")


@dataclass
class Part:
    """One label file and the sample list it answers. `supplement` lists may leave reserves unlabelled."""
    labels_name: str
    labels_bytes: bytes
    samples_name: str
    samples_bytes: bytes
    supplement: bool = False
    split: str = "dev"
    labels: LabelSet = field(init=False)
    samples: SampleList = field(init=False)

    def __post_init__(self):
        self.labels = LabelSet.model_validate(json.loads(self.labels_bytes))
        self.samples = SampleList.model_validate(json.loads(self.samples_bytes))
        if self.labels.split != self.split or self.samples.split != self.split:
            raise ValueError(f"{self.labels_name}: {self.split} label sets only")
        check_complete(self.labels, self.samples, require_all=not self.supplement)


def build_scored(parts: list[Part], sha_of: dict[str, str], test: bool = False) -> dict:
    """The scored.json document for `parts` (base first). `sha_of` maps clip_id -> sha256 (dev inventory).
    test=True (the sealed TEST labels, labels/TEST_RULES.md "Reserve windows are labelled like every other window;
    which windows are scored is decided later from the list order"): every `-rN` window is a reserve, a reserve that
    would be scored after its press already has TARGET_PER_PRESS scored windows is `not_needed`, and a press that ends
    below the target is reported in counts["short_presses"] instead of refused (the labels are sealed)."""
    def csha(cid: str) -> str:
        if cid not in sha_of:
            raise ValueError(f"clip {cid} is not in the dev inventory")
        return sha_of[cid]

    frames_out, wins_out = [], []
    seen_f: dict[tuple, str] = {}
    seen_w: dict[tuple, str] = {}
    n_scored: dict[str, int] = {}
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
            reserve = (part.supplement or test) and bool(RESERVE_RE.search(it.sample_id))
            row = dict(sample_id=it.sample_id, label_set=part.labels_name, press=it.press, clip_id=it.clip_id,
                       clip_sha256=csha(it.clip_id), t0=it.t0, t1=it.t1, dup_of=None, n_strokes=0, hidden=[],
                       edge=[], supplement_reserve=reserve)
            if w is None:
                if not reserve:
                    raise ValueError(f"stroke window {it.sample_id} is not labelled")
                row["status"] = "not_needed"
                wins_out.append(row)
                continue
            note = w.note or ""
            key = (it.press, row["clip_sha256"], round(float(it.t0 or 0.0), 6), round(float(it.t1 or 0.0), 6))
            dup = seen_w.get(key)
            seen_w.setdefault(key, it.sample_id)
            hid, edg = hidden_spans(note), edge_spans(note)
            if EDGE_NOT_CLICKED in note and not edg:
                raise ValueError(f"{it.sample_id}: a BDC left unclicked at the window edge needs an edge span "
                                 f"('edge a-b s' reaching the window edge)")
            _check_spans(it.sample_id, hid + edg, float(it.t0 or 0.0), float(it.t1 or 0.0))
            row.update(n_strokes=len(w.strokes), hidden=hid, edge=edg)
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
            elif test and reserve and n_scored.get(str(it.press), 0) >= TARGET_PER_PRESS:
                row["status"] = "not_needed"
            else:
                row["status"] = "scored"
            if row["status"] == "scored":
                n_scored[str(it.press)] = n_scored.get(str(it.press), 0) + 1
            wins_out.append(row)
    if not test:
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
        "scored_edge_seconds": round(sum(b - a for w in wins_out if w["status"] == "scored"
                                         for a, b in w["edge"]), 3),
        "scored_strokes": sum(w["n_strokes"] for w in wins_out if w["status"] == "scored"),
    }
    if test:
        counts["short_presses"] = {p: n_scored.get(p, 0) for p in sorted({w["press"] for w in wins_out})
                                   if n_scored.get(p, 0) < TARGET_PER_PRESS}
    parts_out = [dict(labels=p.labels_name, samples=p.samples_name, supplement=p.supplement) for p in parts]
    return dict(contract=CONTRACT, rule=RULE, parts=parts_out, files=files, frames=frames_out,
                stroke_windows=wins_out, counts=counts)


def _check_stopping(wins: list[dict]) -> None:
    """Stopping rule (labels/RULES.md "Supplementary lists"): per press, in list order, a supplementary reserve is
    left unlabelled (not_needed) only once the press has TARGET_PER_PRESS scored windows and is labelled only while it
    has fewer; scored windows of one press have distinct clip sha256; each press ends with exactly the target."""
    for press in sorted({w["press"] for w in wins}):
        n = 0
        shas = set()
        for w in (x for x in wins if x["press"] == press):
            if w["status"] == "not_needed":
                if not w.get("supplement_reserve"):
                    raise ValueError(f"{w['sample_id']}: only a supplementary reserve can be not_needed")
                if n < TARGET_PER_PRESS:
                    raise ValueError(f"{w['sample_id']} left unlabelled while {press} has only {n} scored windows")
            elif w.get("supplement_reserve") and n >= TARGET_PER_PRESS:
                raise ValueError(f"{w['sample_id']}: supplementary reserve labelled after {press} reached "
                                 f"{TARGET_PER_PRESS} scored windows (it is not needed)")
            if w["status"] == "scored":
                if w["clip_sha256"] in shas:
                    raise ValueError(f"{w['sample_id']}: a second scored window on the same clip bytes")
                shas.add(w["clip_sha256"])
                n += 1
        if n != TARGET_PER_PRESS:
            raise ValueError(f"press {press} ends with {n} scored windows, the rule is exactly {TARGET_PER_PRESS}")


def dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class ScoredDev:
    frames: list[FrameLabel]
    windows: list[StrokeWindow]
    hidden: dict[str, list[list[float]]]      # sample_id -> hidden spans of a scored window
    edge: dict[str, list[list[float]]]        # sample_id -> edge spans (possible BDC left unclicked at a window edge)
    ignore: dict[str, list[list[float]]]      # sample_id -> hidden + edge, sorted: detections inside are not scored
    doc: dict


def load_scored(labels_dir: Path) -> ScoredDev:
    """The scored DEV frames and stroke windows. Fails closed (ValueError) on any hash mismatch, and when the statuses
    rebuilt from the hashed label files and sample lists (build_scored, stopping rule included) differ from
    scored.json. The clip sha256 used for the rebuild are the ones scored.json records (no inventory read here)."""
    labels_dir = Path(labels_dir)
    raw = (labels_dir / "scored.json").read_bytes()
    want = (labels_dir / "scored.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_bytes(raw) != want:
        raise ValueError("scored.json does not match scored.sha256")
    doc = json.loads(raw)
    if doc.get("contract") != CONTRACT:
        raise ValueError(f"unknown scored contract {doc.get('contract')!r}")
    blobs: dict[str, bytes] = {}
    for name, h in doc["files"].items():
        b = (labels_dir / name).read_bytes()
        if sha256_bytes(b) != h:
            raise ValueError(f"{name} changed after scored.json was made")
        blobs[name] = b
    parts = [Part(p["labels"], blobs[p["labels"]], p["samples"], blobs[p["samples"]], supplement=bool(p["supplement"]))
             for p in doc["parts"]]
    sha_of: dict[str, str] = {}
    for r in doc["frames"] + doc["stroke_windows"]:
        if sha_of.setdefault(r["clip_id"], r["clip_sha256"]) != r["clip_sha256"]:
            raise ValueError(f"scored.json gives clip {r['clip_id']} two sha256")
    if build_scored(parts, sha_of) != doc:
        raise ValueError("scored.json is not what the label files give (statuses rebuilt by build_scored differ)")
    fmap = {(p.labels_name, f.sample_id): f for p in parts for f in p.labels.frames}
    wmap = {(p.labels_name, w.sample_id): w for p in parts for w in p.labels.stroke_windows}
    frames = [fmap[(r["label_set"], r["sample_id"])] for r in doc["frames"] if r["status"] == "scored"]
    rows = [r for r in doc["stroke_windows"] if r["status"] == "scored"]
    windows = [wmap[(r["label_set"], r["sample_id"])] for r in rows]
    return ScoredDev(frames=frames, windows=windows, hidden={r["sample_id"]: r["hidden"] for r in rows},
                     edge={r["sample_id"]: r["edge"] for r in rows},
                     ignore={r["sample_id"]: sorted(r["hidden"] + r["edge"]) for r in rows}, doc=doc)


def load_scored_test(label_files: dict[str, bytes], samples_name: str, sha_of: dict[str, str]) -> ScoredDev:
    """The scored TEST frames and stroke windows (sealed scoring, scripts/sealed_run.py score). `label_files` maps
    each LABELS-hashed label file name to its bytes (the caller has checked them against sealed/LABELS.sha256), among
    them the sample list `samples_name`; `sha_of` maps clip_id -> sha256 (the MANIFEST). Each label file is a test
    LabelSet answering that sample list; together they label every sampled item exactly once (fail closed). Statuses:
    build_scored(test=True), same rules as dev."""
    samples = label_files[samples_name]
    names = sorted(n for n in label_files if n != samples_name)
    if not names:
        raise ValueError("no test label file besides the sample list")
    parts = [Part(n, label_files[n], samples_name, samples, supplement=True, split="test") for n in names]
    labelled = [f.sample_id for p in parts for f in p.labels.frames] +                [w.sample_id for p in parts for w in p.labels.stroke_windows]
    if len(labelled) != len(set(labelled)):
        raise ValueError("a test sample is labelled in more than one label file")
    want = {i.sample_id for i in parts[0].samples.items}
    if set(labelled) != want:
        raise ValueError(f"test labels leave {len(want - set(labelled))} sampled items unlabelled (every item is "
                         f"labelled, TEST_RULES)")
    # one base part per label file; reserves are decided by list order across the files (build_scored(test=True))
    for p in parts:
        p.supplement = False
    doc = build_scored(_test_parts_in_list_order(parts), sha_of, test=True)
    fmap = {f.sample_id: f for p in parts for f in p.labels.frames}
    wmap = {w.sample_id: w for p in parts for w in p.labels.stroke_windows}
    frames = [fmap[r["sample_id"]] for r in doc["frames"] if r["status"] == "scored"]
    rows = [r for r in doc["stroke_windows"] if r["status"] == "scored"]
    windows = [wmap[r["sample_id"]] for r in rows]
    return ScoredDev(frames=frames, windows=windows, hidden={r["sample_id"]: r["hidden"] for r in rows},
                     edge={r["sample_id"]: r["edge"] for r in rows},
                     ignore={r["sample_id"]: sorted(r["hidden"] + r["edge"]) for r in rows}, doc=doc)


def _test_parts_in_list_order(parts: list[Part]) -> list[Part]:
    """build_scored walks each part's sample list in order; the test sample list is shared by every label file, so
    the merged labels are given as ONE part (all labels of all files) to keep the list order across frames and
    windows. The merged LabelSet is re-validated."""
    base = json.loads(parts[0].labels_bytes)
    base["frames"] = [json.loads(f.model_dump_json()) for p in parts for f in p.labels.frames]
    base["stroke_windows"] = [json.loads(w.model_dump_json()) for p in parts for w in p.labels.stroke_windows]
    for p in parts[1:]:
        for k in ("rules_sha256", "sample_list_sha256", "split"):
            if getattr(p.labels, k) != getattr(parts[0].labels, k):
                raise ValueError(f"test label files disagree on {k}")
    merged = Part("+".join(p.labels_name for p in parts), json.dumps(base).encode("utf-8"), parts[0].samples_name,
                  parts[0].samples_bytes, supplement=False, split="test")
    return [merged]
