"""Labels contract (PLAN_TWIN §2 labels row, §5 "Split and labels", U14). Frozen at G0.

Where: `labels/dev/` (repo) and the custodian's sealed store (test; only its hash enters the repo, through
`sealed/LABELS.sha256`, format in twin/schemas/prereg.py).

* `SampleList` (`*.samples.json`): the hashed list of what must be labelled, made BEFORE labelling starts. Frames come
  from a seeded RNG over (group, clip), 1 frame every 2 s, stratified by camera; stroke windows (the whole clip up to
  10 s with a running press in the ROI) start at seeded-random times per press clip, never where the motion detector
  fires. `items_sha256` is the sha256 of the canonical JSON of `items` (`sample_items_sha256`).
* `LabelSet` (`*.labels.json`): one labeller's labels for one sample list. Every sampled frame is labelled, including
  frames with no person or forklift (an empty `boxes` list is the label "nothing there"). Boxes are person or forklift
  in the clip's native pixels; a person box carries its foot point (u, v). Each stroke window lists every clicked
  stroke time. The labeller id, its kind (Claude agent or human), whether it worked blind and whether detector
  proposals were shown are recorded (U14: every real-clip figure is "against independent Claude-agent labels").
  Test labels must be blind with no proposals.
* `check_complete(labels, samples)`: the label set answers exactly the hashed sample list (or a subset of it, for the
  blind relabel of 20 dev frames, with `require_all=False`).

All times are container PTS seconds from the clip start (never frame_index / 24).
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Camera, Strict, unique

SHA_RE = r"^[0-9a-f]{64}$"
LabelSplit = Literal["dev", "test"]
MAX_WINDOW_S = 10.0
T_EPS = 1e-6


class Box(Strict):
    cls: Literal["person", "forklift"]
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float
    y1: float
    foot_u: float | None = None              # person: foot point (pixels); forklift: None
    foot_v: float | None = None
    truncated: bool = False                  # cut by the image border
    occluded: bool = False                   # partly hidden (labelling rule: box the visible extent)
    src: Literal["labeller", "proposal_accepted", "proposal_corrected"] = "labeller"

    @model_validator(mode="after")
    def _shape(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"box ({self.x0}, {self.y0}, {self.x1}, {self.y1}) is empty")
        if self.cls == "person":
            if self.foot_u is None or self.foot_v is None:
                raise ValueError("a person box needs its foot point (foot_u, foot_v)")
            if not (self.x0 <= self.foot_u <= self.x1 and self.y0 <= self.foot_v <= self.y1):
                raise ValueError("the foot point must lie inside its box")
        elif self.foot_u is not None or self.foot_v is not None:
            raise ValueError("a forklift box has no foot point")
        return self


class SampleItem(Strict):
    sample_id: str = Field(min_length=1)
    kind: Literal["frame", "stroke_window"]
    clip_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    camera: Camera
    split: LabelSplit
    t_pts: float | None = Field(default=None, ge=0)      # frame
    press: str | None = None                             # stroke window
    t0: float | None = Field(default=None, ge=0)
    t1: float | None = None

    @model_validator(mode="after")
    def _kind(self):
        if self.kind == "frame":
            if self.t_pts is None or any(v is not None for v in (self.press, self.t0, self.t1)):
                raise ValueError(f"{self.sample_id}: a frame sample has t_pts only")
        else:
            if self.t_pts is not None or self.press is None or self.t0 is None or self.t1 is None:
                raise ValueError(f"{self.sample_id}: a stroke-window sample has press, t0 and t1")
            if not 0 < self.t1 - self.t0 <= MAX_WINDOW_S + T_EPS:
                raise ValueError(f"{self.sample_id}: window length must be in (0, {MAX_WINDOW_S}] s")
        return self


def sample_items_sha256(items: list[SampleItem]) -> str:
    canon = json.dumps([i.model_dump(mode="json") for i in items], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class SampleList(Strict):
    contract: Literal["twin.label_samples/1"]
    split: LabelSplit
    seed: int = Field(ge=0)
    rule: str = Field(min_length=1)          # the written sampling rule, as run
    items: list[SampleItem] = Field(min_length=1)
    items_sha256: str = Field(pattern=SHA_RE)

    @model_validator(mode="after")
    def _hash(self):
        unique([i.sample_id for i in self.items], "sample")
        wrong = sorted({i.split for i in self.items} - {self.split})
        if wrong:
            raise ValueError(f"sample list for {self.split} holds {wrong} items")
        if self.items_sha256 != sample_items_sha256(self.items):
            raise ValueError("items_sha256 does not match the items (the sample list changed after hashing)")
        return self


class FrameLabel(Strict):
    sample_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    t_pts: float = Field(ge=0)
    boxes: list[Box]                         # [] = no person / forklift in the frame (still a label)
    note: str | None = None


class StrokeWindow(Strict):
    sample_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    press: str = Field(min_length=1)
    t0: float = Field(ge=0)
    t1: float
    strokes: list[float]                     # clicked stroke times (PTS s), strictly increasing, inside [t0, t1]
    note: str | None = None

    @model_validator(mode="after")
    def _strokes(self):
        if not 0 < self.t1 - self.t0 <= MAX_WINDOW_S + T_EPS:
            raise ValueError(f"{self.sample_id}: window length must be in (0, {MAX_WINDOW_S}] s")
        if any(b <= a for a, b in zip(self.strokes, self.strokes[1:])):
            raise ValueError(f"{self.sample_id}: stroke times must increase strictly")
        if any(not self.t0 - T_EPS <= s <= self.t1 + T_EPS for s in self.strokes):
            raise ValueError(f"{self.sample_id}: a stroke lies outside its window")
        return self


class LabelSet(Strict):
    contract: Literal["twin.labels/1"]
    split: LabelSplit
    labeller_id: str = Field(min_length=1)   # e.g. agent:dev-labeller-a (distinct from every tuning lane)
    labeller_kind: Literal["claude_agent", "human"]
    blind: bool                              # saw only the frames: no proposals, no other labels, no results
    proposals_shown: bool                    # detector proposals were pre-filled (dev only; never accepted by agents)
    rules_sha256: str = Field(pattern=SHA_RE)        # the committed one-page labelling rule
    sample_list_sha256: str = Field(pattern=SHA_RE)  # SampleList.items_sha256 this set answers
    frames: list[FrameLabel]
    stroke_windows: list[StrokeWindow]

    @model_validator(mode="after")
    def _honest(self):
        unique([f.sample_id for f in self.frames] + [w.sample_id for w in self.stroke_windows], "labelled sample")
        if self.blind and self.proposals_shown:
            raise ValueError("a blind label set cannot have seen proposals")
        if self.split == "test" and not self.blind:
            raise ValueError("test labels are made blind, with no proposals (PLAN_TWIN §5, U14)")
        srcs = {b.src for f in self.frames for b in f.boxes}
        if not self.proposals_shown and srcs - {"labeller"}:
            raise ValueError("boxes marked as proposals in a set that saw no proposals")
        if self.labeller_kind == "claude_agent" and "proposal_accepted" in srcs:
            raise ValueError("agents may pre-fill boxes but never accept proposals unchanged (PLAN_TWIN §5)")
        if not self.frames and not self.stroke_windows:
            raise ValueError("empty label set")
        return self


def check_complete(labels: LabelSet, samples: SampleList, *, require_all: bool = True) -> None:
    """Raise ValueError unless `labels` answers `samples`: same hash and split, every label is a sampled item of the
    same kind, clip and time, and (require_all) every sampled item is labelled, empty frames included."""
    if labels.sample_list_sha256 != samples.items_sha256:
        raise ValueError("label set answers a different sample list (hash mismatch)")
    if labels.split != samples.split:
        raise ValueError(f"label split {labels.split} != sample split {samples.split}")
    items = {i.sample_id: i for i in samples.items}
    for f in labels.frames:
        it = items.get(f.sample_id)
        if it is None or it.kind != "frame" or it.clip_id != f.clip_id or abs((it.t_pts or 0.0) - f.t_pts) > T_EPS:
            raise ValueError(f"frame label {f.sample_id} does not match a sampled frame")
    for w in labels.stroke_windows:
        it = items.get(w.sample_id)
        if it is None or it.kind != "stroke_window" or it.clip_id != w.clip_id or it.press != w.press \
                or abs((it.t0 or 0.0) - w.t0) > T_EPS or abs((it.t1 or 0.0) - w.t1) > T_EPS:
            raise ValueError(f"stroke window {w.sample_id} does not match a sampled window")
    if require_all:
        done = {f.sample_id for f in labels.frames} | {w.sample_id for w in labels.stroke_windows}
        missing = sorted(set(items) - done)
        if missing:
            raise ValueError(f"{len(missing)} sampled items are not labelled (empty frames too): {missing[:5]}")
