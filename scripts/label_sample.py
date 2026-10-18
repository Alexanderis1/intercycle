"""Seeded DEV label sample list (lane S4-labels; PLAN_TWIN §5 "Split and labels", U14).

Draws, BEFORE any labelling, the list of what the dev labeller must label, and writes it as a `twin.label_samples/1`
SampleList (twin/schemas/labels.py) plus its hash file:

* Person frames: per camera (stratified, equal allocation), a seeded round-robin over the camera's dev groups in a
  seeded order; inside a group, one frame drawn at random from the group's remaining grid frames (all its clips).
  The grid is 1 frame every 2 s per clip: t = pts[0] + 1 + 2k s, snapped to the nearest container PTS.
* Stroke windows: per press, the same seeded round-robin over the dev groups of the press's framings, one window per
  (press, clip): the whole clip if it lasts <= 10 s, else 10 s starting at a seeded-random time (snapped to a PTS).
  The start comes from the RNG alone: no motion detector, track or proposal is read. n primary windows per press are
  followed by n_reserve reserve windows (ids ...-r1, -r2, ...), used in list order to replace primary windows the
  labeller finds with the press not running. Every item, reserves included, is labelled.

Only the dev split is accepted (fail closed); clips are listed through esk.load("dev").

Reproduce the committed, hashed lists (GPU env: needs PyAV for the PTS lists; nothing is written):

    python scripts/label_sample.py --legacy-base --check labels/dev/sample.json
    python scripts/label_sample.py --supplement labels/dev/sample.json --check labels/dev/sample_s1.json

labels/dev/sample.json was drawn before the fix-round-1 dedupe (dedupe=False, `--legacy-base`); without that flag the
base draw dedupes by sha256 and gives a DIFFERENT list. A new list is written only with `--out` to a path that does not
exist yet: an existing hashed list is never overwritten.

SEALED mode (custodian session only; `build_sealed_list`): the TEST sample list, drawn by the same rules as the dev
base list with the fix-round-1 dedupe (same grid, round-robin, window rule, reserve count 3, P9a on c9_f0 only as in
the dev supplement), from the TEST rows of the active split (group and framing from the split, camera from the class
prefix) and the container PTS of the sealed clips (demux only, no pixel decode), with the committed seed SEALED_SEED.
A stratum with fewer eligible items than its target is drawn whole and the shortfall is written in the list's rule.

    python scripts/label_sample.py --sealed [--clips C:\\intercycle_sealed\\clips] [--out C:\\intercycle_sealed\\labels\\sample.json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.schemas.common import Camera  # noqa: E402
from twin.schemas.labels import MAX_WINDOW_S, SampleItem, SampleList, sample_items_sha256  # noqa: E402

GRID_STEP_S = 2.0
GRID_OFFSET_S = 1.0
DEFAULT_SEED = 20261018
DEFAULT_FRAMES = {9: 20, 14: 20}
DEFAULT_WINDOWS = {"P9a": 10, "P9b": 10}
DEFAULT_RESERVE = 3
ROIS = ROOT / "docs" / "probes" / "S3a-P4_rois.json"
# Fix round 1 supplement (labels/dev/sample_s1.json). The base list gave 6 distinct running P9a windows (all c9_f0;
# its c9_f1 windows were not scorable) and 5 distinct running P9b windows; the targets are 10 + 10 distinct running
# windows. P9a is drawn on c9_f0 only (its ROI is labelled there; in c9_f1 the ram is not identifiable). Reserves are
# generous because P9b ran in 5 of 10 distinct base clips; RULES.md "Supplementary lists" says how they are used.
S1_PRESSES = {"P9a": ["c9_f0"], "P9b": ["c9_f1"]}
S1_WINDOWS = {"P9a": 4, "P9b": 5}
S1_RESERVE = {"P9a": 8, "P9b": 12}
S1_FRAMES = {9: 1}
# Sealed TEST list (custodian; PLAN_TWIN §5, U14). The seed is fixed here, before the list is drawn, and differs from
# the dev seed. Volumes: 20 person frames per camera, 10 + 10 stroke windows (+ the dev base reserve count per press).
SEALED_SEED = 7318502
SEALED_ROOT = Path(r"C:\intercycle_sealed")
SEALED_CLIPS = SEALED_ROOT / "clips"
SEALED_OUT = SEALED_ROOT / "labels" / "sample.json"
SEALED_FRAMES = {9: 20, 14: 20}
SEALED_WINDOWS = {"P9a": 10, "P9b": 10}
SEALED_RESERVE = DEFAULT_RESERVE
SEALED_PRESSES = S1_PRESSES

PtsFn = Callable[[str], list[float]]


def _cam(x) -> Camera:
    c = int(x)
    if c not in (9, 14):
        raise ValueError(f"camera {c} is not 9 or 14")
    return cast(Camera, c)


def frame_grid(pts: list[float], step: float = GRID_STEP_S, offset: float = GRID_OFFSET_S) -> list[float]:
    """Grid frames of one clip: t = pts[0] + offset + k*step (<= last PTS), each snapped to the nearest PTS."""
    if not pts:
        return []
    p = np.asarray(sorted(pts), dtype=float)
    out: list[float] = []
    t = p[0] + offset
    while t <= p[-1] + 1e-9:
        v = float(p[int(np.argmin(np.abs(p - t)))])
        if not out or v != out[-1]:
            out.append(v)
        t += step
    return out


def window_for(pts: list[float], rng: np.random.Generator, length: float = MAX_WINDOW_S) -> tuple[float, float]:
    """(t0, t1) on frame PTS: the whole clip if it spans <= length, else `length` s from a seeded-random start."""
    p = np.asarray(sorted(pts), dtype=float)
    first, last = float(p[0]), float(p[-1])
    if last - first <= length:
        return first, last
    start = float(rng.uniform(first, last - length))
    t0 = float(p[np.searchsorted(p, start - 1e-9)])
    t1 = float(p[np.searchsorted(p, t0 + length + 1e-9) - 1])
    return t0, t1


def _round_robin(pools: dict[str, list], order: list[str], n: int, rng: np.random.Generator, what: str) -> list:
    total = sum(len(v) for v in pools.values())
    if total < n:
        raise ValueError(f"{what}: only {total} candidates for {n}")
    picked: list = []
    while len(picked) < n:
        for g in order:
            if pools[g] and len(picked) < n:
                picked.append(pools[g].pop(int(rng.integers(len(pools[g])))))
    return picked


def rule_text(seed: int, n_frames: dict[int, int], n_windows: dict[str, int], n_reserve: int,
              presses: dict[str, list[str]]) -> str:
    return (
        f"scripts/label_sample.py, seed {seed}, split dev. Frames: per camera {dict(sorted(n_frames.items()))} "
        f"(stratified, equal allocation); grid 1 frame every {GRID_STEP_S:g} s per clip at t = pts0 + "
        f"{GRID_OFFSET_S:g} + {GRID_STEP_S:g}k snapped to the nearest PTS; seeded round-robin over the camera's dev "
        f"groups in seeded order (rng [seed, 1, camera]), one random remaining grid frame of the group per turn; every "
        f"sampled frame is labelled, empty frames included. Stroke windows: per press {n_windows} primary + "
        f"{n_reserve} reserve (ids -r1..), press framings {presses}; seeded round-robin over the dev groups of those "
        f"framings (rng [seed, 2, press index]), one window per (press, clip): whole clip if <= {MAX_WINDOW_S:g} s, "
        f"else {MAX_WINDOW_S:g} s from a seeded-random start snapped to a PTS; no motion detector, track or proposal "
        f"is read. Reserves replace, in list order, primary windows whose press the labeller finds not running; every "
        f"item is labelled. Times are container PTS seconds."
    )


def dedupe_by_sha(inv: pd.DataFrame) -> pd.DataFrame:
    """One clip per byte-identical set (Amendment 1: sha256 / dup_of): keep the smallest clip_id of each sha256."""
    if "sha256" not in inv.columns:
        raise ValueError("inventory has no sha256 column: cannot drop byte-identical copies (fail closed)")
    return inv.sort_values("clip_id").drop_duplicates("sha256", keep="first").reset_index(drop=True)


def _check_dev(inv: pd.DataFrame, split: str) -> None:
    if split != "dev":
        raise ValueError("this sampler draws the dev split only (the test labeller works in the custodian session)")
    if set(inv["split"]) - {"dev"}:
        raise ValueError(f"inventory rows outside dev: {sorted(set(inv['split']) - {'dev'})}")


def build_sample_list(inv: pd.DataFrame, pts_fn: PtsFn, presses: dict[str, list[str]], *, split: str, seed: int,
                      n_frames: dict[int, int], n_windows: dict[str, int], n_reserve: int,
                      dedupe: bool = True) -> SampleList:
    """The DEV sample list. `dedupe` (default since fix round 1) keeps one clip per sha256 before sampling; the
    committed labels/dev/sample.json was drawn before that fix (commit 87fa5c1), i.e. with dedupe=False."""
    _check_dev(inv, split)
    inv = dedupe_by_sha(inv) if dedupe else inv.sort_values("clip_id").reset_index(drop=True)
    pts_cache: dict[str, list[float]] = {}

    def pts(cid: str) -> list[float]:
        if cid not in pts_cache:
            pts_cache[cid] = sorted(float(x) for x in pts_fn(cid))
        return pts_cache[cid]

    items: list[SampleItem] = []
    for cam in sorted(n_frames):
        rng = np.random.default_rng([seed, 1, cam])
        sub = inv[inv["camera"] == cam]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: [(c, t) for c in sub.loc[sub["group_id"] == g, "clip_id"] for t in frame_grid(pts(c))]
                 for g in groups}
        picked = _round_robin(pools, order, n_frames[cam], rng, f"camera {cam} frames")
        gid = dict(zip(sub["clip_id"], sub["group_id"]))
        for k, (c, t) in enumerate(picked, 1):
            items.append(SampleItem(sample_id=f"{split}-f-c{cam}-{k:02d}", kind="frame", clip_id=c, group_id=gid[c],
                                    camera=_cam(cam), split="dev", t_pts=t))
    names = sorted(presses)
    for pi, press in enumerate(names):
        if press not in n_windows:
            continue
        rng = np.random.default_rng([seed, 2, pi])
        sub = inv[inv["framing_id"].isin(presses[press])]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: list(sub.loc[sub["group_id"] == g, "clip_id"]) for g in groups}
        n = n_windows[press]
        picked = _round_robin(pools, order, n + n_reserve, rng, f"press {press} windows")
        row = sub.set_index("clip_id")
        for k, c in enumerate(picked, 1):
            t0, t1 = window_for(pts(c), rng)
            sid = f"{split}-sw-{press}-{k:02d}" if k <= n else f"{split}-sw-{press}-r{k - n}"
            items.append(SampleItem(sample_id=sid, kind="stroke_window", clip_id=c, group_id=str(row.loc[c, "group_id"]),
                                    camera=_cam(row.loc[c, "camera"]), split="dev", press=press, t0=t0, t1=t1))
    rule = rule_text(seed, n_frames, n_windows, n_reserve, presses)
    if dedupe:
        rule += " Byte-identical clips (same sha256) are reduced to one clip (smallest clip_id) before sampling."
    return SampleList(contract="twin.label_samples/1", split="dev", seed=seed, rule=rule, items=items,
                      items_sha256=sample_items_sha256(items))


def build_supplement(inv: pd.DataFrame, pts_fn: PtsFn, presses: dict[str, list[str]], base: SampleList, *,
                     split: str, seed: int, n_windows: dict[str, int], n_reserve: dict[str, int],
                     n_frames: dict[int, int], tag: str = "s1") -> SampleList:
    """A seeded supplementary DEV list drawn after `base` (fix round 1 of lane S4-labels).

    * Inventory reduced to one clip per sha256 first (byte-identical copies never appear twice).
    * Stroke windows, per press: candidates are the deduped clips of that press's framings whose sha256 was NOT
      already sampled for that press in `base`; same seeded round-robin over groups and window rule as the base list
      (rng [seed, 3, press index]); n primary + n_reserve reserves (ids <split>-sw<tag>-<press>-NN / -rK).
    * Frames, per camera: replacement grid frames whose (sha256, PTS) is not already a sampled frame of `base`
      (rng [seed, 4, camera]); ids <split>-f<tag>-c<cam>-NN.
    """
    _check_dev(inv, split)
    if base.split != "dev":
        raise ValueError("the base sample list is not a dev list")
    full_sha = dict(zip(inv["clip_id"], inv["sha256"]))        # before dedupe: base items may name dropped copies
    inv = dedupe_by_sha(inv)
    sha_of = dict(zip(inv["clip_id"], inv["sha256"]))
    pts_cache: dict[str, list[float]] = {}

    def pts(cid: str) -> list[float]:
        if cid not in pts_cache:
            pts_cache[cid] = sorted(float(x) for x in pts_fn(cid))
        return pts_cache[cid]

    def item_sha(clip_id: str) -> str:
        if clip_id not in full_sha:
            raise ValueError(f"base item clip {clip_id} is not in the dev inventory")
        return full_sha[clip_id]

    items: list[SampleItem] = []
    for cam in sorted(n_frames):
        rng = np.random.default_rng([seed, 4, cam])
        used = {(item_sha(i.clip_id), round(float(i.t_pts or 0.0), 6)) for i in base.items
                if i.kind == "frame" and i.camera == cam}
        sub = inv[inv["camera"] == cam]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: [(c, t) for c in sub.loc[sub["group_id"] == g, "clip_id"] for t in frame_grid(pts(c))
                     if (sha_of[c], round(t, 6)) not in used] for g in groups}
        picked = _round_robin(pools, order, n_frames[cam], rng, f"camera {cam} supplementary frames")
        gid = dict(zip(sub["clip_id"], sub["group_id"]))
        for k, (c, t) in enumerate(picked, 1):
            items.append(SampleItem(sample_id=f"{split}-f{tag}-c{cam}-{k:02d}", kind="frame", clip_id=c,
                                    group_id=gid[c], camera=_cam(cam), split="dev", t_pts=t))
    names = sorted(presses)
    for pi, press in enumerate(names):
        if press not in n_windows:
            continue
        rng = np.random.default_rng([seed, 3, pi])
        used = {item_sha(i.clip_id) for i in base.items if i.kind == "stroke_window" and i.press == press}
        sub = inv[inv["framing_id"].isin(presses[press]) & ~inv["sha256"].isin(sorted(used))]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: list(sub.loc[sub["group_id"] == g, "clip_id"]) for g in groups}
        n, nr = n_windows[press], n_reserve.get(press, 0)
        picked = _round_robin(pools, order, n + nr, rng, f"press {press} supplementary windows")
        row = sub.set_index("clip_id")
        for k, c in enumerate(picked, 1):
            t0, t1 = window_for(pts(c), rng)
            sid = f"{split}-sw{tag}-{press}-{k:02d}" if k <= n else f"{split}-sw{tag}-{press}-r{k - n}"
            items.append(SampleItem(sample_id=sid, kind="stroke_window", clip_id=c, group_id=str(row.loc[c, "group_id"]),
                                    camera=_cam(row.loc[c, "camera"]), split="dev", press=press, t0=t0, t1=t1))
    rule = (
        f"scripts/label_sample.py build_supplement, seed {seed}, split dev, supplement '{tag}' to the base list "
        f"{base.items_sha256}. Inventory reduced to one clip per sha256 (smallest clip_id) before sampling. Frames: "
        f"per camera {dict(sorted(n_frames.items()))} replacement grid frames whose (sha256, PTS) is not a base frame "
        f"(rng [seed, 4, camera]). Stroke windows: per press {n_windows} primary + {n_reserve} reserve, press framings "
        f"{presses}; candidates exclude every sha256 already sampled for that press in the base list; seeded "
        f"round-robin over groups (rng [seed, 3, press index]); window = whole clip if <= {MAX_WINDOW_S:g} s, else "
        f"{MAX_WINDOW_S:g} s from a seeded-random start snapped to a PTS; no motion detector, track or proposal is "
        f"read. Labelled in list order per labels/RULES.md 'Supplementary lists'. Times are container PTS seconds."
    )
    return SampleList(contract="twin.label_samples/1", split="dev", seed=seed, rule=rule, items=items,
                      items_sha256=sample_items_sha256(items))


# ------------------------------------------------------------------------------------------------ sealed TEST list


def _round_robin_upto(pools: dict[str, list], order: list[str], n: int, rng: np.random.Generator) -> list:
    """`_round_robin` without the supply check: the same draws while supply lasts (identical to `_round_robin` when
    the pools hold >= n items), and every remaining item when they hold fewer (the caller records the shortfall)."""
    picked: list = []
    while len(picked) < n and any(pools[g] for g in order):
        for g in order:
            if pools[g] and len(picked) < n:
                picked.append(pools[g].pop(int(rng.integers(len(pools[g])))))
    return picked


def sealed_inventory(split_csv: Path) -> pd.DataFrame:
    """The TEST rows of a split CSV (prereg format, with a framing_id column) as an inventory frame: clip_id,
    group_id, split, sha256, framing_id and camera (from the class prefix). Train/dev rows are never returned."""
    from twin.schemas.prereg import camera_class_of, parse_split

    text = split_csv.read_text(encoding="utf-8")
    rows = parse_split(text)                                   # header, sha256 and uniqueness checks
    rd = list(csv.DictReader(io.StringIO(text)))
    if not rd or "framing_id" not in rd[0]:
        raise ValueError(f"{split_csv.name} has no framing_id column (fail closed)")
    test = [r for r in rd if r["clip_id"] in rows and rows[r["clip_id"]].split == "test"]
    if not test:
        raise ValueError(f"{split_csv.name} has no test rows")
    return pd.DataFrame({
        "clip_id": [r["clip_id"] for r in test], "group_id": [r["group_id"] for r in test],
        "split": ["test"] * len(test), "sha256": [r["sha256"] for r in test],
        "framing_id": [r["framing_id"] for r in test], "camera": [camera_class_of(r["clip_id"])[0] for r in test]})


def check_sealed_clips(clips_dir: Path, test_ids: set[str], split_of: dict[str, str]) -> None:
    """Fail closed unless `clips_dir` holds exactly <test id>.mp4 for every test id: a dev/train id, an unknown id or
    a missing test clip refuses the draw. Only file names are read."""
    if not clips_dir.is_dir():
        raise SystemExit(f"sealed clips folder {clips_dir} does not exist")
    names = sorted(p.name for p in clips_dir.iterdir())
    for name in names:
        stem, _, ext = name.rpartition(".")
        if split_of.get(stem, "test") != "test":
            raise SystemExit(f"refusing: {name} is a {split_of[stem]} clip, not a test clip")
        if ext != "mp4" or stem not in test_ids:
            raise SystemExit(f"refusing: {name} is not <test clip id>.mp4 of the active split")
    missing = sorted(test_ids - {n[:-4] for n in names})
    if missing:
        raise SystemExit(f"refusing: {len(missing)} test clips are missing from {clips_dir}")


def sealed_rule_text(seed: int, n_frames: dict[int, int], n_windows: dict[str, int], n_reserve: int,
                     presses: dict[str, list[str]], shortfall: list[str]) -> str:
    return (
        f"scripts/label_sample.py --sealed, seed {seed} (SEALED_SEED, committed before the draw), split test, from the "
        f"test rows of the active split and the container PTS of the sealed clips (demux only). Test inventory reduced "
        f"to one clip per sha256 (smallest clip_id) before sampling. Frames: per camera "
        f"{dict(sorted(n_frames.items()))} (stratified, equal allocation); grid 1 frame every {GRID_STEP_S:g} s per "
        f"clip at t = pts0 + {GRID_OFFSET_S:g} + {GRID_STEP_S:g}k snapped to the nearest PTS; seeded round-robin over "
        f"the camera's test groups in seeded order (rng [seed, 1, camera]), one random remaining grid frame of the "
        f"group per turn; every sampled frame is labelled, empty frames included. Stroke windows: per press "
        f"{n_windows} primary + {n_reserve} reserve (ids -r1..), press framings {presses}; seeded round-robin over the "
        f"test groups of those framings (rng [seed, 2, press index]), one window per (press, clip): whole clip if <= "
        f"{MAX_WINDOW_S:g} s, else {MAX_WINDOW_S:g} s from a seeded-random start snapped to a PTS; no motion detector, "
        f"track or proposal is read. Reserves replace, in list order, primary windows whose press the labeller finds "
        f"not running or not scorable; every item is labelled. A stratum with fewer eligible items than its target is "
        f"drawn whole and never filled from another stratum. Shortfall: "
        + ("; ".join(shortfall) if shortfall else "none") + ". Times are container PTS seconds."
    )


def build_sealed_list(inv: pd.DataFrame, pts_fn: PtsFn, *, seed: int = SEALED_SEED,
                      n_frames: dict[int, int] | None = None, n_windows: dict[str, int] | None = None,
                      n_reserve: int = SEALED_RESERVE,
                      presses: dict[str, list[str]] | None = None) -> tuple[SampleList, list[str]]:
    """The sealed TEST sample list and its shortfall lines (also written in its rule). Same draws as
    build_sample_list(dedupe=True), except: split test, ids "test-...", and a short stratum is drawn whole."""
    n_frames = SEALED_FRAMES if n_frames is None else n_frames
    n_windows = SEALED_WINDOWS if n_windows is None else n_windows
    presses = SEALED_PRESSES if presses is None else presses
    if seed == DEFAULT_SEED:
        raise ValueError("the sealed list must not use the dev seed")
    if set(inv["split"]) - {"test"}:
        raise ValueError(f"sealed inventory rows outside test: {sorted(set(inv['split']) - {'test'})}")
    inv = dedupe_by_sha(inv)
    pts_cache: dict[str, list[float]] = {}

    def pts(cid: str) -> list[float]:
        if cid not in pts_cache:
            pts_cache[cid] = sorted(float(x) for x in pts_fn(cid))
        return pts_cache[cid]

    items: list[SampleItem] = []
    short: list[str] = []
    for cam in sorted(n_frames):
        rng = np.random.default_rng([seed, 1, cam])
        sub = inv[inv["camera"] == cam]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: [(c, t) for c in sub.loc[sub["group_id"] == g, "clip_id"] for t in frame_grid(pts(c))]
                 for g in groups}
        picked = _round_robin_upto(pools, order, n_frames[cam], rng)
        if len(picked) < n_frames[cam]:
            short.append(f"camera {cam} frames {len(picked)} of {n_frames[cam]}")
        gid = dict(zip(sub["clip_id"], sub["group_id"]))
        for k, (c, t) in enumerate(picked, 1):
            items.append(SampleItem(sample_id=f"test-f-c{cam}-{k:02d}", kind="frame", clip_id=c, group_id=gid[c],
                                    camera=_cam(cam), split="test", t_pts=t))
    names = sorted(presses)
    for pi, press in enumerate(names):
        if press not in n_windows:
            continue
        rng = np.random.default_rng([seed, 2, pi])
        sub = inv[inv["framing_id"].isin(presses[press])]
        groups = sorted(set(sub["group_id"]))
        order = [groups[i] for i in rng.permutation(len(groups))]
        pools = {g: list(sub.loc[sub["group_id"] == g, "clip_id"]) for g in groups}
        n = n_windows[press]
        picked = _round_robin_upto(pools, order, n + n_reserve, rng)
        if len(picked) < n + n_reserve:
            short.append(f"press {press} windows {len(picked)} of {n} primary + {n_reserve} reserve "
                         f"({min(len(picked), n)} primary, {max(0, len(picked) - n)} reserve)")
        row = sub.set_index("clip_id")
        for k, c in enumerate(picked, 1):
            t0, t1 = window_for(pts(c), rng)
            sid = f"test-sw-{press}-{k:02d}" if k <= n else f"test-sw-{press}-r{k - n}"
            items.append(SampleItem(sample_id=sid, kind="stroke_window", clip_id=c,
                                    group_id=str(row.loc[c, "group_id"]), camera=_cam(row.loc[c, "camera"]),
                                    split="test", press=press, t0=t0, t1=t1))
    test_ids = set(inv["clip_id"])
    if any(i.clip_id not in test_ids for i in items):
        raise ValueError("a sampled clip is not a test row (fail closed)")
    rule = sealed_rule_text(seed, n_frames, n_windows, n_reserve, presses, short)
    return (SampleList(contract="twin.label_samples/1", split="test", seed=seed, rule=rule, items=items,
                       items_sha256=sample_items_sha256(items)), short)


def main_sealed(clips_dir: Path, out: Path, split_csv: Path | None = None) -> int:
    """Draw and write the sealed TEST list (custodian only). Prints counts and hashes only."""
    from twin.schemas.prereg import active_split_path, parse_split

    if out.exists() or out.with_suffix(".sha256").exists():
        raise SystemExit(f"refusing to overwrite {out} (or its .sha256): a hashed sample list is never redrawn")
    split_csv = split_csv or active_split_path()
    rows = parse_split(split_csv.read_text(encoding="utf-8"))
    inv = sealed_inventory(split_csv)
    check_sealed_clips(clips_dir, set(inv["clip_id"]), {k: v.split for k, v in rows.items()})
    sl, short = build_sealed_list(inv, lambda c: container_pts(clips_dir / f"{c}.mp4"))
    data = dump_list(sl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    out.with_suffix(".sha256").write_text(hash_text(sl, data, out.name), encoding="utf-8", newline="\n")
    fr = [i for i in sl.items if i.kind == "frame"]
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    per = {p: sum(i.press == p for i in sw) for p in sorted(SEALED_WINDOWS)}
    print(f"sealed test list: {len(fr)} frames (camera 9: {sum(i.camera == 9 for i in fr)}, camera 14: "
          f"{sum(i.camera == 14 for i in fr)}), {len(sw)} stroke windows {per}; shortfall: "
          f"{'; '.join(short) if short else 'none'}")
    print(f"items_sha256 {sl.items_sha256}")
    print(f"file sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


def hash_text(sl: SampleList, file_bytes: bytes, name: str = "sample.json") -> str:
    return (f"{sl.items_sha256}  {name}#items (twin.labels sample_items_sha256)\n"
            f"{hashlib.sha256(file_bytes).hexdigest()}  {name}\n")


def container_pts(path: Path) -> list[float]:
    """Sorted container PTS (s) of every video packet of the clip (demux only, no decode)."""
    import av  # pyright: ignore[reportMissingImports]

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb = st.time_base
        assert tb is not None
        return sorted(float(p.pts * tb) for p in c.demux(st) if p.pts is not None)


def press_framings(rois_path: Path = ROIS, keep: tuple[str, ...] = ("P9a", "P9b")) -> dict[str, list[str]]:
    d = json.loads(rois_path.read_text(encoding="utf-8"))
    return {p["id"]: sorted(p["rois"]) for p in d["presses"] if p["id"] in keep}


def dump_list(sl: SampleList) -> bytes:
    return (json.dumps(sl.model_dump(mode="json"), indent=1) + "\n").encode("utf-8")


def build_committed(inv: pd.DataFrame, pts_fn: PtsFn, *, which: str, seed: int = DEFAULT_SEED,
                    base: SampleList | None = None) -> SampleList:
    """Rebuild a committed DEV list. `which`: "legacy-base" = labels/dev/sample.json (drawn before the fix-round-1
    dedupe, i.e. dedupe=False); "base" = the same draw with dedupe (NOT the committed list); "s1" = sample_s1.json
    against `base` (the committed base list)."""
    if which == "s1":
        if base is None:
            raise ValueError("the supplement needs the committed base list")
        return build_supplement(inv, pts_fn, S1_PRESSES, base, split="dev", seed=seed, n_windows=S1_WINDOWS,
                                n_reserve=S1_RESERVE, n_frames=S1_FRAMES, tag="s1")
    if which not in ("legacy-base", "base"):
        raise ValueError(f"unknown list {which!r}")
    return build_sample_list(inv, pts_fn, press_framings(), split="dev", seed=seed, n_frames=DEFAULT_FRAMES,
                             n_windows=DEFAULT_WINDOWS, n_reserve=DEFAULT_RESERVE, dedupe=which == "base")


def check_against(sl: SampleList, data: bytes, committed: Path) -> str | None:
    """None if `sl` (serialised as `data`) equals the committed list file byte for byte, else what differs."""
    want = SampleList.model_validate(json.loads(committed.read_text(encoding="utf-8")))
    if want.items_sha256 != sl.items_sha256:
        return f"items_sha256 {sl.items_sha256} != committed {want.items_sha256}"
    if committed.read_bytes() != data:
        return "same items, but the file bytes differ (rule text or format)"
    return None


def main(argv: list[str] | None = None) -> int:
    """Reproduce the committed lists (no file is written):
        python scripts/label_sample.py --legacy-base --check labels/dev/sample.json
        python scripts/label_sample.py --supplement labels/dev/sample.json --check labels/dev/sample_s1.json
    Writing a list needs --out to a path that does not exist yet: a hashed sample list is never overwritten."""
    from intercycle import esk

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=None, help="write the list here (refused if the file already exists)")
    ap.add_argument("--check", default=None, metavar="COMMITTED_JSON",
                    help="rebuild and compare with this committed list; writes nothing; exit 1 on a difference")
    ap.add_argument("--seed", type=int, default=None, help=f"dev seed (default {DEFAULT_SEED}; not with --sealed)")
    ap.add_argument("--legacy-base", action="store_true",
                    help="the base list as committed (labels/dev/sample.json, drawn before the dedupe fix)")
    ap.add_argument("--supplement", default=None, metavar="BASE_SAMPLE_JSON",
                    help="draw the fix-round-1 supplementary list (S1_*) against this base sample list")
    ap.add_argument("--sealed", action="store_true",
                    help=f"custodian only: draw the sealed TEST list (seed SEALED_SEED) to --out (default {SEALED_OUT})")
    ap.add_argument("--clips", default=str(SEALED_CLIPS), help="--sealed: the sealed test clips folder")
    a = ap.parse_args(argv)
    if a.sealed:
        if a.check or a.legacy_base or a.supplement or a.seed is not None:
            ap.error("--sealed takes only --clips and --out (its seed is the committed SEALED_SEED)")
        return main_sealed(Path(a.clips), Path(a.out) if a.out else SEALED_OUT)
    if a.seed is None:
        a.seed = DEFAULT_SEED
    if bool(a.out) == bool(a.check):
        ap.error("give exactly one of --out (a new file) or --check (a committed list)")
    if a.supplement and a.legacy_base:
        ap.error("--legacy-base and --supplement exclude each other")
    if a.out and (Path(a.out).exists() or Path(a.out).with_suffix(".sha256").exists()):
        raise SystemExit(f"refusing to overwrite {a.out} (or its .sha256): a hashed sample list is never redrawn in "
                         f"place; use --check to reproduce a committed list")
    inv = esk.load("dev")
    pts_fn: PtsFn = lambda c: container_pts(esk.clip_path(c))  # noqa: E731
    if a.supplement:
        base = SampleList.model_validate(json.loads(Path(a.supplement).read_text(encoding="utf-8")))
        sl = build_committed(inv, pts_fn, which="s1", seed=a.seed, base=base)
    else:
        sl = build_committed(inv, pts_fn, which="legacy-base" if a.legacy_base else "base", seed=a.seed)
    data = dump_list(sl)
    if a.check:
        diff = check_against(sl, data, Path(a.check))
        print(f"{a.check}: " + ("reproduced (items_sha256 " + sl.items_sha256 + ")" if diff is None else diff))
        return 0 if diff is None else 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    out.with_suffix(".sha256").write_text(hash_text(sl, data, out.name), encoding="utf-8", newline="\n")
    fr = [i for i in sl.items if i.kind == "frame"]
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    print(f"{len(fr)} frames, {len(sw)} stroke windows ({sum((i.t1 or 0) - (i.t0 or 0) for i in sw):.1f} s), "
          f"items_sha256 {sl.items_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
