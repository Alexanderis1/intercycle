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

    python scripts/label_sample.py --out labels/dev/sample.json      (GPU env: needs PyAV for the PTS lists)
"""
from __future__ import annotations

import argparse
import hashlib
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


def main(argv: list[str] | None = None) -> int:
    from intercycle import esk

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--supplement", default=None, metavar="BASE_SAMPLE_JSON",
                    help="draw the fix-round-1 supplementary list (S1_*) against this base sample list")
    a = ap.parse_args(argv)
    inv = esk.load("dev")
    pts_fn: PtsFn = lambda c: container_pts(esk.clip_path(c))  # noqa: E731
    if a.supplement:
        base = SampleList.model_validate(json.loads(Path(a.supplement).read_text(encoding="utf-8")))
        sl = build_supplement(inv, pts_fn, S1_PRESSES, base, split="dev", seed=a.seed, n_windows=S1_WINDOWS,
                              n_reserve=S1_RESERVE, n_frames=S1_FRAMES, tag="s1")
        default_out = ROOT / "labels" / "dev" / "sample_s1.json"
    else:
        sl = build_sample_list(inv, pts_fn, press_framings(), split="dev", seed=a.seed, n_frames=DEFAULT_FRAMES,
                               n_windows=DEFAULT_WINDOWS, n_reserve=DEFAULT_RESERVE)
        default_out = ROOT / "labels" / "dev" / "sample.json"
    out = Path(a.out) if a.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(sl.model_dump(mode="json"), indent=1) + "\n").encode("utf-8")
    out.write_bytes(data)
    out.with_suffix(".sha256").write_text(hash_text(sl, data, out.name), encoding="utf-8", newline="\n")
    fr = [i for i in sl.items if i.kind == "frame"]
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    print(f"{len(fr)} frames, {len(sw)} stroke windows ({sum((i.t1 or 0) - (i.t0 or 0) for i in sw):.1f} s), "
          f"items_sha256 {sl.items_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
