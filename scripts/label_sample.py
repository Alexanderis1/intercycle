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


def build_sample_list(inv: pd.DataFrame, pts_fn: PtsFn, presses: dict[str, list[str]], *, split: str, seed: int,
                      n_frames: dict[int, int], n_windows: dict[str, int], n_reserve: int) -> SampleList:
    if split != "dev":
        raise ValueError("this sampler draws the dev split only (the test labeller works in the custodian session)")
    if set(inv["split"]) - {"dev"}:
        raise ValueError(f"inventory rows outside dev: {sorted(set(inv['split']) - {'dev'})}")
    inv = inv.sort_values("clip_id").reset_index(drop=True)
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
    return SampleList(contract="twin.label_samples/1", split="dev", seed=seed,
                      rule=rule_text(seed, n_frames, n_windows, n_reserve, presses), items=items,
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
    ap.add_argument("--out", default=str(ROOT / "labels" / "dev" / "sample.json"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    inv = esk.load("dev")
    sl = build_sample_list(inv, lambda c: container_pts(esk.clip_path(c)), press_framings(), split="dev",
                           seed=a.seed, n_frames=DEFAULT_FRAMES, n_windows=DEFAULT_WINDOWS, n_reserve=DEFAULT_RESERVE)
    out = Path(a.out)
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
