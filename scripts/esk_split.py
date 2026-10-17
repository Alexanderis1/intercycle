"""Session groups and the train/dev/test split for the press-shop clips (PLAN_TWIN §3 P3, §5 "Split and labels").

Committed BEFORE Pass 1 finishes, and run exactly ONCE (inside the Modal container that holds the Pass 1 features, so
no test-clip image feature ever leaves Modal). Every threshold and the seed live in this file; changing any of them
after the run is an amendment.

Input (one per clip, written by cloud/modal_esk.py Pass 1):
    <features>/<clip_id>.npz   phash  uint64[n]       pHash series (1 fps; 4 fps for clips < 3 s)
                               bg     uint8[36,64,3]  per-pixel median background (RGB) over the sampled frames
                               luma   float32[4]      mean, std, p10, p90 of the luma over the sampled frames (0-255)
    <meta>.json                [{"clip_id", "camera", "cls", "sha256"}, ...]   cls = class prefix 0-7

Rule:
1. pHash link (near-duplicate / overlapping footage): a pair of clips of the same camera is linked when at least
   min(PHASH_MIN_FRAMES, n_frames) frames of the shorter series have a frame of the other series at Hamming distance
   <= PHASH_HAMMING_MAX (n_frames = length of the shorter series).
2. Agglomerative clustering (average linkage, per camera) on the distance
       d = W_BG * MAD(background RGB) + W_LUMA * |delta mean luma| + W_LAYOUT * MAD(object layout)
   where all terms are in grey levels (0-255) and "object layout" is the 16x9 grid of the mean Sobel gradient magnitude
   of the median-background luma (where static objects such as pallets, racks and machines stand), cut at
   CLUSTER_DISTANCE.
3. Groups = connected components of (pHash links  OR  same cluster).  Groups never span cameras.
4. Split 60/20/20 by group, stratified by camera and safe/unsafe: each group's stratum is its majority class type
   (unsafe = prefixes 0-3, safe = 4-7; ties -> unsafe). Within each (camera, stratum), groups are shuffled with the
   seeded RNG, stable-sorted by clip count (largest first), and each goes to the split with the largest remaining
   deficit of clips against the 60/20/20 target (ties -> train, dev, test).

Output: split.csv (clip_id, group_id, split, sha256), sorted by clip_id, plus a JSON summary of groups.

    python scripts/esk_split.py --features DIR --meta meta.json --out split.csv [--summary summary.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- pre-registered constants
SEED = 20261017
PHASH_FPS = 1.0  # sampling rate of the pHash series (Pass 1)
PHASH_FPS_SHORT = 4.0  # for clips shorter than SHORT_CLIP_S
SHORT_CLIP_S = 3.0
PHASH_HAMMING_MAX = 6
PHASH_MIN_FRAMES = 3
BG_SIZE = (64, 36)  # (w, h) of the median background thumbnail
LAYOUT_GRID = (16, 9)  # (w, h) cells of the object-layout map (4x4 thumbnail pixels per cell)
W_BG = 1.0
W_LUMA = 0.5
W_LAYOUT = 1.0
CLUSTER_LINKAGE = "average"
CLUSTER_DISTANCE = 8.0  # grey levels
SPLIT_FRACTIONS = (("train", 0.6), ("dev", 0.2), ("test", 0.2))
UNSAFE_PREFIXES = (0, 1, 2, 3)
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}


# ----------------------------------------------------------------------------- features
def layout_map(bg: np.ndarray) -> np.ndarray:
    """16x9 mean Sobel gradient magnitude of the background luma (grey levels)."""
    rgb = bg.astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    p = np.pad(y, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.hypot(gx, gy) / 8.0
    h, w = mag.shape
    gw, gh = LAYOUT_GRID
    return mag.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))


def phash_linked(a: np.ndarray, b: np.ndarray) -> bool:
    """Rule 1 for one pair of pHash series."""
    if len(a) == 0 or len(b) == 0:
        return False
    if len(a) > len(b):
        a, b = b, a
    ham = np.bitwise_count(np.bitwise_xor(a[:, None], b[None, :]))
    hits = int((ham.min(axis=1) <= PHASH_HAMMING_MAX).sum())
    return hits >= min(PHASH_MIN_FRAMES, len(a))


def pair_distance(f1: dict, f2: dict) -> float:
    d_bg = float(np.abs(f1["bg"].astype(np.float32) - f2["bg"].astype(np.float32)).mean())
    d_luma = abs(float(f1["luma"][0]) - float(f2["luma"][0]))
    d_lay = float(np.abs(f1["layout"] - f2["layout"]).mean())
    return W_BG * d_bg + W_LUMA * d_luma + W_LAYOUT * d_lay


# ----------------------------------------------------------------------------- grouping
class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def group_camera(ids: list[str], feats: dict[str, dict]) -> tuple[list[int], dict]:
    """Group the clips of one camera. Returns a component index per clip and diagnostics."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(ids)
    dsu = _DSU(n)
    n_links = 0
    for i in range(n):
        for j in range(i + 1, n):
            if phash_linked(feats[ids[i]]["phash"], feats[ids[j]]["phash"]):
                dsu.union(i, j)
                n_links += 1
    n_clusters = n
    if n > 1:
        cond = np.array([pair_distance(feats[ids[i]], feats[ids[j]]) for i in range(n) for j in range(i + 1, n)])
        lab = fcluster(linkage(cond, method=CLUSTER_LINKAGE), t=CLUSTER_DISTANCE, criterion="distance")
        n_clusters = int(len(set(lab.tolist())))
        first: dict[int, int] = {}
        for i, c in enumerate(lab.tolist()):
            if c in first:
                dsu.union(first[c], i)
            else:
                first[c] = i
    comp = [dsu.find(i) for i in range(n)]
    return comp, {"n_clips": n, "phash_links": n_links, "agglomerative_clusters": n_clusters}


def assign_splits(groups: dict[str, dict], seed: int = SEED) -> dict[str, str]:
    """Rule 4. groups: gid -> {"camera", "n_clips", "n_unsafe"}. Returns gid -> split."""
    names = [s for s, _ in SPLIT_FRACTIONS]
    fr = np.array([f for _, f in SPLIT_FRACTIONS])
    out: dict[str, str] = {}
    strata: dict[tuple[int, int], list[str]] = {}
    for gid, g in groups.items():
        unsafe = 1 if 2 * g["n_unsafe"] >= g["n_clips"] else 0
        strata.setdefault((int(g["camera"]), unsafe), []).append(gid)
    for (cam, unsafe), gids in sorted(strata.items()):
        rng = np.random.default_rng([seed, cam, unsafe])
        gids = sorted(gids)
        order = [gids[k] for k in rng.permutation(len(gids))]
        order.sort(key=lambda g: -groups[g]["n_clips"])  # stable: ties keep the seeded order
        total = sum(groups[g]["n_clips"] for g in order)
        got = np.zeros(len(names))
        for g in order:
            deficit = fr * total - got
            k = int(np.argmax(deficit))  # argmax returns the first max: ties -> train, dev, test
            out[g] = names[k]
            got[k] += groups[g]["n_clips"]
    return out


def make_split(meta: list[dict], feats: dict[str, dict], seed: int = SEED) -> tuple[list[dict], dict]:
    """Rules 1-4. meta rows: clip_id, camera, cls, sha256. feats: clip_id -> {phash, bg, luma, layout}."""
    rows_out: list[dict] = []
    groups: dict[str, dict] = {}
    gid_of: dict[str, str] = {}
    diag: dict = {"constants": {k: v for k, v in globals().items() if k.isupper()}, "cameras": {}}
    for cam in sorted({int(m["camera"]) for m in meta}):
        ids = sorted(m["clip_id"] for m in meta if int(m["camera"]) == cam)
        comp, d = group_camera(ids, feats)
        # stable group ids: number components by their smallest clip_id
        roots = sorted(set(comp), key=lambda r: min(ids[i] for i in range(len(ids)) if comp[i] == r))
        name = {r: f"c{cam}_g{k:03d}" for k, r in enumerate(roots)}
        for i, cid in enumerate(ids):
            gid_of[cid] = name[comp[i]]
        d["groups"] = len(roots)
        diag["cameras"][str(cam)] = d
    by_id = {m["clip_id"]: m for m in meta}
    for cid, gid in gid_of.items():
        g = groups.setdefault(gid, {"camera": int(by_id[cid]["camera"]), "n_clips": 0, "n_unsafe": 0})
        g["n_clips"] += 1
        g["n_unsafe"] += int(int(by_id[cid]["cls"]) in UNSAFE_PREFIXES)
    split_of = assign_splits(groups, seed)
    for cid in sorted(gid_of):
        rows_out.append({"clip_id": cid, "group_id": gid_of[cid], "split": split_of[gid_of[cid]],
                         "sha256": by_id[cid]["sha256"]})
    diag["groups"] = {g: dict(v, split=split_of[g]) for g, v in sorted(groups.items())}
    return rows_out, diag


def load_features(features_dir: Path, ids: list[str]) -> dict[str, dict]:
    out = {}
    for cid in ids:
        z = np.load(features_dir / f"{cid}.npz")
        bg = z["bg"]
        out[cid] = {"phash": z["phash"].astype(np.uint64), "bg": bg, "luma": z["luma"], "layout": layout_map(bg)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args(argv)
    meta = json.loads(a.meta.read_text(encoding="utf-8"))
    feats = load_features(a.features, [m["clip_id"] for m in meta])
    rows, diag = make_split(meta, feats)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "group_id", "split", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if a.summary:
        a.summary.write_text(json.dumps(diag, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"clips": len(rows), "cameras": diag["cameras"]}, default=str))


if __name__ == "__main__":
    main()
