"""scripts/esk_split.py: pHash link rule, grouping and the seeded group split (synthetic features, no video)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("esk_split", Path(__file__).resolve().parents[1] / "scripts" / "esk_split.py")
assert _SPEC and _SPEC.loader
S = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S)


def _flip(h: int, nbits: int, rng) -> int:
    for b in rng.choice(64, size=nbits, replace=False):
        h ^= 1 << int(b)
    return h


def test_phash_link_rule():
    rng = np.random.default_rng(0)
    base = [int(x) for x in rng.integers(0, 2**63, size=5)]
    a = np.array(base, dtype=np.uint64)
    near = np.array([_flip(h, 6, rng) for h in base], dtype=np.uint64)
    far = np.array([_flip(h, 7, rng) for h in base], dtype=np.uint64)
    assert S.phash_linked(a, near)  # Hamming 6 on every frame
    assert not S.phash_linked(a, far)  # Hamming 7: no frame matches
    two = np.array([base[0], _flip(base[1], 20, rng), _flip(base[2], 20, rng)], dtype=np.uint64)
    assert not S.phash_linked(a, two)  # 1 matching frame < min(3, 3)
    one = np.array([base[3]], dtype=np.uint64)
    assert S.phash_linked(a, one)  # 1-frame clip: min(3, 1) = 1


def _session(rng, cam: int, sess: int, n: int, cls: list[int]):
    bg = rng.integers(0, 255, size=(36, 64, 3)).astype(np.uint8)
    meta, feats = [], {}
    for k in range(n):
        cid = f"{cls[k % len(cls)]}_s{cam}x{sess}x{k}"
        noisy = np.clip(bg.astype(int) + rng.integers(-2, 3, size=bg.shape), 0, 255).astype(np.uint8)
        ph = np.array([int(x) for x in rng.integers(0, 2**63, size=4)], dtype=np.uint64)
        feats[cid] = {"phash": ph, "bg": noisy, "luma": np.array([noisy.mean(), 1, 0, 0], np.float32),
                      "layout": S.layout_map(noisy)}
        meta.append({"clip_id": cid, "camera": cam, "cls": cls[k % len(cls)], "sha256": f"{cid}sha"})
    return meta, feats


def test_groups_follow_sessions_and_split_is_by_group_and_deterministic():
    rng = np.random.default_rng(1)
    meta, feats = [], {}
    for s in range(20):
        m, f = _session(rng, 9, s, 3 + s % 4, [0, 4])
        meta += m
        feats.update(f)
    for s in range(8):
        m, f = _session(rng, 14, s, 2 + s % 3, [3, 7])
        meta += m
        feats.update(f)
    rows, diag = S.make_split(meta, feats)
    assert diag["cameras"]["9"]["groups"] == 20 and diag["cameras"]["14"]["groups"] == 8
    sess_of = {r["clip_id"]: r["clip_id"].split("_s")[1].rsplit("x", 1)[0] for r in rows}
    by_group: dict[str, set] = {}
    split_of_group: dict[str, set] = {}
    for r in rows:
        by_group.setdefault(r["group_id"], set()).add(sess_of[r["clip_id"]])
        split_of_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())  # one session per group
    assert all(len(v) == 1 for v in split_of_group.values())  # a group never spans splits
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    assert n["train"] > n["dev"] > 0 and n["test"] > 0
    rows2, _ = S.make_split(meta, feats)
    assert rows == rows2  # seeded and deterministic


def test_phash_link_merges_distinct_backgrounds():
    rng = np.random.default_rng(2)
    m1, f1 = _session(rng, 9, 0, 1, [0])
    m2, f2 = _session(rng, 9, 1, 1, [0])
    f2[m2[0]["clip_id"]]["phash"] = f1[m1[0]["clip_id"]]["phash"].copy()
    rows, diag = S.make_split(m1 + m2, {**f1, **f2})
    assert len({r["group_id"] for r in rows}) == 1


# ----------------------------------------------------------------------------- amendment A1+
def _a1_data():
    rng = np.random.default_rng(3)
    meta, feats = [], {}
    for s in range(16):
        m, f = _session(rng, 9, s, 2 + s % 3, [0, 4])
        meta += m
        feats.update(f)
    for s in range(10):
        m, f = _session(rng, 14, s, 2 + s % 2, [3, 7])
        meta += m
        feats.update(f)
    return meta, feats


def test_a1plus_ignores_phash_and_merges_byte_identical():
    meta, feats = _a1_data()
    ph = feats[meta[0]["clip_id"]]["phash"]
    for m in meta:  # identical pHash everywhere: the committed rule would make one group per camera
        if m["camera"] == 9:
            feats[m["clip_id"]]["phash"] = ph.copy()
    a = next(m for m in meta if "_s9x0x" in m["clip_id"])
    b = next(m for m in meta if "_s9x5x" in m["clip_id"])
    b["sha256"] = a["sha256"]  # byte-identical file in another cluster
    framing = {m["clip_id"]: f"c{m['camera']}_f0" for m in meta}
    rows, diag = S.make_split_a1plus(meta, feats, set(), framing)
    assert diag["cameras"]["9"]["agglomerative_clusters"] == 16
    assert diag["cameras"]["9"]["groups"] == 15 and diag["cameras"]["9"]["cluster_merges_by_sha256"] == 1
    g = {r["clip_id"]: r["group_id"] for r in rows}
    assert g[a["clip_id"]] == g[b["clip_id"]]
    assert {r["framing_id"] for r in rows} == {"c9_f0", "c14_f0"}
    rows2, _ = S.make_split_a1plus(meta, feats, set(), framing)
    assert rows == rows2


def test_a1plus_viewed_and_framing_rules():
    meta, feats = _a1_data()
    framing = {m["clip_id"]: f"c{m['camera']}_f0" for m in meta}
    lone = [m["clip_id"] for m in meta if "_s9x3x" in m["clip_id"]]
    for c in lone:  # a framing seen in one group only can never reach test
        framing[c] = "c9_f9"
    rows0, _ = S.make_split_a1plus(meta, feats, set(), framing)
    test0 = {r["group_id"] for r in rows0 if r["split"] == "test"}
    assert test0
    viewed = {next(r["clip_id"] for r in rows0 if r["group_id"] == g) for g in test0}
    rows, diag = S.make_split_a1plus(meta, feats, viewed, framing)
    split = {r["clip_id"]: r["split"] for r in rows}
    assert all(split[c] != "test" for c in viewed)
    assert all(split[c] != "test" for c in lone)
    train_fr = {r["framing_id"] for r in rows if r["split"] == "train"}
    assert all(r["framing_id"] in train_fr for r in rows if r["split"] == "test")
    sp: dict[str, set] = {}
    for r in rows:
        sp.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in sp.values())
    assert any(r["split"] == "test" for r in rows)


def test_a1plus_cli_writes_framing_column(tmp_path):
    import json

    meta, feats = _a1_data()
    fd = tmp_path / "feat"
    fd.mkdir()
    for cid, f in feats.items():
        np.savez(fd / f"{cid}.npz", phash=f["phash"], bg=f["bg"], luma=f["luma"])
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "viewed.txt").write_text("# none\n", encoding="utf-8")
    (tmp_path / "framing.csv").write_text("clip_id,framing_id\n" + "".join(
        f"{m['clip_id']},c{m['camera']}_f0\n" for m in meta), encoding="utf-8")
    out = tmp_path / "split_a1.csv"
    S.main(["--features", str(fd), "--meta", str(tmp_path / "meta.json"), "--out", str(out), "--rule", "a1plus",
            "--viewed", str(tmp_path / "viewed.txt"), "--framing", str(tmp_path / "framing.csv")])
    assert out.read_text(encoding="utf-8").splitlines()[0] == "clip_id,group_id,split,sha256,framing_id"
