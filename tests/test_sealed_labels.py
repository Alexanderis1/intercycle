"""Sealed TEST label tooling (lane S4-labels, sealed mode): label_sample --sealed, label_extract --sealed,
custodian_seal and labels/TEST_RULES.md. Synthetic inputs only (tiny mp4s written in tmp); no test data is read."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twin import schemas as S
from twin.schemas import prereg
from twin.schemas.labels import SampleItem, SampleList, sample_items_sha256

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LS = _load("label_sample")
LE = _load("label_extract")
CS = _load("custodian_seal")

# (clip_id, group_id, split, framing_id, duration s); class prefix 0/1/6 = camera 9, 3/7 = camera 14
ROWS = [
    ("0_tr1", "c9_a0", "train", "c9_f1", 6.0), ("0_dv1", "c9_a1", "dev", "c9_f1", 6.0),
    ("0_te1", "c9_a7", "test", "c9_f1", 12.0), ("0_te2", "c9_a7", "test", "c9_f1", 6.0),
    ("1_te3", "c9_a8", "test", "c9_f1", 8.0), ("1_te4", "c9_a8", "test", "c9_f1", 8.0),  # 1_te4 = copy of 1_te3
    ("6_te5", "c9_a9", "test", "c9_f0", 11.0),
    ("3_te6", "c14_a1", "test", "c14_f0", 5.0), ("7_te7", "c14_a2", "test", "c14_f0", 3.0),
    ("3_dv2", "c14_a3", "dev", "c14_f0", 4.0),
]


def _sha(cid: str) -> str:
    return hashlib.sha256(("1_te3" if cid == "1_te4" else cid).encode()).hexdigest()


def _split_csv(tmp: Path) -> Path:
    p = tmp / "split_t.csv"
    lines = ["clip_id,group_id,split,sha256,framing_id"]
    lines += [f"{c},{g},{s},{_sha(c)},{f}" for c, g, s, f, _ in ROWS]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def _pts_fn():
    dur = {c: d for c, _, _, _, d in ROWS}
    return lambda cid: [round(i / 25.0, 6) for i in range(int(round(dur[cid] * 25)))]


# ------------------------------------------------------------------------------------------------ label_sample


def test_sealed_inventory_keeps_test_rows_with_group_framing_and_camera(tmp_path):
    inv = LS.sealed_inventory(_split_csv(tmp_path))
    assert set(inv["split"]) == {"test"}
    assert sorted(inv["clip_id"]) == sorted(c for c, _, s, _, _ in ROWS if s == "test")
    r = inv.set_index("clip_id")
    assert r.loc["6_te5", "framing_id"] == "c9_f0" and r.loc["6_te5", "camera"] == 9
    assert r.loc["7_te7", "camera"] == 14 and r.loc["3_te6", "group_id"] == "c14_a1"


def test_round_robin_upto_equals_round_robin_with_enough_supply():
    pools = {"a": list(range(5)), "b": list(range(10, 13)), "c": [20]}
    order = ["b", "a", "c"]
    got = LS._round_robin_upto({k: list(v) for k, v in pools.items()}, order, 6, np.random.default_rng(3))
    want = LS._round_robin({k: list(v) for k, v in pools.items()}, order, 6, np.random.default_rng(3), "x")
    assert got == want
    short = LS._round_robin_upto({k: list(v) for k, v in pools.items()}, order, 99, np.random.default_rng(3))
    assert sorted(short) == sorted(sum(pools.values(), []))


def test_sealed_list_mirrors_the_dev_draw_when_supply_suffices():
    """Same inventory relabelled dev / test, same seed and volumes: same draws, ids and split aside."""
    rows = [(c, g, f, d) for c, g, s, f, d in ROWS if s == "test"]
    base = pd.DataFrame({"clip_id": [r[0] for r in rows], "group_id": [r[1] for r in rows],
                         "framing_id": [r[2] for r in rows], "sha256": [_sha(r[0]) for r in rows],
                         "camera": [prereg.camera_class_of(r[0])[0] for r in rows]})
    kw = dict(n_frames={9: 4, 14: 2}, n_windows={"P9a": 1, "P9b": 2}, n_reserve=0)
    presses = {"P9a": ["c9_f0"], "P9b": ["c9_f1"]}
    dev = LS.build_sample_list(base.assign(split="dev"), _pts_fn(), presses, split="dev", seed=99, dedupe=True, **kw)
    test, short = LS.build_sealed_list(base.assign(split="test"), _pts_fn(), seed=99, presses=presses, **kw)
    assert test.split == "test" and len(test.items) == len(dev.items)
    for a, b in zip(dev.items, test.items):
        da, db = a.model_dump(), b.model_dump()
        assert da.pop("sample_id").replace("dev-", "test-", 1) == db.pop("sample_id")
        assert da.pop("split") == "dev" and db.pop("split") == "test"
        assert da == db
    assert short == []


def test_sealed_list_records_shortfall_and_never_fills_from_another_stratum(tmp_path):
    inv = LS.sealed_inventory(_split_csv(tmp_path))
    sl, short = LS.build_sealed_list(inv, _pts_fn())
    S.SampleList.model_validate(sl.model_dump(mode="json"))
    assert sl.seed == LS.SEALED_SEED != LS.DEFAULT_SEED and sl.split == "test"
    fr = [i for i in sl.items if i.kind == "frame"]
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    # camera 14 grid: 3_te6 (5 s) -> 1, 3, 5? no: last pts 4.96 -> 1, 3; 7_te7 (3 s) -> 1  => 3 frames
    assert sum(i.camera == 14 for i in fr) == 3 and all(i.clip_id[0] in "37" for i in fr if i.camera == 14)
    assert "camera 14 frames 3 of 20" in short and "camera 9 frames" in "; ".join(short)
    p9a = [i for i in sw if i.press == "P9a"]
    assert [i.clip_id for i in p9a] == ["6_te5"] and p9a[0].sample_id == "test-sw-P9a-01"
    assert {i.clip_id for i in sw if i.press == "P9b"} <= {"0_te1", "0_te2", "1_te3"}       # c9_f1, deduped
    assert "1_te4" not in {i.clip_id for i in sl.items}                                   # byte copy dropped
    assert all(i.t1 - i.t0 <= 10.0 + 1e-6 for i in sw)
    assert "Shortfall: camera 9 frames" in sl.rule and "press P9a windows 1 of 10 primary + 3 reserve" in sl.rule
    again, _ = LS.build_sealed_list(inv, _pts_fn())
    assert again.items_sha256 == sl.items_sha256


def test_sealed_list_refuses_dev_seed_and_non_test_rows(tmp_path):
    inv = LS.sealed_inventory(_split_csv(tmp_path))
    with pytest.raises(ValueError, match="dev seed"):
        LS.build_sealed_list(inv, _pts_fn(), seed=LS.DEFAULT_SEED)
    bad = inv.copy()
    bad.loc[0, "split"] = "dev"
    with pytest.raises(ValueError, match="outside test"):
        LS.build_sealed_list(bad, _pts_fn())


def test_check_sealed_clips_refuses_dev_unknown_and_missing(tmp_path):
    split_of = {c: s for c, _, s, _, _ in ROWS}
    test_ids = {c for c, s in split_of.items() if s == "test"}
    d = tmp_path / "clips"
    d.mkdir()
    for c in test_ids:
        (d / f"{c}.mp4").write_bytes(b"x")
    LS.check_sealed_clips(d, test_ids, split_of)
    (d / "0_dv1.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit, match="dev clip"):
        LS.check_sealed_clips(d, test_ids, split_of)
    (d / "0_dv1.mp4").unlink()
    (d / "9_zz.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit, match="not <test clip id>.mp4"):
        LS.check_sealed_clips(d, test_ids, split_of)
    (d / "9_zz.mp4").unlink()
    (d / "0_te1.mp4").unlink()
    with pytest.raises(SystemExit, match="missing"):
        LS.check_sealed_clips(d, test_ids, split_of)


def test_main_sealed_arguments():
    with pytest.raises(SystemExit):
        LS.main(["--sealed", "--check", "labels/dev/sample.json"])
    with pytest.raises(SystemExit):
        LS.main(["--sealed", "--seed", "5"])


# ------------------------------------------------------------------------------------------------ end to end


def _write_mp4(path: Path, seconds: float, fps: int = 5, w: int = 1920, h: int = 1080) -> None:
    av = pytest.importorskip("av")
    from fractions import Fraction

    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.time_base = Fraction(1, fps)
        for k in range(int(seconds * fps)):
            img = np.full((h, w, 3), (k * 7) % 255, np.uint8)
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts = k
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


@pytest.fixture()
def sealed_root(tmp_path, monkeypatch, capsys):
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    split = _split_csv(tmp_path)
    monkeypatch.setattr(prereg, "active_split_path", lambda repo=None: split)
    root = tmp_path / "sealed_store"
    (root / "clips").mkdir(parents=True)
    for c, _, s, _, d in ROWS:
        if s == "test":
            _write_mp4(root / "clips" / f"{c}.mp4", min(d, 4.0))
    out = root / "labels" / "sample.json"
    assert LS.main_sealed(root / "clips", out, split) == 0
    return root, split, capsys.readouterr().out


def test_end_to_end_sample_extract_and_seal(sealed_root, tmp_path):
    root, split, printed = sealed_root
    out = root / "labels" / "sample.json"
    sl = S.SampleList.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert sl.items_sha256 in printed and not any(c in printed for c, *_ in ROWS)    # counts and hashes only
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        LS.main_sealed(root / "clips", out, split)

    sealed = LE.Sealed(root)
    LE.cmd_frames(sealed=sealed)
    LE.cmd_windows(sealed=sealed)
    fr = [i for i in sl.items if i.kind == "frame"]
    sw = [i for i in sl.items if i.kind == "stroke_window"]
    assert all((root / "labelling" / "frames" / f"{i.sample_id}.png").is_file() for i in fr)
    assert all((root / "labelling" / "windows" / f"{i.sample_id}.npz").is_file() for i in sw)
    assert LE.cmd_crop(fr[0].sample_id, 0, 0, 200, 200, 20, None, sealed=sealed).is_file()
    w = sw[0]
    box = np.load(root / "labelling" / "windows" / f"{w.sample_id}.npz")["box"]
    assert LE.cmd_kymo(w.sample_id, int(box[0]) + 2, int(box[0]) + 6, sealed=sealed)

    rules = hashlib.sha256(b"synthetic rules").hexdigest()
    test_ids = {c for c, _, s, _, _ in ROWS if s == "test"}
    repo_out = tmp_path / "repo_sealed"
    line = CS.cmd_sample(root / "labels", repo_out / "SAMPLE.sha256", test_ids)
    assert line.startswith(sl.items_sha256 + "  sample.json#items frames=") and line.count("\n") == 1

    def labelset(items):
        return {"contract": "twin.labels/1", "split": "test", "labeller_id": "agent:t", "labeller_kind": "claude_agent",
                "blind": True, "proposals_shown": False, "rules_sha256": rules, "sample_list_sha256": sl.items_sha256,
                "frames": [{"sample_id": i.sample_id, "clip_id": i.clip_id, "t_pts": i.t_pts, "boxes": []}
                           for i in items if i.kind == "frame"],
                "stroke_windows": [{"sample_id": i.sample_id, "clip_id": i.clip_id, "press": i.press, "t0": i.t0,
                                    "t1": i.t1, "strokes": []} for i in items if i.kind == "stroke_window"]}

    lab = root / "labels"
    (lab / "labels_a.json").write_text(json.dumps(labelset(sl.items[:3])), encoding="utf-8")
    with pytest.raises(SystemExit, match="BEFORE labelling"):
        CS.cmd_sample(lab, tmp_path / "other" / "SAMPLE.sha256", test_ids)
    with pytest.raises(SystemExit, match="not labelled"):
        CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, rules)
    (lab / "labels_b.json").write_text(json.dumps(labelset(sl.items[2:])), encoding="utf-8")
    with pytest.raises(SystemExit, match="labelled twice"):
        CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, rules)
    (lab / "labels_b.json").write_text(json.dumps(labelset(sl.items[3:])), encoding="utf-8")
    with pytest.raises(SystemExit, match="rules_sha256"):
        CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, "0" * 64)
    text, counts = CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, rules)
    assert counts == {"label_files": 2, "frames": len(fr), "stroke_windows": len(sw)}
    back = prereg.parse_sealed((repo_out / "LABELS.sha256").read_text(encoding="utf-8"))
    assert back.kind == "LABELS" and back.sample_list_sha256 == sl.items_sha256
    assert [f.path for f in back.files] == ["labels/labels_a.json", "labels/labels_b.json", "labels/sample.json"]
    assert back.files[2].sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert back.ids == sorted({i.clip_id for i in sl.items})
    CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, rules)                     # same inputs: no-op
    (lab / "labels_b.json").write_text(json.dumps(labelset(sl.items[3:])) + " ", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        CS.cmd_labels(lab, repo_out / "LABELS.sha256", test_ids, rules)


def test_extract_refuses_tampered_lists_and_non_test_items(sealed_root):
    root, _, _ = sealed_root
    sealed = LE.Sealed(root)
    assert len(LE.sealed_items(sealed)) > 0
    p = root / "labels" / "sample.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(d), encoding="utf-8")                                         # bytes changed
    with pytest.raises(SystemExit, match="do not match"):
        LE.sealed_items(sealed)
    # a valid, hashed list that names a dev clip is refused as a whole
    it = SampleItem(sample_id="test-f-c9-01", kind="frame", clip_id="0_dv1", group_id="c9_a1", camera=9,
                      split="test", t_pts=1.0)
    sl = SampleList(contract="twin.label_samples/1", split="test", seed=1, rule="r", items=[it],
                      items_sha256=sample_items_sha256([it]))
    data = LS.dump_list(sl)
    p.write_bytes(data)
    p.with_suffix(".sha256").write_text(LS.hash_text(sl, data, p.name), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a test clip"):
        LE.sealed_items(sealed)
    with pytest.raises(SystemExit, match="not test clips"):
        CS.load_sample(root / "labels", {c for c, _, s, _, _ in ROWS if s == "test"})


def test_sealed_store_refuses_repo_and_data_root(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="inside the repo"):
        LE.sealed_store(LE.Sealed(REPO / "tmp_sealed_should_not_exist"))
    monkeypatch.setenv("INTERCYCLE_DATA", str(tmp_path))
    with pytest.raises(SystemExit, match="INTERCYCLE_DATA"):
        LE.sealed_store(LE.Sealed(tmp_path / "x"))
    assert not (REPO / "tmp_sealed_should_not_exist").exists()


def test_extract_main_flags():
    with pytest.raises(SystemExit):
        LE.main(["--sealed", "--sample", "labels/dev/sample.json", "frames"])
    with pytest.raises(SystemExit):
        LE.main(["--sealed-root", "x", "frames"])


# ------------------------------------------------------------------------------------------------ TEST_RULES.md


def test_test_rules_example_validates_and_holds_no_dev_specifics():
    text = (REPO / "labels" / "TEST_RULES.md").read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)```", text, re.S)
    assert block
    ex = block.group(1).replace("<sha256 of labels/TEST_RULES.md>", "a" * 64) \
        .replace("<items_sha256 of sample.json>", "b" * 64).replace("<clip_id>", "0_x")
    S.LabelSet.model_validate(json.loads(ex))
    split = (REPO / "prereg" / "split_a1.csv").read_text(encoding="utf-8").splitlines()[1:]
    ids = {ln.split(",")[0] for ln in split if ln}
    words = set(re.findall(r"[0-7]_[a-z]+[0-9]+", text))
    assert not words & ids, "a real clip id appears in TEST_RULES.md"
    for dev in ("labels/dev", "scored.json", "agreement", "sample_s1", "dev_labels", "Revisions"):
        assert dev not in text
    assert not re.search(r"\b[0-9a-f]{64}\b", text)
