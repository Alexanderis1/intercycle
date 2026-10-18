"""scripts/sealed_run.py refusals (PLAN_TWIN §5), each on a scratch git repo with a scratch sealed root.

Nothing here touches the real sealed root or the real repo state: every repo, clip, label and seed file is made up
in tmp_path."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sealed_run_script", REPO / "scripts" / "sealed_run.py")
assert _spec and _spec.loader
SR = importlib.util.module_from_spec(_spec)
sys.modules["sealed_run_script"] = SR
_spec.loader.exec_module(SR)

from twin.rng import check_world_seed  # noqa: E402
from twin.schemas.labels import SampleItem, sample_items_sha256  # noqa: E402
from twin.schemas.prereg import make_sealed, render_sealed  # noqa: E402

CLIPS = {"0_te1": b"clip zero", "3_te2": b"clip three"}
FRAMING = {"0_te1": "c9_f1", "3_te2": "c14_f0"}
INFO = {"genesis_time": 1000, "period": 30, "hash": "testchain"}


def g(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()


def commit(repo: Path, msg: str, *paths: str) -> str:
    g(repo, "add", "--", *paths)
    g(repo, "commit", "-q", "-m", msg)
    return g(repo, "rev-parse", "HEAD")


def write(p: Path, data: str | bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data, encoding="utf-8", newline="\n")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_beacon(rnd: int) -> dict:
    sig = hashlib.sha512(f"sig{rnd}".encode()).digest() + bytes(32)
    return {"round": rnd, "randomness": sha(sig), "signature": sig.hex()}


def fake_fetch(path: str) -> dict:
    if path == "/info":
        return dict(INFO)
    return fake_beacon(int(path.rsplit("/", 1)[1]))


def build(tmp: Path, source: str = "fallback") -> dict:
    repo, root = tmp / "repo", tmp / "sealed"
    repo.mkdir()
    g(repo, "init", "-q", "-b", "main")
    g(repo, "config", "user.email", "t@example.invalid")
    g(repo, "config", "user.name", "t")
    g(repo, "config", "core.autocrlf", "false")
    write(repo / "README.md", "scratch\n")
    c0 = commit(repo, "init", "README.md")
    # sealed root: clips, labels and the sample list
    for cid, data in CLIPS.items():
        write(root / "clips" / f"{cid}.mp4", data)
    items = [SampleItem(sample_id=f"f{k}", kind="frame", clip_id=cid, group_id="g", camera=9 if cid[0] == "0" else 14,
                        split="test", t_pts=1.0) for k, cid in enumerate(CLIPS)]
    samples = {"contract": "twin.label_samples/1", "split": "test", "seed": 7, "rule": "1 frame / 2 s",
               "items": [i.model_dump(mode="json") for i in items], "items_sha256": sample_items_sha256(items)}
    write(root / "labels" / "samples.json", json.dumps(samples))
    write(root / "labels" / "frames.json", json.dumps({"labels": []}))
    lab_files = [(sha((root / p).read_bytes()), p) for p in ("labels/frames.json", "labels/samples.json")]
    # repo inputs
    write(repo / "twin" / "daygen.py", "GEN = 1\n")
    write(repo / "twin" / "dynamics.json", "{}\n")
    rows = ["clip_id,group_id,split,sha256,framing_id"]
    rows += [f"{cid},g,test,{sha(d)},{FRAMING[cid]}" for cid, d in CLIPS.items()]
    rows.append(f"1_tr1,g,train,{'0' * 64},c9_f1")
    write(repo / "prereg" / "split_a1.csv", "\n".join(rows) + "\n")
    write(repo / "prereg" / "ACTIVE_SPLIT", "split_a1.csv\n")
    write(repo / "sealed" / "MANIFEST.sha256", render_sealed(make_sealed(
        "MANIFEST", [(sha(d), f"clips/{cid}.mp4") for cid, d in CLIPS.items()], list(CLIPS))))
    write(repo / "sealed" / "LABELS.sha256", render_sealed(make_sealed(
        "LABELS", lab_files, list(CLIPS), sample_list_sha256=samples["items_sha256"])))
    code = commit(repo, "code", "twin", "prereg", "sealed")
    write(repo / "prereg" / "FREEZE", code + "\n")
    commit(repo, "FREEZE", "prereg/FREEZE")
    if source == "fallback":
        line = SR.derive(repo, root, fallback=True)["seed_commit"]
    else:
        line = SR.derive(repo, root, fallback=False, fetch=fake_fetch, now=lambda: 4e9)["seed_commit"]
    write(repo / "prereg" / "TEST_SEED_COMMIT", line + "\n")
    commit(repo, "TEST_SEED_COMMIT", "prereg/TEST_SEED_COMMIT")
    return {"repo": repo, "root": root, "code": code, "c0": c0}


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    return build(tmp_path)


def refused(s: dict, match: str, fetch=fake_fetch):
    with pytest.raises(SR.Refusal, match=match):
        SR.preflight(s["repo"], s["root"], fetch)


# ------------------------------------------------------------------------------------------------ passes

def test_preflight_passes_and_records_the_inputs(s):
    rec = SR.preflight(s["repo"], s["root"], fake_fetch)
    assert rec["freeze_sha"] == s["code"] and rec["n_clips"] == 2
    assert set(rec["inputs"]) == {"manifest", "labels", "generator", "dynamics", "test_seed_commit"}
    assert rec["changed_since_freeze"] == sorted(SR.ALLOWED_AFTER_FREEZE)
    sd = rec["seeds"]
    assert sd["source"] == "fallback" and len(sd["seeds"]) == 16 and all(v > 999 for v in sd["seed_ints"])
    assert {j["clip_id"]: (j["camera"], j["framing"]) for j in rec["jobs"]} == {"0_te1": (9, "c9_f1"),
                                                                              "3_te2": (14, "c14_f0")}
    assert os.environ.get(SR.SEALED_ENV) is None            # preflight alone never sets the sealed flag


def test_drand_reveal_passes_and_records_the_round(tmp_path, monkeypatch):
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    s = build(tmp_path, source="drand")
    rec = SR.preflight(s["repo"], s["root"], fake_fetch)
    t = SR.freeze_commit_time(s["repo"])
    assert rec["seeds"]["source"] == "drand" and rec["seeds"]["round"] == SR.first_round_after(INFO, t)
    assert SR.round_time(INFO, rec["seeds"]["round"] - 1) <= t < SR.round_time(INFO, rec["seeds"]["round"])


# ------------------------------------------------------------------------------------------------ tree and FREEZE

def test_refuses_a_dirty_tree(s):
    write(s["repo"] / "twin" / "daygen.py", "GEN = 2\n")
    refused(s, "not clean")


def test_refuses_an_untracked_file(s):
    write(s["repo"] / "scratch.txt", "x\n")
    refused(s, "not clean")


def test_refuses_without_freeze(s):
    g(s["repo"], "rm", "-q", "prereg/FREEZE")
    g(s["repo"], "commit", "-q", "-m", "drop FREEZE")
    refused(s, "does not exist")


def test_refuses_a_short_freeze_sha(s):
    write(s["repo"] / "prereg" / "FREEZE", s["code"][:12] + "\n")
    commit(s["repo"], "short", "prereg/FREEZE")
    refused(s, "40-hex")


def test_refuses_a_freeze_sha_that_is_no_commit(s):
    write(s["repo"] / "prereg" / "FREEZE", "ab" * 20 + "\n")
    commit(s["repo"], "bogus", "prereg/FREEZE")
    refused(s, "not a commit")


def test_refuses_a_freeze_that_is_not_an_ancestor(s):
    repo = s["repo"]
    g(repo, "checkout", "-q", "-b", "side", s["c0"])
    write(repo / "side.txt", "side\n")
    side = commit(repo, "side", "side.txt")
    g(repo, "checkout", "-q", "main")
    write(repo / "prereg" / "FREEZE", side + "\n")
    commit(repo, "FREEZE on a side branch", "prereg/FREEZE")
    refused(s, "not an ancestor")


def test_refuses_any_other_change_since_freeze(s):
    write(s["repo"] / "twin" / "daygen.py", "GEN = 2\n")
    commit(s["repo"], "late change", "twin/daygen.py")
    refused(s, r"changed since FREEZE.*twin/daygen.py")


def test_refuses_a_rename_into_an_allowed_name(s):
    repo = s["repo"]
    g(repo, "rm", "-q", "prereg/TEST_SEED_COMMIT")
    g(repo, "mv", "twin/dynamics.json", "prereg/TEST_SEED_COMMIT")
    g(repo, "commit", "-q", "-m", "rename")
    refused(s, r"changed since FREEZE.*twin/dynamics.json")


# ------------------------------------------------------------------------------------------------ input hashes

@pytest.mark.parametrize("name", ["manifest", "labels", "generator", "dynamics"])
def test_input_hash_mismatch_is_refused(s, name):
    p = s["repo"] / SR.INPUTS[name]
    p.write_bytes(p.read_bytes() + b"# tampered\n")
    with pytest.raises(SR.Refusal, match=f"input {name} .*hash"):
        SR.check_inputs(s["repo"], s["code"])


def test_input_missing_from_the_freeze_tree_is_refused(s):
    with pytest.raises(SR.Refusal, match="not in the FREEZE tree"):
        SR.check_inputs(s["repo"], s["c0"])


def test_crlf_checkout_hashes_like_the_frozen_blob(s):
    p = s["repo"] / SR.INPUTS["dynamics"]
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert SR.check_inputs(s["repo"], s["code"])["dynamics"]["sha256"] == sha(b"{}\n")


def test_custodian_fetch_manifest_format_is_refused(s):
    """scripts/custodian_fetch.py writes '# ids <n>: ...' lines, not the twin.sealed/1 contract: refused, fail closed."""
    legacy = "".join(f"{sha(d)}  clips/{c}.mp4\n" for c, d in CLIPS.items()) + "# ids 2: 0_te1 3_te2\n"
    write(s["repo"] / SR.INPUTS["manifest"], legacy)
    with pytest.raises(SR.Refusal, match="not a twin.sealed/1 file"):
        SR.load_sealed(s["repo"], "manifest", "MANIFEST")


@pytest.mark.parametrize("change,match", [
    ("tamper", "hashes to"), ("missing", "missing"), ("extra", "extra")])
def test_clips_must_match_the_manifest(s, change, match):
    clips = s["root"] / "clips"
    if change == "tamper":
        (clips / "0_te1.mp4").write_bytes(b"other bytes")
    elif change == "missing":
        (clips / "3_te2.mp4").unlink()
    else:
        (clips / "9_x.mp4").write_bytes(b"x")
    refused(s, match)


def test_manifest_ids_must_be_test_rows_of_the_active_split(s):
    p = s["repo"] / "prereg" / "split_a1.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("3_te2,g,test", "3_te2,g,dev"), encoding="utf-8")
    man = SR.load_sealed(s["repo"], "manifest", "MANIFEST")
    with pytest.raises(SR.Refusal, match="not a test row"):
        SR.check_manifest_files(s["repo"], s["root"], man)


def test_label_files_must_match_labels(s):
    (s["root"] / "labels" / "frames.json").write_text('{"labels": [1]}', encoding="utf-8")
    refused(s, "label file labels/frames.json")


def test_labels_need_the_hashed_sample_list(s):
    lab = SR.load_sealed(s["repo"], "labels", "LABELS")
    lab = lab.model_copy(update={"sample_list_sha256": "0" * 64})
    with pytest.raises(SR.Refusal, match="sample list"):
        SR.check_label_files(s["root"], lab)


# ------------------------------------------------------------------------------------------------ seeds

def _edit_reveal(s: dict, **kw) -> None:
    p = s["root"] / SR.SEEDS_FILE
    d = json.loads(p.read_text(encoding="utf-8"))
    d.update(kw)
    p.write_text(json.dumps(d), encoding="utf-8")


def test_seed_commit_must_match_the_revealed_salt(s):
    _edit_reveal(s, salt="00" * 16)
    refused(s, r"TEST_SEED_COMMIT != sha256\(salt")


def test_seeds_must_derive_from_r(s):
    d = json.loads((s["root"] / SR.SEEDS_FILE).read_text(encoding="utf-8"))
    _edit_reveal(s, seeds=d["seeds"][::-1])
    refused(s, r"not sha256\(R")


def test_seed_count_and_missing_reveal(s):
    d = json.loads((s["root"] / SR.SEEDS_FILE).read_text(encoding="utf-8"))
    _edit_reveal(s, seeds=d["seeds"][:15])
    refused(s, "15 seeds")
    (s["root"] / SR.SEEDS_FILE).unlink()
    refused(s, "does not exist")


def test_seed_commit_file_must_exist_after_freeze(s):
    g(s["repo"], "rm", "-q", "prereg/TEST_SEED_COMMIT")
    g(s["repo"], "commit", "-q", "-m", "drop")
    refused(s, "TEST_SEED_COMMIT does not exist")


@pytest.mark.parametrize("bad,match", [("round", "not the first round"), ("randomness", "differs from drand"),
                                       ("signature", r"sha256\(signature\)")])
def test_drand_checks_fail_closed(tmp_path, monkeypatch, bad, match):
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    s = build(tmp_path, source="drand")
    fetch = fake_fetch
    if bad == "round":
        rnd = json.loads((s["root"] / SR.SEEDS_FILE).read_text(encoding="utf-8"))["round"]
        _edit_reveal(s, round=rnd + 1)
    elif bad == "randomness":
        def fetch(path):  # noqa: F811 - the beacon disagrees with the reveal
            b = fake_fetch(path)
            return dict(b, randomness="1" * 64) if path != "/info" else b
    else:
        def fetch(path):  # noqa: F811 - randomness is not sha256(signature)
            b = fake_fetch(path)
            return dict(b, signature="00" * 96) if path != "/info" else b
    refused(s, match, fetch)


def test_derive_runs_once_and_waits_for_the_round(s, tmp_path):
    with pytest.raises(SR.Refusal, match="derived once"):
        SR.derive(s["repo"], s["root"], fallback=True)
    with pytest.raises(SR.Refusal, match="wait and re-run"):
        SR.derive(s["repo"], tmp_path / "fresh_root", fallback=False, fetch=fake_fetch, now=lambda: 0.0)


def test_test_seeds_need_the_sealed_flag(monkeypatch):
    """The generator's guard (twin.rng.check_world_seed, called by twin/daygen.py) refuses test seeds without the flag."""
    seeds = SR.derive_seeds("ab" * 16)
    ints = [SR.seed_int(x) for x in seeds]
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    for v in ints:
        with pytest.raises(PermissionError, match="not a dev seed"):
            check_world_seed(v)
    monkeypatch.setenv(SR.SEALED_ENV, "1")
    for v in ints:
        check_world_seed(v)
    with pytest.raises(SR.Refusal, match="dev range"):
        SR.seed_int("0" * 13 + "3e7" + "0" * 48)


# ------------------------------------------------------------------------------------------------ run

def fake_runner(batches):
    out = []
    for b in batches:
        for j in b:
            SR.verify_clip_bytes(j["clip_id"], j["data"], j["sha256"])
            out.append({"clip_id": j["clip_id"], "stats": {"clip_id": j["clip_id"]},
                        "files": {f"tracks/{j['clip_id']}.parquet": b"t" + j["data"]}})
    return out


def test_run_sets_the_flag_only_after_the_refusals_and_writes_runlog(s):
    try:
        write(s["repo"] / "scratch.txt", "x\n")
        with pytest.raises(SR.Refusal):
            SR.run(s["repo"], s["root"], ["clips"], fake_fetch, fake_runner)
        assert os.environ.get(SR.SEALED_ENV) is None and not (s["repo"] / SR.RUNLOG).exists()
        (s["repo"] / "scratch.txt").unlink()
        rec = SR.run(s["repo"], s["root"], ["clips"], fake_fetch, fake_runner)
        assert os.environ.get(SR.SEALED_ENV) == "1"
    finally:
        os.environ.pop(SR.SEALED_ENV, None)
    log = json.loads((s["repo"] / SR.RUNLOG).read_text(encoding="utf-8"))
    for k in ("timestamp", "head", "freeze_sha", "inputs", "seeds", "stages", "finished"):
        assert k in log
    assert log["seeds"]["salt"] and len(log["seeds"]["seeds"]) == 16
    st = log["stages"]["clips"]
    assert st["status"] == "done" and st["n_clips"] == 2 and st["ingest"] == "intercycle.perception.esk_ingest.ingest_clip"
    assert (s["root"] / "perc" / "tracks" / "0_te1.parquet").read_bytes() == b"tclip zero"
    assert rec["stages"]["clips"]["outputs_sha256"]["tracks/0_te1.parquet"] == sha(b"tclip zero")


def test_upload_check_rejects_changed_bytes():
    with pytest.raises(PermissionError, match="MANIFEST"):
        SR.verify_clip_bytes("0_te1", b"changed", sha(b"clip zero"))


def test_unknown_stage_is_refused(s):
    with pytest.raises(SR.Refusal, match="not wired"):
        SR.run(s["repo"], s["root"], ["isaac"], fake_fetch, fake_runner)
    assert os.environ.get(SR.SEALED_ENV) is None


def test_runner_returning_fewer_clips_is_an_error(s):
    try:
        with pytest.raises(RuntimeError, match="returned 1 clips"):
            SR.run(s["repo"], s["root"], ["clips"], fake_fetch, lambda b: fake_runner(b)[:1])
    finally:
        os.environ.pop(SR.SEALED_ENV, None)
    assert json.loads((s["repo"] / SR.RUNLOG).read_text(encoding="utf-8"))["stages"]["clips"]["status"] == "error"


def test_sealed_run_names_no_truth_path():
    from _truth_scan import parse, string_literals
    from twin.truth_guard import TRUTH_TOKENS

    lits = string_literals(parse(REPO / "scripts" / "sealed_run.py"))
    assert not [x for x in lits if any(t in x for t in TRUTH_TOKENS)]
