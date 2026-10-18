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
_REAL_CLIPS_ROOTS = SR.CLIPS_STAGE_ROOTS      # before any fixture swaps in the stand-in roots

from twin.rng import check_world_seed  # noqa: E402
from twin.schemas.labels import SampleItem, sample_items_sha256  # noqa: E402
from twin.schemas.prereg import make_sealed, render_sealed  # noqa: E402

CLIPS = {"0_te1": b"clip zero", "3_te2": b"clip three"}
FRAMING = {"0_te1": "c9_f1", "3_te2": "c14_f0"}
DEV_CLIPS = {"4_dv1": b"dev clip four"}
INFO = {"genesis_time": 1000, "period": 30, "hash": "testchain"}
G2_FILE = "prereg/G2_HASHES.json"
G2_OTHER = {"twin/engine.py": "ENG = 1\n", "twin/camera.py": "CAM = 1\n", "twin/site.yaml": "site: 1\n"}


@pytest.fixture(autouse=True)
def _g2_file(monkeypatch):
    """The scratch repos commit a G2 hash record at this path (the real file is the lead's call), and their clips-stage
    code is pkg/clipcode.py (which imports pkg.helper), standing in for the real CLIPS_STAGE_ROOTS."""
    monkeypatch.setattr(SR, "G2_HASHES_FILE", G2_FILE)
    monkeypatch.setattr(SR, "CLIPS_STAGE_ROOTS", ("pkg/clipcode.py",))


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


PKG = {"pkg/__init__.py": "", "pkg/clipcode.py": "from pkg import helper\n\n\ndef f():\n    return helper.X\n",
       "pkg/helper.py": "X = 1\n", "pkg/robot.py": "R = 1\n"}


def build(tmp: Path, source: str = "fallback", stage: str = "robot") -> dict:
    """A scratch repo. stage "clips": code, then the commit writing prereg/FREEZE_REALCLIP = code (HEAD; the detached
    09:15 worktree). stage "robot": that, then prereg/FREEZE = the FREEZE_REALCLIP commit, the seeds and
    TEST_SEED_COMMIT; s["code"] is the robot FREEZE sha there, s["rc"] the FREEZE_REALCLIP sha."""
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
    for rel, text in G2_OTHER.items():
        write(repo / rel, text)
    rows = ["clip_id,group_id,split,sha256,framing_id"]
    rows += [f"{cid},g,test,{sha(d)},{FRAMING[cid]}" for cid, d in CLIPS.items()]
    rows.append(f"1_tr1,g,train,{'0' * 64},c9_f1")
    rows += [f"{cid},g,dev,{sha(d)},c9_f1" for cid, d in DEV_CLIPS.items()]
    write(repo / "prereg" / "split_a1.csv", "\n".join(rows) + "\n")
    write(repo / "prereg" / "ACTIVE_SPLIT", "split_a1.csv\n")
    write(repo / "sealed" / "MANIFEST.sha256", render_sealed(make_sealed(
        "MANIFEST", [(sha(d), f"clips/{cid}.mp4") for cid, d in CLIPS.items()], list(CLIPS))))
    write(repo / "sealed" / "LABELS.sha256", render_sealed(make_sealed(
        "LABELS", lab_files, list(CLIPS), sample_list_sha256=samples["items_sha256"])))
    write(repo / G2_FILE, json.dumps(SR.g2_record(repo)) + "\n")
    for rel, text in PKG.items():
        write(repo / rel, text)
    code = commit(repo, "code", "twin", "prereg", "sealed", "pkg")
    write(repo / "prereg" / "FREEZE_REALCLIP", code + "\n")
    rc_commit = commit(repo, "FREEZE_REALCLIP", "prereg/FREEZE_REALCLIP")
    if stage == "clips":
        return {"repo": repo, "root": root, "code": code, "rc": code, "c0": c0, "stage": "clips"}
    write(repo / "prereg" / "FREEZE", rc_commit + "\n")
    commit(repo, "FREEZE", "prereg/FREEZE")
    if source == "fallback":
        line = SR.derive(repo, root, fallback=True)["seed_commit"]
    else:
        t_freeze = SR.freeze_commit_time(repo)
        line = SR.derive(repo, root, fallback=False, fetch=fake_fetch, now=lambda: t_freeze + 60.0)["seed_commit"]
    write(repo / "prereg" / "TEST_SEED_COMMIT", line + "\n")
    commit(repo, "TEST_SEED_COMMIT", "prereg/TEST_SEED_COMMIT")
    return {"repo": repo, "root": root, "code": rc_commit, "rc": code, "c0": c0, "stage": "robot"}


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    return build(tmp_path)


@pytest.fixture
def c(tmp_path, monkeypatch):
    """The clips-stage repo: HEAD is the commit that wrote prereg/FREEZE_REALCLIP (the detached 09:15 worktree)."""
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    return build(tmp_path, stage="clips")


def refused(s: dict, match: str, fetch=fake_fetch):
    with pytest.raises(SR.Refusal, match=match):
        if s.get("stage") == "clips":
            SR.preflight_clips(s["repo"], s["root"])
        else:
            SR.preflight(s["repo"], s["root"], fetch)


# ------------------------------------------------------------------------------------------------ passes

def test_preflight_passes_and_records_the_inputs(s):
    rec = SR.preflight(s["repo"], s["root"], fake_fetch)
    assert rec["freeze_sha"] == s["code"] and rec["n_clips"] == 2
    assert set(rec["inputs"]) == {"manifest", "labels", "generator", "dynamics", "engine", "camera", "site",
                                 "test_seed_commit"}
    assert rec["changed_since_freeze"] == sorted([SR.FREEZE, SR.SEED_COMMIT]) and rec["runlog"] == []
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

@pytest.mark.parametrize("name", ["manifest", "labels", "generator", "dynamics", "engine", "camera", "site"])
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


def test_run_sets_the_flag_only_after_the_refusals_and_writes_runlog(c):
    s = c
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
    assert log["contract"] == SR.RUNLOG_CONTRACT and [r["status"] for r in log["records"]] == ["started", "done"]
    start, done = log["records"]
    for k in ("attempt", "timestamp", "head", "freeze_sha", "inputs", "stages", "status", "perc_dir"):
        assert k in start
    assert start["attempt"] == done["attempt"] == 1 and start["stages"] == ["clips"]
    assert start["freeze"] == SR.FREEZE_REALCLIP and start["freeze_sha"] == s["rc"] and "seeds" not in start
    assert set(start["inputs"]) == {"manifest", "labels"}                     # no G2 files, no seeds for real clips
    st = done["stages"]["clips"]
    assert st["status"] == "done" and st["n_clips"] == 2 and st["ingest"] == "intercycle.perception.esk_ingest.ingest_clip"
    assert (s["root"] / "perc" / "attempt_001" / "tracks" / "0_te1.parquet").read_bytes() == b"tclip zero"
    assert rec["stages"]["clips"]["outputs_sha256"]["tracks/0_te1.parquet"] == sha(b"tclip zero")


def test_double_stage_runs_after_the_refusals_with_the_test_seeds(s):
    """The CPU-double stage (cut design) gets the 16 revealed test seeds, a fresh runs root, and the sealed flag."""
    seen = {}

    def fake_double(out, seeds):
        seen.update(out=out, seeds=seeds, flag=os.environ.get(SR.SEALED_ENV))
        return {"runs_root": str(out / "runs"), "invalid": str(out / "runs" / "invalid_all.json"), "n_missions": 192,
                "usd": 0.0, "wall_s": 1.0}

    try:
        rec = SR.run(s["repo"], s["root"], ["double"], fake_fetch, fake_runner, double_runner=fake_double)
    finally:
        os.environ.pop(SR.SEALED_ENV, None)
    assert seen["flag"] == "1" and len(seen["seeds"]) == 16 and all(v > 999 for v in seen["seeds"])
    RB = SR._robot_module()                   # scripts/sealed_robot.py: the robot stage lives outside sealed_run.py
    assert seen["out"] == s["repo"] / RB.DOUBLE_OUT / "attempt_001"
    assert rec["stages"]["double"]["status"] == "done" and rec["stages"]["double"]["n_missions"] == 192
    with pytest.raises(RB.StageRefusal, match="sealed flag"):
        RB.run_double_stage(s["repo"], 9, [1000] * 16, fake_double)   # never outside an attempt
    with pytest.raises(RB.StageRefusal, match="not wired"):
        RB.run_stage("isaac", {"attempt": 1, "seeds": {"seed_ints": [1000] * 16}})


def test_robot_stage_module_is_outside_the_realclip_closure():
    """Wiring / fixing the double stage after FREEZE_REALCLIP must not touch a real-clip input (robot check 5)."""
    closure = SR.import_closure(REPO, "HEAD", _REAL_CLIPS_ROOTS)
    assert "scripts/sealed_run.py" in closure
    assert not {"scripts/sealed_robot.py", "scripts/run_double.py", "intercycle/eval/twin_batch.py",
                "twin/plant.py"} & closure


def test_double_stage_refuses_before_any_runlog_record_while_the_backend_is_not_wired(s):
    """No fake runner: run_double.SEALED_BACKEND_READY is False in this build, so sealed_run refuses before the drand
    fetch and before the RUNLOG `started` record (no attempt number used up)."""
    called = []

    def fetch(url):
        called.append(url)
        return fake_fetch(url)

    log = s["repo"] / SR.RUNLOG
    before = log.read_bytes() if log.exists() else None
    with pytest.raises(SR.Refusal, match="not wired"):
        SR.run(s["repo"], s["root"], ["double"], fetch, fake_runner)
    assert called == [] and (log.read_bytes() if log.exists() else None) == before
    assert SR.SEALED_ENV not in os.environ


def test_upload_check_rejects_changed_bytes():
    with pytest.raises(PermissionError, match="MANIFEST"):
        SR.verify_clip_bytes("0_te1", b"changed", sha(b"clip zero"))


def test_unknown_stage_is_refused(s):
    with pytest.raises(SR.Refusal, match="not wired"):
        SR.run(s["repo"], s["root"], ["isaac"], fake_fetch, fake_runner)
    assert os.environ.get(SR.SEALED_ENV) is None


def test_runner_returning_fewer_clips_is_an_error(c):
    s = c
    try:
        with pytest.raises(RuntimeError, match="returned 1 clips"):
            SR.run(s["repo"], s["root"], ["clips"], fake_fetch, lambda b: fake_runner(b)[:1])
    finally:
        os.environ.pop(SR.SEALED_ENV, None)
    recs = json.loads((s["repo"] / SR.RUNLOG).read_text(encoding="utf-8"))["records"]
    assert [r["status"] for r in recs] == ["started", "error"]
    assert recs[-1]["stages"]["clips"]["status"] == "error"


def test_sealed_run_names_no_truth_path():
    from _truth_scan import parse, string_literals
    from twin.truth_guard import TRUTH_TOKENS

    lits = string_literals(parse(REPO / "scripts" / "sealed_run.py"))
    assert not [x for x in lits if any(t in x for t in TRUTH_TOKENS)]


# ------------------------------------------------------------------------------------------------ derive / clocks

def test_derive_runs_the_tree_and_freeze_refusals_first(s, tmp_path):
    write(s["repo"] / "scratch.txt", "x\n")
    with pytest.raises(SR.Refusal, match="not clean"):
        SR.derive(s["repo"], tmp_path / "r1", fallback=True)
    (s["repo"] / "scratch.txt").unlink()
    write(s["repo"] / "prereg" / "FREEZE", "ab" * 20 + "\n")
    commit(s["repo"], "bogus", "prereg/FREEZE")
    with pytest.raises(SR.Refusal, match="not a commit"):
        SR.derive(s["repo"], tmp_path / "r2", fallback=True)
    assert not (tmp_path / "r1").exists() and not (tmp_path / "r2").exists()


def test_derive_refuses_long_after_the_freeze_commit(s, tmp_path):
    t = SR.freeze_commit_time(s["repo"])
    with pytest.raises(SR.Refusal, match="after the FREEZE commit"):
        SR.derive(s["repo"], tmp_path / "late", fallback=False, fetch=fake_fetch,
                  now=lambda: t + SR.MAX_DERIVE_LAG_S + 120.0)
    rec = SR.derive(s["repo"], tmp_path / "ok", fallback=False, fetch=fake_fetch, now=lambda: t + 60.0)
    rv = json.loads((tmp_path / "ok" / SR.SEEDS_FILE).read_text(encoding="utf-8"))
    assert rec["source"] == "drand" and rv["derived_at"] == int(t + 60.0) and rv["freeze_commit_time"] == t


def _refreeze_with_dates(s: dict, author: int, committer: int) -> None:
    repo = s["repo"]
    p = repo / "prereg" / "FREEZE"
    write(p, p.read_text(encoding="utf-8") + "\n")
    g(repo, "add", "--", "prereg/FREEZE")
    env = dict(os.environ, GIT_AUTHOR_DATE=f"{author} +0000", GIT_COMMITTER_DATE=f"{committer} +0000")
    subprocess.run(["git", "commit", "-q", "-m", "re-dated FREEZE"], cwd=repo, check=True, env=env)


def test_freeze_commit_dates_must_agree(s):
    t = SR.freeze_commit_time(s["repo"])
    _refreeze_with_dates(s, author=t - 2 * SR.DATE_SKEW_S, committer=t)
    with pytest.raises(SR.Refusal, match="committer date .* author date"):
        SR.check_commit_dates(s["repo"])
    _refreeze_with_dates(s, author=t + 5, committer=t + 10)
    assert SR.check_commit_dates(s["repo"]) == t + 10


def test_drand_reveal_needs_its_own_clock_in_the_window(tmp_path, monkeypatch):
    monkeypatch.delenv(SR.SEALED_ENV, raising=False)
    s = build(tmp_path, source="drand")
    d = json.loads((s["root"] / SR.SEEDS_FILE).read_text(encoding="utf-8"))
    _edit_reveal(s, derived_at=None)
    refused(s, "needs derived_at")
    _edit_reveal(s, derived_at=d["freeze_commit_time"] + SR.MAX_DERIVE_LAG_S + 1)
    refused(s, "after the FREEZE commit")
    _edit_reveal(s, derived_at=d["freeze_commit_time"] - 100)
    refused(s, "before drand round")
    _edit_reveal(s, derived_at=d["derived_at"])
    assert SR.preflight(s["repo"], s["root"], fake_fetch)["seeds"]["derived_at"] == d["derived_at"]


# ------------------------------------------------------------------------------------------------ RUNLOG append-only

def _run(s: dict, runner=fake_runner) -> dict:
    try:
        return SR.run(s["repo"], s["root"], ["clips"], fake_fetch, runner)
    finally:
        os.environ.pop(SR.SEALED_ENV, None)


def _records(s: dict) -> list[dict]:
    return json.loads((s["repo"] / SR.RUNLOG).read_text(encoding="utf-8"))["records"]


def test_a_second_run_appends_a_new_attempt_to_a_dirty_runlog(c):
    s = c
    r1 = _run(s)
    first = _records(s)
    assert g(s["repo"], "status", "--porcelain", "--untracked-files=all") == f"?? {SR.RUNLOG}"   # dirty, allowed
    r2 = _run(s)
    recs = _records(s)
    assert recs[:2] == first and [(r["attempt"], r["status"]) for r in recs] == [(1, "started"), (1, "done"),
                                                                               (2, "started"), (2, "done")]
    assert (r1["perc_dir"], r2["perc_dir"]) == ("perc/attempt_001", "perc/attempt_002")
    for d in ("attempt_001", "attempt_002"):
        assert (s["root"] / "perc" / d / "tracks" / "3_te2.parquet").is_file()


def test_a_rerun_after_a_stage_error_is_a_new_attempt(c):
    s = c
    def broken(batches):
        raise RuntimeError("provider failure")

    with pytest.raises(RuntimeError, match="provider failure"):
        _run(s, broken)
    assert not (s["root"] / "perc" / "attempt_001").exists()
    r2 = _run(s)
    assert r2["attempt"] == 2 and [(r["attempt"], r["status"]) for r in _records(s)] == [
        (1, "started"), (1, "error"), (2, "started"), (2, "done")]


def test_a_committed_runlog_is_allowed_and_must_only_grow(c):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    assert SR.check_diff(s["repo"], s["code"], SR.ALLOWED_AFTER_FREEZE_REALCLIP) == sorted([SR.FREEZE_REALCLIP,
                                                                                            SR.RUNLOG])
    _run(s)                                                             # passes check_clean / check_diff, appends
    assert len(_records(s)) == 4
    commit(s["repo"], "RUNLOG attempt 2", SR.RUNLOG)
    p = s["repo"] / SR.RUNLOG
    d = json.loads(p.read_text(encoding="utf-8"))
    d["records"] = d["records"][:2]                                     # drop committed records in the working file
    p.write_text(json.dumps(d), encoding="utf-8")
    refused(s, "not an append-only extension")


def test_a_committed_rewrite_of_runlog_is_refused(c):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    p = s["repo"] / SR.RUNLOG
    d = json.loads(p.read_text(encoding="utf-8"))
    d["records"][0]["timestamp"] = "2000-01-01T00:00:00+00:00"          # history edited in the working file
    p.write_text(json.dumps(d), encoding="utf-8")
    refused(s, "not an append-only extension")
    d["records"][1]["status"] = "started"                               # and an edit that breaks the structure
    p.write_text(json.dumps(d), encoding="utf-8")
    refused(s, "opens out of order")


def _commit_edited_runlog(s: dict, edit, msg: str) -> None:
    p = s["repo"] / SR.RUNLOG
    d = json.loads(p.read_text(encoding="utf-8"))
    edit(d)
    p.write_text(json.dumps(d), encoding="utf-8")
    commit(s["repo"], msg, SR.RUNLOG)


@pytest.mark.parametrize("edit", [
    lambda d: d["records"][1].update(stages={"clips": "edited"}),      # a closed record's stages rewritten
    lambda d: d["records"].__delitem__(slice(1, None)),                 # truncated to the opening record
    lambda d: d.update(records=[]),                                     # emptied
])
def test_a_rewrite_committed_after_freeze_is_refused(c, edit):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    _commit_edited_runlog(s, edit, "RUNLOG history edited")
    assert g(s["repo"], "status", "--porcelain", "--untracked-files=all") == ""   # working file == HEAD
    with pytest.raises(SR.Refusal, match=r"not an append-only extension.*commit [0-9a-f]{12} has 2 records"):
        SR.check_runlog(s["repo"], s["code"])
    refused(s, "not an append-only extension")


def test_a_committed_rewrite_stays_refused_after_a_later_restore(c):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    good = (s["repo"] / SR.RUNLOG).read_text(encoding="utf-8")
    _commit_edited_runlog(s, lambda d: d["records"][0].update(timestamp="2000-01-01T00:00:00+00:00"), "rewrite")
    (s["repo"] / SR.RUNLOG).write_text(good, encoding="utf-8")         # the HEAD version looks untouched again
    commit(s["repo"], "restore", SR.RUNLOG)
    refused(s, "not an append-only extension")


def test_a_committed_deletion_of_runlog_is_refused(c):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    g(s["repo"], "rm", "-q", "--", SR.RUNLOG)
    g(s["repo"], "commit", "-q", "-m", "drop RUNLOG")
    refused(s, "not an append-only extension")


def test_runlog_check_starts_from_the_freeze_tree(c):
    s = c
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    head = g(s["repo"], "rev-parse", "HEAD")
    assert len(SR.check_runlog(s["repo"], head)) == 2                  # FREEZE tree holds the 2 records: a prefix
    with pytest.raises(SR.Refusal, match="the FREEZE tree"):
        _commit_edited_runlog(s, lambda d: d["records"].pop(), "truncate")
        SR.check_runlog(s["repo"], head)


def test_only_runlog_may_be_dirty(c):
    s = c
    _run(s)
    write(s["repo"] / "results" / "sealed" / "other.json", "{}\n")
    refused(s, r"not clean:\n\?\? results/sealed/other.json")


def test_append_refuses_when_the_file_is_not_the_known_prefix(c):
    s = c
    _run(s)
    known = _records(s)
    with pytest.raises(SR.Refusal, match="not a prefix"):
        SR.append_runlog(s["repo"], known[:1], {"attempt": 2, "status": "started", "timestamp": "t", "head": "h",
                                                "stages": ["clips"]})
    assert _records(s) == known


def test_runlog_must_parse_and_attempt_dirs_are_fresh(c):
    s = c
    p = s["repo"] / SR.RUNLOG
    write(p, json.dumps({"timestamp": "x", "stages": {}}))              # the old single-record format
    refused(s, "not a twin.sealed_runlog/2")
    write(p, json.dumps({"contract": SR.RUNLOG_CONTRACT, "records": [{"attempt": 1, "status": "done", "timestamp": "t",
                                                                   "head": "h", "stages": {}}]}))
    refused(s, "before it started")
    p.unlink()
    (s["root"] / "perc" / "attempt_001").mkdir(parents=True)
    with pytest.raises(SR.Refusal, match="refusing to reuse"):
        _run(s)
    assert not p.exists()


# ------------------------------------------------------------------------------------------------ G2 hash record

def test_g2_record_must_be_named(s, monkeypatch):
    monkeypatch.setattr(SR, "G2_HASHES_FILE", None)
    refused(s, "no committed G2")


def test_g2_record_must_match_and_be_frozen(s, monkeypatch):
    rec = SR.check_inputs(s["repo"], s["code"])
    assert SR.G2_HASHES_FILE == G2_FILE                                  # the lead's file name is the default
    assert SR.check_g2_hashes(s["repo"], s["code"], rec) == {
        "generator": sha(b"GEN = 1\n"), "dynamics": sha(b"{}\n"), "engine": sha(b"ENG = 1\n"), "camera": sha(b"CAM = 1\n"),
        "site": sha(b"site: 1\n")}
    bad = dict(rec, dynamics=dict(rec["dynamics"], sha256="1" * 64))
    with pytest.raises(SR.Refusal, match="dynamics .* != G2 record"):
        SR.check_g2_hashes(s["repo"], s["code"], bad)
    monkeypatch.setattr(SR, "G2_HASHES_FILE", "prereg/elsewhere.json")
    with pytest.raises(SR.Refusal, match="not in the FREEZE tree"):
        SR.check_g2_hashes(s["repo"], s["code"], rec)


@pytest.mark.parametrize("edit", ["contract", "missing", "extra"])
def test_g2_record_must_be_the_contract_with_exactly_the_g2_files(tmp_path, edit):
    s = build(tmp_path)
    repo = s["repo"]
    rec = SR.g2_record(repo)
    if edit == "contract":
        rec["contract"] = "twin.g2_hashes/0"
    elif edit == "missing":
        rec["files"].pop("twin/camera.py")
    else:
        rec["files"]["twin/other.py"] = "0" * 64
    write(repo / G2_FILE, json.dumps(rec) + "\n")
    code = commit(repo, "bad G2", G2_FILE)
    with pytest.raises(SR.Refusal, match="is not a twin.g2_hashes/1 record"):
        SR.check_inputs(repo, code)


def test_write_g2_hashes_writes_the_record_sealed_run_checks(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("write_g2_hashes", REPO / "scripts" / "write_g2_hashes.py")
    assert spec and spec.loader
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    s = build(tmp_path)
    repo = s["repo"]
    (repo / G2_FILE).unlink()
    out = W.write(repo)
    assert json.loads(out.read_text(encoding="utf-8")) == SR.g2_record(repo) and W.check(repo)["contract"] == SR.G2_CONTRACT
    write(repo / "twin" / "engine.py", "ENG = 2\n")                      # an uncommitted G2 file: refused
    with pytest.raises(Exception, match="not committed"):
        W.write(repo)
    with pytest.raises(Exception, match="differs from the working files"):
        W.check(repo)
    (repo / "twin" / "camera.py").unlink()
    with pytest.raises(Exception, match="does not exist"):
        W.write(repo)


# ------------------------------------------------------------------------------------------------ re-runs

def test_a_rerun_after_committed_results_sealed_outputs_is_allowed(c):
    """LEAD DECISION: sealed_run's own outputs under results/sealed/** are exempt from the FREEZE..HEAD diff rule; each
    invocation appends a new RUNLOG attempt and keeps the earlier records."""
    s = c
    _run(s)
    write(s["repo"] / "results" / "sealed" / "attempt_001" / "summary.json", "{}\n")
    commit(s["repo"], "sealed attempt 1 outputs", SR.RUNLOG, "results/sealed/attempt_001/summary.json")
    assert SR.check_diff(s["repo"], s["code"], SR.ALLOWED_AFTER_FREEZE_REALCLIP) == sorted([
        SR.FREEZE_REALCLIP, SR.RUNLOG, "results/sealed/attempt_001/summary.json"])
    first = _records(s)
    r2 = _run(s)
    assert r2["attempt"] == 2 and _records(s)[:len(first)] == first and len(_records(s)) == len(first) + 2
    write(s["repo"] / "results" / "other.json", "{}\n")                    # outside results/sealed/: still refused
    commit(s["repo"], "other", "results/other.json")
    refused(s, r"changed since FREEZE.*results/other.json")


# ------------------------------------------------------------------------------------------------ container + dry run

def test_container_function_is_imported_by_name_not_serialized():
    """Audit fix: container_ingest lives in scripts/sealed_ingest.py (a stdlib-only module Modal imports by name in the
    container) and is registered without serialized=True, so the client Python need not match the image."""
    spec = SR.modal_function_spec()
    assert spec == {"module": "sealed_ingest", "qualname": "container_ingest", "serialized": False,
                    "include_source": False, "local_python_source": "sealed_ingest"}
    import ast

    tree = ast.parse((REPO / "scripts" / "sealed_run.py").read_text(encoding="utf-8"))
    kws = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert "serialized" not in kws and "include_source" in kws
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "container_ingest"]
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import sealed_ingest; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('twin', 'intercycle', 'modal')))")
    r = subprocess.run([sys.executable, "-I", "-c", code, str(REPO / "scripts")], capture_output=True, text=True,
                       check=True)
    assert r.stdout.strip() == "[]"                                      # importable without the repo or modal


def test_container_batch_check_fails_closed():
    SR.modal_function_spec()                                             # puts scripts/ on sys.path
    import sealed_ingest as SI  # pyright: ignore[reportMissingImports]

    job = {"clip_id": "0_te1", "data": b"clip zero", "sha256": sha(b"clip zero"), "split": "test"}
    assert SI.check_batch([job]) == "test"
    with pytest.raises(PermissionError, match="differ from MANIFEST"):
        SI.check_batch([dict(job, data=b"other")])
    with pytest.raises(PermissionError, match="one split"):
        SI.check_batch([job, dict(job, split="dev")])
    with pytest.raises(PermissionError, match="one split"):
        SI.check_batch([dict(job, split="train")])


def test_dev_dry_run_runs_the_clips_path_on_dev_clips_only(s, tmp_path):
    data = tmp_path / "data"
    for cid, d in DEV_CLIPS.items():
        write(data / "clips" / f"{cid}.mp4", d)
    write(data / "clips" / "0_te1.mp4", b"clip zero")
    dev = tmp_path / "devroot"
    seen = []

    def runner(batches):
        seen.extend(j["split"] for b in batches for j in b)
        return fake_runner(batches)

    rec = SR.dev_dry_run(s["repo"], dev, list(DEV_CLIPS), data_root=data, runner=runner)
    st = rec["clips_stage"]
    assert rec["dev_dry_run"] and st["n_clips"] == 1 and seen == ["dev"] and os.environ.get(SR.SEALED_ENV) is None
    assert not (s["repo"] / SR.RUNLOG).exists()
    man = SR.parse_sealed((dev / "MANIFEST.dev.sha256").read_text(encoding="utf-8"))
    assert man.kind == "MANIFEST" and [f.path for f in man.files] == ["clips/4_dv1.mp4"]
    assert (Path(st["out"]) / "tracks" / "4_dv1.parquet").read_bytes() == b"t" + DEV_CLIPS["4_dv1"]
    with pytest.raises(SR.Refusal, match="dev clips only"):
        SR.dev_dry_run(s["repo"], tmp_path / "d2", ["0_te1"], data_root=data, runner=fake_runner)
    with pytest.raises(SR.Refusal, match="outside the sealed root"):
        SR.dev_dry_run(s["repo"], SR.DEFAULT_SEALED_ROOT / "dry", list(DEV_CLIPS), data_root=data, runner=fake_runner)
    assert not (tmp_path / "d2").exists()


# ------------------------------------------------------------------------------------------------ split freeze

def test_clips_preflight_checks_only_freeze_realclip(c):
    rec = SR.preflight_clips(c["repo"], c["root"])
    assert rec["freeze"] == SR.FREEZE_REALCLIP and rec["freeze_sha"] == c["rc"] and rec["n_clips"] == 2
    assert set(rec["inputs"]) == {"manifest", "labels"} and "seeds" not in rec
    assert rec["changed_since_freeze"] == [SR.FREEZE_REALCLIP]
    assert not (c["repo"] / SR.FREEZE).exists()                          # no robot FREEZE, no seeds needed


def test_clips_refuses_without_freeze_realclip(c):
    g(c["repo"], "rm", "-q", SR.FREEZE_REALCLIP)
    g(c["repo"], "commit", "-q", "-m", "drop FREEZE_REALCLIP")
    refused(c, "FREEZE_REALCLIP does not exist")


def test_clips_refuses_a_bad_or_foreign_freeze_realclip(c):
    repo = c["repo"]
    write(repo / SR.FREEZE_REALCLIP, c["rc"][:12] + "\n")
    commit(repo, "short", SR.FREEZE_REALCLIP)
    refused(c, "40-hex")
    g(repo, "checkout", "-q", "-b", "side", c["c0"])
    write(repo / "side.txt", "side\n")
    side = commit(repo, "side", "side.txt")
    g(repo, "checkout", "-q", "main")
    write(repo / SR.FREEZE_REALCLIP, side + "\n")
    commit(repo, "FREEZE_REALCLIP on a side branch", SR.FREEZE_REALCLIP)
    refused(c, "FREEZE_REALCLIP commit .* is not an ancestor of HEAD")


def test_clips_refuses_a_dirty_tree(c):
    write(c["repo"] / "pkg" / "helper.py", "X = 2\n")
    refused(c, "not clean")


@pytest.mark.parametrize("rel", ["prereg/FREEZE", "prereg/TEST_SEED_COMMIT", "pkg/robot.py", "twin/engine.py"])
def test_clips_refuses_any_change_after_freeze_realclip(c, rel):
    """Only prereg/FREEZE_REALCLIP and results/sealed/** may differ: the robot FREEZE, the seeds and any later
    commit (robot code included) are refused, so the 09:15 run must use the detached worktree at FREEZE_REALCLIP."""
    write(c["repo"] / rel, "later\n")
    commit(c["repo"], "later", rel)
    refused(c, rf"changed since FREEZE_REALCLIP.*{rel}")


@pytest.mark.parametrize("name", ["manifest", "labels"])
def test_clips_manifest_and_labels_must_match_freeze_realclip(c, name):
    p = c["repo"] / SR.INPUTS[name]
    p.write_bytes(p.read_bytes() + b"# tampered\n")
    with pytest.raises(SR.Refusal, match=f"input {name} .*hash"):
        SR.check_inputs(c["repo"], c["rc"], keys=("manifest", "labels"))
    with pytest.raises(SR.Refusal, match="not in the FREEZE tree"):
        SR.check_inputs(c["repo"], c["c0"], keys=(name,))


def test_clips_still_checks_the_sealed_root(c):
    (c["root"] / "clips" / "0_te1.mp4").write_bytes(b"changed")
    refused(c, "hashes to")


def test_run_refuses_mixing_clips_and_robot_stages(c):
    with pytest.raises(SR.Refusal, match="mix"):
        SR.run(c["repo"], c["root"], ["clips", "double"], fake_fetch, fake_runner)
    assert os.environ.get(SR.SEALED_ENV) is None


def test_robot_preflight_records_the_realclip_freeze(s):
    rec = SR.preflight(s["repo"], s["root"], fake_fetch)
    rf = rec["realclip_freeze"]
    assert rf["freeze_realclip_sha"] == s["rc"] and rf["n_realclip_inputs"] > 0
    assert SR.import_closure(s["repo"], s["rc"]) == {"pkg/clipcode.py", "pkg/__init__.py", "pkg/helper.py"}


def _robot_change(s: dict, rel: str, text: str | None) -> None:
    """A commit between FREEZE_REALCLIP and a re-written robot FREEZE (then fresh seeds), as the 10:15 freeze would."""
    repo = s["repo"]
    if text is None:
        g(repo, "rm", "-q", rel)
        g(repo, "commit", "-q", "-m", f"drop {rel}")
    else:
        write(repo / rel, text)
        commit(repo, f"robot work {rel}", rel)
    head = g(repo, "rev-parse", "HEAD")
    write(repo / "prereg" / "FREEZE", head + "\n")
    commit(repo, "FREEZE again", "prereg/FREEZE")
    (s["root"] / SR.SEEDS_FILE).unlink()
    line = SR.derive(repo, s["root"], fallback=True)["seed_commit"]
    write(repo / "prereg" / "TEST_SEED_COMMIT", line + "\n")
    commit(repo, "TEST_SEED_COMMIT again", "prereg/TEST_SEED_COMMIT")


def test_robot_passes_when_only_robot_files_changed_after_freeze_realclip(s):
    _robot_change(s, "pkg/robot.py", "R = 2\n")
    rec = SR.preflight(s["repo"], s["root"], fake_fetch)
    assert rec["realclip_freeze"]["freeze_realclip_sha"] == s["rc"]


@pytest.mark.parametrize("rel,text", [
    ("pkg/helper.py", "X = 2\n"),                          # imported by the clips stage (transitively)
    ("pkg/__init__.py", "# changed\n"),                     # a parent package of an imported module
    ("pkg/clipcode.py", "from pkg import helper, newmod\n"), # a root
    ("twin/schemas/labels.py", "SCHEMA = 2\n"),            # the labels schema (added here)
    ("sealed/MANIFEST.sha256", None),                       # sealed/* (deleted)
    ("sealed/NEW.txt", "x\n"),                              # sealed/* (added)
    ("prereg/split_a1.csv", "clip_id,group_id,split,sha256,framing_id\n"),   # the active split
])
def test_robot_refuses_a_realclip_input_changed_between_the_freezes(s, rel, text):
    _robot_change(s, rel, text)
    with pytest.raises(SR.Refusal):
        SR.preflight(s["repo"], s["root"], fake_fetch)
    with pytest.raises(SR.Refusal, match=rf"real-clip inputs differ between FREEZE_REALCLIP.*{rel}"):
        SR.check_realclip_unchanged(s["repo"], SR.read_freeze(s["repo"]))


def test_robot_refuses_a_module_newly_imported_after_freeze_realclip(s):
    """A module the clips stage imports only at FREEZE (closure taken at both commits) is guarded too."""
    repo = s["repo"]
    write(repo / "pkg" / "newmod.py", "N = 1\n")
    write(repo / "pkg" / "clipcode.py", "from pkg import helper, newmod\n")
    commit(repo, "new import", "pkg/newmod.py", "pkg/clipcode.py")
    with pytest.raises(SR.Refusal, match=r"real-clip inputs differ.*pkg/clipcode.py.*pkg/newmod.py"):
        SR.check_realclip_unchanged(repo, g(repo, "rev-parse", "HEAD"))


def test_robot_refuses_without_freeze_realclip_or_when_it_is_not_an_ancestor_of_freeze(s):
    repo = s["repo"]
    g(repo, "checkout", "-q", "-b", "side", s["c0"])
    write(repo / "side.txt", "side\n")
    side = commit(repo, "side", "side.txt")
    g(repo, "checkout", "-q", "main")
    write(repo / SR.FREEZE_REALCLIP, side + "\n")
    with pytest.raises(SR.Refusal, match="FREEZE_REALCLIP commit .* is not an ancestor of"):
        SR.check_realclip_unchanged(repo, SR.read_freeze(repo))
    (repo / SR.FREEZE_REALCLIP).unlink()
    with pytest.raises(SR.Refusal, match="FREEZE_REALCLIP does not exist"):
        SR.check_realclip_unchanged(repo, SR.read_freeze(repo))


def test_robot_stages_are_refused_until_wired(s, monkeypatch):
    monkeypatch.setattr(SR, "ROBOT_MODULE", "sealed_robot_missing")     # no robot module -> refused before anything
    with pytest.raises(SR.Refusal, match="not wired"):
        SR.run(s["repo"], s["root"], ["double"], fake_fetch, fake_runner)
    assert os.environ.get(SR.SEALED_ENV) is None and not (s["repo"] / SR.RUNLOG).exists()


# ------------------------------------------------------------------------------------------------ score (fix1)

def _scored(monkeypatch):
    import types

    import twin.dev_labels as DL

    seen = {}

    def fake_load(blobs, samples, sha_of, split="test"):
        seen["load"] = (sorted(blobs), samples, sha_of)
        seen["split"] = split
        return types.SimpleNamespace(doc={"counts": {"frames_scored": 0}})

    def fake_scorer(labels_dir, data, site, splits, watchdog, scored, perc, clips_dir):
        seen["score"] = dict(labels_dir=labels_dir, splits=splits, perc=perc, clips_dir=clips_dir, site=site)
        return {"person": {"n": 0}, "frames": [{"label": "secret"}], "windows": [], "clips": ["0_te1"]}

    monkeypatch.setattr(DL, "load_scored_test", fake_load)
    return seen, fake_scorer


def _ran(c):
    try:
        SR.run(c["repo"], c["root"], ["clips"], fake_fetch, fake_runner)
    finally:
        os.environ.pop(SR.SEALED_ENV, None)


def test_score_runs_the_same_scorer_on_the_attempt_and_keeps_label_content_sealed(c, monkeypatch, tmp_path):
    seen, scorer = _scored(monkeypatch)
    with pytest.raises(SR.Refusal, match="no done clips attempt"):              # nothing ran yet
        SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)
    _ran(c)
    d = SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)
    assert d["attempt"] == 1
    assert seen["load"][1] == "labels/samples.json" and seen["load"][2] == {k: sha(v) for k, v in CLIPS.items()}
    assert seen["score"]["splits"] == ("test",) and seen["score"]["labels_dir"] is None and seen["split"] == "test"
    assert seen["load"][0] == ["labels/samples.json"]     # from <sealed-root>/labels only (frames.json: no contract)
    assert seen["score"]["perc"] == c["root"] / "perc" / "attempt_001"
    assert seen["score"]["clips_dir"] == c["root"] / "clips" and seen["score"]["site"] == c["repo"] / SR.SITE
    full = json.loads((c["root"] / SR.SCORES_DIR / "attempt_001_realclip_test.json").read_text(encoding="utf-8"))
    summ = json.loads(Path(d["summary"]).read_text(encoding="utf-8"))
    assert full["result"]["frames"] == [{"label": "secret"}] and full["freeze_realclip_sha"] == c["rc"]
    assert "frames" not in summ["result"] and "windows" not in summ["result"] and "clips" not in summ["result"]
    assert summ["result"]["person"] == {"n": 0} and summ["attempt"] == 1
    with pytest.raises(SR.Refusal):                                             # scored once (and tree now dirty)
        SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)


def test_score_refuses_changed_perc_outputs(c, monkeypatch, tmp_path):
    seen, scorer = _scored(monkeypatch)
    _ran(c)
    write(c["root"] / "perc" / "attempt_001" / "tracks" / "0_te1.parquet", b"edited")
    with pytest.raises(SR.Refusal, match="changed since the clips attempt"):
        SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)
    write(c["root"] / "perc" / "attempt_001" / "tracks" / "0_te1.parquet", b"tclip zero")
    write(c["root"] / "perc" / "attempt_001" / "boxes" / "extra.parquet", b"x")
    with pytest.raises(SR.Refusal, match="differs from RUNLOG outputs"):
        SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)
    assert "score" not in seen


def test_score_runs_the_clips_refusals_first(c, monkeypatch, tmp_path):
    seen, scorer = _scored(monkeypatch)
    _ran(c)
    write(c["repo"] / "pkg" / "helper.py", "X = 2\n")                          # a code change after FREEZE_REALCLIP
    commit(c["repo"], "edit", "pkg/helper.py")
    with pytest.raises(SR.Refusal, match="FREEZE_REALCLIP"):
        SR.score(c["repo"], c["root"], data_root=tmp_path, scorer=scorer)
    assert "score" not in seen and "load" not in seen


def test_the_sealed_scorer_is_in_the_guarded_real_clip_set(monkeypatch):
    monkeypatch.undo()                                  # the real CLIPS_STAGE_ROOTS, not the scratch-repo stand-in
    for rel in ("intercycle/eval/realclip.py", "intercycle/observer/pixel_adapter.py", "twin/dev_labels.py"):
        assert rel in SR.CLIPS_STAGE_ROOTS
    assert SR.SITE in SR.REALCLIP_DATA_PATHS
    got = SR.import_closure(REPO, "HEAD", SR.CLIPS_STAGE_ROOTS)
    assert {"intercycle/eval/realclip.py", "twin/dev_labels.py", "twin/schemas/labels.py"} <= got
    # the container module (shipped by add_local_python_source, imported by name) is a guarded real-clip input
    assert "scripts/sealed_ingest.py" in SR.CLIPS_STAGE_ROOTS and "scripts/sealed_ingest.py" in got


def test_a_sealed_ingest_change_after_freeze_realclip_is_refused(c):
    """RC-final: scripts/sealed_ingest.py (the container code) is covered by C3 like any other file: a commit that
    changes it after FREEZE_REALCLIP refuses the clips stage."""
    write(c["repo"] / "scripts" / "sealed_ingest.py", "X = 1\n")
    commit(c["repo"], "container edit", "scripts/sealed_ingest.py")
    refused(c, "FREEZE_REALCLIP")


def test_labels_paths_must_sit_under_the_sealed_labels_dir(c):
    lab = SR.load_sealed(c["repo"], "labels", "LABELS")
    SR.check_label_files(c["root"], lab)
    moved = lab.model_copy(update={"files": [f.model_copy(update={"path": "elsewhere/" + f.path}) for f in lab.files]})
    with pytest.raises(SR.Refusal, match="not under labels/"):
        SR.check_label_files(c["root"], moved)
    up = lab.model_copy(update={"files": [f.model_copy(update={"path": "labels/../" + f.path}) for f in lab.files]})
    with pytest.raises(SR.Refusal, match="not under labels/"):
        SR.check_label_files(c["root"], up)


def test_dev_score_runs_the_score_core_with_split_dev(s, monkeypatch, tmp_path):
    """dev-dry-run --labels: the dev labels go through score's own core (LABELS-checked files under <root>/labels,
    load_scored_test, the scorer) with split dev; PERC outputs re-hashed against dev_dry_run.json."""
    seen, scorer = _scored(monkeypatch)
    data = tmp_path / "data"
    for cid, d in DEV_CLIPS.items():
        write(data / "clips" / f"{cid}.mp4", d)
    items = [SampleItem(sample_id="dev-f-1", kind="frame", clip_id="4_dv1", group_id="g", camera=14, split="dev",
                        t_pts=1.0)]
    smp = {"contract": "twin.label_samples/1", "split": "dev", "seed": 1, "rule": "r",
           "items": [i.model_dump(mode="json") for i in items], "items_sha256": sample_items_sha256(items)}
    lab_dir = tmp_path / "labels_dev"
    write(lab_dir / "sample.json", json.dumps(smp))
    write(lab_dir / "labels.json", json.dumps({"labels": []}))
    dev = tmp_path / "devroot"
    rec = SR.dev_dry_run(s["repo"], dev, list(DEV_CLIPS), data_root=data, runner=fake_runner, labels=lab_dir,
                         scorer=scorer)
    assert seen["split"] == "dev" and seen["score"]["splits"] == ("dev",)
    assert seen["load"][0] == ["labels/sample.json"] and seen["load"][1] == "labels/sample.json"
    assert seen["load"][2] == {k: sha(v) for k, v in DEV_CLIPS.items()}
    assert seen["score"]["perc"] == Path(rec["clips_stage"]["out"]) and seen["score"]["clips_dir"] == dev.resolve() / "clips"
    assert "frames" not in rec["score"]["result"] and Path(rec["score"]["full"]).is_file()
    assert os.environ.get(SR.SEALED_ENV) is None and not (s["repo"] / SR.RUNLOG).exists()
    write(Path(rec["clips_stage"]["out"]) / "tracks" / "4_dv1.parquet", b"edited")
    with pytest.raises(SR.Refusal, match="changed since the clips attempt"):
        SR.dev_score(s["repo"], dev, lab_dir, data_root=data, scorer=scorer)
