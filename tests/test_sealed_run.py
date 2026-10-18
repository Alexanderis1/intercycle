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
DEV_CLIPS = {"4_dv1": b"dev clip four"}
INFO = {"genesis_time": 1000, "period": 30, "hash": "testchain"}
G2_FILE = "prereg/G2_HASHES.json"
G2_OTHER = {"twin/engine.py": "ENG = 1\n", "twin/camera.py": "CAM = 1\n", "twin/site.yaml": "site: 1\n"}


@pytest.fixture(autouse=True)
def _g2_file(monkeypatch):
    """The scratch repos commit a G2 hash record at this path (the real file is the lead's call)."""
    monkeypatch.setattr(SR, "G2_HASHES_FILE", G2_FILE)


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
    code = commit(repo, "code", "twin", "prereg", "sealed")
    write(repo / "prereg" / "FREEZE", code + "\n")
    commit(repo, "FREEZE", "prereg/FREEZE")
    if source == "fallback":
        line = SR.derive(repo, root, fallback=True)["seed_commit"]
    else:
        t_freeze = SR.freeze_commit_time(repo)
        line = SR.derive(repo, root, fallback=False, fetch=fake_fetch, now=lambda: t_freeze + 60.0)["seed_commit"]
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
    assert log["contract"] == SR.RUNLOG_CONTRACT and [r["status"] for r in log["records"]] == ["started", "done"]
    start, done = log["records"]
    for k in ("attempt", "timestamp", "head", "freeze_sha", "inputs", "seeds", "stages", "status", "perc_dir"):
        assert k in start
    assert start["attempt"] == done["attempt"] == 1 and start["stages"] == ["clips"]
    assert start["seeds"]["salt"] and len(start["seeds"]["seeds"]) == 16
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
    assert seen["out"] == s["repo"] / SR.DOUBLE_OUT / "attempt_001"
    assert rec["stages"]["double"]["status"] == "done" and rec["stages"]["double"]["n_missions"] == 192
    with pytest.raises(SR.Refusal, match="sealed flag"):
        SR.run_double_stage(s["repo"], 9, [1000] * 16, fake_double)   # never outside an attempt


def test_double_stage_refuses_before_any_runlog_record_while_the_backend_is_not_wired(s, monkeypatch):
    """No fake runner and run_double.SEALED_BACKEND_READY False: sealed_run refuses before the drand fetch and before
    the RUNLOG `started` record (no attempt number used up)."""
    monkeypatch.setattr(SR._run_double_module(), "SEALED_BACKEND_READY", False)
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


def test_runner_returning_fewer_clips_is_an_error(s):
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


def test_a_second_run_appends_a_new_attempt_to_a_dirty_runlog(s):
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


def test_a_rerun_after_a_stage_error_is_a_new_attempt(s):
    def broken(batches):
        raise RuntimeError("provider failure")

    with pytest.raises(RuntimeError, match="provider failure"):
        _run(s, broken)
    assert not (s["root"] / "perc" / "attempt_001").exists()
    r2 = _run(s)
    assert r2["attempt"] == 2 and [(r["attempt"], r["status"]) for r in _records(s)] == [
        (1, "started"), (1, "error"), (2, "started"), (2, "done")]


def test_a_committed_runlog_is_allowed_and_must_only_grow(s):
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    assert SR.check_diff(s["repo"], s["code"]) == sorted([SR.FREEZE, SR.SEED_COMMIT, SR.RUNLOG])
    _run(s)                                                             # passes check_clean / check_diff, appends
    assert len(_records(s)) == 4
    commit(s["repo"], "RUNLOG attempt 2", SR.RUNLOG)
    p = s["repo"] / SR.RUNLOG
    d = json.loads(p.read_text(encoding="utf-8"))
    d["records"] = d["records"][:2]                                     # drop committed records in the working file
    p.write_text(json.dumps(d), encoding="utf-8")
    refused(s, "not an append-only extension")


def test_a_committed_rewrite_of_runlog_is_refused(s):
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
def test_a_rewrite_committed_after_freeze_is_refused(s, edit):
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    _commit_edited_runlog(s, edit, "RUNLOG history edited")
    assert g(s["repo"], "status", "--porcelain", "--untracked-files=all") == ""   # working file == HEAD
    with pytest.raises(SR.Refusal, match=r"not an append-only extension.*commit [0-9a-f]{12} has 2 records"):
        SR.check_runlog(s["repo"], s["code"])
    refused(s, "not an append-only extension")


def test_a_committed_rewrite_stays_refused_after_a_later_restore(s):
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    good = (s["repo"] / SR.RUNLOG).read_text(encoding="utf-8")
    _commit_edited_runlog(s, lambda d: d["records"][0].update(timestamp="2000-01-01T00:00:00+00:00"), "rewrite")
    (s["repo"] / SR.RUNLOG).write_text(good, encoding="utf-8")         # the HEAD version looks untouched again
    commit(s["repo"], "restore", SR.RUNLOG)
    refused(s, "not an append-only extension")


def test_a_committed_deletion_of_runlog_is_refused(s):
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    g(s["repo"], "rm", "-q", "--", SR.RUNLOG)
    g(s["repo"], "commit", "-q", "-m", "drop RUNLOG")
    refused(s, "not an append-only extension")


def test_runlog_check_starts_from_the_freeze_tree(s):
    _run(s)
    commit(s["repo"], "RUNLOG attempt 1", SR.RUNLOG)
    head = g(s["repo"], "rev-parse", "HEAD")
    assert len(SR.check_runlog(s["repo"], head)) == 2                  # FREEZE tree holds the 2 records: a prefix
    with pytest.raises(SR.Refusal, match="the FREEZE tree"):
        _commit_edited_runlog(s, lambda d: d["records"].pop(), "truncate")
        SR.check_runlog(s["repo"], head)


def test_only_runlog_may_be_dirty(s):
    _run(s)
    write(s["repo"] / "results" / "sealed" / "other.json", "{}\n")
    refused(s, r"not clean:\n\?\? results/sealed/other.json")


def test_append_refuses_when_the_file_is_not_the_known_prefix(s):
    _run(s)
    known = _records(s)
    with pytest.raises(SR.Refusal, match="not a prefix"):
        SR.append_runlog(s["repo"], known[:1], {"attempt": 2, "status": "started", "timestamp": "t", "head": "h",
                                                "stages": ["clips"]})
    assert _records(s) == known


def test_runlog_must_parse_and_attempt_dirs_are_fresh(s):
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

def test_a_rerun_after_committed_results_sealed_outputs_is_allowed(s):
    """LEAD DECISION: sealed_run's own outputs under results/sealed/** are exempt from the FREEZE..HEAD diff rule; each
    invocation appends a new RUNLOG attempt and keeps the earlier records."""
    _run(s)
    write(s["repo"] / "results" / "sealed" / "attempt_001" / "summary.json", "{}\n")
    commit(s["repo"], "sealed attempt 1 outputs", SR.RUNLOG, "results/sealed/attempt_001/summary.json")
    assert SR.check_diff(s["repo"], s["code"]) == sorted([SR.FREEZE, SR.SEED_COMMIT, SR.RUNLOG,
                                                         "results/sealed/attempt_001/summary.json"])
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
