"""scripts/gen_dev_worlds.py: dev seeds only (PREREG v1 §2); no world is generated here (the runner is faked).

The PREREG-committed gate (`prereg_gate`) is tested on scratch git repositories; the CLI tests pass a fake gate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gen_dev_worlds", ROOT / "scripts" / "gen_dev_worlds.py")
assert _spec is not None and _spec.loader is not None
G = importlib.util.module_from_spec(_spec)
sys.modules["gen_dev_worlds"] = G
_spec.loader.exec_module(G)

SUF = ".x.json"     # a neutral suffix: the fake runner writes nothing


CALLS: list[list[str]] = []


def _ok(cmd, **kw):
    CALLS.append(cmd)
    return SimpleNamespace(returncode=0)


GATE = {"head": "h" * 40, "prereg_sha256_lf": "s" * 64, "tree_clean": True}


def _gate():
    return dict(GATE)


def _no_gate():
    raise G.Refusal("PREREG not committed")


def test_sets_are_the_preregistered_seeds():
    assert G.SETS["robot"] == (0, 1, 2, 3)
    assert G.SETS["calib"] == tuple(range(100, 132)) and len(G.SETS["calib"]) == 32


@pytest.mark.parametrize("set_name", ["robot", "calib"])
@pytest.mark.parametrize("seeds", [[4], [99], [132], [999], [1000], [-1], [0, 0], [100, 100]])
def test_refuses_other_seeds(seeds, set_name):
    with pytest.raises(G.Refusal):
        G.check_seeds(seeds, set_name)


def test_refuses_seed_of_the_other_set():
    with pytest.raises(G.Refusal):
        G.check_seeds([100], "robot")
    with pytest.raises(G.Refusal):
        G.check_seeds([0], "calib")
    with pytest.raises(G.Refusal):
        G.check_seeds([0, 100], "robot")
    with pytest.raises(G.Refusal):
        G.check_seeds([0], None)  # type: ignore[arg-type]  # the set is always required
    assert G.check_seeds([3, 0], "robot") == [0, 3]


def test_refuses_bad_days_and_suffix(tmp_path):
    for d in ([0], [11], [1, 1], []):
        with pytest.raises(G.Refusal):
            G.check_days(d)
    with pytest.raises(G.Refusal):
        G.plan([0], [1], tmp_path, "../x.json")


def test_cli_refuses_test_seed_mixed_sets_and_sealed_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    CALLS.clear()
    o = str(tmp_path / "o")
    assert G.main(["--set", "robot", "--seeds", "5000", "--out-dir", o, "--suffix", SUF], runner=_ok,
                  gate=_gate) == 2
    # a batch that spans both sets is refused (the set is always recorded)
    assert G.main(["--set", "robot", "--seeds", "0,100", "--out-dir", o, "--suffix", SUF], runner=_ok,
                  gate=_gate) == 2
    with pytest.raises(SystemExit):     # --set is required
        G.main(["--seeds", "0", "--out-dir", o, "--suffix", SUF], runner=_ok, gate=_gate)
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    assert G.main(["--set", "robot", "--out-dir", o, "--suffix", SUF], runner=_ok, gate=_gate) == 2
    assert CALLS == [] and not (tmp_path / "o").exists()


def test_cli_refuses_without_committed_prereg(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    CALLS.clear()
    assert G.main(["--set", "robot", "--out-dir", str(tmp_path / "o"), "--suffix", SUF], runner=_ok,
                  gate=_no_gate) == 2
    assert CALLS == [] and not (tmp_path / "o").exists()


def test_cli_launches_the_generator_per_seed_day(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    CALLS.clear()
    out = tmp_path / "o"
    assert G.main(["--set", "robot", "--days", "2-3", "--out-dir", str(out), "--suffix", SUF], runner=_ok,
                  gate=_gate) == 0
    assert len(CALLS) == 8
    assert all(c[1:3] == ["-m", "twin.daygen"] for c in CALLS)
    m = json.loads((out / "dev_worlds.json").read_text(encoding="utf-8"))
    assert m["set"] == "robot" and m["seeds"] == [0, 1, 2, 3] and m["days"] == [2, 3]
    assert m["prereg"] == GATE
    assert set(m["inputs_sha256_lf"]) == set(G.HASHED)
    # a second run into the same directory is refused
    assert G.main(["--set", "robot", "--days", "2-3", "--out-dir", str(out), "--suffix", SUF], runner=_ok,
                  gate=_gate) == 2


def test_generator_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    bad = lambda cmd, **kw: SimpleNamespace(returncode=3)  # noqa: E731
    assert G.main(["--set", "robot", "--seeds", "1", "--days", "1", "--out-dir", str(tmp_path / "o"),
                   "--suffix", SUF], runner=bad, gate=_gate) == 1
    assert not (tmp_path / "o" / "dev_worlds.json").exists()


# ------------------------------------------------------------------ prereg_gate on a scratch git repository


def _g(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _g(r, "init", "-q", "-b", "main")
    _g(r, "config", "user.email", "t@example.invalid")
    _g(r, "config", "user.name", "t")
    _g(r, "config", "core.autocrlf", "false")
    _g(r, "config", "commit.gpgsign", "false")
    (r / "README").write_text("x\n", encoding="utf-8")
    _g(r, "add", "README")
    _g(r, "commit", "-q", "-m", "base")
    return r


def _commit_prereg(r: Path, text: str = "PREREG v1\n") -> None:
    (r / "prereg").mkdir(exist_ok=True)
    (r / "prereg" / "PREREG_TWIN.md").write_text(text, encoding="utf-8", newline="\n")
    _g(r, "add", "prereg/PREREG_TWIN.md")
    _g(r, "commit", "-q", "-m", "prereg")


def test_gate_refuses_when_prereg_missing(repo):
    with pytest.raises(G.Refusal, match="not tracked"):
        G.prereg_gate(repo)


def test_gate_refuses_modified_prereg_or_dirty_tree(repo):
    _commit_prereg(repo)
    (repo / "prereg" / "PREREG_TWIN.md").write_text("PREREG v1 edited\n", encoding="utf-8")
    with pytest.raises(G.Refusal, match="not clean"):
        G.prereg_gate(repo)
    _g(repo, "checkout", "--", "prereg/PREREG_TWIN.md")
    (repo / "stray.py").write_text("x = 1\n", encoding="utf-8")       # untracked files count as dirty
    with pytest.raises(G.Refusal, match="not clean"):
        G.prereg_gate(repo)


def test_gate_refuses_prereg_not_on_main(repo):
    _g(repo, "checkout", "-q", "-b", "lane")
    _commit_prereg(repo)
    with pytest.raises(G.Refusal, match="not on main"):
        G.prereg_gate(repo)


def test_gate_refuses_prereg_changed_after_main(repo):
    _commit_prereg(repo)
    _g(repo, "checkout", "-q", "-b", "lane")
    (repo / "prereg" / "PREREG_TWIN.md").write_text("PREREG v1 changed on a lane\n", encoding="utf-8")
    _g(repo, "commit", "-q", "-am", "edit")
    with pytest.raises(G.Refusal, match="differs from main"):
        G.prereg_gate(repo)


def test_gate_passes_and_records_head_and_hash(repo):
    _commit_prereg(repo)
    _g(repo, "checkout", "-q", "-b", "lane")
    (repo / "code.py").write_text("y = 2\n", encoding="utf-8")
    _g(repo, "add", "code.py")
    _g(repo, "commit", "-q", "-m", "later work")
    rec = G.prereg_gate(repo)
    assert rec["head"] == _g(repo, "rev-parse", "HEAD") and rec["tree_clean"] is True
    assert rec["main"] == _g(repo, "rev-parse", "main")
    assert rec["prereg_added_in"] == _g(repo, "rev-parse", "main")
    assert rec["prereg_git_blob"] == _g(repo, "rev-parse", "HEAD:prereg/PREREG_TWIN.md")
    assert rec["prereg_sha256_lf"] == hashlib.sha256(b"PREREG v1\n").hexdigest()
