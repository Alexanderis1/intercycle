"""scripts/gen_dev_worlds.py: dev seeds only (PREREG v1 §2); no world is generated here (the runner is faked)."""
from __future__ import annotations

import importlib.util
import json
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


def test_sets_are_the_preregistered_seeds():
    assert G.SETS["robot"] == (0, 1, 2, 3)
    assert G.SETS["calib"] == tuple(range(100, 132)) and len(G.SETS["calib"]) == 32


@pytest.mark.parametrize("seeds", [[4], [99], [132], [999], [1000], [-1], [0, 0]])
def test_refuses_other_seeds(seeds):
    with pytest.raises(G.Refusal):
        G.check_seeds(seeds, None)


def test_refuses_seed_of_the_other_set():
    with pytest.raises(G.Refusal):
        G.check_seeds([100], "robot")
    with pytest.raises(G.Refusal):
        G.check_seeds([0], "calib")
    assert G.check_seeds([3, 0], "robot") == [0, 3]


def test_refuses_bad_days_and_suffix(tmp_path):
    for d in ([0], [11], [1, 1], []):
        with pytest.raises(G.Refusal):
            G.check_days(d)
    with pytest.raises(G.Refusal):
        G.plan([0], [1], tmp_path, "../x.json")


def test_cli_refuses_test_seed_and_sealed_flag(tmp_path, monkeypatch):
    CALLS.clear()
    assert G.main(["--seeds", "5000", "--out-dir", str(tmp_path / "o"), "--suffix", SUF], runner=_ok) == 2
    monkeypatch.setenv("INTERCYCLE_SEALED_RUN", "1")
    assert G.main(["--set", "robot", "--out-dir", str(tmp_path / "o"), "--suffix", SUF], runner=_ok) == 2
    assert CALLS == [] and not (tmp_path / "o").exists()


def test_cli_launches_the_generator_per_seed_day(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    CALLS.clear()
    out = tmp_path / "o"
    assert G.main(["--set", "robot", "--days", "2-3", "--out-dir", str(out), "--suffix", SUF], runner=_ok) == 0
    assert len(CALLS) == 8
    assert all(c[1:3] == ["-m", "twin.daygen"] for c in CALLS)
    m = json.loads((out / "dev_worlds.json").read_text(encoding="utf-8"))
    assert m["seeds"] == [0, 1, 2, 3] and m["days"] == [2, 3]
    assert set(m["inputs_sha256_lf"]) == set(G.HASHED)
    # a second run into the same directory is refused
    assert G.main(["--set", "robot", "--days", "2-3", "--out-dir", str(out), "--suffix", SUF], runner=_ok) == 2


def test_generator_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    bad = lambda cmd, **kw: SimpleNamespace(returncode=3)  # noqa: E731
    assert G.main(["--seeds", "1", "--days", "1", "--out-dir", str(tmp_path / "o"), "--suffix", SUF],
                  runner=bad) == 1
    assert not (tmp_path / "o" / "dev_worlds.json").exists()
