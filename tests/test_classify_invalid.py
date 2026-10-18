"""scripts/classify_invalid.py: infrastructure-invalid from the logs alone (PLAN_TWIN §5 Rules), on scratch batches."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("classify_invalid_script", REPO / "scripts" / "classify_invalid.py")
assert _spec and _spec.loader
CI = importlib.util.module_from_spec(_spec)
sys.modules["classify_invalid_script"] = CI
_spec.loader.exec_module(CI)


def good_record(tag: str) -> dict:
    return {"tag": tag, "phases": {"boot_wall_s": 4.0, "total_wall_s": 300.0}, "error": None,
            "sim": {"sim_s": 600.0}, "goal": {"result": "SUCCEEDED", "t_goal_sent": 14.7}, "plant": {"trips": 0},
            "cfg": {"tag": tag, "twin": True}}


def make_batch(tmp: Path, tags: list[str]) -> Path:
    b = tmp / "batch"
    b.mkdir()
    (b / "batch.json").write_text(json.dumps({"cfgs": [{"tag": t, "twin": True} for t in tags]}), encoding="utf-8")
    for t in tags:
        (b / f"{t}.json").write_text(json.dumps(good_record(t)), encoding="utf-8")
        d = b / t
        d.mkdir()
        for n in ("episode.json", "sim.log", "summary.json", "log.jsonl"):
            (d / n).write_text("{}\n" if n.endswith("json") else "ok\n", encoding="utf-8")
        (d / "goal.log").write_text('LEG {"goal": [1, 2, 0]}\n', encoding="utf-8")
    return b


def edit(b: Path, tag: str, **kw) -> None:
    p = b / f"{tag}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for k, v in kw.items():
        if v is None and k in ("phases",):
            d.pop(k, None)
        else:
            d[k] = v
    p.write_text(json.dumps(d), encoding="utf-8")


def test_good_episodes_are_valid(tmp_path):
    b = make_batch(tmp_path, ["a", "b"])
    out = CI.classify(b)
    assert out["invalid"] == [] and out["n"] == 2 and out["contract"] == "twin.invalid/1"


def test_missing_record_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a", "b"])
    (b / "b.json").unlink()
    out = CI.classify(b)
    assert out["invalid"] == ["b"] and "no episode record" in out["episodes"]["b"]["reason"]


@pytest.mark.parametrize("gone", ["episode.json", "sim.log", "summary.json"])
def test_missing_required_log_is_invalid(tmp_path, gone):
    b = make_batch(tmp_path, ["a"])
    (b / "a" / gone).unlink()
    e = CI.classify(b)["episodes"]["a"]
    assert e["invalid"] and gone in e["reason"]


def test_twin_run_needs_the_run_log(tmp_path):
    b = make_batch(tmp_path, ["a"])
    (b / "a" / "log.jsonl").unlink()
    assert "log.parquet|log.jsonl" in CI.classify(b)["episodes"]["a"]["reason"]


def test_record_without_sim_summary_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", sim=None)
    assert CI.classify(b)["episodes"]["a"]["invalid"]


def test_modal_error_stub_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    (b / "a.json").write_text(json.dumps({"tag": "a", "error": "FunctionTimeoutError('timeout')", "cfg": {}}),
                              encoding="utf-8")
    e = CI.classify(b)["episodes"]["a"]
    assert e["invalid"] and e["reason"].startswith("Modal/RunPod returned an error")


def test_runpod_failed_status_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", provider_status="FAILED")
    assert CI.classify(b)["episodes"]["a"]["invalid"]


@pytest.mark.parametrize("err", ["RuntimeError('sim or bridge exited during boot')",
                                 "RuntimeError('timeout waiting for the first bridge frame')",
                                 "KeyError('press.P14a.p_running')", "RemoteError('ValueError in plant')"])
def test_error_before_first_command_without_infra_sign_is_a_failure(tmp_path, err):
    """PLAN §5: our own crash before the first goal (plant / runner / cfg bug) is a failure (Y = 0), not invalid."""
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", error=err, goal=None, log_tail="Traceback (most recent call last):\n  ValueError: bad cfg\n")
    (b / "a" / "goal.log").write_text("waiting for nav2\n", encoding="utf-8")
    e = CI.classify(b)["episodes"]["a"]
    assert not e["invalid"] and not e["commanded"]
    assert e["reason"].startswith("error before first command, scored as failure")


@pytest.mark.parametrize("err, sig", [("FunctionTimeoutError('Function timed out')", "FunctionTimeoutError"),
                                      ("RuntimeError('container pre-empted by the provider')", "pre-empt"),
                                      ("RuntimeError('sim exited with code 137')", "exited with code 137")])
def test_error_before_first_command_with_infra_sign_is_invalid(tmp_path, err, sig):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", error=err, goal=None)
    (b / "a" / "goal.log").write_text("waiting for nav2\n", encoding="utf-8")
    e = CI.classify(b)["episodes"]["a"]
    assert e["invalid"] and not e["commanded"] and "before the robot's first command" in e["reason"]
    assert sig in e["reason"]


def test_infra_sign_in_the_log_tail_before_first_command_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", error="RuntimeError('sim or bridge exited during boot')", goal=None,
         log_tail="== sim ==\nKilled process 41 (python) total-vm: out of memory\n")
    (b / "a" / "goal.log").write_text("waiting for nav2\n", encoding="utf-8")
    assert CI.classify(b)["episodes"]["a"]["invalid"]


def test_oom_signature_before_first_command_is_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", goal=None)
    (b / "a" / "goal.log").write_text("starting\n", encoding="utf-8")
    (b / "a" / "sim.log").write_text("worker: Killed process 77 (python3) out of memory\n", encoding="utf-8")
    e = CI.classify(b)["episodes"]["a"]
    assert e["invalid"] and "infrastructure sign" in e["reason"]


def test_error_after_first_command_is_a_failure_not_invalid(tmp_path):
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", error="RuntimeError('planner crashed at t=200')")
    e = CI.classify(b)["episodes"]["a"]
    assert not e["invalid"] and e["commanded"] and "failure" in e["reason"]


def test_oom_after_first_command_is_a_failure(tmp_path):
    b = make_batch(tmp_path, ["a"])
    (b / "a" / "sim.log").write_text("out of memory\n", encoding="utf-8")
    assert not CI.classify(b)["episodes"]["a"]["invalid"]


def test_no_command_and_no_infra_sign_is_a_failure(tmp_path):
    """A planner that never commands the robot, with no infrastructure sign, is our failure (Y = 0), not invalid."""
    b = make_batch(tmp_path, ["a"])
    edit(b, "a", goal={"result": "REFUSED"})
    (b / "a" / "goal.log").write_text("refused to start\n", encoding="utf-8")
    e = CI.classify(b)["episodes"]["a"]
    assert not e["invalid"] and not e["commanded"]


def test_refuses_after_a_metric_or_twice_or_without_batch_json(tmp_path):
    b = make_batch(tmp_path, ["a"])
    (b / "primary_metric.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CI.ClassifyError, match="metric file exists"):
        CI.classify(b)
    (b / "primary_metric.json").unlink()
    CI.write(b, CI.classify(b))
    with pytest.raises(CI.ClassifyError, match="classified once"):
        CI.classify(b)
    (b / "batch.json").unlink()
    (b / "invalid.json").unlink()
    with pytest.raises(CI.ClassifyError, match="episode set is unknown"):
        CI.classify(b)


def test_commit_commits_only_the_list(tmp_path):
    repo = tmp_path
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    b = make_batch(tmp_path, ["a"])
    (repo / "other.txt").write_text("staged elsewhere\n", encoding="utf-8")
    subprocess.run(["git", "add", "other.txt"], cwd=repo, check=True, capture_output=True)
    assert CI.main([str(b), "--commit"]) == 0
    files = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"], cwd=repo, check=True,
                           capture_output=True, text=True).stdout.split()
    assert files == ["batch/invalid.json"]


def test_main_refusal_exit_code(tmp_path):
    b = make_batch(tmp_path, ["a"])
    (b / "metrics.json").write_text("{}", encoding="utf-8")
    assert CI.main([str(b)]) == 2
