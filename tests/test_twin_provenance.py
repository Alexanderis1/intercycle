"""Provenance (PLAN_TWIN §5): every clip-derived value names its clips, and they are TRAIN ids of the active split."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from twin import provenance as PV
from twin import schemas as S
from twin.schemas import prereg as PR
from twin.schemas.common import Tagged

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SHA = "ab" * 32


def _raw(name: str):
    text = (FIX / name).read_text(encoding="utf-8")
    return yaml.safe_load(text) if name.endswith(".yaml") else json.loads(text)


def _split_text(rows: list[tuple[str, str, str]]) -> str:
    return "clip_id,group_id,split,sha256\n" + "".join(f"{c},{g},{s},{SHA}\n" for c, g, s in rows)


FIXTURE_SPLIT = PR.parse_split(_split_text([("c9_train_0001", "g01", "train"), ("c14_train_0002", "g03", "train"),
                                            ("c9_dev_0003", "g02", "dev"), ("0_te1", "g09", "test")]))


# ------------------------------------------------------------------------------------------------ schema rules


@pytest.mark.parametrize("source", ["cal_v0", "cal_v1", "traced_v0", "traced_v1", "measured"])
def test_clip_derived_tagged_value_needs_clips(source):
    with pytest.raises(ValidationError, match="clip ids"):
        Tagged[float].model_validate({"value": 1.0, "source": source})
    Tagged[float].model_validate({"value": 1.0, "source": source, "clips": ["c9_train_0001"]})
    Tagged[float].model_validate({"value": 1.0, "source": "prior"})


@pytest.mark.parametrize("source", ["fitted", "fitted_censored"])
def test_fitted_param_needs_clips(source):
    d = _raw("twin_dynamics.json")
    p = d["observed"]["person.speed_mps"]
    p["source"], p["clips"] = source, []
    with pytest.raises(ValidationError, match="clip ids it was fitted on"):
        S.Dynamics.model_validate(d)
    p["clips"] = [""]
    with pytest.raises(ValidationError, match="clip ids it was fitted on"):
        S.Dynamics.model_validate(d)


def test_traced_checkpoint_and_site_need_clips():
    d = _raw("twin_tasks.yaml")
    d["checkpoints"][0]["clips"] = []
    with pytest.raises(ValidationError, match="clip ids"):
        S.Tasks.model_validate(d)
    s = _raw("twin_site.yaml")
    del s["presses"][0]["dilation_m"]["clips"]
    with pytest.raises(ValidationError, match="clip ids"):
        S.Site.model_validate(s)


# ------------------------------------------------------------------------------------------------ split format


def test_split_parser_rules():
    assert FIXTURE_SPLIT["0_te1"].split == "test"
    with pytest.raises(ValueError, match="header"):
        PR.parse_split("clip,group_id,split,sha256\nx,g,train," + SHA + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        PR.parse_split(_split_text([("a", "g", "train"), ("a", "g", "dev")]))
    with pytest.raises(ValidationError):
        PR.parse_split(_split_text([("a", "g", "holdout")]))
    # an amended split may append columns after the four contract columns
    extra = "clip_id,group_id,split,sha256,a1_reason\n" + f"a,g,train,{SHA},moved\n"
    assert PR.parse_split(extra)["a"].split == "train"


def test_active_split_pointer(tmp_path):
    (tmp_path / "prereg").mkdir()
    with pytest.raises(FileNotFoundError):
        PR.active_split_path(tmp_path)
    (tmp_path / "prereg" / "split.csv").write_text(_split_text([("a", "g", "train")]), encoding="utf-8")
    assert PR.active_split_path(tmp_path).name == "split.csv"
    (tmp_path / "prereg" / "ACTIVE_SPLIT").write_text("# amended at A1+\nprereg/split_a1.csv\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        PR.active_split_path(tmp_path)                                    # named file missing: fail closed
    (tmp_path / "prereg" / "split_a1.csv").write_text(_split_text([("a", "g", "dev")]), encoding="utf-8")
    assert PR.load_active_split(tmp_path)["a"].split == "dev"


def test_repo_active_split_parses_if_present():
    try:
        rows = PR.load_active_split()
    except FileNotFoundError:
        pytest.skip("no split committed on this branch yet")
    assert rows and {r.split for r in rows.values()} <= {"train", "dev", "test"}


# ------------------------------------------------------------------------------------------------ train-only check


def _fixture_refs():
    return PV.source_clips(site=S.load(S.Site, FIX / "twin_site.yaml"),
                           dynamics=S.load(S.Dynamics, FIX / "twin_dynamics.json"),
                           safety=S.load(S.Safety, FIX / "twin_safety.yaml"),
                           tasks=S.load(S.Tasks, FIX / "twin_tasks.yaml"),
                           day_scripts=[S.load(S.DayScript, FIX / "twin_day_script.json")])


def test_fixture_sources_are_all_train():
    refs = _fixture_refs()
    assert set(refs) == {"site", "dynamics", "safety", "tasks", "generated_days"}
    assert refs["safety"] == ["c14_train_0002"] and refs["dynamics"] == ["c9_train_0001"]
    assert PV.assert_train_only(refs, FIXTURE_SPLIT) > 0


@pytest.mark.parametrize("where,cid,why", [("dynamics", "c9_dev_0003", "dev"), ("safety", "0_te1", "test"),
                                           ("site", "nowhere_9", "not in split")])
def test_non_train_source_fails(where, cid, why):
    refs = _fixture_refs()
    refs[where] = [*refs[where], cid]
    with pytest.raises(PV.ProvenanceError, match=why):
        PV.assert_train_only(refs, FIXTURE_SPLIT)


def test_safety_evidence_counts_as_a_source():
    d = copy.deepcopy(_raw("twin_safety.yaml"))
    d["presses"][1]["guard"]["evidence_clips"] = ["c9_dev_0003"]
    refs = PV.source_clips(safety=S.Safety.model_validate(d))
    with pytest.raises(PV.ProvenanceError, match="c9_dev_0003"):
        PV.assert_train_only(refs, FIXTURE_SPLIT)
