"""S1b item 3 part 1: one site path S.load(path) threaded through run_cfg into the broadcaster, the timing planner and
the masks (twin mode), body-shop mode unchanged.

run_cfg["scenario"] (absent = body shop) is read by the episode-side broadcaster (`load_scenario`). The robot side
(observer_bridge -> masks.Site, timing_planner -> TimingCore) never opens run_cfg.json (it holds the plant seed): it
gets the same path as its existing `scenario` ROS parameter, formatted by `masks.scenario_ros_arg`.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "ros2" / "intercycle_eval", ROOT / "ros2" / "intercycle_nav"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _truth_scan import parse, py_files, string_literals  # noqa: E402
from intercycle import scenario as S  # noqa: E402
from intercycle_eval.broadcaster import check_robot_scenario, load_scenario  # noqa: E402
from intercycle_nav import masks as K  # noqa: E402
from intercycle_nav.timing import TimingCore  # noqa: E402

NAV = ROOT / "ros2" / "intercycle_nav" / "intercycle_nav"
TWIN_SITE_ID = 0x5A1B2C


@pytest.fixture
def world_dir(tmp_path) -> Path:
    """A world dir holding a stand-in 'compiled twin' scenario: the body-shop yaml with another site id."""
    d = yaml.safe_load(Path(S.DEFAULT).read_text(encoding="utf-8"))
    d["site_id"] = TWIN_SITE_ID
    wd = tmp_path / "world"
    wd.mkdir()
    (wd / "scenario.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    return wd


def _launch(cfg: dict, arg: str | None = None) -> dict:
    """cfg plus robot-side launch strings carrying `arg` (default: masks.scenario_ros_arg(cfg))."""
    arg = K.scenario_ros_arg(cfg) if arg is None else arg
    return dict(cfg, goal_cmd="python3 -u -m intercycle_nav.timing_planner --ros-args -p condition:=learned" + arg,
                extra_launch=["exec $IC_SIM_PY -u -m intercycle_eval.broadcaster --work $IC_WORK --condition learned",
                              "exec python3 -u -m intercycle_nav.observer_bridge --ros-args -p condition:=learned" + arg])


def test_body_shop_mode_is_unchanged():
    cfg = {"world_dir": "/tmp/x/world", "plant": {"seed": 3, "plant_t0": 36000.0}}
    assert K.run_cfg_scenario(cfg) is None and K.scenario_ros_arg(cfg) == ""
    scn, path = load_scenario(cfg)
    assert path is None and scn == S.load()
    assert K.Site.from_scenario().site_id == S.load().site_id
    assert K.run_cfg_scenario({"scenario": ""}) is None and K.scenario_ros_arg({"scenario": ""}) == ""


def test_twin_scenario_reaches_broadcaster_masks_and_timing(world_dir, monkeypatch):
    cfg = _launch({"world_dir": str(world_dir), "scenario": "scenario.yaml"})
    scn, path = load_scenario(cfg)
    assert path is not None
    assert path == os.path.join(str(world_dir), "scenario.yaml") and scn.site_id == TWIN_SITE_ID
    arg = K.scenario_ros_arg(cfg)
    assert arg == " -p scenario:=$IC_WORLD/scenario.yaml"
    monkeypatch.setenv("IC_WORLD", str(world_dir))             # what the episode exports to launched commands
    robot_path = os.path.expandvars(arg.split(":=", 1)[1])
    assert os.path.abspath(robot_path) == os.path.abspath(path)
    assert K.Site.from_scenario(robot_path).site_id == TWIN_SITE_ID          # observer_bridge -> masks
    core = TimingCore.from_scenario(S.load(robot_path))                      # timing_planner -> timing
    assert core.graph.scn.site_id == TWIN_SITE_ID


def test_absolute_paths_are_kept(world_dir):
    p = str(world_dir / "scenario.yaml")
    cfg = {"scenario": p}
    assert K.run_cfg_scenario(cfg) == p and K.scenario_ros_arg(cfg) == f" -p scenario:={p}"
    assert load_scenario(_launch(cfg))[0].site_id == TWIN_SITE_ID


def test_bad_scenario_values_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="world_dir"):
        K.run_cfg_scenario({"scenario": "scenario.yaml"})
    with pytest.raises(TypeError):
        K.run_cfg_scenario({"scenario": 7, "world_dir": "/w"})
    for bad in (7, "a b.yaml", "x;rm.yaml", "$HOME/s.yaml", "s'.yaml"):
        with pytest.raises(ValueError):
            K.scenario_ros_arg({"scenario": bad})
    with pytest.raises(FileNotFoundError):
        load_scenario(_launch({"scenario": str(tmp_path / "missing.yaml")}))


def _param_flows(path: Path, sink: str) -> bool:
    """The module declares the `scenario` parameter and passes a name bound from it to `sink` (S.load / from_scenario)."""
    tree = parse(path)
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(c, ast.Constant) and c.value == "scenario"
                                             for c in ast.walk(n.value)):
            bound |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == sink:
            if any(isinstance(a, ast.Name) and a.id in bound for a in ast.walk(n)):
                return True
    return False


def test_robot_side_takes_the_scenario_parameter():
    assert _param_flows(NAV / "timing_planner.py", "load")
    assert _param_flows(NAV / "observer_bridge.py", "from_scenario")


def test_robot_side_never_names_run_cfg():
    hits = {p.name: s for p in py_files("ros2/intercycle_nav/intercycle_nav") for s in string_literals(parse(p))
            if "run_cfg.json" in s}
    assert not hits, f"robot-side code names the run_cfg.json file (it holds the plant seed): {hits}"


# ------------------------------------------------------------------ fail closed on a half-threaded scenario (fix1)


def test_twin_cfg_from_the_real_batch_builder_without_the_argument_is_refused(world_dir):
    """The real launch path: batch.episode_cfg's strings do not carry the scenario yet (patch sent to its owner). A twin
    run_cfg built on them would sign frames with the twin site_id while the robot loads the body shop (every frame
    'wrong_site', silently no feed): the broadcaster refuses it before signing anything."""
    from intercycle_eval.batch import episode_cfg

    for cond in ("learned", "oracle"):
        cfg = dict(episode_cfg("x", 20, cond), world_dir=str(world_dir), scenario="scenario.yaml")
        with pytest.raises(ValueError, match="scenario"):
            load_scenario(cfg)
        arg = K.scenario_ros_arg(cfg)                          # the patch: append the argument to bridge and goal
        fixed = dict(cfg, goal_cmd=cfg["goal_cmd"] + arg,
                     extra_launch=[c + arg if "intercycle_nav.observer_bridge" in c else c for c in cfg["extra_launch"]])
        scn, path = load_scenario(fixed)
        assert scn.site_id == TWIN_SITE_ID and path == os.path.join(str(world_dir), "scenario.yaml")
        load_scenario(episode_cfg("x", 20, cond))              # body-shop batch cfgs are accepted unchanged


def test_half_threaded_twin_cfgs_are_refused(world_dir):
    base = {"world_dir": str(world_dir), "scenario": "scenario.yaml"}
    good = _launch(base)
    check_robot_scenario(good)
    assert good["extra_launch"][1].endswith(" -p scenario:=$IC_WORLD/scenario.yaml")
    bare = _launch(base, "")
    cases = {
        "no goal_cmd": dict(good, goal_cmd=""),
        "no bridge": dict(good, extra_launch=good["extra_launch"][:1]),
        "goal lacks arg": dict(good, goal_cmd=bare["goal_cmd"]),
        "bridge lacks arg": dict(good, extra_launch=bare["extra_launch"]),
        "a second bridge lacks arg": dict(good, extra_launch=good["extra_launch"] + bare["extra_launch"][1:]),
        "other path": _launch(base, " -p scenario:=$IC_WORLD/other.yaml"),
        "overridden later": dict(good, goal_cmd=good["goal_cmd"] + " -p scenario:=/x/bodyshop.yaml"),
        "empty value": _launch(base, " -p scenario:="),
    }
    for name, cfg in cases.items():
        try:
            check_robot_scenario(cfg)
        except ValueError:
            continue
        pytest.fail(f"accepted a half-threaded twin cfg: {name}")
    with pytest.raises(TypeError):
        check_robot_scenario(dict(good, extra_launch="exec python3 -m intercycle_nav.observer_bridge"))


def test_body_shop_cfg_with_a_stray_scenario_argument_is_refused(world_dir):
    """The reverse mismatch: no run_cfg scenario (broadcaster signs the body shop) but the robot would load another."""
    stray = _launch({"world_dir": str(world_dir)}, " -p scenario:=$IC_WORLD/scenario.yaml")
    with pytest.raises(ValueError, match="body shop"):
        load_scenario(stray)
    check_robot_scenario(_launch({"world_dir": str(world_dir)}))
