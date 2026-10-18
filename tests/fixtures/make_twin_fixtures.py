"""Regenerate the twin golden fixtures (tests/fixtures/twin_*). Run from the repo root:

    python tests/fixtures/make_twin_fixtures.py

Every value is synthetic (no real clip, track or site data). tests/test_twin_contracts.py validates the committed files;
regenerate only when a contract changes on purpose (and say so in the commit).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("INTERCYCLE_TRUTH_LOG", str(Path(tempfile.gettempdir()) / "fixture_truth_access.log"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

from twin.daygen_stub import generate_day, write_day  # noqa: E402
from twin.schemas.runs import MissionMetric, world_values  # noqa: E402

HDR = "# Golden fixture for {c} (tests/test_twin_contracts.py). Synthetic values, not real data.\n"
C9, C14 = "c9_train_0001", "c14_train_0002"
SHA = "0123456789abcdef" * 4


def dump_yaml(name: str, contract: str, data: dict) -> None:
    text = HDR.format(c=contract) + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120)
    (HERE / name).write_text(text, encoding="utf-8", newline="\n")


def dump_json(name: str, data) -> None:
    (HERE / name).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def site() -> None:
    s = yaml.safe_load((REPO / "twin" / "site_v0.yaml").read_text(encoding="utf-8"))
    s["site_id"] = "fixture_site"
    s["cal_sources"] = [C9, C14]
    f9, f14 = s["framings"]
    f9["H_img_to_floor"].update(source="cal_v0", clips=[C9])
    f9["k1"] = {"value": -0.12, "source": "cal_v0", "clips": [C9]}
    f14["H_img_to_floor"].update(source="cal_v0", clips=[C14])
    s["zones"][0]["floor_polygon"].update(source="traced_v0", clips=[C9])
    s["presses"][0]["footprint"].update(source="traced_v0", clips=[C9])
    s["presses"][0]["dilation_m"] = {"value": 0.22, "source": "cal_v1", "clips": [C9], "note": "p95 reprojection error"}
    s["scale"] = {"value": {"lo": 0.9, "mid": 1.0, "hi": 1.12}, "source": "cal_v1", "clips": [C9, C14]}
    dump_yaml("twin_site.yaml", "twin.site/1", s)


def tasks() -> None:
    def cp(i, zone, press, x, z, inside):
        return {"id": f"cp{i}", "zone": zone, "press": press, "target_kind": "press" if press else "panel",
                "target_xzh": [x, z, 1.2], "normal_xz": [0.0, -1.0],
                "candidate_poses": [{"pose": {"x": x, "z": z - 1.5, "yaw": 0.0}, "source": "generated", "standoff_m": 1.5}],
                "inside_danger": inside, "dwell_s": 8.0, "zoom": 20.0 if press else 4.0, "thermal": bool(press),
                "source": "traced_v0", "clips": [C14 if zone == "z14" else C9]}
    data = {
        "contract": "twin.tasks/1", "site_id": "fixture_site",
        "inspection": {"defect_rate": 0.25, "zoom_max": 20.0,
                       "detectability": "p_detect = 1 - exp(-(px_per_mm / 2.0)) * cos(incidence)"},
        "zones": [{"zone": "z9", "dock": {"x": 1.0, "z": 1.0, "yaw": 0.0}, "mission_order": ["cp1", "cp2"]},
                  {"zone": "z14", "dock": {"x": 31.0, "z": 1.0, "yaw": 0.0}, "mission_order": ["cp3"]}],
        "checkpoints": [cp(1, "z9", "P9a", 6.2, 5.5, True), cp(2, "z9", None, 11.0, 3.0, False),
                        cp(3, "z14", "P14a", 34.2, 4.5, True)],
    }
    dump_yaml("twin_tasks.yaml", "twin.tasks/1", data)


def safety() -> None:
    rules = [{"press_state": "running", "action": "trip"}, {"press_state": "idle", "action": "hold"},
             {"press_state": "stopped", "action": "allow"}, {"press_state": "tripped", "action": "hold"},
             {"press_state": "held", "action": "hold"}, {"press_state": "resetting", "action": "hold"}]
    data = {"contract": "twin.safety/1", "site_id": "fixture_site", "presses": [
        {"press": "P9a", "guard": {"type": "ASSUMED", "device": "light_curtain", "reset": "manual"},
         "reset": {"family": "lognormal", "median_s": 240.0, "p90_s": 900.0, "sweep_s": {"lo": 60.0, "hi": 900.0}},
         "entry_rules": rules},
        {"press": "P14a", "guard": {"type": "OBSERVED", "device": "fixed_fence", "reset": "auto", "evidence_clips": [C14]},
         "reset": {"family": "lognormal", "median_s": 60.0, "p90_s": 180.0, "sweep_s": {"lo": 30.0, "hi": 300.0}},
         "entry_rules": rules}]}
    dump_yaml("twin_safety.yaml", "twin.safety/1", data)


def dynamics() -> None:
    d = json.loads((REPO / "twin" / "dynamics_v0.json").read_text(encoding="utf-8"))
    d["site_id"], d["version"] = "fixture_site", "fixture"
    d["observed"]["person.speed_mps"] = {"value": 1.15, "unit": "m/s", "source": "fitted", "n": 412,
                                         "sweep": {"lo": 0.9, "hi": 1.4}, "clips": [C9],
                                         "dist": {"family": "lognormal", "params": {"median": 1.15, "sigma": 0.22}}}
    d["observed"]["press.P9a.run_bout_median_s"] = {"value": 420.0, "unit": "s", "source": "fitted_censored", "n": 17,
                                                    "sweep": {"lo": 200.0, "hi": 900.0}, "clips": [C9],
                                                    "dist": {"family": "kaplan_meier", "params": {}}}
    dump_json("twin_dynamics.json", d)


def day_script() -> None:
    write_day(generate_day(7, 2, duration_s=300.0), HERE / "twin_day_script.json")


def realised() -> None:
    lines = [{"kind": "header", "contract": "twin.realised_state/1", "run_id": "fixture_run", "world": 0, "day": 2,
              "hz": 2.0, "t0_s": 86400.0, "zones": ["z9", "z14"], "presses": ["P9a", "P14a"]}]
    for i in range(8):
        t = 86400.0 + 0.5 * i
        ram = 0.5 * (1 - math.cos(2 * math.pi * (0.5 * i) / 4.0))
        lines.append({"kind": "tick", "t": t, "presses": [
            {"id": "P9a", "state": "running", "ram": round(ram, 6), "stroke": i % 8 == 0, "freeze": False},
            {"id": "P14a", "state": "stopped", "ram": 0.0, "stroke": False, "freeze": False}],
            "movers": [
                {"id": "walker:z9:0000", "cls": "person", "zone": "z9", "x": 2.0 + 0.6 * i, "z": 3.0, "heading": 1.5708,
                 "speed": 1.2, "r": 0.3},
                {"id": "walker:z9:0001", "cls": "person", "zone": "z9", "x": 6.0, "z": 5.7, "heading": 0.0,
                 "speed": 0.0, "r": 0.3},
                {"id": "forklift:0", "cls": "forklift", "zone": "z9", "x": 10.0, "z": 2.0 + 0.9 * i, "heading": 0.0,
                 "speed": 1.8, "r": 1.0},
                {"id": "robot", "cls": "robot", "zone": "z14", "x": 31.0, "z": 1.0, "heading": 0.0, "speed": 0.0,
                 "r": 0.45}]})
    (HERE / "twin_realised_state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8",
                                                    newline="\n")


def summary() -> None:
    dump_json("twin_summary.json", {
        "contract": "twin.summary/1", "run_id": "w0_d9_z9_learned", "engine": "double", "world": 0, "world_split": "dev",
        "day": 9, "zone": "z9", "condition": "learned", "T_s": 720.0, "status": "ok", "status_reason": None,
        "checkpoints_planned": 2, "attempted": 2, "passed": 1, "declined": 0, "robot_caused_press_stops": 0,
        "holds": 0, "incidents": 0, "press_time_lost_s": 0.0, "path_length_m": 41.5,
        "code": {"head": "8c52e12", "generator_sha256": SHA, "dynamics_sha256": SHA}, "wall_s": 380.2,
        "camera": {"model": "stat_v1", "cal_source": "cal_v1 draws", "h_mode": "joint",
                   "draw_index": {"9": 17, "14": 103}, "hold_s": 34.0, "hold_s_by_camera": {"9": 34.0, "14": 7.5},
                   "params_sha256": SHA}})


def primary_metric() -> None:
    def m(run, world, day, zone, cond, outs, stop=False, hold=False, lost=0.0):
        zeroed = stop or hold
        return {"run_id": run, "world": world, "day": day, "zone": zone, "condition": cond, "T_s": 720.0,
                "Y": 0 if zeroed else [o for _, o in outs].count("passed"), "robot_caused_press_stop": stop,
                "hold": hold, "incident": False, "error": False, "press_time_lost_s": lost,
                "checkpoints": [{"checkpoint": c, "outcome": o} for c, o in outs]}
    missions = [
        m("w0_d9_z9_learned", 0, 9, "z9", "learned", [("cp1", "passed"), ("cp2", "passed")]),
        m("w0_d9_z14_learned", 0, 9, "z14", "learned", [("cp3", "declined")]),
        m("w0_d9_z9_reactive", 0, 9, "z9", "reactive", [("cp1", "tripped"), ("cp2", "passed")], stop=True, lost=310.0),
        m("w0_d9_z14_reactive", 0, 9, "z14", "reactive", [("cp3", "held")], hold=True, lost=95.0),
        m("w1_d9_z9_learned", 1, 9, "z9", "learned", [("cp1", "failed"), ("cp2", "unattempted")]),
    ]
    pw = [w.model_dump() for w in world_values([MissionMetric.model_validate(x) for x in missions])]
    dump_json("twin_primary_metric.json", {"contract": "twin.primary_metric/1", "engine": "double", "split": "dev",
                                           "missions": missions, "per_world": pw})


def parquet() -> None:
    inv = {"clip_id": ["c9_train_0001", "c9_dev_0003", "c14_train_0002", "c14_dev_0004"],
           "camera": [9, 9, 14, 14], "framing_id": ["cam9_f0", "cam9_f0", "cam14_f0", "cam14_f0"],
           "group_id": ["g01", "g02", "g03", "g04"], "class": [0, 4, 3, 7], "duration": [5.2, 4.8, 6.0, 3.1],
           "sha256": [SHA] * 4, "split": ["train", "dev", "train", "dev"]}
    pq.write_table(pa.table(inv, schema=pa.schema([("clip_id", pa.string()), ("camera", pa.int64()),
                                                   ("framing_id", pa.string()), ("group_id", pa.string()),
                                                   ("class", pa.int64()), ("duration", pa.float64()),
                                                   ("sha256", pa.string()), ("split", pa.string())])),
                   HERE / "twin_inventory.parquet")
    nan = float("nan")
    trk = {"split": ["train"] * 5, "t": [0.0, 0.0833, 0.1667, 0.25, 0.25], "track_id": [1, 1, 1, 1, 2],
           "class": ["person", "person", "person", "person", "forklift"], "u": [812.0, 815.5, 819.0, 822.1, 1500.0],
           "v": [640.0, 641.0, 642.5, 643.0, 300.0], "x": [5.92, 5.95, 5.97, 6.0, nan],
           "y": [4.07, 4.06, 4.05, 4.04, nan], "conf": [0.81, 0.79, 0.83, 0.8, 0.55],
           "framing": ["cam9_f0"] * 5}
    pq.write_table(pa.table(trk, schema=pa.schema([("split", pa.string()), ("t", pa.float64()), ("track_id", pa.int64()),
                                                   ("class", pa.string()), ("u", pa.float64()), ("v", pa.float64()),
                                                   ("x", pa.float64()), ("y", pa.float64()), ("conf", pa.float64()),
                                                   ("framing", pa.string())])),
                   HERE / "twin_tracks.parquet")
    mac = {"split": ["train"] * 4, "clip_id": [C9] * 4, "press": ["P9a"] * 4, "framing": ["cam9_f0"] * 4,
           "t": [0.0, 0.0833, 0.1667, 0.25], "flow_v": [0.1, 14.2, nan, -9.8], "roi_energy": [0.02, 0.9, nan, 0.7],
           "stroke": [False, True, False, False], "missing": [False, False, True, False]}
    pq.write_table(pa.table(mac, schema=pa.schema([("split", pa.string()), ("clip_id", pa.string()), ("press", pa.string()),
                                                   ("framing", pa.string()), ("t", pa.float64()), ("flow_v", pa.float64()),
                                                   ("roi_energy", pa.float64()), ("stroke", pa.bool_()),
                                                   ("missing", pa.bool_())])),
                   HERE / "twin_machines.parquet")
    log = {"run_id": ["w0_d9_z9_learned"] * 5, "t": [0.0, 31.0, 39.5, 75.0, 720.0],
           "kind": ["mission_start", "checkpoint_attempt", "checkpoint_pass", "press_hold", "mission_end"],
           "entity": ["robot", "robot", "robot", "P9a", "robot"], "checkpoint": [None, "cp1", "cp1", None, None],
           "zone": ["z9"] * 5, "value": [None, None, None, 12.5, None],
           "detail": [None, None, '{"capture_px_per_mm": 2.4}', None, None]}
    pq.write_table(pa.table(log, schema=pa.schema([("run_id", pa.string()), ("t", pa.float64()), ("kind", pa.string()),
                                                   ("entity", pa.string()), ("checkpoint", pa.string()),
                                                   ("zone", pa.string()), ("value", pa.float64()),
                                                   ("detail", pa.string())])),
                   HERE / "twin_log.parquet")


def labels() -> None:
    from twin.schemas.labels import SampleItem, sample_items_sha256

    raw = [
        {"sample_id": "f0001", "kind": "frame", "clip_id": "c9_dev_0003", "group_id": "g02", "camera": 9, "split": "dev",
         "t_pts": 2.0},
        {"sample_id": "f0002", "kind": "frame", "clip_id": "c9_dev_0003", "group_id": "g02", "camera": 9, "split": "dev",
         "t_pts": 4.0},
        {"sample_id": "w0001", "kind": "stroke_window", "clip_id": "c9_dev_0003", "group_id": "g02", "camera": 9,
         "split": "dev", "press": "P9a", "t0": 0.4, "t1": 4.8},
    ]
    items = [SampleItem.model_validate(r) for r in raw]
    sha = sample_items_sha256(items)
    dump_json("twin_label_samples.json", {
        "contract": "twin.label_samples/1", "split": "dev", "seed": 20261017,
        "rule": "frames: seeded RNG over (group, clip), 1 frame every 2 s, stratified by camera; stroke windows: "
                "seeded-random start per press clip, whole clip up to 10 s",
        "items": [i.model_dump(mode="json") for i in items], "items_sha256": sha})
    dump_json("twin_labels.json", {
        "contract": "twin.labels/1", "split": "dev", "labeller_id": "agent:dev-labeller-a",
        "labeller_kind": "claude_agent", "blind": True, "proposals_shown": False, "rules_sha256": SHA,
        "sample_list_sha256": sha,
        "frames": [
            {"sample_id": "f0001", "clip_id": "c9_dev_0003", "t_pts": 2.0, "boxes": [
                {"cls": "person", "x0": 800.0, "y0": 400.0, "x1": 860.0, "y1": 600.0, "foot_u": 830.0, "foot_v": 598.0},
                {"cls": "forklift", "x0": 1300.0, "y0": 200.0, "x1": 1600.0, "y1": 420.0, "occluded": True}]},
            {"sample_id": "f0002", "clip_id": "c9_dev_0003", "t_pts": 4.0, "boxes": [], "note": "empty frame"}],
        "stroke_windows": [{"sample_id": "w0001", "clip_id": "c9_dev_0003", "press": "P9a", "t0": 0.4, "t1": 4.8,
                            "strokes": [0.9167, 2.9167, 4.75]}]})


def sealed() -> None:
    from twin.schemas.prereg import make_sealed, render_sealed

    ids = ["0_fx1", "3_fx2", "3_fx3"]
    man = make_sealed("MANIFEST", [(f"{k:02x}" * 32, f"clips/{c}.mp4") for k, c in enumerate(ids)], ids)
    (HERE / "twin_MANIFEST.sha256").write_text(render_sealed(man), encoding="utf-8", newline="\n")
    lab = make_sealed("LABELS", [("cd" * 32, "test.labels.json"), ("ef" * 32, "test.samples.json")], ids,
                      sample_list_sha256="12" * 32)
    (HERE / "twin_LABELS.sha256").write_text(render_sealed(lab), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    for fn in (site, tasks, safety, dynamics, day_script, realised, summary, primary_metric, parquet, labels, sealed):
        fn()
    print("fixtures written to", HERE)
