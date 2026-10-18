"""Twin broadcaster (episode side, NOT the robot): tails Observation v2 through twin/observer_live.py and broadcasts the
signed slow / fast / map layers to the robot's observer_bridge (PLAN_TWIN §2: "The broadcaster and observer_live read
only obs_v2.jsonl and never open realised_state or plant_state").

    $IC_SIM_PY -m twin.broadcast_live --work $IC_WORK --obs $IC_OUT/obs_v2.jsonl --world $IC_WORLD
        (run with PYTHONPATH=$IC_SIM_PYTHONPATH: numpy, pydantic, pynacl; no ROS)

Inputs (all non-truth): the mission's `obs_v2.jsonl` (written by the plant's camera model while the mission runs) and,
from the world dir, `observer_spec.json` (the site-wide twin_spec plus the mission zone's machine ids in the zone
scenario's order), `m7_margin.json` (the frozen M-W7 margin) and `learn_obs.jsonl` (the Observation v2 lines of the
learning period, copied to <work>/learn/obs_v2.jsonl so ObsV2Reader reads them under its only allowed name). It never
reads run_cfg.json (the plant's cfg) or any truth file.

"learned" condition: the observer first ingests the learning period (it learned before the mission), the map layer is
packed ONCE from that state at mission start (`ObserverLive.map_snapshot`, M-W7-inflated), and the slow / fast layers
are live (`ObserverLive.slow/fast` at each tick's delivery time). The observer's clock is the stream: "now" = the
delivery time `t` of the newest ingested observation, never the plant's clock.

The observer is site-wide (Observation v2 carries every press and camera); the robot's site map is the mission zone's
scenario, so the slow / fast entries are re-indexed to the zone's machine order (`ZoneView`). The map keeps the
site-wide machine grids: the robot looks machines up by id.

Keys (C-W5): a fresh Ed25519 key per episode; only its PUBLIC verify key leaves the process, as <work>/observer_vk.txt
("<key_id_of(vk)>:<hex>"), which the bridge's keyring parameter reads. Headers carry key_id_of(key).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

from nacl.signing import SigningKey

from intercycle.codec.trust import key_id_of
from intercycle.types import DAY0_UNIX, FastLayer, SlowLayer
from intercycle_nav.broadcast import Broadcaster, pack_map_for_key  # pyright: ignore[reportMissingImports]  # ROS-free (tests/test_isaac_deps.py ISAAC_SAFE_ROS)
from intercycle_nav.feed import FeedWriter  # pyright: ignore[reportMissingImports]
from twin import observer_live as OL

SPEC_FILE = "observer_spec.json"
MARGIN_FILE = "m7_margin.json"
LEARN_FILE = "learn_obs.jsonl"
VK_FILE = "observer_vk.txt"
SPEC_CONTRACT = "twin.observer_spec/1"


def episode_key() -> tuple[SigningKey, str]:
    """(fresh signing key, its keyring line "key_id:hex" with key_id = key_id_of(verify key))."""
    key = SigningKey.generate()
    return key, f"{key_id_of(key.verify_key)}:{key.verify_key.encode().hex()}"


def spec_record(spec: dict, zone: str, zone_machines: list[str], camera_model: str) -> dict:
    """observer_spec.json: the site-wide twin_spec, the mission zone's machine ids (zone scenario order) and the camera
    model that writes the Observation (the plant's `camera.provenance()["model"]`; it picks the observer config,
    `observer_live.config_for_camera`, fail closed)."""
    ids = [m["id"] for m in spec["machines"]]
    bad = [m for m in zone_machines if m not in ids]
    if bad or not zone_machines:
        raise ValueError(f"zone {zone} machines {zone_machines} are not all in the observer spec {ids}")
    OL.config_for_camera(camera_model, "")             # refuse a camera model without a twin-run config now
    return {"contract": SPEC_CONTRACT, "zone": zone, "zone_machines": list(zone_machines), "camera_model": camera_model,
            "spec": spec}


class ZoneView:
    """Re-index the site-wide observer's slow / fast layers to the mission zone's machine order."""

    def __init__(self, ob: OL.ObserverLive, zone_machines: list[str]):
        self.ob = ob
        self.idx = [ob.ids.index(m) for m in zone_machines]

    def slow(self, t: float) -> SlowLayer:
        s = self.ob.slow(t)
        return replace(s, machines=[replace(s.machines[i], index=k) for k, i in enumerate(self.idx)])

    def fast(self, t: float) -> FastLayer:
        f = self.ob.fast(t)
        return replace(f, machines=[replace(f.machines[i], index=k) for k, i in enumerate(self.idx)])


def load_inputs(world: Path, work: Path) -> tuple[dict, list[str], str, Path, str]:
    """(spec, zone machines, margin path, learn obs path under work/learn/obs_v2.jsonl, camera model). A spec record
    without a camera model is refused (fail closed)."""
    rec = json.loads((world / SPEC_FILE).read_text(encoding="utf-8"))
    if rec.get("contract") != SPEC_CONTRACT:
        raise ValueError(f"{SPEC_FILE}: contract {rec.get('contract')!r} != {SPEC_CONTRACT!r}")
    if not rec.get("camera_model"):
        raise ValueError(f"{SPEC_FILE}: no camera_model (the observer config depends on it)")
    learn = work / "learn" / OL.OBS_FILE
    learn.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(world / LEARN_FILE, learn)
    return rec["spec"], list(rec["zone_machines"]), str(world / MARGIN_FILE), learn, str(rec["camera_model"])


def learned_observer(spec: dict, margin_path: str, learn_path: Path,
                     camera_model: str) -> tuple[OL.ObserverLive, float, int]:
    """An ObserverLive (config by camera model, `OL.config_for_camera`) that has ingested the learning period.
    Returns (observer, newest delivery time, ticks)."""
    ob = OL.ObserverLive(spec, OL.config_for_camera(camera_model, margin_path))
    rd = OL.ObsV2Reader(learn_path, ob.M, len(spec["cameras"]))
    t_now, n = -math.inf, 0
    for o in rd.poll():
        ob.ingest(o)
        t_now = max(t_now, float(o.t))
        n += 1
    if n == 0:
        raise ValueError(f"{learn_path}: no learning-period observations (the learned condition needs them)")
    return ob, t_now, n


def write_vk(work: Path, vk: str) -> None:
    tmp = work / (VK_FILE + ".tmp")
    tmp.write_text(vk, encoding="utf-8")
    os.replace(tmp, work / VK_FILE)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="twin broadcaster: obs_v2.jsonl -> observer_live -> signed layers")
    ap.add_argument("--work", required=True)
    ap.add_argument("--obs", required=True, help="the mission's obs_v2.jsonl")
    ap.add_argument("--world", required=True)
    ap.add_argument("--log", default="", help="JSONL log (default <obs dir>/broadcast_live.jsonl)")
    ap.add_argument("--fast-every", type=float, default=0.5)
    ap.add_argument("--slow-every", type=float, default=10.0)
    a = ap.parse_args(argv)
    work, world, obs_path = Path(a.work), Path(a.world), Path(a.obs)
    log = open(a.log or obs_path.parent / "broadcast_live.jsonl", "a", encoding="utf-8")
    key, vk = episode_key()
    write_vk(work, vk)
    spec, zone_machines, margin_path, learn_path, camera_model = load_inputs(world, work)
    ob, t_now, n_learn = learned_observer(spec, margin_path, learn_path, camera_model)
    bundle = pack_map_for_key(ob.map_snapshot(), key, version=1, issue_time=DAY0_UNIX + t_now)
    zv = ZoneView(ob, zone_machines)
    diag = ob.diagnostics()
    log.write(json.dumps({"start": {"t_wall": time.time(), "verify_key": vk, "n_learn": n_learn, "t_learn_end": t_now,
                                    "zone_machines": zone_machines, "map_bytes": bundle.stats.get("compressed_bytes"),
                                    "map_frames": len(bundle.carousel()), "models": diag["models"],
                                    "availability": diag["availability"]}}, default=str) + "\n")
    log.flush()
    w = FeedWriter(f"unix://{work / 'observer.sock'}")
    bc = Broadcaster(w, key, bundle, zv.slow, zv.fast, fast_every_s=a.fast_every, slow_every_s=a.slow_every)
    rd = OL.ObsV2Reader(obs_path, ob.M, len(spec["cameras"]))
    stop = work / "STOP"
    n_ticks, last = 0, -math.inf
    t_wall0 = time.time()
    while not stop.exists():
        for o in rd.poll():
            if float(o.t_cap) <= float(ob.t_last if ob.t_last is not None else -math.inf):
                continue                                  # already learned (overlap with the learning period)
            ob.ingest(o)
            t_now = max(t_now, float(o.t))
        if t_now > last:
            bc.tick(t_now)
            last = t_now
            n_ticks += 1
            if n_ticks % 200 == 0:
                log.write(json.dumps({"t": t_now, "ticks": n_ticks, "sent": w.sent, "dropped": w.dropped,
                                      "wall_s": round(time.time() - t_wall0, 1)}) + "\n")
                log.flush()
        time.sleep(0.02)
    log.write(json.dumps({"end": {"t": last, "ticks": n_ticks, "sent": w.sent, "dropped": w.dropped,
                                  "models": ob.diagnostics()["models"]}}, default=str) + "\n")
    log.close()
    w.close()


if __name__ == "__main__":
    main()
