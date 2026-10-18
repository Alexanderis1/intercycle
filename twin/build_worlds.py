"""Build the per-zone world files of the twin through `intercycle_sim/world.py` (PLAN_TWIN §2; CPU side only).

    python -m twin.build_worlds [--site P] [--dynamics P] [--tasks P] [--out twin]

writes, for every zone of the site, `<out>/<zone>/`:

  scenario.yaml   the zone's compiled Scenario (twin/compile.py); build_world_files reads the dock from it
  boxes.json      every static body as an upright box (press footprints, unseen floor as walls, the border ring)
  static.usda     the same boxes as one static triangle-mesh collider (world.py's factory_l0.usda)
  mesh.npz        the same mesh for the Warp lidar
  map.pgm/.yaml   the Nav2 occupancy map (0.1 m) rasterised from the same rects
  world.json      world.py's metadata (extent, dock, frames)
  inputs.json     the repo-relative paths and sha256 of the site / dynamics / tasks files it was built from

Unseen floor is a wall in both the USD and the map, because both come from the same StaticMap.rects() over the
zone's extent (tests/test_twin_worlds.py checks it). Inputs default to twin.compile.default_inputs() (the v0 stubs
while site.yaml / dynamics.json / tasks.yaml do not exist). The provenance check always uses the active split
(prereg/ACTIVE_SPLIT, else prereg/split.csv) from the command line, and the build fails closed without one; only a
caller of build() (tests) may pass another split.

This module imports scipy (via intercycle.sim.geometry); it never runs in the Isaac container (CPU_ONLY in
tests/test_isaac_deps.py), which receives the written files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from twin.compile import TWIN, CompiledZone, compile_paths, default_inputs, scenario_yaml
from twin.schemas.prereg import SplitRow

_SIM_PKG = Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"
if str(_SIM_PKG) not in sys.path:
    sys.path.insert(0, str(_SIM_PKG))

from intercycle_sim import world as W  # noqa: E402

FILES = ("scenario.yaml", "boxes.json", "static.usda", "mesh.npz", "map.pgm", "map.yaml", "world.json", "inputs.json")
REPO = TWIN.parent


def _input_record(paths: Mapping[str, str | Path]) -> bytes:
    rec = {}
    for k, p in paths.items():
        q = Path(p).resolve()
        try:
            name = q.relative_to(REPO).as_posix()
        except ValueError:
            name = q.as_posix()
        rec[k] = {"path": name, "sha256": text_sha256(q)}
    return (json.dumps(rec, indent=2) + "\n").encode()


def text_sha256(p: Path) -> str:
    """sha256 of a text file with CRLF normalised to LF (the same on a Windows checkout and on Linux)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def zone_files(cz: CompiledZone, scenario_path: str | Path) -> dict[str, bytes]:
    """world.py's files for one compiled zone (scenario_path must hold scenario_yaml(cz.scenario))."""
    files = W.build_world_files(scenario_path=str(scenario_path), rects=cz.rects, extent=cz.extent)
    files["static.usda"] = files.pop("factory_l0.usda")
    return files


def build(site: str | Path, dynamics: str | Path, tasks: str | Path, out: str | Path = TWIN, *,
          split: Mapping[str, SplitRow] | None = None) -> dict[str, Path]:
    """Compile the site and write every zone's files under out/<zone>/. Returns {zone: directory}."""
    zones = compile_paths(site, dynamics, tasks, split=split)
    written: dict[str, Path] = {}
    for zid, cz in zones.items():
        d = Path(out) / zid
        d.mkdir(parents=True, exist_ok=True)
        scn_path = d / "scenario.yaml"
        scn_path.write_text(scenario_yaml(cz.scenario), encoding="utf-8", newline="\n")
        files = zone_files(cz, scn_path)
        files["inputs.json"] = _input_record({"site": site, "dynamics": dynamics, "tasks": tasks})
        W.write_world(files, str(d))
        written[zid] = d
    return written


def main(argv: list[str] | None = None) -> int:
    d = default_inputs()
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--site", default=str(d["site"]))
    ap.add_argument("--dynamics", default=str(d["dynamics"]))
    ap.add_argument("--tasks", default=str(d["tasks"]))
    ap.add_argument("--out", default=str(TWIN))
    a = ap.parse_args(argv)
    for zid, path in build(a.site, a.dynamics, a.tasks, a.out).items():
        print(f"{zid}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
