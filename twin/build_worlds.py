"""Build the per-zone world files of the twin through `intercycle_sim/world.py` (PLAN_TWIN §2; CPU side only).

    python -m twin.build_worlds [--site P] [--dynamics P] [--tasks P] [--out twin] [--allow-stubs]

writes, for every zone of the site, `<out>/<zone>/`:

  scenario.yaml   the zone's compiled Scenario (twin/compile.py); build_world_files reads the dock from it
  boxes.json      every static body as an upright box (press footprints, unseen floor as walls, the border ring)
  static.usda     the same boxes as one static triangle-mesh collider (world.py's factory_l0.usda)
  mesh.npz        the same mesh for the Warp lidar
  map.pgm/.yaml   the Nav2 occupancy map (0.1 m) rasterised from the same rects
  world.json      world.py's metadata (extent, dock, frames)
  cameras.json    per framing: the camera placement compile used and its focal length (fitted from H, or the prior)
  sitegraph.json  the zone's TwinSiteGraph as data (twin.compile.graph_json): the robot-side timing layer builds its
                  SiteGraph from it (intercycle_nav.timing.GraphData; the robot never imports twin)
  inputs.json     the repo-relative paths, sha256 and stub flag of the site / dynamics / tasks files it was built from

Unseen floor is a wall in both the USD and the map, because both come from the same StaticMap.rects() over the
zone's extent (tests/test_twin_worlds.py checks it). Inputs default to twin.compile.default_inputs(): site.yaml /
dynamics.json / tasks.yaml; a missing one, or a v0 stub given explicitly, fails closed unless --allow-stubs is passed
(pre-G2 builds only; the choice is logged and inputs.json flags every stub). The provenance check always uses the
active split (prereg/ACTIVE_SPLIT, else prereg/split.csv) and the build fails closed without one; only tests may
substitute a split (twin.compile `_split_for_tests`).

This module imports scipy (via intercycle.sim.geometry); it never runs in the Isaac container (CPU_ONLY in
tests/test_isaac_deps.py), which receives the written files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from twin.compile import (TWIN, CompileError, CompiledZone, compile_paths, default_inputs, graph_json, is_stub,
                          scenario_yaml)
from twin.schemas.prereg import SplitRow

_SIM_PKG = Path(__file__).resolve().parents[1] / "ros2" / "intercycle_sim"
if str(_SIM_PKG) not in sys.path:
    sys.path.insert(0, str(_SIM_PKG))

from intercycle_sim import world as W  # noqa: E402

log = logging.getLogger(__name__)
FILES = ("scenario.yaml", "boxes.json", "static.usda", "mesh.npz", "map.pgm", "map.yaml", "world.json", "inputs.json",
         "cameras.json", "sitegraph.json")
REPO = TWIN.parent


def _input_record(paths: Mapping[str, str | Path]) -> bytes:
    rec = {}
    for k, p in paths.items():
        q = Path(p).resolve()
        try:
            name = q.relative_to(REPO).as_posix()
        except ValueError:
            name = q.as_posix()
        rec[k] = {"path": name, "sha256": text_sha256(q), "stub": is_stub(k, q)}
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
          _split_for_tests: Mapping[str, SplitRow] | None = None) -> dict[str, Path]:
    """Compile the site and write every zone's files under out/<zone>/. Returns {zone: directory}."""
    zones = compile_paths(site, dynamics, tasks, _split_for_tests=_split_for_tests)
    written: dict[str, Path] = {}
    for zid, cz in zones.items():
        d = Path(out) / zid
        d.mkdir(parents=True, exist_ok=True)
        scn_path = d / "scenario.yaml"
        scn_path.write_text(scenario_yaml(cz.scenario), encoding="utf-8", newline="\n")
        files = zone_files(cz, scn_path)
        files["inputs.json"] = _input_record({"site": site, "dynamics": dynamics, "tasks": tasks})
        files["cameras.json"] = (json.dumps([c.as_json() for c in cz.cameras.values()], indent=2) + "\n").encode()
        files["sitegraph.json"] = (json.dumps(graph_json(cz.graph), indent=1) + "\n").encode()
        W.write_world(files, str(d))
        written[zid] = d
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--site")
    ap.add_argument("--dynamics")
    ap.add_argument("--tasks")
    ap.add_argument("--out", default=str(TWIN))
    ap.add_argument("--allow-stubs", action="store_true",
                    help="a missing site.yaml / dynamics.json / tasks.yaml falls back to its G0 v0 stub (pre-G2 only)")
    a = ap.parse_args(argv)
    given = {"site": a.site, "dynamics": a.dynamics, "tasks": a.tasks}
    paths = {k: Path(v) for k, v in given.items() if v}
    if len(paths) < len(given):
        d = default_inputs(allow_stubs=a.allow_stubs)            # fails closed on a missing file without the flag
        paths = {k: paths.get(k, d[k]) for k in given}
    for k, p in paths.items():
        if is_stub(k, p):
            if not a.allow_stubs:
                raise CompileError(f"{k}: {p} is the G0 v0 stub; pass --allow-stubs to build on it (pre-G2 only)")
            log.warning("building on the G0 v0 STUB %s for %s (recorded as stub in inputs.json)", p, k)
    for zid, path in build(paths["site"], paths["dynamics"], paths["tasks"], a.out).items():
        print(f"{zid}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
