"""Twin site graph on the robot side (PLAN_TWIN §4 S1b item 3; R-W10): one press-shop zone's topological graph built
from tasks.yaml, with the `intercycle.robot.sitegraph.SiteGraph` interface (scn, pos, adj, targets, inspect_pose,
mission) plus `capture_psi`.

The robot side never imports twin.* (tests/test_no_truth_import.py: intercycle/robot is a forbidden-prefix module), so
this module reads the contract files as plain YAML and repeats the graph rules of twin/compile.py's TwinSiteGraph
(tests/test_twin_sitegraph.py pins node-for-node, edge-for-edge parity on the v0 contracts and the golden fixtures):

* nodes: DOCK (the zone's dock), W_<checkpoint> for a checkpoint whose capture pose is inside the press's danger rect
  (a "machine" target: its wait pose is GENERATED, not hand-made, by stepping from the capture pose along the target
  normal, in any direction, out of the danger rect plus WAIT_OFFSET_M), P_<checkpoint> for any other checkpoint (a
  "point" target at its capture pose), and C<k>_<corner> visibility corners of the inflated footprints and danger
  rects;
* edges: every node pair whose segment keeps ROBOT_R clear of the visible-floor boundary, every press footprint and
  every danger rect; weights are seconds at the scenario's robot speed;
* the mission is the zone's fixed `mission_order` from tasks.yaml; every target must be reachable from the dock.
Fail closed: a pose on unseen floor, too close to a footprint / danger rect, a capture pose of an inside_danger
checkpoint outside its danger rect, or an unreachable target raises TwinGraphError.

Loading in a run (`load_twin_graph`): the compiled zone's world dir (twin/build_worlds.py) holds scenario.yaml and
inputs.json; inputs.json names the site.yaml and tasks.yaml the zone was compiled from (repo-relative) with their
sha256 (CRLF normalised). The loader re-reads exactly those files and refuses a hash mismatch, so the robot's graph is
the one compile validated. `site_graph(scn, scenario_path)` is the one place a SHARED module gets its graph: the twin
graph when the scenario path has the twin inputs, the body-shop SiteGraph for the body-shop scenario, and an error
for a twin-compiled scenario without them (never the body-shop graph in twin mode).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from intercycle.robot.sitegraph import Target
from intercycle.scenario import Scenario

REPO = Path(__file__).resolve().parents[2]
ROBOT_R = 0.55                  # = twin.compile.ROBOT_R (clearance radius of the robot body)
WAIT_OFFSET_M = 1.05            # = twin.compile.WAIT_OFFSET_M (= timing.TimingCore.wait_offset_m default)
CORNER_PAD_M = 0.05             # = twin.compile.CORNER_PAD_M
INPUTS_FILE = "inputs.json"
_POSE_RANK = {"confirmed": 0, "generated": 1, "prior": 2}
_EPS = 1e-9

AABB = tuple[float, float, float, float]          # (x0, z0, x1, z1)
Point = tuple[float, float]


class TwinGraphError(ValueError):
    """The twin site graph cannot be built (inconsistent contracts, blocked or unreachable poses, bad inputs)."""


# ------------------------------------------------------------------------------------------------ geometry


def point_in_polygon(x: float, z: float, poly: Sequence[Sequence[float]]) -> bool:
    """Even-odd ray casting (= twin.schemas.site.point_in_polygon)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if (z0 > z) != (z1 > z):
            xc = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if x < xc:
                inside = not inside
    return inside


def _seg_seg_dist(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between segments pq and ab (0 if they intersect)."""
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    r, s = q - p, b - a
    den = cross(r, s)
    if abs(den) > _EPS:
        t, u = cross(a - p, s) / den, cross(a - p, r) / den
        if -_EPS <= t <= 1 + _EPS and -_EPS <= u <= 1 + _EPS:
            return 0.0

    def pt_seg(x: np.ndarray, a_: np.ndarray, b_: np.ndarray) -> float:
        d = b_ - a_
        L2 = float(d @ d)
        w = 0.0 if L2 < _EPS else min(1.0, max(0.0, float((x - a_) @ d) / L2))
        return float(np.hypot(*(x - (a_ + w * d))))

    return min(pt_seg(p, a, b), pt_seg(q, a, b), pt_seg(a, p, q), pt_seg(b, p, q))


def _seg_rect_dist(p: np.ndarray, q: np.ndarray, r: AABB) -> float:
    x0, z0, x1, z1 = r
    for v in (p, q):
        if x0 <= v[0] <= x1 and z0 <= v[1] <= z1:
            return 0.0
    c = [np.array(v, float) for v in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    return min(_seg_seg_dist(p, q, c[k], c[(k + 1) % 4]) for k in range(4))


def _in_rect(x: float, z: float, r: AABB) -> bool:
    return r[0] < x < r[2] and r[1] < z < r[3]


def aabb(poly: Sequence[Sequence[float]]) -> AABB:
    xs, zs = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
    return min(xs), min(zs), max(xs), max(zs)


def ray_exit(x: float, z: float, ux: float, uz: float, r: AABB) -> float:
    """Distance from (x, z) inside rect r along the unit direction (ux, uz) to the rect boundary."""
    tx = ((r[2] - x) / ux if ux > 0 else (r[0] - x) / ux) if abs(ux) > _EPS else math.inf
    tz = ((r[3] - z) / uz if uz > 0 else (r[1] - z) / uz) if abs(uz) > _EPS else math.inf
    return min(tx, tz)


def plan_yaw_to_psi(yaw: float) -> float:
    """Plan yaw (atan2(dx, dz): 0 faces +z, pi/2 faces +x) -> ROS heading psi (atan2(dy, dx), y = plan z)."""
    return math.atan2(math.cos(yaw), math.sin(yaw))


# ------------------------------------------------------------------------------------------------ graph


class TwinSiteGraph:
    """Topological site graph of one twin zone, built from tasks.yaml (a parsed twin.tasks/1 mapping)."""

    def __init__(self, scn: Scenario, tasks: Mapping[str, Any], zone: str, floor: Sequence[Sequence[float]],
                 footprints: Sequence[AABB]):
        self.scn = scn
        self.zone = zone
        self.pos: dict[str, tuple[float, float]] = {}
        self.adj: dict[str, dict[str, float]] = {}
        self.targets: list[Target] = []
        self.inspect: dict[str, tuple[float, float]] = {}    # target id -> capture pose (x, z)
        self.yaw: dict[str, float] = {}                      # target id -> capture yaw (plan yaw)
        self._floor_poly = [(float(p[0]), float(p[1])) for p in floor]
        if len(self._floor_poly) < 3:
            raise TwinGraphError(f"zone {zone}: the visible-floor polygon needs >= 3 points")
        self._floor = [np.array(p, float) for p in self._floor_poly]
        self._footprints: list[AABB] = [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in footprints]
        self._danger = [m.rect for m in scn.machines]
        if tasks.get("contract") != "twin.tasks/1":
            raise TwinGraphError(f"tasks contract {tasks.get('contract')!r} is not twin.tasks/1")
        zt = next((z for z in tasks.get("zones") or [] if z.get("zone") == zone), None)
        if zt is None:
            raise TwinGraphError(f"tasks.yaml has no entry for zone {zone}")
        self._order = [str(c) for c in zt["mission_order"]]
        cps = {str(c["id"]): c for c in tasks.get("checkpoints") or [] if c.get("zone") == zone}
        if len(set(self._order)) != len(self._order) or set(self._order) != set(cps):
            raise TwinGraphError(f"zone {zone}: mission_order {self._order} != its checkpoints {sorted(cps)}")
        self._node_required("DOCK", float(zt["dock"]["x"]), float(zt["dock"]["z"]))
        for cid in self._order:
            self._add_target(cps[cid])
        self._add_corners()
        self._connect()
        self._check_reachable()

    # ---- clearance
    def _clear_pt(self, x: float, z: float, hazards: bool = True) -> str | None:
        p = np.array((x, z), float)
        if not point_in_polygon(x, z, self._floor_poly):
            return "outside the visible floor"
        nf = len(self._floor)
        if min(_seg_seg_dist(p, p, self._floor[k], self._floor[(k + 1) % nf]) for k in range(nf)) < ROBOT_R - _EPS:
            return f"closer than {ROBOT_R} m to unseen floor"
        for r in self._footprints:
            if _seg_rect_dist(p, p, r) < ROBOT_R - _EPS:
                return f"closer than {ROBOT_R} m to a press footprint {r}"
        if hazards:
            for r in self._danger:
                if _seg_rect_dist(p, p, r) < ROBOT_R - _EPS:
                    return f"closer than {ROBOT_R} m to a danger rect {r}"
        return None

    def _clear_seg(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        p, q = np.array(a, float), np.array(b, float)
        nf = len(self._floor)
        if any(_seg_seg_dist(p, q, self._floor[k], self._floor[(k + 1) % nf]) < ROBOT_R - _EPS for k in range(nf)):
            return False
        return all(_seg_rect_dist(p, q, r) >= ROBOT_R - _EPS for r in self._footprints + self._danger)

    def _node(self, n: str, x: float, z: float) -> None:
        self.pos[n] = (float(x), float(z))
        self.adj.setdefault(n, {})

    def _node_required(self, n: str, x: float, z: float, hazards: bool = True) -> None:
        why = self._clear_pt(x, z, hazards)
        if why:
            raise TwinGraphError(f"zone {self.zone}: node {n} at ({x:.3f}, {z:.3f}) is {why}")
        self._node(n, x, z)

    # ---- targets
    def _add_target(self, cp: Mapping[str, Any]) -> None:
        cid = str(cp["id"])
        poses = cp.get("candidate_poses") or []
        if not poses:
            raise TwinGraphError(f"checkpoint {cid}: no candidate pose")
        cand = sorted(poses, key=lambda c: _POSE_RANK[c["source"]])[0]
        px, pz, yaw = float(cand["pose"]["x"]), float(cand["pose"]["z"]), float(cand["pose"]["yaw"])
        self.inspect[cid], self.yaw[cid] = (px, pz), yaw
        dwell = float(cp["dwell_s"])
        if cp.get("inside_danger"):
            press = cp.get("press")
            ms = [m for m in self.scn.machines if m.id == press] if press else []
            if not ms:
                raise TwinGraphError(f"checkpoint {cid}: inside_danger needs a press of zone {self.zone}")
            m = ms[0]
            r = m.rect
            if not _in_rect(px, pz, r):
                raise TwinGraphError(f"checkpoint {cid}: capture pose ({px}, {pz}) is not inside the danger rect {r}")
            why = self._clear_pt(px, pz, hazards=False)
            if why:
                raise TwinGraphError(f"checkpoint {cid}: capture pose is {why}")
            nx_, nz_ = (float(v) for v in cp["normal_xz"])
            L = math.hypot(nx_, nz_)
            if L < _EPS:
                raise TwinGraphError(f"checkpoint {cid}: zero target normal")
            ux, uz = nx_ / L, nz_ / L
            t = ray_exit(px, pz, ux, uz, r) + WAIT_OFFSET_M
            node = f"W_{cid}"
            self._node_required(node, px + ux * t, pz + uz * t)
            if any(_seg_rect_dist(np.array((px, pz)), np.array(self.pos[node]), f) < ROBOT_R - _EPS
                   for f in self._footprints):
                raise TwinGraphError(f"checkpoint {cid}: the entry line from {node} passes a press footprint")
            self.targets.append(Target(cid, "machine", node, m.id, dwell))
        else:
            node = f"P_{cid}"
            self._node_required(node, px, pz)
            self.targets.append(Target(cid, "point", node, None, dwell))

    def _add_corners(self) -> None:
        d = ROBOT_R + CORNER_PAD_M
        for k, (x0, z0, x1, z1) in enumerate(self._footprints + self._danger):
            for c, (x, z) in enumerate(((x0 - d, z0 - d), (x1 + d, z0 - d), (x1 + d, z1 + d), (x0 - d, z1 + d))):
                if self._clear_pt(x, z) is None:
                    self._node(f"C{k}_{c}", x, z)

    def _connect(self) -> None:
        v = self.scn.inspection.robot_speed_mps
        names = list(self.pos)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if self._clear_seg(self.pos[a], self.pos[b]):
                    d = math.hypot(self.pos[a][0] - self.pos[b][0], self.pos[a][1] - self.pos[b][1]) / v
                    self.adj[a][b] = d
                    self.adj[b][a] = d

    def _check_reachable(self) -> None:
        seen, todo = {"DOCK"}, ["DOCK"]
        while todo:
            for b in self.adj[todo.pop()]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        lost = [t.id for t in self.targets if t.node not in seen]
        if lost:
            raise TwinGraphError(f"zone {self.zone}: targets not reachable from the dock: {lost}")

    # ---- SiteGraph interface
    def inspect_pose(self, target: Target) -> tuple[float, float]:
        """Robot centre while capturing: the chosen candidate pose (inside the danger rect for machine targets)."""
        return self.inspect[target.id]

    def mission(self) -> list[Target]:
        """One inspection round in the zone's fixed mission order (tasks.yaml)."""
        by_id = {t.id: t for t in self.targets}
        return [by_id[i] for i in self._order]

    def capture_psi(self, target: Target) -> float:
        """ROS heading of the chosen candidate pose (tasks.yaml plan yaw converted)."""
        return plan_yaw_to_psi(self.yaw[target.id])


# ------------------------------------------------------------------------------------------------ loading


def text_sha256(p: Path) -> str:
    """sha256 of a text file with CRLF normalised to LF (= twin.build_worlds.text_sha256)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _input_file(rec: Mapping[str, Any], kind: str, repo: Path) -> Path:
    ent = rec.get(kind)
    if not isinstance(ent, Mapping) or not isinstance(ent.get("path"), str) or not isinstance(ent.get("sha256"), str):
        raise TwinGraphError(f"{INPUTS_FILE}: no {kind} entry with path and sha256")
    p = Path(ent["path"])
    p = p if p.is_absolute() else repo / p
    if not p.is_file():
        raise TwinGraphError(f"{INPUTS_FILE}: {kind} file {p} does not exist")
    got = text_sha256(p)
    if got != ent["sha256"]:
        raise TwinGraphError(f"{INPUTS_FILE}: {kind} file {p} has sha256 {got}, the zone was compiled from "
                             f"{ent['sha256']} (re-run python -m twin.build_worlds)")
    return p


def load_twin_graph(scenario_path: str | Path, scn: Scenario | None = None, repo: str | Path | None = None
                    ) -> TwinSiteGraph:
    """The TwinSiteGraph of a compiled twin zone from its world dir (scenario.yaml + inputs.json, see module doc)."""
    from intercycle import scenario as S

    sp = Path(scenario_path)
    scn = S.load(sp) if scn is None else scn
    inputs = sp.parent / INPUTS_FILE
    if not inputs.is_file():
        raise TwinGraphError(f"{sp}: no {INPUTS_FILE} next to the twin scenario (twin/build_worlds.py writes it)")
    rec = json.loads(inputs.read_text(encoding="utf-8"))
    root = REPO if repo is None else Path(repo)
    site = yaml.safe_load(_input_file(rec, "site", root).read_text(encoding="utf-8"))
    tasks = yaml.safe_load(_input_file(rec, "tasks", root).read_text(encoding="utf-8"))
    if len(scn.zones) != 1:
        raise TwinGraphError(f"{sp}: a compiled twin zone has exactly one zone, got {[z.id for z in scn.zones]}")
    zone = scn.zones[0].id
    if not isinstance(site, Mapping) or site.get("contract") != "twin.site/1":
        raise TwinGraphError("inputs.json site file is not a twin.site/1 contract")
    if not isinstance(tasks, Mapping) or tasks.get("site_id") != site.get("site_id"):
        raise TwinGraphError(f"tasks site_id {tasks.get('site_id') if isinstance(tasks, Mapping) else None!r} != "
                             f"site site_id {site.get('site_id')!r}")
    sz = next((z for z in site.get("zones") or [] if z.get("id") == zone), None)
    if sz is None:
        raise TwinGraphError(f"site has no zone {zone}")
    presses = [p for p in site.get("presses") or [] if p.get("zone") == zone]
    if sorted(p["id"] for p in presses) != sorted(m.id for m in scn.machines):
        raise TwinGraphError(f"zone {zone}: scenario machines {sorted(m.id for m in scn.machines)} != site presses "
                             f"{sorted(p['id'] for p in presses)}")
    return TwinSiteGraph(scn, tasks, zone, sz["floor_polygon"]["value"],
                         [aabb(p["footprint"]["value"]) for p in presses])


def is_twin_scenario(scn: Scenario) -> bool:
    """A twin-compiled zone (twin/compile.py): every machine is an off-line press and there is no body-shop hall annex."""
    return not scn.data_hall and bool(scn.machines) and all(m.kind == "press" and not m.on_line for m in scn.machines)


def site_graph(scn: Scenario, scenario_path: str | Path | None = None):
    """The site graph a run's planner uses (see module doc). Twin mode never falls back to the body-shop graph."""
    if scenario_path and (Path(scenario_path).parent / INPUTS_FILE).is_file():
        return load_twin_graph(scenario_path, scn)
    if is_twin_scenario(scn):
        raise TwinGraphError("a twin-compiled scenario needs its scenario path with inputs.json (tasks.yaml) to build "
                             "the twin site graph; refusing the body-shop graph")
    from intercycle.robot.sitegraph import SiteGraph

    return SiteGraph(scn)
