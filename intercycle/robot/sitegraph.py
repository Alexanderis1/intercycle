"""Topological site graph the visiting robot is provisioned with (public site map, no machine data)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from intercycle.scenario import Scenario

AISLE_N, AISLE_S = 32.0, 16.0
CROSS_X = 52.5                      # walkway across the line between S50 and S60
SOUTH_ACCESS = {"S20", "S40", "S70"}
TRANSFER_S = 8.0                    # skids move for this long after every line index
WAIT_OFFSET = 0.7                   # wait point distance outside the hazard boundary (robot body fully outside)
INSPECT_DEPTH = 1.0                 # robot centre this far inside the boundary while inspecting


@dataclass(frozen=True)
class Target:
    id: str
    kind: str                       # "machine" | "rack"
    node: str                       # wait node at the hazard boundary (or the rack stop)
    machine: str | None
    dwell_s: float


class SiteGraph:
    def __init__(self, scn: Scenario):
        self.scn = scn
        self.pos: dict[str, tuple[float, float]] = {}
        self.adj: dict[str, dict[str, float]] = {}
        self.targets: list[Target] = []
        v = scn.inspection.robot_speed_mps
        self._build(v)

    def _node(self, n: str, x: float, z: float) -> None:
        self.pos[n] = (x, z)
        self.adj.setdefault(n, {})

    def _edge(self, a: str, b: str, v: float) -> None:
        (xa, za), (xb, zb) = self.pos[a], self.pos[b]
        d = math.hypot(xa - xb, za - zb) / v
        self.adj[a][b] = d
        self.adj[b][a] = d

    def _chain(self, names: list[str], v: float) -> None:
        names = sorted(set(names), key=lambda n: self.pos[n][0])
        for a, b in zip(names, names[1:]):
            self._edge(a, b, v)

    def _build(self, v: float) -> None:
        scn = self.scn
        xs = {2.0, CROSS_X, 90.0} | {m.x for m in scn.machines}
        for x in xs:
            self._node(f"N{x:g}", x, AISLE_N)
            self._node(f"S{x:g}", x, AISLE_S)
        self._node("DOCK", *scn.robot["dock"])
        self._node("X", CROSS_X, 24.0)                 # line crossing walkway (timed)
        self._node("E", 90.0, 24.0)
        self._node("DOOR", *scn.data_hall["door"])
        self._node("DH", 100.0, 24.0)
        self._chain([n for n in self.pos if n.startswith("N")], v)
        self._chain([n for n in self.pos if n.startswith("S") and n[1:].replace(".", "").isdigit()], v)
        self._edge("DOCK", "N2", v)
        self._edge("N2", "S2", v)                      # around the line start
        self._edge(f"N{CROSS_X:g}", "X", v)
        self._edge("X", f"S{CROSS_X:g}", v)
        self._edge("N90", "E", v)
        self._edge("S90", "E", v)
        self._edge("E", "DOOR", v)
        self._edge("DOOR", "DH", v)
        # machine targets: wait node on the hazard boundary, reached from the access aisle
        for m in scn.machines:
            x0, z0, x1, z1 = m.rect
            if m.id in SOUTH_ACCESS:
                n, aisle = f"W_{m.id}", f"S{m.x:g}"
                self._node(n, m.x, z0 - WAIT_OFFSET)
            elif m.on_line:
                n, aisle = f"W_{m.id}", f"N{m.x:g}"
                self._node(n, m.x, z1 + WAIT_OFFSET)
            else:                                       # closures cells north of the north aisle
                n, aisle = f"W_{m.id}", f"N{m.x:g}"
                self._node(n, m.x, z0 - WAIT_OFFSET)
            self._edge(aisle, n, v)
            self.targets.append(Target(m.id, "machine", n, m.id, scn.inspection.dwell_s))
        # data-hall rack rows (no machine hazard)
        rows = scn.data_hall_targets["rows"]
        for r in range(rows):                       # cold aisles just north of each rack row
            z = 14.4 + r * 8.0
            self._node(f"DA{r}", 99.0, z)
            self._edge("DH", f"DA{r}", v)
            prev = f"DA{r}"
            for k, x in enumerate((103.0, 106.6)):
                n = f"R{r}{k}"
                self._node(n, x, z)
                self._edge(prev, n, v)
                prev = n
                self.targets.append(Target(n, "rack", n, None, 6.0))

    def inspect_pose(self, target: Target) -> tuple[float, float]:
        """Robot centre while inspecting: INSPECT_DEPTH inside the boundary, straight in from the wait node."""
        m = self.scn.machine(target.machine)
        x, z = self.pos[target.node]
        dz = 1.0 if m.z > z else -1.0
        edge = m.rect[1] if dz > 0 else m.rect[3]
        return x, edge + dz * INSPECT_DEPTH

    def mission(self) -> list[Target]:
        """One inspection tour in walking order."""
        order = ["S10", "HEM1", "S20", "S30", "S40", "SEAL1", "S50", "S60", "S70", "S80"]
        by_id = {t.id: t for t in self.targets}
        return [by_id[i] for i in order] + [t for t in self.targets if t.kind == "rack"]
