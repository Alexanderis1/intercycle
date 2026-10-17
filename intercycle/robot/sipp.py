"""Safe Interval Path Planning (Phillips & Likhachev, ICRA 2011) on a topological graph.

Each node has a list of safe intervals (None = always safe). The search state is (node, interval);
the robot may wait at a node while it is safe. The goal is reached when the robot is at the goal
node inside a safe interval that still has `need_s` seconds left.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Callable

Interval = tuple[float, float]
INF = math.inf


@dataclass
class Plan:
    nodes: list[str]
    arrive: list[float]          # arrival time at each node
    depart: list[float]          # departure time from each node (last = entry time at goal)
    expanded: int

    @property
    def goal_time(self) -> float:
        return self.depart[-1]


def _shortest_times(adj: dict[str, dict[str, float]], goal: str) -> dict[str, float]:
    dist = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def plan(adj: dict[str, dict[str, float]], start: str, t0: float, goal: str,
         safe: Callable[[str], list[Interval] | None], need_s: float = 0.0,
         horizon_s: float = 1800.0, max_expansions: int = 50_000) -> Plan | None:
    h = _shortest_times(adj, goal)
    if start not in h:
        return None

    def intervals(n: str) -> list[Interval]:
        iv = safe(n)
        return [(-INF, INF)] if iv is None else iv

    def usable(n: str, iv: Interval, t: float) -> bool:
        return n != goal or iv[1] - t >= need_s

    s_iv = [i for i, iv in enumerate(intervals(start)) if iv[0] <= t0 < iv[1]]
    if not s_iv:  # start is unsafe right now: wait in place is not allowed, but plan from t0 anyway
        s_iv = [0]
    tie = itertools.count()
    best: dict[tuple[str, int], float] = {}
    parent: dict[tuple[str, int], tuple[tuple[str, int] | None, float, float]] = {}
    pq = []
    k0 = (start, s_iv[0])
    best[k0] = t0
    parent[k0] = (None, t0, t0)
    heapq.heappush(pq, (t0 + h[start], next(tie), k0))
    expanded = 0
    while pq and expanded < max_expansions:
        _, _, key = heapq.heappop(pq)
        node, ii = key
        t = best[key]
        expanded += 1
        iv = intervals(node)[ii]
        if node == goal and usable(node, iv, t):
            return _unwind(parent, key, expanded)
        if t - t0 > horizon_s:
            continue
        for nb, dur in adj[node].items():
            if nb not in h:
                continue
            for jj, (a, b) in enumerate(intervals(nb)):
                arr = max(t + dur, a)
                if arr >= b or arr - dur > iv[1]:   # must still be safe at `node` when leaving
                    continue
                if not usable(nb, (a, b), arr) and nb == goal:
                    continue
                k = (nb, jj)
                if arr < best.get(k, INF):
                    best[k] = arr
                    parent[k] = (key, arr - dur, arr)
                    heapq.heappush(pq, (arr + h[nb], next(tie), k))
    return None


def _unwind(parent, key, expanded) -> Plan:
    nodes, arrive, depart = [], [], []
    cur = key
    nxt_depart = None
    while cur is not None:
        prev, dep_prev, arr = parent[cur]
        nodes.append(cur[0])
        arrive.append(arr)
        depart.append(nxt_depart if nxt_depart is not None else arr)
        nxt_depart = dep_prev
        cur = prev
    nodes.reverse()
    arrive.reverse()
    depart.reverse()
    return Plan(nodes, arrive, depart, expanded)
