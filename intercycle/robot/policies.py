"""Entry decisions at a hazard boundary for the four robots. No ground truth in here.

Every policy sees the world only through callables:
  sees_motion(t)   -> bool   robot's own camera, already delayed by its latency
  knowledge(t)     -> (MachineModel | None, MachineNow | None, issue_unix | None)   decoded BLE (B only)
  planned_stop(t)  -> (t0, t1) | None   public shift calendar
  oracle(t)        -> list of true future idle windows (C only)
A policy returns (t_cross | None, note). None means "keep waiting"; the runner calls it again later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from intercycle.robot.predict import idle_windows
from intercycle.types import DAY0_UNIX, State


@dataclass
class Ctx:
    need_s: float               # time inside the zone
    margin_s: float
    sees_motion: Callable[[float], bool]
    planned_stop: Callable[[float], tuple[float, float] | None]
    knowledge: Callable[[float], tuple] | None = None
    oracle: Callable[[float], list[tuple[float, float]]] | None = None
    distrust_until: float = -1.0     # set when the code disagrees with what the robot sees


def _break_entry(ctx: Ctx, t: float) -> float | None:
    ps = ctx.planned_stop(t)
    if ps and ps[1] - t >= ctx.need_s + ctx.margin_s and not ctx.sees_motion(t):
        return t
    return None


def a0_shutdown_only(ctx: Ctx, t: float) -> tuple[float | None, str, float]:
    """Returns (t_cross, note, next_check_t)."""
    e = _break_entry(ctx, t)
    if e is not None:
        return e, "planned stop", t
    return None, "waiting for a planned stop", t + 5.0


def a_reactive(ctx: Ctx, t: float) -> tuple[float | None, str, float]:
    if not ctx.sees_motion(t):
        return t, "saw the machine stop", t
    return None, "machine moving", t + 0.25


def b_intercycle(ctx: Ctx, t: float, waited_unknown: float = 0.0) -> tuple[float | None, str, float]:
    e = _break_entry(ctx, t)
    if e is not None:
        return e, "planned stop", t
    model, now, issue = ctx.knowledge(t)
    wins = idle_windows(model, now, issue, t) if (model and now and issue) else None
    if wins is not None and now.state == State.ACTIVE and not ctx.sees_motion(t):
        t_idle = issue - DAY0_UNIX + now.next_idle_in_s
        if t_idle - t > 2.0:                 # code says "running for a while yet", camera sees it stopped
            ctx.distrust_until = t + 120.0
    if t < ctx.distrust_until:
        wins = None
        waited_unknown = max(waited_unknown, 181.0)
    if wins is None:
        if waited_unknown > 180.0:      # no usable prediction for 3 min: behave like A
            r = a_reactive(ctx, t)
            return r[0], "no prediction, reactive fallback", r[2]
        return None, "prediction unavailable", t + 0.5
    need = ctx.need_s + ctx.margin_s
    for a, b in wins:
        start = max(a, t)
        if b - start >= need:
            if start <= t + 1e-6:
                if ctx.sees_motion(t):
                    return None, "predicted idle but camera sees motion", t + 0.25
                return t, "predicted idle window fits", t
            return None, "waiting for predicted window", min(a, t + 2.0)
    return None, "no window fits", t + 2.0


def c_oracle(ctx: Ctx, t: float) -> tuple[float | None, str, float]:
    for a, b in ctx.oracle(t):
        start = max(a, t)
        if b - start >= ctx.need_s:
            if start <= t + 1e-6:
                return t, "oracle window", t
            return None, "oracle wait", start
    return None, "oracle: no window", t + 5.0


def fits_ever(model, need_s: float) -> bool:
    """Can this machine's normal idle window ever fit an inspection? (B decides to defer to breaks.)"""
    return model is not None and (1 - model.busy_fraction) * model.period_s * (1 - 3 * model.jitter) >= need_s
