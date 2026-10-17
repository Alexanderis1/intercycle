"""Shared data contracts. Every module talks through these types.

Time convention: simulation time `t` is seconds since 00:00 local time of sim day 0.
Wall-clock (unix) time for issued codes is DAY0_UNIX + t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

DAY0_UNIX = 1_792_360_800  # 2026-10-19 00:00:00 CEST (UTC+2)


@dataclass(frozen=True)
class Window:
    """One machine cycle as it happens on the floor (ground truth, before any robot effects).

    The hazard zone is ACTIVE on [bs, is_) and IDLE on [is_, ie). The next window starts at ie.
    """
    bs: float
    is_: float
    ie: float
    kind: str = "cycle"  # "cycle" | "andon" (unplanned stop extends idle) | "break" | "shift_change"

    @property
    def period(self) -> float:
        return self.ie - self.bs

    @property
    def idle(self) -> float:
        return self.ie - self.is_


class State(IntEnum):
    IDLE = 0
    ACTIVE = 1
    STOPPED = 2   # idle longer than the learned cycle predicts (andon, break): remaining time unknown
    UNKNOWN = 3


@dataclass(frozen=True)
class MachineModel:
    """Slow layer: what the observer learned about one machine. Valid for hours."""
    index: int            # position in the scenario's machine list (robot has the site map)
    period_s: float
    busy_fraction: float
    jitter: float         # cycle-to-cycle sigma, fraction of period
    drift_per_h: float    # fraction per hour, signed
    stops_per_h: float
    confidence: int       # 0..15; robot ignores < 4


@dataclass(frozen=True)
class MachineNow:
    """Fast layer: where one machine is in its cycle at the moment of issue. Valid for seconds."""
    index: int
    state: State
    next_idle_in_s: float     # from issue time to the start of the next idle window (0 if idle now)
    next_idle_len_s: float    # predicted length of that idle window (remaining length if idle now)
    ttl_s: float              # entry is valid until issue_time + ttl_s
    confidence: int           # 0..15
    seq: int                  # 16-bit, increments every issue


@dataclass(frozen=True)
class SlowLayer:
    site_id: int
    key_id: int
    issue_time: float                     # unix seconds
    traffic: list[list[int]]              # [zone][hour 0..23] -> level 0..15
    machines: list[MachineModel]


@dataclass(frozen=True)
class FastLayer:
    site_id: int
    key_id: int
    issue_time: float                     # unix seconds, millisecond precision
    seq: int
    machines: list[MachineNow]


class MoverClass(IntEnum):
    PERSON = 0
    AGV = 1
    TUGGER = 2
    FORKLIFT = 3
    OTHER = 4


@dataclass(frozen=True)
class Mover:
    """Live layer: one tracked person or vehicle as the observer sees it at issue time."""
    track_id: int          # 8-bit, wraps
    cls: MoverClass
    x: float               # metres, site frame
    z: float
    heading: float         # radians, 0 = +z, clockwise towards +x
    speed: float           # m/s


@dataclass(frozen=True)
class LiveLayer:
    site_id: int
    key_id: int
    issue_time: float      # unix seconds, millisecond precision
    seq: int
    movers: list[Mover]


@dataclass
class Decoded:
    """What the robot ends up with after receiving and verifying a broadcast."""
    ok: bool
    reason: str = ""                      # "", "bad_signature", "expired", "unknown_key", "malformed", "stale_seq"
    slow: SlowLayer | None = None
    fast: FastLayer | None = None
    live: LiveLayer | None = None
    extra: dict = field(default_factory=dict)
