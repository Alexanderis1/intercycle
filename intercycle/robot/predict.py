"""Turn a decoded broadcast (slow model + fast nowcast) into predicted idle windows in sim time."""
from __future__ import annotations

import math

from intercycle.types import DAY0_UNIX, MachineModel, MachineNow, State


def idle_windows(model: MachineModel, now: MachineNow, issue_unix: float, t_now: float,
                 horizon_s: float = 900.0, k_sigma: float = 2.0) -> list[tuple[float, float]] | None:
    """Predicted idle windows (sim time), each shrunk by the growing phase uncertainty.

    Returns None when the entry cannot be trusted (low confidence, expired, stopped, unknown).
    """
    t_issue = issue_unix - DAY0_UNIX
    if now.confidence < 4 or model.confidence < 4 or now.state in (State.STOPPED, State.UNKNOWN):
        return None
    if t_now > t_issue + now.ttl_s:
        return None
    T = model.period_s
    idle_len = (1 - model.busy_fraction) * T
    out = []
    if now.state == State.IDLE:
        s0, l0 = t_issue, now.next_idle_len_s
        nxt = t_issue + l0 + model.busy_fraction * T
    else:
        s0, l0 = t_issue + now.next_idle_in_s, now.next_idle_len_s
        nxt = s0 + T
    k = 0
    s, length = s0, l0
    while s < t_now + horizon_s:
        u = k_sigma * model.jitter * T * math.sqrt(k + 1) + 0.05     # 0.05 s quantisation + latency
        a, b = s + (u if k > 0 or now.state != State.IDLE else 0.0), s + length - u
        if b > a and b > t_now:
            out.append((max(a, t_issue), b))
        k += 1
        s, length = (nxt, idle_len) if k == 1 else (s + T, idle_len)
    return out


def transfer_windows_from_index(index_t: float, takt: float, t_now: float, horizon_s: float,
                                transfer_s: float, cross_s: float) -> list[tuple[float, float]]:
    """Safe windows for the line crossing: gaps between skid transfers, shortened by the crossing time."""
    k0 = math.floor((t_now - index_t) / takt)
    out = []
    for k in range(k0, k0 + int(horizon_s / takt) + 2):
        a = index_t + k * takt + transfer_s
        b = index_t + (k + 1) * takt - cross_s
        if b > a and b > t_now:
            out.append((a, b))
    return out
