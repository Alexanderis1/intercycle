"""AMCL initial-pose spread helpers for timing_planner (R-W4 start-pose noise; pure logic, no ROS).

The planner sends /initialpose with the covariance of the start-pose draw only when the episode applied start noise
(fix round 2: the spread defaults to 0, so a run with an exact start pose, e.g. a body-shop R3 batch run, keeps AMCL
on the exact pose as before). AMCL's acknowledgement is judged on the components that were actually sent non-zero.
"""
from __future__ import annotations

COV_EPS = 1e-6
SPREAD_INDEX = {"xx": 0, "yy": 7, "yawyaw": 35}     # row-major 6x6 (x, y, z, roll, pitch, yaw)


def spread_components(cov_sent) -> list[int]:
    """Indices (0, 7, 35) of the covariance components sent non-zero."""
    return [i for i in SPREAD_INDEX.values() if float(cov_sent[i]) > 0.0]


def ack_ok(cov_sent, cov_got) -> bool:
    """AMCL answered our spread init: every component we sent non-zero is non-zero in its answer. False when nothing
    was spread (there is nothing to acknowledge; the planner does not publish then)."""
    idx = spread_components(cov_sent)
    return bool(idx) and all(float(cov_got[i]) > COV_EPS for i in idx)
