"""twin/inspect_model.py (lane S2 TASKS, E-W3): the thermal capture and the keyed CRN defect draw."""
from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from twin import inspect_model as IM


class _Zero:
    def normal(self, *a, **k):
        return 0.0


class _Wall:
    """A static caster that blocks every ray at 1 m."""

    def cast(self, origin, dirs, rmax):
        return np.full(len(dirs), 1.0)


CAM = np.array((0.0, 0.0, 0.35))
N = np.array((-1.0, 0.0, 0.0))


def _shot(tgt, v=(0.0, 0.0, 0.0), wz=0.0, static=None, normal=N):
    return IM.thermal_capture(0.0, CAM, np.array(tgt, float), normal, np.array(v, float), wz, static, None,
                              cast(np.random.Generator, _Zero()))


def test_focal_length_and_constants():
    assert math.isclose(IM.F_TH, 320 / math.tan(math.radians(12.0)))
    assert round(IM.F_TH, 1) == 1505.5
    assert IM.THERMAL_PXM == 100.0 and IM.TH_EXPOSURE_S == 0.010 and IM.DEFECT_RATE == 0.25


def test_thermal_pass_and_range_limit():
    assert _shot((5.0, 0.0, 0.35))["pass"]
    r_max = IM.F_TH / IM.THERMAL_PXM                         # 15.06 m at theta = 0
    assert _shot((r_max - 0.1, 0.0, 0.35))["pass"]
    far = _shot((r_max + 0.1, 0.0, 0.35))
    assert not far["pass"] and not far["checks"]["pxm_ok"]


def test_thermal_fails_on_occlusion_incidence_blur_and_frame():
    assert not _shot((5.0, 0.0, 0.35), static=_Wall())["checks"]["unoccluded"]
    oblique = np.array((-math.cos(math.radians(65)), math.sin(math.radians(65)), 0.0))
    assert not _shot((5.0, 0.0, 0.35), normal=oblique)["checks"]["incidence_ok"]
    # blur: omega * f_th * 10 ms >= 1 px at omega = 0.07 rad/s
    b = _shot((5.0, 0.0, 0.35), wz=0.07)
    assert not b["checks"]["blur_ok"] and b["blur_px"] == pytest.approx(0.07 * IM.F_TH * 0.010, abs=1e-3)
    assert _shot((5.0, 0.0, 0.35), wz=0.05)["checks"]["blur_ok"]

    class _Big:
        def normal(self, *a, **k):
            return math.radians(15.0)                        # beyond the 12 deg half-HFOV
    off = IM.thermal_capture(0.0, CAM, np.array((5.0, 0.0, 0.35)), N, np.zeros(3), 0.0, None, None,
                              cast(np.random.Generator, _Big()))
    assert not off["checks"]["in_image"] and not off["pass"]


def test_defect_draw_is_keyed_crn_and_near_the_rate():
    a = [IM.defect_draw(3, 1, "cp_P9a_ram", m) for m in range(4000)]
    assert a == [IM.defect_draw(3, 1, "cp_P9a_ram", m) for m in range(4000)]      # same key, same answer
    assert abs(sum(a) / len(a) - 0.25) < 0.025
    assert a != [IM.defect_draw(4, 1, "cp_P9a_ram", m) for m in range(4000)]      # another world differs
    assert a != [IM.defect_draw(3, 1, "cp_P9a_die", m) for m in range(4000)]      # another checkpoint differs
    assert not any(IM.defect_draw(3, 1, "x", m, rate=0.0) for m in range(100))
    assert all(IM.defect_draw(3, 1, "x", m, rate=1.0) for m in range(100))
    with pytest.raises(ValueError):
        IM.defect_draw(3, 1, "x", 0, rate=1.5)
