"""PRODUCTION_BRIEF A4: the broadcast may only ADD keep-out and only LOWER speed; unknown phase -> union of all L1 bins."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO))

from intercycle.observer import mapping as M  # noqa: E402

from intercycle_nav import masks as K  # noqa: E402

SITE = K.Site.from_scenario()
GRID = K.Grid(-2.0, -2.0, 0.25, 496, 208)
T0 = 10 * 3600.0


def _cell(x, y):
    return int((x - GRID.x0) / GRID.res), int((y - GRID.y0) / GRID.res)


def _snap(windows):
    ms = [K.MachineView(mid, i, "IDLE", 0.5, windows) for i, mid in enumerate(SITE.ids)]
    return K.Snapshot(T0, True, 1, True, 0.1, M.RUNNING, ms)


def test_broadcast_cannot_free_floor_cells():
    site = replace(SITE, hard_keepout=[(4.0, 2.0, 6.0, 4.0)])
    p = K.MaskParams()
    free_everything = GRID.zeros()                                # a broadcast-derived mask that frees every cell
    stale = _snap(None)                                           # stale / missing windows = occupied
    floor = K.keepout_floor(GRID, "learned", site, stale, object(), p)
    k, _ = K.enforce_monotone(free_everything, GRID.zeros(), floor, site, GRID)
    for mid, (x0, y0, x1, y1) in site.rects.items():
        assert k[_cell((x0 + x1) / 2, (y0 + y1) / 2)] == K.KEEP_OUT, mid
    assert k[_cell(5.0, 3.0)] == K.KEEP_OUT                       # hard keep-out in every condition
    # no verified map at all -> every zone is floor
    floor2 = K.keepout_floor(GRID, "learned", site, None, None, p)
    k2, _ = K.enforce_monotone(free_everything, GRID.zeros(), floor2, site, GRID)
    assert all(k2[_cell((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)] == K.KEEP_OUT for r in site.rects.values())
    # an open, verified window legitimately opens its zone, but never the hard keep-out
    open_ = _snap([[T0 - 5.0, T0 + 30.0]])
    floor3 = K.keepout_floor(GRID, "oracle", site, open_, object(), p)
    k3, _ = K.enforce_monotone(free_everything, GRID.zeros(), floor3, site, GRID)
    r = site.rects["S10"]
    assert k3[_cell((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)] == K.FREE
    assert k3[_cell(5.0, 3.0)] == K.KEEP_OUT
    # none: only the hard keep-out
    k4, _ = K.enforce_monotone(free_everything, GRID.zeros(), K.keepout_floor(GRID, "none", site, None, None, p), site, GRID)
    assert k4.sum() == (K.keepout_floor(GRID, "none", site, None, None, p) > 0).sum() * K.KEEP_OUT


def test_broadcast_cannot_raise_speed_limits():
    site = replace(SITE, speed_limits=[(10.0, 30.0, 20.0, 34.0, 40)])
    raise_ = np.full((GRID.nx, GRID.ny), 100, np.int16)
    raise_[_cell(15.0, 32.0)] = 120                               # out of range
    _, sp = K.enforce_monotone(GRID.zeros(), raise_.astype(np.int16), GRID.zeros(), site, GRID)
    assert sp.max() <= 100
    assert sp[_cell(15.0, 32.0)] == 40 and sp[_cell(12.0, 31.0)] == 40      # static limit wins
    _, sp2 = K.enforce_monotone(GRID.zeros(), GRID.zeros().astype(np.int16), GRID.zeros(), site, GRID)
    assert sp2[_cell(15.0, 32.0)] == 40                           # "no limit" from the broadcast cannot lift it
    assert sp2[_cell(50.0, 32.0)] == 0


@pytest.mark.skipif(not (REPO / "out" / "layered_map.npz").exists(), reason="map not built")
def test_unknown_phase_projects_the_union_of_all_bins():
    view = M.LayeredMap.load(REPO / "out" / "layered_map.npz")
    p = K.MaskParams()
    for prod in (M.RUNNING, M.ANDON):
        mv = K.MachineView("S10", 0, "UNKNOWN", None, None)
        assert K.machine_bins(view, 0, mv, prod, p) == M.ALL_BINS
