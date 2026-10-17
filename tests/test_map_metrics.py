"""Held-out map metrics (intercycle/eval/map_metrics.py) on hand-made records with known answers (fast)."""
from __future__ import annotations

import numpy as np

from intercycle.eval import map_metrics as MM
from intercycle.observer import mapping as M
from intercycle.scenario import load

SCN = load()
MIDS = [m.id for m in SCN.machines]


def _map():
    m = M.LayeredMap(SCN)
    m.visits_cam[:] = 1000
    m.l1_visits[:] = 100
    i = m._mi("HEM1")
    m._l1(i)[3, 10:14, 10:14] = 100.0          # bin 3: a 4 x 4 cell block
    m.period[:] = 45.0
    m.jitter[:] = 0.005
    return m, i


def _rec(m, i, inside, outside_centre, outside_overlap, n_ticks=3):
    g = MM.Grids(m.spec)
    nz = g.nz[i]
    cells_in = [(10 + k % 4) * nz + 10 + k // 4 for k in range(inside)]
    cells_c = [(20 + k) * nz + 20 for k in range(outside_centre)]
    cells_o = [(25 + k) * nz + 25 for k in range(outside_overlap)]
    et, em, ec, ecen = [], [], [], []
    for n in range(n_ticks):
        for c, cen in [(c, True) for c in cells_in + cells_c] + [(c, False) for c in cells_o]:
            et.append(n)
            em.append(i)
            ec.append(c)
            ecen.append(cen)
    Mn = len(MIDS)
    bins = np.full((n_ticks, Mn), -1, np.int16)
    bins[:, i] = 3
    ph = np.full((n_ticks, Mn), np.nan)
    ph[:, i] = 3.5 / 16
    return dict(l1_t=np.arange(n_ticks) * 2.0, l1_tod=np.full(n_ticks, 20, np.int16), l1_prod=np.zeros(n_ticks, np.int8),
                l1_bin=bins, l1_phase=ph, e_tick=np.array(et, np.int32), e_mach=np.array(em, np.int8), e_cell=np.array(ec, np.int32),
                e_centre=np.array(ecen, bool), mids=np.array(MIDS))


def test_unsafe_free_and_false_occupied_exact():
    m, i = _map()
    rec = _rec(m, i, inside=10, outside_centre=2, outside_overlap=3)
    r = MM.score_l1(m, [rec], preds=(("env", "bin", 0.0, 0),))["env|bin|h0|d0"]
    assert abs(r["unsafe_free_centre"] - 2 / 12) < 1e-12           # 2 of 12 centre cells outside the 16-cell envelope
    assert abs(r["unsafe_free_overlap"] - 5 / 15) < 1e-12
    # false-occupied: 16 predicted cells, 10 truly occupied, per tick, for HEM1 (other machines predict nothing: empty envelopes)
    assert abs(r["false_occupied_share_centre"] - 6 / 16) < 1e-12
    assert r["per_machine_unsafe_free_centre"]["HEM1"] == 2 / 12
    # inflating by enough cells closes the misses next to the block but not the far ones
    r2 = MM.score_l1(m, [rec], preds=(("env", "bin", 0.0, 12),))["env|bin|h0|d12"]
    assert r2["unsafe_free_centre"] < r["unsafe_free_centre"]


def test_robot_bins_and_safe_boxes_never_lose_cells():
    m, i = _map()
    rec = _rec(m, i, inside=16, outside_centre=0, outside_overlap=0)
    out = MM.score_l1(m, [rec], preds=(("box", "robot", 0.0, 0), ("env", "bin", 0.0, 0)))
    assert out["box|robot|h0|d0"]["unsafe_free_centre"] == 0.0 and out["env|bin|h0|d0"]["unsafe_free_centre"] == 0.0
    # horizon: the prediction made at tick n is scored against the truth at tick n + h
    out = MM.score_l1(m, [_rec(m, i, 16, 0, 0, n_ticks=10)], preds=(("box", "robot", 10.0, 0),))
    assert out["box|robot|h10|d0"]["instants"] == (10 - 5) * len(MIDS)


def test_auc_from_histograms():
    pos = np.array([0, 0, 0, 5])
    neg = np.array([5, 0, 0, 0])
    assert MM._auc(pos, neg) == 1.0
    assert MM._auc(neg, pos) == 0.0
    assert MM._auc(np.array([1, 1]), np.array([1, 1])) == 0.5
