"""M-W4 on the body-shop LayeredMap: in-zone change detection + forgetting on each machine's L1 envelope grid.

This is the adapter lane S8 (A-lite) uses for its "new arm 0.5-2 m from an existing cell" and "program change" scenarios
on the existing body-shop code. It runs NEXT TO `LayeredMap.update` and `observer/change.py` (which keeps watching the
static space outside every zone + 1 m) and never imports ground truth.

Per known machine i it keeps one `change_detect.ChangeLayer` over the machine's L1 grid (zone + margin, env_res cells)
with the 17 phase bins (16 + STOP) as slices. Hits use the map's own L1 rule (`LayeredMap.update`): PART or SKID
detections whose observation `machine == i`, rasterised with radius r - shrink_m on cell centres. A machine's grid is
observed on a tick when the tracker knows its phase bin (cond.phase_bin[i] >= 0) and its camera is up and not offline /
bumped (the same condition under which the map counts an L1 visit).

    det = L1ChangeDetector(mp)                  # mp: LayeredMap (known machines at construction time)
    events = det.step(obs, cond)                # after or before mp.update(obs, t, cond), same tick
    det.sync()                                  # UNKNOWN boxes -> mp.unknown_rects (other owners' rects are kept)
    det.unknown_rects()                         # [(x0, z0, x1, z1)] world boxes covering every UNKNOWN cell
    det.envelope_prob(machine_id, phase_bin)    # decayed (forgetting) occupancy grid
    det.write_decayed_l1()                      # copy the decayed counts into mp's L1 (so the broadcast forgets too)

Events: {"kind": "inzone_change", "t", "machine", "cells", "rect"} when cells turn UNKNOWN,
        {"kind": "inzone_cleared", "t", "machine", "cells"} when UNKNOWN is lifted.
"""
from __future__ import annotations

import numpy as np

from intercycle.observer import mapping as M
from intercycle.observer.change_detect import ChangeConfig, ChangeLayer

L1_CFG = ChangeConfig(dilate=2)        # 2 cells = 0.5 m around every UNKNOWN cell on the 0.25 m envelope grid


class L1ChangeDetector:
    def __init__(self, mp: M.LayeredMap, cfg: ChangeConfig | None = None):
        self.map = mp
        self.cfg = cfg or L1_CFG
        self.ids = [d["id"] for d in mp.spec["machines"]]
        self.layers = [ChangeLayer(int(mp.l1_nx[i] * mp.l1_nz[i]), M.N_BINS, self.cfg,
                                   shape=(int(mp.l1_nx[i]), int(mp.l1_nz[i]))) for i in range(len(self.ids))]
        self._owned: list[tuple] = []

    def _hits(self, obs: M.Observation, i: int) -> np.ndarray:
        mp = self.map
        nx, nz = int(mp.l1_nx[i]), int(mp.l1_nz[i])
        out = np.zeros(nx * nz, bool)
        sel = ((obs.cls == M.PART) | (obs.cls == M.SKID)) & (obs.machine == i)
        if sel.any():
            _, ix, iz = M.raster(obs.x[sel], obs.z[sel], np.maximum(obs.r[sel] - mp.shrink_m, 0.05), mp.l1_x0[i],
                                 mp.l1_z0[i], nx, nz, mp.spec["env_res"], centre=True)
            out[ix * nz + iz] = True
        return out

    def step(self, obs: M.Observation, cond: M.Cond) -> list[dict]:
        mp = self.map
        ok = np.asarray(obs.cam_ok, bool).copy()
        if mp.cam_offline:
            ok[list(mp.cam_offline)] = False
        ev: list[dict] = []
        for i, layer in enumerate(self.layers):
            b = int(cond.phase_bin[i]) if i < len(cond.phase_bin) else -1
            if b < 0 or not ok[mp.cam_of_machine[i]]:
                continue                                   # not observed: no learning, no decay, no CUSUM
            r = layer.step(b, self._hits(obs, i), np.ones(layer.n_cells, bool))
            if len(r.alarmed):
                ev.append(dict(kind="inzone_change", t=obs.t_cap, machine=self.ids[i], cells=int(len(r.alarmed)),
                               rect=list(self._rect(i, r.alarmed))))
            if len(r.lifted):
                ev.append(dict(kind="inzone_cleared", t=obs.t_cap, machine=self.ids[i], cells=int(len(r.lifted))))
        return ev

    def _rect(self, i: int, cells: np.ndarray) -> tuple[float, float, float, float]:
        mp = self.map
        nz, res = int(mp.l1_nz[i]), mp.spec["env_res"]
        ix, iz = np.divmod(np.asarray(cells, np.int64), nz)
        return (float(mp.l1_x0[i] + ix.min() * res), float(mp.l1_z0[i] + iz.min() * res),
                float(mp.l1_x0[i] + (ix.max() + 1) * res), float(mp.l1_z0[i] + (iz.max() + 1) * res))

    def unknown_mask(self, machine_id: str) -> np.ndarray:
        i = self.ids.index(machine_id)
        return self.layers[i].unknown_mask().reshape(self.layers[i].shape)

    def unknown_rects(self, max_boxes: int = 12) -> list[tuple[float, float, float, float]]:
        """World boxes that together cover every UNKNOWN cell (dilated by cfg.dilate cells) of every machine."""
        mp = self.map
        out = []
        res = mp.spec["env_res"]
        for i, layer in enumerate(self.layers):
            U = layer.unknown_mask()
            if not U.any():
                continue
            for i0, j0, i1, j1 in M.safe_box_cover(U.reshape(layer.shape), max_boxes):
                out.append((float(mp.l1_x0[i] + i0 * res), float(mp.l1_z0[i] + j0 * res),
                            float(mp.l1_x0[i] + i1 * res), float(mp.l1_z0[i] + j1 * res)))
        return out

    def sync(self) -> None:
        """Replace this detector's previous boxes in map.unknown_rects by the current ones (others' boxes are kept)."""
        base = [r for r in self.map.unknown_rects if r not in self._owned]
        self._owned = self.unknown_rects()
        self.map.unknown_rects = base + self._owned

    def envelope_prob(self, machine_id: str, phase_bin: int) -> np.ndarray:
        i = self.ids.index(machine_id)
        return np.nan_to_num(self.layers[i].p(phase_bin)).reshape(self.layers[i].shape)

    def write_decayed_l1(self) -> None:
        """Overwrite the map's L1 counts of the known machines with the decayed ones (every cell of a machine grid is
        observed together, so the per-cell decayed visits are equal across the grid and one value per bin is exact)."""
        mp = self.map
        for i, layer in enumerate(self.layers):
            view = mp.l1_view(i)
            for b in range(M.N_BINS):
                view[b] = layer.k[b].reshape(layer.shape)
                mp.l1_visits[i, b] = float(layer.n[b].max()) if layer.n_cells else 0.0
        mp._cover_cache.clear()
