"""M-W4 on the body-shop LayeredMap: in-zone change detection + forgetting on each machine's L1 envelope grid.

This is the adapter lane S8 (A-lite) uses for its "new arm" and "program change" scenarios on the existing body-shop
code. It runs NEXT TO `LayeredMap.update` and `observer/change.py` (which keeps watching the static space outside every
zone + 1 m) and never imports ground truth.

Per known machine i it keeps one `change_detect.ChangeLayer` over the machine's L1 grid (zone + margin, env_res cells)
with the 17 phase bins (16 + STOP) as slices.

Hits = the map's own L1 rule (`LayeredMap.update`): first the detections located in a camera that is offline / bumped
(map.cam_offline) or whose floor view is invalid (mapping.floor_ok_of) are dropped, exactly as `_drop_offline` does;
then PART or SKID detections whose observation `machine == i` are rasterised with radius r - shrink_m on cell centres.

Observed (Observation v2 only, never "the whole grid"): machine i's grid is observed on a tick when the tracker knows its
phase bin (cond.phase_bin[i] >= 0), its camera's floor view is valid and not offline (the map's L1 visit rule), and its
zone is fully visible (obs.zone_vis[i] >= vis_min, default mapping.ZONE_VIS_MIN = 1: an occluder or an unhealthy camera
on any part of the zone makes the whole grid unobserved, since zone_vis does not say which cells are hidden). Within an
observed grid, the cells whose site cell belongs to an offline / unhealthy camera are not observed. An unobserved cell
learns nothing, does not decay, does not feed the CUSUM and is never lifted. A v1 observation (no zone_vis /
cam_floor_ok) is refused with ValueError: without visibility the adapter cannot tell "seen empty" from "not seen".
Availability cost: with the measured shim a zone is not fully visible on ~3 % of the ticks (person false positives near
the zone, blackouts); those ticks are lost for learning, decay and lifting.

What is broadcast (audit S7a #1, #2). The robot receives the map: `LayeredMap.envelope` (undecayed counts, per phase
bin) plus map.unknown_rects. The detector does not change the envelope. Instead it judges its CUSUM and its lift against
that broadcast: on each tick, the FREE cells passed to `ChangeLayer.step(..., free=)` are the cells outside the map's
envelope of the current bin (at cfg.p_min, the robot's 0.05). So
- the cells the map broadcasts FREE are the cells monitored (in a bin the layer has learned, n >= min_visits);
- a hit on a cell the map broadcasts FREE is unexplained and blocks the lift of the UNKNOWN cells around it, so an
  UNKNOWN box is lifted only when the change went away or the map's own counts took it in: whatever the decayed counts
  say, a lift never exposes a FREE cell the change still occupies;
- forgetting never removes anything from the broadcast (the envelope's counts are not touched), so a rarely used
  envelope cell (a weekly program variant) is exactly as occupied as without this module.
Forgetting serves the CUSUM's null here (the decayed counts are the learned behaviour of each cell). Availability cost:
an UNKNOWN box stays until the map's undecayed counts take the change in (~13 h for a newly always-occupied cell against
a week of history), and usually as long as the change persists, because detection noise keeps hitting FREE fringe cells
inside its 0.5 m dilation. In a bin the layer has not learned yet (fewer than min_visits observed ticks, e.g. the STOP bin
of a machine that never stops, or the first ~8.5 min after a start since nothing is persisted) the cells are not
monitored: the map shows its own fallback there (union of the learned bins), as without this module.

Every tick: `mp.update(obs, t, cond)` then `det.step(obs, cond)` (the comparison must see this tick's counts; a step
without a new map update is refused with RuntimeError). `step` pushes the UNKNOWN boxes into map.unknown_rects itself.

    det = L1ChangeDetector(mp)                  # mp: LayeredMap (known machines at construction time), v2 observations
    events = det.step(obs, cond)                # after mp.update(obs, t, cond), same tick; also syncs the boxes
    det.unknown_rects()                         # [(x0, z0, x1, z1)] world boxes covering every UNKNOWN cell (dilated)
    det.unknown_mask(machine_id)                # [nx, nz] UNKNOWN cells (dilated by cfg.dilate)
    det.monitored(machine_id, phase_bin)        # [nx, nz] cells the CUSUM watches in that bin (learned and map-FREE)
    det.envelope_prob(machine_id, phase_bin)    # decayed (forgetting) occupancy grid
    det.evidence_hits(obs, i)                   # the hit rule (flat bool over machine i's grid)

Events: {"kind": "inzone_change", "t", "machine", "cells", "rect"} when cells turn UNKNOWN,
        {"kind": "inzone_cleared", "t", "machine", "cells"} when UNKNOWN is lifted.
"""
from __future__ import annotations

import numpy as np

from intercycle.observer import mapping as M
from intercycle.observer.change_detect import ChangeConfig, ChangeLayer

L1_CFG = ChangeConfig(dilate=2)        # 2 cells = 0.5 m around every UNKNOWN cell on the 0.25 m envelope grid


class L1ChangeDetector:
    def __init__(self, mp: M.LayeredMap, cfg: ChangeConfig | None = None, vis_min: float = M.ZONE_VIS_MIN):
        self.map = mp
        self.cfg = cfg or L1_CFG
        self.vis_min = float(vis_min)
        self.ids = [d["id"] for d in mp.spec["machines"]]
        self.layers = [ChangeLayer(int(mp.l1_nx[i] * mp.l1_nz[i]), M.N_BINS, self.cfg,
                                   shape=(int(mp.l1_nx[i]), int(mp.l1_nz[i]))) for i in range(len(self.ids))]
        s, res = mp.spec["site"], mp.spec["env_res"]
        self.cam_of_l1_cell = []                         # per machine: camera of each L1 cell (flat), as cam_of_cell
        for i in range(len(self.ids)):
            gx = mp.l1_x0[i] + (np.arange(mp.l1_nx[i]) + 0.5) * res
            gz = mp.l1_z0[i] + (np.arange(mp.l1_nz[i]) + 0.5) * res
            ix = np.clip(np.floor((gx - s["x0"]) / s["res"]).astype(np.int64), 0, mp.nx - 1)
            iz = np.clip(np.floor((gz - s["z0"]) / s["res"]).astype(np.int64), 0, mp.nz - 1)
            self.cam_of_l1_cell.append(mp.cam_of_cell[ix[:, None], iz[None, :]].reshape(-1))
        self._owned: list[tuple] = []
        self._map_tick: int | None = None

    # ------------------------------------------------------------------ evidence (the map's L1 rule)
    def _gone(self, obs: M.Observation) -> list[int]:
        """Cameras whose detections are not evidence this tick: offline / bumped or floor view invalid."""
        return sorted(set(np.flatnonzero(~M.floor_ok_of(obs)).tolist()) | set(self.map.cam_offline))

    def evidence_hits(self, obs: M.Observation, i: int, gone=None) -> np.ndarray:
        """Flat bool over machine i's L1 grid: the cells LayeredMap.update would add a hit to (same drop as
        `_drop_offline`, same raster). `gone` overrides the dropped cameras (default: `_gone(obs)`)."""
        mp = self.map
        gone = self._gone(obs) if gone is None else sorted(gone)
        nx, nz = int(mp.l1_nx[i]), int(mp.l1_nz[i])
        out = np.zeros(nx * nz, bool)
        sel = ((obs.cls == M.PART) | (obs.cls == M.SKID)) & (obs.machine == i)
        if gone and sel.any():
            s = mp.spec["site"]
            ix = np.clip(np.floor((obs.x - s["x0"]) / s["res"]).astype(np.int64), 0, mp.nx - 1)
            iz = np.clip(np.floor((obs.z - s["z0"]) / s["res"]).astype(np.int64), 0, mp.nz - 1)
            sel &= ~np.isin(mp.cam_of_cell[ix, iz], gone)
        if sel.any():
            _, ix, iz = M.raster(obs.x[sel], obs.z[sel], np.maximum(obs.r[sel] - mp.shrink_m, 0.05), mp.l1_x0[i],
                                 mp.l1_z0[i], nx, nz, mp.spec["env_res"], centre=True)
            out[ix * nz + iz] = True
        return out

    def _map_free(self, i: int, b: int) -> np.ndarray:
        """Flat bool: cells of machine i's grid outside the map's broadcast envelope of bin b."""
        return ~self.map.envelope(self.ids[i], b, self.cfg.p_min).reshape(-1)

    # ------------------------------------------------------------------ step
    def step(self, obs: M.Observation, cond: M.Cond) -> list[dict]:
        mp = self.map
        if obs.zone_vis is None or obs.cam_floor_ok is None:
            raise ValueError("L1ChangeDetector needs Observation v2 (zone_vis, cam_floor_ok): convert with "
                             "mapping.ObsV2Converter; without visibility a hidden cell would read as seen empty")
        if mp.n_ticks == 0 or (self._map_tick is not None and mp.n_ticks <= self._map_tick):
            raise RuntimeError("call L1ChangeDetector.step after LayeredMap.update of the same tick")
        self._map_tick = mp.n_ticks
        gone = self._gone(obs)
        zv = np.asarray(obs.zone_vis, float)
        ev: list[dict] = []
        for i, layer in enumerate(self.layers):
            b = int(cond.phase_bin[i]) if i < len(cond.phase_bin) else -1
            if b < 0 or i >= len(zv) or not zv[i] >= self.vis_min or int(mp.cam_of_machine[i]) in gone:
                continue                                   # not observed: no learning, no decay, no CUSUM, no lift
            observed = ~np.isin(self.cam_of_l1_cell[i], gone) if gone else np.ones(layer.n_cells, bool)
            r = layer.step(b, self.evidence_hits(obs, i, gone), observed, free=self._map_free(i, b))
            if len(r.alarmed):
                ev.append(dict(kind="inzone_change", t=obs.t_cap, machine=self.ids[i], cells=int(len(r.alarmed)),
                               rect=list(self._rect(i, r.alarmed))))
            if len(r.lifted):
                ev.append(dict(kind="inzone_cleared", t=obs.t_cap, machine=self.ids[i], cells=int(len(r.lifted))))
        self.sync()
        return ev

    def _rect(self, i: int, cells: np.ndarray) -> tuple[float, float, float, float]:
        mp = self.map
        nz, res = int(mp.l1_nz[i]), mp.spec["env_res"]
        ix, iz = np.divmod(np.asarray(cells, np.int64), nz)
        return (float(mp.l1_x0[i] + ix.min() * res), float(mp.l1_z0[i] + iz.min() * res),
                float(mp.l1_x0[i] + (ix.max() + 1) * res), float(mp.l1_z0[i] + (iz.max() + 1) * res))

    # ------------------------------------------------------------------ queries / broadcast
    def unknown_mask(self, machine_id: str) -> np.ndarray:
        i = self.ids.index(machine_id)
        return self.layers[i].unknown_mask().reshape(self.layers[i].shape)

    def monitored(self, machine_id: str, phase_bin: int) -> np.ndarray:
        i = self.ids.index(machine_id)
        return self.layers[i].monitored(phase_bin, free=self._map_free(i, phase_bin)).reshape(self.layers[i].shape)

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
        """Replace this detector's previous boxes in map.unknown_rects by the current ones. Other owners' boxes are
        kept, also when one equals a box of ours (one instance is removed per box we owned). Called by `step`."""
        base = list(self.map.unknown_rects)
        for r in self._owned:
            if r in base:
                base.remove(r)
        self._owned = self.unknown_rects()
        self.map.unknown_rects = base + self._owned

    def envelope_prob(self, machine_id: str, phase_bin: int) -> np.ndarray:
        i = self.ids.index(machine_id)
        return np.nan_to_num(self.layers[i].p(phase_bin)).reshape(self.layers[i].shape)
