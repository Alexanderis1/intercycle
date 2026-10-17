"""Observer broadcast -> Nav2 costmap-filter masks (pure logic, no ROS). Used by `observer_bridge`.

What it does (DIRECTION 5.3):
  * `ObserverState` ingests untrusted transport records and keeps only what verifies: slow/fast/live layers through
    `codec.ble.Receiver` (Ed25519, site id, freshness, seq), the map layer through `codec.maplayer.MapReceiver` (signed
    manifest + chunk hash table, valid_s). Fast entries past their TTL become UNKNOWN; an expired map is dropped.
  * KeepoutFilter mask = union over [now, now + H] of each machine's L1 envelope at its projected phase bins
    (`MapView.predicted_bins`: the received phase advanced with the learned period, widened by k_sigma * phase
    uncertainty, +- `pad_bins`), plus every UNKNOWN cell. A machine whose fast entry is missing / stale / untrusted gets
    the union of all its bins (= its whole learned envelope). No map at all -> static keep-out of every hazard zone.
  * A machine whose hazard zone contains the robot's (AMCL) pose is HELD: the safety system stops it (hold if idle,
    trip if active; BRIEF), so its STOP-bin envelope is used, and the cells under the robot are cleared so Nav2 can
    always plan the way out.
  * SpeedFilter mask (percent of max speed, 0 = no limit) from the L2 vehicle / L3 people occupancy probability of the
    current and the next time-of-day bin in the current production state.

Never imports intercycle.truth or intercycle.sim (checked by `assert_no_truth`). The site map (scenario YAML: hall,
hazard-zone rectangles, public break calendar) is the robot's provisioned public map, not truth.

Frames: plan (x along the line, z across) -> ROS map frame x = plan x, y = plan z. Grids here are [ix, iy] (x-major);
`to_occupancy_data` flattens them row-major in y for nav_msgs/OccupancyGrid.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from intercycle.codec import ble
from intercycle.codec import maplayer as ML
from intercycle.codec.sign import Keyring, demo_key, keyring_from
from intercycle.observer import mapping as M
from intercycle.robot.predict import idle_windows
from intercycle.types import DAY0_UNIX, MachineModel, MachineNow, State

from intercycle_nav.feed import BLOB, FRAME

CONDITIONS = ("none", "static", "learned", "oracle")
KEEP_OUT, FREE = 100, 0


def assert_no_truth() -> None:
    bad = [m for m in sys.modules if m.startswith(("intercycle.truth", "intercycle.sim"))]
    if bad:
        raise RuntimeError(f"ground-truth / simulator modules imported in the robot process: {bad}")


def default_keyring(key_id: int) -> Keyring:
    """Verify key of the public DEMO signing key (codec.sign.demo_key; the seed is public, so it is a test key)."""
    return keyring_from({key_id: demo_key(key_id)})


def keyring_from_hex(spec: str) -> Keyring:
    """'1:abcd...,2:...' (key_id:hex verify key) -> Keyring."""
    from nacl.signing import VerifyKey

    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        k, h = part.split(":")
        out[int(k)] = VerifyKey(bytes.fromhex(h))
    return out


# ------------------------------------------------------------------------------------------------ grid
@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int

    def zeros(self, dtype=np.int8) -> np.ndarray:
        return np.zeros((self.nx, self.ny), dtype)

    def box_slice(self, x0, y0, x1, y1):
        """Cells touched by a world box (conservative: any overlap)."""
        i0 = max(0, int(math.floor((x0 - self.x0) / self.res)))
        i1 = min(self.nx, int(math.ceil((x1 - self.x0) / self.res)))
        j0 = max(0, int(math.floor((y0 - self.y0) / self.res)))
        j1 = min(self.ny, int(math.ceil((y1 - self.y0) / self.res)))
        return (slice(i0, max(i0, i1)), slice(j0, max(j0, j1)))

    def fill_boxes(self, g: np.ndarray, boxes, value) -> None:
        for b in np.asarray(boxes, float).reshape(-1, 4):
            g[self.box_slice(*b)] = np.maximum(g[self.box_slice(*b)], value)

    def resample(self, src: np.ndarray, sx0: float, sy0: float, sres: float, fill=0) -> np.ndarray:
        """Nearest-cell lookup of a coarser [nx, ny] grid (e.g. the 1 m site grid) at this grid's cell centres."""
        cx = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        cy = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        ii = np.floor((cx - sx0) / sres).astype(int)
        jj = np.floor((cy - sy0) / sres).astype(int)
        okx, oky = (ii >= 0) & (ii < src.shape[0]), (jj >= 0) & (jj < src.shape[1])
        out = np.full((self.nx, self.ny), fill, src.dtype)
        sub = src[np.clip(ii, 0, src.shape[0] - 1)][:, np.clip(jj, 0, src.shape[1] - 1)]
        out[np.ix_(okx, oky)] = sub[np.ix_(okx, oky)]
        return out

    def disc(self, x: float, y: float, r: float) -> np.ndarray:
        cx = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        cy = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return (cx[:, None] - x) ** 2 + (cy[None, :] - y) ** 2 <= r * r


def to_occupancy_data(g: np.ndarray):
    """[nx, ny] -> OccupancyGrid.data (row-major, row = y), as array('b') (rclpy takes it without a per-element copy)."""
    import array

    return array.array("b", np.ascontiguousarray(g.T).astype(np.int8).tobytes())


# ------------------------------------------------------------------------------------------------ site (public map)
@dataclass
class Site:
    """The robot's provisioned public site map: machine ids (broadcast index order), hazard-zone rectangles, calendar."""
    ids: list[str]
    rects: dict[str, tuple[float, float, float, float]]   # plan x0, z0, x1, z1 == ROS x0, y0, x1, y1
    stops: list[tuple[float, float, str]]                 # public calendar (tod_start, tod_end, kind)
    site_id: int
    key_id: int
    extent: tuple[float, float, float, float]             # x0, y0, x1, y1 of hall + data hall

    @classmethod
    def from_scenario(cls, path: str | None = None) -> Site:
        from intercycle import scenario as S

        scn = S.load(path) if path else S.load()
        rects = {m.id: tuple(float(v) for v in m.rect) for m in scn.machines}
        ext = (float(scn.hall["x"][0]), float(scn.hall["z"][0]), float(scn.data_hall["x"][1]), float(scn.hall["z"][1]))
        return cls([m.id for m in scn.machines], rects, list(scn.line_stops()), int(scn.site_id), int(scn.key_id), ext)

    def calendar(self, t: float) -> int | None:
        tod = t % M.DAY
        for a, b, kind in self.stops:
            if a <= tod < b:
                return M.BREAK if kind == "break" else M.SHIFT_CHANGE
        return None

    def zone_of(self, x: float, y: float, pad: float = 0.0) -> str | None:
        for mid, (x0, y0, x1, y1) in self.rects.items():
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                return mid
        return None


# ------------------------------------------------------------------------------------------------ verified state
@dataclass
class MachineView:
    id: str
    index: int
    state: str                    # IDLE / ACTIVE / STOPPED / UNKNOWN (UNKNOWN also = stale / untrusted / missing)
    phase: float | None           # fraction of the map's cycle now (None = unknown -> whole envelope)
    windows: list | None          # predicted idle windows (plant time); None = cannot be trusted
    held: bool = False


@dataclass
class Snapshot:
    t: float
    map_ok: bool
    map_version: int | None
    slow_ok: bool
    fast_age_s: float | None
    prod: int
    machines: list[MachineView] = field(default_factory=list)


class ObserverState:
    """Everything the robot believes about the plant, built only from verified broadcast bytes."""

    def __init__(self, site: Site, keyring: Keyring | None = None, fast_max_age_s: float = 30.0):
        self.site = site
        kr = keyring if keyring is not None else default_keyring(site.key_id)
        self.rx = ble.Receiver(kr, site.site_id)
        self.map_rx = ML.MapReceiver(kr, site.site_id)
        self.fast_max_age_s = fast_max_age_s
        self.stats: Counter[str] = Counter()

    # ---- input
    def ingest(self, t_rx: float, record: bytes) -> None:
        kind, data = record[:1], record[1:]
        if kind == FRAME:
            if ML.is_map_frame(data):
                if self.map_rx.ingest(t_rx, data) is not None:
                    self.stats["map_complete"] += 1
            else:
                d = self.rx.ingest(t_rx, data)
                if d is not None:
                    self.stats["ok" if d.ok else f"rej_{d.reason}"] += 1
        elif kind == BLOB:
            d = self.rx.accept(t_rx, data)
            self.stats["ok" if d.ok else f"rej_{d.reason}"] += 1
        else:
            self.stats["rej_bad_record"] += 1

    def all_stats(self) -> dict:
        return {"bridge": dict(self.stats), "layers": dict(self.rx.stats), "map": dict(self.map_rx.stats)}

    # ---- views
    def map_view(self, t: float) -> ML.DecodedMap | None:
        return self.map_rx.latest(t)

    def map_version(self) -> int | None:
        man = self.map_rx.manifest
        return None if man is None or not self.map_rx.done else int(man.version)

    def snapshot(self, t: float, horizon_s: float = 600.0, robot_xy: tuple[float, float] | None = None,
                 k_sigma: float = 2.0, held_pad_m: float = 0.5) -> Snapshot:
        """held_pad_m: the robot holds a machine while its body (half-length 0.5 m) reaches into the zone, as a light
        curtain / area scanner would see it, not only while its centre is inside."""
        d = self.rx.latest(t)
        slow, fast = (d.slow, d.fast) if d.ok else (None, None)
        fast_age = None if fast is None else DAY0_UNIX + t - fast.issue_time
        if fast_age is not None and fast_age > self.fast_max_age_s:
            fast = None
        models = {m.index: m for m in slow.machines} if slow else {}
        nows = {m.index: m for m in fast.machines} if fast else {}
        view = self.map_view(t)
        held_id = self.site.zone_of(*robot_xy, pad=held_pad_m) if robot_xy is not None else None
        out = []
        n_stopped = 0
        for i, mid in enumerate(self.site.ids):
            mod, now = models.get(i), nows.get(i)
            st = "UNKNOWN" if now is None else State(now.state).name
            n_stopped += st == "STOPPED"
            ph = own_phase(mod, now, fast.issue_time if fast else None, t)
            if ph is not None:
                ph = ph % 1.0                          # map phase convention: 0 = start of motion in the zone (see README)
            wins = idle_windows(mod, now, fast.issue_time, t, horizon_s, k_sigma) if (mod and now and fast) else None
            out.append(MachineView(mid, i, st, ph, wins, held=(mid == held_id)))
        cal = self.site.calendar(t)
        prod = cal if cal is not None else (M.ANDON if out and n_stopped >= 0.5 * len(out) else M.RUNNING)
        return Snapshot(t, view is not None, self.map_version(), slow is not None, fast_age, prod, out)


def own_phase(model: MachineModel | None, now: MachineNow | None, issue_unix: float | None, t: float) -> float | None:
    """Phase (fraction of the cycle since the zone's motion started) at plant time t, from one fast entry.
    None when it cannot be trusted (missing, low confidence, expired TTL, STOPPED, UNKNOWN)."""
    if model is None or now is None or issue_unix is None:
        return None
    t_issue = issue_unix - DAY0_UNIX
    if now.confidence < 4 or model.confidence < 4 or now.state in (State.STOPPED, State.UNKNOWN) or t > t_issue + now.ttl_s:
        return None
    T, b = float(model.period_s), float(model.busy_fraction)
    if not (T > 0 and np.isfinite(T)):
        return None
    if now.state == State.ACTIVE:
        ph = min(max(b - now.next_idle_in_s / T, 0.0), b)
    else:
        ph = min(max(1.0 - now.next_idle_len_s / T, b), 1.0)
    return ph + (t - t_issue) / T


# ------------------------------------------------------------------------------------------------ masks
@dataclass
class MaskParams:
    horizon_s: float = 6.0          # keep-out union over [now, now + H]
    step_s: float = 0.5
    k_sigma: float = 2.0
    pad_bins: int = 1               # extra phase bins each side (phase-convention / latency margin)
    inflate_m: float = 0.2          # grow every envelope box (ISO 13855-style S = K*T + C margin)
    static_margin_m: float = 0.15   # static keep-out margin; the wait pose (1.05 m out) must keep >= 0.6 m to turn in place
    self_clear_r: float = 0.75      # cells cleared under the robot (so Nav2 can always plan out)
    andon_valid_s: float = 3.0
    speed_levels: tuple = ((0.30, 40), (0.10, 60), (0.03, 80))   # (p_min, percent of max speed)


def idle_through(windows, t0: float, t1: float) -> bool:
    """True if one predicted idle window covers [t0, t1]."""
    return any(a <= t0 and b >= t1 for a, b in windows or [])


def machine_bins(view: M.MapView, i: int, mv: MachineView, prod: int, p: MaskParams) -> tuple:
    if mv.held:
        return (M.STOP_BIN,)
    bins: set[int] = set()
    for dt in np.arange(0.0, p.horizon_s + 1e-9, p.step_s):
        b = view.predicted_bins(i, mv.phase, prod, float(dt), p.k_sigma, andon_valid_s=p.andon_valid_s)
        if b == M.ALL_BINS:
            return M.ALL_BINS
        for c in b:
            if c == M.STOP_BIN:
                bins.add(c)
            else:
                bins.update((c + d) % M.N_PHASE for d in range(-p.pad_bins, p.pad_bins + 1))
    return tuple(sorted(bins))


def keepout_mask(grid: Grid, condition: str, site: Site, snap: Snapshot | None, view: M.MapView | None,
                 p: MaskParams, robot_xy: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """(mask [nx, ny] int8 in {0, 100}, info). Conservative fallbacks are described in the module docstring."""
    g = grid.zeros()
    info: dict = {"condition": condition, "source": condition}
    if condition == "none":
        return g, info
    if condition == "static" or view is None or snap is None:
        info["source"] = "static" if condition == "static" else "static_fallback_no_map"
        for r in site.rects.values():
            m = p.static_margin_m
            grid.fill_boxes(g, [(r[0] - m, r[1] - m, r[2] + m, r[3] + m)], KEEP_OUT)
    else:
        ids = view.machine_ids()
        per = {}
        for mv in snap.machines:
            if mv.id not in ids:                   # a machine the map does not know: its whole zone
                r = site.rects[mv.id]
                grid.fill_boxes(g, [r], KEEP_OUT)
                per[mv.id] = "zone"
                continue
            i = ids.index(mv.id)
            # light-curtain semantics (BRIEF): crossing the boundary of an ACTIVE zone trips the line, so the zone
            # rectangle itself is keep-out unless a predicted idle window covers all of [now, now + H] (or the robot
            # is inside and therefore holds the machine). The L1 envelope adds the swept floor around it.
            zone_open = mv.held or idle_through(mv.windows, snap.t, snap.t + p.horizon_s)
            if not zone_open:
                grid.fill_boxes(g, [site.rects[mv.id]], KEEP_OUT)
            bins = machine_bins(view, i, mv, snap.prod, p)
            boxes = view.boxes_for(mv.id, bins)
            if len(boxes) and p.inflate_m > 0:
                boxes = boxes + np.array([-p.inflate_m, -p.inflate_m, p.inflate_m, p.inflate_m])
            grid.fill_boxes(g, boxes, KEEP_OUT)
            per[mv.id] = {"bins": "held" if mv.held else ("all" if bins == M.ALL_BINS else len(bins)),
                          "zone": "open" if zone_open else "closed"}
        s = view.spec["site"]
        U = view.unknown_mask(M.tod_bin(snap.t), snap.prod)
        if U.any():
            g = np.maximum(g, np.where(grid.resample(U.astype(np.int8), s["x0"], s["z0"], s["res"]) > 0, KEEP_OUT, 0)
                           .astype(np.int8))
        info["bins"] = per
        info["unknown_cells_1m"] = int(U.sum())
    if robot_xy is not None and p.self_clear_r > 0:
        d = grid.disc(robot_xy[0], robot_xy[1], p.self_clear_r)
        info["self_cleared"] = int((g[d] > 0).sum())
        g[d] = FREE
    info["keepout_cells"] = int((g > 0).sum())
    return g, info


def speed_mask(grid: Grid, condition: str, snap: Snapshot | None, view: M.MapView | None, p: MaskParams) -> tuple[np.ndarray, dict]:
    """(mask [nx, ny] int8: percent of max speed, 0 = no limit, info)."""
    g = grid.zeros()
    if condition in ("none", "static") or view is None or snap is None:
        return g, {"speed_cells": 0}
    s = view.spec["site"]
    tb = M.tod_bin(snap.t)
    P = np.zeros((s["nx"], s["nz"]), np.float32)
    for tt in (tb, (tb + 1) % M.N_TOD):
        for cls in (M.VEHICLE, M.PERSON):
            pr, _ = view.traffic(cls, tt, snap.prod)
            P = np.maximum(P, pr)
    lim = np.zeros(P.shape, np.int8)
    for pmin, pct in sorted(p.speed_levels, key=lambda v: v[0]):      # ascending p: higher p overwrites with a lower %
        lim = np.where(P >= pmin, np.int8(pct), lim).astype(np.int8)
    g = grid.resample(lim, s["x0"], s["z0"], s["res"])
    return g, {"speed_cells": int((g > 0).sum()), "p_max": float(P.max()) if P.size else 0.0}


def snapshot_json(snap: Snapshot, condition: str, extra: dict | None = None) -> dict:
    return {"t": snap.t, "condition": condition, "map_ok": snap.map_ok, "map_version": snap.map_version,
            "slow_ok": snap.slow_ok, "fast_age_s": snap.fast_age_s, "prod": M.PROD_NAMES[snap.prod],
            "machines": {m.id: {"state": m.state, "phase": m.phase, "held": m.held,
                                "windows": None if m.windows is None else [[round(a, 3), round(b, 3)] for a, b in m.windows[:8]]}
                         for m in snap.machines},
            **(extra or {})}
