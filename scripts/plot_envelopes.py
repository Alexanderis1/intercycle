"""Eyeball check of the ground-truth machine envelopes: S30 swept area per 1/16 cycle-phase bin.

Samples the kinematics of the whole floor for 30 min and, for every phase bin of S30's cycle window, paints the
union of (a) S30's two weld arms, (b) the skids / car bodies and (c) the AGV docks in view onto a 0.1 m grid.
This is the object the observer has to learn later (layer L1: machine envelopes by phase bin).

    python scripts/plot_envelopes.py [--seed 0] [--start-h 7] [--minutes 30] [--out out/envelopes_S30.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.scenario import load  # noqa: E402
from intercycle.sim.world import World  # noqa: E402
from intercycle.truth.kinematics import Kinematics  # noqa: E402

MID = "S30"
NBINS = 16
RES = 0.1
VIEW = (22.0, 14.0, 38.0, 34.0)                 # x0, z0, x1, z1 shown around S30
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdad3"
C_ARM, C_SKID, C_AGV = "#2a78d6", "#b9b8b2", "#eb6834"      # arms (blue), car bodies (neutral), AGV dock (orange)


class Canvas:
    def __init__(self):
        x0, z0, x1, z1 = VIEW
        self.xs = x0 + RES * (np.arange(int(round((x1 - x0) / RES))) + 0.5)
        self.zs = z0 + RES * (np.arange(int(round((z1 - z0) / RES))) + 0.5)
        self.g = np.zeros((len(self.xs), len(self.zs)), dtype=bool)

    def paint(self, c: np.ndarray) -> None:
        for cx, cz, r in c:
            i0, i1 = np.searchsorted(self.xs, [cx - r, cx + r])
            j0, j1 = np.searchsorted(self.zs, [cz - r, cz + r])
            if i1 <= i0 or j1 <= j0:
                continue
            m = ((self.xs[i0:i1] - cx) ** 2)[:, None] + ((self.zs[j0:j1] - cz) ** 2)[None, :] <= r * r
            self.g[i0:i1, j0:j1] |= m

    @property
    def area(self) -> float:
        return float(self.g.sum() * RES * RES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-h", type=float, default=7.0)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "out" / "envelopes_S30.png"))
    a = ap.parse_args()

    scn = load()
    m = scn.machine(MID)
    t0 = a.start_h * 3600.0
    t1 = t0 + a.minutes * 60.0
    world = World(scn, a.seed, t0, t1)
    kin = Kinematics(scn, world)
    wins = {id(w): w for w in world.W[MID]}
    bs = np.array([w.bs for w in world.W[MID]])
    kinds = [w.kind for w in world.W[MID]]

    arms = [Canvas() for _ in range(NBINS)]
    skids = [Canvas() for _ in range(NBINS)]
    docks = [Canvas() for _ in range(NBINS)]
    n_samples = np.zeros(NBINS, int)
    n_active = np.zeros(NBINS, int)
    vx0, vz0, vx1, vz1 = VIEW
    seen: set = set()
    for t in np.arange(t0, t1, 1.0 / a.hz):
        i = int(np.searchsorted(bs, t, side="right")) - 1
        if i < 0 or kinds[i] != "cycle":                      # stops (andon / breaks) are not part of the takt
            continue
        k = min(int(kin.cycle_phase(MID, float(t)) * NBINS), NBINS - 1)
        b = kin.bodies(float(t))
        n_samples[k] += 1
        n_active[k] += kin.phase(MID, float(t))[0] == "active"
        arms[k].paint(b[MID])
        for tag, src, cv in ((0, b["SKIDS"], skids[k]), (1, b["AGVDOCK"], docks[k])):
            key = (k, tag, src.tobytes())
            if key in seen:                                   # rest states repeat: paint each distinct set once
                continue
            seen.add(key)
            cv.paint(src[(src[:, 0] + src[:, 2] > vx0) & (src[:, 0] - src[:, 2] < vx1)] if len(src) else src)

    fig, axs = plt.subplots(4, 4, figsize=(11.5, 13.6), facecolor=SURFACE, constrained_layout=True)
    x0, z0, x1, z1 = m.rect
    extent = (vx0, vx1, vz0, vz1)
    for k, ax in enumerate(axs.ravel()):
        ax.set_facecolor(SURFACE)
        img = np.ones((len(arms[k].zs), len(arms[k].xs), 3))
        for cv, col in ((skids[k], C_SKID), (arms[k], C_ARM), (docks[k], C_AGV)):
            rgb = matplotlib.colors.to_rgb(col)
            mask = cv.g.T
            img[mask] = rgb
        bgc = matplotlib.colors.to_rgb(SURFACE)
        blank = np.all(img == 1.0, axis=-1)
        img[blank] = bgc
        ax.imshow(img, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=INK2, lw=1.0, ls=(0, (4, 3))))
        for zc in (22.8, 25.2):
            ax.axhline(zc, color=GRID, lw=0.8, zorder=0)
        for (mid_, bx, bz) in kin.arm_bases:
            if mid_ == MID:
                ax.plot([bx], [bz], marker="s", ms=4, color=INK, zorder=3)
        act = n_active[k] / n_samples[k] if n_samples[k] else float("nan")
        ax.set_title(f"{k}-{k + 1}/16 | {act:.0%} active | arms {arms[k].area:.1f} m2", fontsize=8, color=INK, loc="left")
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vz0, vz1)
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if k % 4:
            ax.set_yticklabels([])
        if k < 12:
            ax.set_xticklabels([])
    union = Canvas()
    for cv in arms:
        union.g |= cv.g
    fig.suptitle(f"{MID} ground-truth envelope by cycle phase (seed {a.seed}, {a.minutes:g} min from {a.start_h:g}:00, "
                 f"{int(n_samples.sum())} samples); arm union over all bins {union.area:.0f} of {(x1 - x0) * (z1 - z0):.0f} m2 zone",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.legend(handles=[Patch(color=C_ARM, label="S30 weld arms (5 circles each)"), Patch(color=C_SKID, label="skids / car bodies"),
                        Patch(color=C_AGV, label="AGV dock (every 6th cycle, 20 s)"),
                        Patch(fill=False, ec=INK2, ls="--", label="hazard zone"), Patch(color=INK, label="robot base")],
               loc="outside lower center", ncol=3, fontsize=8.5, frameon=False, labelcolor=INK2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"saved {out}  (samples per bin: {n_samples.tolist()})")


if __name__ == "__main__":
    main()
