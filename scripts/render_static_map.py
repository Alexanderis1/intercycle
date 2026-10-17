"""Render the static collision map with the site graph overlaid, for eyeballing: out/static_map.png.

    .venv\\Scripts\\python.exe scripts\\render_static_map.py [--out out/static_map.png] [--routes]

10 px per metre (0.1 m per pixel) over x -2..122, z -2..50.  Dark = occupied raster cells (what the lidar and the
clearance field see), amber dashed = machine hazard zones, blue = site-graph edges/nodes, red = nodes/edges where the
robot footprint (swept every 0.25 m, aligned to the edge) hits static geometry, grey dotted = mover routes (--routes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from intercycle.robot.sitegraph import SiteGraph  # noqa: E402
from intercycle.scenario import load  # noqa: E402
from intercycle.sim import geometry as G  # noqa: E402

INK, MUTED = "#1f2933", "#52606d"
OCC, FREE = "#3e4c59", "#fbfbfa"
BLUE, RED, AMBER, GREEN = "#2f6fb0", "#d0312d", "#c27c0e", "#2d7d46"
# lighter fill per body class so the eye can tell what is what (the raster underneath is what the sim uses)
FILL = {"wall": "#27323d", "conveyor": "#8f9ba8", "robot_base": "#c27c0e", "tip_dresser": "#e0a43c", "curtain_post": "#d0312d",
        "lifter_post": "#8a5a00", "fixture_table": "#6b7b8c", "column": "#27323d", "rack_market": "#7c6f9b",
        "rack_dh": "#2f6fb0", "crah": "#2d7d46"}


def graph_report(sm: G.StaticMap, g: SiteGraph):
    """(bad node names, bad edges as (a, b, x, z))."""
    bad_nodes = []
    for n, (x, z) in g.pos.items():
        if any(G.static_hit(sm, G.footprint("robot", x, z, yaw)) for yaw in (0.0, np.pi / 2)):
            bad_nodes.append(n)
    bad_edges = []
    for a, nb in g.adj.items():
        for b in nb:
            if a < b:
                hit = G.segment_hit(sm, g.pos[a], g.pos[b], "robot", 0.25)
                if hit is not None:
                    bad_edges.append((a, b, *hit))
    return bad_nodes, bad_edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "static_map.png"))
    ap.add_argument("--routes", action="store_true", help="overlay the mover routes from truth/agents.py")
    args = ap.parse_args()

    scn = load()
    sm = G.StaticMap(scn)
    g = SiteGraph(scn)
    bad_nodes, bad_edges = graph_report(sm, g)
    bad_edge_keys = {(a, b) for a, b, *_ in bad_edges}

    # figure: axes exactly 1240 x 520 px => 0.1 m per pixel
    dpi, ml, mr, mb, mt = 100, 60, 20, 44, 40
    W, H = G.NX + ml + mr, G.NZ + mb + mt
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([ml / W, mb / H, G.NX / W, G.NZ / H])
    ax.imshow(np.where(sm.grid.T, 0.0, 1.0), origin="lower", extent=(G.X0, G.X1, G.Z0, G.Z1), cmap=matplotlib.colors.ListedColormap([OCC, FREE]),
              vmin=0, vmax=1, interpolation="nearest", zorder=0)
    for r in sm.rects():
        c = FILL.get(r["cls"])
        if c and r["cls"] not in ("wall", "column", "curtain_post"):
            ax.add_patch(Rectangle((r["cx"] - r["w"] / 2, r["cz"] - r["d"] / 2), r["w"], r["d"], fc=c, ec="none", alpha=0.55, zorder=1))
    for m in scn.machines:
        x0, z0, x1, z1 = m.rect
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=AMBER, lw=0.9, ls=(0, (4, 3)), zorder=2))
        ax.text(m.x, z1 + 0.25 if m.on_line else z0 + 0.25, m.id, ha="center", va="bottom" if m.on_line else "bottom", fontsize=7, color=AMBER, zorder=6)
    for zid, rect in (("walkway", (51.5, 22.8, 53.5, 25.2)),):
        ax.add_patch(Rectangle(rect[:2], rect[2] - rect[0], rect[3] - rect[1], fill=False, ec=GREEN, lw=0.9, ls=":", zorder=2))
    if args.routes:
        from intercycle.truth.agents import agents
        for a in agents(scn, 0):
            p = np.array(a._pts)
            ax.plot(p[:, 0], p[:, 1], color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=3)
    for a, nb in g.adj.items():
        for b in nb:
            if a < b:
                (xa, za), (xb, zb) = g.pos[a], g.pos[b]
                bad = (a, b) in bad_edge_keys
                ax.plot([xa, xb], [za, zb], color=RED if bad else BLUE, lw=1.6 if bad else 1.0, zorder=4)
    for n, (x, z) in g.pos.items():
        bad = n in bad_nodes
        ax.plot(x, z, "o", ms=4.5 if bad else 3.0, mfc=RED if bad else BLUE, mec="white", mew=0.6, zorder=5)
        ax.annotate(n, (x, z), xytext=(2, 3), textcoords="offset points", fontsize=5.5, color=RED if bad else INK, zorder=7)
    ax.set_xlim(G.X0, G.X1)
    ax.set_ylim(G.Z0, G.Z1)
    ax.set_xlabel("x (m, along the line)", fontsize=8, color=MUTED)
    ax.set_ylabel("z (m, across the hall)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.set_xticks(np.arange(0, 121, 10))
    ax.set_yticks(np.arange(0, 49, 8))
    for s in ax.spines.values():
        s.set_visible(False)
    msg = f"static collision map (SYNTHETIC)   robot-footprint check: {len(bad_nodes)} nodes and {len(bad_edges)} edges collide"
    fig.text(ml / W, 1 - 14 / H, msg, fontsize=9, color=INK, va="center", ha="left", fontweight="bold")
    fig.text(ml / W, 1 - 28 / H, "dark = occupied raster (0.1 m cells); amber dashed = hazard zones; blue = site graph; red = collides; green dotted = walkway gap",
             fontsize=7, color=MUTED, va="center", ha="left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"wrote {out}  ({W}x{H} px, 0.1 m/px)")
    print("colliding nodes:", bad_nodes or "none")
    for a, b, x, z in bad_edges:
        print(f"colliding edge {a}-{b} first at ({x:.2f}, {z:.2f})")


if __name__ == "__main__":
    main()
