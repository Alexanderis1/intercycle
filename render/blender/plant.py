"""Heavy-industry plant dressing for the realistic Blender look (workstream X, DIRECTION.md §7 X item 3).

Everything here is static and placed from scenario/bodyshop.yaml so the render matches the sim and the map:
building shell (precast plinth + corrugated cladding, roller doors), column grid outside every zone used by
the sim, roof trusses + purlins + deck with skylight strips, a high-bay LED grid, floor slab with saw-cut
joints and painted aisle markings, overhead services (cable trays, spiral ducts, sprinkler mains) and
floor clutter (pallets, bins, carts, extinguishers, bollards, signage). Functions take the BScene `bs`
(render/blender/replay_bpy.py) and use its helpers (box/cyl/light/material).

Materials: world-space mapped (so scaled unit cubes need no UVs). `mat()` takes the keys in PBR_KEYS from the
CC0 PBR library (render/blender/materials.py, coords="world"); the rest are procedural.
Seeded RNG: the same YAML always gives the same plant.
"""
from __future__ import annotations

import math
import random

ROOF_EAVE = 12.0          # bottom chord of the roof trusses (m); CCTV is at 9 m, lights hang at 10.6 m
ROOF_TOP = 13.6
COL_DX = 12.0             # column pitch along x
COL_ROWS_Y = (0.0, 12.0, 35.5, 48.0)   # wall rows + two internal rows that fall between sim zones


# =====================================================================================
# node helpers
# =====================================================================================
class _NT:
    """Tiny node-tree builder: values may be sockets or floats."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.nodes, self.links = self.nt.nodes, self.nt.links

    def n(self, kind, **props):
        node = self.nodes.new(kind)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def link(self, out, inp):
        self.links.new(out, inp)

    def _feed(self, sock, v):
        if hasattr(v, "is_output"):
            self.link(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=0.0, c=None):
        m = self.n("ShaderNodeMath", operation=op)
        self._feed(m.inputs[0], a)
        self._feed(m.inputs[1], b)
        if c is not None:
            self._feed(m.inputs[2], c)
        return m.outputs[0]

    def pos(self):
        return self.n("ShaderNodeNewGeometry").outputs["Position"]

    def xyz(self):
        s = self.n("ShaderNodeSeparateXYZ")
        self.link(self.pos(), s.inputs[0])
        return s.outputs[0], s.outputs[1], s.outputs[2]

    def noise(self, scale, detail=6.0, rough=0.55, vec=None):
        t = self.n("ShaderNodeTexNoise")
        t.inputs["Scale"].default_value = scale
        t.inputs["Detail"].default_value = detail
        t.inputs["Roughness"].default_value = rough
        self.link(vec if vec is not None else self.pos(), t.inputs["Vector"])
        return t.outputs["Fac"]

    def ramp(self, fac, stops):
        r = self.n("ShaderNodeValToRGB")
        el = r.color_ramp.elements
        el[0].position, el[0].color = stops[0][0], (*stops[0][1], 1.0)
        el[1].position, el[1].color = stops[-1][0], (*stops[-1][1], 1.0)
        for p, c in stops[1:-1]:
            e = el.new(p)
            e.color = (*c, 1.0)
        self._feed(r.inputs["Fac"], fac)
        return r.outputs["Color"]

    def scale_col(self, col, s):
        v = self.n("ShaderNodeVectorMath", operation="SCALE")
        self._feed(v.inputs[0], col)
        self._feed(v.inputs["Scale"], s)
        return v.outputs["Vector"]

    def bump(self, height, strength, dist=0.01, normal=None):
        b = self.n("ShaderNodeBump")
        b.inputs["Strength"].default_value = strength
        b.inputs["Distance"].default_value = dist
        self._feed(b.inputs["Height"], height)
        if normal is not None:
            self.link(normal, b.inputs["Normal"])
        return b.outputs["Normal"]

    def bsdf(self):
        return self.nodes["Principled BSDF"]


def _darker(rgb, k):
    return tuple(max(0.0, c * k) for c in rgb)


def proc_mat(bs, name, rgb, rough=0.5, metal=0.0, var=0.18, scale=1.5, bump=0.15, grime=0.25):
    """Worn industrial surface: world-space noise varies colour/roughness, fine noise bumps,
    large-scale grime darkens patches. Works on any object without UVs."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    t = _NT(m)
    b = t.bsdf()
    n1 = t.noise(scale)
    col = t.ramp(n1, [(0.3, _darker(rgb, 1 - var)), (0.7, tuple(min(1.0, c * (1 + var * 0.5)) for c in rgb))])
    g = t.noise(scale * 0.12, detail=3.0)
    gk = t.math("MULTIPLY_ADD", t.math("SMOOTH_MIN", g, 0.62, 0.1), -grime, 1.0 + grime * 0.45)
    t.link(t.scale_col(col, gk), b.inputs["Base Color"])
    r = t.math("MULTIPLY_ADD", n1, 0.25, rough - 0.12)
    t.link(r, b.inputs["Roughness"])
    b.inputs["Metallic"].default_value = metal
    if bump > 0:
        t.link(t.bump(t.noise(scale * 18.0, detail=4.0), bump, 0.004), b.inputs["Normal"])
    bs.mats[name] = m
    return m


def concrete_floor(bs, name="plant_floor"):
    """Power-floated slab: tonal noise, oil/tyre stains, saw-cut joints every 6 m, light polish."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    t = _NT(m)
    b = t.bsdf()
    x, y, _ = t.xyz()

    def joint(c):
        f = t.math("FRACT", t.math("DIVIDE", c, 6.0))
        d = t.math("MINIMUM", f, t.math("SUBTRACT", 1.0, f))
        return t.math("LESS_THAN", d, 0.0009)

    j = t.math("MAXIMUM", joint(x), joint(y))
    base = t.ramp(t.noise(0.35, detail=8.0), [(0.25, (0.27, 0.27, 0.26)), (0.55, (0.41, 0.405, 0.39)), (0.8, (0.52, 0.51, 0.49))])
    stain = t.math("SMOOTH_MIN", t.noise(0.06, detail=4.0, rough=0.7), 0.58, 0.05)
    sk = t.math("MULTIPLY_ADD", t.math("SUBTRACT", 0.58, stain), -3.0, 1.0)
    # tyre scuffs: noise stretched along x (traffic runs along the hall), thresholded into dark streaks
    sv = t.n("ShaderNodeVectorMath", operation="MULTIPLY")
    t.link(t.pos(), sv.inputs[0])
    sv.inputs[1].default_value = (0.08, 2.5, 1.0)
    scuff = t.math("GREATER_THAN", t.noise(1.0, detail=3.0, vec=sv.outputs["Vector"]), 0.66)
    sc = t.math("MULTIPLY_ADD", scuff, -0.18, 1.0)
    fleck = t.math("MULTIPLY_ADD", t.noise(9.0, detail=2.0), 0.25, 0.88)
    k = t.math("MULTIPLY", t.math("MULTIPLY", sk, sc), t.math("MULTIPLY", fleck, t.math("MULTIPLY_ADD", j, -0.6, 1.0)))
    t.link(t.scale_col(base, k), b.inputs["Base Color"])
    t.link(t.math("MULTIPLY_ADD", t.noise(0.8), 0.35, 0.28), b.inputs["Roughness"])
    nb = t.bump(t.noise(40.0, detail=3.0), 0.08, 0.002)
    t.link(t.bump(j, 0.6, 0.004, normal=nb), b.inputs["Normal"])
    bs.mats[name] = m
    return m


def stripe_mat(bs, name, c1, c2, period=0.25, rough=0.55):
    """Diagonal hazard stripes (column wraps, guard rails)."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    t = _NT(m)
    b = t.bsdf()
    x, y, z = t.xyz()
    s = t.math("FRACT", t.math("DIVIDE", t.math("ADD", t.math("ADD", x, y), z), period))
    col = t.ramp(t.math("GREATER_THAN", s, 0.5), [(0.0, c1), (1.0, c2)])
    wear = t.math("MULTIPLY_ADD", t.math("SMOOTH_MIN", t.noise(3.0), 0.6, 0.1), -0.35, 1.2)
    t.link(t.scale_col(col, wear), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    bs.mats[name] = m
    return m


def corrugated(bs, name, rgb, axis="z", pitch=0.2, metal=0.6, rough=0.42):
    """Trapezoidal sheet look: ribs RUN ALONG `axis` (the wave varies across it).
    z = vertical ribs on walls; x / y = horizontal ribs on the roof deck; h = horizontal slats on a wall
    (roller shutters)."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    t = _NT(m)
    b = t.bsdf()
    x, y, z = t.xyz()
    coord = {"z": lambda: t.math("ADD", x, y), "y": lambda: x, "x": lambda: y, "h": lambda: z}[axis]()
    w = t.math("SINE", t.math("MULTIPLY", coord, 2 * math.pi / pitch))
    rib = t.math("SMOOTH_MAX", w, 0.2, 0.3)
    col = t.ramp(t.noise(0.8), [(0.3, _darker(rgb, 0.85)), (0.75, rgb)])
    t.link(col, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = metal
    t.link(t.math("MULTIPLY_ADD", t.noise(2.0), 0.2, rough - 0.08), b.inputs["Roughness"])
    t.link(t.bump(rib, 0.5, 0.02), b.inputs["Normal"])
    bs.mats[name] = m
    return m


def mesh_mat(bs, name, rgb=(0.04, 0.04, 0.045), cell=0.05, wire=0.07):
    """Welded wire mesh (guard panels, grating): wires are opaque metal, holes transparent.
    The grid uses (x+y, z) on vertical faces and (x, y) on horizontal ones (tray bottoms, platform deck)."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    t = _NT(m)
    b = t.bsdf()
    x, y, z = t.xyz()
    sn = t.n("ShaderNodeSeparateXYZ")
    t.link(t.n("ShaderNodeNewGeometry").outputs["Normal"], sn.inputs[0])
    flat = t.math("GREATER_THAN", t.math("ABSOLUTE", sn.outputs[2]), 0.5)
    uc = t.math("SUBTRACT", t.math("ADD", x, y), t.math("MULTIPLY", flat, y))        # x+y, or x when flat
    vc = t.math("ADD", z, t.math("MULTIPLY", flat, t.math("SUBTRACT", y, z)))       # z, or y when flat
    u = t.math("FRACT", t.math("DIVIDE", uc, cell))
    v = t.math("FRACT", t.math("DIVIDE", vc, cell))
    hole = t.math("MULTIPLY", t.math("GREATER_THAN", u, wire), t.math("GREATER_THAN", v, wire))
    t.link(t.math("SUBTRACT", 1.0, hole), b.inputs["Alpha"])
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Metallic"].default_value = 0.7
    b.inputs["Roughness"].default_value = 0.45
    bs.mats[name] = m
    return m


def glow_cam_only(bs, name, rgb, strength):
    """Emitter seen by the camera (lamp lenses, skylights, signs) that does not light the scene itself:
    real lighting comes from area lights, so there is no double counting and no extra noise."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    t = _NT(m)
    out = t.n("ShaderNodeOutputMaterial")
    em = t.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    tr = t.n("ShaderNodeBsdfTransparent")
    lp = t.n("ShaderNodeLightPath")
    mix = t.n("ShaderNodeMixShader")
    t.link(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    t.link(tr.outputs[0], mix.inputs[1])
    t.link(em.outputs[0], mix.inputs[2])
    t.link(mix.outputs[0], out.inputs["Surface"])
    bs.mats[name] = m
    return m


def skylight_mat(bs, name="skylight"):
    """Frosted polycarbonate: camera sees a bright diffuse panel, other rays pass straight to the sky
    (a transparent hop, not a caustic, so daylight reaches the floor without fireflies)."""
    if name in bs.mats:
        return bs.mats[name]
    m = bs.bpy.data.materials.new(name)
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    t = _NT(m)
    out = t.n("ShaderNodeOutputMaterial")
    em = t.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1.0)
    t.link(t.math("MULTIPLY_ADD", t.noise(0.5), 1.5, 2.2), em.inputs["Strength"])
    tr = t.n("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.75, 0.77, 0.8, 1.0)
    lp = t.n("ShaderNodeLightPath")
    mix = t.n("ShaderNodeMixShader")
    t.link(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    t.link(tr.outputs[0], mix.inputs[1])
    t.link(em.outputs[0], mix.inputs[2])
    t.link(mix.outputs[0], out.inputs["Surface"])
    bs.mats[name] = m
    return m


# plant key -> CC0 PBR key in render/blender/materials.py PLANT (world-mapped). Keys not listed stay procedural:
# glows, thin markings, wire mesh/grating (need alpha), and finishes with no matching scan.
PBR_KEYS = {
    "floor": "floor_light", "safety_yellow": "safety_yellow", "ral7035": "painted_grey", "steel_grey": "machine_grey",
    "galv": "galvanized", "rubber": "rubber_black", "wood": "wood_pallet", "cardboard": "cardboard",
    "cladding": "cladding", "hazard": "guard_hazard",
}
USE_PBR = True            # False: every key procedural (offline / A-B comparison)


def mat(bs, key):
    """Named plant material: the CC0 PBR library entry in PBR_KEYS when USE_PBR, else procedural."""
    if USE_PBR and key in PBR_KEYS:
        import materials  # render/blender/materials.py; a failure here is a real error, not a silent fallback
        return materials.plant(PBR_KEYS[key], coords="world")
    table = {
        "steel_grey": lambda: proc_mat(bs, "steel_grey", (0.30, 0.32, 0.34), rough=0.5, metal=0.6),
        "ral7035": lambda: proc_mat(bs, "ral7035", (0.70, 0.71, 0.69), rough=0.45, var=0.08, grime=0.3),
        "safety_yellow": lambda: proc_mat(bs, "safety_yellow", (0.85, 0.62, 0.02), rough=0.45, grime=0.35),
        "galv": lambda: proc_mat(bs, "galv", (0.62, 0.64, 0.66), rough=0.38, metal=0.9, var=0.25, scale=4.0),
        "dark_steel": lambda: proc_mat(bs, "dark_steel", (0.06, 0.065, 0.07), rough=0.5, metal=0.5),
        "rubber": lambda: proc_mat(bs, "rubber", (0.02, 0.02, 0.02), rough=0.85, bump=0.1),
        "wood": lambda: proc_mat(bs, "wood", (0.45, 0.33, 0.2), rough=0.8, var=0.3, scale=6.0, bump=0.3),
        "cardboard": lambda: proc_mat(bs, "cardboard", (0.48, 0.36, 0.22), rough=0.9, var=0.15),
        "bin_blue": lambda: proc_mat(bs, "bin_blue", (0.04, 0.16, 0.45), rough=0.5, var=0.08, bump=0.05),
        "bin_yellow": lambda: proc_mat(bs, "bin_yellow", (0.8, 0.55, 0.02), rough=0.5, var=0.08, bump=0.05),
        "red_paint": lambda: proc_mat(bs, "red_paint", (0.5, 0.03, 0.02), rough=0.4),
        "green_paint": lambda: proc_mat(bs, "green_paint", (0.05, 0.3, 0.1), rough=0.5),
        "precast": lambda: proc_mat(bs, "precast", (0.55, 0.55, 0.53), rough=0.85, var=0.12, scale=0.8, bump=0.25),
        "cladding": lambda: corrugated(bs, "cladding", (0.68, 0.70, 0.72), axis="z", pitch=0.2),
        "roof_deck": lambda: corrugated(bs, "roof_deck", (0.78, 0.79, 0.79), axis="y", pitch=0.25, metal=0.3),
        "roof_steel": lambda: proc_mat(bs, "roof_steel", (0.72, 0.73, 0.72), rough=0.5, metal=0.2, var=0.08, grime=0.2),
        "floor": lambda: concrete_floor(bs),
        "hazard": lambda: stripe_mat(bs, "hazard", (0.85, 0.62, 0.02), (0.02, 0.02, 0.02)),
        "line_yellow": lambda: proc_mat(bs, "line_yellow", (0.8, 0.6, 0.05), rough=0.5, var=0.25, bump=0.0, grime=0.5),
        "line_white": lambda: proc_mat(bs, "line_white", (0.75, 0.75, 0.72), rough=0.5, var=0.2, bump=0.0, grime=0.5),
        "line_green": lambda: proc_mat(bs, "line_green", (0.05, 0.35, 0.12), rough=0.55, var=0.2, bump=0.0, grime=0.4),
        "mesh": lambda: mesh_mat(bs, "mesh"),
        "grating": lambda: mesh_mat(bs, "grating", rgb=(0.35, 0.36, 0.37), cell=0.034, wire=0.18),
    }
    return table[key]()


# =====================================================================================
# building
# =====================================================================================
def h_column(bs, name, x, y, h, flange=0.4, depth=0.4, matk="steel_grey"):
    """Wide-flange (HEB-like) steel column from three boxes, with a hazard wrap at the base."""
    st = mat(bs, matk)
    bs.box(name + "_web", (0.02, depth - 0.04, h), (x, y, h / 2), st)
    for s in (-1, 1):
        bs.box(name + f"_fl{s}", (flange, 0.025, h), (x, y + s * (depth / 2 - 0.0125), h / 2), st)
    bs.box(name + "_wrap", (flange + 0.06, depth + 0.06, 1.5), (x, y, 0.75), mat(bs, "hazard"))
    bs.box(name + "_plate", (0.7, 0.7, 0.03), (x, y, 0.015), mat(bs, "dark_steel"))


def building(bs, rng: random.Random):
    s = bs.scn
    X, Y = s["hall"]["x"][1], s["hall"]["z"][1]
    dh = s["data_hall"]
    bs.box("floor", (X + 30, Y + 30, 0.2), (X / 2 + 12, Y / 2, -0.1), mat(bs, "floor"))
    # walls: precast plinth 0-3 m, corrugated cladding above, translucent daylight band 8.5-10 m
    for nm, (cx, cy, lx, ly) in {"S": (X / 2, -0.15, X, 0.3), "N": (X / 2, Y + 0.15, X, 0.3), "W": (-0.15, Y / 2, 0.3, Y)}.items():
        bs.box(f"plinth{nm}", (lx, ly, 3.0), (cx, cy, 1.5), mat(bs, "precast"))
        bs.box(f"clad{nm}", (lx, ly, 5.5), (cx, cy, 5.75), mat(bs, "cladding"))
        bs.box(f"band{nm}", (lx * 0.999 if lx > 1 else lx, ly * 0.999 if ly > 1 else ly, 1.5), (cx, cy, 9.25), skylight_mat(bs))
        bs.box(f"cladT{nm}", (lx, ly, ROOF_TOP - 10.0), (cx, cy, 10.0 + (ROOF_TOP - 10.0) / 2), mat(bs, "cladding"))
    # east wall with the data-hall opening (door at dh.door)
    z0, z1 = dh["z"]
    for k, (a, b) in enumerate(((0.0, z0), (z1, Y))):
        bs.box(f"wallE{k}", (0.3, b - a, ROOF_TOP), (X + 0.15, (a + b) / 2, ROOF_TOP / 2), mat(bs, "cladding"))
    bs.box("wallE_dh_top", (0.3, z1 - z0, ROOF_TOP - 5.2), (X + 0.15, (z0 + z1) / 2, 5.2 + (ROOF_TOP - 5.2) / 2), mat(bs, "cladding"))
    door_y = dh["door"][1]
    for k, (a, b) in enumerate(((z0, door_y - 1.6), (door_y + 1.6, z1))):
        bs.box(f"dh_front{k}", (0.25, b - a, 5.2), (X + 0.1, (a + b) / 2, 2.6), mat(bs, "ral7035"))
    bs.box("dh_door_frame", (0.3, 3.4, 0.25), (X + 0.1, door_y, 3.2), mat(bs, "safety_yellow"))
    # two roller doors on the west wall: transport aisle and dock aisle (horizontal slats)
    for k, (cy, w) in enumerate(((8.0, 4.5), (32.0, 4.5))):
        bs.box(f"rdoor{k}", (0.12, w, 4.8), (0.05, cy, 2.4), corrugated(bs, "rdoor", (0.55, 0.57, 0.6), axis="h", pitch=0.11))
        bs.box(f"rdoor{k}_box", (0.6, w + 0.4, 0.6), (0.3, cy, 5.1), mat(bs, "ral7035"))
        for sgn in (-1, 1):
            bs.box(f"rdoor{k}_g{sgn}", (0.2, 0.2, 4.8), (0.15, cy + sgn * (w / 2 + 0.1), 2.4), mat(bs, "hazard"))
    # columns
    for i in range(0, int(X / COL_DX) + 1):
        x = i * COL_DX
        for j, y in enumerate(COL_ROWS_Y):
            yy = min(max(y, 0.25), Y - 0.25)
            h_column(bs, f"col_{i}_{j}", x if 0 < x < X else min(max(x, 0.25), X - 0.25), yy, ROOF_EAVE,
                     matk="ral7035")             # light grey structure, as in most body shops
    roof(bs, X, Y)


def roof(bs, X, Y):
    st = mat(bs, "roof_steel")                    # roof structure painted light grey, as in most body shops
    # primary trusses along y at each column line: bottom + top chord + verticals/diagonals
    for i in range(0, int(X / COL_DX) + 1):
        x = min(max(i * COL_DX, 0.2), X - 0.2)
        bs.box(f"tr{i}_bot", (0.25, Y, 0.25), (x, Y / 2, ROOF_EAVE + 0.12), st)
        bs.box(f"tr{i}_top", (0.25, Y, 0.25), (x, Y / 2, ROOF_TOP - 0.3), st)
        n = 24
        for k in range(n + 1):
            y = k * Y / n
            bs.box(f"tr{i}_v{k}", (0.12, 0.12, ROOF_TOP - ROOF_EAVE - 0.4), (x, y, (ROOF_EAVE + ROOF_TOP - 0.3) / 2), st)
            if k < n:
                dy, dz = Y / n, ROOF_TOP - ROOF_EAVE - 0.4
                o = bs.box(f"tr{i}_d{k}", (0.1, math.hypot(dy, dz), 0.1), (x, y + dy / 2, (ROOF_EAVE + ROOF_TOP - 0.3) / 2), st)
                o.rotation_euler = (math.atan2(dz, dy) * (1 if k % 2 else -1), 0.0, 0.0)
    # purlins along x every 2 m
    for k in range(0, int(Y / 2) + 1):
        bs.box(f"purlin{k}", (X, 0.15, 0.3), (X / 2, k * 2.0, ROOF_TOP - 0.05), mat(bs, "galv"))
    # deck with skylight strips (2.4 m wide, every 12 m across, between trusses)
    deck_z = ROOF_TOP + 0.15
    sky_y = [6.0, 18.0, 30.0, 42.0]
    edges = [0.0]
    for y in sky_y:
        edges += [y - 1.2, y + 1.2]
    edges.append(Y)
    for k in range(0, len(edges), 2):
        a, b = edges[k], edges[k + 1]
        bs.box(f"deck{k}", (X, b - a, 0.1), (X / 2, (a + b) / 2, deck_z), mat(bs, "roof_deck"))
    for k, y in enumerate(sky_y):
        for i in range(int(X / COL_DX)):
            x0 = i * COL_DX
            bs.box(f"sky{k}_{i}", (COL_DX - 1.2, 2.4, 0.06), (x0 + COL_DX / 2, y, deck_z), skylight_mat(bs))
            bs.box(f"sky{k}_{i}_fr", (COL_DX, 2.6, 0.12), (x0 + COL_DX / 2, y, deck_z + 0.12), mat(bs, "galv")).hide_render = True
        for i in range(int(X / COL_DX)):        # mullions between panels
            bs.box(f"skym{k}_{i}", (0.6, 2.4, 0.1), (i * COL_DX, y, deck_z), mat(bs, "roof_deck"))
    # sprinkler mains (red) and branch lines under the deck
    red = mat(bs, "red_paint")
    for k, y in enumerate((3.0, 15.0, 27.0, 39.0, 45.0)):
        bs.cyl(f"spr_main{k}", 0.08, X, (X / 2, y, ROOF_TOP - 0.55), red, axis="x")


def high_bay_lights(bs, rng: random.Random, energy: float):
    """LED high-bay grid (UFO type) hung from the bottom chords: housing, cam-only lens, disk area light."""
    X, Y = bs.scn["hall"]["x"][1], bs.scn["hall"]["z"][1]
    housing = mat(bs, "dark_steel")
    lens = glow_cam_only(bs, "led_lens", (1.0, 0.97, 0.92), 40.0)
    z = 10.6
    for i in range(16):
        for j in range(8):
            x, y = 3.0 + i * 6.0, 3.0 + j * 6.0
            if (i * 7 + j * 3) % 37 == 0:                       # a couple of dead fittings, as in every plant
                e, lm = 0.0, glow_cam_only(bs, "led_dead", (0.2, 0.2, 0.2), 0.3)
            else:
                e, lm = energy * rng.uniform(0.93, 1.05), lens
            bs.cyl(f"hb_{i}_{j}_rod", 0.01, ROOF_EAVE - z, (x, y, (ROOF_EAVE + z) / 2), housing)
            bs.cyl(f"hb_{i}_{j}_body", 0.21, 0.14, (x, y, z + 0.07), housing)
            bs.cyl(f"hb_{i}_{j}_fins", 0.16, 0.08, (x, y, z + 0.18), mat(bs, "galv"))
            bs.no_light(bs.cyl(f"hb_{i}_{j}_lens", 0.18, 0.01, (x, y, z - 0.005), lm))
            if e > 0:
                li = bs.light(f"hb_{i}_{j}", "AREA", (x, y, z - 0.02), e, (1.0, 0.95, 0.88), 0.36)
                li.data.shape = "DISK"
                li.data.spread = math.radians(120)
                bs.lights_ceiling.append((li, e))


def floor_markings(bs):
    """Painted aisle edges from the YAML traffic zones, zebra crossings, walkway arrows, cell outlines."""
    s = bs.scn
    X = s["hall"]["x"][1]
    yel, wht, grn = mat(bs, "line_yellow"), mat(bs, "line_white"), mat(bs, "line_green")
    zones = {z["id"]: z["rect"] for z in s["zones"]}
    zc = 0.003
    for zid in ("Z_AISLE_N", "Z_AISLE_S", "Z_TRANSPORT", "Z_MARKET"):
        if zid not in zones:
            continue
        x0, y0, x1, y1 = zones[zid]
        for y in (y0, y1):
            if 0.0 < y < s["hall"]["z"][1]:
                bs.box(f"mk_{zid}_{y}", (x1 - x0, 0.1, 0.004), ((x0 + x1) / 2, y, zc), yel)
    # pedestrian aisles: green walkway band down the middle of N and S aisles, dashed white centreline in transport
    for zid in ("Z_AISLE_N", "Z_AISLE_S"):
        if zid in zones:
            x0, y0, x1, y1 = zones[zid]
            bs.box(f"walk_{zid}", (x1 - x0, 1.5, 0.003), ((x0 + x1) / 2, (y0 + y1) / 2, zc - 0.001), grn)
    if "Z_TRANSPORT" in zones:
        x0, y0, x1, y1 = zones["Z_TRANSPORT"]
        for k in range(int((x1 - x0) / 4)):
            bs.box(f"dash{k}", (2.0, 0.12, 0.004), (x0 + 1.5 + k * 4.0, (y0 + y1) / 2, zc), wht)
        for xc in (24.0, 48.0, 72.0):                         # zebra crossings over the transport aisle
            for k in range(8):
                bs.box(f"zeb{xc}_{k}", (2.5, 0.25, 0.004), (xc, y0 + 0.3 + k * 0.5, zc), wht)
    # outlines around every machine footprint (yellow, 0.1 m inside the sim boundary)
    for m in s["machines"]:
        w, d = m["size"]
        cx, cy = m["x"], m["z"]
        for sgn in (-1, 1):
            bs.box(f"cell_{m['id']}_h{sgn}", (w, 0.08, 0.004), (cx, cy + sgn * (d / 2 - 0.1), zc), yel)
            bs.box(f"cell_{m['id']}_v{sgn}", (0.08, d, 0.004), (cx + sgn * (w / 2 - 0.1), cy, zc), yel)


def overhead_services(bs, rng: random.Random):
    """Cable trays (with cable bundles) over both pedestrian aisles, spiral ducts, a busbar over the line."""
    X = bs.scn["hall"]["x"][1]
    galv, dark = mat(bs, "galv"), mat(bs, "dark_steel")
    for k, y in enumerate((14.5, 17.5, 30.5, 33.5)):
        z = 7.2 + 0.4 * (k % 2)
        bs.box(f"tray{k}_b", (X, 0.6, 0.02), (X / 2, y, z), mat(bs, "grating"))
        for s in (-1, 1):
            bs.box(f"tray{k}_s{s}", (X, 0.02, 0.12), (X / 2, y + s * 0.3, z + 0.06), galv)
        bs.box(f"tray{k}_cab", (X, 0.45, 0.07), (X / 2, y, z + 0.045), dark)
        for i in range(0, int(X), 3):                       # hangers
            bs.cyl(f"tray{k}_h{i}", 0.008, ROOF_EAVE - z, (i + 1.5, y + 0.32, (ROOF_EAVE + z) / 2), galv)
    for k, (y, r) in enumerate(((10.5, 0.45), (37.5, 0.4))):
        bs.cyl(f"duct{k}", r, X, (X / 2, y, 10.9), galv, axis="x")
        for i in range(6, int(X), 12):
            bs.cyl(f"duct{k}_out{i}", r * 0.6, 1.0, (i, y, 10.2), galv)
    bs.box("busbar", (X - 10, 0.25, 0.18), (X / 2, 21.0, 8.4), mat(bs, "ral7035"))


def clutter(bs, rng: random.Random):
    """Pallets with mesh containers and bins, carts, extinguishers, bollards, bins: avoids the sim's
    walkable zones' centre lines so no prop sits where traffic is drawn."""
    s = bs.scn
    X = s["hall"]["x"][1]
    wood, mesh_m, galv = mat(bs, "wood"), mat(bs, "mesh"), mat(bs, "galv")
    bins = [mat(bs, "bin_blue"), mat(bs, "bin_yellow")]
    # parts supermarket: racking bays along the south wall (Z_MARKET 0-5 m), loaded with containers
    for i in range(16):
        x = 3.0 + i * 5.8
        rack_bay(bs, f"mk{i}", x, 1.4, rng)
    # floor stock along the market edge (y 4.2..4.8 strip stays clear of the transport aisle at 6)
    for i in range(30):
        x = 2.0 + i * 3.1 + rng.uniform(-0.3, 0.3)
        if rng.random() < 0.65:
            pallet_load(bs, f"fs{i}", x, 4.35, rng, wood, mesh_m, galv, bins)
    # line-side material near every station's south side (between aisle S edge 18 and cell)
    for m in s["machines"]:
        if not m.get("on_line"):
            continue
        for k in range(2):
            pallet_load(bs, f"ls_{m['id']}_{k}", m["x"] - 2.5 + 5.0 * k, 13.2, rng, wood, mesh_m, galv, bins)
        part_cart(bs, f"pc_{m['id']}", m["x"], 13.2, rng)
    # extinguishers + signs on internal columns, bollards at the roller doors
    for i in range(0, int(X / COL_DX) + 1):
        x = i * COL_DX
        for y in (12.0, 35.5):
            if 0 < x < X:
                extinguisher(bs, f"ext_{i}_{y}", x + 0.3, y - 0.25)
    for k, cy in enumerate((8.0, 32.0)):
        for sgn in (-1, 1):
            bs.cyl(f"bollard{k}{sgn}", 0.08, 1.2, (1.0, cy + sgn * 2.8, 0.6), mat(bs, "hazard"))
    # wheelie bins / scrap skips near the closures cells
    for k, x in enumerate((33.0, 58.0, 62.0)):
        bs.box(f"skip{k}", (1.6, 1.0, 1.0), (x, 46.6, 0.5), mat(bs, "green_paint" if k % 2 else "steel_grey"))


def rack_bay(bs, name, x, y, rng: random.Random, w=5.4, d=1.1, h=5.5):
    """Pallet racking bay: blue uprights, orange beams, three load levels."""
    up = proc_mat(bs, "rack_blue", (0.05, 0.15, 0.45), rough=0.4, metal=0.3)
    beam = proc_mat(bs, "rack_orange", (0.85, 0.28, 0.02), rough=0.4, metal=0.3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            bs.box(f"{name}_u{sx}{sy}", (0.09, 0.07, h), (x + sx * w / 2, y + sy * d / 2, h / 2), up)
    for lv, z in enumerate((0.15, 1.8, 3.5)):
        for sy in (-1, 1):
            bs.box(f"{name}_b{lv}{sy}", (w, 0.05, 0.12), (x, y + sy * d / 2, z + 1.2 if lv else z + 0.0), beam)
        zz = z + (1.26 if lv else 0.06)
        for k in range(2):
            if rng.random() < 0.85:
                pallet_load(bs, f"{name}_l{lv}{k}", x - w / 4 + k * w / 2, y, rng, mat(bs, "wood"), mat(bs, "mesh"), mat(bs, "galv"),
                            [mat(bs, "bin_blue"), mat(bs, "bin_yellow")], z0=zz)


def pallet_load(bs, name, x, y, rng: random.Random, wood, mesh_m, galv, bins, z0=0.0):
    bs.box(name + "_pal", (1.2, 0.8, 0.144), (x, y, z0 + 0.072), wood)
    kind = rng.random()
    if kind < 0.45:                                          # wire mesh container ("gitterbox")
        hh = 0.97
        for sx in (-1, 1):
            bs.box(f"{name}_gs{sx}", (0.01, 0.8, hh), (x + sx * 0.6, y, z0 + 0.144 + hh / 2), mesh_m)
            bs.box(f"{name}_gf{sx}", (1.2, 0.01, hh), (x, y + sx * 0.4, z0 + 0.144 + hh / 2), mesh_m)
        for sx in (-1, 1):
            for sy in (-1, 1):
                bs.box(f"{name}_gp{sx}{sy}", (0.04, 0.04, hh), (x + sx * 0.6, y + sy * 0.4, z0 + 0.144 + hh / 2), galv)
        bs.box(name + "_gfill", (1.1, 0.7, hh * rng.uniform(0.3, 0.8)), (x, y, z0 + 0.144 + hh * 0.2), mat(bs, "steel_grey"))
    elif kind < 0.8:                                          # stacked small-load carriers
        bm = bins[int(rng.random() * len(bins))]
        for lv in range(rng.randint(2, 4)):
            for ix in range(2):
                for iy in range(2):
                    bs.box(f"{name}_c{lv}{ix}{iy}", (0.58, 0.38, 0.27), (x - 0.3 + ix * 0.6, y - 0.2 + iy * 0.4, z0 + 0.144 + 0.135 + lv * 0.28), bm)
    else:                                                     # cardboard, shrink-wrapped
        hh = rng.uniform(0.5, 1.2)
        bs.box(name + "_box", (1.15, 0.78, hh), (x, y, z0 + 0.144 + hh / 2), mat(bs, "cardboard"))


def part_cart(bs, name, x, y, rng: random.Random):
    """Wheeled closures rack at the line side: steel frame on four casters, a row of slotted door/hood panels
    standing on edge (a random number of slots filled)."""
    galv, rubber = mat(bs, "galv"), mat(bs, "rubber")
    steel = proc_mat(bs, "panel_steel", (0.6, 0.62, 0.64), rough=0.3, metal=0.9, var=0.08, scale=2.0, bump=0.03, grime=0.1)
    L, W, deck = 1.6, 0.9, 0.3
    bs.box(name + "_deck", (L, W, 0.05), (x, y, deck), galv)
    for sx in (-1, 1):
        for sy in (-1, 1):
            bs.cyl(f"{name}_w{sx}{sy}", 0.08, 0.06, (x + sx * (L / 2 - 0.12), y + sy * (W / 2 - 0.1), 0.08), rubber, axis="x")
            bs.box(f"{name}_up{sx}{sy}", (0.04, 0.04, 1.2), (x + sx * (L / 2 - 0.02), y + sy * (W / 2 - 0.02), deck + 0.6), galv)
        bs.box(f"{name}_top{sx}", (0.04, W, 0.04), (x + sx * (L / 2 - 0.02), y, deck + 1.2), galv)
    bs.box(name + "_handle", (L, 0.03, 0.03), (x, y - W / 2 - 0.05, deck + 0.95), galv)
    for k in range(9):                                        # slots every 0.16 m along x
        sx = x - L / 2 + 0.16 + k * 0.16
        if rng.random() < 0.75:
            p = bs.box(f"{name}_pan{k}", (0.012, 0.8, rng.uniform(0.75, 0.95)), (sx, y, deck + 0.45), steel)
            p.rotation_euler = (0.0, math.radians(rng.uniform(-4, 4)), 0.0)


def extinguisher(bs, name, x, y):
    bs.cyl(name + "_body", 0.08, 0.5, (x, y, 0.95), mat(bs, "red_paint"))
    bs.box(name + "_sign", (0.02, 0.3, 0.3), (x + 0.05, y, 2.2), glow_cam_only(bs, "sign_red", (0.8, 0.05, 0.03), 0.6))


# =====================================================================================
# dressing of the floor the sim never uses
# =====================================================================================
def blocked_rects(scn: dict, margin: float = 0.8) -> list[tuple[float, float, float, float]]:
    """Every rectangle the sim uses (traffic zones, machine footprints + margin, the conveyor strip)."""
    out = [tuple(z["rect"]) for z in scn["zones"]]
    for m in scn["machines"]:
        w, d = m["size"]
        out.append((m["x"] - w / 2 - margin, m["z"] - d / 2 - margin, m["x"] + w / 2 + margin, m["z"] + d / 2 + margin))
    xs = [m["x"] for m in scn["machines"] if m.get("on_line")]
    out.append((min(xs) - 7.5, 22.5, max(xs) + 7.5, 25.5))
    return out


def is_free(rects, x0, y0, x1, y1, hall) -> bool:
    if x0 < 0.6 or y0 < 0.6 or x1 > hall[0] - 0.6 or y1 > hall[1] - 0.6:
        return False
    for i in range(0, int(hall[0] / COL_DX) + 1):              # keep 0.5 m off the columns
        for cy in COL_ROWS_Y:
            cx = i * COL_DX
            if x0 - 0.5 < cx < x1 + 0.5 and y0 - 0.5 < cy < y1 + 0.5:
                return False
    return not any(x0 < r[2] and r[0] < x1 and y0 < r[3] and r[1] < y1 for r in rects)


def dressing(bs, rng: random.Random):
    """Fill the unused floor (outside all sim zones) so cells never sit on an empty slab: panel storage
    racks, a maintenance corner, a shop-floor office, stacked empty skids, sealer drums."""
    s = bs.scn
    hall = (s["hall"]["x"][1], s["hall"]["z"][1])
    rects = blocked_rects(s)
    placed: list[tuple[float, float, float, float]] = []
    kinds = [("office", 6.0, 2.5, 1), ("skids", 5.2, 1.4, 4), ("maint", 3.2, 1.6, 3), ("drums", 1.3, 1.3, 6), ("panels", 2.6, 1.3, 40)]
    n = 0
    for kind, w, d, limit in kinds:
        got = 0
        for yi in range(int(hall[1] * 2), 0, -1):               # north first: the closures side is the emptiest
            for xi in range(int(hall[0] * 2)):
                if got >= limit:
                    break
                x, y = xi * 0.5 + w / 2, yi * 0.5 - d / 2
                box = (x - w / 2 - 0.4, y - d / 2 - 0.4, x + w / 2 + 0.4, y + d / 2 + 0.4)
                if not is_free(rects + placed, *box, hall):
                    continue
                placed.append(box)
                n += 1
                got += 1
                {"office": office, "skids": skid_stack, "maint": maintenance, "drums": drums, "panels": panel_rack}[kind](bs, f"dr{n}", x, y, rng)
    return n


def panel_rack(bs, name, x, y, rng: random.Random):
    """A-frame steel rack holding pressed door/hood panels on edge."""
    st, biw = mat(bs, "steel_grey"), proc_mat(bs, "biw_steel", (0.6, 0.62, 0.64), rough=0.32, metal=0.85, var=0.1, scale=3.0, bump=0.05, grime=0.15)
    bs.box(name + "_base", (2.4, 1.2, 0.15), (x, y, 0.15), st)
    for sx in (-1, 1):
        bs.box(name + f"_post{sx}", (0.08, 0.08, 1.9), (x + sx * 1.15, y, 1.1), st)
    bs.box(name + "_top", (2.4, 0.08, 0.08), (x, y, 2.05), st)
    for k in range(rng.randint(3, 8)):
        o = bs.box(name + f"_p{k}", (1.3, 0.025, 1.05), (x - 0.9 + k * 0.25, y, 0.85), biw)
        o.rotation_euler = (0.0, 0.0, math.radians(90 + rng.uniform(-6, 6)))


def office(bs, name, x, y, rng: random.Random):
    """Shop-floor office container: ribbed walls, a window band lit from inside, a door."""
    bs.box(name + "_body", (6.0, 2.5, 2.8), (x, y, 1.4), corrugated(bs, "container", (0.62, 0.64, 0.62), axis="z", pitch=0.3, metal=0.4))
    bs.box(name + "_win", (4.0, 2.52, 0.8), (x - 0.6, y, 1.6), glow_cam_only(bs, "office_win", (0.75, 0.82, 0.9), 2.5))
    bs.box(name + "_door", (0.9, 2.52, 2.0), (x + 2.4, y, 1.0), mat(bs, "steel_grey"))
    bs.box(name + "_roof", (6.1, 2.6, 0.08), (x, y, 2.84), mat(bs, "galv"))
    bs.box(name + "_sign", (1.2, 0.02, 0.3), (x - 0.6, y - 1.27, 2.45), glow_cam_only(bs, "sign_green", (0.05, 0.6, 0.2), 0.8))


def skid_stack(bs, name, x, y, rng: random.Random):
    """Empty BIW skids (two runners + cross members) stacked four high."""
    st = mat(bs, "dark_steel")
    for lv in range(rng.randint(2, 5)):
        z = 0.12 + lv * 0.3
        for s in (-1, 1):
            bs.box(f"{name}_r{lv}{s}", (5.0, 0.12, 0.16), (x, y + s * 0.5, z), st)
        for k in range(4):
            bs.box(f"{name}_c{lv}{k}", (0.1, 1.2, 0.1), (x - 2.0 + k * 1.33, y, z + 0.1), st)


def maintenance(bs, name, x, y, rng: random.Random):
    """Workbench with vise and parts, a red tool trolley, a tall grey cabinet."""
    wood, st, red = mat(bs, "wood"), mat(bs, "steel_grey"), mat(bs, "red_paint")
    bs.box(name + "_top", (2.0, 0.8, 0.05), (x - 0.5, y, 0.9), wood)
    for sx in (-1, 1):
        bs.box(name + f"_leg{sx}", (0.06, 0.7, 0.88), (x - 0.5 + sx * 0.95, y, 0.44), st)
    bs.box(name + "_vise", (0.25, 0.15, 0.15), (x - 1.3, y, 1.0), mat(bs, "dark_steel"))
    for k in range(3):
        bs.box(name + f"_part{k}", (rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), rng.uniform(0.05, 0.2)), (x - 0.6 + k * 0.35, y + 0.1, 1.0), mat(bs, "galv"))
    bs.box(name + "_trolley", (0.7, 0.5, 1.0), (x + 0.95, y + 0.1, 0.55), red)
    bs.box(name + "_cab", (1.0, 0.5, 1.95), (x + 1.0, y - 0.5, 0.98), mat(bs, "ral7035"))


def drums(bs, name, x, y, rng: random.Random):
    """Sealer / adhesive drums on a pallet (four 200 l drums)."""
    bs.box(name + "_pal", (1.2, 1.2, 0.144), (x, y, 0.072), mat(bs, "wood"))
    col = proc_mat(bs, "drum_blue", (0.04, 0.12, 0.4), rough=0.4, metal=0.5, var=0.1, bump=0.05)
    for ix in (-1, 1):
        for iy in (-1, 1):
            if rng.random() < 0.9:
                bs.cyl(f"{name}_d{ix}{iy}", 0.29, 0.88, (x + ix * 0.3, y + iy * 0.3, 0.144 + 0.44), col)


def build(bs, light_energy: float, seed: int = 11):
    rng = random.Random(seed)
    building(bs, rng)
    high_bay_lights(bs, rng, light_energy)
    floor_markings(bs)
    overhead_services(bs, rng)
    clutter(bs, rng)
    dressing(bs, rng)
