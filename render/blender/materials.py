"""CC0 PBR materials (Poly Haven + ambientCG) for the Blender body-shop scene.

    import materials as M
    mat = M.pbr("my_floor", "concrete_floor_worn_001", res="2k", scale=4.0)   # Poly Haven id
    mat = M.pbr("my_rubber", "Rubber004", source="ambientcg")                # ambientCG id
    mat = M.plant("floor_epoxy")                                             # curated, see PLANT
    M.used_credits()                                                          # [{asset, owner, licence, ...}]

Mapping: BOX (triplanar) projection on OBJECT coordinates by default, so imported meshes with transforms applied
look right at real-world scale (`scale` = metres per texture tile). Scaled unit cubes (the static plant boxes) pass
coords="world" instead: object coordinates ignore the object's scale and would stretch one tile over the whole box. Tangent
normal maps need UVs, so detail comes from the displacement (height) map through a Bump node instead.
Downloads are cached in D:\\icdata\\assets\\polyhaven\\<id>\\ and D:\\icdata\\assets\\ambientcg\\<id>\\ (only image
files are extracted from ambientCG zips; nothing downloaded is ever executed).
Licences: Poly Haven https://polyhaven.com/license (CC0), ambientCG https://docs.ambientcg.com/license/ (CC0).
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import bpy

PH_DIR = Path(r"D:\icdata\assets\polyhaven")
ACG_DIR = Path(r"D:\icdata\assets\ambientcg")
UA = {"User-Agent": "intercycle-render/1.0"}
_USED: dict[str, dict] = {}


# ----------------------------------------------------------------------------------------------- download
def _get(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    return dst


def _json(url: str, cache: Path) -> dict:
    if not cache.exists():
        _get(url, cache)
    return json.loads(cache.read_text(encoding="utf-8"))


PH_MAPS = {"color": "Diffuse", "rough": "Rough", "ao": "AO", "disp": "Displacement", "metal": "Metal"}


def fetch_polyhaven(asset_id: str, res: str = "2k") -> dict:
    """{map: path} for a Poly Haven texture + 'tile_m' (real-world width of one tile, from the asset info)."""
    d = PH_DIR / asset_id
    files = _json(f"https://api.polyhaven.com/files/{asset_id}", d / "files.json")
    info = _json(f"https://api.polyhaven.com/info/{asset_id}", d / "info.json")
    out = {}
    for key, ph in PH_MAPS.items():
        if ph in files and res in files[ph]:
            fmt = "jpg" if "jpg" in files[ph][res] else next(iter(files[ph][res]))
            url = files[ph][res][fmt]["url"]
            out[key] = _get(url, d / Path(url).name)
    dims = info.get("dimensions") or [2000, 2000]
    out["tile_m"] = dims[0] / 1000.0
    _USED[asset_id] = dict(asset=asset_id, owner=", ".join(info.get("authors", {}) or ["Poly Haven"]),
                           licence="CC0-1.0", source_url=f"https://polyhaven.com/a/{asset_id}",
                           licence_url="https://polyhaven.com/license", kind="PBR texture (Poly Haven)")
    return out


ACG_SUFFIX = {"color": "_Color", "rough": "_Roughness", "ao": "_AmbientOcclusion", "disp": "_Displacement",
              "metal": "_Metalness"}


def fetch_ambientcg(asset_id: str, res: str = "2k") -> dict:
    """{map: path} for an ambientCG material; downloads the <res>-JPG zip once, extracts only the image maps."""
    d = ACG_DIR / asset_id
    tag = f"{res.upper()}-JPG"
    marker = d / f".{tag}.done"
    if not marker.exists():
        z = _get(f"https://ambientcg.com/get?file={asset_id}_{tag}.zip", d / f"{asset_id}_{tag}.zip")
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                name = Path(m.filename).name                       # basename only: no path traversal
                if name.lower().endswith((".jpg", ".png")) and any(s in name for s in ACG_SUFFIX.values()):
                    (d / name).write_bytes(zf.read(m))
        z.unlink()
        marker.write_text("ok")
    out = {}
    for key, suf in ACG_SUFFIX.items():
        hit = sorted(d.glob(f"{asset_id}_{tag}{suf}.*"))
        if hit:
            out[key] = hit[0]
    out["tile_m"] = 2.0
    _USED[asset_id] = dict(asset=asset_id, owner="ambientCG (Lennart Demes)", licence="CC0-1.0",
                           source_url=f"https://ambientcg.com/view?id={asset_id}",
                           licence_url="https://docs.ambientcg.com/license/", kind="PBR texture (ambientCG)")
    return out


def world_hdri(asset_id: str = "autoshop_01", res: str = "1k", strength: float = 1.0, scene=None):
    """Poly Haven HDRI as the world background (cached in PH_DIR/<id>/)."""
    d = PH_DIR / asset_id
    files = _json(f"https://api.polyhaven.com/files/{asset_id}", d / "files.json")
    info = _json(f"https://api.polyhaven.com/info/{asset_id}", d / "info.json")
    url = files["hdri"][res]["hdr"]["url"]
    path = _get(url, d / Path(url).name)
    sc = scene or bpy.context.scene
    w = sc.world or bpy.data.worlds.new("world")
    sc.world = w
    if hasattr(w, "use_nodes") and not w.use_nodes:
        w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(path), check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    _USED[asset_id] = dict(asset=asset_id, owner=", ".join(info.get("authors", {}) or ["Poly Haven"]),
                           licence="CC0-1.0", source_url=f"https://polyhaven.com/a/{asset_id}",
                           licence_url="https://polyhaven.com/license", kind="HDRI (Poly Haven)")
    return w


def used_credits() -> list[dict]:
    return list(_USED.values())


# ----------------------------------------------------------------------------------------------- node helpers
def _mix(nt, blend: str, fac, a, b):
    """ShaderNodeMix in RGBA mode; a/b are sockets or RGBA tuples. Returns the Result socket."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    ins = [s for s in n.inputs if s.type == "RGBA"]
    n.inputs["Factor"].default_value = fac if not hasattr(fac, "links") else 0.5
    if hasattr(fac, "links"):
        nt.links.new(fac, n.inputs["Factor"])
    for sock, v in zip(ins[:2], (a, b)):
        if hasattr(v, "links"):
            nt.links.new(v, sock)
        else:
            sock.default_value = (*v[:3], 1.0)
    return next(s for s in n.outputs if s.type == "RGBA")


def _math(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if hasattr(v, "links"):
            nt.links.new(v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n.outputs[0]


def _img(nt, path, vec, non_color: bool, blend: float = 0.25):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        n.image.colorspace_settings.name = "Non-Color"
    n.projection = "BOX"
    n.projection_blend = blend
    nt.links.new(vec, n.inputs["Vector"])
    return n


def _new_mat(name: str):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m, None, None
    m = bpy.data.materials.new(name)
    if hasattr(m, "use_nodes") and not m.use_nodes:          # 5.x: node tree always present; keep 4.x working
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bsdf


def _coord(nt, coords: str):
    """Texture-space position: "object" (follows the object; size of the mesh before its scale) or "world"
    (Geometry Position: real metres on static, scaled unit cubes such as the plant boxes)."""
    if coords == "world":
        return nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"]
    return nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def _vector(nt, tile_m: float, coords: str = "object"):
    mp = nt.nodes.new("ShaderNodeMapping")
    s = 1.0 / max(tile_m, 1e-3)
    mp.inputs["Scale"].default_value = (s, s, s)
    nt.links.new(_coord(nt, coords), mp.inputs["Vector"])
    return mp.outputs["Vector"]


# ----------------------------------------------------------------------------------------------- pbr
def pbr(name: str, asset_id: str, res: str = "2k", scale: float | None = None, source: str = "auto",
        tint=None, tint_mix: float = 1.0, desaturate: bool = False, color_override=None,
        rough_mul: float = 1.0, rough_add: float = 0.0, metallic=None, bump: float = 0.3, ao: float = 0.6,
        coat: float = 0.0, coat_rough: float = 0.1, coords: str = "object"):
    """Principled material from a CC0 texture set.

    scale: metres per tile (default: Poly Haven real-world size, 2 m for ambientCG).
    tint / tint_mix: multiply the albedo by an RGB (linear) colour; desaturate first if `desaturate`.
    color_override: RGB (linear) paint colour; the texture then only drives roughness/bump/AO (paint over metal).
    metallic: None = use the metal map if any, else 0; a float forces it.
    coords: "object" (default) or "world" (static scaled boxes; see _coord).
    """
    m, nt, bsdf = _new_mat(name)
    if nt is None:
        return m
    src = source if source != "auto" else ("ambientcg" if asset_id[:1].isupper() else "polyhaven")
    maps = fetch_polyhaven(asset_id, res) if src == "polyhaven" else fetch_ambientcg(asset_id, res)
    vec = _vector(nt, scale or maps["tile_m"], coords)
    col = None
    if "color" in maps:
        col = _img(nt, maps["color"], vec, False).outputs["Color"]
        if desaturate:
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            hsv.inputs["Saturation"].default_value = 0.0
            nt.links.new(col, hsv.inputs["Color"])
            col = hsv.outputs["Color"]
        if tint is not None:
            col = _mix(nt, "MULTIPLY", tint_mix, col, tint)
    if color_override is not None:
        col = None
        bsdf.inputs["Base Color"].default_value = (*color_override[:3], 1.0)
    if "ao" in maps and ao > 0:
        a = _img(nt, maps["ao"], vec, True).outputs["Color"]
        col = _mix(nt, "MULTIPLY", ao, col if col is not None else (*bsdf.inputs["Base Color"].default_value[:3],), a)
    if col is not None:
        nt.links.new(col, bsdf.inputs["Base Color"])
    if "rough" in maps:
        r = _img(nt, maps["rough"], vec, True).outputs["Color"]
        r = _math(nt, "MULTIPLY_ADD", r, rough_mul) if rough_mul != 1.0 or rough_add else r
        if rough_mul != 1.0 or rough_add:
            r.node.inputs[2].default_value = rough_add
            r.node.use_clamp = True
        nt.links.new(r, bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = min(1.0, 0.5 * rough_mul + rough_add)
    if metallic is not None:
        bsdf.inputs["Metallic"].default_value = metallic
    elif "metal" in maps:
        nt.links.new(_img(nt, maps["metal"], vec, True).outputs["Color"], bsdf.inputs["Metallic"])
    if "disp" in maps and bump > 0:
        h = _img(nt, maps["disp"], vec, True).outputs["Color"]
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = 0.002 * (scale or maps["tile_m"])
        nt.links.new(h, b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    if coat > 0:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = coat_rough
    m["pbr_source"] = f"{src}:{asset_id}"
    return m


# ----------------------------------------------------------------------------------------------- procedural extras
def _hazard_stripes(m, period_m: float = 0.2, coords: str = "object"):
    """Overlay 45-degree yellow/black hazard stripes (object or world space) on a paint material's base colour."""
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(_coord(nt, coords), sep.inputs["Vector"])
    s = _math(nt, "ADD", sep.outputs["X"], sep.outputs["Z"])
    s = _math(nt, "ADD", s, sep.outputs["Y"])
    s = _math(nt, "PINGPONG", s, period_m)
    s = _math(nt, "GREATER_THAN", s, period_m / 2)
    yellow = (0.82, 0.52, 0.0)
    nt.links.new(_mix(nt, "MIX", s, yellow, (0.015, 0.015, 0.015)), bsdf.inputs["Base Color"])


def _galv_spangle(m, cell_m: float = 0.02, coords: str = "object"):
    """Hot-dip galvanising spangle: Voronoi cells nudge colour and roughness."""
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 1.0 / cell_m
    nt.links.new(_coord(nt, coords), vor.inputs["Vector"])
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(vor.outputs["Color"], bw.inputs["Color"])
    r = _math(nt, "MULTIPLY_ADD", bw.outputs["Val"], 0.12)
    r.node.inputs[2].default_value = 0.28
    nt.links.new(r, bsdf.inputs["Roughness"])
    c = _mix(nt, "MIX", bw.outputs["Val"], (0.56, 0.58, 0.60), (0.64, 0.65, 0.66))
    nt.links.new(c, bsdf.inputs["Base Color"])


# ----------------------------------------------------------------------------------------------- curated plant set
def _srgb(r, g, b):
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return (f(r / 255), f(g / 255), f(b / 255))


SAFETY_YELLOW = _srgb(240, 168, 0)        # RAL 1003 signal yellow
RAL7035 = _srgb(197, 199, 196)            # light grey (machine guards, cabinets)
MACHINE_GREY = _srgb(96, 102, 105)        # RAL 7043-ish machine body
CLADDING_GREY = _srgb(196, 200, 201)        # light grey liner sheet (RAL 9002-ish), as in most halls
BIW_STEEL = (0.42, 0.43, 0.45)            # linear F0 of cold-rolled steel, lightly oiled (not chrome)

# key -> pbr() kwargs (+ "post": procedural overlay). Painted metals take roughness/scratch bump from
# ambientCG PaintedMetal004 under a solid paint colour.
PLANT: dict[str, dict] = {
    "floor_epoxy": dict(asset_id="concrete_floor_worn_001", scale=4.0, tint=_srgb(178, 182, 178), tint_mix=0.6,
                        desaturate=True, rough_mul=0.6, rough_add=0.18, bump=0.15, coat=0.35, coat_rough=0.3),
    "floor_concrete": dict(asset_id="concrete_floor_worn_001", scale=4.0, bump=0.3),
    "safety_yellow": dict(asset_id="PaintedMetal004", color_override=SAFETY_YELLOW, metallic=0.0, rough_mul=0.6,
                          rough_add=0.1, bump=0.08, scale=1.5),
    "painted_grey": dict(asset_id="PaintedMetal004", color_override=RAL7035, metallic=0.0, rough_mul=0.6,
                         rough_add=0.12, bump=0.08, scale=1.5),
    "machine_grey": dict(asset_id="PaintedMetal004", color_override=MACHINE_GREY, metallic=0.0, rough_mul=0.6,
                         rough_add=0.12, bump=0.08, scale=1.5),
    "guard_hazard": dict(asset_id="PaintedMetal004", color_override=SAFETY_YELLOW, metallic=0.0, rough_mul=0.6,
                         rough_add=0.1, bump=0.08, scale=1.5, post="hazard"),
    "worn_paint": dict(asset_id="PaintedMetal004", scale=1.0, bump=0.3),
    "galvanized": dict(asset_id="Metal032", metallic=1.0, rough_mul=0.5, rough_add=0.15, bump=0.05, post="galv"),
    "biw_steel": dict(asset_id="Metal032", color_override=BIW_STEEL, metallic=1.0,
                      rough_mul=0.3, rough_add=0.30, bump=0.03, scale=1.0, ao=0.0),
    "rubber_black": dict(asset_id="Rubber004", rough_mul=1.0, bump=0.2, scale=0.5),
    "plastic_black": dict(asset_id="Plastic013A", tint=(0.02, 0.02, 0.02), rough_mul=0.8, rough_add=0.15, scale=0.5),
    "plastic_white": dict(asset_id="Plastic013A", scale=0.5),
    "cardboard": dict(asset_id="Cardboard004", scale=0.6, bump=0.2),
    "wood_pallet": dict(asset_id="raw_plank_wall", scale=1.2, bump=0.3),
    # pre-painted (polyester-coated) steel sheet: a dielectric paint, so metallic 0; the scan gives ribs (bump) + AO
    "cladding": dict(asset_id="box_profile_metal_sheet", color_override=CLADDING_GREY, metallic=0.0,
                     rough_mul=0.8, bump=0.6),
}


def plant(key: str, res: str = "2k", coords: str = "object"):
    """Curated PLANT material by key (built once per .blend, named 'PLANT_<key>', '_w' suffix for world coords)."""
    spec = dict(PLANT[key])
    post = spec.pop("post", None)
    name = f"PLANT_{key}" + ("_w" if coords == "world" else "")
    fresh = bpy.data.materials.get(name) is None
    m = pbr(name, res=res, coords=coords, **spec)
    if fresh and post == "hazard":
        _hazard_stripes(m, coords=coords)
    elif fresh and post == "galv":
        _galv_spangle(m, coords=coords)
    return m


def preview_all(res: str = "1k") -> list:
    """Build every PLANT material (used by the library build and the material test render)."""
    return [plant(k, res) for k in PLANT]
