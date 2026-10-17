"""NVIDIA SimReady (CC-BY-4.0) prop library for the Blender body-shop scene.

Three jobs, one file:

1. fetch   (plain Python, stdlib)  download the chosen asset folders from the public NVIDIA bucket,
           skipping ``source_assets/`` (MDL sources + duplicate textures, never referenced by the USD).
               python render/blender/assets.py fetch [name ...]
2. build   (inside Blender)        import every USD, normalise (metres, Z-up, origin at floor centre,
           transforms applied), swap MDL materials for materials.py PBR ones, assemble the body-in-white,
           save D:\\icdata\\assets\\library\\nvidia_props.blend (one collection per asset + "BIW").
               blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/assets.py build
3. test    (inside Blender)        one CPU proof render per asset into D:\\icdata\\renders\\asset_tests\\<name>.png
               blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/assets.py test [name ...]

Scene API (inside Blender):
    import assets
    assets.load_library()                          # links every asset collection from the library .blend
    obj = assets.instance("forklift_blue", (12.0, 30.0, 0.0), yaw_deg=90)
    assets.ASSETS["forklift_blue"]["size"]         # real-world (x, y, z) extent in metres, measured at build time

Licence: every asset here ships a CC-BY-4.0 LICENSE.txt next to it (checked by ``fetch``; recorded in
D:\\icdata\\assets\\nvidia\\manifest.json and render/credits.json). Assets are never committed to git.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NV_ROOT = Path(r"D:\icdata\assets\nvidia\6.0")
LIBRARY = Path(r"D:\icdata\assets\library\nvidia_props.blend")
TEST_DIR = Path(r"D:\icdata\renders\asset_tests")
CREDITS = HERE.parent / "credits.json"
BUCKET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
CC_BY = "CC-BY-4.0"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
OWNER = "NVIDIA Corporation (Isaac Sim / SimReady assets)"

_WH = "SimReady/Industrial/Warehouse/"
_BP = "SimReady/Industrial/Machines/Automotive/Body_Parts/"


def _a(folder, usd, kind, used_for, mats=None, textures=True):
    return dict(folder=folder, usd=usd, kind=kind, used_for=used_for, mats=mats or {}, textures=textures,
                size=None, licence=CC_BY, owner=OWNER, source_url=BUCKET_ROOT + folder + "/" + usd,
                credit=f"{folder.rsplit('/', 1)[-1]} by NVIDIA Corporation, {CC_BY} ({CC_BY_URL})")


# name -> spec. `mats`: substring of the imported material name (lower case) -> materials.PLANT key;
# "*" is the fallback for anything unmatched; a value of None keeps the imported (USD preview) material.
ASSETS: dict[str, dict] = {
    "forklift_blue": _a(_WH + "Equipment/Vehicle_Forklift_Blue_A01", "sm_vehicle_forklift_blue_a01_01.usd", "vehicle",
                        "forklift traffic", {"*": None}),
    "forklift_red": _a(_WH + "Equipment/Vehicle_Forklift_Red_B01", "sm_vehicle_forklift_red_b01_01.usd", "vehicle",
                       "forklift traffic", {"*": None}),
    "forklift_c": _a(_WH + "Equipment/Vehicle_Forklift_Blue_C01", "sm_vehicle_forklift_blue_c01_01.usd", "vehicle",
                     "forklift traffic", {"*": None}),
    "conveyor": _a(_WH + "Equipment/Conveyor_Belt_A35", "sm_conveyor_belt_a35_01.usd", "conveyor",
                   "NOT USABLE AS IS: imports with the roller bed standing vertical (A02 is broken too); "
                   "keep the procedural conveyor", {"*": None}),
    "machine_guard": _a(_WH + "Barriers/Machine_Guard_A01", "sm_machineGuard_a01_01.usd", "guard",
                        "solid machine guard panels", {"painted_yellow": "safety_yellow", "*": None}),
    "wire_guard_a": _a(_WH + "Barriers/Wire_Machine_Guard_A01", "sm_wire_machineGuard_a01_01.usd", "guard",
                       "wire-mesh robot cell fencing", {"painted_yellow": "safety_yellow", "*": None}),
    "wire_guard_b": _a(_WH + "Barriers/Wire_Machine_Guard_A02", "sm_wire_machineGuard_a02_01.usd", "guard",
                       "wire-mesh robot cell fencing", {"painted_yellow": "safety_yellow", "*": None}),
    "fence_metal": _a(_WH + "Barriers/Barrier_Fence_Metal_A02", "sm_barrier_fence_metal_a02_01.usd", "guard",
                      "walkway fencing", {"*": "safety_yellow"}),
    "safety_gate": _a(_WH + "Barriers/Barrier_Gate_Safety_A02", "sm_barrier_gate_safety_a02_01.usd", "guard",
                      "cell access gate", {"painted_yellow": "safety_yellow", "painted_unique_gate": "safety_yellow",
                                           "*": None}),
    "rail_guard": _a(_WH + "Barriers/Rail_Guard_A09", "sm_rail_guard_a09_01.usd", "guard",
                     "floor rail / bollard guard", {"painted_yellow": "safety_yellow", "*": None}),
    "rack_3m": _a(_WH + "Racks/3m_S01_Rack", "sm_rack_h3m_s01_01.usd", "rack", "pallet racking", {"*": None}),
    "rack_large": _a(_WH + "Racks/Large_Empty_Rack_A01", "sm_rack_large_empty_a01_01.usd", "rack",
                     "pallet racking", {"*": None}),
    "shelf_heavy": _a(_WH + "Racks/Heavy_Duty_Steel_Shelve_A06", "sm_shelve_steel_heavyDuty_a06_01.usd", "rack",
                      "parts shelving", {"*": None}),
    "column": _a(_WH + "Structures/Wall_Column_A01", "sm_structure_wall_column_a01_01.usd", "structure",
                 "building columns", {"*": "painted_grey"}, textures=False),
    "beam": _a(_WH + "Structures/Structure_Beam_A01", "sm_structure_beam_a01_01.usd", "structure",
               "roof beams", {"*": "painted_grey"}, textures=False),
    "sign_warning": _a(_WH + "Signs/Warning_Sign_A04", "sm_sign_warning_a04_01.usd", "sign", "safety signage",
                       {"*": None}),
    "sign_floor": _a(_WH + "Signs/Movable_Floor_Sign_A01", "sm_sign_floor_movable_a01_01.usd", "sign",
                     "floor signage", {"*": None}),
    "pallet_wood": _a(_WH + "Pallets/Pallet_A01", "sm_pallet_a01_01.usd", "pallet", "pallets", {"*": None}),
    "pallet_block": _a(_WH + "Pallets/Block_Pallet_B02_01", "sm_pallet_block_b02_01.usd", "pallet", "pallets",
                       {"*": None}),
    "pallet_steel": _a(_WH + "Pallets/Steel_Galvanized_Pallet_A01_01", "sm_pallet_steel_galvanized_a01_01.usd",
                       "pallet", "parts pallets", {"*": None}),
    # body-in-white sources (bare steel everywhere)
    "car_frame": _a("SimReady/Industrial/Machines/Automotive/Frames/Car_Frame_Car_Frame_Assembled_A01",
                    "sm_carFrame_car_frame_assembled_a01_01.usd", "biw", "body-in-white frame", {"*": "biw_steel"}),
    "door_front": _a(_BP + "Door_Front_A01", "sm_bodyPart_door_front_a01_01.usd", "biw", "body-in-white panel",
                     {"*": "biw_steel"}),
    "door_rear": _a(_BP + "Door_Rear_A01", "sm_bodyPart_door_rear_a01_01.usd", "biw", "body-in-white panel",
                    {"*": "biw_steel"}),
    "hood": _a(_BP + "Body_Part_Hood_D01", "sm_bodyPart_hood_d01_01.usd", "biw", "body-in-white panel",
               {"*": "biw_steel"}),
    "fender_front": _a(_BP + "Body_Part_Fender_Front_D01", "sm_bodyPart_fender_front_d01_01.usd", "biw",
                       "body-in-white panel", {"*": "biw_steel"}),
    "fender_rear": _a(_BP + "Fender_Rear_A01", "sm_bodyPart_fender_rear_a01_01.usd", "biw", "body-in-white panel",
                      {"*": "biw_steel"}),
    "roof_panel": _a(_BP + "Body_Part_Exterior_Roof_Panel_A01", "sm_bodyPart_exterior_roof_panel_a01_01.usd", "biw",
                     "body-in-white panel", {"*": "biw_steel"}),
    "bumper_front": _a(_BP + "Front_Bumper_D01", "sm_bodyPart_bumper_front_d01_01.usd", "biw",
                       "body-in-white panel", {"*": "biw_steel"}),
    "bumper_rear": _a(_BP + "Rear_Bumper_C01", "sm_bodyPart_bumper_rear_c01_01.usd", "biw", "body-in-white panel",
                      {"*": "biw_steel"}),
    "boot_lid": _a(_BP + "Body_Part_Boot_Lid_C01", "sm_bodyPart_boot_lid_c01_01.usd", "biw", "body-in-white panel",
                   {"*": "biw_steel"}),
}
SIZES_JSON = LIBRARY.with_suffix(".sizes.json")       # written by build: measured extents per collection


# ----------------------------------------------------------------------------------------------- fetch
def fetch(names=None, workers: int = 8) -> None:
    """Parallel download of the chosen folders (no source_assets/, no Textures/ when textures=False)."""
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(HERE))
    import fetch_assets as fa
    base = fa.ROOTS["6.0"]
    jobs, lic = [], {}
    for n in names or ASSETS:
        spec = ASSETS[n]
        keys = [(k, s) for k, s in fa.list_keys(base + spec["folder"] + "/")
                if not any(x in k for x in fa.SKIP) and "/source_assets/" not in k
                and (spec["textures"] or "/Textures/" not in k)]
        if not any(Path(k).name.lower() in fa.LICENCE_NAMES for k, _ in keys):
            print(f"  WARNING {n}: no licence file, skipped", file=sys.stderr)
            continue
        print(f"{n}: {len(keys)} files {sum(s for _, s in keys) / 1e6:.1f} MB", flush=True)
        jobs += keys
    dest_root = fa.DEFAULT_DEST / "6.0"
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda ks: fa.download(ks[0], ks[1], dest_root, base), jobs))
    for p in paths:
        if p.name.lower() in fa.LICENCE_NAMES:
            first = next((ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                          if ln.strip()), "")
            lic[str(p.parent.relative_to(dest_root)).replace("\\", "/")] = {"licence_file": p.name,
                                                                            "first_line": first[:200]}
    bad = {k: v for k, v in lic.items() if "Attribution 4.0" not in v["first_line"]}
    if bad:
        print("  WARNING licence is not CC-BY-4.0:", bad, file=sys.stderr)
    fa.update_manifest(fa.DEFAULT_DEST, "6.0", "assets.py:" + ",".join(names or ["all"]), lic)
    print(f"done: {len(jobs)} files, {len(lic)} licences (all CC-BY-4.0: {not bad})")


# ----------------------------------------------------------------------------------------------- scene API
def _bpy():
    import bpy  # noqa: PLC0415  (only available inside Blender)
    return bpy


def load_library(path: Path | str = LIBRARY, names=None, link: bool = True) -> list:
    """Link (or append) asset collections from the library .blend; returns the collections.
    Fills ASSETS[name]["size"] from the sizes json written by build."""
    bpy = _bpy()
    path = Path(path)
    if SIZES_JSON.exists():
        for k, v in json.loads(SIZES_JSON.read_text(encoding="utf-8")).items():
            if k in ASSETS:
                ASSETS[k]["size"] = tuple(v)
    have = {c.name for c in bpy.data.collections}
    with bpy.data.libraries.load(str(path), link=link) as (src, dst):
        avail = list(src.collections)
        dst.collections = [c for c in avail if (names is None or c in names) and c not in have]
    want = [c for c in avail if names is None or c in names]
    return [bpy.data.collections[c] for c in want if bpy.data.collections.get(c) is not None]


def instance(name: str, loc, yaw_deg: float = 0.0, scale: float = 1.0, collection=None):
    """Collection-instance empty of asset `name` at loc (Blender X/Y/Z metres), yaw about Z in degrees."""
    bpy = _bpy()
    src = bpy.data.collections.get(name)
    if src is None:
        load_library(names=[name])
        src = bpy.data.collections[name]
    ob = bpy.data.objects.new(f"{name}.inst", None)
    ob.instance_type = "COLLECTION"
    ob.instance_collection = src
    ob.location = tuple(loc) if len(loc) == 3 else (loc[0], loc[1], 0.0)
    ob.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
    ob.scale = (scale, scale, scale)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


# ----------------------------------------------------------------------------------------------- build
def _usd_path(name: str) -> Path:
    s = ASSETS[name]
    return NV_ROOT / s["folder"] / s["usd"]


def _bbox(meshes):
    """World-space bounds from the vertices (Object.bound_box is stale right after Mesh.transform)."""
    import numpy as np
    from mathutils import Vector
    pts = []
    for o in meshes:
        a = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", a)
        a = a.reshape(-1, 3)
        m = np.array(o.matrix_world)
        pts.append(a @ m[:3, :3].T + m[:3, 3])
    pts = np.concatenate(pts)
    return Vector(pts.min(0)), Vector(pts.max(0))


def _import_one(bpy, name: str):
    """Import one USD; returns the new objects. SimReady USDs are metersPerUnit 1, Z-up after Blender's
    axis conversion; UsdPreviewSurface materials arrive as Principled + textures (the MDL ones do not)."""
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(_usd_path(name)), import_cameras=False, import_lights=False,
                          import_materials=True, import_usd_preview=True, set_material_blend=True,
                          import_proxy=False, import_guide=False, import_render=True, import_visible_only=True,
                          support_scene_instancing=False, import_textures_mode="IMPORT_NONE",
                          create_world_material=False, scale=1.0, apply_unit_conversion_scale=True)
    return [o for o in bpy.data.objects if o not in before]


def _normalise(bpy, objs, name: str, col):
    """Bake every mesh into world space (transforms applied), drop empties, origin = floor centre."""
    from mathutils import Matrix, Vector
    meshes = [o for o in objs if o.type == "MESH"]
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        if o.data.users > 1:
            o.data = o.data.copy()
        o.data.transform(mw)
        o.matrix_world = Matrix.Identity(4)
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    for o in objs:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    lo, hi = _bbox(meshes)
    shift = Matrix.Translation(-Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)))
    for i, o in enumerate(meshes):
        o.data.transform(shift)
        o.data.update()
        o.name = o.data.name = f"{name}.{i:03d}"
    return meshes, tuple(round(x, 3) for x in (hi - lo))


def _swap_materials(meshes, name: str, M):
    rules = ASSETS[name]["mats"]
    for o in meshes:
        if not o.material_slots:
            o.data.materials.append(M.plant(rules.get("*") or "painted_grey"))
        for slot in o.material_slots:
            old = slot.material.name.lower() if slot.material else ""
            key = next((v for k, v in rules.items() if k != "*" and k in old), rules.get("*"))
            if key is not None:
                slot.material = M.plant(key)
            elif slot.material is None:
                slot.material = M.plant("painted_grey")


def build_assets(bpy, names, M, root_col) -> tuple[dict, dict]:
    sizes, report = {}, {}
    for n in names:
        if not _usd_path(n).exists():
            report[n] = "missing (run fetch)"
            continue
        col = bpy.data.collections.new(n)
        root_col.children.link(col)
        objs = _import_one(bpy, n)
        meshes, size = _normalise(bpy, objs, n, col)
        _swap_materials(meshes, n, M)
        sizes[n] = size
        report[n] = dict(meshes=len(meshes), polys=sum(len(o.data.polygons) for o in meshes), size=size,
                         mats=sorted({s.material.name for o in meshes for s in o.material_slots if s.material}))
        print(f"[assets] {n}: {report[n]}", flush=True)
    return sizes, report


def build(names=None, out: Path = LIBRARY) -> dict:
    """Import, normalise and save the library. Collections already in `out` are kept unless rebuilt."""
    bpy = _bpy()
    import materials as M
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if out.exists():
        bpy.ops.wm.open_mainfile(filepath=str(out))
    names = list(names or ASSETS)
    for n in names + ["BIW"]:
        old = bpy.data.collections.get(n)
        if old is not None:
            for o in list(old.all_objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(old)
    root = bpy.context.scene.collection
    sizes = json.loads(SIZES_JSON.read_text(encoding="utf-8")) if SIZES_JSON.exists() and out.exists() else {}
    new_sizes, report = build_assets(bpy, [n for n in names if n != "BIW"], M, root)
    sizes.update(new_sizes)
    if bpy.data.collections.get("car_frame") is not None:
        sizes["BIW"] = build_biw(bpy, M, root)
        report["BIW"] = dict(size=sizes["BIW"])
    for m in M.preview_all("2k"):                          # every PLANT material ships in the library
        m.use_fake_user = True
    M.world_hdri("autoshop_01", "1k", scene=bpy.context.scene)   # CC0 workshop HDRI (credited, optional)
    bpy.data.orphans_purge(do_recursive=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    SIZES_JSON.write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    bpy.context.preferences.filepaths.save_version = 0       # no .blend1 backup of a 300+ MB file
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    write_credits(M)
    return report


# ----------------------------------------------------------------------------------------------- body-in-white
# Placement of the closures on the SimReady frame (frame coords: +X = car front, +Y = car left, z = 0 at the
# lowest sill point; frame 4.93 x 1.84 x 1.32 m). Openings were measured from a vertex profile of the frame:
# front door x 0.0..0.9, B-pillar -0.27..0.0, rear door -0.95..-0.25, roof x -1.1..0.5 at z 1.25-1.32, hood
# region x 0.95..2.35 at z ~0.8, boot deck x -2.4..-1.5 at z ~1.0. Parts come from other SimReady car models,
# so each is scaled to its slot. Native part axes: doors have the outer skin at +X; door_front's front edge is
# +Y (a right-hand door), door_rear's rear edge is +Y (a left-hand door); boot_lid's rear face is +X.
# (part, x, y, z, yaw_deg, scale, mirror_y)
BIW_LAYOUT = [
    ("door_front", 0.47, -0.83, 0.05, -90.0, 1.0, False),
    ("door_front", 0.47, 0.83, 0.05, -90.0, 1.0, True),
    ("door_rear", -0.68, 0.765, 0.04, 90.0, 0.95, False),
    ("door_rear", -0.68, -0.765, 0.04, 90.0, 0.95, True),
    ("hood", 1.62, 0.0, 0.66, 0.0, 0.85, False),
    ("roof_panel", -0.30, 0.0, 1.20, 0.0, 0.73, False),
    ("boot_lid", -2.00, 0.0, 0.58, 180.0, 1.0, False),
]
BIW_LENGTH = 4.65            # target overall length (m); the SimReady frame is 4.93 m, scaled uniformly
SKID_H = 0.30                # skid rail + support pins: BIW sill sits this high above the floor


def _skid(bpy, M, col, length: float, pin_top=lambda x, y: SKID_H):
    """Procedural body skid: two rectangular-tube rails, three cross members, four locating pins that reach
    up to the underbody (`pin_top(x, y)` = underbody z above that pin)."""
    import bmesh
    from mathutils import Matrix
    me = bpy.data.meshes.new("BIW.skid")
    bm = bmesh.new()

    def box(cx, cy, cz, sx, sy, sz):
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((cx, cy, cz)) @
                              Matrix.Diagonal((sx, sy, sz, 1.0)))
    for y in (-0.42, 0.42):
        box(0, y, 0.06, length * 0.98, 0.10, 0.12)                       # rails
    for x in (-length * 0.38, 0.0, length * 0.38):
        box(x, 0, 0.10, 0.08, 0.84, 0.06)                                # cross members
    for x in (-length * 0.30, length * 0.27):
        for y in (-0.42, 0.42):
            top = max(pin_top(x, y), 0.14)
            box(x, y, 0.12 + (top - 0.12) / 2, 0.08, 0.08, top - 0.12)  # pins / pads
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M.plant("machine_grey"))
    ob = bpy.data.objects.new("BIW.skid", me)
    col.objects.link(ob)
    return ob


def build_biw(bpy, M, root):
    """Collection "BIW": frame + closures in bare steel on a procedural skid, origin at floor centre."""
    import bmesh
    from mathutils import Matrix
    col = bpy.data.collections.new("BIW")
    root.children.link(col)
    k = BIW_LENGTH / _size("car_frame")[0]
    lift = Matrix.Translation((0, 0, SKID_H)) @ Matrix.Scale(k, 4)
    parts = [("car_frame", 0, 0, 0, 0, 1.0, False)] + [p for p in BIW_LAYOUT if bpy.data.collections.get(p[0])]
    for i, (n, x, y, z, yaw, s, mirror) in enumerate(parts):
        place = lift @ Matrix.Translation((x, y, z)) @ (Matrix.Scale(-1, 4, (0, 1, 0)) if mirror else Matrix()) \
            @ Matrix.Rotation(math.radians(yaw), 4, "Z") @ Matrix.Scale(s, 4)
        for src in bpy.data.collections[n].objects:
            me = src.data.copy()
            me.transform(place)
            if mirror:
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.reverse_faces(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
            ob = bpy.data.objects.new(f"BIW.{n}.{i:02d}", me)
            col.objects.link(ob)
    import numpy as np
    vs = []
    for o in col.objects:
        a = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", a)
        vs.append(a.reshape(-1, 3))
    vs = np.concatenate(vs)

    def pin_top(x, y, r=0.06):
        near = vs[(np.abs(vs[:, 0] - x) < r) & (np.abs(vs[:, 1] - y) < r) & (vs[:, 2] < SKID_H + 0.5)]
        return float(near[:, 2].min()) if len(near) else SKID_H
    _skid(bpy, M, col, BIW_LENGTH, pin_top)
    lo, hi = _bbox(list(col.objects))
    return tuple(round(v, 3) for v in (hi - lo))


def _size(name):
    bpy = _bpy()
    lo, hi = _bbox(list(bpy.data.collections[name].objects))
    return tuple(hi - lo)


# ----------------------------------------------------------------------------------------------- credits
REFERENCE_JSON = Path(r"D:\icdata\assets\reference\reference.json")
SCENE_HDRI = "kloofendal_overcast_puresky"     # scene_real.py default sky (Poly Haven, CC0)


def _other_credits() -> list:
    """Rows for what the scene uses besides this library: FANUC meshes (render/blender/fanuc.py), MakeHuman
    workers (render/blender/humans.py CREDITS), the sky HDRI, and the reference photos of the acceptance check."""
    rows = [dict(asset="fanuc_r2000ic_210f", source_asset="Robots/Fanuc/r2000ic_210f", owner=OWNER, licence=CC_BY,
                 licence_url=CC_BY_URL, source_url=BUCKET_ROOT + "Robots/Fanuc/r2000ic_210f/r2000ic_210f.usd",
                 used_for="spot-welding robots (meshes + joint frames, render/blender/fanuc.py)"),
            dict(asset=SCENE_HDRI, owner="Poly Haven", licence="CC0-1.0", licence_url="https://polyhaven.com/license",
                 source_url=f"https://polyhaven.com/a/{SCENE_HDRI}", kind="HDRI (Poly Haven)",
                 used_for="sky seen through skylights and wall bands (render/blender/scene_real.py)")]
    try:
        import humans                                   # needs bpy: only inside Blender
        rows += [dict(asset=c["asset"], owner=c["owner"], licence=c["licence"], licence_url=c["url"], source_url=c["url"],
                      used_for="rigged workers in PPE (render/blender/humans.py)") for c in humans.CREDITS]
    except ImportError:
        pass
    if REFERENCE_JSON.exists():
        rows += [dict(asset=r["file"], owner=r["author"], licence=r["licence"], licence_url=r["licence_url"],
                      source_url=r["source_page_url"], used_for="reference photo for the side-by-side check only "
                      "(render/side_by_side.py); never inside a render")
                 for r in json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))]
    return rows


def write_credits(M=None, path: Path = CREDITS) -> list:
    rows = []
    for n, s in ASSETS.items():
        if _usd_path(n).exists():
            rows.append(dict(asset=n, source_asset=s["folder"].rsplit("/", 1)[-1], owner=s["owner"],
                             licence=s["licence"], licence_url=CC_BY_URL, source_url=s["source_url"],
                             used_for=s["used_for"]))
    cc0 = {}                                            # keep material rows from earlier runs (cached builds
    if path.exists():                                   # do not re-fetch, so they would be lost otherwise)
        cc0 = {r["asset"]: r for r in json.loads(path.read_text(encoding="utf-8")) if r["licence"].startswith("CC0")}
    if M is not None:
        for c in M.used_credits():
            cc0[c["asset"]] = dict(c, used_for="HDRI (world lighting)" if "HDRI" in c["kind"]
                                   else "CC0 PBR material in render/blender/materials.py")
    rows += list(cc0.values())
    have = {r["asset"] for r in rows}
    rows += [r for r in _other_credits() if r["asset"] not in have]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


# ----------------------------------------------------------------------------------------------- proof renders
LIGHTS = {"key": ((4, -5, 7), 1100), "fill": ((-6, -2, 4), 300), "rim": ((0, 7, 6), 500)}


def _studio(bpy, M):
    """Neutral studio: grey world, epoxy floor, three area lights, a 35 mm camera; CPU 16 spp + OIDN."""
    import bmesh
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.resolution_x, sc.render.resolution_y = 640, 360
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    M.world_hdri("autoshop_01", "1k", strength=0.45, scene=sc)
    me = bpy.data.meshes.new("studio_floor")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=40)
    bm.to_mesh(me)
    bm.free()
    fl = bpy.data.objects.new("studio_floor", me)
    me.materials.append(M.plant("floor_epoxy"))
    sc.collection.objects.link(fl)
    lights = []
    for nm, (loc, e) in LIGHTS.items():
        ld = bpy.data.lights.new(nm, "AREA")
        ld.energy, ld.size = e, 4.0
        lo = bpy.data.objects.new(nm, ld)
        lo.location = loc
        sc.collection.objects.link(lo)
        lo.constraints.new("TRACK_TO").target = fl
        lights.append(lo)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = 35
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam, lights


def _frame(cam, size, lights):
    """3/4 elevated view fitting a box of `size` (x, y, z) standing on the floor origin."""
    from mathutils import Vector
    sx, sy, sz = size
    r = max(0.5 * Vector(size).length, 0.3)               # bounding-sphere radius
    tgt = Vector((0, 0, sz * 0.5))
    d = Vector((1.0, -1.35, 0.75)).normalized()
    vfov = 2 * math.atan(cam.data.sensor_width * 9 / 16 / (2 * cam.data.lens))
    cam.location = tgt + d * (r / math.sin(vfov / 2) * 1.02)
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    k = max(1.0, r / 3.5)
    for lo in lights:
        loc, e = LIGHTS[lo.name]
        lo.location = Vector(loc) * k
        lo.data.energy = e * k * k
        lo.data.size = 4.0 * k


def test(names=None, lib: Path = LIBRARY, out_dir: Path = TEST_DIR) -> list:
    bpy = _bpy()
    import materials as M
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cam, lights = _studio(bpy, M)
    sizes = json.loads(SIZES_JSON.read_text(encoding="utf-8"))
    names = [n for n in (names or sizes) if sizes.get(n)]
    load_library(lib, names=names, link=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for n in names:
        if bpy.data.collections.get(n) is None:
            continue
        ob = instance(n, (0, 0, 0))
        _frame(cam, sizes[n], lights)
        bpy.context.scene.render.filepath = str(out_dir / f"{n}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(ob, do_unlink=True)
        outs.append(str(out_dir / f"{n}.png"))
        print("[assets] rendered", outs[-1], flush=True)
    return outs


def test_materials(out: Path = TEST_DIR / "materials_plant.png") -> str:
    """One render: rounded cubes, one per PLANT material (row-major, order printed), on the epoxy floor."""
    bpy = _bpy()
    import materials as M
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cam, lights = _studio(bpy, M)
    keys = [k for k in M.PLANT if k != "floor_epoxy"]
    cols = 7
    for i, k in enumerate(keys):
        bpy.ops.mesh.primitive_cube_add(size=0.8, location=((i % cols - (cols - 1) / 2) * 1.1,
                                                            (i // cols) * 1.3 - 0.6, 0.4))
        o = bpy.context.active_object
        bv = o.modifiers.new("bevel", "BEVEL")
        bv.width, bv.segments = 0.06, 4
        o.data.materials.append(M.plant(k))
        print("[materials]", i, k, flush=True)
    _frame(cam, (cols * 1.1, 3.0, 0.8), lights)
    bpy.context.scene.render.filepath = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    write_credits(M)
    return str(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "fetch"
    if cmd == "fetch":
        fetch(args[1:] or None)
    elif cmd == "build":
        print(json.dumps(build(args[1:] or None), indent=1, default=str))
    elif cmd == "test":
        print(test(args[1:] or None))
    elif cmd == "test_materials":
        print(test_materials())
    else:
        sys.exit(f"unknown command {cmd}")


