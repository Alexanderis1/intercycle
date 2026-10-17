"""Articulated FANUC R-2000iC/210F spot-welding robot for the Blender renders.

Meshes: NVIDIA Isaac Sim 6.0 asset Robots/Fanuc/r2000ic_210f (CC-BY 4.0, LICENSE next to the USD).
Kinematics: joint frames, axes and limits read from r2000ic_210f_physics.usd (UsdPhysics revolute joints).
At zero pose every link frame is unrotated, so each joint is an empty at the joint origin and a joint
angle is a single Euler component about the joint's local axis (sign folds in the USD axis direction).

    from fanuc import build_fanuc, set_q, HOME, weld_q
    joints, tip = build_fanuc("arm_N0", (x, y, 0.0), yaw_deg)      # robot faces local +X
    set_q(joints, weld_q(phase, k))

CLI (inside Blender, via bl_run.py):
    -- fanuc.py --build-lib              # USD -> D:\\icdata\\assets\\library\\fanuc.blend (mesh cache)
    -- fanuc.py --test [--out DIR]       # CPU proof renders of HOME + 3 weld poses from 2 angles
    -- fanuc.py --ik                     # re-solve the weld waypoints (prints degrees)
Pure-python parts (FK / IK / waypoints) need only numpy.
"""
import math
import os
import sys

import numpy as np

USD_PATH = r"D:\icdata\assets\nvidia\6.0\Robots\Fanuc\r2000ic_210f\r2000ic_210f.usd"
LIB_PATH = r"D:\icdata\assets\library\fanuc.blend"
ROOT = "/r2000ic_210f"

# name, origin relative to the parent joint (m), local axis index (0=X 1=Y 2=Z), sign, limits (deg), USD link
# World joint origins at zero pose: J1 (0,0,.67) +Z | J2 (.312,0,.67) +Y | J3 (.312,0,1.745) -Y |
# J4 (.312,0,1.97) -X | J5 (1.592,0,1.97) -Y | J6 (1.592,0,1.97) -X ; flange at x=1.807.
JOINTS = (
    ("a1", (0.0, 0.0, 0.67), 2, +1, (-185.0, 185.0), "J1_link", "j1"),
    ("a2", (0.312, 0.0, 0.0), 1, +1, (-60.0, 76.0), "J2_link", "j2"),
    ("a3", (0.0, 0.0, 1.075), 1, -1, (-79.0, 180.0), "J3_link", "j3"),
    ("a4", (0.0, 0.0, 0.225), 0, -1, (-360.0, 360.0), "J4_link", "j4"),
    ("a5", (1.28, 0.0, 0.0), 1, -1, (-125.0, 125.0), "J5_link", "j5"),
    ("a6", (0.0, 0.0, 0.0), 0, -1, (-360.0, 360.0), "J6_link", "j6"),
)
JN = tuple(j[0] for j in JOINTS)
LIMITS = np.radians([j[4] for j in JOINTS])
FLANGE_X = 0.215                  # flange face, in the a6 frame (tool axis = local +X)
TIP_LOCAL = (0.86, 0.0, 0.0)      # electrode contact point of the servo C-gun, in the a6 frame
# the gun's electrodes close along a6-local Z; the C opens towards a6-local +X


# ----------------------------------------------------------------------------------------- kinematics
def _rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(q, tip=TIP_LOCAL):
    """Base-frame pose of the gun tip: (position (3,), rotation (3,3)); columns = a6 local X, Y, Z."""
    R, p = np.eye(3), np.zeros(3)
    for (_, org, ax, sg, _, _, _), a in zip(JOINTS, q):
        p = p + R @ np.array(org)
        R = R @ _rot(ax, sg * a)
    return p + R @ np.array(tip), R


def tool_frame(fwd, stroke):
    """Rotation whose X = gun forward (into the part), Z = electrode stroke axis."""
    x = np.array(fwd, float)
    x /= np.linalg.norm(x)
    z = np.array(stroke, float)
    z -= x * (z @ x)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def ik(target, R_target, seed, iters=400):
    """Damped least squares on position + orientation, clamped to the USD joint limits."""
    q = np.array(seed, float)
    target = np.asarray(target, float)

    def err(qq):
        p, R = fk(qq)
        Re = R_target @ R.T
        w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * 0.5
        return np.concatenate([target - p, 0.5 * w])

    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e) < 1e-6:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-6
            J[:, i] = (err(q) - err(q + dq)) / 1e-6
        lam = 0.05
        q = q + J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        q = np.clip(q, LIMITS[:, 0], LIMITS[:, 1])
    p, _ = fk(q)
    return q, float(np.linalg.norm(p - target))


# --------------------------------------------------------------------------------- poses / waypoints
# Base frame: robot faces +X; a car body on the conveyor ~1.5-2 m in front, panels at 1.0-1.4 m height.
HOME = [math.radians(v) for v in (0.0, -15.0, -5.0, 0.0, -70.0, 0.0)]
# (target xyz, gun forward, electrode stroke axis) -> solved with ik() (see --ik), hard-coded below.
WELD_TARGETS = (
    ((1.70, -0.45, 1.05), (1, 0, 0), (0, 0, 1)),          # rocker / door-opening lower flange, stroke vertical
    ((1.85, 0.05, 1.25), (1, 0, 0), (0, 1, 0)),           # B-pillar vertical flange, gun rolled 90 deg
    ((1.55, 0.40, 1.40), (0.8, 0, -0.6), (0.6, 0, 0.8)),  # roof-rail flange, gun pitched down
)
WELD_Q_DEG = (
    (-28.18, -29.88, -64.99, 42.97, 43.85, -33.89),
    (1.54, -33.24, -47.77, -90.1, -0.02, -0.1),
    (24.89, -34.59, -28.43, 26.54, -48.92, -33.73),
)
WELD_Q = [[math.radians(v) for v in q] for q in WELD_Q_DEG]
_KW = [HOME, WELD_Q[0], WELD_Q[1], WELD_Q[2], WELD_Q[0], HOME]


def weld_q(phase: float, k: int):
    """Smooth-step through HOME -> 3 weld spots -> HOME over phase 0..1; odd k mirrors the pose (J1/J4/J6)."""
    n = len(_KW) - 1
    u = min(max(phase, 0.0), 1.0) * n
    i = min(int(u), n - 1)
    a = u - i
    a = a * a * (3 - 2 * a)
    q = [_KW[i][j] + a * (_KW[i + 1][j] - _KW[i][j]) for j in range(6)]
    if k % 2:
        q[0], q[3], q[5] = -q[0], -q[3], -q[5]
    return q


# ---------------------------------------------------------------------------------------- materials
def _principled(bpy, name, rgb, rough, metal, worn=False):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if worn:   # low-frequency grime + scuffs: darker, rougher patches
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 6.0
        nz.inputs["Detail"].default_value = 8.0
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.45
        ramp.color_ramp.elements[1].position = 0.75
        nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["A"].default_value = (rgb[0] * 0.55, rgb[1] * 0.5, rgb[2] * 0.4, 1.0)
        mix.inputs["B"].default_value = (*rgb, 1.0)
        nt.links.new(ramp.outputs["Color"], mix.inputs["Factor"])
        nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = rough + 0.25
        mr.inputs["To Max"].default_value = rough
        nt.links.new(ramp.outputs["Color"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.05
        nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def materials(bpy):
    """Named so render/blender/materials.py can swap them for CC0 PBR later."""
    return {
        "yellow": _principled(bpy, "fanuc_yellow", (0.85, 0.62, 0.02), 0.38, 0.0, worn=True),
        "black": _principled(bpy, "fanuc_black", (0.025, 0.025, 0.028), 0.55, 0.0),
        "grey": _principled(bpy, "fanuc_grey", (0.35, 0.36, 0.38), 0.45, 0.6),
        "steel": _principled(bpy, "gun_steel", (0.42, 0.43, 0.45), 0.42, 1.0),
        "copper": _principled(bpy, "gun_copper", (0.8, 0.42, 0.25), 0.3, 1.0),
        "alu": _principled(bpy, "gun_alu", (0.63, 0.64, 0.65), 0.35, 1.0),           # cast transformer housing
        "hose_blue": _principled(bpy, "hose_blue", (0.02, 0.12, 0.45), 0.5, 0.0),     # cooling water in
        "hose_red": _principled(bpy, "hose_red", (0.5, 0.03, 0.02), 0.5, 0.0),        # cooling water out
    }


SLOTS = ("yellow", "black", "grey")


def _classify(rgb):
    r, g, b = rgb
    if max(rgb) < 0.15:
        return "black"
    if r > 0.8 and g < 0.2:          # Isaac paints the servo motors red; real FANUC motors are black
        return "black"
    if r > 0.5 and g > 0.4 and b < 0.25:
        return "yellow"
    return "grey"


# ----------------------------------------------------------------------------------- USD mesh import
def _joint_world_origins():
    out, p = {}, np.zeros(3)
    for jn, org, *_ in JOINTS:
        p = p + np.array(org)
        out[jn] = p.copy()
    return out


def _meshes_from_usd(bpy):
    """One Blender mesh per link ('fanuc_base', 'fanuc_a1'..'fanuc_a6'), vertices relative to the joint origin."""
    import bmesh
    from pxr import Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(USD_PATH)
    if abs(UsdGeom.GetStageMetersPerUnit(st) - 1.0) > 1e-9 or UsdGeom.GetStageUpAxis(st) != "Z":
        raise RuntimeError("unexpected USD units/up axis")
    xc = UsdGeom.XformCache()
    origins = _joint_world_origins()
    links = [("base", "robot_base", "base", np.zeros(3))] + [(j[0], j[5], j[6], origins[j[0]]) for j in JOINTS]
    mats = materials(bpy)
    out = {}
    for key, link, short, org in links:
        prim = st.GetPrimAtPath(f"{ROOT}/{link}/visuals/{short}/mesh")
        m = UsdGeom.Mesh(prim)
        M = np.array(xc.GetLocalToWorldTransform(prim)).T          # Gf is row-vector convention
        pts = np.array(m.GetPointsAttr().Get(), float)
        pts = (M[:3, :3] @ pts.T).T + M[:3, 3] - org
        counts = np.array(m.GetFaceVertexCountsAttr().Get(), np.int64)
        idx = np.array(m.GetFaceVertexIndicesAttr().Get(), np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        mat_idx = np.zeros(len(counts), np.int64)
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        subsets = UsdGeom.Subset.GetAllGeomSubsets(m)

        def slot_of(material):
            rgb = (0.5, 0.5, 0.5)
            for ch in (material.GetPrim().GetChildren() if material else []):    # MDL shader (OmniPBR)
                inp = UsdShade.Shader(ch).GetInput("diffuse_color_constant")
                if inp and inp.Get() is not None:
                    rgb = tuple(inp.Get())
            return SLOTS.index(_classify(rgb))

        mat_idx[:] = slot_of(bound) if bound else SLOTS.index("yellow")
        for s in subsets:
            mat = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
            mat_idx[np.array(s.GetIndicesAttr().Get(), np.int64)] = slot_of(mat)

        me = bpy.data.meshes.new(f"fanuc_{key}")
        me.vertices.add(len(pts))
        me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
        me.loops.add(len(idx))
        me.loops.foreach_set("vertex_index", idx.astype(np.int32))
        me.polygons.add(len(counts))
        me.polygons.foreach_set("loop_start", starts.astype(np.int32))
        me.polygons.foreach_set("material_index", mat_idx.astype(np.int32))
        me.update()
        me.validate()
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bm.to_mesh(me)
        bm.free()
        for s in SLOTS:
            me.materials.append(mats[s])
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(35.0))
        out[key] = me
        print(f"[fanuc] {key}: {len(me.vertices)} verts, {len(me.polygons)} faces", flush=True)
    return out


def _load_meshes(bpy):
    keys = ["base"] + list(JN)
    if all(f"fanuc_{k}" in bpy.data.meshes for k in keys):
        return {k: bpy.data.meshes[f"fanuc_{k}"] for k in keys}
    if os.path.exists(LIB_PATH):
        with bpy.data.libraries.load(LIB_PATH, link=False) as (src, dst):
            dst.meshes = [f"fanuc_{k}" for k in keys if f"fanuc_{k}" in src.meshes]
        mats = materials(bpy)
        out = {}
        for k in keys:
            me = bpy.data.meshes[f"fanuc_{k}"]
            for i, s in enumerate(SLOTS):          # rebind to scene materials (appended copies may be .001)
                me.materials[i] = mats[s]
            out[k] = me
        for m in list(bpy.data.materials):
            if m.users == 0 and m.name.split(".")[0] in {x.name for x in mats.values()} and "." in m.name:
                bpy.data.materials.remove(m)
        return out
    return _meshes_from_usd(bpy)


def build_library():
    import bpy
    meshes = _meshes_from_usd(bpy)
    os.makedirs(os.path.dirname(LIB_PATH), exist_ok=True)
    bpy.data.libraries.write(LIB_PATH, set(meshes.values()), fake_user=True, compress=True)
    print(f"[fanuc] wrote {LIB_PATH}", flush=True)


# ------------------------------------------------------------------------------- procedural servo gun
def _box(bm, size, loc, mat):
    r = bmesh_ops().create_cube(bm, size=1.0)
    vs = r["verts"]
    for v in vs:
        v.co.x = v.co.x * size[0] + loc[0]
        v.co.y = v.co.y * size[1] + loc[1]
        v.co.z = v.co.z * size[2] + loc[2]
    for f in {f for v in vs for f in v.link_faces}:
        f.material_index = mat


def _cyl(bm, r, h, loc, axis, mat, seg=24):
    from mathutils import Matrix
    rot = {"x": Matrix.Rotation(math.pi / 2, 4, "Y"), "y": Matrix.Rotation(math.pi / 2, 4, "X"), "z": Matrix.Identity(4)}[axis]
    res = bmesh_ops().create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=h,
                                  matrix=Matrix.Translation(loc) @ rot)
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = mat
        f.smooth = True


def _c_frame(bm, center, r_out, r_in, a0, a1, thick, mat, seg=28, y0=0.0):
    """Thick arc plate in the X-Z plane (the C of the gun), extruded along Y around y0."""
    cx, cz = center
    rings = []
    for i in range(seg + 1):
        a = math.radians(a0 + (a1 - a0) * i / seg)
        ring = []
        for r, y in ((r_out, y0 - thick / 2), (r_out, y0 + thick / 2), (r_in, y0 + thick / 2), (r_in, y0 - thick / 2)):
            ring.append(bm.verts.new((cx + r * math.cos(a), y, cz + r * math.sin(a))))
        rings.append(ring)
    for i in range(seg):
        for k in range(4):
            f = bm.faces.new((rings[i][k], rings[i + 1][k], rings[i + 1][(k + 1) % 4], rings[i][(k + 1) % 4]))
            f.material_index = mat
    for ring in (rings[0], rings[-1][::-1]):
        bm.faces.new(ring).material_index = mat


def bmesh_ops():
    import bmesh
    return bmesh.ops


def _gun_mesh(bpy):
    """Servo C-gun in the a6 frame: flange face at x=FLANGE_X, C opening to +X, electrodes closing along Z at TIP_LOCAL.
    Silhouette of a real MFDC servo C-gun: two C side plates, a ribbed cast transformer behind the C, the servo
    actuator + motor above the moving electrode, electrode holders on both arms."""
    me = bpy.data.meshes.get("fanuc_gun_v2")
    if me is not None:
        return me
    import bmesh
    bm = bmesh.new()
    STEEL, BLACK, COPPER, GREY, ALU = 0, 1, 2, 3, 4
    fx, tip_x = FLANGE_X, TIP_LOCAL[0]
    _cyl(bm, 0.125, 0.03, (fx + 0.015, 0, 0), "x", GREY)                 # tool changer / adapter plate
    _box(bm, (0.04, 0.26, 0.34), (fx + 0.05, 0, 0), STEEL)               # mounting bracket
    cxc, r_out, r_in, a_open = fx + 0.40, 0.30, 0.20, 40.0               # C yoke: centre, radii, half opening (deg)
    for y0 in (-0.045, 0.045):
        _c_frame(bm, (cxc, 0.0), r_out, r_in, a_open, 360.0 - a_open, 0.022, STEEL, y0=y0)
    _box(bm, (0.20, 0.21, 0.24), (fx + 0.19, 0, -0.04), ALU)             # MFDC transformer behind the C
    for k in range(5):                                                    # cooling ribs
        _box(bm, (0.21, 0.225, 0.012), (fx + 0.19, 0, -0.14 + k * 0.05), ALU)
    _box(bm, (0.03, 0.18, 0.20), (fx + 0.305, 0, -0.04), BLACK)          # secondary bus bars / terminal cover
    r_mid = (r_out + r_in) / 2
    jx = cxc + r_mid * math.cos(math.radians(a_open))                    # jaw root at the C opening
    jaw = r_mid * math.sin(math.radians(a_open))
    for sz in (1, -1):                                                    # arms from the C tips to the electrodes
        _box(bm, (tip_x - jx + 0.05, 0.11, 0.06), ((tip_x + jx) / 2 + 0.01, 0, sz * jaw), STEEL)
        _cyl(bm, 0.032, 0.06, (tip_x, 0, sz * (jaw - 0.04)), "z", GREY)  # electrode holders
        _cyl(bm, 0.02, jaw - 0.07, (tip_x, 0, sz * (jaw - 0.07) / 2 + sz * 0.005), "z", COPPER)   # electrode shanks
        _cyl(bm, 0.009, 0.02, (tip_x, 0, sz * 0.012), "z", COPPER, seg=12)                        # caps
    _cyl(bm, 0.05, 0.22, (tip_x, 0, jaw + 0.14), "z", GREY)               # servo actuator (ball screw)
    _cyl(bm, 0.062, 0.14, (tip_x, 0, jaw + 0.32), "z", BLACK)             # servo motor
    _box(bm, (0.06, 0.08, 0.06), (tip_x - 0.08, 0, jaw + 0.34), BLACK)    # motor connector
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("fanuc_gun_v2")
    bm.to_mesh(me)
    bm.free()
    mats = materials(bpy)
    for s in ("steel", "black", "copper", "grey", "alu"):
        me.materials.append(mats[s])
    return me


def _retract_mesh(bpy):
    """Dress-pack retract unit (spring cylinder + two clamp collars) along the forearm, in the a4 frame."""
    me = bpy.data.meshes.get("fanuc_retract")
    if me is not None:
        return me
    import bmesh
    bm = bmesh.new()
    _cyl(bm, 0.05, 0.36, (0.32, 0.0, 0.29), "x", 1)
    for x in (0.16, 0.48):
        _cyl(bm, 0.062, 0.03, (x, 0.0, 0.29), "x", 0)
    _box(bm, (0.05, 0.06, 0.08), (0.2, 0.0, 0.23), 0)                    # bracket onto the arm casting
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("fanuc_retract")
    bm.to_mesh(me)
    bm.free()
    mats = materials(bpy)
    for s in ("grey", "black"):
        me.materials.append(mats[s])
    return me


def _hose(bpy, name, pts, radius, parent, col, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    col.objects.link(o)
    o.parent = parent
    return o


# ------------------------------------------------------------------------------------------- public
def build_fanuc(name, loc, yaw_deg, collection=None, gun=True):
    """Returns ({'a1'..'a6': joint empty}, tip empty). Robot faces local +X; yaw about world Z."""
    import bpy
    col = collection or bpy.context.scene.collection
    meshes = _load_meshes(bpy)
    mats = materials(bpy)

    def empty(n, at, parent):
        o = bpy.data.objects.new(n, None)
        o.empty_display_size = 0.15
        col.objects.link(o)
        o.parent = parent
        o.location = at
        return o

    def mesh_obj(n, me, parent):
        o = bpy.data.objects.new(n, me)
        col.objects.link(o)
        o.parent = parent
        return o

    root = empty(name, loc, None)
    root.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
    mesh_obj(name + "_base", meshes["base"], root)
    joints, parent = {}, root
    for jn, org, ax, sg, lim, _, _ in JOINTS:
        e = empty(f"{name}_{jn}", org, parent)
        e["axis"], e["sign"] = ax, sg
        e["lim_deg"] = list(lim)
        mesh_obj(f"{name}_l{jn[1]}", meshes[jn], e)
        joints[jn] = e
        parent = e
    a6 = joints["a6"]
    if gun:
        g = mesh_obj(name + "_gun", _gun_mesh(bpy), a6)
        bev = g.modifiers.new("bevel", "BEVEL")                          # soften the primitive edges
        bev.width, bev.segments, bev.limit_method = 0.004, 2, "ANGLE"
        # dress pack (moves with J4): retract unit on the forearm, main hose + cooling water in/out along it, and
        # a loop from the wrist to the transformer (J6)
        a4 = joints["a4"]
        mesh_obj(name + "_retract", _retract_mesh(bpy), a4)
        _hose(bpy, name + "_dress", [(0.12, 0.0, 0.2), (0.5, 0.0, 0.25), (0.85, 0.0, 0.22), (1.05, 0.0, 0.16)], 0.042, a4, col, mats["black"])
        for dy, m in ((0.05, "hose_blue"), (-0.05, "hose_red")):
            _hose(bpy, f"{name}_water{m[-1]}", [(0.15, dy, 0.17), (0.6, dy, 0.21), (1.05, dy * 0.6, 0.15)], 0.013, a4, col, mats[m])
            _hose(bpy, f"{name}_gunwater{m[-1]}", [(0.05, dy, 0.14), (0.14, dy, 0.24), (FLANGE_X + 0.1, dy * 0.8, 0.13)], 0.012, a6, col, mats[m])
        _hose(bpy, name + "_gunhose", [(0.05, 0.0, 0.16), (0.12, 0.0, 0.28), (FLANGE_X + 0.15, 0.0, 0.2)], 0.03, a6, col, mats["black"])
    tip = empty(name + "_tip", TIP_LOCAL, a6)
    tip.empty_display_size = 0.05
    set_q(joints, HOME)
    return joints, tip


def set_q(joints, q):
    """Apply 6 joint angles (rad, FANUC sign convention as in the Isaac USD) about each joint's local axis."""
    for jn, ang in zip(JN, q):
        e = joints[jn]
        r = [0.0, 0.0, 0.0]
        r[int(e["axis"])] = float(e["sign"]) * ang
        e.rotation_euler = r


# --------------------------------------------------------------------------------------------- CLI
def _solve_waypoints():
    seed = np.radians([0, 20, 0, 0, -30, 0])
    for tgt, fwd, stroke in WELD_TARGETS:
        best = None
        for s4 in (-90, 0, 90):
            for s5 in (-60, 0, 60):
                s = seed.copy()
                s[0] = math.atan2(tgt[1], tgt[0])
                s[3], s[4] = math.radians(s4), math.radians(s5)
                q, e = ik(tgt, tool_frame(fwd, stroke), s)
                cost = e * 100 + np.abs(q[3:]).sum() * 0.01
                if best is None or cost < best[0]:
                    best = (cost, q, e)
        _, q, e = best
        print(f"{tgt}: q_deg={tuple(round(float(v), 2) for v in np.degrees(q))} pos_err={e * 1000:.2f} mm", flush=True)


def _test_scene(out_dir, poses=None, cam_names=("a", "b")):
    import bpy
    from mathutils import Vector
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = 16
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPENIMAGEDENOISE"
    scn.render.resolution_x, scn.render.resolution_y = 640, 360
    scn.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.37, 0.4, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    col = scn.collection
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.data.materials.append(_principled(bpy, "test_floor", (0.3, 0.3, 0.3), 0.7, 0.0))
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 1.5
    sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
    col.objects.link(sun)
    area = bpy.data.objects.new("area", bpy.data.lights.new("area", "AREA"))
    area.data.energy, area.data.size = 250.0, 4.0
    area.location = (1.0, 0.0, 6.0)
    col.objects.link(area)
    joints, tip = build_fanuc("fanuc_test", (0, 0, 0), 0.0, col)
    marks = []
    red = _principled(bpy, "test_marker", (1.0, 0.05, 0.02), 0.4, 0.0)
    for i, (tgt, _, _) in enumerate(WELD_TARGETS):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.025, location=tgt)
        o = bpy.context.active_object
        o.data.materials.append(red)
        marks.append(o)
    # body-side proxy panel just behind the targets so reach reads in the image
    bpy.ops.mesh.primitive_cube_add(size=1, location=(2.1, 0.0, 0.7))
    panel = bpy.context.active_object
    panel.scale = (0.12, 2.6, 1.4)
    panel.data.materials.append(_principled(bpy, "test_biw", (0.6, 0.62, 0.64), 0.45, 0.7))
    cams = {}
    for cn, at in (("a", (1.2, -4.6, 2.0)), ("b", (-1.8, 3.2, 5.2))):
        cam = bpy.data.objects.new("cam_" + cn, bpy.data.cameras.new("cam_" + cn))
        cam.data.lens = 24
        col.objects.link(cam)
        cam.location = at
        d = Vector((1.0, 0.0, 1.1)) - Vector(at)
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cams[cn] = cam
    poses = poses or [("home", HOME)] + [(f"weld{i + 1}", q) for i, q in enumerate(WELD_Q)]
    cams = {k: v for k, v in cams.items() if k in cam_names}
    os.makedirs(out_dir, exist_ok=True)
    for pn, q in poses:
        set_q(joints, q)
        bpy.context.view_layer.update()
        p_fk, _ = fk(q)
        p_bl = np.array(tip.matrix_world.translation)
        print(f"[fanuc] {pn}: tip blender={p_bl.round(4)} fk={p_fk.round(4)} diff={np.linalg.norm(p_bl - p_fk) * 1000:.2f} mm", flush=True)
        for cn, cam in cams.items():
            scn.camera = cam
            scn.render.filepath = os.path.join(out_dir, f"fanuc_{pn}_{cn}.png")
            bpy.ops.render.render(write_still=True)


def _sweep_poses():
    """HOME with each joint in turn moved +35 deg (articulation check)."""
    out = []
    for j in range(6):
        q = list(HOME)
        q[j] += math.radians(35.0)
        out.append((f"sweep_a{j + 1}", q))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--out") + 1] if "--out" in args else r"D:\icdata\renders\asset_tests"
    if "--ik" in args:
        _solve_waypoints()
    elif "--build-lib" in args:
        build_library()
    elif "--sweep" in args:
        _test_scene(out, _sweep_poses(), ("a",))
    else:
        _test_scene(out)
