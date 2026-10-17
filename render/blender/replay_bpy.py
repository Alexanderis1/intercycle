#!/usr/bin/env python
"""Blender/Cycles fallback replay of the InterCycle sim trace   (`pip install bpy==4.5.14`, Python 3.11).

Same TRACE CONTRACT, same scenario YAML, same procedural placement, same shots and same camera
functions as render/isaac/replay.py - this file IMPORTS the pure-python half of that module
(`load_trace`, `build_ctx`, `SHOTS`, `arm_layout`, `datahall_layout`, `ble_events`, ...), so the two
renderers cannot drift apart. Only the Kit half is replaced by the bpy half below.

    python replay_bpy.py --scenario scenario/bodyshop.yaml --trace trace.parquet --out render/out \
        --shots overview,curtain,ble,pov,datahall,ab_cam --samples 64 --res 1920x1080
    python replay_bpy.py ... --dry-run                       # no bpy needed
    python replay_bpy.py --smoke --out render/out             # cube + light + device report, 10 s

NO third-party asset is required: everything is procedural (KUKA KR210-proportioned arms, block
body-in-white, racks, forklifts, people capsules). Optional CC0/CC-BY drop-ins, all fetched at run
time and never committed (details + URLs in render/blender/README.md):
    --hdri-id autoshop_01         Poly Haven HDRI, CC0 (default; fetched from api.polyhaven.com)
    --body-glb path.glb           Kenney Car Kit sedan (CC0) or any converted BIW; wheels/glass stripped
    --arm-dae-dir dir             KUKA KR210 L150 visual meshes (Apache-2.0) from ros-industrial/kuka_experimental
STATUS: written against the Blender 4.5 Python API from memory of the docs; not run yet. `# VERIFY`
marks the lines most likely to need a tweak on first contact. Cycles OptiX needs the NVIDIA driver's
libnvoptix (Modal: NVIDIA_DRIVER_CAPABILITIES=all); the script falls back OPTIX -> CUDA -> CPU and says which.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _cand in (HERE.parent / "isaac", HERE, Path("/root")):
    if (_cand / "replay.py").exists():
        sys.path.insert(0, str(_cand))
        break
import replay as core  # noqa: E402  (render/isaac/replay.py, pure-python half)

ORANGE = (1.0, 0.33, 0.02)
STEEL = (0.55, 0.57, 0.60)
YELLOW = (0.95, 0.75, 0.05)
CONCRETE = (0.38, 0.38, 0.40)


def log(*a):
    print("[bpy]", *a, flush=True)


# =====================================================================================
# bpy half
# =====================================================================================
class BScene:
    def __init__(self, scn: dict, ctx: "core.Ctx", args):
        import bpy
        import bmesh
        import mathutils
        self.bpy, self.bmesh, self.mu = bpy, bmesh, mathutils
        self.scn, self.ctx, self.args = scn, ctx, args
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.scene = bpy.context.scene
        self.scene.unit_settings.system = "METRIC"
        self.col_real = self._collection("Real")
        self.col_skins = self._collection("Skins")
        self.col_misc = self._collection("Misc")
        self.mats: dict[str, object] = {}
        self.prims: dict[str, object] = {}          # entity_id -> root object
        self.zone_mat: dict[str, object] = {}
        self.curtain_mat: dict[str, object] = {}
        self.sparks: dict[str, tuple] = {}
        self.arms: dict[str, list] = {}             # machine -> [(joints dict, k, tip empty)]
        self.skins: list[tuple] = []
        self.lights_ceiling: list[tuple] = []
        self.ble_pool: list[object] = []
        self._unit_cube = self._make_unit(kind="cube")
        self._unit_cyl = self._make_unit(kind="cyl")
        self._unit_sph = self._make_unit(kind="sph")

    # ----- helpers -----
    def _collection(self, name):
        c = self.bpy.data.collections.new(name)
        self.scene.collection.children.link(c)
        return c

    def _make_unit(self, kind):
        bm = self.bmesh.new()
        if kind == "cube":
            self.bmesh.ops.create_cube(bm, size=1.0)
        elif kind == "cyl":
            self.bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=24, radius1=0.5, radius2=0.5, depth=1.0)
        else:
            self.bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.5)
        me = self.bpy.data.meshes.new(f"unit_{kind}")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = kind != "cube"
        me.materials.append(None)          # ONE empty slot; each object links its own material (object-linked slot)
        return me

    def material(self, name, rgb, rough=0.5, metal=0.0, emit=None, strength=0.0, alpha=1.0):
        if name in self.mats:
            return self.mats[name]
        m = self.bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        b.inputs["Alpha"].default_value = alpha
        if emit is not None:
            b.inputs["Emission Color"].default_value = (*emit, 1.0)          # VERIFY socket name (4.x)
            b.inputs["Emission Strength"].default_value = strength
        self.mats[name] = m
        return m

    def glow(self, name, rgb, strength, alpha):
        """Translucent emitter: Mix(Transparent, Emission, fac=alpha). Used for hazard zones, curtains, BLE rings."""
        if name in self.mats:
            return self.mats[name]
        m = self.bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        em = nt.nodes.new("ShaderNodeEmission")
        em.name, mix.name = "GlowEmission", "GlowMix"
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        em.inputs["Color"].default_value = (*rgb, 1.0)
        em.inputs["Strength"].default_value = strength
        mix.inputs["Fac"].default_value = alpha
        self.mats[name] = m
        return m

    @staticmethod
    def no_light(o):
        """Glow overlays must be seen by the camera but must not light the scene."""
        o.visible_diffuse = o.visible_glossy = o.visible_shadow = False
        return o

    def set_emit(self, mat, rgb, strength, alpha=None):
        nt = mat.node_tree
        if "GlowEmission" in nt.nodes:
            nt.nodes["GlowEmission"].inputs["Color"].default_value = (*rgb, 1.0)
            nt.nodes["GlowEmission"].inputs["Strength"].default_value = strength
            if alpha is not None:
                nt.nodes["GlowMix"].inputs["Fac"].default_value = alpha
            return
        b = nt.nodes["Principled BSDF"]
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = strength
        if alpha is not None:
            b.inputs["Alpha"].default_value = alpha

    def _obj(self, name, mesh, loc, scale, mat, col=None, parent=None, yaw_deg=0.0):
        o = self.bpy.data.objects.new(name, mesh)
        (col or self.col_real).objects.link(o)
        o.location = loc
        o.scale = scale
        o.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
        if mat is not None:
            slot = o.material_slots[0]      # meshes are shared between objects: link the material per OBJECT
            slot.link = "OBJECT"
            slot.material = mat
        if parent is not None:
            o.parent = parent
        return o

    def box(self, name, size, loc, mat, col=None, parent=None, yaw_deg=0.0):
        return self._obj(name, self._unit_cube, loc, size, mat, col, parent, yaw_deg)

    def cyl(self, name, r, h, loc, mat, col=None, parent=None, axis="z"):
        o = self._obj(name, self._unit_cyl, loc, (2 * r, 2 * r, h), mat, col, parent)
        if axis == "x":
            o.rotation_euler = (0.0, math.radians(90), 0.0)
            o.scale = (2 * r, 2 * r, h)
        return o

    def sphere(self, name, r, loc, mat, col=None, parent=None):
        return self._obj(name, self._unit_sph, loc, (2 * r, 2 * r, 2 * r), mat, col, parent)

    def empty(self, name, loc=(0, 0, 0), parent=None, col=None):
        o = self.bpy.data.objects.new(name, None)
        (col or self.col_misc).objects.link(o)
        o.location = loc
        if parent is not None:
            o.parent = parent
        return o

    def light(self, name, kind, loc, energy, rgb=(1, 1, 1), size=1.0, size_y=None, rot=None):
        li = self.bpy.data.lights.new(name, kind)
        li.energy = energy
        li.color = rgb
        if kind == "AREA":
            li.shape = "RECTANGLE"
            li.size = size
            li.size_y = size_y or size
        elif kind == "POINT":
            li.shadow_soft_size = size
        o = self.bpy.data.objects.new(name, li)
        self.col_misc.objects.link(o)
        o.location = loc
        if rot:
            o.rotation_euler = rot
        return o

    # ----- scene build -----
    def build(self):
        self._render_settings()
        self._world()
        self._shell()
        self._line()
        self._market()
        self._data_hall()
        self._traffic()
        self._hero()
        self._ble_rig()
        self.cam_obj = self.bpy.data.objects.new("Cam", self.bpy.data.cameras.new("Cam"))
        self.col_misc.objects.link(self.cam_obj)
        self.cam_obj.data.sensor_width = 36.0
        self.cam_obj.data.clip_end = 1000.0
        self.scene.camera = self.cam_obj

    def _render_settings(self):
        sc, a = self.scene, self.args
        W, H = (int(v) for v in a.res.split("x"))
        sc.render.engine = "CYCLES"
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.use_persistent_data = True          # keeps BVH/textures between frames of the loop
        cy = sc.cycles
        cy.samples = a.samples
        cy.use_adaptive_sampling = True
        cy.adaptive_threshold = 0.02
        cy.use_denoising = True
        cy.max_bounces = 6
        cy.diffuse_bounces = 3
        cy.glossy_bounces = 4
        cy.transmission_bounces = 4
        try:
            sc.view_settings.view_transform = "AgX"
        except Exception:  # noqa: BLE001
            sc.view_settings.view_transform = "Filmic"
        self._view_transform = sc.view_settings.view_transform
        sc.view_settings.exposure = a.exposure
        self.device = setup_cycles_device(sc)
        if self.device == "OPTIX":
            try:
                cy.denoiser = "OPTIX"
            except Exception:  # noqa: BLE001
                pass
        elif self.device == "CPU":
            cy.samples = min(cy.samples, 16)
            log("WARNING no GPU: samples capped at 16")

    def _world(self):
        w = self.bpy.data.worlds.new("World")
        self.scene.world = w
        w.use_nodes = True
        bg = w.node_tree.nodes["Background"]
        bg.inputs["Strength"].default_value = 0.15
        hdr = fetch_polyhaven_hdri(self.args.hdri_id, Path(self.args.cache))
        if hdr:
            env = w.node_tree.nodes.new("ShaderNodeTexEnvironment")
            env.image = self.bpy.data.images.load(str(hdr))
            w.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
        else:
            bg.inputs["Color"].default_value = (0.5, 0.52, 0.55, 1.0)
        self.world_bg = bg

    def _shell(self):
        X, Z = self.scn["hall"]["x"][1], self.scn["hall"]["z"][1]
        floor = self.material("floor", CONCRETE, rough=0.35)
        wall = self.material("wall", (0.8, 0.81, 0.83), rough=0.85)
        steel = self.material("steel", STEEL, rough=0.4, metal=0.8)
        h = 11.0
        self.box("floor", (X + 26, Z + 2, 0.2), ((X + 24) / 2, Z / 2, -0.1), floor)
        for n, sz, loc in (("wallN", (X, 0.3, h), (X / 2, Z + 0.15, h / 2)), ("wallS", (X, 0.3, h), (X / 2, -0.15, h / 2)),
                           ("wallW", (0.3, Z, h), (-0.15, Z / 2, h / 2))):
            self.box(n, sz, loc, wall)
        for i in range(0, int(X) + 1, 12):
            self.box(f"truss{i}", (0.4, Z, 0.6), (i, Z / 2, h - 0.3), steel)
        for i in range(8):
            for j in range(4):
                o = self.light(f"ceil_{i}_{j}", "AREA", (6 + i * 12, 6 + j * 12, h - 0.9), self.args.light_w, (1.0, 0.97, 0.92), 5.0, 1.0)
                self.lights_ceiling.append((o, self.args.light_w))

    def _line(self):
        s = self.scn
        grey = self.material("cell", (0.35, 0.37, 0.4), rough=0.5, metal=0.5)
        yellow = self.material("yellow", YELLOW, rough=0.5)
        xs = [m["x"] for m in s["machines"] if m.get("on_line")]
        self.box("rail", (max(xs) - min(xs) + 14, 1.6, 0.35), ((max(xs) + min(xs)) / 2, 24, 0.17), grey)
        for m in s["machines"]:
            mid, kind = m["id"], m["kind"]
            w, d = m["size"]
            cx, cz = m["x"], m["z"]
            zm = self.glow(f"zone_{mid}", (0.0, 0.7, 0.15), 1.0, 0.25)
            self.no_light(self.box(f"zone_{mid}", (w, d, 0.02), (cx, cz, 0.012), zm))
            self.zone_mat[mid] = zm
            for sx in (-1, 1):
                for sy in (-1, 1):
                    self.box(f"post_{mid}_{sx}{sy}", (0.3, 0.3, 5.0), (cx + sx * (w / 2 - 0.4), cz + sy * (d / 2 - 0.4), 2.5), grey)
            self.box(f"beamA_{mid}", (w - 0.2, 0.3, 0.3), (cx, cz + d / 2 - 0.4, 5.0), grey)
            self.box(f"beamB_{mid}", (w - 0.2, 0.3, 0.3), (cx, cz - d / 2 + 0.4, 5.0), grey)
            lc = self.glow(f"lc_{mid}", (0.0, 1.0, 0.2), 2.0, 0.18)
            self.curtain_mat[mid] = lc
            for side, sgn in (("N", 1), ("S", -1)):
                zc = cz + sgn * (d / 2 + 0.15)
                for ex in (-1, 1):
                    self.box(f"lcpost_{mid}_{side}{ex}", (0.12, 0.12, 1.9), (cx + ex * w / 2, zc, 0.95), yellow)
                self.no_light(self.box(f"lcbeam_{mid}_{side}", (w, 0.02, 1.8), (cx, zc, 0.95), lc))
            if kind in ("weld_station", "respot", "framing", "roller_hemming", "sealer"):
                sm = self.material(f"spark_{mid}", (1, 0.8, 0.3), emit=(1.0, 0.7, 0.2), strength=0.0)
                sph = self.sphere(f"spark_{mid}", 0.05, (cx, cz, -50), sm)
                li = self.light(f"sparklight_{mid}", "POINT", (cx, cz, -50), 0.0, (1.0, 0.8, 0.5), 0.1)
                self.sparks[mid] = (sph, li, sm)
            self.arms[mid] = []
            for pl in core.arm_layout(m):
                joints, tip = self._kuka(pl.path.replace("/", "_"), pl.x, pl.y, pl.yaw_deg)
                self.arms[mid].append((joints, pl.meta["idx"] + (2 if pl.meta["side"] == "S" else 0), tip))
            if kind == "unload_lifter":
                self.box(f"scissor_{mid}", (w - 2, d - 3, 0.5), (cx, cz, 0.5), yellow)

    def _kuka(self, name, x, y, yaw_deg):
        """KR210 L150 kinematics from ros-industrial/kuka_experimental (Apache-2.0) joint origins/axes,
        procedural link geometry. Returns ({'a1'..'a6': empty}, tip_empty)."""
        orange = self.material("kuka_orange", ORANGE, rough=0.35, metal=0.2)
        black = self.material("kuka_black", (0.03, 0.03, 0.035), rough=0.5, metal=0.6)
        steel = self.material("steel", STEEL, rough=0.4, metal=0.8)
        root = self.empty(name, (x, y, 0.0), col=self.col_real)
        root.rotation_euler = (0, 0, math.radians(yaw_deg))
        self.cyl(name + "_base", 0.36, 0.33, (0, 0, 0.165), black, parent=root)
        a1 = self.empty(name + "_a1", (-0.003, 0.001, 0.331), root)
        self.box(name + "_l1", (0.6, 0.5, 0.45), (0.17, 0, 0.21), orange, parent=a1)
        a2 = self.empty(name + "_a2", (0.3528, -0.0375, 0.4192), a1)
        self.box(name + "_l2", (0.32, 0.26, 1.3), (0.0, -0.07, 0.62), orange, parent=a2)
        a3 = self.empty(name + "_a3", (0.0, -0.1475, 1.2499), a2)
        self.box(name + "_l3", (1.0, 0.3, 0.32), (0.45, 0.17, -0.04), orange, parent=a3)
        a4 = self.empty(name + "_a4", (0.958, 0.184, -0.055), a3)
        self.cyl(name + "_l4", 0.11, 0.54, (0.27, 0, 0), steel, parent=a4, axis="x")
        a5 = self.empty(name + "_a5", (0.542, 0.0, 0.0), a4)
        self.box(name + "_l5", (0.22, 0.2, 0.2), (0.09, 0, 0), orange, parent=a5)
        a6 = self.empty(name + "_a6", (0.1925, 0.0, 0.0), a5)
        self.cyl(name + "_l6", 0.06, 0.12, (0.06, 0, 0), black, parent=a6, axis="x")
        # servo C-gun (the thing that actually spot-welds)
        self.box(name + "_gun_a", (0.10, 0.10, 0.45), (0.2, 0, -0.2), steel, parent=a6)
        self.box(name + "_gun_b", (0.45, 0.10, 0.10), (0.4, 0, -0.42), steel, parent=a6)
        tip = self.empty(name + "_tip", (0.45, 0, -0.42), a6)
        return {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6}, tip

    def _market(self):
        shelf = self.material("shelf", (0.1, 0.25, 0.6), rough=0.5, metal=0.4)
        card = self.material("card", (0.55, 0.42, 0.25), rough=0.9)
        for i in range(8):
            x = 6 + i * 11
            self.box(f"mkt_{i}", (8.0, 1.2, 3.0), (x, 2.5, 1.5), shelf)
            for k in range(4):
                self.box(f"mktbox_{i}_{k}", (1.6, 0.9, 0.9), (x - 3 + k * 2, 2.5, 0.5 + (k % 2) * 1.2), card)

    def _data_hall(self):
        dh, s = self.ctx.dh, self.scn
        x0, x1 = s["data_hall"]["x"]
        z0, z1 = s["data_hall"]["z"]
        conc = self.material("dh_floor", (0.55, 0.57, 0.6), rough=0.25)
        crah_m = self.material("crah", (0.8, 0.82, 0.85), rough=0.4, metal=0.3)
        glass = self.glow("contain", (0.6, 0.8, 1.0), 0.15, 0.10)
        rackm = self.material("rack", (0.05, 0.05, 0.06), rough=0.35, metal=0.5)
        led = self.material("rack_led", (0.0, 0.0, 0.0), emit=(0.1, 1.0, 0.2), strength=4.0)
        self.box("dh_floor", (x1 - x0, z1 - z0, 0.05), ((x0 + x1) / 2, (z0 + z1) / 2, 0.025), conc)
        self.box("dh_wallE", (0.3, z1 - z0, 5.0), (x1 + 0.15, (z0 + z1) / 2, 2.5), crah_m)
        self.box("dh_wallN", (x1 - x0, 0.3, 5.0), ((x0 + x1) / 2, z1 + 0.15, 2.5), crah_m)
        self.box("dh_wallS", (x1 - x0, 0.3, 5.0), ((x0 + x1) / 2, z0 - 0.15, 2.5), crah_m)
        self.box("dh_roof", (x1 - x0, z1 - z0, 0.3), ((x0 + x1) / 2, (z0 + z1) / 2, 5.0), crah_m)
        for k in range(2):
            o = self.light(f"dh_light{k}", "AREA", ((x0 + x1) / 2, (z0 + z1) / 2 + (k - 0.5) * 8, 4.8), self.args.light_w * 0.5, (1, 1, 1), 6.0, 1.0)
            self.lights_ceiling.append((o, self.args.light_w * 0.5))
        for pl in dh["racks"]:
            face = 1.0 if pl.yaw_deg == 0.0 else -1.0
            self.box(pl.path.replace("/", "_"), (0.6, 1.0, 2.0), (pl.x, pl.y, 1.0), rackm)
            self.box(pl.path.replace("/", "_") + "_leds", (0.02, 0.5, 1.6), (pl.x + face * 0.51, pl.y, 1.0), led)
            for zi in range(3):
                mp = self.material(f"skin_{pl.meta['row']}_{pl.meta['idx']}_{zi}", (0, 0, 0), emit=(0.1, 0.1, 0.1), strength=1.0)
                o = self.box(f"skin_{pl.meta['row']}_{pl.meta['idx']}_{zi}", (0.62, 1.05, 0.66), (pl.x, pl.y, 0.2 + zi * 0.68 + 0.33), mp, col=self.col_skins)
                self.skins.append((mp, pl.meta["row"], pl.meta["idx"], (zi + 0.5) / 3))
        a = dh["cold_aisle"]
        self.box("contain_roof", (a["w"] + 0.1, a["z1"] - a["z0"], 0.05), (a["x"], (a["z0"] + a["z1"]) / 2, 2.2), glass)
        for k, dz in enumerate((a["z0"], a["z1"])):
            self.box(f"contain_door{k}", (a["w"], 0.05, 2.2), (a["x"], dz, 1.1), glass)
        for pl in dh["crah"]:
            self.box(pl.path.replace("/", "_"), (1.2, 2.4, 2.4), (pl.x, pl.y, 1.2), crah_m)
        self.col_skins.hide_render = True

    def _traffic(self):
        c = self.ctx
        hivis = self.material("hivis", (0.95, 0.4, 0.02), rough=0.8)
        skin = self.material("skin", (0.75, 0.55, 0.45), rough=0.7)
        navy = self.material("navy", (0.08, 0.1, 0.2), rough=0.8)
        steel = self.material("steel", STEEL, rough=0.4, metal=0.8)
        blue = self.material("agvblue", (0.1, 0.35, 0.7), rough=0.4, metal=0.3)
        black = self.material("tyre", (0.02, 0.02, 0.02), rough=0.9)
        green = self.material("tugger", (0.15, 0.45, 0.2), rough=0.5)
        biw = self.material("biw", (0.62, 0.64, 0.66), rough=0.45, metal=0.7)
        skid = self.material("skid", (0.2, 0.2, 0.22), rough=0.6)
        for eid, tr in c.tracks.items():
            r = self.empty(f"ent_{eid}", (0, 0, -50), col=self.col_real)
            if tr.cls == "person":
                self.cyl(eid + "_torso", 0.22, 0.75, (0, 0, 1.05), hivis, parent=r)
                self.cyl(eid + "_legs", 0.17, 0.7, (0, 0, 0.35), navy, parent=r)
                self.sphere(eid + "_head", 0.11, (0, 0, 1.65), skin, parent=r)
                self.cyl(eid + "_helmet", 0.13, 0.08, (0, 0, 1.76), self.material("helmet", (0.95, 0.95, 0.9), rough=0.4), parent=r)
            elif tr.cls == "forklift":
                self.box(eid + "_body", (1.9, 1.05, 0.9), (0, 0, 0.7), self.material("fork_yel", YELLOW, rough=0.5), parent=r)
                self.box(eid + "_mast", (0.12, 0.7, 2.3), (1.05, 0, 1.15), steel, parent=r)
                for sy in (-0.3, 0.3):
                    self.box(eid + f"_fork{sy}", (1.1, 0.12, 0.05), (1.65, sy, 0.1), steel, parent=r)
                for sx in (-0.6, 0.5):
                    for sy in (-0.5, 0.5):
                        self.box(eid + f"_g{sx}{sy}", (0.06, 0.06, 1.35), (sx, sy, 1.55), black, parent=r)
                self.box(eid + "_roof", (1.1, 1.1, 0.05), (-0.05, 0, 2.2), black, parent=r)
                self.box(eid + "_weight", (0.7, 1.05, 0.7), (-1.1, 0, 0.7), black, parent=r)
            elif tr.cls == "agv":
                self.box(eid + "_body", (1.5, 0.8, 0.28), (0, 0, 0.24), blue, parent=r)
                self.box(eid + "_deck", (1.3, 0.7, 0.04), (0, 0, 0.4), steel, parent=r)
            elif tr.cls == "tugger":
                self.box(eid + "_cab", (1.3, 0.9, 1.3), (0, 0, 0.7), green, parent=r)
                for k in range(1, 4):
                    self.box(eid + f"_cart{k}", (1.4, 0.9, 0.35), (-1.7 * k, 0, 0.35), steel, parent=r)
            elif tr.cls == "body":
                if self.args.body_glb:
                    self._body_glb(r, eid)
                else:
                    self._biw(r, eid, biw, skid)
            else:
                self.bpy.data.objects.remove(r)
                continue
            self.prims[eid] = r

    def _biw(self, root, eid, biw, skid):
        lower = self.box(eid + "_lower", (4.6, 1.85, 0.6), (0, 0, 0.8), biw, parent=root)
        cabin = self.box(eid + "_cabin", (2.4, 1.7, 0.6), (-0.2, 0, 1.4), biw, parent=root)
        for o, w, seg in ((lower, 0.16, 4), (cabin, 0.2, 4)):
            m = o.modifiers.new("bevel", "BEVEL")
            m.width, m.segments, m.limit_method = w, seg, "NONE"
            sm = o.modifiers.new("subsurf", "SUBSURF")
            sm.levels = sm.render_levels = 1
        self.box(eid + "_skid", (4.9, 1.2, 0.2), (0, 0, 0.35), skid, parent=root)

    def _body_glb(self, root, eid):
        """Kenney Car Kit sedan (CC0) or a converted BIW. Strips wheels/glass/lights, paints everything bare steel."""
        bpy = self.bpy
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=self.args.body_glb)
        new = [o for o in bpy.data.objects if o not in before]
        steel = self.material("biw", (0.62, 0.64, 0.66), rough=0.45, metal=0.7)
        for o in new:
            nm = o.name.lower()
            if o.type == "MESH" and any(k in nm for k in ("wheel", "tire", "tyre", "glass", "window", "light", "lamp", "steer")):
                bpy.data.objects.remove(o)
                continue
            if o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(steel)
            if o.parent is None:
                o.parent = root
                o.rotation_euler = (math.radians(90) if self.args.body_glb_yup else 0.0, 0.0, math.radians(self.args.body_glb_yaw))
                o.scale = (self.args.body_glb_scale,) * 3
            for c in o.users_collection:
                c.objects.unlink(o)
            self.col_real.objects.link(o)

    def _hero(self):
        h = self.ctx.hero
        r = self.empty(f"ent_{h.entity_id}", (0, 0, -50), col=self.col_real)
        white = self.material("amr_white", (0.9, 0.92, 0.95), rough=0.3)
        teal = self.material("amr_teal", (0.0, 0.5, 0.55), rough=0.3)
        black = self.material("tyre", (0.02, 0.02, 0.02), rough=0.9)
        dark = self.material("mast_head", (0.05, 0.05, 0.07), rough=0.3)
        self.beacon = self.material("beacon", (0.1, 0.1, 0.1), emit=(0.0, 1.0, 0.3), strength=6.0)
        self.box("amr_body", (0.75, 0.55, 0.22), (0, 0, 0.22), white, parent=r)
        self.box("amr_band", (0.77, 0.57, 0.05), (0, 0, 0.3), teal, parent=r)
        for sx in (-0.25, 0.25):
            for sy in (-0.3, 0.3):
                self.cyl(f"amr_w{sx}{sy}", 0.1, 0.06, (sx, sy, 0.1), black, parent=r).rotation_euler = (math.radians(90), 0, 0)
        self.cyl("amr_mast", 0.025, 1.4, (-0.1, 0, 1.0), white, parent=r)
        self.box("amr_head", (0.18, 0.26, 0.16), (-0.1, 0, 1.75), dark, parent=r)
        self.cyl("amr_lens_rgb", 0.025, 0.03, (-0.0, 0.06, 1.75), self.material("lens", (0, 0, 0.02), rough=0.05, metal=1.0), parent=r).rotation_euler = (0, math.radians(90), 0)
        self.cyl("amr_lens_ir", 0.03, 0.03, (-0.0, -0.06, 1.75), self.material("lens_ir", (0.2, 0.0, 0.0), rough=0.05, metal=1.0), parent=r).rotation_euler = (0, math.radians(90), 0)
        self.box("amr_beacon", (0.08, 0.08, 0.08), (-0.1, 0, 1.88), self.beacon, parent=r)
        self.light("amr_beacon_light", "POINT", (0, 0, 0), 15.0, (0.2, 1.0, 0.4), 0.05).parent = r
        self.prims[h.entity_id] = r

    def _ble_rig(self):
        ok = self.material("ble_ok", (0, 0, 0), emit=(0.1, 0.8, 1.0), strength=40.0)
        bad = self.material("ble_bad", (0, 0, 0), emit=(1.0, 0.1, 0.05), strength=40.0)
        tx = self.material("ble_tx", (0.1, 0.1, 0.1), emit=(0.2, 0.6, 1.0), strength=4.0)
        for bi, (bx, bz) in enumerate(self.scn["ble"]["broadcasters"]):
            self.box(f"ble_tx_{bi}", (0.25, 0.25, 0.1), (bx, bz, 9.0), tx)
        self._ble_ok, self._ble_bad = ok, bad
        for k in range(24):
            o = self.sphere(f"ble_pkt_{k}", 0.14, (0, 0, -50), ok, col=self.col_misc)
            self.ble_pool.append(o)

    # ----- per-frame -----
    def _place(self, obj, x, y, yaw, z=0.0):
        obj.location = (x, y, z)
        obj.rotation_euler = (0.0, 0.0, core.YAW_SIGN * yaw + math.radians(core.YAW_OFFSET_DEG))

    def apply(self, t: float, shot: "core.Shot"):
        c, scn = self.ctx, self.scn
        for eid, o in self.prims.items():
            tr = c.tracks.get(eid)
            if tr is None:
                continue
            x, y, yaw = tr.pose(t)
            self._place(o, x, y, yaw, 0.0 if tr.active(t) else -50.0)
            if tr.cls == "person":                                   # walking bob
                o.location.z += 0.03 * abs(math.sin(2 * math.pi * 1.8 * t)) if tr.speed(t) > 0.2 else 0.0
        for m in scn["machines"]:
            mid = m["id"]
            tr = c.tracks.get(mid)
            st = tr.state_at(t) if tr else "IDLE"
            col, a, e = {"ACTIVE": ((1.0, 0.1, 0.05), 0.35, 1.5), "IDLE": ((0.0, 0.7, 0.15), 0.18, 0.8),
                         "HELD": ((1.0, 0.7, 0.0), 0.5, 4.0), "TRIPPED": ((1.0, 0.0, 0.0), 0.7, 6.0),
                         "STOPPED": ((0.7, 0.7, 0.0), 0.3, 1.5)}.get(st, ((0.0, 0.7, 0.15), 0.25, 1.0))
            self.set_emit(self.zone_mat[mid], col, e, a)
            self.set_emit(self.curtain_mat[mid], col, 1.5 + 2.0 * (st in ("ACTIVE", "TRIPPED")), 0.06 + 0.16 * (st in ("ACTIVE", "TRIPPED")))
            active = tr.intervals("ACTIVE") if tr else []
            on, ph = core.phase_of(active, t) if active else (False, 0.0)
            for joints, k, tip in self.arms.get(mid, []):
                q = self.kuka_q(ph, k) if on else self.KUKA_HOME
                for jn, ang in zip(("a1", "a2", "a3", "a4", "a5", "a6"), q):
                    ax = {"a1": 2, "a2": 1, "a3": 1, "a4": 0, "a5": 1, "a6": 0}[jn]
                    e3 = [0.0, 0.0, 0.0]
                    e3[ax] = ang
                    joints[jn].rotation_euler = e3
            if mid in self.sparks:
                sph, li, sm = self.sparks[mid]
                flick = on and 0.2 < ph < 0.85 and ((int(t * 30) * 2654435761) % 7 < 3)
                if flick and self.arms.get(mid):
                    self.bpy.context.view_layer.update()
                    j, k, tip = self.arms[mid][int(t * 30) % len(self.arms[mid])]
                    p = tip.matrix_world.translation
                    sph.location = li.location = p
                else:
                    sph.location = li.location = (0, 0, -50)
                li.data.energy = 4000.0 if flick else 0.0
                self.set_emit(sm, (1.0, 0.7, 0.2), 200.0 if flick else 0.0)
        hs = c.hero.state_at(t)
        self.set_emit(self.beacon, {"enter": (1.0, 0.1, 0.05), "wait": (1.0, 0.6, 0.0), "exit": (1.0, 0.6, 0.0)}.get(hs, (0.0, 1.0, 0.3)), 6.0)
        self.apply_ble(t, shot)
        if shot.thermal:
            for mp, row, idx, zf in self.skins:
                cc = core.rack_temp_c(row, idx, zf, t)
                self.set_emit(mp, core.thermal_lut((cc - 20.0) / 22.0), 1.3)

    # KUKA KR210 (degrees): joint limits a2 [-45,85], a3 [-210,65], a5 +-125
    KUKA_HOME = [math.radians(v) for v in (0, 35, -35, 0, 20, 0)]
    _KW = [[math.radians(v) for v in q] for q in (
        (0, 35, -35, 0, 20, 0), (30, 45, -20, 40, 50, 60), (40, 55, -10, -20, 60, 120), (10, 40, -25, 50, 40, -60),
        (-25, 50, -15, -30, 55, 30), (0, 35, -35, 0, 20, 0))]

    def kuka_q(self, phase: float, k: int):
        n = len(self._KW) - 1
        u = min(max(phase, 0.0), 1.0) * n
        i = min(int(u), n - 1)
        a = u - i
        a = a * a * (3 - 2 * a)
        q = [self._KW[i][j] + a * (self._KW[i + 1][j] - self._KW[i][j]) for j in range(6)]
        sg = 1.0 if k % 2 == 0 else -1.0
        q[0] *= sg
        q[3] *= sg
        q[5] *= sg
        return q

    def apply_ble(self, t, shot):
        for o in self.ble_pool:
            o.location = (0, 0, -50)
        if not shot.show_ble:
            return
        hero, k = self.ctx.hero, 0
        for ev in core.ble_events(self.scn, hero, t - 0.4, t + 1e-3):
            age = t - ev["t"]
            if age < 0 or k >= len(self.ble_pool):
                continue
            u = min(age / 0.35, 1.0)
            rx, ry, _ = hero.pose(t)
            p0, p1 = (ev["bx"], ev["bz"], 9.0), (rx - 0.1, ry, 1.75)
            if not ev["ok"]:
                p1 = core.lerp3(p0, p1, 0.55)
                if u > 0.9:
                    continue
            o = self.ble_pool[k]
            o.location = core.lerp3(p0, p1, u)
            mat = self._ble_ok if ev["ok"] else self._ble_bad
            o.material_slots[0].material = mat
            k += 1

    def set_thermal(self, on: bool):
        self.scene.view_settings.view_transform = "Standard" if on else self._view_transform
        self.col_real.hide_render = on
        self.col_skins.hide_render = not on
        for o, w in self.lights_ceiling:
            o.data.energy = 0.0 if on else w
        self.world_bg.inputs["Strength"].default_value = 0.0 if on else 0.15

    def point_camera(self, cam: "core.Cam"):
        mu = self.mu
        self.cam_obj.data.lens = cam.focal_mm
        eye, tgt = mu.Vector(cam.eye), mu.Vector(cam.tgt)
        self.cam_obj.location = eye
        self.cam_obj.rotation_euler = (tgt - eye).to_track_quat("-Z", "Y").to_euler()


def setup_cycles_device(scene) -> str:
    """OPTIX -> CUDA -> CPU. Returns the backend actually enabled."""
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()                                       # VERIFY (refresh_devices() on older builds)
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                scene.cycles.device = "GPU"
                log("Cycles device:", kind, [d.name for d in gpus])
                return kind
        except Exception as e:  # noqa: BLE001
            log(f"{kind} unavailable: {e!r}")
    scene.cycles.device = "CPU"
    log("Cycles device: CPU (no GPU backend found)")
    return "CPU"


def fetch_polyhaven_hdri(asset_id: str, cache: Path) -> "Path | None":
    """Poly Haven HDRI (CC0). Fetched at run time, cached, never committed."""
    if not asset_id:
        return None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        dst = cache / f"{asset_id}_1k.hdr"
        if dst.exists():
            return dst
        req = urllib.request.Request(f"https://api.polyhaven.com/files/{asset_id}", headers={"User-Agent": "intercycle-render"})
        files = json.load(urllib.request.urlopen(req, timeout=30))
        url = files["hdri"]["1k"]["hdr"]["url"]
        urllib.request.urlretrieve(url, dst)
        return dst
    except Exception as e:  # noqa: BLE001
        log(f"HDRI {asset_id} unavailable ({e!r}); using flat world colour")
        return None


def render_shot(bs: BScene, shot: "core.Shot", a: float, b: float, args, out_dir: Path) -> Path:
    bpy = bs.bpy
    frames = out_dir / "frames" / (shot.name + (f"_{args.label}" if args.label else ""))
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    bs.set_thermal(shot.thermal)
    n = max(1, int((b - a) * args.fps / args.speed))
    for i in range(n):
        t = a + i * args.speed / args.fps
        bs.apply(t, shot)
        bs.point_camera(shot.cam(t, a, b, bs.ctx))
        bs.scene.render.filepath = str(frames / f"f_{i:05d}.png")
        bpy.ops.render.render(write_still=True)
        if i % 30 == 0:
            log(f"[{shot.name}] frame {i}/{n} t={t:.1f}s")
    bs.set_thermal(False)
    pngs = sorted(frames.glob("f_*.png"))
    mp4 = out_dir / (shot.name + (f"_{args.label}" if args.label else "") + ".mp4")
    core.ffmpeg_frames_to_mp4(pngs, mp4, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
    return mp4


def smoke(out_dir: Path) -> Path:
    """Cube on a plane, one area light, device report, 64 spp 640x360 PNG. Proves bpy + GPU + denoiser."""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = 640, 360
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True
    dev = setup_cycles_device(sc)
    if dev == "OPTIX":
        sc.cycles.denoiser = "OPTIX"
    bpy.ops.mesh.primitive_plane_add(size=20)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    li = bpy.data.objects.new("L", bpy.data.lights.new("L", "AREA"))
    li.data.energy = 800
    li.data.size = 4
    li.location = (2, -2, 5)
    sc.collection.objects.link(li)
    cam = bpy.data.objects.new("C", bpy.data.cameras.new("C"))
    cam.location = (6, -6, 4)
    cam.rotation_euler = (math.radians(65), 0, math.radians(45))
    sc.collection.objects.link(cam)
    sc.camera = cam
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "bpy_smoke.png"
    sc.render.filepath = str(dst)
    bpy.ops.render.render(write_still=True)
    log("SMOKE_OK", dst, "device", dev)
    return dst


def run_bpy(scn, ctx, shots, args, out_dir: Path, scene_cls=None):
    bs = (scene_cls or BScene)(scn, ctx, args)
    bs.build()
    outs = []
    for name, p in zip(shots, core.plan(ctx, shots, args.fps, args.t0, args.t1, args.speed)):
        outs.append(str(render_shot(bs, core.SHOTS[name], p["t0"], p["t1"], args, out_dir)))
    (out_dir / "credits_blender.json").write_text(json.dumps([
        dict(credit="Procedural geometry (this repo); arm kinematics from ros-industrial/kuka_experimental (Apache-2.0)", licence="MIT / Apache-2.0"),
        dict(credit=f"Poly Haven HDRI '{args.hdri_id}'", licence="CC0-1.0"),
        dict(credit="All scenes, schedules and temperatures are SYNTHETIC (InterCycle design assumptions).", licence="n/a")], indent=2))
    (out_dir / "manifest_blender.json").write_text(json.dumps(dict(outputs=outs, device=bs.device, args=vars(args)), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Blender/Cycles replay of the InterCycle sim trace")
    ap.add_argument("--scenario", default="scenario/bodyshop.yaml")
    ap.add_argument("--trace", default="trace.parquet")
    ap.add_argument("--out", default="render/out")
    ap.add_argument("--shots", default="overview,curtain,ble,pov,datahall,ab_cam")
    ap.add_argument("--hero", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--light-w", type=float, default=150.0, help="ceiling area-light power (W, ~1 W/m2 over the hall); tune on first render")
    ap.add_argument("--hdri-id", default="autoshop_01")
    ap.add_argument("--cache", default="/tmp/ph_cache")
    ap.add_argument("--body-glb", default=None)
    ap.add_argument("--body-glb-yup", action="store_true")
    ap.add_argument("--body-glb-yaw", type=float, default=0.0)
    ap.add_argument("--body-glb-scale", type=float, default=1.0)
    ap.add_argument("--synth-bodies", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(out_dir)
        return 0
    scn = core.load_scenario(args.scenario)
    tracks = core.load_trace(args.trace)
    ctx = core.build_ctx(scn, tracks, args.hero, args.synth_bodies)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    if args.dry_run:
        class A:  # core.dry_run wants these
            fps, t0, t1, speed = args.fps, args.t0, args.t1, args.speed
        core.dry_run(scn, tracks, ctx, shots, A)
        return 0
    run_bpy(scn, ctx, shots, args, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
