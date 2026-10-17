#!/usr/bin/env python
"""Realistic look for the Blender replay (workstream X, DIRECTION.md §7 X): RealScene extends BScene.

Same trace contract, YAML layout, shots and per-frame replay as replay_bpy.py; differences:
  * plant.py dresses the hall (shell, roof + skylights, LED high-bays, markings, services, clutter);
  * stations look like a body shop instead of a diagram: roller-skid conveyor, fixture tables, mesh guards
    with a light-curtain opening on each side, stack lights driven by the machine state, hanging station
    signs, an overhead controller platform; NO glowing hazard-zone floor decals or curtain planes (those are
    map overlays, not things a CCTV camera sees);
  * weld spark bursts (emissive streaks + flash light) at the welding gun while a cell is ACTIVE;
  * no capsule people: persons are hidden until humans.py (rigged workers) is available;
  * the hero AMR is hidden (robot shots belong to workstream R4 / Isaac).
Stills (one frame each, labelled SYNTHETIC):
    blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/scene_real.py \
        --trace D:/icdata/renders/trace_synth.parquet --synth-bodies --out D:/icdata/renders/stills \
        --stills wide,weld_close,cctv_2 --t 36030 --samples 128 --res 1920x1080
`cctv_<i>` uses camera i of bodyshop.yaml `cameras` (ceiling, 9 m) and is written UNLABELLED as *_clean.png:
the CCTV post (render/cctv) burns in the timestamp + SYNTHETIC label.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assets  # noqa: E402
import fanuc  # noqa: E402
import humans  # noqa: E402
import plant  # noqa: E402
import replay_bpy as rb  # noqa: E402

core = rb.core
CCTV_Z = 9.0
FORKLIFT_YAW_DEG = 90.0       # forklift_c forks point to -Y in the library; +90 deg turns them to +X
BODY_CORNER = (2.1, 0.8)     # m from the station centre: where the corner robots weld (BIW is 4.65 x 1.74 m)
COMPARE_LINE_X, COMPARE_LINE_Z, COMPARE_LINE_DX = 15.5, 7.0, 14.0   # compare_line camera: x, height, look-at run (m)
FRAME_Z = 4.4               # m, top of the overhead cell frame (dress cables, trays); CCTV at 9 m looks over it
END_REACH = 1.8             # m, base -> gun tip of the FANUC weld poses (fanuc.WELD_Q)
ARM_REACH_STANDOFF = 2.65    # m, line centre -> FANUC base for on-line cells
STACK = {"ACTIVE": "g", "IDLE": "g", "STOPPED": "y", "HELD": "y", "TRIPPED": "r"}


class RealScene(rb.BScene):
    def build(self):
        self.rng = random.Random(5)
        self._render_settings()
        self._world()
        plant.build(self, self.args.light_e)
        self._line()
        self._data_hall()
        self._traffic()
        self._hero()
        self._hide_hero()
        self._ble_rig()
        for o in self.ble_pool:
            o.hide_render = True
        self.cam_obj = self.bpy.data.objects.new("Cam", self.bpy.data.cameras.new("Cam"))
        self.col_misc.objects.link(self.cam_obj)
        self.cam_obj.data.sensor_width = 36.0
        self.cam_obj.data.clip_end = 1000.0
        self.scene.camera = self.cam_obj

    # ----- world: daylight through skylights and wall bands only -----
    def _world(self):
        super()._world()
        self.world_bg.inputs["Strength"].default_value = self.args.sky_strength
        if self.args.haze > 0:                                  # weld-fume haze: soft shafts under the skylights
            nt = self.scene.world.node_tree
            vs = nt.nodes.new("ShaderNodeVolumeScatter")
            vs.inputs["Density"].default_value = self.args.haze
            vs.inputs["Anisotropy"].default_value = 0.3
            out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
            nt.links.new(vs.outputs[0], out.inputs["Volume"])
            self.scene.cycles.volume_bounces = 1

    def set_thermal(self, on: bool):
        super().set_thermal(on)                      # BScene resets the world to its own 0.15 when leaving thermal
        if not on:
            self.world_bg.inputs["Strength"].default_value = self.args.sky_strength

    def _hide_hero(self):
        r = self.prims.pop(self.ctx.hero.entity_id, None)
        if r is not None:
            for o in [r, *r.children_recursive]:
                o.hide_render = True

    # ----- stations -----
    def _line(self):
        s = self.scn
        dark, galv = plant.mat(self, "dark_steel"), plant.mat(self, "galv")
        xs = [m["x"] for m in s["machines"] if m.get("on_line")]
        x0, x1 = min(xs) - 7.0, max(xs) + 7.0
        self._conveyor(x0, x1, 24.0, dark, galv)
        self._platform(x0, x1)
        for m in s["machines"]:
            mid, kind = m["id"], m["kind"]
            w, d = m["size"]
            cx, cy = m["x"], m["z"]
            self._guards(m)
            self._stack_light(mid, cx + w / 2 - 0.6, cy + d / 2 + 0.25)
            self._sign(mid, cx, cy + d / 2 + 0.3 if m.get("on_line") else cy - d / 2 - 0.3)
            if kind in ("weld_station", "respot", "framing", "roller_hemming", "sealer"):
                self._fixture(m)
                self._spark_pool(mid)
            if kind == "framing":
                self._framing_gates(m)
            if kind == "unload_lifter":
                self.box(f"lifter_{mid}", (w - 2.5, 2.6, 0.5), (cx, cy, 0.3), plant.mat(self, "safety_yellow"))
                for sx in (-1, 1):
                    self.box(f"lifter_{mid}_col{sx}", (0.4, 0.4, 4.5), (cx + sx * (w / 2 - 1.5), cy - 2.0, 2.25), plant.mat(self, "ral7035"))
            self.arms[mid] = []
            # On-line robots stand ARM_REACH_STANDOFF from the line centre so the FANUC weld poses (gun tip ~1.7-1.85 m in
            # front of the base) land on the body side (half-width ~0.9 m); replay.py's 3.2 m was sized for the KR210.
            def _y(pl_y):
                return cy + math.copysign(ARM_REACH_STANDOFF, pl_y - cy) if m.get("on_line") else pl_y
            places = [(pl.path.replace("/", "_"), pl.x, _y(pl.y), pl.yaw_deg, pl.meta["idx"] + (2 if pl.meta["side"] == "S" else 0), True)
                      for pl in core.arm_layout(m)]
            if kind in ("weld_station", "respot") and w >= 8:   # real weld cells are dense: a third robot per side
                for side, sg in (("N", 1), ("S", -1)):
                    places.append((f"_Line_{mid}_arm_{side}mid", cx, cy + sg * ARM_REACH_STANDOFF, -90.0 * sg, 4 + (sg < 0), False))
            if kind in ("weld_station", "respot"):              # and one robot per corner of the body, angled onto it
                places += self._end_robots(mid, cx, cy, w)
            for name, x, y, yaw, k, cabinet in places:
                joints, tip = fanuc.build_fanuc(name, (x, y, 0.0), yaw, collection=self.col_real)
                self.arms[mid].append((joints, k, tip))
                self.box(name + "_riser", (1.0, 1.0, 0.05), (x, y, 0.025), dark)
                if cabinet and m.get("on_line"):                 # robot controller + weld timer behind each corner robot
                    sg = 1 if y > cy else -1
                    yc = cy + sg * (d / 2 - 0.45)
                    self.box(name + "_ctrl", (0.75, 0.55, 1.45), (x, yc, 0.73), plant.mat(self, "ral7035"))
                    self.box(name + "_ctrl_hmi", (0.2, 0.01, 0.15), (x, yc - sg * 0.28, 1.1), plant.glow_cam_only(self, "hmi", (0.2, 0.5, 0.9), 2.0))
                    self.box(name + "_valve", (0.5, 0.25, 0.35), (x + 0.65, yc, 1.2), plant.mat(self, "steel_grey"))
            if m.get("on_line") and kind in ("weld_station", "respot", "framing"):
                bases = [(x, y) for _, x, y, _, _, _ in places]
                for name, x, y, yaw, _, _ in places:
                    self._tip_dresser(name, x, y, yaw, m, bases)
                self._cell_frame(m, places)

    def _tip_dresser(self, name, x, y, yaw, m, bases):
        """Electrode tip dresser on a pedestal beside the robot, on whichever side stays inside the cell rect and
        farthest from the other robot bases and the skid tooling (|y - line| > 1.6 m)."""
        cx, cy, (w, d) = m["x"], m["z"], m["size"]
        hx, hy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        best = None
        for sd in (1, -1):                                      # beside the base and 0.3 m behind it, out of the gun's path
            px, py = x - sd * 0.85 * hy - 0.3 * hx, y + sd * 0.85 * hx - 0.3 * hy
            if abs(px - cx) > w / 2 - 0.3 or abs(py - cy) > d / 2 - 0.3 or abs(py - cy) < 1.6:
                continue
            clear = min(math.hypot(px - bx, py - by) for bx, by in bases if (bx, by) != (x, y))   # other robots only
            if best is None or clear > best[0]:
                best = (clear, px, py)
        if best is None or best[0] < 0.7:
            return
        _, px, py = best
        self.box(name + "_td_post", (0.14, 0.14, 1.0), (px, py, 0.5), plant.mat(self, "steel_grey"))
        self.box(name + "_td_head", (0.3, 0.22, 0.16), (px, py, 1.08), plant.mat(self, "ral7035"))
        self.cyl(name + "_td_motor", 0.06, 0.22, (px, py, 1.27), plant.mat(self, "dark_steel"))

    def _cell_frame(self, m, places):
        """Overhead steel frame inside the cell rect: four corner posts, edge and end beams at FRAME_Z, a cable
        tray over each robot row and a dress-cable drop from the tray to every robot (behind its base)."""
        mid, cx, cy, (w, d) = m["id"], m["x"], m["z"], m["size"]
        st, galv, cab = plant.mat(self, "ral7035"), plant.mat(self, "galv"), plant.mat(self, "rubber")
        hx, hy = w / 2 - 0.2, d / 2 - 0.2
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.box(f"cf_{mid}_post{sx}{sy}", (0.2, 0.2, FRAME_Z), (cx + sx * hx, cy + sy * hy, FRAME_Z / 2), st)
            self.box(f"cf_{mid}_end{sx}", (0.2, 2 * hy, 0.3), (cx + sx * hx, cy, FRAME_Z - 0.15), st)
        for sy in (-1, 1):
            self.box(f"cf_{mid}_edge{sy}", (2 * hx, 0.2, 0.3), (cx, cy + sy * hy, FRAME_Z - 0.15), st)
            ty = cy + sy * ARM_REACH_STANDOFF
            self.box(f"cf_{mid}_tray{sy}", (2 * hx, 0.45, 0.08), (cx, ty, FRAME_Z - 0.35), galv)
            self.box(f"cf_{mid}_cables{sy}", (2 * hx, 0.32, 0.06), (cx, ty, FRAME_Z - 0.28), cab)
        for name, x, y, yaw, _, _ in places:
            bx, by = x - 0.35 * math.cos(math.radians(yaw)), y - 0.35 * math.sin(math.radians(yaw))
            top = FRAME_Z - 0.35
            self.cyl(name + "_drop", 0.045, top - 2.3, (bx, by, 2.3 + (top - 2.3) / 2), cab)
            self.box(name + "_drop_arm", (0.08, abs(by - (cy + math.copysign(ARM_REACH_STANDOFF, by - cy))) + 0.08, 0.08),
                     (bx, (by + cy + math.copysign(ARM_REACH_STANDOFF, by - cy)) / 2, top), galv)

    @staticmethod
    def _end_robots(mid, cx, cy, w):
        """Four robots facing the body's corners from inside the cell rect. The FANUC weld poses put the gun
        END_REACH in front of the base, so each base sits END_REACH from its corner, as far out along x as the
        cell allows (base half-width 0.55 m inside the rect)."""
        bx = min(w / 2 - 0.55, BODY_CORNER[0] + END_REACH * math.cos(math.radians(45)))
        by = BODY_CORNER[1] + math.sqrt(max(END_REACH ** 2 - (bx - BODY_CORNER[0]) ** 2, 0.0))
        out = []
        for side, sg in (("N", 1), ("S", -1)):
            for end, sx in (("W", -1), ("E", 1)):
                yaw = math.degrees(math.atan2(sg * (BODY_CORNER[1] - by), sx * (BODY_CORNER[0] - bx)))
                out.append((f"_Line_{mid}_arm_{side}{end}", cx + sx * bx, cy + sg * by, yaw, 6 + (sx < 0) + 2 * (sg < 0), False))
        return out

    def _conveyor(self, x0, x1, y, dark, galv):
        """Roller-bed skid conveyor: two side frames, rollers every 0.6 m, legs every 2.4 m."""
        L = x1 - x0
        for s in (-1, 1):
            self.box(f"conv_frame{s}", (L, 0.12, 0.22), ((x0 + x1) / 2, y + s * 0.62, 0.42), plant.mat(self, "ral7035"))
        n = int(L / 0.6)
        for k in range(n):
            o = self.cyl(f"conv_roll{k}", 0.045, 1.15, (x0 + 0.3 + k * 0.6, y, 0.47), galv)
            o.rotation_euler = (math.radians(90), 0.0, 0.0)
        for k in range(int(L / 2.4) + 1):
            for s in (-1, 1):
                self.box(f"conv_leg{k}{s}", (0.08, 0.08, 0.32), (x0 + k * 2.4, y + s * 0.62, 0.16), dark)
        for k in range(int(L / 9)):                          # drive units
            self.box(f"conv_drive{k}", (0.45, 0.35, 0.35), (x0 + 4.5 + k * 9.0, y - 0.9, 0.35), plant.mat(self, "steel_grey"))

    def _platform(self, x0, x1):
        """Overhead controller platform along the south side of the line (weld timers, robot cabinets)."""
        y0, y1, z = 16.6, 19.2, 5.2
        st, ral = plant.mat(self, "steel_grey"), plant.mat(self, "ral7035")
        self.box("plat_deck", (x1 - x0, y1 - y0, 0.06), ((x0 + x1) / 2, (y0 + y1) / 2, z), plant.mat(self, "grating"))
        for s, yy in ((0, y0), (1, y1)):
            self.box(f"plat_beam{s}", (x1 - x0, 0.2, 0.35), ((x0 + x1) / 2, yy, z - 0.2), st)
            self.box(f"plat_rail{s}", (x1 - x0, 0.05, 0.05), ((x0 + x1) / 2, yy, z + 1.1), plant.mat(self, "safety_yellow"))
            self.box(f"plat_rail{s}m", (x1 - x0, 0.04, 0.04), ((x0 + x1) / 2, yy, z + 0.55), plant.mat(self, "safety_yellow"))
        for k in range(int((x1 - x0) / 6) + 1):
            for yy in (y0, y1):
                self.box(f"plat_col{k}_{yy}", (0.25, 0.25, z), (x0 + k * 6.0, yy, z / 2), st)
        x = x0 + 1.5
        while x < x1 - 1.0:                                   # cabinets on the deck
            w = self.rng.choice((0.8, 1.0, 1.2))
            self.box(f"cab_{x:.1f}", (w, 0.6, 2.0), (x, y0 + 0.6, z + 1.03), ral)
            self.box(f"cab_{x:.1f}_hmi", (0.25, 0.01, 0.2), (x, y0 + 0.29, z + 1.5), plant.glow_cam_only(self, "hmi", (0.2, 0.5, 0.9), 2.0))
            x += w + self.rng.choice((0.05, 0.05, 0.6))
        for k in range(int((x1 - x0) / 4)):                   # dress-pack cable drops to the cell
            self.cyl(f"drop{k}", 0.05, z - 2.2, (x0 + 2.0 + k * 4.0, y1 + 0.3, 2.2 + (z - 2.2) / 2), plant.mat(self, "rubber"))

    def _guards(self, m):
        """2.2 m mesh guard panels on the boundary; one 3.2 m light-curtain opening per long side (on-line
        cells: north and south; closures cells: south only). End faces of on-line cells stay open to the line."""
        mid, w, d = m["id"], *m["size"]
        cx, cy = m["x"], m["z"]
        mesh_m, post_m = plant.mat(self, "mesh"), plant.mat(self, "safety_yellow")
        sides = [("N", +1), ("S", -1)]
        for side, sg in sides:
            yy = cy + sg * (d / 2 + 0.15)
            opening = m.get("on_line") or side == "S"
            segs = [(cx - w / 2, cx - 1.6), (cx + 1.6, cx + w / 2)] if opening else [(cx - w / 2, cx + w / 2)]
            for k, (a, b) in enumerate(segs):
                self._fence_run(f"g_{mid}_{side}{k}", a, b, yy, "x", mesh_m, post_m)
            if opening:
                for ex in (-1, 1):
                    self.box(f"lc_{mid}_{side}{ex}", (0.06, 0.06, 1.9), (cx + ex * 1.6, yy, 0.95), plant.mat(self, "dark_steel"))
                    self.box(f"lc_{mid}_{side}{ex}_led", (0.006, 0.006, 1.7), (cx + ex * 1.6 - ex * 0.033, yy, 0.95),
                             plant.glow_cam_only(self, "lc_led", (1.0, 0.05, 0.02), 0.8))
        if not m.get("on_line"):
            for ex in (-1, 1):
                self._fence_run(f"g_{mid}_E{ex}", cy - d / 2, cy + d / 2, cx + ex * (w / 2 + 0.15), "y", mesh_m, post_m)

    def _fence_run(self, name, a, b, c, axis, mesh_m, post_m, h=2.2):
        n = max(1, round((b - a) / 1.5))
        step = (b - a) / n
        for k in range(n + 1):
            p = a + k * step
            loc = (p, c, h / 2) if axis == "x" else (c, p, h / 2)
            self.box(f"{name}_p{k}", (0.06, 0.06, h), loc, post_m)
        mid = (a + b) / 2
        if axis == "x":
            self.box(f"{name}_mesh", (b - a, 0.01, h - 0.15), (mid, c, (h - 0.15) / 2 + 0.15), mesh_m)
        else:
            self.box(f"{name}_mesh", (0.01, b - a, h - 0.15), (c, mid, (h - 0.15) / 2 + 0.15), mesh_m)

    def _stack_light(self, mid, x, y):
        self.cyl(f"stk_{mid}_pole", 0.02, 2.3, (x, y, 1.15), plant.mat(self, "dark_steel"))
        mats = {}
        for k, (c, rgb) in enumerate((("g", (0.05, 1.0, 0.15)), ("y", (1.0, 0.6, 0.0)), ("r", (1.0, 0.05, 0.02)))):
            mm = self.material(f"stk_{mid}_{c}", (0.05, 0.05, 0.05), rough=0.3, emit=rgb, strength=0.0)
            self.cyl(f"stk_{mid}_{c}", 0.045, 0.09, (x, y, 2.35 + k * 0.095), mm)
            mats[c] = (mm, rgb)
        self.stack = getattr(self, "stack", {})
        self.stack[mid] = mats

    def _sign(self, mid, x, y):
        bpy = self.bpy
        board = plant.proc_mat(self, "sign_blue", (0.02, 0.12, 0.42), rough=0.4, var=0.05, bump=0.0, grime=0.1)
        z = 6.4
        self.box(f"sign_{mid}", (1.6, 0.04, 0.8), (x, y, z), board)
        for sx in (-0.6, 0.6):
            self.cyl(f"sign_{mid}_wire{sx}", 0.004, plant.ROOF_EAVE - z, (x + sx, y, (plant.ROOF_EAVE + z) / 2), plant.mat(self, "galv"))
        cu = bpy.data.curves.new(f"signtxt_{mid}", "FONT")
        cu.body, cu.size, cu.align_x, cu.align_y, cu.extrude = mid, 0.5, "CENTER", "CENTER", 0.004
        font = self._sign_font()
        if font is not None:
            cu.font = font
        cu.materials.append(self.material("sign_white", (0.9, 0.9, 0.9), rough=0.5))
        for sg in (-1, 1):
            o = bpy.data.objects.new(f"signtxt_{mid}{sg}", cu)
            self.col_real.objects.link(o)
            o.location = (x, y + sg * 0.03, z)
            o.rotation_euler = (math.radians(90), 0.0, 0.0 if sg < 0 else math.radians(180))

    def _sign_font(self):
        """A plain bold sans font for station boards (Blender's built-in font makes '1' read as 'I')."""
        if not hasattr(self, "_font"):
            self._font = None
            for f in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
                if Path(f).exists():
                    self._font = self.bpy.data.fonts.load(f)
                    break
        return self._font

    def _fixture(self, m):
        """Body fixture: a dark base table under the skid, locating pins, pneumatic clamps (blue bodies, red arms)."""
        mid, cx, cy = m["id"], m["x"], m["z"]
        on = m.get("on_line")
        dark = plant.mat(self, "dark_steel")
        if not on:
            self.box(f"fx_{mid}_table", (4.2, 2.2, 0.85), (cx, cy, 0.425), dark)
            self._biw_static(f"fx_{mid}_part", cx, cy, 0.85, panel=True)
        blue = plant.proc_mat(self, "clamp_blue", (0.05, 0.2, 0.55), rough=0.35, var=0.06, bump=0.0, grime=0.2)
        red = plant.proc_mat(self, "clamp_red", (0.6, 0.05, 0.03), rough=0.35, var=0.06, bump=0.0, grime=0.2)
        for k in range(6):
            x = cx - 2.0 + k * 0.8
            for s in (-1, 1):
                y = cy + s * (0.95 if on else 1.2)
                self.box(f"fx_{mid}_cb{k}{s}", (0.12, 0.12, 0.55), (x, y, 0.55 + (0.0 if on else 0.4)), blue)
                self.box(f"fx_{mid}_ca{k}{s}", (0.06, 0.25, 0.06), (x, y - s * 0.12, 0.85 + (0.0 if on else 0.4)), red)
        if on:                                                 # tooling frame both sides of the skid, as in real BIW cells
            white = plant.proc_mat(self, "tool_white", (0.8, 0.8, 0.78), rough=0.4, var=0.05, bump=0.0, grime=0.3)
            for s in (-1, 1):
                y = cy + s * 1.3
                self.box(f"fx_{mid}_rail{s}", (5.2, 0.3, 0.3), (cx, y, 0.15), red)
                for k in range(4):
                    x = cx - 2.1 + k * 1.4
                    self.box(f"fx_{mid}_tw{k}{s}", (0.18, 0.18, 0.75), (x, y, 0.65), red)
                    self.box(f"fx_{mid}_tp{k}{s}", (0.3, 0.3, 0.04), (x, y, 1.04), white)
                    self.box(f"fx_{mid}_tc{k}{s}", (0.08, 0.35, 0.08), (x, y - s * 0.2, 1.1), plant.mat(self, "dark_steel"))
                self.box(f"fx_{mid}_vt{s}", (0.6, 0.2, 0.3), (cx + 2.2, y + s * 0.3, 0.5), plant.mat(self, "galv"))

    def _biw_static(self, name, x, y, z, panel=False):
        steel = plant.proc_mat(self, "biw_steel", (0.6, 0.62, 0.64), rough=0.32, metal=0.85, var=0.1, scale=3.0, bump=0.05, grime=0.15)
        if panel:                                              # a door/hood panel in a hemming/sealing nest
            o = self.box(name, (1.4, 1.1, 0.05), (x, y, z + 0.05), steel)
            o.rotation_euler = (math.radians(4), 0, 0)
            return o
        return self.box(name, (4.4, 1.8, 0.9), (x, y, z + 0.45), steel)

    def _framing_gates(self, m):
        mid, cx, cy = m["id"], m["x"], m["z"]
        st = plant.proc_mat(self, "gate_steel", (0.18, 0.2, 0.22), rough=0.4, metal=0.7)
        for s in (-1, 1):
            y = cy + s * 1.6
            for ex in (-1, 1):
                self.box(f"gate_{mid}_{s}{ex}", (0.3, 0.3, 2.9), (cx + ex * 2.2, y, 1.45), st)
            self.box(f"gate_{mid}_{s}top", (4.7, 0.3, 0.3), (cx, y, 2.9), st)
            for k in range(4):
                self.box(f"gate_{mid}_{s}cl{k}", (0.12, 0.35, 0.12), (cx - 1.5 + k, y - s * 0.3, 2.1 - (k % 2) * 0.9), plant.mat(self, "safety_yellow"))

    def _spark_pool(self, mid, n=36):
        sm = glow = plant.glow_cam_only(self, "spark", (1.0, 0.62, 0.18), 60.0)
        pool = [self.cyl(f"spk_{mid}_{k}", 0.004, 1.0, (0, 0, -50), sm) for k in range(n)]
        flash = self.light(f"flash_{mid}", "POINT", (0, 0, -50), 0.0, (1.0, 0.85, 0.65), 0.05)
        self.sparks[mid] = (pool, flash, glow)

    # ----- traffic: SimReady body-in-white + forklift, rigged MakeHuman workers instead of capsules -----
    def _asset_copy(self, root, name: str, yaw_deg: float = 0.0) -> None:
        """Parent local objects that share the library's linked meshes to `root`. Not a collection instance: each
        entity's own objects carry its pass_index for the ground-truth pass, like every other entity."""
        bpy, mu = self.bpy, self.mu
        col = bpy.data.collections.get(name) or assets.load_library(names=[name])[0]
        rot = mu.Matrix.Rotation(math.radians(yaw_deg), 4, "Z")
        for src in col.all_objects:
            if src.type != "MESH":
                continue
            o = bpy.data.objects.new(f"{root.name}.{src.name}", src.data)
            self.col_real.objects.link(o)
            o.parent = root
            o.matrix_parent_inverse = mu.Matrix.Identity(4)
            o.matrix_basis = rot @ src.matrix_world

    def _biw(self, root, eid, biw, skid):                       # replay_bpy hook: real BIW (+X = car front)
        self._asset_copy(root, "BIW")

    def _forklift(self, eid):
        """forklift_c with its forks turned to +X, the heading of the replay's procedural forklift."""
        r = self.empty(f"ent_{eid}", (0, 0, -50), col=self.col_real)
        self._asset_copy(r, "forklift_c", FORKLIFT_YAW_DEG)
        return r

    def _traffic(self):
        persons = sorted(eid for eid, tr in self.ctx.tracks.items() if tr.cls == "person")
        forklifts = sorted(eid for eid, tr in self.ctx.tracks.items() if tr.cls == "forklift")
        saved = {eid: self.ctx.tracks.pop(eid) for eid in persons + forklifts}
        try:
            super()._traffic()
        finally:
            self.ctx.tracks.update(saved)
        for eid in forklifts:
            self.prims[eid] = self._forklift(eid)
        self.workers: dict[str, object] = {}
        if not persons or not Path(humans.LIB_PATH).exists():
            if persons:
                rb.log(f"WARNING no worker library at {humans.LIB_PATH}: {len(persons)} persons stay hidden")
            return
        names = humans.load_workers()
        for k, eid in enumerate(persons):                       # stable worker choice + phase per entity
            root = humans.spawn(names[k % len(names)], (0.0, 0.0, -50.0), 0.0, "walk", phase_s=0.37 * k, collection=self.col_real)
            self.prims[eid] = root
            self.workers[eid] = root

    def _pose_workers(self, t: float) -> None:
        """Walk when the track moves (cycle time scaled by speed so the feet don't slide), idle otherwise."""
        for eid, root in self.workers.items():
            tr = self.ctx.tracks[eid]
            v = tr.speed(t)
            if v > 0.2:
                humans.set_time(root, t * v / humans.WALK_SPEED, "walk")
            else:
                humans.set_time(root, t, "idle")

    # ----- per frame -----
    def apply(self, t: float, shot=None):
        c, scn = self.ctx, self.scn
        for eid, o in self.prims.items():
            tr = c.tracks.get(eid)
            if tr is None:
                continue
            x, y, yaw = tr.pose(t)
            self._place(o, x, y, yaw, 0.0 if tr.active(t) else -50.0)
        self._pose_workers(t)
        for m in scn["machines"]:
            mid = m["id"]
            tr = c.tracks.get(mid)
            st = tr.state_at(t) if tr else "IDLE"
            lit = STACK.get(st, "g")
            for col, (mm, rgb) in self.stack[mid].items():
                self.set_emit(mm, rgb, 25.0 if col == lit else 0.0)
            active = tr.intervals("ACTIVE") if tr else []
            on, ph = core.phase_of(active, t) if active else (False, 0.0)
            for joints, k, _tip in self.arms.get(mid, []):
                fanuc.set_q(joints, fanuc.weld_q(ph, k) if on else fanuc.HOME)
            if mid in self.sparks:
                self._sparks(mid, t, on, ph)
        if shot is not None and shot.thermal:        # data-hall thermal pass (SYNTHETIC temperatures)
            for mp, row, idx, zf in self.skins:
                self.set_emit(mp, core.thermal_lut((core.rack_temp_c(row, idx, zf, t) - 20.0) / 22.0), 1.3)

    def _sparks(self, mid, t, on, ph):
        pool, flash, _ = self.sparks[mid]
        fi = int(t * 30)
        weld = on and 0.2 < ph < 0.85 and ((fi * 2654435761 + sum(map(ord, mid))) % 5 < 2) and self.arms.get(mid)
        for o in pool:
            o.location = (0, 0, -50)
        flash.location = (0, 0, -50)
        flash.data.energy = 0.0
        if not weld:
            return
        self.bpy.context.view_layer.update()
        joints, _, tip = self.arms[mid][fi % len(self.arms[mid])]
        p = tip.matrix_world.translation.copy()
        mu = self.mu
        out = joints["a1"].matrix_world.translation - p      # spatter leaves the electrode gap on the robot's (open) side,
        out.z = 0.0                                          # not into the panel the gun is clamped on
        out = out.normalized() if out.length > 1e-6 else mu.Vector((0.0, 0.0, 0.0))
        rng = random.Random(fi * 131 + len(mid))
        for o in pool[: rng.randint(len(pool) // 3, len(pool))]:
            d = (out * 1.3 + mu.Vector((rng.gauss(0, 0.8), rng.gauss(0, 0.8), rng.gauss(-0.3, 0.7)))).normalized()
            ln = rng.uniform(0.05, 0.35)
            o.location = p + d * (rng.uniform(0.02, 0.4) + ln / 2)
            o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
            o.scale = (0.008, 0.008, ln)
        flash.location = p
        flash.data.energy = rng.uniform(150.0, 600.0)


# =====================================================================================
# stills
# =====================================================================================
def still_cam(name: str, scn: dict) -> "core.Cam":
    if name == "wide":                                         # high corner view down the whole line
        return core.Cam((1.2, 46.5, 8.4), (50.0, 19.0, 0.0), 19.0)
    if name == "compare_line":                                 # the reference photo's viewpoint: above the line axis near
        y = next(m["z"] for m in scn["machines"] if m.get("on_line"))  # S20, looking down it (+x), ~95 deg wide lens
        return core.Cam((COMPARE_LINE_X, y, COMPARE_LINE_Z), (COMPARE_LINE_X + COMPARE_LINE_DX, y, 0.0), 16.5)
    if name == "weld_close":                                   # in the S30 north light-curtain opening, on the middle robot's gun
        m = next(mm for mm in scn["machines"] if mm["id"] == "S30")
        y = m["z"] + m["size"][1] / 2
        return core.Cam((m["x"] - 0.6, y + 1.7, 1.9), (m["x"] + 0.4, m["z"] + 1.6, 1.1), 35.0)
    if name.startswith("cctv_"):
        i = int(name.split("_")[1])
        x, y = scn["cameras"][i]
        dh = scn["data_hall"]
        if x > scn["hall"]["x"][1]:                            # data-hall cameras look at the hall centre
            tgt = (sum(dh["x"]) / 2, sum(dh["z"]) / 2, 0.0)
        else:
            tgt = (x + 10.0 if x < 60 else x - 10.0, y + 16.0 if y < 24 else y - 16.0, 0.0)
        return core.Cam((x, y, CCTV_Z), tgt, 17.5)            # ~91 deg horizontal, like a 2.8 mm CCTV lens
    raise SystemExit(f"unknown still {name!r}: wide | compare_line | weld_close | cctv_<i>")


def label_synthetic(png: Path) -> None:
    """Burn a small 'SYNTHETIC RENDER' label (bottom-left) into a non-CCTV still."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    sz = max(14, im.height // 45)
    font = None
    for f in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(f, sz)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    d.text((sz, im.height - 2 * sz), "SYNTHETIC RENDER", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(png)


# =====================================================================================
# ground truth (CCTV frames): instance ids + depth, in the layout render/cctv/gt.py reads
# =====================================================================================
def assign_ids(bs) -> dict[str, dict]:
    """Give every mesh of every tracked entity, and of every welding arm, an object pass index.
    Returns {id: {"cls", "entity_id"}}; 0 stays background/static plant."""
    ids: dict[str, dict] = {}
    nxt = 1
    groups = [(eid, bs.ctx.tracks[eid].cls, root) for eid, root in bs.prims.items() if eid in bs.ctx.tracks]
    for mid, arms in bs.arms.items():
        for k, (joints, _, _) in enumerate(arms):
            root = joints["a1"]
            while root.parent is not None:
                root = root.parent
            groups.append((f"{mid}/arm{k}", "robot_arm", root))
    for eid, cls, root in groups:
        for o in [root, *root.children_recursive]:
            o.pass_index = nxt
        ids[str(nxt)] = {"cls": cls, "entity_id": eid}
        nxt += 1
    vl = bs.scene.view_layers[0]
    vl.use_pass_object_index = True
    vl.use_pass_z = True
    return ids


def write_gt(bs, stem: str, gt_dir: Path, ids: dict) -> None:
    """Save the render result's passes as multilayer EXR, then split into <stem>_inst.png (uint16),
    <stem>_depth.npy (float32 m, inf = sky/none) and <stem>_ids.json."""
    import numpy as np
    import OpenEXR  # type: ignore
    from PIL import Image
    gt_dir.mkdir(parents=True, exist_ok=True)
    exr = gt_dir / f"{stem}.exr"
    st = bs.scene.render.image_settings
    saved = (st.media_type, st.file_format, st.color_depth, st.use_exr_interleave)
    # Blender 5.x: set media_type MULTI_LAYER_IMAGE first (its only format is OPEN_EXR_MULTILAYER); interleaved =
    # one part, so the channels come back as "<layer>.Object Index.X" / "<layer>.Depth.Z"
    try:
        st.media_type = "MULTI_LAYER_IMAGE"
        st.file_format, st.color_depth, st.use_exr_interleave = "OPEN_EXR_MULTILAYER", "32", True
        bs.bpy.data.images["Render Result"].save_render(str(exr), scene=bs.scene)
    finally:
        st.media_type = saved[0]
        st.file_format, st.color_depth, st.use_exr_interleave = saved[1:]
    with OpenEXR.File(str(exr), separate_channels=True) as f:
        ch = f.channels()
        idx = next(v.pixels for k, v in ch.items() if k.endswith(("Object Index.X", "IndexOB.X")))   # 5.x name, 4.x name
        z = next(v.pixels for k, v in ch.items() if k.endswith("Depth.Z"))
    inst = np.rint(idx).astype(np.uint16)
    Image.fromarray(inst).save(gt_dir / f"{stem}_inst.png")
    z = z.astype(np.float32)
    z[z > 1e6] = np.inf
    np.save(gt_dir / f"{stem}_depth.npy", z)
    (gt_dir / f"{stem}_ids.json").write_text(json.dumps(ids, indent=1))
    exr.unlink()


def run_stills(scn, ctx, args, out_dir: Path) -> list[str]:
    bs = RealScene(scn, ctx, args)
    bs.build()
    ids = assign_ids(bs) if args.gt else {}
    outs = []
    W, H = (int(v) for v in args.res.split("x"))
    for name in [s.strip() for s in args.stills.split(",") if s.strip()]:
        bs.apply(args.t)
        bs.point_camera(still_cam(name, scn))
        # CCTV stills are rendered with `--overscan` extra field per side (same focal, wider sensor) so the
        # barrel distortion in render/cctv can keep the full nominal view: pass the same value to cctv_post.
        k = 1.0 / (1.0 - 2.0 * args.overscan) if name.startswith("cctv_") else 1.0
        bs.cam_obj.data.sensor_width = 36.0 * k
        bs.scene.render.resolution_x, bs.scene.render.resolution_y = round(W * k), round(H * k)
        dst = out_dir / (f"{name}_clean.png" if name.startswith("cctv_") else f"{name}.png")
        bs.scene.render.filepath = str(dst)
        bs.bpy.ops.render.render(write_still=True)
        if not name.startswith("cctv_"):
            label_synthetic(dst)
        elif args.gt:
            write_gt(bs, dst.stem, out_dir / "gt", ids)
        rb.log("still", dst)
        outs.append(str(dst))
    (out_dir / "stills_manifest.json").write_text(json.dumps(dict(stills=outs, t=args.t, device=bs.device, args=vars(args)), indent=2, default=str))
    return outs


def _key_pose(bs, frame: int) -> None:
    """Keyframe everything apply() moves (entity roots, arm joints) so Cycles can motion-blur between frames."""
    for o in bs.prims.values():
        o.keyframe_insert("location", frame=frame)
        o.keyframe_insert("rotation_euler", frame=frame)
    for arms in bs.arms.values():
        for joints, _, _ in arms:
            for j in joints.values():
                j.keyframe_insert("rotation_euler", frame=frame)


def run_cctv_seq(scn, ctx, args, out_dir: Path) -> Path:
    """One CCTV camera over [t, t + seconds] at --fps: clean frames (overscanned) + GT per frame, with motion blur
    over a 1/(2*fps) s shutter that ends on the frame (like a CCTV sensor exposing the frame interval).
    Feed the result to render/cctv: python -m render.cctv.cctv_post --in <out>/frames --gt <out>/gt --overscan ..."""
    bs = RealScene(scn, ctx, args)
    bs.build()
    ids = assign_ids(bs) if args.gt else {}
    bpy, sc = bs.bpy, bs.scene
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.render.motion_blur_position = "END"
    W, H = (int(v) for v in args.res.split("x"))
    k = 1.0 / (1.0 - 2.0 * args.overscan)
    bs.cam_obj.data.sensor_width = 36.0 * k
    sc.render.resolution_x, sc.render.resolution_y = round(W * k), round(H * k)
    bs.point_camera(still_cam(args.cctv_seq, scn))
    seq = out_dir / args.cctv_seq
    frames = seq / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(args.seconds * args.fps)))
    dt = 1.0 / args.fps
    bs.apply(args.t - dt)
    _key_pose(bs, 0)
    for i in range(1, n + 1):
        t = args.t + (i - 1) * dt
        bs.apply(t)
        _key_pose(bs, i)
        sc.frame_set(i)
        bs.apply(t)                      # frame_set re-evaluates the curves; re-apply for the non-keyed state (sparks, lamps)
        dst = frames / f"f_{i - 1:05d}.png"
        sc.render.filepath = str(dst)
        bpy.ops.render.render(write_still=True)
        if args.gt:
            write_gt(bs, dst.stem, seq / "gt", ids)
        if (i - 1) % 25 == 0:
            rb.log(f"[{args.cctv_seq}] frame {i}/{n} t={t:.2f}s")
    (seq / "seq_manifest.json").write_text(json.dumps(dict(cam=args.cctv_seq, t0=args.t, fps=args.fps, frames=n, overscan=args.overscan,
                                                          device=bs.device, args=vars(args)), indent=2, default=str))
    return seq


def main(argv=None) -> int:
    ap = rb.build_parser()
    ap.add_argument("--stills", default="", help="comma list: wide, weld_close, cctv_<i>")
    ap.add_argument("--t", type=float, default=36030.0, help="sim time of the stills (s since 00:00)")
    ap.add_argument("--light-e", type=float, default=150.0, help="power of each LED high-bay area light (W)")
    ap.add_argument("--sky-strength", type=float, default=1.0)
    ap.add_argument("--overscan", type=float, default=0.055, help="extra field per side for cctv_* stills (fraction of the rendered width)")
    ap.add_argument("--cctv-seq", default="", help="render a CCTV sequence from camera cctv_<i> starting at --t")
    ap.add_argument("--seconds", type=float, default=4.0, help="length of --cctv-seq (s)")
    ap.add_argument("--gt", action="store_true", help="also write instance/depth ground truth for cctv_* stills (<out>/gt)")
    ap.add_argument("--haze", type=float, default=0.0, help="world volume scatter density (1/m), e.g. 0.002; 0 = off")
    ap.set_defaults(hdri_id="kloofendal_overcast_puresky", cache=r"D:\icdata\assets\ph_cache")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scn = core.load_scenario(args.scenario)
    ctx = core.build_ctx(scn, core.load_trace(args.trace), args.hero, args.synth_bodies)
    if args.stills:
        run_stills(scn, ctx, args, out_dir)
        return 0
    if args.cctv_seq:
        run_cctv_seq(scn, ctx, args, out_dir)
        return 0
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    rb.run_bpy(scn, ctx, shots, args, out_dir, scene_cls=RealScene)
    return 0


if __name__ == "__main__":
    sys.exit(main())
