"""Rigged industrial workers (MPFB2 / MakeHuman, CC0) with PPE and procedural walk / idle / work clips.

Build the library (headless, Blender 5.1, MPFB extension installed in the user profile):
    blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/humans.py build
Proof renders (Cycles CPU, 640x360, 16 spp + OIDN) into D:\\icdata\\renders\\asset_tests:
    blender -b --factory-startup --python render/blender/bl_run.py -- render/blender/humans.py proofs

Runtime API (import inside any Blender script; MPFB is NOT needed at runtime):
    names = load_workers()                                  # appends the template collections (not rendered)
    root = spawn("W03", (x, y, 0), yaw_deg=30, clip="walk", phase_s=0.4)
    set_time(root, t_s)                                     # per frame, after setting root location / yaw

Conventions: Blender X = scenario x, Blender Y = scenario z, Z up, metres. `yaw_deg` is the heading of the
worker's facing direction measured counter-clockwise from +X (yaw 0 faces +X, yaw 90 faces +Y). Clips are
in place: `walk` is authored for 1.3 m/s, the caller moves the root along its heading at that speed.
Clips: walk (looping, ~0.9-1.07 s cycle scaled to leg length, see clip_duration()), idle (looping, 8 s), work (looping, 6 s, at a fixture ~0.45 m in front).

Asset licences (verified from the MakeHuman asset-pack pages and the .mhmat headers, all CC0):
MPFB base mesh / targets / rig (CC0 output), "MakeHuman system assets" pack (skins, eyes, eyebrows,
eyelashes, hair, male_casualsuit01/02, shoes03), "Gloves 01" pack (toigo_gloves_short, author MargaretToigo).
PPE (hard hat, hi-vis vest, safety glasses) is procedural (ours). Animation clips are procedural (ours).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

LIB_PATH = r"D:\icdata\assets\library\workers.blend"
MPFB_DATA = r"D:\icdata\assets\mpfb\data"
OUT_DIR = r"D:\icdata\renders\asset_tests"
FPS = 30.0
WALK_SPEED = 1.3                       # m/s
CLIPS = {"walk": 1.05, "idle": 8.0, "work": 6.0}   # loop durations, seconds
CLIP_SAMPLES_PER_S = 30.0

# ---------------------------------------------------------------------------------------------------
# worker roster: phenotype (MakeHuman macro 0..1), skin, hair, outfit tint, PPE colours
# age: 0.5 = 25 y, 0.69 ~ 40 y, 0.81 ~ 55 y, 0.875 ~ 62 y
# ---------------------------------------------------------------------------------------------------
NAVY, GREY, GREEN, ORANGE, BLUE, KHAKI = ((0.05, 0.07, 0.14), (0.20, 0.21, 0.22), (0.07, 0.11, 0.07),
                                           (0.55, 0.16, 0.03), (0.06, 0.13, 0.30), (0.30, 0.25, 0.15))
HV_YELLOW, HV_ORANGE = (0.80, 1.0, 0.02), (1.0, 0.28, 0.0)
HAT_WHITE, HAT_YELLOW, HAT_BLUE, HAT_ORANGE = (0.85, 0.85, 0.82), (0.95, 0.70, 0.02), (0.03, 0.15, 0.55), (0.95, 0.30, 0.02)

ROSTER = [
    dict(name="W01", gender=1.0, age=0.50, muscle=0.60, weight=0.50, height=0.55, race=(0.0, 1.0, 0.0),
         skin="young_caucasian_male", hair="short02", brows="eyebrow001", suit="male_casualsuit01", tint=NAVY,
         vest=HV_YELLOW, hat=HAT_WHITE, gloves=(0.25, 0.25, 0.25)),
    dict(name="W02", gender=1.0, age=0.72, muscle=0.55, weight=0.72, height=0.50, race=(0.9, 0.0, 0.1),
         skin="middleage_african_male", hair="short04", brows="eyebrow002", suit="male_casualsuit01", tint=GREY,
         vest=HV_ORANGE, hat=HAT_YELLOW, gloves=(0.45, 0.32, 0.18)),
    dict(name="W03", gender=0.0, age=0.52, muscle=0.50, weight=0.45, height=0.62, race=(0.0, 0.0, 1.0),
         skin="young_asian_female", hair="bob02", brows="eyebrow010", suit="male_casualsuit01", tint=NAVY,
         vest=HV_YELLOW, hat=HAT_WHITE, gloves=(0.08, 0.08, 0.09)),
    dict(name="W04", gender=1.0, age=0.85, muscle=0.45, weight=0.62, height=0.42, race=(0.0, 1.0, 0.0),
         skin="old_caucasian_male", hair="short01", brows="eyebrow003", suit="male_casualsuit01", tint=GREEN,
         vest=HV_YELLOW, hat=HAT_BLUE, gloves=(0.25, 0.25, 0.25)),
    dict(name="W05", gender=0.0, age=0.70, muscle=0.50, weight=0.62, height=0.40, race=(1.0, 0.0, 0.0),
         skin="middleage_african_female", hair="short03", brows="eyebrow011", suit="male_casualsuit01", tint=BLUE,
         vest=HV_ORANGE, hat=HAT_WHITE, gloves=(0.45, 0.32, 0.18)),
    dict(name="W06", gender=1.0, age=0.55, muscle=0.70, weight=0.45, height=0.62, race=(0.0, 0.0, 1.0),
         skin="young_asian_male", hair="short02", brows="eyebrow004", suit="male_casualsuit01", tint=GREY,
         vest=HV_YELLOW, hat=HAT_YELLOW, gloves=(0.08, 0.08, 0.09)),
    dict(name="W07", gender=1.0, age=0.70, muscle=0.40, weight=0.85, height=0.55, race=(0.0, 1.0, 0.0),
         skin="middleage_caucasian_male", hair="short04", brows="eyebrow005", suit="male_casualsuit01", tint=ORANGE,
         vest=HV_YELLOW, hat=HAT_WHITE, gloves=(0.25, 0.25, 0.25)),
    dict(name="W08", gender=0.0, age=0.55, muscle=0.55, weight=0.50, height=0.50, race=(0.0, 1.0, 0.0),
         skin="young_caucasian_female", hair="ponytail01", brows="eyebrow009", suit="male_casualsuit01", tint=KHAKI,
         vest=HV_YELLOW, hat=HAT_ORANGE, gloves=(0.08, 0.08, 0.09)),
    dict(name="W09", gender=1.0, age=0.52, muscle=0.75, weight=0.55, height=0.52, race=(1.0, 0.0, 0.0),
         skin="young_african_male", hair="short02", brows="eyebrow006", suit="male_casualsuit01", tint=NAVY,
         vest=HV_ORANGE, hat=HAT_WHITE, gloves=(0.45, 0.32, 0.18)),
    dict(name="W10", gender=0.0, age=0.80, muscle=0.45, weight=0.62, height=0.64, race=(0.0, 0.3, 0.7),
         skin="middleage_asian_female", hair="bob01", brows="eyebrow012", suit="male_casualsuit01", tint=GREEN,
         vest=HV_YELLOW, hat=HAT_YELLOW, gloves=(0.25, 0.25, 0.25)),
]


CREDITS = [
    dict(asset="MPFB 2.0.18 base mesh, targets, game_engine rig + weights (generated humans)",
         owner="MakeHuman team / Joel Palmius", licence="CC0 (assets and output; MPFB code is GPL-3.0)",
         url="https://static.makehumancommunity.org/about/license.html"),
    dict(asset="MakeHuman system assets: skins, low-poly eyes, eyebrows, eyelashes, hair, male_casualsuit01, shoes03",
         owner="makehuman_system (Data Collection AB, Joel Palmius, Jonas Hauquier)", licence="CC0",
         url="https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html"),
    dict(asset="Gloves 01 pack: toigo_gloves_short", owner="MargaretToigo", licence="CC0",
         url="https://static.makehumancommunity.org/assets/assetpacks/gloves01.html"),
    dict(asset="hard hat, hi-vis vest, safety glasses (procedural), walk/idle/work clips (procedural)",
         owner="InterCycle (ours)", licence="ours", url=""),
]


# ===================================================================================================
# small math helpers
# ===================================================================================================
def _rx(deg):
    return Quaternion((1, 0, 0), math.radians(deg))


def _ry(deg):
    return Quaternion((0, 1, 0), math.radians(deg))


def _rz(deg):
    return Quaternion((0, 0, 1), math.radians(deg))


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _lerp(a, b, u):
    return a + (b - a) * u


def _cos01(u):
    """0..1 ease (cosine)."""
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))


# ===================================================================================================
# rest-pose rig description + pose solver (armature-space deltas -> bone-local quaternions)
# ===================================================================================================
class Rig:
    """Rest data of a game_engine armature. Armature space: character faces -Y, left side is +X."""

    def __init__(self, arm_obj):
        self.bones = [b.name for b in arm_obj.data.bones]
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in arm_obj.data.bones}
        self.R = {b.name: b.matrix_local.to_3x3().to_quaternion() for b in arm_obj.data.bones}
        self.head = {b.name: b.head_local.copy() for b in arm_obj.data.bones}
        self.tail = {b.name: b.tail_local.copy() for b in arm_obj.data.bones}
        self.dir = {n: (self.tail[n] - self.head[n]).normalized() for n in self.bones}
        self.length = {n: (self.tail[n] - self.head[n]).length for n in self.bones}
        self.scale = self.head["pelvis"].z / 0.923          # body-size factor vs the reference male
        # heel point (behind and below the ankle) for both feet
        self.heel = {}
        for s in "lr":
            a = self.head["foot_" + s]
            self.heel[s] = Vector((a.x, a.y + 0.055 * self.scale, 0.0))


class Pose:
    """Builds armature-space rotation deltas D[b] (rest -> pose) bone by bone, parents first."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.D: dict[str, Quaternion] = {}
        self.H: dict[str, Vector] = {}
        self.pelvis_offset = Vector((0, 0, 0))

    def Dp(self, b):
        p = self.rig.parent[b]
        return self.getD(p) if p else Quaternion()

    def getD(self, b):
        if b not in self.D:
            self.D[b] = self.Dp(b).copy()
        return self.D[b]

    def headpos(self, b):
        if b in self.H:
            return self.H[b]
        p = self.rig.parent[b]
        if p is None:
            h = self.rig.head[b].copy()
        else:
            h = self.headpos(p) + self.getD(p) @ (self.rig.head[b] - self.rig.head[p])
        if b == "pelvis":
            h = h + self.pelvis_offset
        self.H[b] = h
        return h

    def rot(self, b, q):
        """Rotate bone b by world-space q on top of its parent's motion."""
        self.D[b] = q @ self.Dp(b)

    def absolute(self, b, q):
        self.D[b] = q.copy()

    def aim(self, b, direction):
        cur = self.Dp(b) @ self.rig.dir[b]
        self.D[b] = cur.rotation_difference(Vector(direction).normalized()) @ self.Dp(b)

    def local(self, b, q_local):
        R = self.rig.R[b]
        self.D[b] = self.Dp(b) @ R @ q_local @ R.inverted()

    def ik2(self, upper, lower, target, pole):
        """Analytic two-bone IK: aims upper/lower so that lower's tail-end (= child head) reaches target."""
        h = self.headpos(upper)
        L1, L2 = self.rig.length[upper], self.rig.length[lower]
        d = Vector(target) - h
        dist = min(max(d.length, abs(L1 - L2) + 1e-3), (L1 + L2) * 0.9995)
        dn = d.normalized()
        pole = Vector(pole)
        pn = pole - dn * pole.dot(dn)
        pn = pn.normalized() if pn.length > 1e-6 else Vector((0, -1, 0))
        ca = max(-1.0, min(1.0, (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)))
        sa = math.sqrt(max(0.0, 1 - ca * ca))
        knee = h + L1 * (ca * dn + sa * pn)
        self.aim(upper, knee - h)
        self.aim(lower, (h + dn * dist) - knee)

    def to_local(self):
        out = {}
        for b in self.rig.bones:
            D = self.getD(b)
            R = self.rig.R[b]
            out[b] = R.inverted() @ self.Dp(b).inverted() @ D @ R
        loc = self.rig.R["pelvis"].inverted() @ self.pelvis_offset
        return out, loc


# ---------------------------------------------------------------------------------------------------
# shared pose pieces
# ---------------------------------------------------------------------------------------------------
FINGERS = ("index", "middle", "ring", "pinky")


def _fingers(P: Pose, side, curl_deg, thumb_deg):
    for f in FINGERS:
        for k, w in (("01", 0.8), ("02", 1.0), ("03", 0.7)):
            P.local(f"{f}_{k}_{side}", Quaternion((1, 0, 0), math.radians(curl_deg * w)))
    P.local(f"thumb_02_{side}", Quaternion((1, 0, 0), math.radians(thumb_deg)))
    P.local(f"thumb_03_{side}", Quaternion((1, 0, 0), math.radians(thumb_deg * 0.7)))


def _arm_dir(fwd_deg, out_deg, side):
    """Direction of a hanging limb segment, swung forward by fwd_deg and abducted outward by out_deg."""
    s = 1.0 if side == "l" else -1.0
    return _rx(-fwd_deg) @ _ry(-s * out_deg) @ Vector((0, 0, -1))


def _hang_arms(P: Pose, thorax_q, swing_l, swing_r, flex_l, flex_r, out_deg=7.0):
    for side, sw, fl in (("l", swing_l, flex_l), ("r", swing_r, flex_r)):
        P.rot("clavicle_" + side, Quaternion())
        P.aim("upperarm_" + side, thorax_q @ _arm_dir(sw, out_deg, side))
        P.aim("lowerarm_" + side, thorax_q @ _arm_dir(sw + fl, out_deg * 0.4, side))
        P.rot("hand_" + side, Quaternion())


def _plant_legs(P: Pose, ankles, pitches, knee_out=0.12):
    """IK both legs to ankle targets; foot pitch (deg, + = heel up) is absolute in armature space."""
    for s in "lr":
        sx = 1.0 if s == "l" else -1.0
        P.ik2("thigh_" + s, "calf_" + s, ankles[s], (sx * knee_out, -1.0, 0.0))
        P.absolute("foot_" + s, _rx(pitches[s]))
        P.absolute("ball_" + s, _rx(min(0.0, pitches[s]) + 0.35 * max(0.0, pitches[s] - 25.0)))


def _ankle_for(rig: Rig, side, flat_fwd, pitch, x=None):
    """Ankle position when the foot's flat-reference ankle is at forward distance flat_fwd and pitched."""
    A0 = rig.head["foot_" + side]
    x = A0.x if x is None else x
    Aflat = Vector((x, -flat_fwd, A0.z))
    if pitch < 0:   # toes up: pivot on the heel
        H0 = rig.heel[side]
        Hflat = Aflat + (H0 - A0)
        return Hflat + _rx(pitch) @ (A0 - H0)
    B0 = rig.head["ball_" + side]
    Bflat = Aflat + (B0 - A0)
    return Bflat + _rx(pitch) @ (A0 - B0)


def _clamp_pelvis(P: Pose, ankles, z_wanted, margin=0.995):
    """Lower the pelvis offset so each leg can reach its ankle target (no IK pops)."""
    rig = P.rig
    Dpel = P.getD("pelvis")
    z_off = z_wanted
    for s in "lr":
        hip_rel = Dpel @ (rig.head["thigh_" + s] - rig.head["pelvis"])
        reach = (rig.length["thigh_" + s] + rig.length["calf_" + s]) * margin
        hip_xy = rig.head["pelvis"] + Vector((P.pelvis_offset.x, P.pelvis_offset.y, 0)) + hip_rel
        a = ankles[s]
        hor2 = (hip_xy.x - a.x) ** 2 + (hip_xy.y - a.y) ** 2
        if hor2 >= reach * reach:
            continue
        zmax = a.z + math.sqrt(reach * reach - hor2) - (rig.head["pelvis"].z + hip_rel.z)
        z_off = min(z_off, zmax)
    P.pelvis_offset.z = z_off
    P.H.clear()


# ---------------------------------------------------------------------------------------------------
# clips
# ---------------------------------------------------------------------------------------------------
DUTY = 0.62           # stance fraction of the gait cycle


def clip_length(rig: Rig, clip: str) -> float:
    """Loop length of a clip for this body. Walk keeps 1.3 m/s: shorter legs -> shorter stride, higher cadence."""
    if clip == "walk":
        return CLIPS["walk"] * rig.scale ** 0.75
    return CLIPS[clip]


def pose_walk(rig: Rig, t: float) -> Pose:
    T = clip_length(rig, "walk")
    S = WALK_SPEED * T                  # stride length
    sc = rig.scale
    s = (t / T) % 1.0
    c0 = S * DUTY * 0.45                # forward position of the flat foot at heel strike
    P = Pose(rig)

    def foot_state(p, side):
        if p < DUTY:
            flat = c0 - S * p
            if p < 0.09:
                pitch = _lerp(-17.0, 0.0, _smooth(p / 0.09))
            elif p < 0.36:
                pitch = 0.0
            else:
                pitch = 40.0 * _smooth((p - 0.36) / (DUTY - 0.36)) ** 1.3
            return _ankle_for(rig, side, flat, pitch, x=rig.head["foot_" + side].x * 0.62), pitch
        u = (p - DUTY) / (1 - DUTY)
        A_off, _ = foot_state(DUTY - 1e-6, side)
        A_on, _ = foot_state(0.0, side)
        a = _cos01(u)
        pos = A_off.lerp(A_on, a)
        pos.z += 0.065 * sc * math.sin(math.pi * min(1.0, u * 1.15)) ** 1.2 if u < 0.87 else 0.0
        if u < 0.4:
            pitch = _lerp(40.0, 5.0, _smooth(u / 0.4))
        else:
            pitch = _lerp(5.0, -17.0, _smooth((u - 0.4) / 0.6))
        return pos, pitch

    aL, pL = foot_state(s, "l")
    aR, pR = foot_state((s + 0.5) % 1.0, "r")
    ankles, pitches = {"l": aL, "r": aR}, {"l": pL, "r": pR}

    yaw = -5.0 * math.cos(2 * math.pi * s)                 # left hip forward at left heel strike
    roll = -3.5 * math.cos(2 * math.pi * (s - 0.30))        # swing-side hip drops
    P.pelvis_offset = Vector((0.018 * sc * math.cos(2 * math.pi * (s - 0.30)), 0.0, 0.0))
    P.rot("pelvis", _rz(yaw) @ _ry(roll))
    bob = -0.028 * sc + 0.014 * sc * math.cos(2 * math.pi * (2 * s - 0.60))
    _clamp_pelvis(P, ankles, bob)

    P.rot("spine_01", _rz(-yaw * 0.5) @ _ry(-roll * 0.6) @ _rx(2.5))
    P.rot("spine_02", _rz(-yaw * 0.5) @ _ry(-roll * 0.3) @ _rx(1.5))
    P.rot("spine_03", _rz(-yaw * 0.6) @ _rx(1.0))
    P.rot("neck_01", _rz(yaw * 0.2) @ _rx(-2.0))
    P.rot("head", _rz(yaw * 0.25) @ _rx(-1.0))
    thorax = P.getD("spine_03")

    swing_l = 17.0 * math.cos(2 * math.pi * (s - 0.5))
    swing_r = 17.0 * math.cos(2 * math.pi * s)
    _hang_arms(P, thorax, swing_l - 2, swing_r - 2, 14 + max(0.0, swing_l) * 0.9, 14 + max(0.0, swing_r) * 0.9)
    _plant_legs(P, ankles, pitches)
    for side in "lr":
        _fingers(P, side, 22, 12)
    return P


def pose_idle(rig: Rig, t: float) -> Pose:
    T = CLIPS["idle"]
    w = 2 * math.pi * t / T
    sc = rig.scale
    P = Pose(rig)
    shift = math.sin(w) * 0.8 + 0.2 * math.sin(3 * w)       # slow weight shift
    P.pelvis_offset = Vector((0.022 * sc * shift, 0.004 * sc * math.sin(2 * w), 0.0))
    roll = 2.2 * shift
    P.rot("pelvis", _rz(2.0 * math.sin(w + 0.7)) @ _ry(roll))
    xs = {s: rig.head["foot_" + s].x * 0.62 for s in "lr"}
    ankles = {"l": _ankle_for(rig, "l", 0.02 * sc, 0.0, xs["l"]) + Vector((0.0, 0, 0)),
              "r": _ankle_for(rig, "r", -0.03 * sc, 0.0, xs["r"])}
    # the leg away from the weight side carries less load and bends a little more
    _clamp_pelvis(P, ankles, -0.012 * sc - 0.006 * sc * abs(shift))
    breath = math.sin(2 * w * 2)                           # 4 s breathing period
    P.rot("spine_01", _ry(-roll * 0.7) @ _rx(1.0))
    P.rot("spine_02", _rx(0.6 * breath))
    P.rot("spine_03", _rx(0.8 * breath) @ _ry(-roll * 0.3))
    look_yaw = 14.0 * math.sin(w + 0.3) + 6.0 * math.sin(3 * w + 1.1) - 4.0 * math.sin(2 * w)
    look_pitch = 3.0 * math.sin(2 * w + 0.5) + 2.0
    P.rot("neck_01", _rz(look_yaw * 0.4) @ _rx(look_pitch * 0.4))
    P.rot("head", _rz(look_yaw * 0.6) @ _rx(look_pitch * 0.6))
    thorax = P.getD("spine_03")
    sway = 1.5 * math.sin(w * 2 + 0.4)
    _hang_arms(P, thorax, 3 + sway, 3 - sway, 12, 14, out_deg=8)
    _plant_legs(P, ankles, {"l": 0.0, "r": 0.0}, knee_out=0.2)
    for side in "lr":
        _fingers(P, side, 25, 12)
    return P


# work timeline: (t, left hand, right hand, torso yaw, lean). Hand targets in reference-male metres,
# armature space (character faces -Y). The loop's last key equals the first.
_WORK_KEYS = [
    (0.0, (0.13, -0.40, 1.00), (-0.13, -0.40, 1.00), 0.0, 14.0),
    (1.4, (0.12, -0.42, 1.02), (-0.14, -0.38, 1.00), 0.0, 14.0),
    (2.1, (0.12, -0.40, 1.02), (-0.36, -0.46, 1.16), -14.0, 12.0),   # right hand reaches for a tool
    (2.7, (0.12, -0.41, 1.02), (-0.33, -0.44, 1.12), -12.0, 12.0),
    (3.3, (0.13, -0.40, 1.00), (-0.10, -0.46, 1.04), 0.0, 16.0),
    (4.8, (0.13, -0.40, 1.00), (-0.10, -0.46, 1.04), 0.0, 16.0),     # tool use (oscillation added)
    (5.3, (0.34, -0.34, 1.04), (-0.12, -0.40, 1.00), 10.0, 13.0),    # left hand to the side bin
    (6.0, (0.13, -0.40, 1.00), (-0.13, -0.40, 1.00), 0.0, 14.0),
]


def _work_key(t):
    t = t % CLIPS["work"]
    for (t0, l0, r0, y0, n0), (t1, l1, r1, y1, n1) in zip(_WORK_KEYS, _WORK_KEYS[1:]):
        if t0 <= t <= t1:
            u = _cos01((t - t0) / (t1 - t0))
            return (Vector(l0).lerp(Vector(l1), u), Vector(r0).lerp(Vector(r1), u), _lerp(y0, y1, u), _lerp(n0, n1, u))
    k = _WORK_KEYS[0]
    return Vector(k[1]), Vector(k[2]), k[3], k[4]


def pose_work(rig: Rig, t: float) -> Pose:
    T = CLIPS["work"]
    t = t % T
    sc = rig.scale
    P = Pose(rig)
    hl, hr, yaw, lean = _work_key(t)
    w = 2 * math.pi * t / T
    # small continuous handling motion; tool use (3.3..4.8 s) adds a 2 Hz push-pull on the right hand
    hl += Vector((0.012 * math.sin(3 * w), 0.010 * math.cos(2 * w), 0.010 * math.sin(4 * w)))
    hr += Vector((0.010 * math.sin(2 * w + 1), 0.0, 0.008 * math.sin(3 * w)))
    tool = _smooth((t - 3.3) / 0.25) * (1 - _smooth((t - 4.55) / 0.25))
    hr += Vector((0.0, -0.035 * tool * math.sin(2 * math.pi * 2.0 * t), 0.01 * tool * math.cos(2 * math.pi * 2.0 * t)))
    sz = 0.5 + 0.5 * sc                     # work height is set by the fixture, not only by body size
    hl = Vector((hl.x * sc, hl.y * sc, hl.z * sz))
    hr = Vector((hr.x * sc, hr.y * sc, hr.z * sz))
    xs = {s: rig.head["foot_" + s].x * 0.72 for s in "lr"}
    ankles = {"l": _ankle_for(rig, "l", 0.06 * sc, 0.0, xs["l"]), "r": _ankle_for(rig, "r", -0.05 * sc, 0.0, xs["r"])}
    P.pelvis_offset = Vector((0.006 * sc * math.sin(w), 0.035 * sc, 0.0))
    P.rot("pelvis", _rz(yaw * 0.25) @ _rx(lean * 0.25))
    _clamp_pelvis(P, ankles, -0.025 * sc)
    P.rot("spine_01", _rz(yaw * 0.25) @ _rx(lean * 0.30))
    P.rot("spine_02", _rz(yaw * 0.25) @ _rx(lean * 0.25))
    P.rot("spine_03", _rz(yaw * 0.25) @ _rx(lean * 0.20))
    P.rot("neck_01", _rx(7.0))
    # look at the hands: head pitch follows the hands' height, yaw follows the busy hand
    P.rot("head", _rz(yaw * 0.6 + 2.0 * math.sin(w)) @ _rx(12.0 + 2.0 * math.sin(2 * w)))
    for side, tgt in (("l", hl), ("r", hr)):
        sx = 1.0 if side == "l" else -1.0
        P.rot("clavicle_" + side, Quaternion())
        P.ik2("upperarm_" + side, "lowerarm_" + side, tgt, (sx * 0.8, 0.35, -1.0))
        # palm down-ish: twist the hand so the back of the hand faces up/out
        P.rot("hand_" + side, Quaternion())
    _plant_legs(P, ankles, {"l": 0.0, "r": 0.0}, knee_out=0.25)
    grip = 38 + 8 * math.sin(3 * w)
    _fingers(P, "l", grip, 18)
    _fingers(P, "r", 45 if tool > 0.5 else grip, 22)
    return P


POSERS = {"walk": pose_walk, "idle": pose_idle, "work": pose_work}


# ===================================================================================================
# baking clips into actions
# ===================================================================================================
def _bake_clip(arm, rig: Rig, clip: str, action_name: str):
    from bpy_extras import anim_utils  # noqa: F401  (documented helper; channelbag created explicitly below)
    T = clip_length(rig, clip)
    n = int(round(T * CLIP_SAMPLES_PER_S))
    times = [T * k / n for k in range(n + 1)]
    frames = [tt * FPS for tt in times]
    poses = []
    for tt in times[:-1]:
        q, loc = POSERS[clip](rig, tt).to_local()
        poses.append((q, loc))
    poses.append(poses[0])                                 # exact loop
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    slot = act.slots.new(id_type='OBJECT', name=arm.name)
    layer = act.layers.new("Layer")
    strip = layer.strips.new(type='KEYFRAME')
    cb = strip.channelbags.new(slot)
    for b in rig.bones:
        if b == "Root":
            continue
        qs = []
        prev = None
        for qd, _ in poses:
            q = qd[b].normalized()
            if prev is not None and prev.dot(q) < 0:
                q = -q
            qs.append(q)
            prev = q
        path = f'pose.bones["{b}"].rotation_quaternion'
        for i in range(4):
            fc = cb.fcurves.new(path, index=i, group_name=b)
            fc.keyframe_points.add(len(frames))
            co = []
            for f, q in zip(frames, qs):
                co += [f, q[i]]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value] * len(frames))
            fc.modifiers.new('CYCLES')
            fc.update()
    for i in range(3):
        fc = cb.fcurves.new('pose.bones["pelvis"].location', index=i, group_name="pelvis")
        fc.keyframe_points.add(len(frames))
        co = []
        for f, (_, loc) in zip(frames, poses):
            co += [f, loc[i]]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value] * len(frames))
        fc.modifiers.new('CYCLES')
        fc.update()
    act["clip"] = clip
    act["duration_s"] = T
    act["fps"] = FPS
    if clip == "walk":
        act["speed_mps"] = WALK_SPEED
    act.frame_range = (0.0, frames[-1])
    return act


# ===================================================================================================
# materials
# ===================================================================================================
def _sock(node, ident):
    for s in list(node.inputs) + list(node.outputs):
        if s.identifier == ident or s.name == ident:
            return s
    raise KeyError(ident)


def _principled(name, color, rough=0.6, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Coat Weight"].default_value = coat
    return m


def _fabric(name, tex_dir, base, tint, rough=0.85, gain=1.6, normal=True):
    """Principled material: grey-scaled source texture (folds, seams) times a tint colour, plus its normal map."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(tex_dir, base + "_diffuse.png"), check_existing=True)
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    _sock(mul, "Factor_Float").default_value = 1.0
    _sock(mul, "A_Color").default_value = (*[min(1.0, c * gain) for c in tint], 1.0)
    gm = nt.nodes.new("ShaderNodeMath")
    gm.operation = 'MULTIPLY'
    gm.inputs[1].default_value = 1.4
    nt.links.new(bw.outputs["Val"], gm.inputs[0])
    nt.links.new(gm.outputs["Value"], _sock(mul, "B_Color"))
    nt.links.new(_sock(mul, "Result_Color"), bsdf.inputs["Base Color"])
    npath = os.path.join(tex_dir, base + "_normal.png")
    if normal and os.path.exists(npath):
        nimg = nt.nodes.new("ShaderNodeTexImage")
        nimg.image = bpy.data.images.load(npath, check_existing=True)
        nimg.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.8
        nt.links.new(nimg.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _vest_material(name, color, z_lo, z_hi):
    """Fluorescent hi-vis fabric with two horizontal and two shoulder retro-reflective stripes.
    Stripes are placed from the rest-pose coordinates stored in the UV map 'vest_rest' (u = x, v = z)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "vest_rest"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs["Vector"])

    def band(src, center, half):
        sub = nt.nodes.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'
        nt.links.new(src, sub.inputs[0]); sub.inputs[1].default_value = center
        ab = nt.nodes.new("ShaderNodeMath"); ab.operation = 'ABSOLUTE'
        nt.links.new(sub.outputs[0], ab.inputs[0])
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = 'LESS_THAN'
        nt.links.new(ab.outputs[0], lt.inputs[0]); lt.inputs[1].default_value = half
        return lt.outputs[0]

    h = z_hi - z_lo
    b1 = band(sep.outputs["Y"], z_lo + 0.16 * h, 0.022)
    b2 = band(sep.outputs["Y"], z_lo + 0.40 * h, 0.022)
    ax = nt.nodes.new("ShaderNodeMath"); ax.operation = 'ABSOLUTE'
    nt.links.new(sep.outputs["X"], ax.inputs[0])
    b3 = band(ax.outputs[0], 0.085, 0.020)
    above = nt.nodes.new("ShaderNodeMath"); above.operation = 'GREATER_THAN'
    nt.links.new(sep.outputs["Y"], above.inputs[0]); above.inputs[1].default_value = z_lo + 0.40 * h
    b3m = nt.nodes.new("ShaderNodeMath"); b3m.operation = 'MULTIPLY'
    nt.links.new(b3, b3m.inputs[0]); nt.links.new(above.outputs[0], b3m.inputs[1])
    mx1 = nt.nodes.new("ShaderNodeMath"); mx1.operation = 'MAXIMUM'
    nt.links.new(b1, mx1.inputs[0]); nt.links.new(b2, mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath"); mx2.operation = 'MAXIMUM'
    nt.links.new(mx1.outputs[0], mx2.inputs[0]); nt.links.new(b3m.outputs[0], mx2.inputs[1])
    f = mx2.outputs[0]
    col = nt.nodes.new("ShaderNodeMix"); col.data_type = 'RGBA'
    nt.links.new(f, _sock(col, "Factor_Float"))
    _sock(col, "A_Color").default_value = (*color, 1.0)
    _sock(col, "B_Color").default_value = (0.62, 0.63, 0.64, 1.0)
    nt.links.new(_sock(col, "Result_Color"), bsdf.inputs["Base Color"])
    nt.links.new(_sock(col, "Result_Color"), bsdf.inputs["Emission Color"])
    for inp, a, b in (("Metallic", 0.0, 0.75), ("Roughness", 0.75, 0.28), ("Emission Strength", 0.12, 0.0)):
        ma = nt.nodes.new("ShaderNodeMath"); ma.operation = 'MULTIPLY_ADD'
        nt.links.new(f, ma.inputs[0]); ma.inputs[1].default_value = b - a; ma.inputs[2].default_value = a
        nt.links.new(ma.outputs[0], bsdf.inputs[inp])
    return m


# ===================================================================================================
# building one worker with MPFB
# ===================================================================================================
def _enable_mpfb():
    import importlib
    mod = "bl_ext.blender_org.mpfb"
    if mod not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=mod)
    mp = importlib.import_module(mod)
    mp.set_preference("mpfb_second_root", MPFB_DATA)
    from bl_ext.blender_org.mpfb.services.locationservice import LocationService
    LocationService._check_set_second_root()
    from bl_ext.blender_org.mpfb.services.humanservice import HumanService
    return HumanService


def _rest_coords(obj, disable_types=('MASK', 'ARMATURE', 'SUBSURF', 'SOLIDIFY')):
    """Vertex coordinates (object space) with shape keys applied but no topology-changing modifiers."""
    saved = []
    for md in obj.modifiers:
        if md.type in disable_types:
            saved.append((md, md.show_viewport))
            md.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = [obj.matrix_world @ v.co for v in me.vertices]
    ev.to_mesh_clear()
    for md, vis in saved:
        md.show_viewport = vis
    return co


def _head_box(body, eyes):
    """Skull extents from the body vertices weighted to the 'head' bone (rest pose, world == armature)."""
    idx = body.vertex_groups["head"].index
    bidx = body.vertex_groups["body"].index
    co = _rest_coords(body)
    eco = _rest_coords(eyes)
    eye_z = sum(p.z for p in eco) / len(eco)
    eye_front = min(p.y for p in eco)
    eye_x = max(abs(p.x) for p in eco)
    pts = []
    for v in body.data.vertices:
        gw = {g.group: g.weight for g in v.groups}
        if gw.get(idx, 0.0) > 0.6 and bidx in gw:
            pts.append(co[v.index])
    up = [p for p in pts if p.z > eye_z]
    xs = [p.x for p in up]; ys = [p.y for p in up]; zs = [p.z for p in pts]
    return dict(xmax=max(abs(x) for x in xs), ymin=min(ys), ymax=max(ys), ztop=max(zs), eye_z=eye_z,
                eye_front=eye_front, eye_x=eye_x)


def _make_helmet(name, hb, color):
    import bmesh
    a = hb["xmax"] + 0.030
    b = (hb["ymax"] - hb["ymin"]) / 2 + 0.030
    cy = (hb["ymax"] + hb["ymin"]) / 2 + 0.004
    zb = hb["eye_z"] + 0.045
    c = hb["ztop"] - zb + 0.035
    bm = bmesh.new()
    nu, nv = 32, 10
    rings = []
    for j in range(nv + 1):                  # dome from base (j=0) to crown
        th = (math.pi / 2) * j / nv
        ring = []
        for i in range(nu if j < nv else 1):
            ph = 2 * math.pi * i / nu
            r = math.cos(th)
            ring.append(bm.verts.new((a * r * math.cos(ph), cy + b * r * math.sin(ph), zb + c * math.sin(th))))
        rings.append(ring)
    for j in range(nv - 1):
        for i in range(nu):
            bm.faces.new((rings[j][i], rings[j][(i + 1) % nu], rings[j + 1][(i + 1) % nu], rings[j + 1][i]))
    top = rings[nv][0]
    for i in range(nu):
        bm.faces.new((rings[nv - 1][i], rings[nv - 1][(i + 1) % nu], top))
    # brim: wide peak at the front (-Y), narrow at the sides and back, drooping slightly
    outer = []
    for i in range(nu):
        ph = 2 * math.pi * i / nu
        front = max(0.0, -math.sin(ph))       # 1 at the front
        w = 0.012 + 0.045 * front ** 2.2 + 0.010 * max(0.0, math.sin(ph))
        x0, y0 = a * math.cos(ph), b * math.sin(ph)
        nrm = Vector((x0 / a ** 2, y0 / b ** 2, 0)).normalized()
        outer.append(bm.verts.new((x0 + nrm.x * w, cy + y0 + nrm.y * w, zb - 0.006 - 0.012 * front * (w / 0.057))))
    for i in range(nu):
        bm.faces.new((rings[0][(i + 1) % nu], rings[0][i], outer[i], outer[(i + 1) % nu]))
    # crown ridge
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    so = ob.modifiers.new("Solidify", 'SOLIDIFY')
    so.thickness = 0.004
    so.offset = 1.0
    ob.data.materials.append(_principled(name + ".mat", color, rough=0.32, coat=0.4))
    return ob


def _make_glasses(name, hb):
    """Wrap-around safety glasses: a curved polycarbonate band hugging the face in front of the eyes."""
    import bmesh
    zc = hb["eye_z"] + 0.004
    y0 = hb["eye_front"] - 0.016
    ax = hb["eye_x"] + 0.030
    by = 0.055
    bm = bmesh.new()
    nu = 24
    lo, hi = [], []
    for i in range(nu + 1):
        ang = math.radians(-88 + 176 * i / nu)
        x, y = ax * math.sin(ang), y0 + by * (1 - math.cos(ang))
        lo.append(bm.verts.new((x, y, zc - 0.020 + 0.006 * (1 - math.cos(ang)))))
        hi.append(bm.verts.new((x, y, zc + 0.017)))
    for i in range(nu):
        bm.faces.new((lo[i], lo[i + 1], hi[i + 1], hi[i]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    so = ob.modifiers.new("Solidify", 'SOLIDIFY')
    so.thickness = 0.0015
    m = _principled(name + ".mat", (0.35, 0.37, 0.38), rough=0.06)
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Transmission Weight"].default_value = 0.75
    bs.inputs["IOR"].default_value = 1.58
    ob.data.materials.append(m)
    return ob


def _make_vest(name, suit, arm, color):
    """Hi-vis vest: torso faces of the shirt, pushed out along the normals, skinned like the shirt."""
    import bmesh
    rig_b = arm.data.bones
    sc = rig_b["pelvis"].head_local.z / 0.923
    z_lo = rig_b["pelvis"].head_local.z + 0.04 * sc
    z_hi = rig_b["neck_01"].head_local.z - 0.015 * sc
    x_max = abs(rig_b["upperarm_l"].head_local.x) * 0.93
    arm_groups = {suit.vertex_groups[g].index for g in suit.vertex_groups.keys()
                  if g.startswith(("upperarm", "lowerarm", "hand"))}
    co = _rest_coords(suit)
    keep_v = set()
    for v in suit.data.vertices:
        p = co[v.index]
        wa = sum(g.weight for g in v.groups if g.group in arm_groups)
        front = p.y < rig_b["spine_03"].head_local.y
        vneck = front and p.z > z_hi - 0.13 * sc + 1.1 * abs(p.x)
        if z_lo <= p.z <= z_hi and abs(p.x) <= x_max and wa < 0.25 and not vneck:
            keep_v.add(v.index)
    vest = suit.copy()
    vest.data = suit.data.copy()
    vest.name = vest.data.name = name
    me = vest.data
    if me.shape_keys:
        vest.shape_key_clear()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    for i, v in enumerate(bm.verts):
        v.co = suit.matrix_world.inverted() @ co[i]
    drop = [f for f in bm.faces if not all(v.index in keep_v for v in f.verts)]
    bmesh.ops.delete(bm, geom=drop, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # face-wise selection leaves a staircase border: relax the border vertices along the edge loop
    for _ in range(8):
        border = [v for v in bm.verts if v.is_boundary]
        new = {}
        for v in border:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * 0.4 + (nb[0].co + nb[1].co) * 0.3
        for v, c in new.items():
            v.co = c
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * 0.010 * sc
    uvl = bm.loops.layers.uv.new("vest_rest")
    for f in bm.faces:
        for lp in f.loops:
            lp[uvl].uv = (lp.vert.co.x, lp.vert.co.z)
    bm.to_mesh(me)
    bm.free()
    me.materials.clear()
    me.materials.append(_vest_material(name + ".mat", color, z_lo, z_hi))
    for md in list(vest.modifiers):
        if md.type != 'ARMATURE':
            vest.modifiers.remove(md)
    return vest


def _bone_parent(ob, arm, bone):
    ob.parent = arm
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    b = arm.data.bones[bone]
    # bone parenting attaches at the bone tail: keep the world transform authored in rest pose
    ob.matrix_parent_inverse = (arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))).inverted()


def _decimate(ob, ratio):
    md = ob.modifiers.new("Decimate", 'DECIMATE')
    md.ratio = ratio
    md.use_collapse_triangulate = False
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_move_to_index(modifier=md.name, index=0)
        bpy.ops.object.modifier_apply(modifier=md.name)


def build_worker(HS, spec, coll):
    hi = HS._create_default_human_info_dict()
    hi["name"] = spec["name"]
    af, ca, asn = spec["race"]
    hi["phenotype"].update(gender=spec["gender"], age=spec["age"], muscle=spec["muscle"], weight=spec["weight"],
                           height=spec["height"], cupsize=0.5, firmness=0.5,
                           race={"african": af, "caucasian": ca, "asian": asn})
    hi["rig"] = "game_engine"
    hi["eyes"] = "low-poly/low-poly.mhclo"
    hi["eyebrows"] = f"{spec['brows']}/{spec['brows']}.mhclo"
    hi["eyelashes"] = "eyelashes02/eyelashes02.mhclo"
    hi["hair"] = f"{spec['hair']}/{spec['hair']}.mhclo"
    hi["clothes"] = [f"{spec['suit']}/{spec['suit']}.mhclo", "shoes03/shoes03.mhclo",
                     "toigo_gloves_short/toigo_gloves_short.mhclo"]
    hi["skin_mhmat"] = f"{spec['skin']}/{spec['skin']}.mhmat"
    hi["skin_material_type"] = "ENHANCED_SSS"
    st = HS.get_default_deserialization_settings()
    st["subdiv_levels"] = 0
    body = HS.deserialize_from_dict(hi, st)
    arm = body.parent
    arm.name = arm.data.name = spec["name"] + ".rig"
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    objs = [arm] + [o for o in bpy.data.objects if o.parent == arm]
    by = {}
    for o in objs:
        for key in (spec["suit"], "shoes03", "gloves", "low-poly", spec["hair"]):
            if key in o.name:
                by[key] = o
    clothes_dir = os.path.join(MPFB_DATA, "clothes")
    suit = by[spec["suit"]]
    suit.data.materials.clear()
    suit.data.materials.append(_fabric(spec["name"] + ".suit", os.path.join(clothes_dir, spec["suit"]), spec["suit"],
                                       spec["tint"], rough=0.9))
    shoes = by["shoes03"]
    shoes.data.materials.clear()
    shoes.data.materials.append(_fabric(spec["name"] + ".boots", os.path.join(clothes_dir, "shoes03"), "shoes03",
                                        (0.10, 0.09, 0.08), rough=0.55, gain=1.0, normal=False))
    gl = by["gloves"]
    gl.data.materials.clear()
    gl.data.materials.append(_principled(spec["name"] + ".gloves", spec["gloves"], rough=0.8))
    # PPE
    hb = _head_box(body, by["low-poly"])
    helmet = _make_helmet(spec["name"] + ".hardhat", hb, spec["hat"])
    glasses = _make_glasses(spec["name"] + ".glasses", hb)
    vest = _make_vest(spec["name"] + ".vest", suit, arm, spec["vest"])
    for o in (helmet, glasses, vest):
        bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.update()
    _bone_parent(helmet, arm, "head")
    _bone_parent(glasses, arm, "head")
    vest.parent = arm
    for md in vest.modifiers:
        if md.type == 'ARMATURE':
            md.object = arm
    # keep each worker light (~30k tris): collapse-decimate the dense clothes (weights are interpolated)
    for o, ratio in ((suit, 0.5), (vest, 0.5), (gl, 0.4), (shoes, 0.5)):
        _decimate(o, ratio)
    objs = [arm] + [o for o in bpy.data.objects if o.parent == arm]
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    # clips
    rig = Rig(arm)
    clips = {}
    for clip in CLIPS:
        act = _bake_clip(arm, rig, clip, f"{spec['name']}_{clip}")
        clips[clip] = act.name
    arm["clips"] = json.dumps(clips)
    arm["worker"] = spec["name"]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    m = ev.to_mesh()
    arm["height_m"] = round(max((ev.matrix_world @ v.co).z for v in m.vertices), 3)      # rest pose, no hat
    ev.to_mesh_clear()
    # triangles
    dg = bpy.context.evaluated_depsgraph_get()
    tris = 0
    for o in objs:
        if o.type == 'MESH':
            ev = o.evaluated_get(dg)
            m = ev.to_mesh(); m.calc_loop_triangles(); tris += len(m.loop_triangles); ev.to_mesh_clear()
    arm["tris"] = tris
    print(f"[humans] built {spec['name']}: {len(objs)} objects, {tris} tris, height {arm['height_m']:.2f} m", flush=True)
    return arm


def build(out=LIB_PATH, only=None, pack=False):
    HS = _enable_mpfb()
    for o in list(bpy.data.objects):            # factory-startup cube / camera / light
        bpy.data.objects.remove(o)
    scene = bpy.context.scene
    for spec in ROSTER:
        if only and spec["name"] not in only:
            continue
        coll = bpy.data.collections.new(spec["name"])
        scene.collection.children.link(coll)
        build_worker(HS, spec, coll)
    scene["credits"] = json.dumps(CREDITS)
    for c in scene.collection.children:
        c["credits"] = json.dumps(CREDITS)
    # templates should not render where they are: keep them excluded in the library file
    for lc in bpy.context.view_layer.layer_collection.children:
        lc.exclude = True
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if pack:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print(f"[humans] saved {out}", flush=True)


# ===================================================================================================
# runtime API
# ===================================================================================================
_ACTION_CACHE: dict[str, list] = {}
_INSTANCE_COUNT = [0]


def load_workers(path=LIB_PATH):
    """Append the worker template collections from the library (not linked to the scene). Returns names."""
    _ACTION_CACHE.clear()
    have = [c.name for c in bpy.data.collections if c.get("is_worker_template")]
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [n for n in src.collections if n.startswith("W") and n not in have]
        new = set(dst.collections)
        dst.actions = [a for a in src.actions if a.split("_")[0] in new and "_" in a]
    for c in dst.collections:
        if c is not None:
            c["is_worker_template"] = True
    return sorted(c.name for c in bpy.data.collections if c.get("is_worker_template"))


def _template(name_or_idx):
    names = sorted(c.name for c in bpy.data.collections if c.get("is_worker_template"))
    if not names:
        raise RuntimeError("call load_workers() first")
    name = names[name_or_idx % len(names)] if isinstance(name_or_idx, int) else name_or_idx
    return bpy.data.collections[name]


def _armature_of(root):
    if root.type == 'ARMATURE':
        return root
    for c in root.children:
        if c.type == 'ARMATURE':
            return c
    raise ValueError(f"{root.name} has no worker armature")


def spawn(name_or_idx, loc, yaw_deg, clip="idle", phase_s=0.0, collection=None):
    """Instance a worker (objects copied, mesh/armature data shared). Returns the root empty.
    The root's +X is the worker's facing direction; set root.location / root.rotation_euler.z per frame."""
    src = _template(name_or_idx)
    _INSTANCE_COUNT[0] += 1
    n = _INSTANCE_COUNT[0]
    parent_coll = collection or bpy.context.scene.collection
    coll = bpy.data.collections.new(f"{src.name}.{n:03d}")
    parent_coll.children.link(coll)
    root = bpy.data.objects.new(f"{src.name}.{n:03d}.root", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.4
    coll.objects.link(root)
    arm_src = next(o for o in src.objects if o.type == 'ARMATURE')
    arm = arm_src.copy()
    arm.name = f"{src.name}.{n:03d}.rig"
    if arm.animation_data:
        arm.animation_data.action = None
    coll.objects.link(arm)
    arm.parent = root
    arm.matrix_parent_inverse = Matrix.Identity(4)
    arm.location = (0, 0, 0)
    arm.rotation_mode = 'XYZ'
    arm.rotation_euler = (0, 0, math.pi / 2)        # rig faces -Y; rotate so it faces the root's +X
    for o in src.objects:
        if o is arm_src:
            continue
        c = o.copy()
        coll.objects.link(c)
        ptype, pbone, pinv = o.parent_type, o.parent_bone, o.matrix_parent_inverse.copy()
        c.parent = arm
        c.parent_type = ptype
        if ptype == 'BONE':
            c.parent_bone = pbone
        c.matrix_parent_inverse = pinv
        for md in c.modifiers:
            if md.type == 'ARMATURE':
                md.object = arm
    root.location = loc
    root.rotation_euler = (0, 0, math.radians(yaw_deg))
    root["worker"] = src.name
    root["clip"] = clip
    root["phase_s"] = float(phase_s)
    set_time(root, 0.0, clip)
    return root


def clip_duration(root, clip):
    """Loop length in seconds of `clip` for this worker instance (walk cadence depends on body size)."""
    arm = _armature_of(root)
    return float(bpy.data.actions[json.loads(arm["clips"])[clip]]["duration_s"])


def _channels(action):
    key = action.name
    if key not in _ACTION_CACHE:
        chans = []
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in cb.fcurves:
                        bone = fc.data_path.split('"')[1]
                        prop = fc.data_path.rsplit(".", 1)[1]
                        chans.append((bone, prop, fc.array_index, fc))
        _ACTION_CACHE[key] = chans
    return _ACTION_CACHE[key]


def set_time(root, t_s, clip=None):
    """Pose the instance at clip time t_s (+ its phase). Loops the clip. Pure pose write, no scene frame needed."""
    arm = _armature_of(root)
    if clip is not None:
        root["clip"] = clip
    clip = root.get("clip", "idle")
    clips = json.loads(arm["clips"])
    act = bpy.data.actions[clips[clip]]
    T = float(act["duration_s"])
    tt = (float(t_s) + float(root.get("phase_s", 0.0))) % T
    frame = tt * float(act.get("fps", FPS))
    pbs = arm.pose.bones
    rot = {}
    for bone, prop, idx, fc in _channels(act):
        v = fc.evaluate(frame)
        if prop == "rotation_quaternion":
            rot.setdefault(bone, [1.0, 0.0, 0.0, 0.0])[idx] = v
        elif prop == "location":
            pbs[bone].location[idx] = v
    for bone, q in rot.items():
        pbs[bone].rotation_quaternion = Quaternion(q).normalized()
    return arm


# ===================================================================================================
# proof renders
# ===================================================================================================
def _proof_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 16
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.render.resolution_x, sc.render.resolution_y = 640, 360
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.37, 0.40, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    sc.world = w
    bpy.ops.mesh.primitive_plane_add(size=60)
    floor = bpy.context.object
    fm = _principled("floor", (0.32, 0.32, 0.31), rough=0.7)
    floor.data.materials.append(fm)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-35))
    sc.collection.objects.link(sun)
    area = bpy.data.objects.new("key", bpy.data.lights.new("key", 'AREA'))
    area.data.energy = 900
    area.data.size = 4
    area.location = (0, -4, 7)
    area.rotation_euler = (math.radians(30), 0, 0)
    sc.collection.objects.link(area)
    return sc


def _camera(sc, loc, target, lens=35):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = lens
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def _render(sc, path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"[humans] wrote {path}", flush=True)


def proofs(lib=LIB_PATH, out=OUT_DIR, which=("lineup", "walk", "work", "cctv")):
    os.makedirs(out, exist_ok=True)
    if "lineup" in which:
        sc = _proof_scene()
        names = load_workers(lib)
        for i in range(6):
            r = spawn(names[i % len(names)], ((i - 2.5) * 0.85, 0, 0), yaw_deg=-90, clip="idle", phase_s=i * 1.3)
            set_time(r, 0.0)
        _camera(sc, (0, -6.2, 1.35), (0, 0, 0.95), lens=40)
        _render(sc, os.path.join(out, "humans_lineup.png"))
    if "walk" in which:
        from PIL import Image
        tiles = []
        for k in range(4):
            sc = _proof_scene()
            sc.render.resolution_x = 160
            names = load_workers(lib)
            r = spawn(names[0], (0, 0, 0), yaw_deg=0, clip="walk")
            set_time(r, k * clip_duration(r, "walk") / 4)
            _camera(sc, (0.1, -3.4, 1.0), (0.1, 0, 0.9), lens=32)
            p = os.path.join(out, f"_walk_{k}.png")
            _render(sc, p)
            tiles.append(Image.open(p).convert("RGB"))
        strip = Image.new("RGB", (640, 360))
        for k, im in enumerate(tiles):
            strip.paste(im, (k * 160, 0))
        strip.save(os.path.join(out, "humans_walk.png"))
        for k in range(4):
            os.remove(os.path.join(out, f"_walk_{k}.png"))
    if "work" in which:
        sc = _proof_scene()
        names = load_workers(lib)
        r = spawn(names[1 % len(names)], (0, 0, 0), yaw_deg=90, clip="work")
        set_time(r, 3.6)
        r2 = spawn(names[2 % len(names)], (1.4, 0.0, 0), yaw_deg=90, clip="work", phase_s=2.0)
        set_time(r2, 0.0)
        for x in (0.0, 1.4):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0.62, 0.45))
            tb = bpy.context.object
            tb.scale = (0.9, 0.45, 0.9)
            tb.data.materials.append(_principled("fixture", (0.15, 0.25, 0.45), rough=0.5, metal=0.3))
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0.50, 0.95))
            pt = bpy.context.object
            pt.scale = (0.35, 0.25, 0.08)
            pt.data.materials.append(_principled("part", (0.6, 0.6, 0.62), rough=0.35, metal=1.0))
        _camera(sc, (-2.2, -2.6, 1.9), (0.6, 0.3, 0.95), lens=35)
        _render(sc, os.path.join(out, "humans_work.png"))
    if "cctv" in which:
        sc = _proof_scene()
        names = load_workers(lib)
        k = 0
        for i, (x, y, yaw, clip) in enumerate([(-2, 1, 0, "walk"), (1, -1.5, 160, "walk"), (3, 2, 90, "work"),
                                               (-3.5, -2, 30, "idle"), (0, 3, -90, "walk"), (4.5, -2.5, 200, "idle"),
                                               (-1, -3.5, 45, "idle"), (2.0, 4.5, 270, "work")]):
            r = spawn(names[i % len(names)], (x, y, 0), yaw_deg=yaw, clip=clip, phase_s=0.37 * i)
            set_time(r, 0.0)
            if clip == "work":
                bpy.ops.mesh.primitive_cube_add(size=1)
                tb = bpy.context.object
                f = Vector((math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0))
                tb.location = Vector((x, y, 0.45)) + f * 0.62
                tb.rotation_euler.z = math.radians(yaw)
                tb.scale = (0.45, 0.9, 0.9)
                tb.data.materials.append(_principled("fixture", (0.15, 0.25, 0.45), rough=0.5, metal=0.3))
        # 9 m high, looking down 45 degrees at the group
        _camera(sc, (0, -9.0, 9.0), (0, 0.5, 0.0), lens=24)
        _render(sc, os.path.join(out, "humans_cctv.png"))


# ===================================================================================================
if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "build"
    if cmd == "build":
        only = [a for a in args[1:] if a.startswith("W")] or None
        build(only=only, pack="--pack" in args)
    elif cmd == "proofs":
        which = [a for a in args[1:] if not a.startswith("-")] or ["lineup", "walk", "work", "cctv"]
        proofs(which=which)
    else:
        sys.exit(f"unknown command {cmd!r}; use build | proofs")
