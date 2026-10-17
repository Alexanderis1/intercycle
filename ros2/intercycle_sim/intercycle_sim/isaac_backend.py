"""Isaac Lab 2.3 / Isaac Sim 5.1 backend, PHYSICS ONLY (headless, no cameras, no RTX sensors).

Robot: ANYmal-D with the pretrained rsl_rl velocity policy (flat if its published checkpoint exists, else rough).
The policy maps the commanded body velocity (from /cmd_vel) to joint position targets at 50 Hz.
DISCLOSURE: like every Isaac Lab velocity policy, it observes the true base linear/angular velocity, standing in for
the robot's onboard state estimator.  That value stays inside the locomotion controller; Nav2 never sees it.

World: the static L0 plant as one triangle-mesh collider (/World/Factory/L0, from world.py) and the movers as
KINEMATIC rigid bodies (/World/Movers/m_XX) that R3's MoverSource drives every step.  The robot physically collides
with both; contacts on non-foot bodies are reported as collision events.
"""
from __future__ import annotations

import math
import os

import numpy as np

_APP = None


def launch_app():
    """Start the Kit app headless (must happen before any isaaclab.* / omni.* import)."""
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=True).app
    return _APP


def _pick_task():
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    tried = []
    for terrain in ("Flat", "Rough"):
        train = f"Isaac-Velocity-{terrain}-Anymal-D-v0"
        try:
            ck = get_published_pretrained_checkpoint("rsl_rl", train)
        except Exception as e:  # noqa: BLE001
            ck = None
            tried.append(f"{train}: {e!r}")
        if ck and os.path.exists(ck):
            return train, train.replace("-v0", "-Play-v0"), ck, tried
        tried.append(f"{train}: no checkpoint ({ck})")
    raise RuntimeError("no published ANYmal-D checkpoint: " + "; ".join(tried))


class IsaacBackend:
    name = "isaaclab-2.3-anymal-d"

    def __init__(self, cfg: dict, world_dir: str, parts: list, device: str = "cuda:0"):
        launch_app()
        import gymnasium as gym
        import torch

        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        self.torch = torch
        self.device = device
        train, play, ckpt, tried = _pick_task()
        print(f"[isaac] task {play} checkpoint {ckpt}", flush=True)
        env_cfg = parse_env_cfg(play, device=device, num_envs=1)
        env_cfg.episode_length_s = 1.0e5                     # no time-out resets
        if getattr(env_cfg.terminations, "base_contact", None) is not None:
            env_cfg.terminations.base_contact = None          # a fall is logged and ends the episode, no teleport
        c = env_cfg.commands.base_velocity
        c.resampling_time_range = (1.0e9, 1.0e9)
        c.heading_command = False
        c.rel_standing_envs = 0.0
        c.rel_heading_envs = 0.0
        c.debug_vis = False
        c.ranges.lin_vel_x = c.ranges.lin_vel_y = c.ranges.ang_vel_z = (0.0, 0.0)
        ev = env_cfg.events
        for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
            if getattr(ev, name, None) is not None:
                setattr(ev, name, None)
        if getattr(ev, "reset_base", None) is not None:
            ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
            ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        if hasattr(env_cfg.scene, "terrain") and "Rough" in train:
            env_cfg.scene.terrain.terrain_type = "plane"      # factory floor is flat
            env_cfg.scene.terrain.terrain_generator = None
            if hasattr(env_cfg, "curriculum"):
                env_cfg.curriculum.terrain_levels = None
        sx, sy, spsi = (float(v) for v in cfg.get("start", [4.0, 32.0, 0.0]))
        self.start = (sx, sy, spsi)
        env_cfg.scene.robot.init_state.pos = (sx, sy, 0.6)
        env_cfg.scene.robot.init_state.rot = (math.cos(spsi / 2), 0.0, 0.0, math.sin(spsi / 2))

        usda = os.path.join(world_dir, "factory_l0.usda")
        self.has_factory = cfg.get("factory", True) and os.path.exists(usda) and os.path.getsize(usda) > 600
        if self.has_factory:
            env_cfg.scene.factory = AssetBaseCfg(prim_path="/World/Factory", spawn=sim_utils.UsdFileCfg(usd_path=usda))

        self.parts = parts
        self.mover_names = []
        for k, p in enumerate(parts):
            rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            cp = sim_utils.CollisionPropertiesCfg()
            mp = sim_utils.MassPropertiesCfg(mass=200.0)
            if p.shape == "cyl":
                sp = sim_utils.CylinderCfg(radius=p.length / 2, height=p.height, axis="Z", rigid_props=rp,
                                           collision_props=cp, mass_props=mp)
            else:
                sp = sim_utils.CuboidCfg(size=(p.length, p.width, p.height), rigid_props=rp, collision_props=cp,
                                         mass_props=mp)
            nm = f"mover_{k:02d}"
            setattr(env_cfg.scene, nm, RigidObjectCfg(
                prim_path=f"/World/Movers/m_{k:02d}", spawn=sp,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-60.0 + 3.0 * k, -60.0, p.height / 2))))
            self.mover_names.append(nm)

        agent_cfg = load_cfg_from_registry(play, "rsl_rl_cfg_entry_point")
        self.env = RslRlVecEnvWrapper(gym.make(play, cfg=env_cfg))
        self.u = self.env.unwrapped
        self.dt = float(self.u.step_dt)
        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(ckpt)
        self.policy = runner.get_inference_policy(device=device)
        self.robot = self.u.scene["robot"]
        self.cmd_term = self.u.command_manager.get_term("base_velocity")
        self.contact = self.u.scene.sensors.get("contact_forces")
        self.foot_ids = []
        if self.contact is not None:
            names = list(self.contact.body_names)
            self.contact_names = names
            self.foot_ids = [i for i, n in enumerate(names) if "FOOT" in n.upper()]
        self.policy_info = {"policy": f"rsl_rl pretrained {train}", "checkpoint": os.path.basename(ckpt),
                            "task": play, "observes_true_velocity": True, "tried": tried,
                            "env_origin": self.u.scene.env_origins[0].tolist(), "factory": bool(self.has_factory),
                            "n_mover_parts": len(parts), "dt": self.dt}
        # place the robot exactly at the start pose (absolute world coordinates)
        root = self.robot.data.default_root_state.clone()
        root[0, 0:3] = torch.tensor((sx, sy, 0.6), device=device)
        root[0, 3:7] = torch.tensor((math.cos(spsi / 2), 0.0, 0.0, math.sin(spsi / 2)), device=device)
        root[0, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.u.scene.write_data_to_sim()
        o = self.env.get_observations()
        self.obs = o[0] if isinstance(o, tuple) else o

    def set_movers(self, part_xyp: np.ndarray) -> None:
        th = self.torch
        for nm, p, (x, y, psi) in zip(self.mover_names, self.parts, part_xyp):
            pose = th.tensor([[x, y, p.height / 2, math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]],
                             device=self.device, dtype=th.float32)
            self.u.scene[nm].write_root_pose_to_sim(pose)

    def step(self, cmd: tuple[float, float, float]) -> dict:
        vx, vy, wz = (max(-1.0, min(1.0, float(v))) for v in cmd)
        vc = self.cmd_term.vel_command_b
        vc[0, 0], vc[0, 1], vc[0, 2] = vx, vy, wz
        with self.torch.inference_mode():
            a = self.policy(self.obs)
            r = self.env.step(a)
        self.obs = r[0]
        return self.state()

    def state(self) -> dict:
        d = self.robot.data
        return {"pos_w": d.root_pos_w[0].tolist(), "quat_wxyz": d.root_quat_w[0].tolist(),
                "v_w": d.root_lin_vel_w[0].tolist(), "v_b": d.root_lin_vel_b[0].tolist(),
                "omega_b": d.root_ang_vel_b[0].tolist()}

    def contacts(self, thresh_n: float = 5.0) -> list[dict]:
        if self.contact is None:
            return []
        f = self.contact.data.net_forces_w[0]
        mag = self.torch.linalg.norm(f, dim=-1).tolist()
        out = []
        for i, m in enumerate(mag):
            if i in self.foot_ids or m < thresh_n:
                continue
            out.append({"body": self.contact_names[i], "force_n": round(m, 1)})
        return out

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            if _APP is not None:
                _APP.close()
