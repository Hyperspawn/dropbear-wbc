"""Velocity-task events: closure-consistent reset from a settled standing NPZ.

Upstream H1 resets with ``reset_root_state_uniform`` + ``reset_joints_by_scale`` around the articulation default.
On Dropbear that is wrong twice: (1) the default state has every passive DOF at its CAD value, so the robot starts
airborne and falls onto its feet (dropbear-research AGENTS.md rule 32), and (2) scaling/perturbing passive or
closure-coupled joints tears the loop closures (CONTRACTS section 1: never command or randomize passive joints;
``logs/robot_task/smoke_env16_200steps.json``: 2.4-3.9 cm gaps from closure-motor noise). This reset writes the FULL
settled joint row + root state of a contract NPZ frame, then

* rotates the whole robot rigidly about the vertical axis through the root frame origin (yaw) and shifts it in x/y
  (feet stay exactly on the ground: a rigid yaw/xy move does not change any height);
* adds uniform noise to the SERIAL motors only (hip roll/yaw/pitch, shoulders, wrists; separate leg/arm ranges),
  clipped to the motor hard limits; the closure motors (knee cranks, calf motors, elbows) are never perturbed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform

from dropbear_wbc.robots.dropbear_names import ARM_MOTORS, CLOSURE_MOTORS, MOTOR_NAMES, ROOT_BODY

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_from_standing_npz(ManagerTermBase):
    """Reset event (``mode="reset"``): full settled joint row + root state from a contract NPZ.

    Params (``EventTermCfg.params``):
        asset_cfg: the robot.
        npz_path: contract NPZ (``dropbear-motion-npz-v1``); validated against the live articulation (names, order,
            fps not checked -- any frame is a state), USD SHA, ankle plant variant and validator verdict (fail closed).
        frame_mode: ``"random"`` (uniform over frames) or ``"first"``.
        pose_range: dict with optional ``x``, ``y`` [m] and ``yaw`` [rad] ranges (rigid move, feet stay grounded).
        velocity_range: dict ``x, y, z, roll, pitch, yaw`` added to the NPZ root CoM velocity (world frame).
        leg_serial_noise: (lo, hi) [rad] added to the 6 serial leg motors (hip roll/yaw/pitch).
        arm_serial_noise: (lo, hi) [rad] added to the 8 serial arm motors (shoulders, wrist roll).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from dropbear_wbc.tasks.tracking.motion_npz import (
            expected_usd_sha256,
            load_motion_npz,
            load_validation_verdict,
            validate_against_articulation,
            validate_provenance,
        )

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[asset_cfg.name]
        path = cfg.params["npz_path"]
        arrays = load_motion_npz(path)
        validate_against_articulation(arrays, self.robot.joint_names, self.robot.body_names, list(MOTOR_NAMES))
        spawn = getattr(self.robot.cfg, "spawn", None)
        spherical = tuple(getattr(spawn, "spherical_joint_overrides", ()) or ())
        self.validation = load_validation_verdict(path)
        validate_provenance(arrays, expected_usd_sha256=expected_usd_sha256(), allow_rejected=False,
                            expected_authored_ankle=len(spherical) == 0, validation=self.validation)
        if self.robot.body_names[0] != ROOT_BODY:
            raise ValueError(f"articulation root body is {self.robot.body_names[0]!r}, expected {ROOT_BODY!r}")
        dev = self.robot.device

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=dev)

        self.npz_path = str(path)
        self.meta = arrays.meta
        self.closure_max_m = float(arrays.closure_residual_m.max())
        self.joint_pos = t(arrays.joint_pos)
        self.joint_vel = t(arrays.joint_vel)
        self.root_pos = t(arrays.body_pos_w[:, 0])
        self.root_quat = t(arrays.body_quat_w[:, 0])
        self.root_lin_vel = t(arrays.body_lin_vel_w[:, 0])
        self.root_ang_vel = t(arrays.body_ang_vel_w[:, 0])
        self.num_frames = int(arrays.num_frames)
        motor_ids, names = self.robot.find_joints(list(MOTOR_NAMES), preserve_order=True)
        if list(names) != list(MOTOR_NAMES):
            raise ValueError(f"motors resolved to {names}")
        closure, arms = set(CLOSURE_MOTORS), set(ARM_MOTORS)
        self.leg_serial_ids = torch.tensor([i for i, n in zip(motor_ids, names) if n not in closure and n not in arms],
                                           dtype=torch.long, device=dev)
        self.arm_serial_ids = torch.tensor([i for i, n in zip(motor_ids, names) if n not in closure and n in arms],
                                           dtype=torch.long, device=dev)
        if len(self.leg_serial_ids) != 6 or len(self.arm_serial_ids) != 8:
            raise ValueError(f"expected 6 serial leg + 8 serial arm motors, got {len(self.leg_serial_ids)} + "
                             f"{len(self.arm_serial_ids)}")
        # the action offset (default pose) the NPZ was built for, if recorded (tools/make_static_npz.py)
        self.npz_default_pose_max_abs_diff: float | None = None
        built_for = self.meta.get("default_motor_pos")
        if isinstance(built_for, dict) and all(n in built_for for n in MOTOR_NAMES):
            live = self.robot.data.default_joint_pos[0, motor_ids].cpu()
            ref = torch.tensor([float(built_for[n]) for n in MOTOR_NAMES])
            self.npz_default_pose_max_abs_diff = float((live - ref).abs().max())
            if self.npz_default_pose_max_abs_diff > 0.02:
                print(f"[dropbear reset_from_standing_npz] WARNING: {path} was built for a default motor pose up to "
                      f"{self.npz_default_pose_max_abs_diff:.3f} rad away from the env's; rebuild it "
                      "(tools/make_static_npz.py)?", flush=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        npz_path: str,
        frame_mode: str = "random",
        pose_range: dict[str, tuple[float, float]] | None = None,
        velocity_range: dict[str, tuple[float, float]] | None = None,
        leg_serial_noise: tuple[float, float] = (0.0, 0.0),
        arm_serial_noise: tuple[float, float] = (0.0, 0.0),
    ):
        dev = self.robot.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=dev)
        n = len(env_ids)
        if n == 0:
            return
        if frame_mode == "first":
            f = torch.zeros(n, dtype=torch.long, device=dev)
        elif frame_mode == "random":
            f = torch.randint(0, self.num_frames, (n,), device=dev)
        else:
            raise ValueError(f"frame_mode {frame_mode!r}")
        pose_range = pose_range or {}
        velocity_range = velocity_range or {}

        # -- root: NPZ link pose (+ env origin), rigid yaw about the root frame's vertical axis, x/y shift
        r = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "yaw")], device=dev)
        rnd = sample_uniform(r[:, 0], r[:, 1], (n, 3), device=dev)
        root_pos = self.root_pos[f] + env.scene.env_origins[env_ids]
        root_pos[:, 0] += rnd[:, 0]
        root_pos[:, 1] += rnd[:, 1]
        zero = torch.zeros(n, device=dev)
        root_quat = quat_mul(quat_from_euler_xyz(zero, zero, rnd[:, 2]), self.root_quat[f])
        v = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=dev)
        vrnd = sample_uniform(v[:, 0], v[:, 1], (n, 6), device=dev)
        # NPZ velocities are world frame: rotate them with the yaw so a moving clip stays consistent
        yaw_q = quat_from_euler_xyz(zero, zero, rnd[:, 2])
        root_lin_vel = quat_apply(yaw_q, self.root_lin_vel[f]) + vrnd[:, :3]
        root_ang_vel = quat_apply(yaw_q, self.root_ang_vel[f]) + vrnd[:, 3:]

        # -- joints: full settled row; noise on serial motors only, clipped to the hard limits
        joint_pos = self.joint_pos[f].clone()
        joint_vel = self.joint_vel[f].clone()
        limits = self.robot.data.joint_pos_limits[env_ids]
        for ids, (lo, hi) in ((self.leg_serial_ids, leg_serial_noise), (self.arm_serial_ids, arm_serial_noise)):
            if lo != 0.0 or hi != 0.0:
                q = joint_pos[:, ids] + sample_uniform(lo, hi, (n, len(ids)), dev)
                joint_pos[:, ids] = torch.clamp(q, limits[:, ids, 0], limits[:, ids, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
                                           env_ids=env_ids)
