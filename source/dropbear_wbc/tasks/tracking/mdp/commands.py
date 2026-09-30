"""BeyondMimic motion command, ported to Dropbear (closed-loop mechanisms, motor-only command).

Upstream: ``whole_body_tracking/tasks/tracking/mdp/commands.py`` (MIT, HybridRobotics). Dropbear patches:

* The motion file is a contract ``dropbear-motion-npz-v1`` NPZ; its ``joint_names``/``body_names`` must equal
  the live articulation's (order included) and ``fps`` must equal the policy rate, else we fail closed.
* The command (policy input) is the reference MOTOR position + velocity (22 + 22), not all 91 joints.
* Reference state initialisation (RSI) writes the FULL settled joint row (all joints, incl. passive
  four-bar/tie-rod/U-joint DOFs and the neck) and the root (``world``) state from the NPZ. Additive noise
  goes on motors only (``joint_position_range``); motors listed in ``closure_joint_names`` (they drive a
  loop closure) use ``closure_joint_position_range`` instead (default 0), because perturbing them without
  re-solving the passive joints tears the closure. Motors are clipped to their hard limits; passive joints
  are never clipped (upstream clipped *every* joint to soft limits, which would break the loop closures).
* The root state comes from the NPZ root body explicitly (upstream used ``body_names[0]``, i.e. G1's pelvis).
* The anchor is a chest-level body rigidly fixed to the root. Its *raw link frame* is used everywhere
  (as upstream), so a deploy runner can rebuild the observations from the root IMU pose and the rigid
  anchor-in-root offset. Only ``bad_anchor_ori`` applies a world-aligning offset (``terminations.py``).
* ``start_at_zero`` (play): every (re)sample starts the motion at frame 0 instead of adaptive sampling.

Frames: world frame (per-env origins added to reference positions), quaternions wxyz, SI units.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from ..motion_npz import (
    MotionArrays,
    expected_usd_sha256,
    load_motion_npz,
    load_validation_verdict,
    validate_against_articulation,
    validate_provenance,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Contract NPZ on the simulation device, validated against the articulation.

    Attributes:
        fps: Motion frame rate [Hz] (== policy rate).
        full_joint_pos, full_joint_vel: (T, J) all articulation joints (RSI rows).
        joint_pos, joint_vel: (T, 22) motor columns (command / metrics).
        root_*: (T, ...) root body (``world``) state, world frame, env origin at 0.
        time_step_total: T.
    """

    def __init__(
        self,
        motion_file: str,
        robot: Articulation,
        motor_ids: Sequence[int],
        motor_names: Sequence[str],
        tracked_body_ids: Sequence[int],
        root_body_index: int,
        expected_fps: float,
        device: str | torch.device,
        allow_rejected: bool = False,
    ):
        arrays: MotionArrays = load_motion_npz(motion_file)
        validate_against_articulation(
            arrays, robot.joint_names, robot.body_names, motor_names, expected_fps=expected_fps, fps_tol=1e-3
        )
        spawn = getattr(getattr(robot, "cfg", None), "spawn", None)
        spherical = tuple(getattr(spawn, "spherical_joint_overrides", ()) or ())
        # sibling <clip>.validation.json (tools/validate_motion_npz.py); a "rejected" verdict fails closed too
        self.validation = load_validation_verdict(motion_file)
        validate_provenance(arrays, expected_usd_sha256=expected_usd_sha256(), allow_rejected=allow_rejected,
                            expected_authored_ankle=len(spherical) == 0, validation=self.validation)
        self.allow_rejected = bool(allow_rejected)
        self.arrays_meta = arrays.meta
        self.arrays_closure_max = float(arrays.closure_residual_m.max())
        self.fps = arrays.fps

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        self.full_joint_pos = t(arrays.joint_pos)
        self.full_joint_vel = t(arrays.joint_vel)
        motor_idx = torch.as_tensor(list(motor_ids), dtype=torch.long, device=device)
        self.joint_pos = self.full_joint_pos[:, motor_idx].contiguous()
        self.joint_vel = self.full_joint_vel[:, motor_idx].contiguous()
        self._body_pos_w = t(arrays.body_pos_w)
        self._body_quat_w = t(arrays.body_quat_w)
        self._body_lin_vel_w = t(arrays.body_lin_vel_w)
        self._body_ang_vel_w = t(arrays.body_ang_vel_w)
        self._body_indexes = torch.as_tensor(list(tracked_body_ids), dtype=torch.long, device=device)
        self.root_pos_w = self._body_pos_w[:, root_body_index].contiguous()
        self.root_quat_w = self._body_quat_w[:, root_body_index].contiguous()
        self.root_lin_vel_w = self._body_lin_vel_w[:, root_body_index].contiguous()
        self.root_ang_vel_w = self._body_ang_vel_w[:, root_body_index].contiguous()
        self.time_step_total = int(self.joint_pos.shape[0])

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    def at(self, name: str, t: torch.Tensor, body: int | None = None) -> torch.Tensor:
        """``<name>[t]`` (tracked bodies) or ``<name>[t, body]`` without gathering every frame first: the properties
        above copy the whole (frames, bodies) array per call, which the live reference's 15000-frame buffer made cost
        ~0.7 ms each (~5.5 ms per policy step on the CPU pipeline, 2026-09-26). Same values."""
        arr = getattr(self, "_" + name)
        if body is None:
            return arr[t][:, self._body_indexes]
        return arr[t, self._body_indexes[body]]


class MotionCommand(CommandTerm):
    """Reference-motion command (BeyondMimic) for Dropbear."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        if self.robot.body_names[0] != cfg.root_body_name:
            raise ValueError(f"articulation root body is {self.robot.body_names[0]!r}, expected {cfg.root_body_name!r}")
        if cfg.anchor_body_name not in cfg.body_names:
            raise ValueError("anchor_body_name must be one of body_names")
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = list(cfg.body_names).index(cfg.anchor_body_name)
        body_ids, body_names = self.robot.find_bodies(list(cfg.body_names), preserve_order=True)
        if list(body_names) != list(cfg.body_names):
            raise ValueError(f"tracked bodies resolved to {body_names}, expected {cfg.body_names}")
        self.body_indexes = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        motor_ids, motor_names = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        if list(motor_names) != list(cfg.joint_names):
            raise ValueError(f"motors resolved to {motor_names}, expected {cfg.joint_names}")
        self.motor_ids = torch.tensor(motor_ids, dtype=torch.long, device=self.device)
        closure = set(cfg.closure_joint_names)
        unknown = closure - set(cfg.joint_names)
        if unknown:
            raise ValueError(f"closure_joint_names not among joint_names: {sorted(unknown)}")
        self._closure_cols = torch.tensor([i for i, n in enumerate(cfg.joint_names) if n in closure], dtype=torch.long, device=self.device)
        self._serial_cols = torch.tensor([i for i, n in enumerate(cfg.joint_names) if n not in closure], dtype=torch.long, device=self.device)

        policy_rate = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        self.motion = MotionLoader(
            cfg.motion_file,
            self.robot,
            motor_ids,
            cfg.joint_names,
            body_ids,
            root_body_index=0,
            expected_fps=policy_rate,
            device=self.device,
            allow_rejected=cfg.allow_rejected_motion,
        )
        # A clip built for a specific default pose (tools/make_static_npz.py records it in meta.default_motor_pos)
        # should match the action offset in use; a mismatch is legal but usually means a stale clip.
        self.npz_default_pose_max_abs_diff: float | None = None
        npz_default = self.motion.arrays_meta.get("default_motor_pos")
        if isinstance(npz_default, dict) and all(n in npz_default for n in cfg.joint_names):
            live = self.robot.data.default_joint_pos[0, self.motor_ids].cpu()
            ref = torch.tensor([float(npz_default[n]) for n in cfg.joint_names])
            self.npz_default_pose_max_abs_diff = float((live - ref).abs().max())
            if self.npz_default_pose_max_abs_diff > 0.02:
                print(f"[dropbear MotionCommand] WARNING: motion NPZ was built for a default motor pose that differs "
                      f"from the env's by up to {self.npz_default_pose_max_abs_diff:.3f} rad "
                      f"(npz: {self.motion.arrays_meta.get('default_pose_source')}); rebuild the clip?", flush=True)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        n_bodies = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, n_bodies, 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // policy_rate) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for key in (
            "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
            "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
            "error_joint_pos", "error_joint_vel", "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ command & reference (motion)
    @property
    def command(self) -> torch.Tensor:
        """Reference motor positions [rad] and velocities [rad/s], shape (num_envs, 44)."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.at("body_pos_w", self.time_steps) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.at("body_quat_w", self.time_steps)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.at("body_lin_vel_w", self.time_steps)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.at("body_ang_vel_w", self.time_steps)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.at("body_pos_w", self.time_steps, self.motion_anchor_body_index) + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.at("body_quat_w", self.time_steps, self.motion_anchor_body_index)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.at("body_lin_vel_w", self.time_steps, self.motion_anchor_body_index)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.at("body_ang_vel_w", self.time_steps, self.motion_anchor_body_index)

    # ------------------------------------------------------------------ robot state
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.motor_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.motor_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # ------------------------------------------------------------------ manager hooks
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        probs = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        probs = torch.nn.functional.pad(
            probs.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
        probs = probs / probs.sum()
        sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        entropy = -(probs * (probs + 1e-12).log()).sum()
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = entropy / math.log(self.bin_count) if self.bin_count > 1 else 0.0
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        if self.cfg.start_at_zero:
            self.time_steps[env_ids] = 0
        else:
            self._adaptive_sampling(env_ids)
        t = self.time_steps[env_ids]
        n = len(env_ids)

        # -- root (the NPZ 'world' body): link pose + CoM velocity, plus BeyondMimic pose/velocity noise
        root_pos = self.motion.root_pos_w[t] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.root_quat_w[t].clone()
        root_lin_vel = self.motion.root_lin_vel_w[t].clone()
        root_ang_vel = self.motion.root_ang_vel_w[t].clone()
        ranges = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=self.device
        )
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=self.device)
        root_pos = root_pos + rand[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]), root_ori)
        ranges = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=self.device)
        root_lin_vel = root_lin_vel + rand[:, :3]
        root_ang_vel = root_ang_vel + rand[:, 3:]

        # -- joints: FULL settled row; noise + hard-limit clip on the motors only
        joint_pos = self.motion.full_joint_pos[t].clone()
        joint_vel = self.motion.full_joint_vel[t].clone()
        motor_pos = joint_pos[:, self.motor_ids]
        for cols, (lo, hi) in (
            (self._serial_cols, self.cfg.joint_position_range),
            (self._closure_cols, self.cfg.closure_joint_position_range),
        ):
            if len(cols) and (lo != 0.0 or hi != 0.0):
                motor_pos[:, cols] += sample_uniform(lo, hi, (n, len(cols)), self.device)
        limits = self.robot.data.joint_pos_limits[env_ids][:, self.motor_ids]
        joint_pos[:, self.motor_ids] = torch.clamp(motor_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
        # The robot was just written onto the reference frame, so until the next _update_command the relative
        # targets are the reference bodies themselves. Without this, the very first env.step() after env.reset()
        # (Isaac Lab evaluates terminations/rewards before the first command update) compares against the zero
        # initialised buffers and every env ends with a spurious 1-step ee_body_pos termination
        # (logs/gpu_pipeline/diag_reset_*.json, play_2026-09-24_11-47-34.json). Mid-episode resets are unaffected:
        # _update_command overwrites these rows before the next termination check.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if self.cfg.continuous_loop and self.cfg.start_at_zero:
            # evaluation: wrap the reference clock only; the robot keeps its state (no teleport to frame 0), so
            # the error/drift of the previous loop carries over. A clip whose last frame != frame 0 then shows a
            # reference jump at the wrap (reported by scripts/play.py).
            self.time_steps[env_ids] = 0
        else:
            self._resample_command(env_ids)

        n = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, n, 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, n, 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, n, 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, n, 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    # ------------------------------------------------------------------ debug visualisation
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
        elif hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer.set_visibility(False)
            self.goal_anchor_visualizer.set_visibility(False)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(False)
                self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionCommand`."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    motion_file: str = MISSING
    """Contract NPZ (``dropbear-motion-npz-v1``)."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    """Tracked bodies (must include the anchor)."""
    joint_names: list[str] = MISSING
    """Motor joints (command, metrics, RSI noise), in contract order."""
    root_body_name: str = "world"

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Additive uniform noise on the serial MOTOR reset positions [rad]."""
    closure_joint_names: list[str] = []
    """Motors (subset of ``joint_names``) that drive a loop closure; they get ``closure_joint_position_range``."""
    closure_joint_position_range: tuple[float, float] = (0.0, 0.0)
    """Additive uniform noise on closure-coupled motor reset positions [rad] (tears the closure if != 0)."""
    start_at_zero: bool = False
    """Play mode: always (re)start the motion at frame 0."""
    continuous_loop: bool = False
    """Evaluation (with ``start_at_zero``): at the clip end only the reference clock wraps to frame 0; the robot state
    is NOT rewritten (default ``False``: the clip end resamples, i.e. writes the NPZ frame-0 state into the sim).
    Episode resets after a termination still write the reference state."""
    allow_rejected_motion: bool = False
    """Accept an NPZ whose ``meta.status`` is ``"rejected"`` (``tools/settle_motion.py`` quality gate) or whose sibling
    ``<clip>.validation.json`` verdict is ``"rejected"`` (``tools/validate_motion_npz.py``). Exploratory/debug only;
    ``scripts/train.py --allow_rejected_motion`` records it in the run dir (``motion_acceptance.json``)."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
