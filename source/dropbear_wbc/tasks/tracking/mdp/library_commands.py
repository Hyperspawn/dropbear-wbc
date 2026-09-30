"""Motion-LIBRARY command (multi-clip BeyondMimic; contract ``dropbear-tracking-library-v1``, docs/CONTRACTS.md 5.3).

:class:`MotionLibraryCommand` is the single-clip :class:`~.commands.MotionCommand` generalised to N contract NPZs
(``tasks/tracking/motion_library.py``): every env carries a clip index and a local time step; the reference
(motor command, tracked bodies, anchor, RSI rows) is read at global row ``starts[clip] + t`` of concatenated tensors.
The single-clip path (``commands.py``) is untouched; this class inherits its robot-state properties, metrics and debug
visualisation and overrides everything that touches the reference.

Differences to the single-clip command, all deliberate:

* **Sampling.** RSI draws (clip, time-bin) from :class:`~..motion_library.LibrarySampler` (per-clip 1 s failure bins as
  upstream, plus a clip-level prior/hazard mixture). A clip end resamples a new (clip, time) like upstream's clip end.
* **Command.** Identical layout to single-clip (reference motor ``joint_pos`` 22 + ``joint_vel`` 22). With
  ``future_steps`` (e.g. ``(5, 10)`` = +0.1/+0.2 s) it appends ``joint_pos``/``joint_vel`` of each future frame of the
  SAME clip (clamped to its last frame): ``[q_t, dq_t, q_{t+k1}, dq_{t+k1}, ...]`` = 44 * (1 + K). Separate task id.
* **Play.** ``start_at_zero``: env ``i`` plays clip ``i % N`` (or ``play_clips``) from frame 0; ``continuous_loop`` wraps
  that clip's clock without rewriting the robot state.
* **Metrics.** The single-clip error metrics, plus per-clip scalars (broadcast so the command manager's per-episode
  mean equals them): ``clip_prob/<name>``, ``clip_fail_per_s/<name>`` (failure hazard x fps) and
  ``clip_err_joint/<name>`` (EMA of the mean motor-joint error of the envs on that clip), for the first
  ``max_logged_clips`` clips; ``clip_*_max`` aggregates cover all clips.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, sample_uniform, yaw_quat

from ..motion_library import LibrarySampler, MotionLibrary, build_library, future_indices, library_report, play_assignment
from ..motion_npz import expected_usd_sha256
from .commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLibraryCommand(MotionCommand):
    """Reference-motion command over a motion library (see module docstring)."""

    cfg: MotionLibraryCommandCfg

    def __init__(self, cfg: MotionLibraryCommandCfg, env: ManagerBasedRLEnv):  # noqa: C901
        CommandTerm.__init__(self, cfg, env)  # NOT MotionCommand.__init__ (that loads a single NPZ)
        self.robot = env.scene[cfg.asset_name]
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
        spawn = getattr(getattr(self.robot, "cfg", None), "spawn", None)
        spherical = tuple(getattr(spawn, "spherical_joint_overrides", ()) or ())
        if not cfg.manifest:
            raise ValueError("MotionLibraryCommandCfg.manifest is empty (scripts/train.py --motion_library <manifest.json>)")
        data = build_library(
            cfg.manifest, joint_names=self.robot.joint_names, body_names=self.robot.body_names,
            motor_names=list(cfg.joint_names), keep_body_names=list(cfg.body_names), root_body_name=cfg.root_body_name,
            expected_fps=policy_rate, expected_usd_sha256=expected_usd_sha256(),
            expected_authored_ankle=len(spherical) == 0, allow_rejected=cfg.allow_rejected_motion,
            require_accepted=cfg.require_accepted)
        self.motion = MotionLibrary(data, cfg.joint_names, cfg.body_names, device=self.device)
        self.library_info = library_report(data)
        self.clip_names = list(self.motion.clip_names)
        self.num_clips = self.motion.num_clips
        self.sampler = LibrarySampler(
            [c.num_frames for c in data.clips], self.motion.fps, [c.weight for c in data.clips],
            clip_weighting=cfg.clip_weighting, clip_adaptive_ratio=cfg.clip_adaptive_ratio,
            adaptive_kernel_size=cfg.adaptive_kernel_size, adaptive_lambda=cfg.adaptive_lambda,
            adaptive_uniform_ratio=cfg.adaptive_uniform_ratio, adaptive_alpha=cfg.adaptive_alpha,
            clip_alpha=cfg.clip_alpha, device=self.device)
        self.future_steps = tuple(int(k) for k in cfg.future_steps)
        if any(k <= 0 for k in self.future_steps):
            raise ValueError(f"future_steps must be positive frame offsets, got {self.future_steps}")

        # static clips built for a default pose (tools/make_static_npz.py): same staleness warning as single-clip
        self.npz_default_pose_max_abs_diff: float | None = None
        live = self.robot.data.default_joint_pos[0, self.motor_ids].cpu()
        for c in data.clips:
            dmp = c.default_motor_pos
            if isinstance(dmp, dict) and all(n in dmp for n in cfg.joint_names):
                ref = torch.tensor([float(dmp[n]) for n in cfg.joint_names])
                d = float((live - ref).abs().max())
                self.npz_default_pose_max_abs_diff = max(d, self.npz_default_pose_max_abs_diff or 0.0)
                if d > 0.02:
                    print(f"[dropbear MotionLibraryCommand] WARNING: clip {c.name!r} was built for a default motor pose "
                          f"{d:.3f} rad away from the env's; rebuild it?", flush=True)

        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._play_clip_ids = play_assignment(self.num_envs, self.num_clips, self.clip_names, cfg.play_clips,
                                              device=self.device)
        n_bodies = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, n_bodies, 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.clip_err_joint_ema = torch.zeros(self.num_clips, device=self.device)
        self._clip_err_seen = torch.zeros(self.num_clips, dtype=torch.bool, device=self.device)

        for key in (
            "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
            "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
            "error_joint_pos", "error_joint_vel", "sampling_entropy", "sampling_top1_prob", "sampling_top1_clip",
            "clip_prob_min", "clip_fail_per_s_max", "clip_err_joint_max",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        self._logged_clips = list(range(min(self.num_clips, max(int(cfg.max_logged_clips), 0))))
        for c in self._logged_clips:
            for prefix in ("clip_prob", "clip_fail_per_s", "clip_err_joint"):
                self.metrics[f"{prefix}/{self.clip_names[c]}"] = torch.zeros(self.num_envs, device=self.device)
        print(f"[dropbear MotionLibraryCommand] library {data.fingerprint['name']!r} sha {data.fingerprint['sha256'][:12]}: "
              f"{self.num_clips} clips, {self.motion.num_frames_total} frames "
              f"({self.library_info['duration_s_total']:.1f} s), {self.sampler.bin_count} time bins, "
              f"future_steps={self.future_steps}", flush=True)

    # ------------------------------------------------------------------ single-clip compatibility
    @property
    def bin_count(self) -> int:
        return self.sampler.bin_count

    @property
    def bin_failed_count(self) -> torch.Tensor:
        return self.sampler.bin_failed_count

    def extra_train_state(self) -> dict:
        """Clip-level sampler state for ``runner.DropbearOnPolicyRunner`` (bins are covered by ``bin_failed_count``)."""
        return {"library_sha256": self.library_info["sha256"], "sampler": self.sampler.state_dict()}

    def load_extra_train_state(self, state: dict) -> bool:
        if state.get("library_sha256") != self.library_info["sha256"]:
            return False
        return self.sampler.load_state_dict(state.get("sampler") or {})

    # ------------------------------------------------------------------ reference (global rows)
    @property
    def _gidx(self) -> torch.Tensor:
        return self.motion.global_index(self.clip_ids, self.time_steps)

    @property
    def command(self) -> torch.Tensor:
        """Reference motor pos [rad] + vel [rad/s] (num_envs, 44), then the same for each future frame."""
        g = self._gidx
        parts = [self.motion.joint_pos[g], self.motion.joint_vel[g]]
        fut = future_indices(self.motion, self.clip_ids, self.time_steps, self.future_steps)
        if fut is not None:
            for j in range(fut.shape[1]):
                parts += [self.motion.joint_pos[fut[:, j]], self.motion.joint_vel[fut[:, j]]]
        return torch.cat(parts, dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self._gidx]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self._gidx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._gidx] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._gidx]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._gidx]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._gidx]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._gidx, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._gidx, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._gidx, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._gidx, self.motion_anchor_body_index]

    # ------------------------------------------------------------------ manager hooks
    def _update_metrics(self):
        super()._update_metrics()
        # per-clip EMA of the mean motor-joint error of the envs on each clip
        err = self.metrics["error_joint_pos"]
        sums = torch.zeros(self.num_clips, device=self.device).index_add_(0, self.clip_ids, err)
        cnt = torch.bincount(self.clip_ids, minlength=self.num_clips).float()
        seen = cnt > 0
        mean = sums / cnt.clamp(min=1.0)
        a = self.cfg.clip_alpha_metrics
        first = seen & ~self._clip_err_seen
        self.clip_err_joint_ema = torch.where(first, mean, torch.where(seen, a * mean + (1 - a) * self.clip_err_joint_ema,
                                                                        self.clip_err_joint_ema))
        self._clip_err_seen |= seen
        probs = self.sampler.clip_probs()
        haz = self.sampler.clip_hazard() * self.motion.fps
        self.metrics["clip_prob_min"][:] = probs.min()
        self.metrics["clip_fail_per_s_max"][:] = haz.max()
        self.metrics["clip_err_joint_max"][:] = self.clip_err_joint_ema.max()
        for c in self._logged_clips:
            name = self.clip_names[c]
            self.metrics[f"clip_prob/{name}"][:] = probs[c]
            self.metrics[f"clip_fail_per_s/{name}"][:] = haz[c]
            self.metrics[f"clip_err_joint/{name}"][:] = self.clip_err_joint_ema[c]

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        self.sampler.record_failures(self.clip_ids[env_ids], self.time_steps[env_ids], episode_failed)
        clips, t = self.sampler.sample(len(env_ids))
        self.clip_ids[env_ids] = clips
        self.time_steps[env_ids] = t
        p = self.sampler.joint_probs()
        pmax, imax = p.max(dim=0)
        ent = -(p * (p + 1e-12).log()).sum()
        b = self.sampler.bin_count
        self.metrics["sampling_entropy"][:] = ent / math.log(b) if b > 1 else 0.0
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_clip"][:] = self.sampler.bin_clip[imax].float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        if self.cfg.start_at_zero:
            self.clip_ids[env_ids] = self._play_clip_ids[env_ids]
            self.time_steps[env_ids] = 0
        else:
            self._adaptive_sampling(env_ids)
        g = self.motion.global_index(self.clip_ids[env_ids], self.time_steps[env_ids])
        n = len(env_ids)

        # -- root (the NPZ 'world' body): link pose + CoM velocity, plus BeyondMimic pose/velocity noise
        root_pos = self.motion.root_pos_w[g] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.root_quat_w[g].clone()
        root_lin_vel = self.motion.root_lin_vel_w[g].clone()
        root_ang_vel = self.motion.root_ang_vel_w[g].clone()
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

        # -- joints: FULL settled row; noise + hard-limit clip on the motors only (as single-clip)
        joint_pos = self.motion.full_joint_pos[g].clone()
        joint_vel = self.motion.full_joint_vel[g].clone()
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
        # relative targets = the reference itself until the next _update_command (single-clip fix, commands.py)
        self.body_pos_relative_w[env_ids] = self.motion.body_pos_w[g] + self._env.scene.env_origins[env_ids][:, None, :]
        self.body_quat_relative_w[env_ids] = self.motion.body_quat_w[g]

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.lengths[self.clip_ids])[0]
        if self.cfg.continuous_loop and self.cfg.start_at_zero:
            self.time_steps[env_ids] = 0  # evaluation: wrap this clip's clock only (robot state carried over)
        else:
            self._resample_command(env_ids)
        self.sampler.record_exposure(self.clip_ids)

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
        self.sampler.step()


@configclass
class MotionLibraryCommandCfg(MotionCommandCfg):
    """Configuration for :class:`MotionLibraryCommand` (all single-clip fields keep their meaning)."""

    class_type: type = MotionLibraryCommand

    motion_file: str = ""
    """Unused (single-clip field); the library comes from :attr:`manifest`."""
    manifest: str = MISSING
    """``dropbear-motion-library-v1`` JSON (tasks/tracking/motion_library.py)."""
    require_accepted: bool = True
    """Every clip needs a non-stale ``accepted`` validator verdict (lifted by ``allow_rejected_motion``)."""
    clip_weighting: str = "duration"
    """Clip prior: ``"duration"`` (weight x frames; time-uniform) or ``"uniform"`` (weight only)."""
    clip_adaptive_ratio: float = 0.5
    """Share of the clip distribution that follows the per-clip failure hazard (0 = prior only)."""
    clip_alpha: float = 0.001
    """EMA rate (per env step) of the clip failure/exposure statistics."""
    clip_alpha_metrics: float = 0.01
    """EMA rate (per env step) of the logged per-clip joint error."""
    future_steps: tuple[int, ...] = ()
    """Future reference frames appended to the command (frames at the policy rate, e.g. (5, 10) = +0.1/+0.2 s)."""
    play_clips: list[str] = []
    """Play (``start_at_zero``): env i plays ``play_clips[i % len]`` (default: every clip, env i -> clip i % N)."""
    max_logged_clips: int = 32
    """Per-clip metric keys are logged for the first this-many clips (aggregates cover all)."""


LIBRARY_ONLY_FIELDS = ("manifest", "require_accepted", "clip_weighting", "clip_adaptive_ratio", "clip_alpha",
                       "clip_alpha_metrics", "future_steps", "play_clips", "max_logged_clips")


def library_command_from(single: MotionCommandCfg, manifest: str = "") -> MotionLibraryCommandCfg:
    """A :class:`MotionLibraryCommandCfg` carrying every field of a single-clip ``MotionCommandCfg`` (ranges, bodies,
    play flags, adaptive parameters, ...) so the library task inherits the tracking task's settings unchanged."""
    lib = MotionLibraryCommandCfg(asset_name=single.asset_name, manifest=manifest, anchor_body_name=single.anchor_body_name,
                                  body_names=list(single.body_names), joint_names=list(single.joint_names))
    for f in fields(single):
        if f.name in ("class_type", "motion_file") or f.name in LIBRARY_ONLY_FIELDS:
            continue
        setattr(lib, f.name, getattr(single, f.name))
    return lib
