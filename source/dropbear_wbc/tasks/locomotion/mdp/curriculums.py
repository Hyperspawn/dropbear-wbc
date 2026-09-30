"""Command-range curriculum (unitree_rl_lab ``lin_vel_cmd_levels`` / ``ang_vel_cmd_levels``, Apache-2.0, adapted).

Every time the global step counter crosses a multiple of the episode length, the mean per-second episode sum of the
tracking reward over the envs being reset is compared with ``success_ratio * weight``; on success the command ranges
widen by ``delta`` at both ends, clamped to ``limit_ranges``. Unlike upstream, the linear-y range has its own delta
and the yaw-rate range is handled by the same term (one term, three axes). The widened ranges live in the command
term's cfg; :class:`dropbear_wbc.tasks.locomotion.runner.LocomotionOnPolicyRunner` checkpoints and restores them,
because the chunked trainer restarts Isaac every chunk.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _widen(rng, delta: float, limit) -> tuple[float, float]:
    lo = max(float(rng[0]) - delta, float(limit[0]))
    hi = min(float(rng[1]) + delta, float(limit[1]))
    return (lo, hi)


def velocity_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    lin_reward_term: str = "track_lin_vel_xy_exp",
    ang_reward_term: str = "track_ang_vel_z_exp",
    success_ratio: float = 0.8,
    delta_lin_x: float = 0.1,
    delta_lin_y: float = 0.1,
    delta_ang_z: float = 0.1,
) -> torch.Tensor:
    """Widen the linear (x, y) and yaw-rate command ranges on tracking success; returns the current max vx."""
    term = env.command_manager.get_term(command_name)
    ranges, limits = term.cfg.ranges, term.cfg.limit_ranges
    if len(env_ids) > 0 and env.common_step_counter % env.max_episode_length == 0:
        ids = torch.as_tensor(env_ids, device=env.device)
        sums = env.reward_manager._episode_sums
        lin = torch.mean(sums[lin_reward_term][ids]) / env.max_episode_length_s
        if lin > env.reward_manager.get_term_cfg(lin_reward_term).weight * success_ratio:
            ranges.lin_vel_x = _widen(ranges.lin_vel_x, delta_lin_x, limits.lin_vel_x)
            ranges.lin_vel_y = _widen(ranges.lin_vel_y, delta_lin_y, limits.lin_vel_y)
        ang = torch.mean(sums[ang_reward_term][ids]) / env.max_episode_length_s
        if ang > env.reward_manager.get_term_cfg(ang_reward_term).weight * success_ratio:
            ranges.ang_vel_z = _widen(ranges.ang_vel_z, delta_ang_z, limits.ang_vel_z)
    return torch.tensor(float(ranges.lin_vel_x[1]), device=env.device)


def command_ranges_dict(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> dict[str, list[float]]:
    """Current command ranges (for logs / checkpoints)."""
    r = env.command_manager.get_term(command_name).cfg.ranges
    return {k: [float(v) for v in getattr(r, k)] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}
