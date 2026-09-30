"""Velocity-task rewards: Isaac Lab H1 locomotion terms, adapted to Dropbear's bodies.

Copied from ``isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`` (BSD-3, Isaac Lab 2.2) where noted;
importing that package would walk every Isaac Lab task and needs ``isaaclab_assets`` (not installed in kit python).

Dropbear-specific:

* **Torso signals come from the anchor (chest) body**, never from the root frame height: the root ``world`` frame
  origin sits ~12.5 cm BELOW the soles (``dropbear_names`` docstring). Orientation uses the anchor link frame times
  ``ANCHOR_FRAME_OFFSET_WXYZ`` (world-aligned at rest, like G1/H1 ``torso_link``).
* **Feet are two bodies each** (sole plate ``*_skateboard_bearing_left_2`` + ankle cross ``*_basis_left_1``,
  ``FOOT_BODIES``). A foot is in contact when ANY of its bodies is; per-foot contact/air times are reduced over the
  foot's bodies. The sensor ``SceneEntityCfg`` must resolve the bodies grouped per foot, in order
  (``preserve_order=True``: left bodies, then right bodies); :func:`foot_groups` checks this.
* Velocities: ``root_lin_vel_w`` is the root body's CoM velocity (Isaac Lab 2.x), i.e. the torso+pelvis CoM, which is
  the intended quantity for the velocity command.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------------------------------- helpers
def foot_groups(sensor_cfg: SceneEntityCfg, bodies_per_foot: int) -> torch.Tensor:
    """``(num_feet, bodies_per_foot)`` sensor body indices; fails closed on a non-grouped selection."""
    ids = sensor_cfg.body_ids
    if isinstance(ids, slice):
        raise ValueError("foot sensor_cfg must select explicit bodies (preserve_order=True), not all bodies")
    ids = list(ids)
    if len(ids) % bodies_per_foot != 0 or len(ids) == 0:
        raise ValueError(f"{len(ids)} foot bodies are not a multiple of bodies_per_foot={bodies_per_foot}")
    return torch.tensor(ids, dtype=torch.long).view(-1, bodies_per_foot)


def anchor_up_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, up_offset_wxyz) -> torch.Tensor:
    """World-aligned anchor orientation ``q_link * offset`` (num_envs, 4)."""
    asset = env.scene[asset_cfg.name]
    q_link = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    off = torch.tensor(up_offset_wxyz, dtype=q_link.dtype, device=q_link.device).expand_as(q_link)
    return math_utils.quat_mul(q_link, off)


def anchor_projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, up_offset_wxyz) -> torch.Tensor:
    """Unit gravity in the world-aligned anchor frame (num_envs, 3); (0, 0, -1) when upright."""
    asset = env.scene[asset_cfg.name]
    return math_utils.quat_apply_inverse(anchor_up_quat_w(env, asset_cfg, up_offset_wxyz), asset.data.GRAVITY_VEC_W)


def _feet_contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, bodies_per_foot: int):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    groups = foot_groups(sensor_cfg, bodies_per_foot).to(env.device)
    contact_time = sensor.data.current_contact_time[:, groups]  # (N, feet, bodies)
    air_time = sensor.data.current_air_time[:, groups]
    in_contact = (contact_time > 0.0).any(dim=-1)  # (N, feet)
    foot_contact_time = contact_time.max(dim=-1).values
    foot_air_time = air_time.min(dim=-1).values
    return in_contact, foot_contact_time, foot_air_time


# ---------------------------------------------------------------------------------------------------- task
def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Isaac Lab (H1): xy velocity command tracking in the gravity-aligned (yaw) frame, exp kernel."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    err = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-err / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Isaac Lab (H1): yaw-rate command tracking in the world frame, exp kernel."""
    asset = env.scene[asset_cfg.name]
    err = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-err / std**2)


# ---------------------------------------------------------------------------------------------------- feet
def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 2
) -> torch.Tensor:
    """Isaac Lab biped air-time reward with multi-body feet.

    Rewards the time spent in the current single-stance mode (min over the two feet of contact time for the stance
    foot / air time for the swing foot), capped at ``threshold`` [s]; zero in double support/flight and for
    |command_xy| <= 0.1 m/s.
    """
    in_contact, contact_time, air_time = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, bodies_per_foot: int = 2
) -> torch.Tensor:
    """Isaac Lab foot-slide penalty with multi-body feet.

    Sum over feet of the planar CoM speed of the foot's first sensor body (the sole plate) while ANY body of that
    foot has had > 1 N contact force in the sensor history. ``asset_cfg`` selects the sole plates, one per foot, in
    the same foot order as ``sensor_cfg``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    groups = foot_groups(sensor_cfg, bodies_per_foot).to(env.device)
    forces = sensor.data.net_forces_w_history[:, :, groups.flatten(), :].norm(dim=-1).max(dim=1)[0]
    contacts = (forces.view(env.num_envs, groups.shape[0], bodies_per_foot) > 1.0).any(dim=-1)
    asset = env.scene[asset_cfg.name]
    vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    if vel.shape[1] != contacts.shape[1]:
        raise ValueError(f"asset_cfg selects {vel.shape[1]} bodies for {contacts.shape[1]} feet")
    return torch.sum(vel.norm(dim=-1) * contacts, dim=1)


# ---------------------------------------------------------------------------------------------------- torso
def anchor_flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, up_offset_wxyz=(1.0, 0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Squared xy of gravity in the world-aligned ANCHOR frame (Isaac Lab ``flat_orientation_l2`` on the chest)."""
    g = anchor_projected_gravity(env, asset_cfg, up_offset_wxyz)
    return torch.sum(torch.square(g[:, :2]), dim=1)


def ground_z(env: ManagerBasedRLEnv, ground_sensor: str | None = None) -> torch.Tensor:
    """Ground height under each robot (num_envs,): the mean hit height of the down-facing ray caster
    ``ground_sensor`` (rough terrain, ``flat_env_cfg.enable_rough_terrain``; used for rewards and terminations only,
    never observed, so the policy stays blind) or the env origin z (flat ground, every earlier run)."""
    if ground_sensor is None:
        return env.scene.env_origins[:, 2]
    hits = env.scene.sensors[ground_sensor].data.ray_hits_w[..., 2]
    ok = torch.isfinite(hits)
    n = ok.sum(dim=1).clamp(min=1)
    z = torch.where(ok, hits, torch.zeros_like(hits)).sum(dim=1) / n
    return torch.where(ok.any(dim=1), z, env.scene.env_origins[:, 2])


def anchor_height_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float,
                     ground_sensor: str | None = None) -> torch.Tensor:
    """(anchor z above the ground - target)^2 [m^2]; ``target_height`` = the settled standing anchor height of the
    reset NPZ (reset-calibrated and physically loaded, never the unloaded CAD height; dropbear-research AGENTS.md
    rules 15/25). Ground = :func:`ground_z`."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2] - ground_z(env, ground_sensor)
    return torch.square(z - target_height)


def motor_torque_over_rated_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, rated: list[float]) -> torch.Tensor:
    """Sum over motors of ``relu(|motor torque| - rated)^2`` [N^2*m^2]: a thermal proxy for real-actuator profiles.

    ``rated`` is the continuous (thermal) torque per motor of ``asset_cfg.joint_ids`` (``hw_motor_specs``); peak torque
    is short-duty only, so a policy that holds a motor above its rated torque would overheat it on the robot.
    ``applied_torque`` of an explicit ``DatasheetMotor`` is the motor torque after the torque-speed envelope.
    """
    asset = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids].abs()
    over = torch.clamp(tau - torch.as_tensor(rated, device=tau.device, dtype=tau.dtype), min=0.0)
    return torch.sum(over * over, dim=1)


# ---------------------------------------------------------------------------------------------------- gait shaping
def _moving_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return ((torch.norm(cmd[:, :2], dim=1) > 0.1) | (cmd[:, 2].abs() > 0.1)).float()


def knee_flexion_in_swing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    target_flex: float, bodies_per_foot: int = 2,
) -> torch.Tensor:
    """Per swinging foot, its knee-crank flexion above the standing default, normalized by ``target_flex`` [rad] and
    capped at 1; summed over feet; zero when the command is ~0.

    Against the stiff-leg ("peg leg") gait: the H1 air-time reward is satisfied by swinging a straight leg from the hip,
    so nothing else asks a knee to bend. ``asset_cfg`` = the two knee motors, in the sensor's foot order (L, R); crank
    flexion is positive (0 = straight stop, ~0.52 rad = ~48 deg knee)."""
    in_contact, _, _ = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    asset = env.scene[asset_cfg.name]
    flex = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    r = torch.sum(torch.clamp(flex / target_flex, 0.0, 1.0) * (~in_contact).float(), dim=1)
    return r * _moving_command(env, command_name)


def feet_swing_height_l2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    stand_z: float, clearance: float, bodies_per_foot: int = 2, ground_sensor: str | None = None,
) -> torch.Tensor:
    """Unitree G1/H1 ``feet_swing_height``: sum over swinging feet of ``(z - (stand_z + clearance))^2`` of the sole
    body (height above the ground under the base, :func:`ground_z`; ``stand_z`` = its standing height); zero when the
    command is ~0."""
    in_contact, _, _ = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - ground_z(env, ground_sensor)[:, None]
    err = torch.square(z - (stand_z + clearance)) * (~in_contact).float()
    return torch.sum(err, dim=1) * _moving_command(env, command_name)


class motor_torque_rate_l2(ManagerTermBase):
    """Sum over motors of ``((tau_t - tau_{t-1}) / peak)^2`` between policy steps (motor torque after the actuator
    model, relative to each motor's peak): penalizes torque spikes a 48 V drive would have to follow."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        ids = list(range(self.asset.num_joints))[ids] if isinstance(ids, slice) else list(ids)
        self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        peak = torch.ones(len(ids), device=env.device)
        for act in self.asset.actuators.values():
            jids = act.joint_indices
            jids = list(range(self.asset.num_joints))[jids] if isinstance(jids, slice) else jids.tolist()
            for k, j in enumerate(jids):
                if j in ids:
                    lim = float(act.effort_limit[0, k])
                    peak[ids.index(j)] = lim if 0.0 < lim < 1.0e6 else 100.0
        self.peak = peak
        self.prev = self.asset.data.applied_torque[:, self.ids].clone()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev[env_ids] = self.asset.data.applied_torque[env_ids][:, self.ids]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        tau = self.asset.data.applied_torque[:, self.ids]
        d = (tau - self.prev) / self.peak
        self.prev = tau.clone()
        return torch.sum(d * d, dim=1)


def feet_mode_time_exceeded(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, max_contact_s: float, max_air_s: float,
    bodies_per_foot: int = 2,
) -> torch.Tensor:
    """Sum over feet of the time a foot has stayed in its current mode beyond the limit [s] (contact > ``max_contact_s``
    or air > ``max_air_s``) while the command is non-zero. Against hopping on one leg with the other held up: the
    biped air-time reward is maximal for a PERMANENT single stance (it is capped, not penalized)."""
    in_contact, contact_time, air_time = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    over = torch.where(in_contact, torch.clamp(contact_time - max_contact_s, min=0.0),
                       torch.clamp(air_time - max_air_s, min=0.0))
    return torch.sum(over, dim=1) * _moving_command(env, command_name)


def feet_air_when_standing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, bodies_per_foot: int = 2
) -> torch.Tensor:
    """Number of feet off the ground while the command is ~0 (stand still on both feet, no marching / one-leg stance)."""
    in_contact, _, _ = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    return torch.sum((~in_contact).float(), dim=1) * (1.0 - _moving_command(env, command_name))


def knee_flexion_in_stance(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, margin: float,
    bodies_per_foot: int = 2,
) -> torch.Tensor:
    """Sum over feet IN CONTACT of the knee-crank flexion beyond ``default + margin`` [rad] (and beyond 0 below the
    default: none). Pairs with :func:`knee_flexion_in_swing`: without it a policy satisfies "bent in swing" by walking
    crouched on the max-flexion stops (gait_scratch_600: both knees at 45-48 deg all the time)."""
    in_contact, _, _ = _feet_contact_state(env, sensor_cfg, bodies_per_foot)
    asset = env.scene[asset_cfg.name]
    flex = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(flex - margin, min=0.0) * in_contact.float(), dim=1)


def feet_lateral_distance_below(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float) -> torch.Tensor:
    """``relu(min_dist - lateral)`` [m], lateral = (left sole - right sole) along the root's yaw-frame y axis.

    Against the crossover ("scissor") gait: self-collisions are off in the sim, so the legs can pass through each
    other; gait_scratch_600 parked both hip rolls on their adduction stops and walked cross-legged (stance width at
    rest is only 0.138 m). ``asset_cfg`` = the two sole bodies, left then right."""
    asset = env.scene[asset_cfg.name]
    p = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    d = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), p[:, 0] - p[:, 1])
    return torch.clamp(min_dist - d[:, 1], min=0.0)


def joint_near_limit_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, margin: float) -> torch.Tensor:
    """Sum over joints of how deep [rad] each sits inside the last ``margin`` before a hard limit: hard stops are for
    over-travel, not for carrying load continuously (policies parked the knee and hip-roll motors on their stops)."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lim = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(lim[..., 0] + margin - q, min=0.0) + torch.clamp(q - (lim[..., 1] - margin), min=0.0), dim=1)
