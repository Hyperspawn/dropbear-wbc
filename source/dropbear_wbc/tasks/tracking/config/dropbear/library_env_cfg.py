"""Dropbear motion-LIBRARY tracking configs (contract ``dropbear-tracking-library-v1``, docs/CONTRACTS.md 5.3).

Same MDP as the single-clip tracking task (``flat_env_cfg``: actions, observations, rewards, terminations,
randomization, timing, solver) with the motion command swapped for :class:`MotionLibraryCommand` (N clips from a
``dropbear-motion-library-v1`` manifest). Only the command term differs:

* ``Dropbear-Tracking-Library-v0`` / ``-Play-v0``: command = reference motor pos + vel (44), identical layout to
  single-clip, so the observation vector (125) and the deploy observation builder are unchanged.
* ``Dropbear-Tracking-Library-Future-v0`` / ``-Play-v0``: command additionally carries the motor pos + vel of the
  reference +0.1 s and +0.2 s ahead (``FUTURE_STEPS`` = (5, 10) frames at 50 Hz): 44 * 3 = 132, obs 213.

The manifest is set by ``scripts/train.py --motion_library <manifest.json>`` (``commands.motion.manifest``).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from ...mdp.library_commands import library_command_from
from .flat_env_cfg import DropbearFlatEnvCfg, DropbearFlatPlayEnvCfg

FUTURE_STEPS: tuple[int, ...] = (5, 10)
"""Future reference frames of the ``-Future`` tasks (policy-rate frames: +0.1 s, +0.2 s)."""


def use_motion_library(cfg, future_steps: tuple[int, ...] = ()) -> None:
    """Replace ``cfg.commands.motion`` by a library command with every single-clip setting carried over."""
    cfg.commands.motion = library_command_from(cfg.commands.motion)
    cfg.commands.motion.future_steps = tuple(future_steps)


@configclass
class DropbearLibraryEnvCfg(DropbearFlatEnvCfg):
    """Training config over a motion library (single-clip MDP, library command)."""

    def __post_init__(self):
        super().__post_init__()
        use_motion_library(self)


@configclass
class DropbearLibraryPlayEnvCfg(DropbearFlatPlayEnvCfg):
    """Play config (no randomization, frame 0): env i plays clip i % N (``commands.motion.play_clips`` to choose)."""

    def __post_init__(self):
        super().__post_init__()
        use_motion_library(self)


@configclass
class DropbearLibraryFutureEnvCfg(DropbearFlatEnvCfg):
    """Library training config whose command also carries the +0.1/+0.2 s reference frames."""

    def __post_init__(self):
        super().__post_init__()
        use_motion_library(self, FUTURE_STEPS)


@configclass
class DropbearLibraryFuturePlayEnvCfg(DropbearFlatPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_motion_library(self, FUTURE_STEPS)


def drop_privileged_policy_obs(cfg) -> None:
    """Remove the policy terms a real robot cannot measure (BeyondMimic ``Wo-State-Estimation``).

    ``motion_anchor_pos_b`` (the reference anchor position in the robot anchor frame needs the robot's world position)
    and ``base_lin_vel`` (needs a velocity estimator) are simulator truth. The critic keeps them (asymmetric
    actor-critic). Policy observation 125 -> 119 (Library-v0 layout).
    """
    cfg.observations.policy.motion_anchor_pos_b = None
    cfg.observations.policy.base_lin_vel = None


@configclass
class DropbearLibraryNoStateEnvCfg(DropbearLibraryEnvCfg):
    """Library training config whose POLICY sees no simulator-only state (deployable without a state estimator)."""

    def __post_init__(self):
        super().__post_init__()
        drop_privileged_policy_obs(self)


@configclass
class DropbearLibraryNoStatePlayEnvCfg(DropbearLibraryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        drop_privileged_policy_obs(self)


@configclass
class DropbearFlatNoStateEnvCfg(DropbearFlatEnvCfg):
    """Single-clip tracking WITHOUT simulator-only state in the policy (no ``motion_anchor_pos_b``, no
    ``base_lin_vel``; the critic keeps them): the deployable single-clip variant (added 2026-09-25 evening)."""

    def __post_init__(self):
        super().__post_init__()
        drop_privileged_policy_obs(self)


@configclass
class DropbearFlatNoStatePlayEnvCfg(DropbearFlatPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        drop_privileged_policy_obs(self)


def use_sensor_dr(cfg, joint_lsb: float = 0.38e-3, action_delay_steps: tuple[int, int] = (0, 3)) -> None:
    """Sensor-side DR on the POLICY group (``mdp/sensor_dr``) + a wider command delay on ``hw_*`` actuators.

    ``joint_lsb``: 0.38e-3 rad = 0x400/0x500 motion-mode replies; 0.01745 = the 0xA1/A2/A4 replies (1 deg).
    ``action_delay_steps``: physics steps (5 ms). (0, 3) = 0-15 ms: notes/39 chain model gives policy tick -> torque
    applied 0.9-9.6 ms (200 Hz MIT round, async HAL), p99 10.7 ms with 1 ms round jitter. Not yet measured (S39).
    White noise already on the terms is kept. The critic group is untouched.
    """
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

    from ... import mdp
    from ...mdp import sensor_dr as sdr
    from ...tracking_env_cfg import motors_cfg

    cfg.events.sensor_errors = EventTerm(func=sdr.reset_imu_errors, mode="reset")
    pol = cfg.observations.policy
    pol.motion_anchor_ori_b = ObsTerm(func=sdr.measured_anchor_ori_b, params={"command_name": "motion"},
                                      noise=Unoise(n_min=-0.05, n_max=0.05))
    pol.base_ang_vel = ObsTerm(func=sdr.measured_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    pol.joint_pos = ObsTerm(func=sdr.measured_joint, params={"func": mdp.joint_pos_rel, "asset_cfg": motors_cfg(),
                                                             "lsb": float(joint_lsb)},
                            noise=Unoise(n_min=-0.01, n_max=0.01))
    pol.joint_vel = ObsTerm(func=sdr.measured_joint, params={"func": mdp.joint_vel_rel, "asset_cfg": motors_cfg()},
                            noise=Unoise(n_min=-0.5, n_max=0.5))
    for act in cfg.scene.robot.actuators.values():
        if hasattr(act, "max_delay") and hasattr(act, "min_delay") and act.max_delay > 0:
            act.min_delay, act.max_delay = int(action_delay_steps[0]), int(action_delay_steps[1])


@configclass
class DropbearLibraryNoStateSensorDREnvCfg(DropbearLibraryNoStateEnvCfg):
    """NoState library training + sensor-side DR (deployment target: 6-axis IMU, motion-mode joint replies)."""

    def __post_init__(self):
        super().__post_init__()
        use_sensor_dr(self)
