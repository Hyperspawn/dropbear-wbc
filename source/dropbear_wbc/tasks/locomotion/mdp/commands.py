"""Velocity command with curriculum limits (unitree_rl_lab ``UniformLevelVelocityCommandCfg``, Apache-2.0)."""
from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` + ``limit_ranges``: the bounds the curriculum may widen ``ranges`` to."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
