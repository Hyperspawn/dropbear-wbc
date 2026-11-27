"""Unitree-style low-level SDK for Dropbear (``dropbear_hg-v1``, CONTRACTS section 6).

Importable with only numpy, msgpack and pyzmq; no simulator dependency. The transport names (pyzmq) load on first
use, so ``motors``/``types`` and :mod:`dropbear_wbc.deploy.config` also import in a Python without pyzmq, such as
Isaac Sim's (``tools/hw_stand_test.py --motor_pose_json``, item 390).
"""
from .motors import (DEFAULT_KD, DEFAULT_KP, DEFAULT_POS, EFFORT_LIMIT, MOTOR_NAMES, NECK_NAMES, NUM_MOTORS,
                     NUM_NECK)
from .signals import STOP_SIGNALS, install_stop_handler
from .types import (IMU_AGE_UNKNOWN, MOTOR_AGE_UNKNOWN, MOTION_MODE_KP500_RAIL_ERR_RAD, MOTION_MODE_TAU_RAIL_NM,
                    TAU_SRC_DIRECT, TAU_SRC_IQ_KT, SCHEMA, CrcError, IMUState, LowCmd, LowState, MotorCmd, MotorCmdBlock, MotorMode, MotorState,
                    MotorStateBlock, SimState, TICK_MOD, TickUnwrapper, motion_mode_tau_valid, tau_est_with_iq_fallback,
                    tick_diff)

__all__ = [
    "DEFAULT_KD", "DEFAULT_KP", "DEFAULT_POS", "EFFORT_LIMIT", "MOTOR_NAMES", "NECK_NAMES", "NUM_MOTORS", "NUM_NECK",
    "LOWCMD", "LOWSTATE", "ChannelFactoryInitialize", "ChannelPublisher", "ChannelSubscriber", "Endpoints",
    "ZmqPublisher", "ZmqSubscriber", "IMU_AGE_UNKNOWN", "MOTOR_AGE_UNKNOWN", "MOTION_MODE_KP500_RAIL_ERR_RAD",
    "MOTION_MODE_TAU_RAIL_NM", "TAU_SRC_DIRECT", "TAU_SRC_IQ_KT", "SCHEMA", "CrcError", "IMUState", "LowCmd", "LowState",
    "MotorCmd", "MotorCmdBlock", "MotorMode", "MotorState", "MotorStateBlock", "SimState",
    "TICK_MOD", "TickUnwrapper", "motion_mode_tau_valid", "tau_est_with_iq_fallback", "tick_diff",
    "STOP_SIGNALS", "install_stop_handler",
]

_TRANSPORT_NAMES = frozenset({"LOWCMD", "LOWSTATE", "ChannelFactoryInitialize", "ChannelPublisher", "ChannelSubscriber",
                              "Endpoints", "ZmqPublisher", "ZmqSubscriber"})


def __getattr__(name: str):
    if name in _TRANSPORT_NAMES:
        from . import transport
        return getattr(transport, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
