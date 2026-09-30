"""Contract name tables used by the motion pipeline (mirrors of docs/CONTRACTS.md sections 1-2).

The authoritative Python constants live in other modules written in parallel
(``dropbear_wbc.robots.dropbear.MOTOR_NAMES``, ``dropbear_wbc.sdk.motors.MOTOR_NAMES``,
``dropbear_wbc.kinematics.semantic``). :func:`check_against_package` asserts that these mirrors
agree with whichever of them are importable, so a contract change cannot silently diverge.
"""

from __future__ import annotations

import importlib

__all__ = ["MOTOR_NAMES", "SEMANTIC_NAMES", "ROOT_COLUMNS", "CSV_COLUMNS", "check_against_package"]

#: ``dropbear-wbc-motors-v1`` (CONTRACTS.md section 1), SDK slot order.
MOTOR_NAMES: tuple[str, ...] = (
    "PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint", "LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81",
    "PG_right_leg_pitch", "PG_right_leg_roll", "RL_hip_joint", "RL_knee_actuator_joint", "RL_Revolute67", "RL_Revolute81",
    "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
    "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
)

#: ``dropbear-semantic-v1`` (CONTRACTS.md section 2) order.
SEMANTIC_NAMES: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
)
SEMANTIC_INDEX: dict[str, int] = {n: i for i, n in enumerate(SEMANTIC_NAMES)}

#: ``dropbear-motion-csv-v1`` root columns (CONTRACTS.md section 3).
ROOT_COLUMNS: tuple[str, ...] = ("root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw")
CSV_COLUMNS: tuple[str, ...] = ROOT_COLUMNS + MOTOR_NAMES


def check_against_package() -> dict[str, str]:
    """Compare the mirrors with the package constants that exist. Returns {module: 'ok'|'absent'}.

    Raises ``AssertionError`` on any mismatch.
    """
    status: dict[str, str] = {}
    for mod_name, attr, ours in (
        ("dropbear_wbc.robots.dropbear", "MOTOR_NAMES", MOTOR_NAMES),
        ("dropbear_wbc.robots.dropbear_names", "MOTOR_NAMES", MOTOR_NAMES),
        ("dropbear_wbc.sdk.motors", "MOTOR_NAMES", MOTOR_NAMES),
        ("dropbear_wbc.kinematics.semantic", "SEMANTIC_NAMES", SEMANTIC_NAMES),
        ("dropbear_wbc.kinematics.semantic", "MOTOR_NAMES", MOTOR_NAMES),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - absent or needs Isaac; both mean "cannot check"
            status[f"{mod_name}.{attr}"] = "absent"
            continue
        theirs = getattr(mod, attr, None)
        if theirs is None:
            status[f"{mod_name}.{attr}"] = "absent"
            continue
        assert tuple(theirs) == ours, f"{mod_name}.{attr} differs from motion.names mirror"
        status[f"{mod_name}.{attr}"] = "ok"
    return status
