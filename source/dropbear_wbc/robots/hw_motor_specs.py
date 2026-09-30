"""Real-actuator ("digital twin v2") motor specs for Dropbear. Pure Python: no Isaac/torch imports.

Numbers come from ``data/robot/actuators_datasheet_v1.json`` (built by ``tools/build_actuator_datasheet.py`` from the
MyActuator PDFs in ``github.com/Hyperspawn/myactuator-can``) plus the user's actuator map of 2026-09-25
(``myactuator-can/AttemptToExplainActuators.md``). Everything is at the actuator OUTPUT shaft, SI units.

Model per joint (``tasks``' ``hw_*`` actuator profiles, :mod:`dropbear_wbc.robots.hw_actuators`):

* torque-speed envelope: ``|tau| <= min(peak, saturation * (1 - |w| / no_load))`` in the driving quadrant
  (Isaac Lab ``DCMotor`` line, chord of the vendor curve);
* armature = rotor inertia x ratio^2 (the vendor "Inertia" field is rotor x ratio, NOT squared);
* Coulomb friction ~ the vendor back-drive torque, plus a small viscous term (both randomized per episode);
* PD gains inside the RMD / CEM motion-mode ("MIT") command range: kp <= 500, kd <= 5 (``PROTO_V39`` p89-90,
  CEM protocol V4.4 p153);
* command latency (position target delayed 0..``max_delay`` physics steps, per episode).

The knee (CEM-60) has NO public datasheet: the CEM manual in the corpus covers CEM-15/25/45 only. Its entry is
PROVISIONAL, extrapolated from that series (all 30:1 cycloid, 48 V, rated ~0.57 x peak, rated speed ~0.8 x no-load,
rotor inertia 0.7 kg*cm^2) and from ``dropbear_docs/docs/03-assembly/legs/actuators.md`` ("CEM-60 60 Nm 60 RPM").
Replace it with measured values as soon as the motor is on a bench (docs/ACTUATORS.md).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .dropbear_names import MOTOR_NAMES


@dataclass(frozen=True)
class MotorModel:
    """One actuator model at its output shaft."""

    name: str
    gear_ratio: float
    peak_torque: float
    """Short-duty peak torque [N*m] (the effort limit)."""
    rated_torque: float
    """Continuous (thermal) torque [N*m]."""
    no_load_speed: float
    """No-load output speed at 48 V [rad/s] (the torque-speed line's zero-torque point)."""
    saturation_effort: float
    """Torque-speed line intercept at w = 0 [N*m] (>= peak; the line is clipped at ``peak_torque``)."""
    rotor_inertia: float
    """Motor-side rotor inertia [kg*m^2]."""
    coulomb_friction: float
    """Output-side Coulomb friction [N*m] (~ vendor back-drive torque)."""
    viscous_friction: float
    """Output-side viscous friction [N*m*s/rad] (not published; small nominal value, randomized)."""
    mass_kg: float
    status: str
    """``datasheet`` or ``provisional`` (extrapolated, see the module docstring)."""

    @property
    def armature(self) -> float:
        """Reflected rotor inertia rotor x ratio^2 [kg*m^2]."""
        return self.rotor_inertia * self.gear_ratio**2


MOTOR_MODELS: dict[str, MotorModel] = {
    m.name: m
    for m in (
        MotorModel("RMD-X10-S2-V3-1:35", 35, 100.0, 50.0, 5.6147, 768.25, 5.675e-4, 2.88, 0.30, 1.70, "datasheet"),
        MotorModel("RMD-X10-V3-1:7", 7, 40.0, 15.0, 19.5389, 132.91, 5.675e-4, 0.62, 0.05, 1.15, "datasheet"),
        MotorModel("RMD-X8-Pro-V2-1:9", 9, 25.0, 10.0, 16.7552, 54.74, 3.4e-4, 0.61, 0.05, 0.71, "datasheet"),
        # PROVISIONAL: no CEM-60 datasheet (see module docstring). no-load 60 rpm (user docs, conservative vs the
        # series trend ~85 rpm); rated 34 N*m at 0.8 x no-load -> saturation 34 / 0.2 = 170 N*m; back-drive unknown.
        MotorModel("EPS-CEM-60", 30, 60.0, 34.0, 60.0 * 2.0 * math.pi / 60.0, 170.0, 0.7e-4, 2.0, 0.30, 1.20,
                   "provisional"),
    )
}

MOTION_MODE_LIMITS: dict[str, float] = {"kp_max": 500.0, "kd_max": 5.0}
"""RMD motion mode (PROTO_V39 p89-90) and CEM protocol V4.4 p153: kp 0..500, kd 0..5 (12-bit fields)."""

JOINT_MOTOR_MAPS: dict[str, dict[str, str]] = {
    # the user's map of 2026-09-25 (AttemptToExplainActuators.md): hip roll ("Hip Spreader") and hip yaw ("Leg
    # Rotator") X10 1:7, hip pitch ("Waist Pivot") X10-S2 1:35, knee ("Knee Bender") CEM-60, calf X8 Pro 1:9,
    # shoulder pitch ("Shoulder Rotator") X10 1:7, other arm joints X8 Pro 1:9
    "user_2026_09_25": {
        **{f"PG_{s}_leg_pitch": "RMD-X10-V3-1:7" for s in ("left", "right")},
        **{f"PG_{s}_leg_roll": "RMD-X10-V3-1:7" for s in ("left", "right")},
        **{f"{p}_hip_joint": "RMD-X10-S2-V3-1:35" for p in ("LL", "RL")},
        **{f"{p}_knee_actuator_joint": "EPS-CEM-60" for p in ("LL", "RL")},
        **{f"{p}_Revolute{n}": "RMD-X8-Pro-V2-1:9" for p in ("LL", "RL") for n in (67, 81)},
        **{f"{p}_yaw": "RMD-X10-V3-1:7" for p in ("LH", "RH")},
        **{f"{p}_{j}": "RMD-X8-Pro-V2-1:9" for p in ("LH", "RH") for j in ("pitch", "roll", "elbow_joint", "wrist_roll")},
    },
}
# hardware A/B (2026-09-25 evening): the user's map with ONLY the knee upgraded to the X10-S2 1:35 (the CAD's knee
# motor): isolates whether a stronger knee lets the bent-knee (human-reference) walk stay inside the motor ratings
JOINT_MOTOR_MAPS["user_knee_x10s2"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"{p}_knee_actuator_joint": "RMD-X10-S2-V3-1:35" for p in ("LL", "RL")},
}
# the CAD/USD motor components (tools/probe_usd_motor_prims.py): X10-S2 on hip roll, hip pitch AND knee
JOINT_MOTOR_MAPS["cad_usd"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"PG_{s}_leg_pitch": "RMD-X10-S2-V3-1:35" for s in ("left", "right")},
    **{f"{p}_knee_actuator_joint": "RMD-X10-S2-V3-1:35" for p in ("LL", "RL")},
}

HW_GAINS: dict[str, tuple[float, float]] = {
    # joint regex role -> (kp [N*m/rad], kd [N*m*s/rad]) at the motor output, inside MOTION_MODE_LIMITS
    "hip_roll": (150.0, 5.0),
    "hip_yaw": (150.0, 5.0),
    "hip_pitch": (200.0, 5.0),
    # knee crank: 500 / 5 reflects to ~155 / ~1.6 at the knee through the four-bar (G ~ 1.79 at the stand)
    "knee": (500.0, 5.0),
    "calf": (80.0, 4.0),
    "shoulder_pitch": (80.0, 3.0),
    "arm": (50.0, 2.0),
}
JOINT_ROLES: dict[str, str] = {
    **{f"PG_{s}_leg_pitch": "hip_roll" for s in ("left", "right")},
    **{f"PG_{s}_leg_roll": "hip_yaw" for s in ("left", "right")},
    **{f"{p}_hip_joint": "hip_pitch" for p in ("LL", "RL")},
    **{f"{p}_knee_actuator_joint": "knee" for p in ("LL", "RL")},
    **{f"{p}_Revolute{n}": "calf" for p in ("LL", "RL") for n in (67, 81)},
    **{f"{p}_yaw": "shoulder_pitch" for p in ("LH", "RH")},
    **{f"{p}_{j}": "arm" for p in ("LH", "RH") for j in ("pitch", "roll", "elbow_joint", "wrist_roll")},
}

MOTOR_CAN_IDS: dict[str, int] = {
    # legs: the Codex GR00T embodiment plan (dropbear_control/.../dropbear_embodiment.json, 0x140 + ID, UNVERIFIED on
    # the robot; which calf motor is "outer"/"inner" is not known, so calf A/B -> the plan's outer/inner is a guess)
    "LL_Revolute67": 1, "LL_Revolute81": 2, "RL_Revolute81": 3, "RL_Revolute67": 4,
    "LL_knee_actuator_joint": 5, "LL_hip_joint": 6, "RL_hip_joint": 7, "RL_knee_actuator_joint": 8,
    "PG_left_leg_roll": 9, "PG_left_leg_pitch": 10, "PG_right_leg_pitch": 11, "PG_right_leg_roll": 12,
    # arms: myactuator-can/README.md "assigned IDs" (right 21-25, left 31-35: wrist, elbow, hand rotate, shoulder out,
    # shoulder rot)
    "RH_wrist_roll": 21, "RH_elbow_joint": 22, "RH_roll": 23, "RH_pitch": 24, "RH_yaw": 25,
    "LH_wrist_roll": 31, "LH_elbow_joint": 32, "LH_roll": 33, "LH_pitch": 34, "LH_yaw": 35,
}
"""CAN motor ID per motor (command frame 0x140 + ID). Legs: planned, unverified; arms: the user's README."""

HW_PROFILE_MAPS: dict[str, str] = {
    "hw_v1": "user_2026_09_25",  # the user's actuator map (hip roll X10 1:7, knee CEM-60 provisional)
    "hw_v1_cad": "cad_usd",  # the CAD/USD motor components (X10-S2 on hip roll and knee)
    "hw_v1_knee_x10s2": "user_knee_x10s2",  # the user's map, knee upgraded to X10-S2 (hardware A/B)
    "hw_v1i": "user_2026_09_25",  # hw_v1 + target interpolation over the 4 physics steps (firmware-side smoothing)
    "hw_v1ie": "user_2026_09_25",  # hw_v1i + stiffer elbows (docs/ISSUES.md #24)
}
HW_PROFILE_OPTIONS: dict[str, dict] = {
    "hw_v1i": {"target_interp_steps": 4},
    # elbow kp 50 -> 200: at the joint kp / G^2 = 2.2 -> 8.7 N*m/rad through the 4.8x speed-up linkage, so the forearm
    # sags ~10 deg instead of ~40 deg under its own weight and the policy needs no large mass-specific target offset
    "hw_v1ie": {"target_interp_steps": 4, "gain_overrides": {"LH_elbow_joint": (200.0, 4.0), "RH_elbow_joint": (200.0, 4.0)}},
}
"""Extra ``make_hw_motor_groups`` options per profile (docs/ISSUES.md #11: 50 Hz targets stepped into a 200 Hz PD
make torque spikes; the motor-side firmware can ramp each new target over the 4 control ticks instead)."""
"""Actuator-profile name (tasks' ``set_actuator_profile``) -> :data:`JOINT_MOTOR_MAPS` key."""

LATENCY_STEPS: tuple[int, int] = (0, 2)
"""Position-target delay range in physics steps (dt 5 ms -> 0-10 ms: ESP32 + CAN round trip, unmeasured)."""
FRICTION_SCALE_RANGE: tuple[float, float] = (0.5, 1.5)
"""Per-episode scale on the Coulomb and viscous friction (unmeasured, see docs/ACTUATORS.md)."""
VELOCITY_LIMIT_SIM_FACTOR: float = 2.0
"""PhysX joint velocity cap = factor x no-load speed (safety net only; the torque-speed line does the physics)."""


def joint_hw_params(map_name: str = "user_2026_09_25") -> dict[str, dict]:
    """Per-motor parameters for ``map_name``: model fields + armature + role gains, in MOTOR_NAMES order."""
    if map_name not in JOINT_MOTOR_MAPS:
        raise ValueError(f"unknown motor map {map_name!r}; known: {sorted(JOINT_MOTOR_MAPS)}")
    jm = JOINT_MOTOR_MAPS[map_name]
    missing = [n for n in MOTOR_NAMES if n not in jm]
    if missing:
        raise ValueError(f"motor map {map_name!r} misses {missing}")
    out: dict[str, dict] = {}
    for name in MOTOR_NAMES:
        m = MOTOR_MODELS[jm[name]]
        kp, kd = HW_GAINS[JOINT_ROLES[name]]
        if kp > MOTION_MODE_LIMITS["kp_max"] or kd > MOTION_MODE_LIMITS["kd_max"]:
            raise ValueError(f"{name}: gains {kp}/{kd} outside the motion-mode range {MOTION_MODE_LIMITS}")
        out[name] = {**asdict(m), "model": m.name, "armature": m.armature, "role": JOINT_ROLES[name], "kp": kp, "kd": kd}
    return out


def hw_motor_torque(tau_pd, dq, peak, saturation, no_load_speed, coulomb, viscous, friction_vel_eps: float = 0.05):
    """The ``DatasheetMotor`` output law in plain numpy (reference for sim ports such as the Newton bridge kernel).

    ``tau_pd`` (the PD demand) is clipped to the DC-motor torque-speed envelope
    ``[sat * (-1 - v / v0), sat * (1 - v / v0)]`` capped at +-``peak`` (v clipped at the speed where the line meets
    the peak, as Isaac Lab's DCMotor). Output-side Coulomb (tanh-smoothed) and viscous friction are then subtracted.
    Returns ``(applied, clipped)``: the joint torque, and the envelope-clipped motor torque (what telemetry calls
    ``tau``)."""
    import numpy as np

    tau_pd, dq = np.asarray(tau_pd, float), np.asarray(dq, float)
    v_e = no_load_speed * (1.0 + peak / saturation)
    v = np.clip(dq, -v_e, v_e)
    top = np.minimum(saturation * (1.0 - v / no_load_speed), peak)
    bottom = np.maximum(saturation * (-1.0 - v / no_load_speed), -peak)
    clipped = np.clip(tau_pd, bottom, top)
    return clipped - (coulomb * np.tanh(dq / friction_vel_eps) + viscous * dq), clipped
