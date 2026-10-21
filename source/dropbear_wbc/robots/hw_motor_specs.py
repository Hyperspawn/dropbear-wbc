"""Real-actuator motor specs for Dropbear (pure Python: no Isaac/torch). docs/ACTUATORS.md is the readable version.

The robot (user, 2026-10-16): CEM-60 at hip pitch and knee, RMD-X10 1:7 at hip yaw and shoulder pitch, RMD-X8 Pro on
the ankles and arms, and no hip roll motor yet. Two candidate builds for the six heavy joints, one profile each:

* ``hw_cem6``: CEM-60 at hip roll, hip pitch and knee;
* ``hw_v4``: RMD-X8-120 (V4) at hip roll and hip pitch, RMD-X10-200 (V4.1) at the knee.

Both carry their motors' masses (:func:`motor_mass_offsets`). Every other profile is an older version kept so the
policies trained on it replay unchanged: ``hw_v2*`` (the CEM-60 knee at the 2026-10-14 rating, but the CAD's X10-S2
at hip pitch and an X10 1:7 at hip roll, neither on the robot) and ``hw_v1*`` (a provisional CEM-60 guess).

Motor model (:mod:`dropbear_wbc.robots.hw_actuators`), at the output shaft, SI units:

* torque-speed envelope: ``|tau| <= min(peak, saturation * (1 - |w| / no_load))`` in the driving quadrant;
* armature = rotor inertia x ratio^2;
* Coulomb friction ~ the back-drive torque, plus a small viscous term (both randomized per episode);
* PD gains inside the motion-mode ("MIT") command range kp <= 500, kd <= 5 (RMD protocol V39 p89-90, CEM V4.4 p153);
* command latency 0-10 ms (position target delayed 0..2 physics steps, per episode).

Sources: MyActuator datasheets (V3 RMD motors: ``data/robot/actuators_datasheet_v1.json``; V4 motors and the CEM-60:
the Hyperspawn web repo's ``data/actuators/datasheets``) and the user's statements of 2026-09-25, 2026-10-14 and
2026-10-16. Unconfirmed: the CEM-60's gear ratio (30:1 assumed; its drawing says 50:1), rotor inertia and friction,
and how the V4 sheets define "Moment of Inertia".
"""
from __future__ import annotations

import copy
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
        """Reflected rotor inertia rotor x ratio^2 [kg*m^2] (datasheet reading of the vendor Inertia field)."""
        return self.rotor_inertia * self.gear_ratio**2

    @property
    def armature_alt(self) -> float | None:
        """Alternative catalog reading: vendor Inertia taken as already output-reflected (= rotor x N), not rotor x N^2.
        None when unknown (CEM-60 provisional). Lower end of the armature DR until bench E4 (notes/13 T2)."""
        if self.name.startswith("EPS-CEM"):
            return None
        return self.rotor_inertia * self.gear_ratio


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
        # VENDOR (2026-10-14, docs/archive/ACTUATORS_HISTORY.md section 14): the user's 48 V / 330 W / 60 N*m / 60 rpm rating, peak
        # 75 N*m (<= 10 s) and 80 rpm peak speed from the vendor table. The torque-speed line passes the rated point
        # (60 N*m at 60 rpm) and ends at 80 rpm: saturation = 60 / (1 - 60/80) = 240 N*m, so the full 75 N*m is
        # available up to 55 rpm. Reduction 30:1 assumed (unconfirmed). Rotor inertia, Coulomb and viscous friction:
        # the series values of the provisional entry (unknown for the CEM-60).
        MotorModel("EPS-CEM-60-V2", 30, 75.0, 60.0, 80.0 * 2.0 * math.pi / 60.0, 240.0, 0.7e-4, 2.0, 0.30, 0.93,
                   "vendor"),
        # the same with the drawing's 50:1 reduction (armature 0.175 instead of 0.063 kg*m^2): sensitivity only
        MotorModel("EPS-CEM-60-V2-50to1", 50, 75.0, 60.0, 80.0 * 2.0 * math.pi / 60.0, 240.0, 0.7e-4, 2.0, 0.30, 0.93,
                   "vendor"),
        # POWER-CONSISTENT reading of the same motor (2026-10-20, notes/213): the V2 line puts the rated point at
        # 60 N*m @ 60 rpm = 377 W at the shaft, above the 330 W rating before any loss. The drawing's 60 N*m @ 40 rpm
        # (251 W) fits it. Line through 60 N*m @ 40 rpm to the 80 rpm no-load: saturation 60 / (1 - 40/80) = 120 N*m,
        # so the 75 N*m peak holds only to 30 rpm and the knee's 55 rpm p99 gets 37.5 N*m (V2: 75). Use this for any
        # sim that is meant to bound the CEM-60 from below until the vendor 48 V T-N curve or a bench sweep exists.
        MotorModel("EPS-CEM-60-V2-P330", 30, 75.0, 60.0, 80.0 * 2.0 * math.pi / 60.0, 120.0, 0.7e-4, 2.0, 0.30, 0.93,
                   "vendor-power-consistent"),
        # CANDIDATES for the hip roll / hip pitch / knee (2026-10-16, the user's MyActuator V4 datasheets in
        # hyperspawn-web data/actuators/datasheets/myactuator/x8-120details.json and x10-200-details.json):
        # RMD-X8-P20-120 (V4): 19.61:1, 43 N*m rated at 127 rpm, 120 N*m peak, 158 rpm no-load, 1.40 kg, back-drive
        # 1.5 N*m, Ø96 x 76 mm. Line through the rated point to no-load: saturation 43 / (1 - 127/158) = 219 N*m.
        # "Moment of Inertia 1.5 kg*cm^2" taken as the ROTOR inertia (the V3 "rotor x ratio" reading would give a
        # 0.08 kg*cm^2 rotor, implausible for a 96 mm motor next to the X8 Pro's 3.4); unconfirmed.
        MotorModel("RMD-X8-P20-120-V4", 19.61, 120.0, 43.0, 158.0 * 2.0 * math.pi / 60.0, 219.2, 1.5e-4, 1.5, 0.05,
                   1.40, "datasheet"),
        # RMD-X10-P20-200 (V4.1): 20:1, 60 N*m rated at 74 rpm, 170 N*m peak, 80 rpm no-load, 1.64 kg, back-drive
        # 1.0 N*m, Ø124 x 63.5 mm. The 48 V load sweep holds ~73 rpm to 84 N*m and gives 65 rpm at 124 N*m:
        # saturation 124 / (1 - 65/80) = 661 N*m (rated 60 N*m up to 72.7 rpm vs the sheet's 74). "Moment of Inertia
        # 0.003387 kg*m^2" read as rotor x ratio (V3 convention): rotor 1.69 kg*cm^2; unconfirmed.
        MotorModel("RMD-X10-P20-200-V4.1", 20, 170.0, 60.0, 80.0 * 2.0 * math.pi / 60.0, 661.0, 1.69e-4, 1.0, 0.05,
                   1.64, "datasheet"),
    )
}

MOTION_MODE_LIMITS: dict[str, float] = {"kp_max": 500.0, "kd_max": 5.0}
"""RMD motion mode (PROTO_V39 p89-90) and CEM protocol V4.4 p153: kp 0..500, kd 0..5 (12-bit fields)."""

DEFAULT_KD_SIDE: str = "output"
"""Which speed the motion-mode kd field acts on (notes/13 T5). ``"output"`` = today's twin (sim damping = field kd);
``"motor"`` = setup-software reading (sim output damping = field kd x gear_ratio). E6/B1 picks the real value; until
then DR across both is not physically meaningful. Profiles may override via ``HW_PROFILE_OPTIONS[...]["kd_side"]``."""
KD_SIDES: tuple[str, ...] = ("output", "motor")


def normalize_kd_side(kd_side: str | None = None) -> str:
    """Return a valid kd_side; None/empty -> :data:`DEFAULT_KD_SIDE`."""
    side = DEFAULT_KD_SIDE if kd_side is None or kd_side == "" else str(kd_side)
    if side not in KD_SIDES:
        raise ValueError(f"kd_side must be one of {KD_SIDES}, got {kd_side!r}")
    return side


def kd_sim_from_field(kd_field: float, gear_ratio: float, kd_side: str | None = None) -> float:
    """IdealPD / joint-output damping from a motion-mode field kd (notes/13 T5).

    Motion-mode torque at the output: if kd acts on motor speed, tau_out = kd_field * omega_m * N
    = kd_field * N * omega_out, so sim damping at the joint is ``kd_field * gear_ratio``. If kd acts on output
    speed, sim damping is ``kd_field`` unchanged (default).
    """
    side = normalize_kd_side(kd_side)
    kd = float(kd_field)
    if side == "output":
        return kd
    return kd * float(gear_ratio)


def _check_motion_mode_gains(name: str, kp: float, kd: float, *, kd_side: str | None = None) -> None:
    """Field gains must stay inside MOTION_MODE_LIMITS. Under kd_side=motor, HW_GAINS/overrides are still FIELD
    values (0..5 for kd); putting an already-output-referred (kd*N) number here is rejected by the same cap."""
    if kp > MOTION_MODE_LIMITS["kp_max"] or kd > MOTION_MODE_LIMITS["kd_max"]:
        side = normalize_kd_side(kd_side)
        hint = ""
        if side == "motor" and kd > MOTION_MODE_LIMITS["kd_max"]:
            hint = (" under kd_side='motor' HW_GAINS/overrides must still be motion-mode FIELD kd (<=5), not "
                    "output-referred (field x N) values; the sim multiplies by gear_ratio itself")
        raise ValueError(
            f"{name}: gains {kp}/{kd} outside the motion-mode range {MOTION_MODE_LIMITS}{hint}"
        )


def _with_gains(entry: dict, kp: float, kd: float, kd_side: str) -> dict:
    """Copy ``entry`` with field kp/kd and recomputed kd_sim / kd_output."""
    out = {**entry, "kp": float(kp), "kd": float(kd), "kd_side": kd_side}
    out["kd_sim"] = kd_sim_from_field(kd, out["gear_ratio"], kd_side)
    out["kd_output"] = out["kd_sim"]  # alias: IdealPD damping at the joint/output
    return out


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
# the user's map with the vendor CEM-60 numbers of 2026-10-14 (docs/archive/ACTUATORS_HISTORY.md section 14)
JOINT_MOTOR_MAPS["vendor_2026_10_14"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"{p}_knee_actuator_joint": "EPS-CEM-60-V2" for p in ("LL", "RL")},
}
JOINT_MOTOR_MAPS["vendor_2026_10_14_r50"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"{p}_knee_actuator_joint": "EPS-CEM-60-V2-50to1" for p in ("LL", "RL")},
}
# THE REAL ROBOT (user, 2026-10-16): the 4 CEM-60 bought in 2024 sit at hip pitch and knee (MyActuator had no X10-S2
# 1:35), hip yaw and shoulder pitch are the 4 X10 1:7, the 12 X8 Pro are the calves and arms. The hip roll motors
# (butt) are NOT installed yet; the two candidate builds below differ in what goes there and in the legs:
# "cem6": 6 x CEM-60 (48 V / 330 W rating) at hip roll, hip pitch and knee -> 2 more CEM-60 to buy;
JOINT_MOTOR_MAPS["cem6_2026_10_16"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"PG_{s}_leg_pitch": "EPS-CEM-60-V2" for s in ("left", "right")},
    **{f"{p}_hip_joint": "EPS-CEM-60-V2" for p in ("LL", "RL")},
    **{f"{p}_knee_actuator_joint": "EPS-CEM-60-V2" for p in ("LL", "RL")},
}
# Build A and the vendor-knee map with the power-consistent CEM-60 envelope (notes/213): the same motors, masses and
# gains; only the torque-speed line changes (saturation 240 -> 120 N*m)
JOINT_MOTOR_MAPS["cem6_p330_2026_10_20"] = {
    j: ("EPS-CEM-60-V2-P330" if m == "EPS-CEM-60-V2" else m) for j, m in JOINT_MOTOR_MAPS["cem6_2026_10_16"].items()
}
JOINT_MOTOR_MAPS["vendor_p330_2026_10_20"] = {
    j: ("EPS-CEM-60-V2-P330" if m == "EPS-CEM-60-V2" else m) for j, m in JOINT_MOTOR_MAPS["vendor_2026_10_14"].items()
}
# "v4": X8-120 (V4) at hip roll and hip pitch, X10-200 (V4.1) at the knee
JOINT_MOTOR_MAPS["v4_2026_10_16"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"PG_{s}_leg_pitch": "RMD-X8-P20-120-V4" for s in ("left", "right")},
    **{f"{p}_hip_joint": "RMD-X8-P20-120-V4" for p in ("LL", "RL")},
    **{f"{p}_knee_actuator_joint": "RMD-X10-P20-200-V4.1" for p in ("LL", "RL")},
}
# "v4_arms" (sim study, 2026-10-20): Build B + X8-120 (V4) at shoulder pitch (was X10 1:7) and shoulder roll (was
# X8 Pro) -- do stronger shoulders let the arms help balance?
JOINT_MOTOR_MAPS["v4_arms_2026_10_20"] = {
    **JOINT_MOTOR_MAPS["v4_2026_10_16"],
    **{f"{p}_{j}": "RMD-X8-P20-120-V4" for p in ("LH", "RH") for j in ("yaw", "pitch")},
}
# the CAD/USD motor components (tools/probe_usd_motor_prims.py): X10-S2 on hip roll, hip pitch AND knee
JOINT_MOTOR_MAPS["cad_usd"] = {
    **JOINT_MOTOR_MAPS["user_2026_09_25"],
    **{f"PG_{s}_leg_pitch": "RMD-X10-S2-V3-1:35" for s in ("left", "right")},
    **{f"{p}_knee_actuator_joint": "RMD-X10-S2-V3-1:35" for p in ("LL", "RL")},
}

HW_GAINS: dict[str, tuple[float, float]] = {
    # role -> (kp, kd) MOTION-MODE FIELD values (PROTO_V39 / CEM V4.4: kp<=500, kd<=5). Always field units, never
    # already-converted sim damping. Sim IdealPD damping = field kd when kd_side="output" (default, today's twin),
    # or field kd * gear_ratio when kd_side="motor" (notes/13 T5 / item 60). Do NOT put kd*N into this table.
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
    # twin v2.1: the vendor CEM-60 knee (2026-10-14); otherwise identical to the matching hw_v1* profile
    "hw_v2": "vendor_2026_10_14",
    "hw_v2i": "vendor_2026_10_14",
    "hw_v2ie": "vendor_2026_10_14",
    "hw_v2ie_r50": "vendor_2026_10_14_r50",  # hw_v2ie with the drawing's 50:1 knee reduction (sensitivity only)
    # twin v3 (2026-10-16): the real robot's legs, two candidate hip-roll/leg builds; hw_v2ie options + motor masses
    "hw_cem6": "cem6_2026_10_16",
    "hw_v4": "v4_2026_10_16",
    "hw_v4_arms": "v4_arms_2026_10_20",  # Build B + X8-120 shoulders (sim study)
    # the same two profiles with the power-consistent CEM-60 line (notes/213): replay hw_cem6 / hw_v2ie policies here
    "hw_cem6_p330": "cem6_p330_2026_10_20",
    "hw_v2ie_p330": "vendor_p330_2026_10_20",
}
HW_PROFILE_OPTIONS: dict[str, dict] = {
    "hw_v1i": {"target_interp_steps": 4},
    # elbow kp 50 -> 200: at the joint kp / G^2 = 2.2 -> 8.7 N*m/rad through the 4.8x speed-up linkage, so the forearm
    # sags ~10 deg instead of ~40 deg under its own weight and the policy needs no large mass-specific target offset
    "hw_v1ie": {"target_interp_steps": 4, "gain_overrides": {"LH_elbow_joint": (200.0, 4.0), "RH_elbow_joint": (200.0, 4.0)}},
    # armature DR (notes/13 T2-T3): log-uniform alt..datasheet on the critical set; ImplicitActuator on joints whose
    # explicit PD at 5 ms goes unstable at the alt read (calves + wrist roll). Upper-arm roll stays explicit (rho_alt<1).
    "hw_v1_armdr": {"armature_dr": True},
    "hw_v1i_armdr": {"target_interp_steps": 4, "armature_dr": True},
    "hw_v1ie_armdr": {"target_interp_steps": 4, "armature_dr": True,
                      "gain_overrides": {"LH_elbow_joint": (200.0, 4.0), "RH_elbow_joint": (200.0, 4.0)}},
}
# copies, so that changing a v2 option never changes the published v1 profiles
HW_PROFILE_OPTIONS["hw_v2i"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1i"])
HW_PROFILE_OPTIONS["hw_v2ie"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1ie"])
HW_PROFILE_OPTIONS["hw_v2ie_r50"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1ie"])
HW_PROFILE_OPTIONS["hw_cem6"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1ie"])
HW_PROFILE_OPTIONS["hw_v4"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1ie"])
HW_PROFILE_OPTIONS["hw_v4_arms"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v1ie"])
HW_PROFILE_OPTIONS["hw_cem6_p330"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_cem6"])
HW_PROFILE_OPTIONS["hw_v2ie_p330"] = copy.deepcopy(HW_PROFILE_OPTIONS["hw_v2ie"])
"""Extra ``make_hw_motor_groups`` options per profile (docs/ISSUES.md #11: 50 Hz targets stepped into a 200 Hz PD
make torque spikes; the motor-side firmware can ramp each new target over the 4 control ticks instead)."""
"""Actuator-profile name (tasks' ``set_actuator_profile``) -> :data:`JOINT_MOTOR_MAPS` key."""

CAD_MOTOR_MODEL = "RMD-X10-S2-V3-1:35"
"""The motor the CAD (plant USD 45586414) has at hip roll, hip pitch and knee (6 slots)."""
MOTOR_STATOR_BODY: dict[str, str] = {
    # body holding each heavy-joint motor's stator in the plant USD (logs/actuators/usd_motor_prims.json): the hip roll
    # stators sit in the pelvis 'world'; the hip pitch motor's stator IS the thigh body, which also holds the knee stator
    **{f"PG_{s}_leg_pitch": "world" for s in ("left", "right")},
    "LL_hip_joint": "LL_RMD_X10_S2_MIR4__3_Stator_1", "RL_hip_joint": "RL_RMD_X10_S2_MIR4__3_Stator_1",
    "LL_knee_actuator_joint": "LL_RMD_X10_S2_MIR4__3_Stator_1", "RL_knee_actuator_joint": "RL_RMD_X10_S2_MIR4__3_Stator_1",
}


ARM_MOTOR_BODY: dict[str, str] = {
    # body holding each shoulder motor in the plant USD (logs/actuators/usd_motor_prims.json): the shoulder-pitch X10 1:7
    # stators sit in the torso 'world'; each shoulder-roll X8 Pro is its own body after the shoulder-pitch rotor
    "LH_yaw": "world", "RH_yaw": "world",
    "LH_pitch": "LH_RMD_X8_Pro_MIR8_MIR1__1__1", "RH_pitch": "RH_RMD_X8_Pro_MIR8_MIR1__1__1",
}
BASE_ARM_MAP = "user_2026_09_25"
"""The arm motors the plant USD and the real robot have (X10 1:7 shoulder pitch, X8 Pro elsewhere)."""
MASS_OFFSET_PROFILES: tuple[str, ...] = ("hw_cem6", "hw_v4", "hw_v4_arms", "hw_cem6_p330")
"""Profiles whose plant mass carries the real motors (twin v3). hw_v2ie_p330 stays on the hw_v2ie plant mass."""

RATED_POWER_W: dict[str, float] = {"EPS-CEM-60-V2": 330.0, "EPS-CEM-60-V2-50to1": 330.0, "EPS-CEM-60-V2-P330": 330.0}
"""Stated rated power [W] where the vendor/user gave one (48 V). Shaft power at the modeled rated point must not
exceed it (:func:`rated_point_shaft_power`)."""


def rated_point_speed(model: MotorModel) -> float:
    """Output speed [rad/s] where the modeled torque-speed line reaches the rated torque."""
    return model.no_load_speed * (1.0 - model.rated_torque / model.saturation_effort)


def rated_point_shaft_power(model: MotorModel) -> float:
    """Shaft power [W] at the modeled rated point (rated torque x :func:`rated_point_speed`)."""
    return model.rated_torque * rated_point_speed(model)


def envelope_torque(model: MotorModel, speed: float) -> float:
    """Available output torque [N*m] at |speed| [rad/s] in the motoring quadrant (DatasheetMotor law)."""
    return max(0.0, min(model.peak_torque, model.saturation_effort * (1.0 - abs(speed) / model.no_load_speed)))


def motor_mass_offsets(profile: str) -> dict[str, float]:
    """Per-body mass change [kg] that swaps the CAD's X10-S2 (vendor 1.70 kg) for the profile's motor at the 6 heavy
    joints, the whole difference on the stator body (the CAD part masses are not separable; first-order).

    Only the twin-v3 profiles (:data:`MASS_OFFSET_PROFILES`) change the plant mass; every older profile returns ``{}`` so
    published policies replay on the plant they were trained on."""
    if profile not in MASS_OFFSET_PROFILES:
        return {}
    jm = JOINT_MOTOR_MAPS[HW_PROFILE_MAPS[profile]]
    cad = MOTOR_MODELS[CAD_MOTOR_MODEL].mass_kg
    out: dict[str, float] = {}
    for joint, body in MOTOR_STATOR_BODY.items():
        out[body] = out.get(body, 0.0) + MOTOR_MODELS[jm[joint]].mass_kg - cad
    base = JOINT_MOTOR_MAPS[BASE_ARM_MAP]
    for joint, body in ARM_MOTOR_BODY.items():  # shoulder swaps (hw_v4_arms): new motor minus the USD's motor
        out[body] = out.get(body, 0.0) + MOTOR_MODELS[jm[joint]].mass_kg - MOTOR_MODELS[base[joint]].mass_kg
    return {b: round(d, 4) for b, d in out.items() if abs(d) > 1e-9}


STOP_MODEL: dict[str, float] = {"omega": 250.0, "zeta": 0.7, "margin_deg": 10.0, "knee_stiffness": 8000.0,
                                "knee_damping": 20.0}
"""Explicit end-stop model (``--stop_model``): past a motor joint's mechanical limit the actuator adds a damped spring
torque ``k * penetration - d * velocity`` with ``k = omega^2 * armature`` and ``d = 2 * zeta * sqrt(k * armature)`` (stable
at the 200 Hz explicit step, ~1-2 deg of penetration at full load), and PhysX's own rigid limit moves ``margin_deg``
outward so that only the modeled stop acts. The knee cranks hold the body's weight through the four-bar (hundreds of
N*m when a landing drives the knee into a stop), so they get a stiffer spring (``knee_stiffness``, ~3.6 deg at 500
N*m; their effective inertia includes the reflected shank, so it stays stable). Joints without a mechanical stop
(travel >= 350 deg: shoulder and wrist rolls) are not modeled. The stop torque is then known exactly (``DatasheetMotor.stop_effort``):
what the printed stop parts carry (e.g. the knee-lock blocks on the knee drive's 300 mm extrusion)."""
KNEE_LOCK_ARM_M: float = 0.28
"""Distance from the knee motor axis to the knee-lock stop blocks (``*_knee_lock_2*`` on the 300 mm extrusion of the knee
drive; plant USD 45586414): stop torque / this = force on the blocks."""


def motor_stop_limits() -> dict[str, tuple[float, float]]:
    """Mechanical travel [rad] of each motor joint (the plant USD's limits, as pinned in the semantic calibration)."""
    import json

    from .dropbear_names import REPO_ROOT

    cal = json.loads((REPO_ROOT / "data/calibration/snapshots/dropbear_semantic_calibration_e51033d4.json")
                     .read_text(encoding="utf-8"))
    return {m: (float(lo), float(hi)) for m, (lo, hi) in zip(cal["motor_names"], cal["motor_limits_rad"])}


LEGACY_HW_ARMATURE_PROFILES: dict[str, str] = {
    "legacy_hwarm": "user_2026_09_25",
}
"""Tracking profiles that keep the LEGACY implicit groups (gains, effort limits) but put each motor's datasheet armature
(rotor x ratio^2 of this motor map) on it instead of 0.01 kg*m^2 (notes/13 T1). Not a hardware twin: no envelope,
friction or delay, and no motor map for the scoreboard; it isolates the armature effect for an A/B against ``legacy``."""


def legacy_group_armature(map_name: str = "user_2026_09_25") -> dict[str, dict[str, float]]:
    """Legacy group name -> {motor: armature of ``map_name``} for the four body-motor groups."""
    from .dropbear_names import ACTUATOR_GROUP_MOTORS

    params = joint_hw_params(map_name)
    return {g: {m: float(params[m]["armature"]) for m in motors} for g, motors in ACTUATOR_GROUP_MOTORS.items()}


# armature DR profiles share the same motor maps as their non-DR parents
HW_PROFILE_MAPS["hw_v1_armdr"] = "user_2026_09_25"
HW_PROFILE_MAPS["hw_v1i_armdr"] = "user_2026_09_25"
HW_PROFILE_MAPS["hw_v1ie_armdr"] = "user_2026_09_25"

ARMATURE_DR_CRITICAL_JOINTS: tuple[str, ...] = (
    "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",  # calves (parallel ankle)
    "LH_wrist_roll", "RH_wrist_roll",
    "LH_roll", "RH_roll",  # upper-arm roll
)
"""First joints for armature DR (notes/13 T2-T3): link_at_rest < armature_alt * 3."""

ARMATURE_DR_IMPLICIT_JOINTS: tuple[str, ...] = (
    "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",
    "LH_wrist_roll", "RH_wrist_roll",
)
"""Critical joints whose explicit PD at dt=5 ms has spectral radius >1 at the alt armature (notes/13 B). Upper-arm
roll is in the critical set but stays on DatasheetMotor (rho_alt 0.88). Alternative to implicit: physics dt 2.5 ms
(``ARMATURE_DR_SAFE_DT``) keeps DatasheetMotor on these joints too."""

ARMATURE_DR_PHYSICS_DT: float = 0.005
ARMATURE_DR_SAFE_DT: float = 0.0025
"""Halving the physics step (and doubling decimation to keep 50 Hz) also stabilises explicit PD at the alt armature."""

LATENCY_STEPS: tuple[int, int] = (0, 2)
"""Position-target delay range in physics steps (dt 5 ms -> 0-10 ms: ESP32 + CAN round trip, unmeasured)."""
FRICTION_SCALE_RANGE: tuple[float, float] = (0.5, 1.5)
"""Per-episode scale on the Coulomb and viscous friction (unmeasured, see docs/ACTUATORS.md)."""
VELOCITY_LIMIT_SIM_FACTOR: float = 2.0
"""PhysX joint velocity cap = factor x no-load speed (safety net only; the torque-speed line does the physics)."""


def joint_hw_params(map_name: str = "user_2026_09_25", *, kd_side: str | None = None) -> dict[str, dict]:
    """Per-motor parameters for ``map_name``: model fields + armature + role gains, in MOTOR_NAMES order.

    Each entry exposes ``kp`` / ``kd`` (motion-mode FIELD values from :data:`HW_GAINS`), ``kd_side``, and
    ``kd_sim`` / ``kd_output`` (what IdealPD damping gets after the notes/13 T5 conversion). Default
    ``kd_side="output"`` keeps ``kd_sim == kd`` (bit-match with pre-item-60 twins).
    """
    side = normalize_kd_side(kd_side)
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
        _check_motion_mode_gains(name, kp, kd, kd_side=side)
        base = {**asdict(m), "model": m.name, "armature": m.armature, "armature_alt": m.armature_alt,
                "role": JOINT_ROLES[name]}
        out[name] = _with_gains(base, kp, kd, side)
    return out



def link_inertia_at_rest(joint: str, datasheet_path: str | None = None) -> float:
    """Link inertia at rest [kg*m^2] from ``data/robot/actuators_datasheet_v1.json`` (notes/13)."""
    import json
    from pathlib import Path

    path = Path(datasheet_path) if datasheet_path else Path(__file__).resolve().parents[3] / "data/robot/actuators_datasheet_v1.json"
    data = json.loads(path.read_text())
    return float(data["joints"][joint]["inertia"]["link_inertia_at_rest_kgm2"])


def armature_dr_ranges(map_name: str = "user_2026_09_25",
                       joints: tuple[str, ...] | list[str] | None = None) -> dict[str, tuple[float, float]]:
    """Per-joint (alt, datasheet) armature range for log-uniform DR (notes/13 T2). Skips motors with no alt (CEM-60).
    Default ``joints`` = :data:`ARMATURE_DR_CRITICAL_JOINTS`."""
    params = joint_hw_params(map_name)
    want = list(joints) if joints is not None else list(ARMATURE_DR_CRITICAL_JOINTS)
    out: dict[str, tuple[float, float]] = {}
    for name in want:
        if name not in params:
            raise ValueError(f"unknown joint {name!r} for map {map_name!r}")
        alt, arm = params[name]["armature_alt"], float(params[name]["armature"])
        if alt is None:
            continue
        lo, hi = float(alt), arm
        if not (0.0 < lo < hi):
            raise ValueError(f"{name}: armature DR needs 0 < alt < datasheet, got {lo}, {hi}")
        out[name] = (lo, hi)
    return out


def explicit_pd_spectral_radius(kp: float, kd: float, inertia: float, dt: float = ARMATURE_DR_PHYSICS_DT) -> float:
    """Spectral radius of semi-implicit Euler explicit PD (notes/13 B). >1 => unstable."""
    import math

    if inertia <= 0.0 or dt <= 0.0:
        raise ValueError("inertia and dt must be > 0")
    a = dt / inertia
    # A = [[1 - a*kp*dt, dt*(1 - a*kd)], [-a*kp, 1 - a*kd]]
    a00 = 1.0 - a * kp * dt
    a01 = dt * (1.0 - a * kd)
    a10 = -a * kp
    a11 = 1.0 - a * kd
    # eigenvalues of 2x2
    tr, det = a00 + a11, a00 * a11 - a01 * a10
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        s = math.sqrt(disc)
        return max(abs(0.5 * (tr + s)), abs(0.5 * (tr - s)))
    # complex conjugate: |lambda| = sqrt(det) when det>0; else use modulus of complex pair
    re = 0.5 * tr
    im = 0.5 * math.sqrt(-disc)
    return math.hypot(re, im)


def joints_unstable_at_alt(map_name: str = "user_2026_09_25", dt: float = ARMATURE_DR_PHYSICS_DT,
                           joints: tuple[str, ...] | list[str] | None = None,
                           *, kd_side: str | None = None) -> list[str]:
    """Joints in ``joints`` (default critical set) whose explicit PD at ``dt`` has rho>1 at the alt armature.
    Uses ``kd_sim`` (IdealPD damping after kd_side conversion), not the raw field kd."""
    params = joint_hw_params(map_name, kd_side=kd_side)
    want = list(joints) if joints is not None else list(ARMATURE_DR_CRITICAL_JOINTS)
    bad: list[str] = []
    for name in want:
        alt = params[name]["armature_alt"]
        if alt is None:
            continue
        J = link_inertia_at_rest(name) + float(alt)
        if explicit_pd_spectral_radius(params[name]["kp"], params[name]["kd_sim"], J, dt) > 1.0:
            bad.append(name)
    return bad


def sample_armature_log_uniform(ranges: dict[str, tuple[float, float]], n_envs: int, rng=None):
    """Sample armature [n_envs, n_joints] log-uniform in each joint's (lo, hi). Joint order = ``ranges`` insertion order.

    ``rng``: ``random.Random``, a numpy Generator, or None (deterministic ``Random(0)``). Returns ``(array, names)``;
    ``array`` is a numpy ndarray when numpy is importable, else a list of rows.
    """
    import math
    import random

    if rng is None:
        rng = random.Random(0)

    def _u() -> float:
        if hasattr(rng, "uniform") and not isinstance(rng, random.Random):
            return float(rng.uniform(0.0, 1.0))
        return float(rng.random())

    names = list(ranges)
    out = []
    for _ in range(n_envs):
        row = []
        for n in names:
            lo, hi = ranges[n]
            row.append(math.exp(math.log(lo) + _u() * (math.log(hi) - math.log(lo))))
        out.append(row)
    try:
        import numpy as np
        return np.asarray(out, dtype=float), names
    except ImportError:  # pragma: no cover
        return out, names


def armature_dr_group_plan(map_name: str = "user_2026_09_25", *,
                           gain_overrides: dict[str, tuple[float, float]] | None = None,
                           critical: tuple[str, ...] | list[str] | None = None,
                           implicit: tuple[str, ...] | list[str] | None = None,
                           kd_side: str | None = None) -> dict:
    """Isaac-free plan for an armature-DR hw profile (notes/13 T2-T3).

    Returns ``explicit`` (model -> joint list for DatasheetMotor), ``implicit`` (joint -> cfg dict for ImplicitActuator),
    ``ranges`` (joint -> (alt, datasheet)), ``spawn_armature`` (geometric mean for the ImplicitActuator default).
    ImplicitActuator ``kd`` is ``kd_sim`` (IdealPD damping after kd_side conversion).
    """
    import math

    side = normalize_kd_side(kd_side)
    params = joint_hw_params(map_name, kd_side=side)
    for name, (kp, kd) in (gain_overrides or {}).items():
        if name not in params:
            raise ValueError(f"gain override for unknown motor {name!r}")
        _check_motion_mode_gains(name, float(kp), float(kd), kd_side=side)
        params[name] = _with_gains(params[name], float(kp), float(kd), side)
    crit = set(critical if critical is not None else ARMATURE_DR_CRITICAL_JOINTS)
    imp = set(implicit if implicit is not None else ARMATURE_DR_IMPLICIT_JOINTS)
    if not imp.issubset(crit):
        raise ValueError(f"implicit joints must be a subset of critical; extra={sorted(imp - crit)}")
    ranges = armature_dr_ranges(map_name, joints=sorted(crit))
    explicit: dict[str, list[str]] = {}
    implicit_cfg: dict[str, dict] = {}
    spawn: dict[str, float] = {}
    for joint, p in params.items():
        if joint in imp:
            lo, hi = ranges[joint]
            spawn_a = math.exp(0.5 * (math.log(lo) + math.log(hi)))  # geometric mean
            spawn[joint] = spawn_a
            implicit_cfg[joint] = {
                "kp": float(p["kp"]), "kd": float(p["kd_sim"]), "effort": float(p["peak_torque"]),
                "armature": spawn_a, "armature_range": (lo, hi), "model": p["model"],
                "kd_field": float(p["kd"]), "kd_side": side,
            }
        else:
            explicit.setdefault(p["model"], []).append(joint)
    return {"explicit": explicit, "implicit": implicit_cfg, "ranges": ranges, "spawn_armature": spawn,
            "critical": sorted(crit), "implicit_joints": sorted(imp), "kd_side": side}


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
