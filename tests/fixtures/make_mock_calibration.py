"""Generate tests/fixtures/mock_semantic_calibration.json -- a clearly labelled MOCK calibration.

Numbers are read from the USD joint-frame dump written by the calibrate_settle builder
(logs/calibrate_settle/usd_joint_frames.json, USD SHA 45586414...) where possible:
  * motor<->semantic signs from the motor joint axes in the world frame at USD rest,
  * semantic ranges from the USD motor limits,
  * hip centre / sole height / segment lengths from joint anchors and the foot collision AABB.
Knee/elbow four-bar ratios (3.0 semantic rad per crank rad) and the ankle 2x2 mix are INVENTED.

Run:  python tests/fixtures/make_mock_calibration.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "logs/calibrate_settle/usd_joint_frames.json"
OUT = Path(__file__).with_name("mock_semantic_calibration.json")

SEMANTIC = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
]
MOTORS = [
    "PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint", "LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81",
    "PG_right_leg_pitch", "PG_right_leg_roll", "RL_hip_joint", "RL_knee_actuator_joint", "RL_Revolute67", "RL_Revolute81",
    "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
    "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
]
# semantic -> (motor, semantic axis in world at rest). Linear scale = sign(dot(motor_axis, semantic_axis)).
DIRECT = {
    "left_hip_roll": ("PG_left_leg_pitch", (1, 0, 0)),
    "left_hip_yaw": ("PG_left_leg_roll", (0, 0, 1)),
    "left_hip_pitch": ("LL_hip_joint", (0, 1, 0)),
    "right_hip_roll": ("PG_right_leg_pitch", (1, 0, 0)),
    "right_hip_yaw": ("PG_right_leg_roll", (0, 0, 1)),
    "right_hip_pitch": ("RL_hip_joint", (0, 1, 0)),
    "left_shoulder_pitch": ("LH_yaw", (0, 1, 0)),
    "left_shoulder_roll": ("LH_pitch", (1, 0, 0)),
    "left_shoulder_yaw": ("LH_roll", (0, 0, 1)),
    "left_wrist_roll": ("LH_wrist_roll", (0, 0, -1)),  # forearm hangs along -z at rest
    "right_shoulder_pitch": ("RH_yaw", (0, 1, 0)),
    "right_shoulder_roll": ("RH_pitch", (1, 0, 0)),
    "right_shoulder_yaw": ("RH_roll", (0, 0, 1)),
    "right_wrist_roll": ("RH_wrist_roll", (0, 0, -1)),
}
FOURBAR = {"left_knee": "LL_knee_actuator_joint", "right_knee": "RL_knee_actuator_joint",
           "left_elbow": "LH_elbow_joint", "right_elbow": "RH_elbow_joint"}
FOURBAR_RATIO = 3.0  # INVENTED: flexion rad per crank rad
# G1 convention (verified with MuJoCo, logs/motion_pipeline/probe_g1_elbow_convention.log):
#   knee  : 0 = straight, positive = flexion
#   elbow : 0 = forearm ~horizontal (~75-90 deg flexed), positive = EXTENSION, straight arm ~ +pi/2.
# Dropbear USD rest (crank 0) = straight leg / straight hanging arm, so:
#   knee_sem  = +3 * crank            -> motor = knee_sem / 3
#   elbow_sem = pi/2 - 3 * crank      -> motor = (pi/2 - elbow_sem) / 3
G1_ELBOW_STRAIGHT = math.pi / 2
ANKLE_MIX = 0.8  # INVENTED: roll coupling


def main() -> None:
    dump = json.loads(DUMP.read_text(encoding="utf-8"))
    joints = dump["joints"]

    def lim(m: str) -> tuple[float, float]:
        j = joints[m]
        return math.radians(j["lowerLimit"]), math.radians(j["upperLimit"])

    def anchor(m: str) -> np.ndarray:
        return np.asarray(joints[m]["anchor0_w"], dtype=float)

    dofs: dict[str, dict] = {}
    for sem, (mot, ax) in DIRECT.items():
        s = float(np.sign(np.dot(joints[mot]["axis0_w"], ax)))
        lo, hi = lim(mot)
        r = sorted((lo / s, hi / s))
        dofs[sem] = {"map": "linear", "motors": [mot], "scale": s, "offset": 0.0, "range": [max(r[0], -math.pi), min(r[1], math.pi)]}
    for sem, mot in FOURBAR.items():
        lo, hi = lim(mot)
        if sem.endswith("knee"):
            dofs[sem] = {"map": "linear", "motors": [mot], "scale": 1.0 / FOURBAR_RATIO, "offset": 0.0,
                         "range": [lo * FOURBAR_RATIO, hi * FOURBAR_RATIO]}
        else:
            dofs[sem] = {"map": "linear", "motors": [mot], "scale": -1.0 / FOURBAR_RATIO,
                         "offset": G1_ELBOW_STRAIGHT / FOURBAR_RATIO,
                         "range": [G1_ELBOW_STRAIGHT - hi * FOURBAR_RATIO, G1_ELBOW_STRAIGHT - lo * FOURBAR_RATIO]}
    for side, pre, rs in (("left", "LL", 1.0), ("right", "RL", -1.0)):
        a, b = f"{pre}_Revolute67", f"{pre}_Revolute81"
        # motor = M @ [pitch, roll]; motor axes are -y so +pitch (toes down, about +y) is -motor.
        mat = [[-1.0, rs * ANKLE_MIX], [-1.0, -rs * ANKLE_MIX]]
        dofs[f"{side}_ankle_pitch"] = {"map": "linear2", "motors": [a, b], "partner": f"{side}_ankle_roll",
                                       "matrix": mat, "offset": [0.0, 0.0], "range": [-0.6, 0.6]}
        dofs[f"{side}_ankle_roll"] = {"map": "pair", "motors": [a, b], "partner": f"{side}_ankle_pitch",
                                      "range": [-0.25, 0.25]}

    sole_z = min(c["aabb_min_w"][2] for body in ("LL_skateboard_bearing_left_2", "RL_skateboard_bearing_left_2")
                 for c in dump["foot_collisions"][body])
    hip_l, hip_r = anchor("LL_hip_joint"), anchor("RL_hip_joint")
    hip_c = 0.5 * (hip_l + hip_r)
    knee_z = 0.5 * (anchor("LL_Revolute47")[2] + anchor("LL_Revolute57")[2])  # polycentric knee, midpoint
    ankle = anchor("LL_Revolute87")
    sh_l = anchor("LH_roll")
    elbow_l = anchor("LH_Revolute123")
    wrist_l = anchor("LH_wrist_roll")
    foot_min = dump["foot_collisions"]["LL_skateboard_bearing_left_2"][0]["aabb_min_w"]
    foot_max = dump["foot_collisions"]["LL_skateboard_bearing_left_2"][0]["aabb_max_w"]

    standing_sem = {n: 0.0 for n in SEMANTIC}
    standing_sem.update({"left_hip_pitch": -0.15, "right_hip_pitch": -0.15, "left_knee": 0.30, "right_knee": 0.30,
                         "left_ankle_pitch": -0.15, "right_ankle_pitch": -0.15,
                         "left_elbow": G1_ELBOW_STRAIGHT - 0.3, "right_elbow": G1_ELBOW_STRAIGHT - 0.3})

    def sem_to_motor(q: dict[str, float]) -> list[float]:
        out = [0.0] * 22
        for sem, spec in dofs.items():
            if spec["map"] == "linear":
                out[MOTORS.index(spec["motors"][0])] = spec["offset"] + spec["scale"] * q[sem]
            elif spec["map"] == "linear2":
                x = np.array([q[sem], q[spec["partner"]]])
                m = np.asarray(spec["offset"]) + np.asarray(spec["matrix"]) @ x
                out[MOTORS.index(spec["motors"][0])], out[MOTORS.index(spec["motors"][1])] = m
        return [float(v) for v in out]

    real = _real_schema_fields(dofs, lim)
    cal = {
        **real,
        "schema": "dropbear-semantic-calibration-v1",
        "MOCK": True,
        "mock_warning": "MOCK calibration for pipeline development only. Signs from USD axes, four-bar ratios "
                        "and ankle mix INVENTED. Never train on outputs produced with this file.",
        "usd_sha256": dump["usd_sha256"],
        "authored_ankle_tierods": False,  # mock of the CONTRACTS 0.2 default plant (robots.defaults checks it)
        "semantic_names": SEMANTIC,
        "motor_names": MOTORS,
        "dofs": {n: {**dofs[n], **real["dofs"][n]} for n in SEMANTIC},
        "semantic_zero_motor_pos": sem_to_motor({n: 0.0 for n in SEMANTIC}),
        "conventions_note": "elbow uses the real G1 convention: 0 = forearm ~horizontal, +pi/2 ~ straight arm",
        "standing_semantic_pos": [standing_sem[n] for n in SEMANTIC],
        "standing_motor_pos": sem_to_motor(standing_sem),
        "key_bodies": {"root": "world", "anchor": "torso_RMD_X10__1_Rotor_1",
                       "left_foot": "LL_skateboard_bearing_left_2", "right_foot": "RL_skateboard_bearing_left_2"},
        "rest_transforms": {
            "pelvis_in_root": {"pos": [float(v) for v in hip_c], "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                               "note": "semantic pelvis frame = midpoint of the hip-pitch joint anchors, axes = root axes"},
        },
        "segment_lengths": {
            "hip_width": float(np.linalg.norm(hip_l - hip_r)),
            "hip_to_knee": float(hip_c[2] - knee_z),
            "knee_to_ankle": float(knee_z - ankle[2]),
            "ankle_to_sole": float(ankle[2] - sole_z),
            "ankle_forward_of_hip": float(ankle[0] - hip_c[0]),
            "foot_front": float(foot_max[0] - ankle[0]),
            "foot_back": float(ankle[0] - foot_min[0]),
            "foot_half_width": float(0.5 * (foot_max[1] - foot_min[1])),
            "shoulder_to_elbow": float(np.linalg.norm(sh_l - elbow_l)),
            "elbow_to_wrist": float(np.linalg.norm(elbow_l - wrist_l)),
            "standing_hip_height": float(hip_c[2] - sole_z),
            "sole_z_in_root": float(sole_z),
        },
        "provenance": {"script": "tests/fixtures/make_mock_calibration.py", "source": str(DUMP.relative_to(ROOT)),
                       "usd_sha256": dump["usd_sha256"]},
    }
    OUT.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(cal["segment_lengths"], indent=1))


def _real_schema_fields(dofs: dict, lim) -> dict:
    """Fields read by the real ``dropbear_wbc.kinematics.semantic.SemanticMap`` (same linear/2x2 MOCK maps):
    ``motor_limits_rad``, per-DOF ``type/scale/offset/valid_range`` (``q_sem = scale * (m - offset)``) and
    tabulated ``ankle_pairs`` built from the invented 2x2 mix."""
    out_dofs: dict[str, dict] = {}
    for sem, spec in dofs.items():
        if spec["map"] == "linear":
            out_dofs[sem] = {"type": "linear", "motors": spec["motors"], "scale": 1.0 / spec["scale"],
                             "offset": spec["offset"], "valid_range": spec["range"]}
        else:
            out_dofs[sem] = {"type": "pair", "motors": spec["motors"], "valid_range": spec["range"]}
    pairs: dict[str, dict] = {}
    for side in ("left", "right"):
        spec = dofs[f"{side}_ankle_pitch"]
        a, b = spec["motors"]
        mat = np.asarray(spec["matrix"])
        inv = np.linalg.inv(mat)
        (alo, ahi), (blo, bhi) = lim(a), lim(b)
        ag, bg = np.linspace(alo, ahi, 41), np.linspace(blo, bhi, 41)
        aa, bb = np.meshgrid(ag, bg, indexing="ij")
        pr = np.einsum("ij,abj->abi", inv, np.stack([aa, bb], -1))
        prange = spec["range"]
        rrange = dofs[f"{side}_ankle_roll"]["range"]
        pgrid, rgrid = np.linspace(*prange, 41), np.linspace(*rrange, 41)
        pp, rr = np.meshgrid(pgrid, rgrid, indexing="ij")
        ab = np.einsum("ij,abj->abi", mat, np.stack([pp, rr], -1))
        valid = (ab[..., 0] >= alo) & (ab[..., 0] <= ahi) & (ab[..., 1] >= blo) & (ab[..., 1] <= bhi)
        pairs[side] = {
            "motors": [a, b], "a_grid": ag.tolist(), "b_grid": bg.tolist(),
            "pitch": pr[..., 0].tolist(), "roll": pr[..., 1].tolist(),
            "inverse": {"pitch_grid": pgrid.tolist(), "roll_grid": rgrid.tolist(),
                        "a": ab[..., 0].tolist(), "b": ab[..., 1].tolist(), "valid": valid.tolist()},
            "a_limits": [alo, ahi], "b_limits": [blo, bhi], "pitch_range": prange, "roll_range": rrange,
        }
    motor_limits = [list(lim(m)) for m in MOTORS]
    return {"dofs": out_dofs, "ankle_pairs": pairs, "motor_limits_rad": motor_limits}


if __name__ == "__main__":
    main()
