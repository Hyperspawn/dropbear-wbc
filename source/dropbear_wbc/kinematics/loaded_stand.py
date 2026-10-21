"""Loaded-stand hold pose derived from the semantic calibration (item 241, notes/219b).

The calibration's ``standing_motor_pos`` is the kinematic reference stand (knee crank ~16 deg, COM ~6.6 cm behind the
mid-sole). Held under load it puts ~80 N*m on each knee crank (1.33x the 60 N*m rating of CEM-60 / X10-200), while the
trained walkers stand with the crank at about 2-4 deg and ~8-12 N*m. :func:`derive_loaded_stand` builds the straighter
pose: both knee cranks set to ``knee_crank_rad``, then one common hip-pitch / ankle-pitch shift that keeps the sole
pitch of the reference stand and puts the whole-body COM over the mid-sole (fore-aft). Everything else (hip roll / yaw,
arms) stays at the reference stand.

:func:`static_stand_torques` is the double-stance static inverse dynamics used by the test gate: gravity only, CoP
under the COM (y split between the soles, clamped to the sole), joint torques mapped to the 22 motors through the
calibration's semantic Jacobian. It needs ``mujoco`` and the serial MJCF; the deploy runtime only reads the JSON this
module writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .semantic import MOTOR_NAMES, SEMANTIC_NAMES, SemanticMap

SCHEMA = "dropbear-loaded-stand-v1"
G = 9.81
SIDES = ("left", "right")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arr(x) -> np.ndarray:
    return np.asarray(x[0] if isinstance(x, tuple) else x, dtype=float)


class StandPlant:
    """Kinematic serial plant (free base, soles at z=0) + calibration semantic map."""

    def __init__(self, calibration: dict, xml_path: Path):
        import mujoco

        self.mj = mujoco
        self.calib = calibration
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.sm = SemanticMap(calibration)
        self.mass = float(self.m.body_mass.sum())
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in SEMANTIC_NAMES])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in SEMANTIC_NAMES])
        self.sole = {s: self.m.site(f"{s}_sole").id for s in SIDES}
        self.sole_body = {s: int(self.m.site_bodyid[self.sole[s]]) for s in SIDES}

    def set_pose(self, q_sem: np.ndarray) -> np.ndarray:
        m, d = self.m, self.d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[self.qadr] = q_sem
        d.qvel[:] = 0.0
        self.mj.mj_forward(m, d)
        d.qpos[2] -= 0.5 * (d.site_xpos[self.sole["left"]][2] + d.site_xpos[self.sole["right"]][2])
        self.mj.mj_forward(m, d)
        return d.subtree_com[0].copy()

    def sole_pitch(self, side: str = "left") -> float:
        R = self.d.site_xmat[self.sole[side]].reshape(3, 3)
        return math.atan2(-R[2, 0], R[0, 0])

    def mid_sole_x(self) -> float:
        return 0.5 * float(self.d.site_xpos[self.sole["left"]][0] + self.d.site_xpos[self.sole["right"]][0])

    def semantic_to_motor(self, q_sem: np.ndarray) -> np.ndarray:
        return _arr(self.sm.semantic_to_motor(q_sem, clip=False))

    def motor_jacobian(self, q_motor: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """J[i, j] = d semantic_i / d motor_j."""
        J = np.zeros((len(SEMANTIC_NAMES), len(MOTOR_NAMES)))
        for j in range(len(MOTOR_NAMES)):
            a, b = q_motor.copy(), q_motor.copy()
            a[j] += h
            b[j] -= h
            J[:, j] = (_arr(self.sm.motor_to_semantic(a, clip=False)) - _arr(self.sm.motor_to_semantic(b, clip=False))) / (2 * h)
        return J


def knee_semantic_for_crank(calibration: dict, side: str, crank_rad: float) -> float:
    dof = calibration["dofs"][f"{side}_knee"]
    return float(np.interp(crank_rad, np.asarray(dof["motor_grid"], float), np.asarray(dof["semantic_values"], float)))


def derive_loaded_stand(plant: StandPlant, knee_crank_rad: float | tuple[float, float]) -> np.ndarray:
    """Semantic pose: knee cranks at ``knee_crank_rad``, COM over mid-sole, sole pitch of the reference stand."""
    calib = plant.calib
    q0 = np.asarray(calib["standing_semantic_pos"], float)
    kc = (knee_crank_rad, knee_crank_rad) if np.isscalar(knee_crank_rad) else tuple(knee_crank_rad)
    iK = [SEMANTIC_NAMES.index(f"{s}_knee") for s in SIDES]
    iHP = [SEMANTIC_NAMES.index(f"{s}_hip_pitch") for s in SIDES]
    iAP = [SEMANTIC_NAMES.index(f"{s}_ankle_pitch") for s in SIDES]
    plant.set_pose(q0)
    sp0 = plant.sole_pitch()

    def pose(x):
        q = q0.copy()
        for k, s in enumerate(SIDES):
            q[iK[k]] = knee_semantic_for_crank(calib, s, float(kc[k]))
            q[iHP[k]] = q0[iHP[k]] + x[0]
            q[iAP[k]] = q0[iAP[k]] + x[1]
        return q

    def res(x):
        q = pose(x)
        com = plant.set_pose(q)
        return np.array([com[0] - plant.mid_sole_x(), plant.sole_pitch() - sp0])

    x = np.zeros(2)
    for _ in range(50):
        r = res(x)
        if np.max(np.abs(r)) < 1e-7:
            break
        Jn = np.zeros((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += 1e-5
            Jn[:, j] = (res(xp) - r) / 1e-5
        x -= np.linalg.lstsq(Jn, r, rcond=None)[0]
    if np.max(np.abs(res(x))) > 1e-5:
        raise RuntimeError(f"loaded stand did not converge (residual {res(x)})")
    return pose(x)


def static_stand_torques(plant: StandPlant, q_sem: np.ndarray, cop_dx: float = 0.0, cop_y: str = "sole") -> np.ndarray:
    """Motor-side (22, SDK order) static double-stance torques [N*m] at ``q_sem`` for the plant mass.

    Load shares between the feet follow the COM y. Per-foot CoP x = COM x + ``cop_dx`` (clamped to +-8 cm of the sole
    centre). ``cop_y="sole"``: each foot's CoP at its own sole centre line (the shares already put the net CoP under
    the COM; least ankle-roll load). ``cop_y="com"``: notes/31 convention, CoP y = COM y clamped to +-6 cm of each
    sole, which loads both feet on their inner edge when the stance is wider than 12 cm.
    """
    mj, m, d = plant.mj, plant.m, plant.d
    com = plant.set_pose(q_sem)
    d.qacc[:] = 0.0
    mj.mj_inverse(m, d)
    tau = d.qfrc_inverse.copy()
    F = plant.mass * np.array([0.0, 0.0, G])
    pl, pr = d.site_xpos[plant.sole["left"]].copy(), d.site_xpos[plant.sole["right"]].copy()
    share_l = float(np.clip((com[1] - pr[1]) / (pl[1] - pr[1]), 0.05, 0.95))
    for side, share in (("left", share_l), ("right", 1.0 - share_l)):
        p = d.site_xpos[plant.sole[side]].copy()
        cop = np.array([np.clip(com[0] + cop_dx, p[0] - 0.08, p[0] + 0.08), (p[1] if cop_y == "sole" else np.clip(com[1], p[1] - 0.06, p[1] + 0.06)), p[2]])
        jp = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jp, None, cop, plant.sole_body[side])
        tau -= jp.T @ (share * F)
    tau_sem = tau[plant.dadr]
    J = plant.motor_jacobian(plant.semantic_to_motor(q_sem))
    return J.T @ tau_sem


def build_record(plant: StandPlant, calibration_path: Path, xml_path: Path, knee_crank_rad: float) -> dict:
    q_sem = derive_loaded_stand(plant, knee_crank_rad)
    q_mot = plant.semantic_to_motor(q_sem)
    tau = static_stand_torques(plant, q_sem)
    tau0 = static_stand_torques(plant, np.asarray(plant.calib["standing_semantic_pos"], float))
    return {
        "schema": SCHEMA,
        "calibration": Path(calibration_path).name,
        "calibration_sha256": file_sha256(calibration_path),
        "plant_xml": Path(xml_path).name,
        "plant_mass_kg": round(plant.mass, 4),
        "knee_crank_rad": float(knee_crank_rad),
        "method": "knee cranks fixed; common hip-pitch/ankle-pitch shift keeps the reference sole pitch and puts the "
                  "COM x over the mid-sole (notes/219b)",
        "loaded_standing_semantic_pos": [round(float(v), 9) for v in q_sem],
        "loaded_standing_motor_pos": [round(float(v), 9) for v in q_mot],
        "static_motor_torque_Nm": {n: round(float(t), 3) for n, t in zip(MOTOR_NAMES, tau)},
        "reference_static_motor_torque_Nm": {n: round(float(t), 3) for n, t in zip(MOTOR_NAMES, tau0)},
    }


def load_record(path: Path) -> dict:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {rec.get('schema')!r}, expected {SCHEMA!r}")
    return rec
