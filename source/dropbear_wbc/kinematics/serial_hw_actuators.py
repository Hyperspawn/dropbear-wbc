"""Hardware (``hw_*`` motor map) surrogate actuators for the DERIVED serial model, and a URDF export of it.

Why: ``tools/build_serial_mjcf.py`` maps the LEGACY ``DROPBEAR_CFG`` motor groups (``ACTUATOR_PARAMS``) through
``J = d(motor)/d(semantic)``. Against the user's motor map (``hw_motor_specs.JOINT_MOTOR_MAPS['user_2026_09_25']``) that
surrogate is 2.2-5.0x too strong on the legs (hip roll 224 vs 46 N*m, knee 167 vs 33.5 N*m peak at the stand), 1.6x
on the X8 arm joints, and its armature is 2.8-67.5x too low (hip pitch 0.010 vs 0.676 kg*m^2). Anything that takes
effort/armature from ``dropbear_serial.xml`` (the mjlab / SONIC plans in docs/SERIAL_MODEL_AND_GMR.md, a MuJoCo
sim2sim) gets a robot several times stronger and lighter-geared than the hardware.

This module recomputes the same J at ``standing_semantic_pos`` (CPU, calibration JSON only; no raw sweep, no USD) and
maps the datasheet motor model instead: effort_peak/effort_rated = sum|J| x peak/rated (an upper bound: coupled motors
cannot all saturate in one direction), velocity = the semantic speed at which the first coupled motor reaches its
no-load speed, armature = sum J^2 x rotor x N^2, kp/kd = sum J^2 x HW_GAINS (motion-mode field, output-referred).
Motor space stays the source of truth (``hw_motor_specs``); these are semantic-space surrogates.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ..robots import hw_motor_specs as hw
from ..robots.dropbear_names import ACTUATOR_GROUP_MOTORS, ACTUATOR_PARAMS, MOTOR_NAMES
from .semantic import DEFAULT_CALIBRATION, SemanticMap

REPO = Path(__file__).resolve().parents[3]
SERIAL_XML = REPO / "data/robot/dropbear_serial.xml"
SERIAL_META = REPO / "data/robot/dropbear_serial.json"


def standing_jacobian(calibration: str | Path = DEFAULT_CALIBRATION, h: float = 1e-3) -> np.ndarray:
    """J[m, s] = d(motor m)/d(semantic s) at ``standing_semantic_pos`` (central differences, same as the builder)."""
    cal = json.loads(Path(calibration).read_text())
    smap = SemanticMap(cal)
    stand = np.asarray(cal.get("standing_semantic_pos", smap.motor_to_semantic(np.asarray(cal["standing_motor_pos"]))))
    n = stand.shape[0]
    jac = np.zeros((len(MOTOR_NAMES), n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = h
        jac[:, i] = (smap.semantic_to_motor(stand + dq, clip=False) - smap.semantic_to_motor(stand - dq, clip=False)) / (2 * h)
    return jac


def legacy_surrogates(jac: np.ndarray, semantic_index: dict[str, int]) -> dict[str, dict]:
    """Reproduces ``dropbear_serial.json['surrogate_actuators']`` (legacy DROPBEAR_CFG groups)."""
    grp = {m: g for g, ms in ACTUATOR_GROUP_MOTORS.items() for m in ms}
    p = np.array([ACTUATOR_PARAMS[grp[m]] for m in MOTOR_NAMES], dtype=float)
    out = {}
    for jn, i in semantic_index.items():
        c = jac[:, i]
        out[jn] = {"effort": float(np.sum(np.abs(c) * p[:, 0])), "kp": float(np.sum(c**2 * p[:, 1])),
                   "kd": float(np.sum(c**2 * p[:, 2])), "armature": float(np.sum(c**2 * p[:, 3]))}
    return out


def hw_surrogates(jac: np.ndarray, semantic_index: dict[str, int], map_name: str = "user_2026_09_25",
                  tol: float = 1e-3) -> dict[str, dict]:
    mm = hw.JOINT_MOTOR_MAPS[map_name]
    models = [hw.MOTOR_MODELS[mm[m]] for m in MOTOR_NAMES]
    pk = np.array([x.peak_torque for x in models])
    rt = np.array([x.rated_torque for x in models])
    nl = np.array([x.no_load_speed for x in models])
    ar = np.array([x.armature for x in models])
    kp = np.array([hw.HW_GAINS[hw.JOINT_ROLES[m]][0] for m in MOTOR_NAMES])
    kd = np.array([hw.HW_GAINS[hw.JOINT_ROLES[m]][1] for m in MOTOR_NAMES])
    out = {}
    for jn, i in semantic_index.items():
        c = jac[:, i]
        nz = np.abs(c) > tol
        out[jn] = {"effort_peak": float(np.sum(np.abs(c) * pk)), "effort_rated": float(np.sum(np.abs(c) * rt)),
                   "velocity": float(np.min(nl[nz] / np.abs(c[nz]))), "armature": float(np.sum(c**2 * ar)),
                   "kp": float(np.sum(c**2 * kp)), "kd": float(np.sum(c**2 * kd)),
                   "motors": {MOTOR_NAMES[k]: mm[MOTOR_NAMES[k]] for k in np.nonzero(nz)[0]}}
    return out


def semantic_index(meta_path: str | Path = SERIAL_META) -> dict[str, int]:
    meta = json.loads(Path(meta_path).read_text())
    return {jn: v["semantic_index"] for jn, v in meta["joints"].items()}


def hw_mjcf_text(xml_text: str, sur: dict[str, dict], map_name: str) -> str:
    """Rewrite joint armature/actuatorfrcrange and the position servos of ``dropbear_serial.xml`` with ``sur``."""
    def joint(mo):
        tag, jn = mo.group(0), mo.group(1)
        if jn not in sur:
            return tag
        s = sur[jn]
        tag = re.sub(r'armature="[^"]*"', f'armature="{s["armature"]:.5f}"', tag)
        return re.sub(r'actuatorfrcrange="[^"]*"', f'actuatorfrcrange="{-s["effort_peak"]:.2f} {s["effort_peak"]:.2f}"', tag)

    def servo(mo):
        tag, jn = mo.group(0), mo.group(1)
        if jn not in sur:
            return tag
        s = sur[jn]
        tag = re.sub(r'forcerange="[^"]*"', f'forcerange="{-s["effort_peak"]:.2f} {s["effort_peak"]:.2f}"', tag)
        tag = re.sub(r'gainprm="[^"]*"', f'gainprm="{s["kp"]:.3f}"', tag)
        return re.sub(r'biasprm="[^"]*"', f'biasprm="0 {-s["kp"]:.3f} {-s["kd"]:.4f}"', tag)

    out = re.sub(r'<joint name="([^"]+)"[^>]*/>', joint, xml_text)
    out = re.sub(r'<general name="([^"]+)"[^>]*/>', servo, out)
    out = out.replace('<mujoco model="dropbear_serial">', f'<mujoco model="dropbear_serial_{map_name}">', 1)
    note = (f"<!-- HW SURROGATE VARIANT ({map_name}): written by tools/export_serial_hw.py from dropbear_serial.xml. "
            f"Joint armature, actuatorfrcrange and the position servos are mapped from hw_motor_specs through "
            f"J = dm/ds at the stand (effort = sum|J| x PEAK; kp/kd = sum J^2 x HW_GAINS). Kinematics and masses are "
            f"unchanged. Do not edit by hand. -->\n")
    return out.replace("<compiler", note + "  <compiler", 1)


def _rpy(r: np.ndarray) -> tuple[float, float, float]:
    """URDF fixed-axis roll-pitch-yaw for R = Rz(y) Ry(p) Rx(r)."""
    p = math.asin(-max(-1.0, min(1.0, r[2, 0])))
    if abs(r[2, 0]) < 1 - 1e-9:
        return math.atan2(r[2, 1], r[2, 2]), p, math.atan2(r[1, 0], r[0, 0])
    return math.atan2(-r[1, 2], r[1, 1]), p, 0.0


def urdf_text(model, sur: dict[str, dict], map_name: str, xml_sha256: str = "", mesh_prefix: str = "meshes/") -> str:
    """URDF of a compiled serial MjModel: link frame = MJCF body frame (all hinge pos are 0), lumped inertials,
    mesh visuals placed as authored (MuJoCo's mesh recentring undone), revolute limits from ``sur``."""
    import mujoco

    def quat2mat(q):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, np.asarray(q, float))
        return r.reshape(3, 3)

    def name(t, i):
        return mujoco.mj_id2name(model, t, i)

    f = lambda v: " ".join(f"{x:.9g}" for x in v)  # noqa: E731
    B, J, M = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_MESH
    lines = ['<?xml version="1.0"?>',
             f"<!-- GENERATED by tools/export_serial_hw.py from data/robot/dropbear_serial.xml (sha256 {xml_sha256[:16]}). "
             f"DERIVED, NOT CANONICAL: dropbear-semantic-v1 serial joints, closed loops collapsed, masses lumped from the "
             f"USD (total {float(model.body_mass.sum()):.3f} kg). effort = sum|J| x PEAK of hw map '{map_name}' at the "
             f"stand (upper bound), velocity = first coupled motor at no-load; armature/rated/kp/kd per joint in the "
             f"comment. Motor-space limits (hw_motor_specs) are the source of truth. Do not edit by hand. -->",
             f'<robot name="dropbear_serial_{map_name}">']
    for b in range(1, model.nbody):
        lines.append(f'  <link name="{name(B, b)}">')
        if model.body_mass[b] > 0:
            r = quat2mat(model.body_iquat[b])
            i = r @ np.diag(model.body_inertia[b]) @ r.T
            lines.append(f'    <inertial><origin xyz="{f(model.body_ipos[b])}" rpy="0 0 0"/><mass value="{model.body_mass[b]:.9g}"/>'
                         f'<inertia ixx="{i[0,0]:.9g}" ixy="{i[0,1]:.9g}" ixz="{i[0,2]:.9g}" iyy="{i[1,1]:.9g}" '
                         f'iyz="{i[1,2]:.9g}" izz="{i[2,2]:.9g}"/></inertial>')
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            rt = quat2mat(model.geom_quat[g]) @ quat2mat(model.mesh_quat[mid]).T
            pt = model.geom_pos[g] - rt @ model.mesh_pos[mid]
            lines.append(f'    <visual><origin xyz="{f(pt)}" rpy="{f(_rpy(rt))}"/><geometry><mesh '
                         f'filename="{mesh_prefix}{name(M, mid)}.stl"/></geometry></visual>')
        lines.append("  </link>")
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p == 0:
            continue
        org = f'<origin xyz="{f(model.body_pos[b])}" rpy="{f(_rpy(quat2mat(model.body_quat[b])))}"/>'
        js = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        if not js:
            lines.append(f'  <joint name="{name(B, b)}_fixed" type="fixed"><parent link="{name(B, p)}"/>'
                         f'<child link="{name(B, b)}"/>{org}</joint>')
            continue
        if len(js) != 1 or model.jnt_type[js[0]] != mujoco.mjtJoint.mjJNT_HINGE or not np.allclose(model.jnt_pos[js[0]], 0):
            raise ValueError(f"body {name(B, b)}: URDF export needs exactly one hinge at the body origin")
        j = js[0]
        jn = name(J, j)
        s = sur[jn]
        lines.append(f'  <joint name="{jn}" type="revolute"><parent link="{name(B, p)}"/><child link="{name(B, b)}"/>{org}'
                     f'<axis xyz="{f(model.jnt_axis[j])}"/><limit lower="{model.jnt_range[j, 0]:.9g}" '
                     f'upper="{model.jnt_range[j, 1]:.9g}" effort="{s["effort_peak"]:.4g}" velocity="{s["velocity"]:.4g}"/>'
                     f'<dynamics damping="{model.dof_damping[model.jnt_dofadr[j]]:.4g}" friction="0"/></joint>'
                     f'<!-- {jn}: armature {s["armature"]:.5f} kg*m^2, rated {s["effort_rated"]:.4g} N*m, kp {s["kp"]:.4g}, '
                     f'kd {s["kd"]:.4g}; motors {s["motors"]} -->')
    lines.append("</robot>")
    return "\n".join(lines) + "\n"
