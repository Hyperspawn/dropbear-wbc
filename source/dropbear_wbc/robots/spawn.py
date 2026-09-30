"""Spawn-time, in-memory adaptations of the Dropbear USD (the source file on P: is never modified).

Each adaptation is switchable and was found by ``tools/inspect_dropbear_articulation.py`` /
``logs/robot_task/usd_*_probe.log`` on USD SHA ``45586414...``:

* ``deactivate_prims`` -- three 18 g knee bearings (``ORPHAN_BODIES``) have colliders but **no joints**, so
  PhysX simulates them as free bodies inside the knee (not articulation links, hence not filtered by
  ``enabled_self_collisions=False``).
* ``joint_friction_override`` -- the USD authors ``physxJoint:jointFriction`` 0.5 on the four ``PG_*`` hip
  motors and 0.1 on ``*_Revolute87``. PhysX scales it by the joint constraint force, which locks the loaded
  hips. Isaac Lab 2.2's ``write_joint_friction_coefficient_to_sim`` does not apply on Isaac Sim 5.x (it edits
  a copy of ``get_dof_friction_properties()`` and never calls the setter), so the attribute is overridden on
  the stage instead.
* ``joint_axis_overrides`` -- the left-knee loop closure ``LL_Revolute121`` is authored with axis X while its
  mirror ``RL_Revolute121`` and every other knee four-bar joint use Z (same local frames), which locks the
  left knee four-bar. Overridden to Z.
* ``min_principal_inertia`` -- ``LH_bicep_1``/``RH_bicep_1`` have a zero principal inertia component, which
  PhysX rejects for articulation links ("components must be > 0"); zero/negative components are raised
  to this value [kg*m^2].
* ``spherical_joint_overrides`` -- joints retyped to ``PhysicsSphericalJoint`` (docs/CONTRACTS.md 0.2: the ankle
  crank->tie-rod closures ``*_Revolute111/112`` are authored revolute with the axis parallel to the calf motor,
  which over-constrains the parallel ankle; rod-end bearings are spherical). Only loop closures excluded from the
  articulation are accepted, so the articulation DOF count is unchanged. The revolute-only attributes
  (``physics:lowerLimit``/``upperLimit``) stay authored but are not part of the spherical schema; the spherical
  cone limits are unauthored (= no limit).
* ``passive_drive_api`` -- OPT-IN, NOT the contract plant (default off): apply ``UsdPhysics.DriveAPI("angular")``
  (stiffness 0) to every movable passive *revolute* tree joint, so that the actuator-group damping Isaac Lab writes
  actually acts. Without an authored drive PhysX ignores it (``tools/probe_passive_damping.py``,
  ``logs/review_fixes/passive_damping/``). The spherical rod-end joints ``*_Revolute115/117`` stay undamped even with
  ``rotX/rotY/rotZ`` drives (same probe), so this is not a complete damping model.
"""
from __future__ import annotations

from collections.abc import Callable

from pxr import Gf, Usd, UsdPhysics

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


def _joints_by_name(root: Usd.Prim) -> dict[str, Usd.Prim]:
    return {p.GetName(): p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}


def apply_stage_fixes(root: Usd.Prim, cfg: "DropbearUsdFileCfg") -> dict[str, list[str]]:
    """Apply the configured adaptations below ``root`` (the spawned robot prim). Returns what was changed."""
    stage = root.GetStage()
    path = root.GetPath().pathString
    changed: dict[str, list[str]] = {"deactivated": [], "friction": [], "axis": [], "inertia": [], "spherical": [],
                                     "passive_drive": []}
    for rel in cfg.deactivate_prims:
        target = stage.GetPrimAtPath(f"{path}/{rel}")
        if not target.IsValid():
            raise ValueError(f"cannot deactivate {path}/{rel}: prim not found (USD revision changed?)")
        target.SetActive(False)
        changed["deactivated"].append(rel)
    joints = _joints_by_name(root)
    if cfg.joint_friction_override is not None:
        for name, prim in joints.items():
            attr = prim.GetAttribute("physxJoint:jointFriction")
            if attr and attr.HasAuthoredValue():
                attr.Set(float(cfg.joint_friction_override))
                changed["friction"].append(name)
    for name, axis in cfg.joint_axis_overrides.items():
        if name not in joints:
            raise ValueError(f"joint {name} not found for axis override")
        UsdPhysics.RevoluteJoint(joints[name]).GetAxisAttr().Set(axis)
        changed["axis"].append(f"{name}:{axis}")
    if cfg.min_principal_inertia is not None:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                attr = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr()
                value = attr.Get()
                if value is not None and min(value) <= 0.0:
                    fixed = Gf.Vec3f(*(max(float(v), cfg.min_principal_inertia) for v in value))
                    attr.Set(fixed)
                    changed["inertia"].append(prim.GetName())
    for name in cfg.spherical_joint_overrides:
        changed["spherical"].append(retype_closure_to_spherical(joints, name))
    if cfg.passive_drive_api:
        exclude = set(cfg.passive_drive_exclude)
        for name, prim in joints.items():
            if name in exclude or prim.GetTypeName() != "PhysicsRevoluteJoint":
                continue
            if UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
                continue
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                continue
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            drive.CreateTypeAttr("force")
            changed["passive_drive"].append(name)
    return changed


def retype_closure_to_spherical(joints: dict[str, Usd.Prim], name: str) -> str:
    """Retype the loop-closure joint ``name`` to ``PhysicsSphericalJoint`` (in memory); fail closed otherwise."""
    if name not in joints:
        raise ValueError(f"joint {name} not found for the spherical retype (USD revision changed?)")
    prim = joints[name]
    joint = UsdPhysics.Joint(prim)
    if not joint.GetExcludeFromArticulationAttr().Get():
        raise ValueError(f"{name} is not a loop closure (excludeFromArticulation false): refusing to retype a tree joint")
    type_from = prim.GetTypeName()
    if type_from not in ("PhysicsRevoluteJoint", "PhysicsSphericalJoint"):
        raise ValueError(f"{name} is a {type_from}, expected PhysicsRevoluteJoint")
    prim.SetTypeName("PhysicsSphericalJoint")
    return f"{name}:{type_from}->PhysicsSphericalJoint"


@clone
def spawn_dropbear_usd(
    prim_path: str,
    cfg: "DropbearUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """``spawn_from_usd`` followed by :func:`apply_stage_fixes` on the source prim (before cloning)."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    changed = apply_stage_fixes(prim, cfg)
    print(f"[dropbear_wbc.robots.spawn] {prim_path}: {changed}", flush=True)
    return prim


@configclass
class DropbearUsdFileCfg(UsdFileCfg):
    """``UsdFileCfg`` with Dropbear stage adaptations (see module docstring)."""

    func: Callable = spawn_dropbear_usd
    deactivate_prims: tuple[str, ...] = ()
    joint_friction_override: float | None = None
    joint_axis_overrides: dict[str, str] = {}
    min_principal_inertia: float | None = None
    spherical_joint_overrides: tuple[str, ...] = ()
    passive_drive_api: bool = False
    """OPT-IN diagnostic (not the contract plant): authored drives on passive revolute tree joints (see module doc)."""
    passive_drive_exclude: tuple[str, ...] = ()
    """Joint names never given a passive drive (the motors and neck screws, which have authored drives)."""
