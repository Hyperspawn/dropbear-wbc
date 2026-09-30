"""CONTRACTS section 0.1 plant fixes for the Newton import (in memory; the USD on P: is never modified).

Mirrors ``dropbear_wbc.robots.spawn.apply_stage_fixes`` (the Isaac implementation, which needs
Isaac Lab to import) using the same pure-Python constants from
:mod:`dropbear_wbc.robots.dropbear_names`, so both simulators load the same plant:

* deactivate the three jointless 18 g knee bearings (``ORPHAN_BODIES``);
* zero ``physxJoint:jointFriction`` wherever it is authored (``USD_JOINT_FRICTION``);
* override the left-knee closure axis ``LL_Revolute121`` X -> Z (``JOINT_AXIS_FIXES``);
* raise zero principal inertia components to ``MIN_PRINCIPAL_INERTIA`` (the bicep links);
* CONTRACTS 0.2 (default, opt-out ``authored_ankle_tierods=True`` / ``$DROPBEAR_AUTHORED_ANKLE=1``): retype the
  ankle crank->tie-rod loop closures ``SPHERICAL_JOINT_FIXES`` revolute -> spherical. SolverMuJoCo then maps each
  to one CONNECT equality (3 translational rows) instead of the revolute's two CONNECTs (5 rows).

Call it on a stage whose edit target is the session layer (Codex's ``prepare_stage`` sets that).
"""
from __future__ import annotations

from ..robots.dropbear_names import (
    JOINT_AXIS_FIXES,
    MIN_PRINCIPAL_INERTIA,
    ORPHAN_BODIES,
    USD_JOINT_FRICTION,
    authored_ankle_requested,
    spherical_joint_fixes,
)


def find_orphan_bodies(stage, root_body: str) -> list[dict]:
    """Rigid bodies that no enabled joint references (except the articulation root)."""
    from pxr import UsdPhysics

    referenced = set()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            if not j.GetJointEnabledAttr().Get():
                continue
            referenced |= {str(t) for t in j.GetBody0Rel().GetTargets()}
            referenced |= {str(t) for t in j.GetBody1Rel().GetTargets()}
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and path not in referenced and path != root_body:
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            out.append({"path": path, "mass_kg": float(mass or 0.0)})
    return out


def apply_contract_fixes(stage, root_path: str = "/humanoid", root_body: str = "/humanoid/world",
                         authored_ankle_tierods: bool | None = None) -> dict:
    """Apply the section 0.1 fixes (and the 0.2 ankle retype unless the authored ankle is requested) below
    ``root_path``; return what changed (fails closed on mismatch)."""
    from pxr import Gf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    authored = authored_ankle_requested(authored_ankle_tierods)
    changed: dict = {"orphans_detected": find_orphan_bodies(stage, root_body), "deactivated": [], "friction": [],
                     "axis": [], "inertia": [], "spherical": [], "authored_ankle_tierods": authored}
    detected = sorted(o["path"].rsplit("/", 1)[-1] for o in changed["orphans_detected"])
    if detected != sorted(ORPHAN_BODIES):
        raise ValueError(f"jointless bodies {detected} differ from CONTRACTS 0.1 ORPHAN_BODIES {sorted(ORPHAN_BODIES)}"
                         " (USD revision changed?)")
    for name in ORPHAN_BODIES:
        prim = stage.GetPrimAtPath(f"{root_path}/{name}")
        if not prim.IsValid():
            raise ValueError(f"orphan body {root_path}/{name} not found")
        prim.SetActive(False)
        changed["deactivated"].append(name)
    joints = {p.GetName(): p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}
    for name, prim in joints.items():
        attr = prim.GetAttribute("physxJoint:jointFriction")
        if attr and attr.HasAuthoredValue():
            changed["friction"].append({"joint": name, "from": float(attr.Get()), "to": 0.0,
                                        "expected": USD_JOINT_FRICTION.get(name)})
            attr.Set(0.0)
    for name, axis in JOINT_AXIS_FIXES.items():
        if name not in joints:
            raise ValueError(f"joint {name} not found for axis override")
        attr = UsdPhysics.RevoluteJoint(joints[name]).GetAxisAttr()
        changed["axis"].append({"joint": name, "from": str(attr.Get()), "to": axis})
        attr.Set(axis)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            attr = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr()
            value = attr.Get()
            if value is not None and min(value) <= 0.0:
                fixed = Gf.Vec3f(*(max(float(v), MIN_PRINCIPAL_INERTIA) for v in value))
                changed["inertia"].append({"body": prim.GetName(), "from": list(value), "to": list(fixed)})
                attr.Set(fixed)
    for name in spherical_joint_fixes(authored):
        if name not in joints:
            raise ValueError(f"joint {name} not found for the CONTRACTS 0.2 spherical retype")
        prim = joints[name]
        if not UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            raise ValueError(f"{name} is not a loop closure: refusing to retype a tree joint")
        type_from = prim.GetTypeName()
        if type_from not in ("PhysicsRevoluteJoint", "PhysicsSphericalJoint"):
            raise ValueError(f"{name} is a {type_from}, expected PhysicsRevoluteJoint")
        prim.SetTypeName("PhysicsSphericalJoint")
        changed["spherical"].append({"joint": name, "from": type_from, "to": "PhysicsSphericalJoint"})
    return changed
