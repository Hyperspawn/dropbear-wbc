"""Dump Dropbear USD joint frames, drives and foot collision geometry (CPU only, pxr).

Run with any python that has ``pxr`` (usd-core), e.g. the team venv:

    .venv-newton/Scripts/python.exe tools/dump_usd_joint_frames.py \
        --out logs/calibrate_settle/usd_joint_frames.json

Output (JSON), all in the USD stage frame at authored rest (all joint coordinates 0), meters:
  bodies[name]   = {path, pos_w[3], quat_w_wxyz[4], mass}
  joints[name]   = {type, body0, body1, excluded, enabled, axis_local, axis_w[3] (from body0 frame),
                    anchor0_w[3], anchor1_w[3], local_pos0/1, local_rot0/1 (wxyz), lower/upper (deg or m),
                    drive{type, stiffness, damping, max_force, target_position}}
  collisions[body] = list of {prim, type, approximation, world AABB min/max} for the foot bodies.
The USD is opened read-only; nothing is written back.
"""
from __future__ import annotations

import sys

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
DEFAULT_USD = str(_paths.usd_path())


def _quat_wxyz(q) -> list[float]:
    return [float(q.GetReal()), *[float(v) for v in q.GetImaginary()]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default=os.environ.get("DROPBEAR_USD", DEFAULT_USD))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--foot-bodies", nargs="*", default=[
        "LL_skateboard_bearing_left_2", "RL_skateboard_bearing_left_2", "LL_basis_left_1", "RL_basis_left_1"])
    args = parser.parse_args()

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Work

    Work.SetConcurrencyLimit(1)
    usd_path = Path(args.usd)
    sha = hashlib.sha256(usd_path.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    bodies: dict[str, dict] = {}
    body_tf: dict[str, Gf.Matrix4d] = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = cache.GetLocalToWorldTransform(prim)
            body_tf[str(prim.GetPath())] = m
            rot = m.ExtractRotationQuat()
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            bodies[prim.GetName()] = {
                "path": str(prim.GetPath()),
                "pos_w": [float(v) for v in m.ExtractTranslation()],
                "quat_w_wxyz": _quat_wxyz(rot),
                "mass": None if mass is None else float(mass),
            }

    joints: dict[str, dict] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        lp0, lp1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
        lr0, lr1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
        entry: dict = {
            "type": prim.GetTypeName(),
            "body0": b0[0].split("/")[-1] if b0 else None,
            "body1": b1[0].split("/")[-1] if b1 else None,
            "excluded": bool(j.GetExcludeFromArticulationAttr().Get()),
            "enabled": bool(j.GetJointEnabledAttr().Get()),
            "local_pos0": [float(v) for v in lp0] if lp0 is not None else None,
            "local_pos1": [float(v) for v in lp1] if lp1 is not None else None,
            "local_rot0": [float(lr0.GetReal()), *map(float, lr0.GetImaginary())] if lr0 is not None else None,
            "local_rot1": [float(lr1.GetReal()), *map(float, lr1.GetImaginary())] if lr1 is not None else None,
        }
        axis_attr = prim.GetAttribute("physics:axis")
        axis = axis_attr.Get() if axis_attr and axis_attr.IsValid() else None
        entry["axis_local"] = str(axis) if axis is not None else None
        for key in ("physics:lowerLimit", "physics:upperLimit"):
            attr = prim.GetAttribute(key)
            val = attr.Get() if attr and attr.IsValid() else None
            entry[key.split(":")[1]] = None if val is None else float(val)
        # World frames of both joint anchors at authored rest.
        for side, bpaths, lp, lr in (("0", b0, lp0, lr0), ("1", b1, lp1, lr1)):
            if bpaths and bpaths[0] in body_tf and lp is not None:
                m = body_tf[bpaths[0]]
                pos = m.Transform(Gf.Vec3d(*lp))
                entry[f"anchor{side}_w"] = [float(v) for v in pos]
                if axis is not None and str(axis) in ("X", "Y", "Z") and lr is not None:
                    ax_local = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[str(axis)]
                    ax_joint = Gf.Rotation(Gf.Quatd(lr.GetReal(), Gf.Vec3d(*lr.GetImaginary()))).TransformDir(ax_local)
                    ax_w = m.TransformDir(ax_joint)
                    ax_w = ax_w / max(ax_w.GetLength(), 1e-12)
                    entry[f"axis{side}_w"] = [round(float(v), 6) for v in ax_w]
            elif not bpaths and lp is not None:
                entry[f"anchor{side}_w"] = [float(v) for v in lp]
        drives = {}
        for name in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, name):
                d = UsdPhysics.DriveAPI(prim, name)
                drives[name] = {
                    "type": d.GetTypeAttr().Get(),
                    "stiffness": d.GetStiffnessAttr().Get(),
                    "damping": d.GetDampingAttr().Get(),
                    "max_force": d.GetMaxForceAttr().Get(),
                    "target_position": d.GetTargetPositionAttr().Get(),
                }
        entry["drive"] = drives
        joints[prim.GetName()] = entry

    collisions: dict[str, list] = {}
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy,
                                                             UsdGeom.Tokens.guide, UsdGeom.Tokens.render])
    for fb in args.foot_bodies:
        if fb not in bodies:
            continue
        root = stage.GetPrimAtPath(bodies[fb]["path"])
        items = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                approx = None
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                items.append({
                    "prim": str(prim.GetPath()), "type": prim.GetTypeName(), "approximation": approx,
                    "aabb_min_w": [float(v) for v in box.GetMin()], "aabb_max_w": [float(v) for v in box.GetMax()],
                })
                # Lowest mesh vertex in world (exact sole height at rest) for meshes.
                if prim.IsA(UsdGeom.Mesh):
                    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    m = cache.GetLocalToWorldTransform(prim)
                    if pts:
                        zs = [m.Transform(Gf.Vec3d(*p)) for p in pts]
                        low = min(zs, key=lambda v: v[2])
                        items[-1]["lowest_vertex_w"] = [float(v) for v in low]
                        items[-1]["num_points"] = len(pts)
        collisions[fb] = items

    out = {
        "usd_path": str(usd_path), "usd_sha256": sha,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "bodies": bodies, "joints": joints, "foot_collisions": collisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} bodies={len(bodies)} joints={len(joints)} sha={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
