"""Collision geometry probe for the tabletop task (CPU, pxr).

For the authored rest pose of the contract USD it reports, in the root (``world`` body) frame:

* the front extent (max x) of every collider that is rigid with the root, binned by height, so the table edge can be
  placed in front of the torso;
* the anchor / head body poses (head camera placement);
* the hand-plate and forearm collision hulls in their BODY frames (the pusher geometry of the scripted policy), plus
  the hand body pose at rest.

Run with the team venv (usd-core + scipy)::

    .venv-newton/Scripts/python.exe tools/tabletop_geometry_probe.py \
        --out logs/tabletop/geometry_probe.json

Stdout goes to the log you redirect it to; the JSON is the machine-readable result.
"""
from __future__ import annotations

import sys

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
DEFAULT_USD = str(_paths.usd_path())
HULL_BODIES = ("LH_shoulder_ex_al_interface_1", "RH_shoulder_ex_al_interface_1", "LH_6mm_bearing__4__1",
               "RH_6mm_bearing__4__1")
POSE_BODIES = ("world", "head_5mm_ujoint_base__5__1", "head_u_joint_center__8__1", "LH_shoulder_ex_al_interface_1",
               "RH_shoulder_ex_al_interface_1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=os.environ.get("DROPBEAR_USD", DEFAULT_USD))
    ap.add_argument("--out", type=Path, default=REPO / "logs/tabletop/geometry_probe.json")
    args = ap.parse_args()

    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Work
    from scipy.spatial import ConvexHull

    Work.SetConcurrencyLimit(1)
    sha = hashlib.sha256(Path(args.usd).read_bytes()).hexdigest()
    stage = Usd.Stage.Open(args.usd, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # rigid-with-root bodies: bodies connected to "world" through fixed joints (transitively)
    fixed_edges: list[tuple[str, str]] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            j = UsdPhysics.Joint(prim)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if b0 and b1:
                fixed_edges.append((b0[0].split("/")[-1], b1[0].split("/")[-1]))
    rigid = {"world"}
    changed = True
    while changed:
        changed = False
        for a, b in fixed_edges:
            if a in rigid and b not in rigid:
                rigid.add(b)
                changed = True
            elif b in rigid and a not in rigid:
                rigid.add(a)
                changed = True

    def body_collision_points(prim):
        pts = []
        for sub in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not sub.IsA(UsdGeom.Mesh):
                continue
            anc, is_col = sub, False
            while anc and anc != prim.GetParent():
                if anc.HasAPI(UsdPhysics.CollisionAPI):
                    is_col = True
                    break
                anc = anc.GetParent()
            if not is_col:
                continue
            m = cache.GetLocalToWorldTransform(sub)
            p = UsdGeom.Mesh(sub).GetPointsAttr().Get()
            if not p:
                continue
            a = np.asarray(p, dtype=float)
            a4 = np.concatenate([a, np.ones((len(a), 1))], 1)
            mm = np.asarray(m, dtype=float)  # row-vector convention: p_w = p_l @ M
            pts.append((a4 @ mm)[:, :3])
        return np.concatenate(pts, 0) if pts else np.zeros((0, 3))

    bodies = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetName()] = prim
    out: dict = {"schema": "dropbear-tabletop-geometry-probe-v1", "usd": args.usd, "usd_sha256": sha,
                 "frame": "root (world body) frame at the authored rest pose (all joints 0), metres",
                 "rigid_with_root": sorted(rigid)}
    # front extent of the rigid torso, binned by height
    allp = []
    for name in sorted(rigid):
        if name in bodies:
            p = body_collision_points(bodies[name])
            if len(p):
                allp.append(p)
    tor = np.concatenate(allp, 0)
    bins = np.arange(0.9, 1.95, 0.05)
    front = []
    for lo in bins:
        sel = (tor[:, 2] >= lo) & (tor[:, 2] < lo + 0.05)
        front.append({"z_lo": round(float(lo), 3), "z_hi": round(float(lo + 0.05), 3),
                      "max_x": float(tor[sel, 0].max()) if sel.any() else None,
                      "min_y": float(tor[sel, 1].min()) if sel.any() else None,
                      "max_y": float(tor[sel, 1].max()) if sel.any() else None})
    out["torso_front_extent"] = front
    out["torso_aabb"] = {"min": tor.min(0).tolist(), "max": tor.max(0).tolist()}
    print("torso (rigid with root) collider AABB:", np.round(tor.min(0), 3), np.round(tor.max(0), 3))
    for f in front:
        print(f"  z {f['z_lo']:.2f}-{f['z_hi']:.2f}: max_x {f['max_x']}")
    # body poses
    poses = {}
    for name in POSE_BODIES:
        m = np.asarray(cache.GetLocalToWorldTransform(bodies[name]), dtype=float)
        rot = m[:3, :3].T  # column-vector rotation
        from scipy.spatial.transform import Rotation as R

        q = R.from_matrix(rot).as_quat()  # xyzw
        poses[name] = {"pos": m[3, :3].tolist(), "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
        print(f"pose {name}: pos {np.round(m[3, :3], 4)} quat_wxyz {np.round([q[3], q[0], q[1], q[2]], 4)}")
    out["body_poses_rest"] = poses
    # hulls in body frames
    hulls = {}
    for name in HULL_BODIES:
        prim = bodies[name]
        m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
        rot, pos = m[:3, :3].T, m[3, :3]
        pw = body_collision_points(prim)
        if not len(pw):
            hulls[name] = {"n_points": 0}
            print(f"hull {name}: NO collision mesh")
            continue
        pb = (pw - pos) @ rot  # world -> body: R^T (p - t)
        hv = pb[ConvexHull(pb).vertices]
        hulls[name] = {"n_points": int(len(pw)), "vertices_b": hv.round(6).tolist(),
                       "aabb_b": {"min": pb.min(0).tolist(), "max": pb.max(0).tolist()},
                       "aabb_w_rest": {"min": pw.min(0).tolist(), "max": pw.max(0).tolist()}}
        print(f"hull {name}: {len(hv)} vertices, body-frame AABB {np.round(pb.min(0), 4)} .. {np.round(pb.max(0), 4)}; "
              f"rest world AABB {np.round(pw.min(0), 4)} .. {np.round(pw.max(0), 4)}")
    out["hulls"] = hulls
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
