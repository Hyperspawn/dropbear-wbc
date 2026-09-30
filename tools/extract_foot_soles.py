"""Extract the foot collision geometry (convex-hull vertices in the foot body frame) from the USD.

CPU only. Needs ``pxr`` (usd-core) and ``scipy``, e.g. the team venv::

    .venv-newton/Scripts/python.exe tools/extract_foot_soles.py

For each foot body the script collects every mesh point under ``<body>/collisions`` (the collision
``Xform`` carries ``PhysicsCollisionAPI`` + ``MeshCollisionAPI(convexHull)``), expresses it in the
foot *body* (link) frame [m], and keeps the convex-hull vertices of the union. Because each mesh is
simulated as its own convex hull, the lowest point of the foot along any direction is a vertex of this
union hull, so ``min_z(R @ v + p)`` over these vertices is the exact lowest contact point of the foot in
a pose ``(p, R)``. Also reports the sole height at the authored rest pose.

Output JSON (``data/calibration/dropbear_foot_sole_hulls.json``)::

    {"schema": "dropbear-foot-sole-hulls-v1", "usd_sha256": ..., "feet": {body: {"vertices_b": [[x,y,z],...],
     "rest_pos_w": [...], "rest_quat_w_wxyz": [...], "rest_min_z_w": float, ...}}}
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
FEET = ("LL_skateboard_bearing_left_2", "RL_skateboard_bearing_left_2", "LL_basis_left_1", "RL_basis_left_1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=os.environ.get("DROPBEAR_USD", DEFAULT_USD))
    ap.add_argument("--out", type=Path, default=REPO / "data/calibration/dropbear_foot_sole_hulls.json")
    args = ap.parse_args()

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Work
    from scipy.spatial import ConvexHull

    Work.SetConcurrencyLimit(1)
    sha = hashlib.sha256(Path(args.usd).read_bytes()).hexdigest()
    stage = Usd.Stage.Open(args.usd, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    feet: dict[str, dict] = {}
    for prim in stage.Traverse():
        if prim.GetName() not in FEET or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body_w = cache.GetLocalToWorldTransform(prim)
        world_to_body = body_w.GetInverse()
        pts_b = []
        n_mesh = 0
        for sub in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not sub.IsA(UsdGeom.Mesh):
                continue
            # only meshes that are below a prim carrying the CollisionAPI
            anc, is_col = sub, False
            while anc and anc != prim:
                if anc.HasAPI(UsdPhysics.CollisionAPI):
                    is_col = True
                    break
                anc = anc.GetParent()
            if not is_col:
                continue
            m = cache.GetLocalToWorldTransform(sub) * world_to_body  # mesh -> body (row-vector convention)
            for p in UsdGeom.Mesh(sub).GetPointsAttr().Get() or []:
                q = m.Transform(Gf.Vec3d(*p))
                pts_b.append((q[0], q[1], q[2]))
            n_mesh += 1
        pts = np.asarray(pts_b, dtype=np.float64)
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        rot = body_w.ExtractRotationQuat()
        pos = np.array(body_w.ExtractTranslation())
        rmat = np.array(Gf.Matrix3d(body_w.ExtractRotationMatrix())).T  # column-vector convention
        verts_w = verts @ rmat.T + pos
        feet[prim.GetName()] = {
            "vertices_b": np.round(verts, 7).tolist(),
            "num_source_points": int(len(pts)), "num_meshes": n_mesh, "num_hull_vertices": int(len(verts)),
            "rest_pos_w": pos.tolist(),
            "rest_quat_w_wxyz": [float(rot.GetReal()), *map(float, rot.GetImaginary())],
            "rest_min_z_w": float(verts_w[:, 2].min()),
            "rest_aabb_min_w": verts_w.min(axis=0).tolist(), "rest_aabb_max_w": verts_w.max(axis=0).tolist(),
            "rest_lowest_vertex_w": verts_w[np.argmin(verts_w[:, 2])].tolist(),
            "rest_num_vertices_within_1mm_of_min": int((verts_w[:, 2] < verts_w[:, 2].min() + 1e-3).sum()),
        }
        print(f"{prim.GetName()}: meshes={n_mesh} points={len(pts)} hull_vertices={len(verts)} "
              f"rest_min_z={feet[prim.GetName()]['rest_min_z_w']:.5f} m "
              f"aabb={np.round(verts_w.min(0), 4).tolist()}..{np.round(verts_w.max(0), 4).tolist()} "
              f"verts_within_1mm_of_min={feet[prim.GetName()]['rest_num_vertices_within_1mm_of_min']}")
    missing = [f for f in FEET if f not in feet]
    if missing:
        raise SystemExit(f"foot bodies not found: {missing}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "dropbear-foot-sole-hulls-v1", "usd_path": args.usd, "usd_sha256": sha,
        "script": "tools/extract_foot_soles.py",
        "frame": "vertices_b are in the body link frame [m]; world = R(quat_w) @ v + pos_w",
        "feet": feet,
    }, indent=1))
    print(f"wrote {args.out} usd_sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
