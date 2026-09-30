"""Dump per-rigid-body mass properties and collision hulls of the Dropbear USD (CPU, pxr).

Used by ``tools/build_serial_mjcf.py`` to lump segment masses/inertias and to size the simple collision
geoms of the DERIVED serial model. The USD is only read (plant authority, docs/CONTRACTS.md section 0).

Run with an interpreter that has ``pxr`` (usd-core) and scipy, e.g.::

    .venv-newton/Scripts/python.exe tools/extract_usd_body_properties.py

Output JSON (``data/robot/usd_body_properties_<sha8>.json``, schema ``dropbear-usd-body-properties-v1``)::

    {"usd_sha256", "usd_path", "script", "frame": ..., "bodies": {name: {
        "path", "mass", "com_b" (3), "diag_inertia" (3), "principal_axes_wxyz" (4),
        "inertia_b" (3x3, about the COM, body axes), "rest_pos_w" (3), "rest_quat_w_wxyz" (4),
        "collision_hull_b" ([[x,y,z],...] convex-hull vertices of all collision meshes, body frame,
                            subsampled to <= --max-hull-vertices), "collision_aabb_b" ([min], [max]),
        "num_collision_meshes"}}}

All values are AS AUTHORED (no contract 0.1 fixes applied; the builder applies the minimum-inertia fix).
Units: kg, m, kg*m^2; quaternions wxyz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
DEFAULT_USD = str(_paths.usd_path())
SCHEMA = "dropbear-usd-body-properties-v1"


def default_out(sha: str) -> Path:
    return REPO / "data" / "robot" / f"usd_body_properties_{sha[:8]}.json"


def _quat_wxyz(q) -> list[float]:
    im = q.GetImaginary()
    return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=os.environ.get("DROPBEAR_USD", DEFAULT_USD))
    ap.add_argument("--out", type=Path, default=None, help="default: data/robot/usd_body_properties_<sha8>.json")
    ap.add_argument("--max-hull-vertices", type=int, default=160)
    args = ap.parse_args(argv)

    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Work
    from scipy.spatial import ConvexHull

    Work.SetConcurrencyLimit(1)
    t0 = time.time()
    sha = hashlib.sha256(Path(args.usd).read_bytes()).hexdigest()
    out = args.out or default_out(sha)
    stage = Usd.Stage.Open(args.usd, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def mat4(m) -> np.ndarray:
        """Gf.Matrix4d (row-vector convention) -> numpy 4x4 acting on column vectors."""
        return np.array(m, dtype=np.float64).T

    bodies: dict[str, dict] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        mapi = UsdPhysics.MassAPI(prim)
        mass = mapi.GetMassAttr().Get()
        com = mapi.GetCenterOfMassAttr().Get()
        diag = mapi.GetDiagonalInertiaAttr().Get()
        pax = mapi.GetPrincipalAxesAttr().Get()
        if mass is None or com is None or diag is None or pax is None:
            raise RuntimeError(f"{name}: mass properties not fully authored (mass={mass}, com={com}, diag={diag}, axes={pax})")
        body_w = mat4(cache.GetLocalToWorldTransform(prim))
        rot_w = body_w[:3, :3]
        scale = np.linalg.norm(rot_w, axis=0)
        if np.max(np.abs(scale - 1.0)) > 1e-5:
            raise RuntimeError(f"{name}: body transform has scale {scale}; unsupported")
        from scipy.spatial.transform import Rotation

        q_w = Rotation.from_matrix(rot_w).as_quat()  # xyzw
        qa = _quat_wxyz(pax)
        r_pa = Rotation.from_quat([qa[1], qa[2], qa[3], qa[0]]).as_matrix()
        inertia_b = r_pa @ np.diag(np.asarray(diag, dtype=np.float64)) @ r_pa.T
        w2b = np.linalg.inv(body_w)
        pts_b = []
        n_mesh = 0
        for sub in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not sub.IsA(UsdGeom.Mesh):
                continue
            anc, is_col = sub, False
            while anc and anc != prim:
                if anc.HasAPI(UsdPhysics.CollisionAPI):
                    is_col = True
                    break
                anc = anc.GetParent()
            if not is_col:
                continue
            pts = UsdGeom.Mesh(sub).GetPointsAttr().Get()
            if not pts:
                continue
            p = np.asarray(pts, dtype=np.float64)
            m = w2b @ mat4(cache.GetLocalToWorldTransform(sub))
            pb = p @ m[:3, :3].T + m[:3, 3]
            try:
                pb = pb[ConvexHull(pb).vertices]
            except Exception:  # degenerate (flat) mesh: keep all points
                pass
            pts_b.append(pb)
            n_mesh += 1
        hull_b: list[list[float]] = []
        aabb = None
        if pts_b:
            allp = np.concatenate(pts_b)
            try:
                allp = allp[ConvexHull(allp).vertices]
            except Exception:
                pass
            aabb = [allp.min(0).round(6).tolist(), allp.max(0).round(6).tolist()]
            if len(allp) > args.max_hull_vertices:  # deterministic farthest-point subsample
                sel = [int(np.argmax(np.linalg.norm(allp - allp.mean(0), axis=1)))]
                d = np.linalg.norm(allp - allp[sel[0]], axis=1)
                while len(sel) < args.max_hull_vertices:
                    k = int(np.argmax(d))
                    sel.append(k)
                    d = np.minimum(d, np.linalg.norm(allp - allp[k], axis=1))
                allp = allp[sel]
            hull_b = allp.round(6).tolist()
        bodies[name] = {
            "path": str(prim.GetPath()),
            "mass": float(mass),
            "com_b": [float(x) for x in com],
            "diag_inertia": [float(x) for x in diag],
            "principal_axes_wxyz": qa,
            "inertia_b": inertia_b.tolist(),
            "rest_pos_w": body_w[:3, 3].tolist(),
            "rest_quat_w_wxyz": [float(q_w[3]), float(q_w[0]), float(q_w[1]), float(q_w[2])],
            "collision_hull_b": hull_b,
            "collision_aabb_b": aabb,
            "num_collision_meshes": n_mesh,
        }
    doc = {
        "schema": SCHEMA,
        "usd_path": str(args.usd),
        "usd_sha256": sha,
        "script": "tools/extract_usd_body_properties.py",
        "frame": "all *_b quantities are in the rigid-body prim (link) frame; rest_* = authored world pose",
        "num_bodies": len(bodies),
        "total_mass": float(sum(b["mass"] for b in bodies.values())),
        "bodies": bodies,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"[extract_usd_body_properties] {len(bodies)} bodies, total mass {doc['total_mass']:.4f} kg, "
          f"usd sha {sha[:12]}, {time.time() - t0:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
