"""Decimated Dropbear GLB for the browser viewer (tools/live_web.py): one node per body, meshes in body frames.

Input: the merged per-body visual meshes cached by ``tools/live_viewer_gl.py`` (``logs/live/dropbear_visual_merged.npz``,
5.9 M triangles from the USD's ~1300 visual meshes). Each body is quadric-decimated (Open3D) to
``max(--min_tris, --ratio * n)`` triangles, then written as a GLB whose node names are the USD body names (the last path
element, which is also Isaac's body name), so the page can pose every node from the physics state.

  .venv-newton/Scripts/python.exe tools/export_dropbear_glb.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cache", type=Path, default=REPO / "logs/live/dropbear_visual_merged.npz")
    ap.add_argument("--out", type=Path, default=REPO / "site/assets/dropbear.glb")
    ap.add_argument("--ratio", type=float, default=0.06)
    ap.add_argument("--min_tris", type=int, default=1500)
    args = ap.parse_args()
    import open3d as o3d
    import trimesh

    t0 = time.perf_counter()
    with np.load(args.cache, allow_pickle=False) as d:
        labels = [str(x) for x in d["labels"]]
        counts, verts, tris, colors, has = d["counts"], d["verts"], d["tris"], d["colors"], d["has_mesh"]
    scene = trimesh.Scene(base_frame="dropbear_scene_root")  # the root body itself is named "world"
    v0 = t0i = 0
    stats = {"bodies": 0, "tris_in": 0, "tris_out": 0}
    for b, lab in enumerate(labels):
        nv, nt = int(counts[b, 0]), int(counts[b, 1])
        V, T = verts[v0:v0 + nv], tris[t0i:t0i + nt]
        v0, t0i = v0 + nv, t0i + nt
        if not has[b]:
            continue
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                      o3d.utility.Vector3iVector(T.astype(np.int32)))
        m.remove_duplicated_vertices()
        target = int(max(args.min_tris, args.ratio * nt))
        if len(m.triangles) > target:
            m = m.simplify_quadric_decimation(target_number_of_triangles=target)
        m.remove_unreferenced_vertices()
        tm = trimesh.Trimesh(np.asarray(m.vertices, dtype=np.float32), np.asarray(m.triangles), process=False)
        c = colors[b] if colors[b, 0] >= 0 else np.array([0.82, 0.82, 0.80])
        tm.visual = trimesh.visual.ColorVisuals(tm, face_colors=np.tile(np.r_[np.clip(c, 0, 1) * 255, 255], (len(tm.faces), 1)))
        name = lab.rsplit("/", 1)[-1]
        scene.add_geometry(tm, node_name=name, geom_name=name)
        stats["bodies"] += 1
        stats["tris_in"] += nt
        stats["tris_out"] += len(tm.faces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(scene.export(file_type="glb"))
    stats.update(mb=round(args.out.stat().st_size / 2**20, 1), seconds=round(time.perf_counter() - t0, 1))
    (args.out.with_suffix(".json")).write_text(json.dumps(stats), encoding="utf-8")
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
