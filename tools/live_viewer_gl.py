"""Lightweight live viewer of a running motor-twin simulation: Newton's OpenGL viewer, no Isaac (runs in .venv-newton).

The physics process is ``scripts/play.py ... --device cpu --realtime --state_out logs/live/state.bin`` (Isaac,
headless). It publishes every link pose of env 0 per policy step (``dropbear_wbc.isaac.state_share``); this viewer
imports the same Dropbear USD into a Newton model (visual meshes only; nothing is simulated here), writes the
published poses into ``state.body_q`` by body name and draws them. It needs ~1-2 GB of RAM where a second Isaac
process needed ~13 GB committed (on the 31.6 GB laptop, two Isaac processes + the Llama text encoder + Kimodo ran out
of commit memory, Win32 error 1455, 2026-09-26).

    .venv-newton/Scripts/python.exe tools/live_viewer_gl.py --state logs/live/state.bin

Mouse: orbit/zoom as in the Newton examples; the camera translates with the robot (``--no_follow`` to stop).
Exits when the window closes, or ``--idle_exit_s`` after the physics process stops publishing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))


def _rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return v @ r.T


def _cache_key(usd: Path) -> dict:
    st = usd.stat()
    return {"usd": str(usd), "size": st.st_size, "mtime": int(st.st_mtime), "version": 1}


def _model_from_cache(cache: Path, device: str):
    import newton

    with np.load(cache, allow_pickle=False) as d:
        labels = [str(x) for x in d["labels"]]
        xforms, counts = d["xforms"], d["counts"]
        verts, tris, colors, has = d["verts"], d["tris"], d["colors"], d["has_mesh"]
    vb = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    v0 = t0 = 0
    for b, lab in enumerate(labels):
        bi = vb.add_link(xform=tuple(float(x) for x in xforms[b]), label=lab)
        nv, nt = int(counts[b, 0]), int(counts[b, 1])
        if has[b]:
            mesh = newton.Mesh(verts[v0:v0 + nv], tris[t0:t0 + nt].reshape(-1), compute_inertia=False)
            vb.add_shape_mesh(bi, mesh=mesh, cfg=cfg, color=tuple(float(c) for c in colors[b]) if colors[b, 0] >= 0
                              else None, label=f"{lab}/merged_visual")
        v0, t0 = v0 + nv, t0 + nt
    vb.add_ground_plane()
    return vb.finalize(device=device), labels


def build_model(usd: Path, device: str, merge: bool = True, cache: Path | None = None):
    """Display model (see ``_build_from_usd``); with ``cache``, the merged meshes are read from / written to an NPZ so
    later starts skip the USD parse (~30 s and ~5 GB of transient commit)."""
    if merge and cache is not None:
        key = _cache_key(usd)
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as d:
                ok = json.loads(str(d["key"])) == key
            if ok:
                return _model_from_cache(cache, device)
        _build_from_usd(usd, device, merge=True, cache=cache)  # writes the cache, then rebuild from it (lean)
        return _model_from_cache(cache, device)
    return _build_from_usd(usd, device, merge=merge)


def _build_from_usd(usd: Path, device: str, merge: bool = True, cache: Path | None = None):
    """Newton model of the Dropbear USD for display: every body's visible meshes merged into one mesh per body.

    The USD has ~1300 visual meshes (5.9 M triangles); the viewer updates each shape instance from Python every frame
    (~120 ms per frame unmerged), so merging per body (~90 meshes) is what makes it interactive.
    """
    import newton

    from dropbear_wbc.newton_sim.plant import configure_warp_cache, load_prepare_stage

    configure_warp_cache()
    src = newton.ModelBuilder()
    stage, _ = load_prepare_stage()(usd)
    src.add_usd(stage, floating=True, enable_self_collisions=False, load_visual_shapes=True, root_path="/humanoid")
    if not merge:
        src.add_ground_plane()
        return src.finalize(device=device), list(src.body_label)
    parts: dict[int, list] = {}
    for s in range(len(src.shape_type)):
        m = src.shape_source[s]
        if not (int(src.shape_flags[s]) & int(newton.ShapeFlags.VISIBLE)) or src.shape_type[s] != newton.GeoType.MESH \
                or m is None:
            continue
        tf = np.asarray(src.shape_transform[s], dtype=np.float64)
        v = _rotate_xyzw(tf[3:7], np.asarray(m.vertices, dtype=np.float64) * np.asarray(src.shape_scale[s])) + tf[:3]
        parts.setdefault(int(src.shape_body[s]), []).append((v, np.asarray(m.indices, dtype=np.int64).reshape(-1, 3),
                                                              src.shape_color[s] if s < len(src.shape_color) else None))
    vb = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    rec = {"verts": [], "tris": [], "counts": [], "colors": [], "has_mesh": [], "xforms": []}
    for b, lab in enumerate(src.body_label):
        rec["xforms"].append(np.asarray(src.body_q[b], dtype=np.float64))
        bi = vb.add_link(xform=src.body_q[b], label=lab)
        if b not in parts:
            rec["counts"].append((0, 0))
            rec["colors"].append((-1.0, -1.0, -1.0))
            rec["has_mesh"].append(False)
            continue
        verts, tris, off = [], [], 0
        for v, t, _ in parts[b]:
            verts.append(v)
            tris.append(t + off)
            off += len(v)
        col = [c for _, _, c in parts[b] if c is not None]
        V, T = np.concatenate(verts).astype(np.float32), np.concatenate(tris).astype(np.int32)
        c = tuple(np.mean(np.asarray(col, dtype=float), axis=0)) if col else None
        rec["verts"].append(V)
        rec["tris"].append(T)
        rec["counts"].append((len(V), len(T)))
        rec["colors"].append(c if c is not None else (-1.0, -1.0, -1.0))
        rec["has_mesh"].append(True)
        if cache is None:
            vb.add_shape_mesh(bi, mesh=newton.Mesh(V, T.reshape(-1), compute_inertia=False), cfg=cfg, color=c,
                              label=f"{lab}/merged_visual")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=np.asarray(json.dumps(_cache_key(usd))), labels=np.asarray(list(src.body_label)),
                 xforms=np.stack(rec["xforms"]), counts=np.asarray(rec["counts"], dtype=np.int64),
                 verts=np.concatenate(rec["verts"]), tris=np.concatenate(rec["tris"]),
                 colors=np.asarray(rec["colors"], dtype=np.float64), has_mesh=np.asarray(rec["has_mesh"]))
        return None
    vb.add_ground_plane()
    return vb.finalize(device=device), list(vb.body_label)


def main() -> int:
    from dropbear_wbc.newton_sim.plant import DEFAULT_USD

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--state", type=Path, default=REPO / "logs/live/state.bin")
    ap.add_argument("--usd", type=Path, default=DEFAULT_USD)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--max_fps", type=float, default=30.0, help="frame cap (leaves CPU to the physics process)")
    ap.add_argument("--no_follow", action="store_true")
    ap.add_argument("--no_merge", action="store_true", help="keep the ~1300 USD meshes separate (slow)")
    ap.add_argument("--mesh_cache", type=Path, default=REPO / "logs/live/dropbear_visual_merged.npz",
                    help="merged-mesh cache (rebuilt when the USD size/mtime changes); '' to disable")
    ap.add_argument("--idle_exit_s", type=float, default=20.0, help="0 = never")
    ap.add_argument("--wait_s", type=float, default=600.0)
    ap.add_argument("--snapshot", default=None,
                    help="'<sim_s>:<png>[,<sim_s>:<png>...]': save the viewport (no UI) once the sim time passes each mark")
    args = ap.parse_args()

    import newton
    import warp as wp

    from dropbear_wbc.isaac.state_share import StateReader

    t0 = time.perf_counter()
    model, labels = build_model(args.usd, args.device, merge=not args.no_merge,
                                cache=args.mesh_cache if str(args.mesh_cache) not in ("", ".") else None)
    state = model.state()
    w, h = (int(v) for v in args.size.lower().split("x"))
    viewer = newton.viewer.ViewerGL(width=w, height=h)
    viewer.set_model(model)
    print(f"[viewer-gl] model ready in {time.perf_counter() - t0:.1f} s ({model.body_count} bodies); waiting for "
          f"{args.state}", flush=True)
    reader = StateReader(args.state if args.state.is_absolute() else REPO / args.state, wait_s=args.wait_s)
    short = {lab.rsplit("/", 1)[-1]: i for i, lab in enumerate(labels)}
    pairs = [(k, short[n]) for k, n in enumerate(reader.body_names) if n in short]
    missing = [n for n in reader.body_names if n not in short]
    if not pairs:
        raise SystemExit(f"no published body matches the Newton model (published {reader.body_names[:5]}...)")
    print(f"[viewer-gl] {len(pairs)}/{len(reader.body_names)} bodies matched"
          + (f"; unmatched {missing}" if missing else ""), flush=True)
    src_idx = np.array([k for k, _ in pairs])
    dst_idx = np.array([i for _, i in pairs])
    body_q = state.body_q.numpy()
    eye = np.array([2.4, 2.4, 0.35])  # camera offset from the robot's body centroid (looks at the centroid)
    d = -eye / np.linalg.norm(eye)
    cam_pitch, cam_yaw = float(np.degrees(np.arcsin(d[2]))), float(np.degrees(np.arctan2(d[1], d[0])))
    last_root = None
    frames, shown, t_rep, t_last_new, got_any = 0, 0, time.time(), time.time(), False
    sim_t = 0.0
    snaps = sorted((float(a), Path(b)) for a, b in (x.split(":", 1) for x in args.snapshot.split(",")))         if args.snapshot else []
    while viewer.is_running():
        t_frame = time.time()
        st = reader.read()
        if st is not None:
            got_any, t_last_new, sim_t = True, time.time(), st["sim_t"]
            q = st["body_quat_wxyz"][src_idx]
            body_q[dst_idx, :3] = st["body_pos"][src_idx]
            body_q[dst_idx, 3:] = q[:, [1, 2, 3, 0]]  # Newton transforms are (p, q_xyzw)
            state.body_q.assign(body_q)
            root = st["body_pos"][src_idx].mean(axis=0)  # body centroid: framing + follow
            if not args.no_follow:
                if last_root is None:
                    viewer.set_camera(wp.vec3(*(root + eye)), cam_pitch, cam_yaw)
                else:
                    c = viewer.camera
                    p = np.array([c.pos[0], c.pos[1], c.pos[2]]) + np.array([*(root - last_root)[:2], 0.0])
                    viewer.set_camera(wp.vec3(*p), c.pitch, c.yaw)
            last_root = root
            shown += 1
        viewer.begin_frame(sim_t)
        viewer.log_state(state)
        viewer.end_frame()
        if snaps and sim_t >= snaps[0][0]:
            from PIL import Image

            out = snaps.pop(0)[1]
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=False).numpy()).save(out)
            print(f"[viewer-gl] snapshot {out} at sim {sim_t:.2f} s", flush=True)
        frames += 1
        now = time.time()
        if now - t_rep >= 5.0:
            print(json.dumps({"viewer_gl": {"fps": round(frames / (now - t_rep), 1), "new_states": shown,
                                            "sim_t": round(sim_t, 2)}}), flush=True)
            frames, shown, t_rep = 0, 0, now
        if args.idle_exit_s > 0 and got_any and now - t_last_new > args.idle_exit_s:
            print(f"[viewer-gl] no new state for {args.idle_exit_s:.0f} s: exiting", flush=True)
            break
        if args.max_fps > 0:
            rest = 1.0 / args.max_fps - (time.time() - t_frame)
            if rest > 0:
                time.sleep(rest)
    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
