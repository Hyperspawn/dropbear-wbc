"""Make a seamless, arbitrarily long CYCLIC motion clip from a steady segment of a settled clip (numpy only).

Why (docs/ISSUES.md #7): single-clip tracking replays one finite clip, so short clips reset mid-walk (the robot
teleports back) and no clip supports walking continuously. This tool finds a gait-cycle loop inside ``[t0, t1]``
(frames i < j with matching motor pose/velocity, root height/tilt and foot-contact state, ``j - i`` >= ``min_len_s``),
takes frames ``[i, j)`` and tiles them ``copies`` times. Copy ``k`` is the segment moved by ``G^k``, where ``G`` is the
planar rigid motion (yaw about vertical + horizontal translation) that maps the root pose at ``i`` onto the root pose at
``j`` -- so every copy starts where the previous one ended and the walk continues (a straight walk stays straight, a
slightly curving one curves). The small pose mismatch at the seam is removed by ramping the last ``blend_s`` of every
copy onto the next copy's first frame (joint positions and body positions; body orientations are left as they are).

Output: a ``dropbear-motion-npz-v1`` NPZ with every per-frame array tiled and ``meta`` extended with a ``cyclic`` block.
Run ``tools/validate_motion_npz.py <out> --write-verdicts`` afterwards (the tracking env needs the verdict).

    python tools/make_cyclic_clip.py data/motions/gmr_lafan1/lafan1_walk1_subject1_v4.npz --t0 2.5 --t1 7.5 \
        --copies 20 --out data/motions_cyclic/lafan1_walk1_cyclic.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(q: np.ndarray) -> np.ndarray:
    """(roll, pitch) of a wxyz quaternion (small-angle proxy via the body z axis in world)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    return np.stack([zx, zy], axis=-1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def find_loop(z, root: int, motor_idx: list[int], fps: float, t0: float, t1: float, min_len_s: float,
              max_len_s: float) -> tuple[int, int, float, dict]:
    q = z["joint_pos"][:, motor_idx]
    qd = z["joint_vel"][:, motor_idx]
    rz_ = z["body_pos_w"][:, root, 2]
    tl = tilt_of(z["body_quat_w"][:, root])
    c = z["contact"]
    a, b = int(t0 * fps), min(int(t1 * fps), len(q) - 1)
    lo, hi = int(min_len_s * fps), int(max_len_s * fps)
    best = (None, None, np.inf)
    for i in range(a, b):
        for j in range(i + lo, min(b, i + hi) + 1):
            if not np.array_equal(c[i], c[j]):
                continue
            d = (np.sum((q[i] - q[j]) ** 2) + 0.01 * np.sum((qd[i] - qd[j]) ** 2)
                 + 25.0 * (rz_[i] - rz_[j]) ** 2 + 4.0 * np.sum((tl[i] - tl[j]) ** 2))
            if d < best[2]:
                best = (i, j, d)
    i, j, d = best
    if i is None:
        raise SystemExit(f"no loop pair with equal contact state in [{t0}, {t1}] s")
    info = {"motor_pose_rms_rad": float(np.sqrt(np.mean((q[i] - q[j]) ** 2))),
            "motor_vel_rms_rad_s": float(np.sqrt(np.mean((qd[i] - qd[j]) ** 2))),
            "root_dz_m": float(rz_[j] - rz_[i]), "contact": [bool(v) for v in c[i]]}
    return i, j, float(d), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", type=Path)
    ap.add_argument("--t0", type=float, required=True, help="start of the steady segment [s]")
    ap.add_argument("--t1", type=float, required=True, help="end of the steady segment [s]")
    ap.add_argument("--min_len_s", type=float, default=0.8, help="shortest loop (about one gait cycle)")
    ap.add_argument("--max_len_s", type=float, default=2.6)
    ap.add_argument("--copies", type=int, default=20)
    ap.add_argument("--blend_s", type=float, default=0.12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    fps = float(z["fps"])
    T = len(z["joint_pos"])
    bn = [str(b) for b in z["body_names"]]
    jn = [str(j) for j in z["joint_names"]]
    root = bn.index("world")
    motor_idx = [jn.index(str(m)) for m in z["motor_names"]]
    i, j, d, info = find_loop(z, root, motor_idx, fps, args.t0, args.t1, args.min_len_s, args.max_len_s)
    L = j - i
    print(json.dumps({"loop_frames": [i, j], "loop_s": L / fps, "score": d, **info}))

    # planar rigid motion G: root pose at i -> root pose at j
    yaw = yaw_of(z["body_quat_w"][:, root])
    dpsi = float(np.arctan2(np.sin(yaw[j] - yaw[i]), np.cos(yaw[j] - yaw[i])))
    ci = np.array([*z["body_pos_w"][i, root, :2], 0.0])
    cj = np.array([*z["body_pos_w"][j, root, :2], 0.0])
    R = rz(dpsi)
    qz = np.array([np.cos(dpsi / 2), 0.0, 0.0, np.sin(dpsi / 2)])

    def apply_g(pos, quat, lin, ang):
        pos = np.einsum("ij,...j->...i", R, pos - ci) + cj
        quat = quat_mul(np.broadcast_to(qz, quat.shape), quat)
        lin = np.einsum("ij,...j->...i", R, lin)
        ang = np.einsum("ij,...j->...i", R, ang)
        return pos, quat, lin, ang

    seg = {k: z[k][i:j].copy() for k in z.files if z[k].ndim >= 1 and len(z[k]) == T}
    # seam blend: ramp the segment end onto G(frame i) (the next copy's first frame) in joint + body position space
    nxt_pos, _, _, _ = apply_g(z["body_pos_w"][i], z["body_quat_w"][i], z["body_lin_vel_w"][i], z["body_ang_vel_w"][i])
    W = max(1, int(round(args.blend_s * fps)))
    w = (np.arange(1, W + 1) / (W + 1.0))  # 0 -> 1 over the last W frames (before the seam)
    dq = z["joint_pos"][i] - z["joint_pos"][j]
    dp = nxt_pos - z["body_pos_w"][j]
    seg["joint_pos"][-W:] += w[:, None] * dq[None]
    seg["body_pos_w"][-W:] += w[:, None, None] * dp[None]

    parts = {k: [] for k in seg}
    pos, quat, lin, ang = seg["body_pos_w"], seg["body_quat_w"], seg["body_lin_vel_w"], seg["body_ang_vel_w"]
    for k in range(args.copies):
        for key in seg:
            if key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                continue
            parts[key].append(seg[key])
        parts["body_pos_w"].append(pos)
        parts["body_quat_w"].append(quat)
        parts["body_lin_vel_w"].append(lin)
        parts["body_ang_vel_w"].append(ang)
        pos, quat, lin, ang = apply_g(pos, quat, lin, ang)
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    # renormalize quaternions (float32 drift over copies)
    out["body_quat_w"] = out["body_quat_w"] / np.linalg.norm(out["body_quat_w"], axis=-1, keepdims=True)
    meta = json.loads(str(z["meta"]))
    meta["cyclic"] = {"tool": "tools/make_cyclic_clip.py", "source": str(args.npz).replace("\\", "/"),
                      "loop_frames": [int(i), int(j)], "loop_s": L / fps, "copies": args.copies,
                      "yaw_per_loop_deg": float(np.degrees(dpsi)),
                      "advance_per_loop_m": float(np.linalg.norm(cj[:2] - ci[:2])), "seam_blend_s": W / fps,
                      "seam_mismatch": info}
    meta["frames"] = int(len(out["joint_pos"]))
    statics = {k: z[k] for k in z.files if k not in out and k != "meta"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **statics, **out, meta=np.array(json.dumps(meta)))
    step = np.abs(np.diff(out["joint_pos"][:, motor_idx], axis=0)).max()
    print(json.dumps({"out": str(args.out), "frames": int(len(out["joint_pos"])), "duration_s": len(out["joint_pos"]) / fps,
                      "max_motor_step_rad": float(step), "yaw_per_loop_deg": float(np.degrees(dpsi)),
                      "advance_per_loop_m": float(np.linalg.norm(cj[:2] - ci[:2]))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
