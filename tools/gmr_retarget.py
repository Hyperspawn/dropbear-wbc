"""Headless GMR retarget of a human BVH (LAFAN1) clip to a robot (``dropbear`` serial model or ``unitree_g1``).

Runs in ``.venv-gmr`` (GMR from ``$DROPBEAR_UPSTREAM/GMR-dropbear`` patched by ``tools/setup_gmr.py``)::

    .venv-gmr/Scripts/python.exe tools/gmr_retarget.py \
        --bvh $DROPBEAR_UPSTREAM/lafan1/bvh/walk1_subject1.bvh --robot dropbear --start 0 --end 20 \
        --out logs/serial_mjcf_gmr/gmr/walk1_subject1_dropbear.npz

Output NPZ (``gmr-retarget-npz-v1``): ``fps``, ``qpos`` (T, nq; free root = pelvis pos + quat wxyz, then hinges in
MuJoCo qpos order), ``qpos_joint_names`` (nq-7,), ``robot``, ``xml``, ``ik_config``, ``bvh``, ``frame_range``,
``task_bodies``, ``task_pos_err`` / ``task_rot_err`` (T, n_tasks; table-2 FrameTask errors after the solve, m / rad),
``human_pos`` (T, n_tasks, 3; scaled + offset human targets), ``solve_ms`` (T,), ``meta`` (JSON).
LAFAN1 is CC BY-NC-ND 4.0 (internal R&D only; outputs are Adapted Material: never redistribute).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh", type=Path, required=True)
    ap.add_argument("--format", default="lafan1", choices=("lafan1", "nokov"))
    ap.add_argument("--robot", default="dropbear")
    ap.add_argument("--fps", type=float, default=30.0, help="source frame rate (LAFAN1: 30)")
    ap.add_argument("--start", type=float, default=0.0, help="start time [s]")
    ap.add_argument("--end", type=float, default=None, help="end time [s] (default: clip end)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warm-start", type=int, default=50, help="IK calls on frame 0 before recording (0 = GMR default)")
    ap.add_argument("--preroll", type=float, default=0.0,
                    help="s: solve this much of the clip BEFORE --start (not recorded) so the recorded window starts from a "
                         "tracked configuration. GMR's G1 warm start at dance2 8.0 s converged to a twisted local minimum "
                         "(waist yaw 80 deg, Spine2 error 135 deg); from 7.0 s it tracks (review fix 2026-09-24)")
    ap.add_argument("--no-motor-limits", action="store_true",
                    help="dropbear only: do NOT add the linearised motor-limit constraint (DropbearMotorLimit)")
    ap.add_argument("--calibration", type=Path, default=REPO / "data/calibration/dropbear_semantic_calibration.json")
    ap.add_argument("--no-comfort-limits", action="store_true",
                    help="dropbear only: do NOT bound the shoulders to G1's anatomical ranges during IK (keeps the YXZ "
                         "shoulder on its anatomical branch; retargeting-level bound, not a model limit)")
    ap.add_argument("--arm-mode", choices=("direction", "orientation"), default="direction",
                    help="dropbear only. 'direction' (default): arms follow the human upper-arm and forearm DIRECTIONS "
                         "(elbow/wrist position targets rebuilt with Dropbear's segment lengths, weak orientation terms); "
                         "'orientation': the template's bone-orientation tasks (sensitive to the human upper-arm twist)")
    ap.add_argument("--no-arm-seed", action="store_true",
                    help="direction mode: do NOT re-seed the arm joints analytically each frame (dropbear_wbc.motion.gmr_arm)")
    ap.add_argument("--arm-seed-max-step", type=float, default=0.35,
                    help="rad: skip the analytic re-seed of an arm when it would move a joint more than this in one "
                         "frame (near the 90-deg-abduction gimbal the seed swings pitch/yaw; let the IK move smoothly)")
    ap.add_argument("--arm-rescue-err", type=float, default=0.12,
                    help="m: accept the analytic arm seed regardless of --arm-seed-max-step when that arm's elbow/wrist "
                         "position error in the previous frame exceeded this (IK stuck at a limit corner)")
    ap.add_argument("--arm-pos-cost", type=float, default=20.0)
    ap.add_argument("--arm-anchor", choices=("robot", "human"), default="robot",
                    help="direction mode: start the rebuilt elbow/wrist targets at the ROBOT shoulder (default; pelvis "
                         "target + previous pelvis orientation) or at the scaled HUMAN shoulder (before 2026-09-24: "
                         "Dropbear's shoulders are ~5.5 cm wider per side, which pulled the arms 7-14 deg off the human "
                         "direction)")
    ap.add_argument("--arm-rot-cost", type=float, default=1.0)
    args = ap.parse_args(argv)

    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    t0 = time.time()
    frames, human_height = load_bvh_file(str(args.bvh), format=args.format)
    a = int(round(args.start * args.fps))
    b = len(frames) if args.end is None else min(len(frames), int(round(args.end * args.fps)))
    a0 = max(0, a - int(round(args.preroll * args.fps)))
    preroll_frames = frames[a0:a]
    frames = frames[a:b]
    if len(frames) < 2:
        raise SystemExit(f"empty frame range {a}:{b}")
    gmr = GMR(src_human=f"bvh_{args.format}", tgt_robot=args.robot, actual_human_height=human_height, verbose=False)
    m = gmr.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise SystemExit("robot root is not a free joint")
    hinge_names = []
    for j in range(1, m.njnt):
        if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise SystemExit(f"unsupported joint type for {names[j]}")
        hinge_names.append(names[j])
    motor_limit = None
    if args.robot == "dropbear" and not args.no_motor_limits:
        sys.path.insert(0, str(REPO / "source"))
        from dropbear_wbc.kinematics.semantic import SemanticMap
        from dropbear_wbc.motion.gmr_limits import DropbearMotorLimit

        motor_limit = DropbearMotorLimit(m, SemanticMap.load(args.calibration))
        gmr.ik_limits.append(motor_limit)
    comfort = None
    if args.robot == "dropbear" and not args.no_comfort_limits:
        sys.path.insert(0, str(REPO / "source"))
        from dropbear_wbc.motion.gmr_limits import comfort_limit

        comfort = comfort_limit(m)
        gmr.ik_limits.append(comfort)
    fallback_frames: list[int] = []

    def solve(fr, idx):
        try:
            return gmr.retarget(fr)
        except Exception as exc:  # QP infeasible with the linearised motor bounds: solve this frame without them
            if motor_limit is None or motor_limit not in gmr.ik_limits:
                raise
            gmr.ik_limits.remove(motor_limit)
            try:
                return gmr.retarget(fr)
            finally:
                gmr.ik_limits.append(motor_limit)
                fallback_frames.append(idx)
                if len(fallback_frames) <= 3:
                    print(f"[gmr_retarget] frame {idx}: {type(exc).__name__} with motor limits -> solved without them")

    # Dropbear arm mode "direction": the forearm is 0.105 m (human ~0.25 m), so bone ORIENTATION targets (whose
    # upper-arm twist is noisy / offset in LAFAN) and raw hand POSITIONS are both poor drivers. Rebuild the elbow and
    # wrist targets along the human upper-arm and forearm directions with Dropbear's segment lengths (divided by the
    # arm scale, so they come out right after GMR's root-relative scaling) and drive them by position.
    arm_info = None
    if args.robot == "dropbear" and args.arm_mode == "direction":
        import mujoco as _mj
        d0 = _mj.MjData(m)
        d0.qpos[:] = 0.0
        d0.qpos[3] = 1.0
        _mj.mj_kinematics(m, d0)
        bp = lambda n: d0.xpos[_mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, n)].copy()  # noqa: E731
        arm_info = {}
        for side, (sh, el, ha) in {"left": ("LeftArm", "LeftForeArm", "LeftHand"),
                                    "right": ("RightArm", "RightForeArm", "RightHand")}.items():
            # shoulder centre = shoulder_yaw_link origin (moves < 6 mm with the shoulder angles; the pitch-link origin
            # is 6.4 cm medial of it)
            l_ua = float(np.linalg.norm(bp(f"{side}_elbow_link") - bp(f"{side}_shoulder_yaw_link")))
            l_fa = float(np.linalg.norm(bp(f"{side}_wrist_roll_link") - bp(f"{side}_elbow_link")))
            s_arm = float(gmr.human_scale_table[el])
            if abs(gmr.human_scale_table[ha] - s_arm) > 1e-9 or abs(gmr.human_scale_table[sh] - s_arm) > 1e-9:
                raise SystemExit("direction arm mode needs one scale for shoulder/elbow/hand")
            arm_info[side] = {"bones": [sh, el, ha], "upper_arm_m": l_ua, "forearm_m": l_fa, "scale": s_arm,
                              "shoulder_in_pelvis": (bp(f"{side}_shoulder_yaw_link") - bp("pelvis")).tolist(),
                              "anchor": args.arm_anchor}
        for tasks_by_body in (gmr.human_body_to_task1, gmr.human_body_to_task2):
            for hb, t in tasks_by_body.items():
                if hb in ("LeftArm", "RightArm"):
                    t.set_position_cost(0.0)
                    t.set_orientation_cost(args.arm_rot_cost)
                elif hb in ("LeftForeArm", "RightForeArm", "LeftHand", "RightHand"):
                    t.set_position_cost(args.arm_pos_cost)
                    t.set_orientation_cost(args.arm_rot_cost)

    seeder = None
    if arm_info is not None and not args.no_arm_seed:
        from dropbear_wbc.motion.gmr_arm import arm_seed
        from dropbear_wbc.motion.gmr_limits import G1_SHOULDER_COMFORT

        seeder = {}
        for side in ("left", "right"):
            jn = [f"{side}_shoulder_pitch", f"{side}_shoulder_roll", f"{side}_shoulder_yaw", f"{side}_elbow"]
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
            rng = np.array([m.jnt_range[j] for j in jid], dtype=np.float64)
            comfort_rng = np.array([[max(rng[k, 0], G1_SHOULDER_COMFORT[n][0]), min(rng[k, 1], G1_SHOULDER_COMFORT[n][1])]
                                    for k, n in enumerate(jn[:3])])
            seeder[side] = (np.array([m.jnt_qposadr[j] for j in jid]), rng, comfort_rng)

    last_dirs: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # human upper-arm / forearm directions of the frame
    seed_skipped = {"left": 0, "right": 0}
    seed_rescued = {"left": 0, "right": 0}
    arm_err = {"left": 0.0, "right": 0.0}

    def seed_arms(fr, first=False):
        if seeder is None:
            return
        q = gmr.configuration.q.copy()
        w, x, y, z = q[3:7] / np.linalg.norm(q[3:7])
        r_pel = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        for side, (qadr, rng, comfort_rng) in seeder.items():
            sh, el, ha = arm_info[side]["bones"]
            s_pos, e_pos, h_pos = (np.asarray(fr[b][0], dtype=np.float64) for b in (sh, el, ha))
            hard = rng.copy()
            if comfort is not None:  # IK bounded to the comfort box: seed inside it too
                hard[:3] = comfort_rng
            u_dir, f_dir = last_dirs.get(side, (e_pos - s_pos, h_pos - e_pos))  # the HUMAN directions (set by prep)
            sd = arm_seed(r_pel.T @ u_dir, r_pel.T @ f_dir, q[qadr], hard, comfort_rng, first)
            if first or np.abs(sd - q[qadr]).max() <= args.arm_seed_max_step:
                q[qadr] = sd
            elif arm_err[side] > args.arm_rescue_err:
                q[qadr] = sd
                seed_rescued[side] += 1
            else:
                seed_skipped[side] += 1
        gmr.configuration.update(q)

    def gmr_transform(fr):
        """GMR's own raw -> target transform (scale, offsets, ground offset) on a copy of the frame."""
        hd = {k: [np.array(v[0], dtype=np.float64), np.array(v[1], dtype=np.float64)] for k, v in fr.items()}
        hd = gmr.scale_human_data(hd, gmr.human_root_name, gmr.human_scale_table)
        hd = gmr.offset_human_data(hd, gmr.pos_offsets1, gmr.rot_offsets1)
        return gmr.apply_ground_offset(hd)

    def prep(fr):
        if arm_info is None:
            return fr
        fr = dict(fr)
        tr = gmr_transform(fr) if args.arm_anchor == "robot" else None
        if tr is not None:
            qc = gmr.configuration.q
            w, x, y, z = qc[3:7] / np.linalg.norm(qc[3:7])
            r_pel = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                              [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                              [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            pel = np.asarray(tr[gmr.human_root_name][0], dtype=np.float64)
        for side, info in arm_info.items():
            sh, el, ha = info["bones"]
            s_pos, e_pos, h_pos = (np.asarray(fr[b][0], dtype=np.float64) for b in (sh, el, ha))
            u = (e_pos - s_pos) / max(np.linalg.norm(e_pos - s_pos), 1e-9)
            f = (h_pos - e_pos) / max(np.linalg.norm(h_pos - e_pos), 1e-9)
            last_dirs[side] = (u, f)
            if tr is None:  # legacy: anchored at the (scaled) human shoulder
                e2 = s_pos + u * info["upper_arm_m"] / info["scale"]
                h2 = e2 + f * info["forearm_m"] / info["scale"]
            else:
                # targets in GMR's target frame, anchored at the robot shoulder; mapped back through GMR's affine
                # per-body transform T_b(p) = s_b * p + c_b, so that after GMR's scaling they land exactly there
                p_sh = pel + r_pel @ np.asarray(info["shoulder_in_pelvis"])
                t_e = p_sh + u * info["upper_arm_m"]
                t_h = t_e + f * info["forearm_m"]
                s_e, s_h = float(gmr.human_scale_table[el]), float(gmr.human_scale_table[ha])
                e2 = (t_e - (tr[el][0] - s_e * e_pos)) / s_e
                h2 = (t_h - (tr[ha][0] - s_h * h_pos)) / s_h
            fr[el] = [e2, fr[el][1]]
            fr[ha] = [h2, fr[ha][1]]
        return fr

    arm_bodies = None
    if arm_info is not None:
        arm_bodies = {side: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{b}")
                             for b in ("shoulder_yaw_link", "elbow_link", "wrist_roll_link")] for side in arm_info}

    task_bodies = list(gmr.human_body_to_task2.keys())
    tasks = [gmr.human_body_to_task2[k] for k in task_bodies]
    robot_bodies = [t.frame_name for t in tasks]
    n = len(frames)
    qpos = np.zeros((n, m.nq))
    pos_err = np.zeros((n, len(tasks)))
    rot_err = np.zeros((n, len(tasks)))
    human_pos = np.zeros((n, len(tasks), 3))
    arm_dir_err = np.zeros((n, 2, 2))
    # raw (unscaled) human ankle + toe positions, for source-foot contact hints (dropbear_wbc.motion.gmr_serial)
    foot_bones = [(f"{s}Foot", f"{s}Toe") for s in ("Left", "Right")]
    have_toes = all(b in frames[0] for pair in foot_bones for b in pair)
    human_feet_raw = np.full((n, 2, 2, 3), np.nan)  # (frame, left/right, upper arm/forearm) angle robot vs human direction [rad]
    solve_ms = np.zeros(n)
    # warm start: GMR starts from the model's default pose at the origin and runs <= 10 IK iterations per frame, so
    # the first frames of a mid-clip window would carry a convergence transient. Converge on frame 0 first.
    first = preroll_frames[0] if preroll_frames else frames[0]
    for k in range(max(1, args.warm_start)):
        f0 = prep(first)
        seed_arms(f0, first=(k == 0))
        solve(f0, -1)
    for fr in preroll_frames:  # tracked but not recorded
        fr = prep(fr)
        seed_arms(fr)
        solve(fr, -1)
    for i, fr in enumerate(frames):
        ts = time.perf_counter()
        if have_toes:
            human_feet_raw[i] = [[np.asarray(fr[b][0], dtype=np.float64) for b in pair] for pair in foot_bones]
        fr = prep(fr)
        seed_arms(fr)
        qpos[i] = solve(fr, i)
        solve_ms[i] = 1e3 * (time.perf_counter() - ts)
        for k, t in enumerate(tasks):
            e = t.compute_error(gmr.configuration)
            pos_err[i, k] = float(np.linalg.norm(e[:3]))
            rot_err[i, k] = float(np.linalg.norm(e[3:]))
            human_pos[i, k] = gmr.scaled_human_data[task_bodies[k]][0]
        if arm_info is not None:
            for side, info in arm_info.items():
                arm_err[side] = max(float(pos_err[i, task_bodies.index(b)]) for b in info["bones"][1:])
            xp = gmr.configuration.data.xpos
            for k_side, side in enumerate(("left", "right")):
                if side not in arm_bodies:
                    continue
                ps, pe_, pw = (xp[b] for b in arm_bodies[side])
                for k_seg, (a_, b_, hum) in enumerate(((ps, pe_, last_dirs[side][0]), (pe_, pw, last_dirs[side][1]))):
                    r_ = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
                    arm_dir_err[i, k_side, k_seg] = float(np.arccos(np.clip(r_ @ hum, -1.0, 1.0)))
    meta = {
        "schema": "gmr-retarget-npz-v1", "robot": args.robot, "xml": str(ROBOT_XML_DICT[args.robot]).replace("\\", "/"),
        "ik_config": str(IK_CONFIG_DICT[f"bvh_{args.format}"][args.robot]).replace("\\", "/"),
        "bvh": str(args.bvh).replace("\\", "/"), "format": args.format, "fps": args.fps,
        "frame_range": [a, a + n], "gmr_human_height_used": human_height, "warm_start_calls": args.warm_start,
        "preroll_frames": len(preroll_frames),
        "motor_limits": None if motor_limit is None else {
            "constraint": "dropbear_wbc.motion.gmr_limits.DropbearMotorLimit (linearised serial3 + ankle-pair motor bounds)",
            "calibration": str(args.calibration).replace("\\", "/"), "fallback_frames": fallback_frames},
        "arm_mode": None if args.robot != "dropbear" else {
            "mode": args.arm_mode, "pos_cost": args.arm_pos_cost, "rot_cost": args.arm_rot_cost,
            "analytic_seed": seeder is not None, "seed_max_step_rad": args.arm_seed_max_step,
            "seed_skipped_frames": seed_skipped, "seed_rescued_frames": seed_rescued,
            "rescue_err_m": args.arm_rescue_err,
            "segments": arm_info},
        "comfort_limits": None if comfort is None else {
            "constraint": "dropbear_wbc.motion.gmr_limits.comfort_limit (G1 anatomical shoulder ranges, IK only)",
            "ranges_rad": comfort.applied_ranges},
        "gmr_commit_file": "third_party/gmr_dropbear/gmr_commit.txt",
        "license": {"license": "CC-BY-NC-ND-4.0",
                    "summary": "Ubisoft La Forge Animation Dataset (LAFAN1). Internal non-commercial R&D only; "
                               "retargeted outputs are Adapted Material and must not be shared/redistributed.",
                    "redistributable": False,
                    "url": "https://github.com/ubisoft/ubisoft-laforge-animation-dataset"},
        "wall_s": None,
    }
    meta["wall_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, fps=np.float64(args.fps), qpos=qpos, qpos_joint_names=np.array(hinge_names),
                        robot=args.robot, task_bodies=np.array(task_bodies), task_robot_bodies=np.array(robot_bodies),
                        task_pos_err=pos_err, task_rot_err=rot_err, human_pos=human_pos, solve_ms=solve_ms,
                        arm_dir_err=arm_dir_err, human_feet_raw=human_feet_raw,
                        meta=json.dumps(meta))
    pe = {tb: float(np.percentile(pos_err[:, k], 95)) for k, tb in enumerate(task_bodies)}
    if arm_info is not None:
        ad = np.degrees(arm_dir_err)
        print(f"[gmr_retarget] arm direction error robot vs human [deg] upper arm p50/p95 L {np.median(ad[:, 0, 0]):.1f}/"
              f"{np.percentile(ad[:, 0, 0], 95):.1f} R {np.median(ad[:, 1, 0]):.1f}/{np.percentile(ad[:, 1, 0], 95):.1f}; "
              f"forearm L {np.median(ad[:, 0, 1]):.1f}/{np.percentile(ad[:, 0, 1], 95):.1f} "
              f"R {np.median(ad[:, 1, 1]):.1f}/{np.percentile(ad[:, 1, 1], 95):.1f}")
    print(f"[gmr_retarget] {args.bvh.name} [{a}:{a + n}] -> {args.robot}: {n} frames in {meta['wall_s']:.1f} s "
          f"({np.mean(solve_ms):.1f} ms/frame); task pos err p95 [m] {json.dumps({k: round(v, 3) for k, v in pe.items()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
