"""Settle Dropbear motion CSVs in Isaac Lab and write the contract motion NPZs (``dropbear-motion-npz-v1``).

Pipeline per clip (docs/CONTRACTS.md sections 3-4):

1. Read ``<clip>.csv`` (``dropbear-motion-csv-v1``) + sidecar ``<clip>.json`` (``fps``, optional
   ``contact_hint``); resample to 50 Hz (lerp root position / motors, slerp root quaternion).
2. Motor targets are clipped to the authored motor limits (reported).
3. Kinematic settle (Isaac Lab 2.2, ``dropbear_wbc.isaac.quasistatic``): gravity off, root link FIXED, the
   contract 0.1 plant fixes, stiff PD on the 22 motors, neck held at 0, passive joints free (lightly damped),
   27 loop closures solved by PhysX. The frames are split into contiguous chunks, one chunk per env. Each env
   ramps its motor targets from the rest pose to its first frame (<= ``--ramp-rate-deg`` per physics step),
   then from frame to frame (ramp of max(``--min-ramp``, needed) steps) and holds each frame for
   ``--hold`` steps before reading it. Fixed step counts; no velocity criterion (the loop joints keep a
   ~0.02 rad/s solver noise floor). Per frame we store the joint-position change over the last
   ``--window`` hold steps (``dq_window``) and the motor tracking error.
4. The settled FULL joint state (all 91 DOFs) and all 90 link poses (relative to the root link) are read and
   composed with the CSV root pose (gravity is off, so the internal configuration does not depend on it).
5. Ground fix: per frame, root z is shifted so the lowest sole point (collision convex hull,
   ``data/calibration/dropbear_foot_sole_hulls.json``) of the contact foot is at z = 0 (sidecar
   ``contact_hint`` if present, else the lower foot); the shift is smoothed (Gaussian, ``--ground-sigma``).
6. Velocities by finite differences (BeyondMimic ``csv_to_npz``): ``joint_vel`` = gradient of joint_pos,
   ``body_lin_vel_w`` = gradient of body COM positions (Isaac Lab ``body_lin_vel_w`` is a COM velocity),
   ``body_ang_vel_w`` from central SO(3) differences.
7. ``closure_residual_m`` = worst anchor gap of the 27 excluded joints per frame. Frames above
   ``--max-residual`` (default 5 mm) are flagged; if more than ``--max-flagged-frac`` of the frames are
   flagged the output is written as ``<out>.rejected.npz`` (meta.status = "rejected") and the exit code is 2.
   Since v2.2 the same happens when the settled MOTOR reference exceeds ``--max-motor-vel`` (10 rad/s, the USD motor
   cap) or jumps more than ``--max-motor-step`` (0.3 rad) between frames (retarget branch flips);
   ``meta.status_reasons`` says why.
8. The written NPZ is re-loaded with ``dropbear_wbc.tasks.tracking.motion_npz`` (the tracking env's reader)
   and its provenance validated.

Several clips can be settled in one Isaac process (``--csv a.csv b.csv ...``); each NPZ goes next to its CSV
(``<clip>.npz``, or ``<clip><--out-suffix>.npz``, e.g. ``_v4`` so that an existing NPZ is never overwritten) unless
``--out`` is given (single clip only). ``--keep-going`` records a clip that raises (e.g. an INVALID sidecar) in the
summary and continues with the next one instead of aborting the batch (foot_contact track, 2026-09-24).

GPU: run under the team lock (``docs/CONTRACTS.md`` section 0), e.g.::

    python tools/gpu_lock_run.py --owner calibrate_settle --log logs/calibrate_settle/settle_x.log -- \
        C:/isaac-sim/python.bat -u tools/settle_motion.py --csv data/motions/synthetic/stand.csv \
        --gpu-lock-held --headless

Also: ``--make-test standing|sweep`` writes a synthetic CSV (+ sidecar) from the semantic calibration and
exits (CPU only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

TOOL_VERSION = "settle_motion 2.2"  # 2.1: CONTRACTS 0.2 ankle variant; 2.2: motor reference-dynamics gate


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", help="input dropbear-motion-csv-v1 file(s)")
    p.add_argument("--sidecar", type=Path, default=None, help="sidecar JSON (single clip; default <csv>.json)")
    p.add_argument("--out", type=Path, default=None, help="output NPZ (single clip; default <csv>.npz)")
    p.add_argument("--out-suffix", default="", help="output NPZ = <csv stem><suffix>.npz (e.g. _v4; default none)")
    p.add_argument("--keep-going", action="store_true", help="on a per-clip error, record it and continue the batch")
    p.add_argument("--fps-out", type=float, default=50.0)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--pos-iters", type=int, default=16)
    p.add_argument("--motor-kp", type=float, default=10000.0)
    p.add_argument("--motor-kd", type=float, default=100.0)
    p.add_argument("--passive-damping", type=float, default=0.5)
    p.add_argument("--ramp-rate-deg", type=float, default=0.5, help="max motor target change per physics step [deg]")
    p.add_argument("--min-ramp", type=int, default=4, help="min ramp steps between consecutive frames")
    p.add_argument("--hold", type=int, default=12, help="hold steps per frame before reading")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--ground-sigma", type=float, default=0.1, help="ground-shift smoothing sigma [s]")
    p.add_argument("--max-residual", type=float, default=0.005, help="closure residual flag threshold [m]")
    p.add_argument("--max-flagged-frac", type=float, default=0.0)
    p.add_argument("--allow-invalid", action="store_true", help="settle a clip whose sidecar is marked INVALID")
    p.add_argument("--max-motor-vel", type=float, default=10.0,
                   help="reject when any settled motor |joint_vel| exceeds this [rad/s] (USD motor cap)")
    p.add_argument("--max-motor-step", type=float, default=0.3,
                   help="reject when a motor moves more than this between two output frames [rad] (branch flip)")
    p.add_argument("--sole", type=Path, default=REPO / "data/calibration/dropbear_foot_sole_hulls.json")
    p.add_argument("--calibration", type=Path, default=REPO / "data/calibration/dropbear_semantic_calibration.json")
    p.add_argument("--log", type=str, default="", help="log path recorded in meta (the wrapper's --log)")
    p.add_argument("--make-test", choices=["standing", "sweep"], default=None,
                   help="write a synthetic test CSV to --csv and exit")
    p.add_argument("--test-seconds", type=float, default=2.0)
    p.add_argument("--test-fps", type=float, default=50.0)
    p.add_argument("--gpu-lock-held", action="store_true")
    p.add_argument("--authored-ankle", action="store_true",
                   help="keep the authored revolute ankle tie rods (CONTRACTS 0.2 opt-out; default: spherical, "
                        "or $DROPBEAR_AUTHORED_ANKLE=1)")
    return p.parse_known_args()


ARGS, KIT_ARGS = _parse()


# ----------------------------------------------------------------------------------------------------
def make_test_csv(kind: str, path: Path, seconds: float, fps: float, calibration: Path) -> None:
    """Synthetic CSV: constant standing pose, or a smooth semantic sweep around it (both feet on ground)."""
    import numpy as np

    from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES, SemanticMap
    from dropbear_wbc.settle.motion_io import MotionClip, write_motion_csv

    smap = SemanticMap.load(calibration)
    n = int(round(seconds * fps)) + 1
    t = np.arange(n) / fps
    stand_m = smap.standing_motor_pos
    stand_s = smap.motor_to_semantic(stand_m)
    hip_h = float(smap.calib.get("hip_height", {}).get("standing_root_z_estimate", 0.0))
    if kind == "standing":
        motors = np.repeat(stand_m[None], n, axis=0)
        notes = "constant standing_motor_pos"
    else:
        sem = np.repeat(stand_s[None], n, axis=0)
        amp = {  # [rad], applied as amp * sin(2 pi f t + phase) around the standing pose
            "left_hip_pitch": (0.35, 0.5, 0.0), "right_hip_pitch": (0.35, 0.5, np.pi),
            "left_knee": (0.3, 0.5, 0.3), "right_knee": (0.3, 0.5, np.pi + 0.3),
            "left_ankle_pitch": (0.25, 0.7, 0.0), "right_ankle_pitch": (0.25, 0.7, 1.0),
            "left_ankle_roll": (0.1, 0.9, 0.0), "right_ankle_roll": (0.1, 0.9, 1.0),
            "left_hip_roll": (0.12, 0.6, 0.0), "right_hip_roll": (0.12, 0.6, 0.5),
            "left_hip_yaw": (0.2, 0.4, 0.0), "right_hip_yaw": (0.2, 0.4, 0.7),
            "left_shoulder_pitch": (0.8, 0.5, 0.0), "right_shoulder_pitch": (0.8, 0.5, np.pi),
            "left_shoulder_roll": (0.4, 0.4, 0.0), "right_shoulder_roll": (-0.4, 0.4, 0.0),
            "left_shoulder_yaw": (0.5, 0.6, 0.0), "right_shoulder_yaw": (0.5, 0.6, 1.0),
            "left_elbow": (0.3, 0.8, 0.0), "right_elbow": (0.3, 0.8, 1.0),
            "left_wrist_roll": (1.0, 0.5, 0.0), "right_wrist_roll": (1.0, 0.5, 1.0),
        }
        for name, (a, f, ph) in amp.items():
            sem[:, SEMANTIC_NAMES.index(name)] += a * np.sin(2 * np.pi * f * t + ph)
        motors, rep = smap.semantic_to_motor(sem, return_report=True)
        notes = f"semantic sinusoid sweep around standing pose; saturation={rep.summary()}"
    root_pos = np.zeros((n, 3))
    root_pos[:, 2] = hip_h
    root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    if kind == "sweep":  # also move the root a bit (x drift + small yaw) to exercise composition/slerp
        root_pos[:, 0] = 0.1 * t
        yaw = 0.2 * np.sin(2 * np.pi * 0.25 * t)
        root_quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
    contact = np.ones((n, 2), dtype=bool)
    clip = MotionClip(fps=fps, root_pos=root_pos, root_quat=root_quat, motor_pos=motors, contact=contact,
                      sidecar={"source": "synthetic", "source_file": None, "source_license": "n/a",
                               "retarget_method": f"tools/settle_motion.py --make-test {kind}",
                               "semantic_names": list(SEMANTIC_NAMES), "notes": notes})
    write_motion_csv(path, clip)
    print(f"[make-test] wrote {path} frames={n} fps={fps} kind={kind}")


# ----------------------------------------------------------------------------------------------------
def settle_frames(qs, motors, ramp_rate: float, min_ramp: int, hold: int, window: int, tag: str = "") -> dict:
    """Kinematically settle every frame of ``motors`` (T, 22); per-frame numpy arrays (see module doc)."""
    import numpy as np
    import torch

    n_env = qs.num_envs
    t_frames = motors.shape[0]
    chunk = int(np.ceil(t_frames / n_env))
    starts = np.arange(n_env) * chunk
    active = starts < t_frames
    n_j, n_b, n_c = len(qs.joint_names), len(qs.body_names), len(qs.closures)
    out = {"joint_pos": np.zeros((t_frames, n_j)), "body_pos": np.zeros((t_frames, n_b, 3)),
           "body_quat": np.zeros((t_frames, n_b, 4)), "motor_pos": np.zeros((t_frames, 22)),
           "gap": np.zeros(t_frames), "gaps": np.zeros((t_frames, n_c)), "dq_window": np.zeros(t_frames),
           "motor_err": np.zeros(t_frames), "ramp_steps": np.zeros(t_frames, dtype=np.int64)}
    done = np.zeros(t_frames, dtype=bool)
    qs.write_joint_state(torch.zeros(n_env, n_j, device=qs.device))
    qs.step(2)
    current = torch.zeros(n_env, 22, device=qs.device)
    steps0 = qs.physics_steps
    t0 = time.time()
    for i in range(chunk):
        idx = np.minimum(starts + i, t_frames - 1)
        tg = motors[idx].copy()
        tg[~active] = 0.0
        tg_t = torch.tensor(tg, dtype=torch.float32, device=qs.device)
        delta = float((tg_t - current).abs().max())
        ramp = max(min_ramp, int(np.ceil(delta / ramp_rate - 1e-9)))
        info = qs.move(current, tg_t, ramp, hold, window)
        current = tg_t
        st = qs.read_state()
        rec = active & (starts + i < t_frames)
        f = starts[rec] + i
        out["joint_pos"][f] = st["joint_pos"][rec]
        out["body_pos"][f] = st["body_pos"][rec]
        out["body_quat"][f] = st["body_quat"][rec]
        out["motor_pos"][f] = st["motor_pos"][rec]
        out["gaps"][f] = st["gaps"][rec]
        out["gap"][f] = info["gap"].cpu().numpy()[rec]
        out["dq_window"][f] = info["dq_window"].cpu().numpy()[rec]
        out["motor_err"][f] = info["motor_err"].cpu().numpy()[rec]
        out["ramp_steps"][f] = ramp
        done[f] = True
        if i % 10 == 0 or i == chunk - 1:
            print(f"[settle{tag}] frame-step {i + 1}/{chunk} ramp={ramp} worst_gap={1e3 * out['gap'][f].max():.3f} mm "
                  f"max_dq={out['dq_window'][f].max():.1e} wall={time.time() - t0:.1f}s", flush=True)
    if not done.all():
        raise RuntimeError(f"frames not settled: {np.nonzero(~done)[0][:10]}")
    # continuity across chunk boundaries (different envs) vs within chunks
    rev = np.array([not n.startswith("head_LeadScrew") for n in qs.joint_names])
    d = out["joint_pos"][1:] - out["joint_pos"][:-1]
    d[:, rev] = (d[:, rev] + np.pi) % (2 * np.pi) - np.pi
    step_max = np.abs(d).max(axis=1)
    boundary = np.zeros(t_frames - 1, dtype=bool)
    b_idx = starts[1:][starts[1:] < t_frames] - 1
    boundary[b_idx] = True
    out["continuity"] = {
        "within_chunk_joint_step_p99": float(np.percentile(step_max[~boundary], 99)) if (~boundary).any() else 0.0,
        "within_chunk_joint_step_max": float(step_max[~boundary].max()) if (~boundary).any() else 0.0,
        "chunk_boundary_joint_step_max": float(step_max[boundary].max()) if boundary.any() else 0.0,
        "chunks": int(active.sum()), "frames_per_chunk": chunk}
    out["wall_s"] = time.time() - t0
    out["physics_steps"] = qs.physics_steps - steps0
    return out


def settle_clip(qs, csv_path: Path, sidecar: Path | None, out_path: Path, usd_sha: str) -> dict:
    import numpy as np

    from dropbear_wbc.kinematics.closures_np import ClosureTable, closure_residuals
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
    from dropbear_wbc.settle.ground import SoleModel, ground_correction
    from dropbear_wbc.settle.kinematics import (angular_velocity, com_positions, compose_root, linear_velocity,
                                                unwrap_joints)
    from dropbear_wbc.settle.motion_io import read_motion_csv, resample
    from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz, validate_provenance

    t_all = time.time()
    clip_in = read_motion_csv(csv_path, sidecar)
    # the motor columns were produced by SemanticMap of some calibration: it must be for THIS plant's ankle variant
    # (review fix 2026-09-24). Sidecars before that record no variant -> recorded as unknown, not refused.
    if (clip_in.sidecar or {}).get("INVALID") and not ARGS.allow_invalid:
        raise RuntimeError(f"{csv_path}: sidecar marked INVALID ({clip_in.sidecar['INVALID']}); pass --allow-invalid "
                           "only for debugging")
    cal_prov = (clip_in.sidecar or {}).get("calibration") or {}
    csv_ankle = cal_prov.get("authored_ankle_tierods")
    if csv_ankle is not None and bool(csv_ankle) != bool(qs.cfg.authored_ankle_tierods):
        raise RuntimeError(f"{csv_path}: motor columns come from a calibration of the "
                           f"{'authored' if csv_ankle else 'spherical'} ankle, but the settle plant uses the "
                           f"{'authored' if qs.cfg.authored_ankle_tierods else 'spherical'} one (CONTRACTS 0.2): "
                           "re-run the retarget with the matching calibration")
    clip = resample(clip_in, ARGS.fps_out)
    dt = 1.0 / ARGS.fps_out
    lim = qs.motor_limits
    motors = np.clip(clip.motor_pos, lim[:, 0], lim[:, 1])
    clipped = np.abs(motors - clip.motor_pos) > 1e-9
    print(f"[settle] {csv_path}: in {clip_in.num_frames} frames @ {clip_in.fps} Hz -> {clip.num_frames} @ "
          f"{ARGS.fps_out} Hz; motor clipping on {int(clipped.any(axis=1).sum())} frames", flush=True)
    res = settle_frames(qs, motors, np.deg2rad(ARGS.ramp_rate_deg), ARGS.min_ramp, ARGS.hold, ARGS.window,
                        tag=f":{csv_path.stem}")

    # ---- compose with the CSV root pose, ground fix ----------------------------------------------------
    joint_names, body_names = list(qs.joint_names), list(qs.body_names)
    if body_names[0] != "world":
        raise RuntimeError(f"body 0 is {body_names[0]!r}, expected the root 'world'")
    body_pos_w, body_quat_w = compose_root(clip.root_pos, clip.root_quat, res["body_pos"], res["body_quat"])
    sole = SoleModel.load(ARGS.sole)
    if sole.usd_sha256 and sole.usd_sha256 != usd_sha:
        raise RuntimeError("sole hull file was extracted from a different USD")
    lowest = sole.lowest_z(body_pos_w, body_quat_w, body_names)
    gfix = ground_correction(lowest, ARGS.fps_out, clip.contact, ARGS.ground_sigma)
    body_pos_w[..., 2] += gfix.dz[:, None]
    root_pos = clip.root_pos.copy()
    root_pos[:, 2] += gfix.dz

    # ---- velocities ---------------------------------------------------------------------------------------
    revolute = np.array([not n.startswith("head_LeadScrew") for n in joint_names])
    joint_pos = unwrap_joints(res["joint_pos"], revolute)
    joint_vel = linear_velocity(joint_pos, dt)
    com_w = com_positions(body_pos_w, body_quat_w, qs.com_pos_b)
    body_lin_vel_w = linear_velocity(com_w, dt)
    body_ang_vel_w = angular_velocity(body_quat_w, dt)

    # ---- closure residuals & flags --------------------------------------------------------------------------
    table = ClosureTable.from_npz(qs.closure_table())
    gaps, angs = closure_residuals(res["body_pos"], res["body_quat"], table)
    residual = gaps.max(axis=1)
    worst_closure = np.asarray(table.names)[gaps.argmax(axis=1)]
    motor_err = np.abs(res["motor_pos"] - motors)
    flags = residual > ARGS.max_residual
    flagged_frac = float(flags.mean())
    root_err = float(np.abs(body_pos_w[:, 0] - root_pos).max())
    contact_after = gfix.contact_sole_z_after
    both = np.stack([gfix.sole_z_after["left"], gfix.sole_z_after["right"]], -1)
    in_contact = both[gfix.contact]

    meta = {
        "schema": "dropbear-motion-npz-v1", "tool": TOOL_VERSION, "script": "tools/settle_motion.py",
        "log": ARGS.log, "created_unix": time.time(),
        "source_csv": str(csv_path).replace("\\", "/"), "sidecar": clip_in.sidecar,
        "input_fps": clip_in.fps, "input_frames": clip_in.num_frames, "fps": ARGS.fps_out, "frames": clip.num_frames,
        "usd_path": qs.cfg.usd_path, "usd_sha256": usd_sha, "sole_hulls": str(ARGS.sole).replace("\\", "/"),
        "authored_ankle_tierods": bool(qs.cfg.authored_ankle_tierods),
        "source_calibration": {"sha256": cal_prov.get("calibration_sha256"),
                               "authored_ankle_tierods": csv_ankle,
                               "path": cal_prov.get("calibration_path"),
                               "note": None if csv_ankle is not None else
                               "sidecar predates the calibration sha/variant record (2026-09-24): variant unknown"},
        "settle": {**qs.cfg.to_dict(), "ramp_rate_deg_per_step": ARGS.ramp_rate_deg, "min_ramp": ARGS.min_ramp,
                   "hold": ARGS.hold, "window": ARGS.window, "gravity": "off",
                   "root": "fixed (fix_root_link); poses composed with the CSV root pose",
                   "plant_fixes": qs.stage_fixes},
        "ground": {"method": "contact-foot lowest sole point to z=0, Gaussian-smoothed", "sigma_s": ARGS.ground_sigma,
                   "contact_source": gfix.contact_source, "dz_min": float(gfix.dz.min()), "dz_max": float(gfix.dz.max()),
                   "contact_sole_z_after_min": float(np.nanmin(contact_after)),
                   "contact_sole_z_after_max": float(np.nanmax(contact_after)),
                   "contact_sole_abs_p95": float(np.nanpercentile(np.abs(contact_after), 95)),
                   "all_contact_feet_sole_z_min": float(in_contact.min()) if in_contact.size else None,
                   "all_contact_feet_sole_z_max": float(in_contact.max()) if in_contact.size else None,
                   "min_sole_z_any_foot": float(both.min())},
        "velocities": "finite differences (BeyondMimic csv_to_npz); body_lin_vel_w = COM velocity",
        "motor_clipping_frames": int(clipped.any(axis=1).sum()),
        "motor_tracking_error_max_rad": float(motor_err.max()),
        "motor_tracking_error_p95_rad": float(np.percentile(motor_err.max(axis=1), 95)),
        "dq_window_max": float(res["dq_window"].max()), "dq_window_p95": float(np.percentile(res["dq_window"], 95)),
        "closure": {"threshold_m": ARGS.max_residual, "max_m": float(residual.max()), "mean_m": float(residual.mean()),
                    "p95_m": float(np.percentile(residual, 95)), "max_angle_rad": float(angs.max()),
                    "flagged_frames": int(flags.sum()), "flagged_fraction": flagged_frac,
                    "worst_closure_names": sorted(set(worst_closure[residual > ARGS.max_residual].tolist())),
                    "worst_closure_overall": str(worst_closure[int(residual.argmax())])},
        "continuity": res["continuity"],
        "runtime": {"settle_s": float(res["wall_s"]), "physics_steps": int(res["physics_steps"]),
                    "frames_per_s": clip.num_frames / max(float(res["wall_s"]), 1e-9),
                    "total_s": time.time() - t_all, "num_envs": qs.num_envs},
        "root_pos_consistency_m": root_err,
    }
    from dropbear_wbc.settle.quality import motor_dynamics

    dyn = motor_dynamics(joint_pos, joint_vel, joint_names, list(MOTOR_NAMES), ARGS.max_motor_vel, ARGS.max_motor_step)
    reasons = list(dyn.pop("problems"))
    meta["motor_dynamics"] = dyn
    if flagged_frac > ARGS.max_flagged_frac:
        reasons.insert(0, f"closure residual > {ARGS.max_residual} m on {flagged_frac:.3f} of frames")
    meta["status_reasons"] = reasons
    status = "rejected" if reasons else "ok"
    if status == "rejected":
        out_path = out_path.with_suffix(".rejected.npz")
    meta["status"] = status
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    np.savez_compressed(
        out_path, fps=np.asarray(ARGS.fps_out),
        joint_pos=f32(joint_pos), joint_vel=f32(joint_vel), body_pos_w=f32(body_pos_w), body_quat_w=f32(body_quat_w),
        body_lin_vel_w=f32(body_lin_vel_w), body_ang_vel_w=f32(body_ang_vel_w),
        joint_names=np.array(joint_names), body_names=np.array(body_names), motor_names=np.array(MOTOR_NAMES),
        closure_residual_m=f32(residual), meta=np.array(json.dumps(meta)),
        # extras (superset of the contract keys)
        closure_residual_rad=f32(angs.max(axis=1)), frame_flags=flags, ground_dz=f32(gfix.dz),
        motor_target=f32(motors), contact=gfix.contact, dq_window=f32(res["dq_window"]),
    )
    # re-load with the tracking env's reader (fail closed on any contract violation)
    m = load_motion_npz(out_path)
    validate_provenance(m, expected_usd_sha256=usd_sha, allow_rejected=True,
                        expected_authored_ankle=bool(qs.cfg.authored_ankle_tierods))
    meta["reload_check"] = {"ok": True, "frames": m.num_frames, "J": len(m.joint_names), "B": len(m.body_names)}
    report = {"status": status, "out": str(out_path).replace("\\", "/"), **meta,
              "per_frame": {"closure_residual_m": residual.tolist(), "dq_window": res["dq_window"].tolist(),
                            "ramp_steps": res["ramp_steps"].tolist(), "ground_dz": gfix.dz.tolist(),
                            "contact_sole_z_after": np.nan_to_num(contact_after, nan=-1.0).tolist(),
                            "motor_tracking_error_max": motor_err.max(axis=1).tolist()}}
    rpath = out_path.with_suffix(".report.json")
    rpath.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: meta[k] for k in ("frames", "closure", "ground", "continuity", "runtime", "status",
                                            "motor_tracking_error_max_rad", "dq_window_max",
                                            "root_pos_consistency_m", "reload_check")}, indent=1), flush=True)
    print(f"[settle] wrote {out_path} ({status}); report {rpath}", flush=True)
    return {"csv": str(csv_path), "out": str(out_path), "status": status, "frames": clip.num_frames,
            "closure_max_mm": 1e3 * float(residual.max()), "settle_s": float(res["wall_s"])}


def run_settle() -> int:
    from dropbear_wbc.isaac.launch import sha256_file
    from dropbear_wbc.isaac.quasistatic import QuasiStaticCfg, QuasiStaticDropbear

    csvs = list(ARGS.csv)
    if (ARGS.out is not None or ARGS.sidecar is not None) and len(csvs) != 1:
        raise SystemExit("--out/--sidecar need exactly one --csv")
    cfg = QuasiStaticCfg(num_envs=ARGS.num_envs, pos_iters=ARGS.pos_iters, motor_kp=ARGS.motor_kp,
                         motor_kd=ARGS.motor_kd, passive_damping=ARGS.passive_damping,
                         authored_ankle_tierods=True if ARGS.authored_ankle else None)
    t_setup = time.time()
    qs = QuasiStaticDropbear(cfg)
    print(f"[settle] articulation ready in {time.time() - t_setup:.1f}s J={len(qs.joint_names)} B={len(qs.body_names)} "
          f"fixes={qs.stage_fixes}", flush=True)
    usd_sha = sha256_file(cfg.usd_path)
    results = []
    for c in csvs:
        out = ARGS.out if ARGS.out is not None else c.with_name(c.stem + ARGS.out_suffix + ".npz")
        try:
            results.append(settle_clip(qs, c, ARGS.sidecar, out, usd_sha))
        except Exception as exc:  # noqa: BLE001
            if not ARGS.keep_going:
                raise
            import traceback

            traceback.print_exc()
            print(f"[settle] FAILED {c}: {type(exc).__name__}: {exc} (--keep-going: next clip)", flush=True)
            results.append({"csv": str(c), "out": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    print("[settle] summary " + json.dumps(results), flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 2


def main() -> int:
    if ARGS.make_test:
        if not ARGS.csv:
            raise SystemExit("--make-test needs --csv")
        make_test_csv(ARGS.make_test, ARGS.csv[0], ARGS.test_seconds, ARGS.test_fps, ARGS.calibration)
        return 0
    if not ARGS.csv:
        raise SystemExit("--csv is required")
    if not ARGS.gpu_lock_held or not (REPO / ".locks/gpu.lock").exists():
        raise SystemExit("run under tools/gpu_lock_run.py with --gpu-lock-held (docs/CONTRACTS.md section 0)")
    from dropbear_wbc.isaac.launch import close_app_and_exit, prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(KIT_ARGS)
    app_args.headless = True
    app = AppLauncher(app_args).app
    rc = 1
    try:
        rc = run_settle()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    close_app_and_exit(app, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
