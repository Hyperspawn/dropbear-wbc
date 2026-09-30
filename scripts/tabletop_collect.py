"""Run the Dropbear tabletop push task: scripted baseline evaluation and GR00T episode recording (Isaac Lab 2.2).

For every placement of the chosen split (``data/groot/placements/*.json``, made by ``tools/tabletop_placements.py``) one
episode is run with the chosen policy; the success rule is the env's (block centre in the zone with 1 cm margin, on
the table, upright, at rest for 0.5 s). With ``--record`` every attempt is written as a raw episode
(``source/dropbear_wbc/tasks/tabletop/episode_io.py``: lowdim.npz + meta.json + one mp4 per camera) that
``tools/build_groot_dataset.py`` assembles into the GR00T LeRobot v2 dataset.

Policies:
* ``scripted``  IK-based pusher on privileged block / zone poses (``tasks/tabletop/scripted.py``).
* ``teleop_keys`` the teleop keyboard device (``teleop.devices.KeyboardSource``) driven by a scripted key sequence
  (headless plumbing check of the teleop path: device -> wrist targets -> teleop IK -> env; NOT a task solution);
  ``teleop_keyboard`` reads a real console keyboard (msvcrt; interactive use, UNVERIFIED headless).

Examples (always under the GPU lock)::

    python tools/gpu_lock_run.py --owner tabletop --log logs/tabletop/baseline_heldout.log --wait-minutes 120 -- \
        C:/isaac-sim/python.bat -u scripts/tabletop_collect.py --headless --split heldout --num_envs 4 \
        --summary logs/tabletop/baseline_heldout.json
    python tools/gpu_lock_run.py ... -- C:/isaac-sim/python.bat -u scripts/tabletop_collect.py --headless \
        --split train --episodes 16 --num_envs 2 --record data/groot/_raw/dropbear_tabletop_push_v1 --wrist_cams
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Dropbear tabletop push: scripted baseline / teleop / recording.")
parser.add_argument("--placements", type=Path, default=_REPO / "data/groot/placements/tabletop_push_v1.json")
parser.add_argument("--split", default="heldout", choices=["train", "heldout"])
parser.add_argument("--start", type=int, default=0, help="first placement index within the split")
parser.add_argument("--episodes", type=int, default=None, help="number of placements (default: the whole split)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--policy", default="scripted", choices=["scripted", "teleop_keys", "teleop_keyboard"])
parser.add_argument("--record", type=Path, default=None, help="raw episode root (enables the head camera)")
parser.add_argument("--wrist_cams", action="store_true", help="also record the two wrist cameras (320x240)")
parser.add_argument("--scene_cam", action="store_true", help="also record a third-person demo camera (not for GR00T)")
parser.add_argument("--no_head_cam", action="store_true", help="evaluation without any camera (fastest)")
parser.add_argument("--gravity_ff", default="on", choices=["on", "off"])
parser.add_argument("--dq_ff", default="on", choices=["on", "off"], help="arm velocity feed-forward (finite difference)")
parser.add_argument("--solver_iters", type=int, nargs=2, default=[32, 4], metavar=("POS", "VEL"))
parser.add_argument("--warmup_steps", type=int, default=6, help="unrecorded steps after each reset (settle, re-render)")
parser.add_argument("--summary", type=Path, default=None)
parser.add_argument("--jpeg_quality", type=int, default=95)
parser.add_argument("--max_wall_s", type=float, default=3000.0)
BLANK_STD = 4.0  # an RGB frame whose pixel std is below this is "blank" (uniform)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
want_head = (args.record is not None or args.scene_cam or args.wrist_cams) and not args.no_head_cam
if want_head or args.scene_cam or args.wrist_cams:
    args.enable_cameras = True
app = AppLauncher(args).app


def main() -> int:  # noqa: C901
    import gymnasium as gym
    import numpy as np
    import torch

    from dropbear_wbc.isaac.launch import sha256_file
    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES, USD_SHA256, resolve_usd_path
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tabletop import episode_io as eio
    from dropbear_wbc.tasks.tabletop.config import PUSH_ID
    from dropbear_wbc.tasks.tabletop.kinematics import TabletopArmIK
    from dropbear_wbc.tasks.tabletop.layout import LAYOUT, load_placements
    from dropbear_wbc.tasks.tabletop.mdp import PlacementFeed, success_now, zone_pos_root
    from dropbear_wbc.tasks.tabletop.scripted import PHASE_ID, PushParams, ScriptedPushPolicy
    from dropbear_wbc.tasks.tabletop.tabletop_env_cfg import ARM_JOINTS
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK
    from dropbear_wbc.teleop.gravity import ArmGravity

    t_start = time.perf_counter()
    register()
    placements, pmeta = load_placements(args.placements, args.split)
    placements = placements[args.start:]
    if args.episodes is not None:
        placements = placements[: args.episodes]
    if not placements:
        raise SystemExit("no placements selected")
    env_cfg = load_entry_point(PUSH_ID, "env_cfg_entry_point")
    env_cfg.scene.num_envs = min(args.num_envs, len(placements))
    env_cfg.enable_cameras(head=want_head, wrists=args.wrist_cams, scene=args.scene_cam)
    env_cfg.set_solver_iterations(*args.solver_iters)
    env_cfg.sim.device = args.device
    env_cfg.seed = 0
    cams = [c for c, on in (("head", want_head), ("left_wrist", args.wrist_cams), ("right_wrist", args.wrist_cams),
                            ("scene", args.scene_cam)) if on]
    cam_attr = {"head": "head_cam", "left_wrist": "left_wrist_cam", "right_wrist": "right_wrist_cam",
                "scene": "scene_cam"}
    env = gym.make(PUSH_ID, cfg=env_cfg)
    uw = env.unwrapped
    n_env = uw.num_envs
    dev = uw.device
    feed = PlacementFeed(list(placements))
    uw.tabletop_feed = feed
    robot = uw.scene["robot"]
    motor_ids, motor_names = robot.find_joints(list(MOTOR_NAMES), preserve_order=True)
    arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    if list(motor_names) != list(MOTOR_NAMES):
        raise RuntimeError(f"motor order mismatch: {motor_names}")
    hand_ids = [robot.find_bodies(n)[0][0] for n in ("LH_shoulder_ex_al_interface_1", "RH_shoulder_ex_al_interface_1")]
    root_w = torch.tensor(LAYOUT.root_pos_w, device=dev)

    base = DropbearArmIK()
    grav = ArmGravity(base)
    params = PushParams()
    iks = [TabletopArmIK(base=base) for _ in range(n_env)]
    pols = [ScriptedPushPolicy(iks[i], LAYOUT, params) for i in range(n_env)]
    q_rest = base.rest_q()
    motor_rest10 = iks[0].to_motor10(q_rest)

    # ---- teleop device path (plumbing): keyboard device -> wrist targets (torso frame) -> teleop IK
    kb = None
    if args.policy.startswith("teleop"):
        from dropbear_wbc.teleop.devices import KeyboardSource

        ref = base.fk_both(q_rest)
        kb = [KeyboardSource(ref, backend="msvcrt" if args.policy == "teleop_keyboard" else "none")
              for _ in range(n_env)]
        # scripted keys: start, left hand 12 cm forward + 10 cm up + 6 cm outward over ~5 s, then hold
        key_script = {0: "r"}
        for k in range(12):
            key_script[4 + 3 * k] = "w" if k % 2 == 0 else "wq"
        for k in range(6):
            key_script[45 + 3 * k] = "qa"
        tele_ik = [DropbearArmIK(base.path) for _ in range(n_env)]

    video = eio.VideoWriterThread() if args.record is not None else None
    run_stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    provenance = {
        "script": "scripts/tabletop_collect.py", "run_stamp": run_stamp, "task": PUSH_ID, "policy": args.policy,
        "placements_file": str(args.placements), "placements_sha256": sha256_file(args.placements),
        "split": args.split, "calibration": str(base.path), "calibration_sha256": base.sha256,
        "usd": resolve_usd_path(), "usd_sha256_contract": USD_SHA256, "layout": LAYOUT.to_dict(),
        "push_params": params.to_dict(), "gravity_ff": args.gravity_ff, "dq_ff": args.dq_ff, "solver_iters": list(args.solver_iters),
        "control_hz": LAYOUT.control_hz, "sim_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation,
        "cameras": cams, "argv": sys.argv,
    }

    def sem_measured(q22: np.ndarray) -> np.ndarray:
        return base.motor_to_semantic_arms_fast(q22)

    obs, _ = env.reset()
    # per-env episode state
    active = [i in feed.current and feed.current[i].pid != "default" for i in range(n_env)]
    warm = [args.warmup_steps] * n_env
    bufs = [eio.EpisodeBuffer(cams if args.record else [], args.jpeg_quality) for _ in range(n_env)]
    t_ep = [0.0] * n_env
    wall_ep = [time.perf_counter()] * n_env
    q_cmd_prev = [q_rest.copy() for _ in range(n_env)]
    ep_placement = [None] * n_env
    blank_frames: list[dict] = [{} for _ in range(n_env)]  # per env: camera -> blank frames in the current episode
    zone_check: list[dict | None] = [None] * n_env  # per env: zone render check of the current episode
    from dropbear_wbc.tasks.tabletop.cameras import green_centroid, project_head
    results: list[dict] = []
    stepped = 0
    step_dt = uw.step_dt

    def start_episode(i: int, q22: np.ndarray) -> None:
        p = feed.current[i]
        ep_placement[i] = p
        warm[i] = args.warmup_steps
        bufs[i] = eio.EpisodeBuffer(cams if args.record else [], args.jpeg_quality)
        t_ep[i] = 0.0
        wall_ep[i] = time.perf_counter()
        q_cmd_prev[i] = q_rest.copy()
        act_prev[i] = motor_rest10  # no dq* spike across a reset
        blank_frames[i] = {}
        zone_check[i] = None
        if args.policy == "scripted":
            pols[i].reset(p.side, p.zone_xy, sem_measured(q22))
        else:
            kb[i].offset = {s: np.zeros(3) for s in kb[i].offset}
            kb[i].roll = {s: 0.0 for s in kb[i].roll}
            tele_ik[i].reset(q_rest)
            kb[i].tele_step = 0
            kb[i].tracking = False

    q22_all = robot.data.joint_pos[:, motor_ids].cpu().numpy().astype(np.float64)
    act_prev = np.tile(motor_rest10, (n_env, 1))
    for i in range(n_env):
        if active[i]:
            start_episode(i, q22_all[i])
    stats = {"ik_err_max": 0.0, "step_wall_s": []}
    blank_stats = {"rerenders": 0, "blank_after_rerender": 0}
    from dropbear_wbc.tasks.tabletop.cameras import WRIST_CAM_STORE_RES

    def read_camera(c: str) -> np.ndarray:
        """(N, H, W, 3) uint8 of camera ``c``. A frame with std < BLANK_STD is re-rendered up to 3 times (smoke_v2 saw
        intermittent uniform frames on TiledCamera sensors); frames still blank are counted per env. Wrist views are
        downscaled to WRIST_CAM_STORE_RES."""
        sensor = uw.scene[cam_attr[c]]
        im = sensor.data.output["rgb"][..., :3]
        for _ in range(3):
            blank = im.float().flatten(1).std(dim=1) < BLANK_STD
            if not bool(blank.any()):
                break
            blank_stats["rerenders"] += 1
            uw.sim.render()
            sensor._is_outdated[:] = True  # noqa: SLF001 -- re-fetch the annotator output after the extra render
            sensor._update_outdated_buffers()  # noqa: SLF001
            im = sensor.data.output["rgb"][..., :3]
        out = im.cpu().numpy()
        blank = out.reshape(out.shape[0], -1).std(axis=1) < BLANK_STD
        for i in np.nonzero(blank)[0].tolist():
            if active[i] and warm[i] == 0:  # frames that are recorded
                blank_frames[i][c] = blank_frames[i].get(c, 0) + 1
                blank_stats["blank_after_rerender"] += 1
        if c in ("left_wrist", "right_wrist") and (out.shape[2], out.shape[1]) != tuple(WRIST_CAM_STORE_RES):
            from PIL import Image

            out = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f)).resize(tuple(WRIST_CAM_STORE_RES),
                                                                                         Image.BILINEAR)) for f in out])
        return out
    cam_stats: dict = {}
    print(f"[collect] reset_info {getattr(uw, 'tabletop_reset_info', None)}", flush=True)
    try:
        while any(active) and time.perf_counter() - t_start < args.max_wall_s:
            ts = time.perf_counter()
            q22_all = robot.data.joint_pos[:, motor_ids].cpu().numpy().astype(np.float64)
            dq22_all = robot.data.joint_vel[:, motor_ids].cpu().numpy().astype(np.float64)
            org = uw.scene.env_origins
            bpos = (uw.scene["block"].data.root_pos_w - org - root_w).cpu().numpy()
            bquat = uw.scene["block"].data.root_quat_w.cpu().numpy()
            zpos = zone_pos_root(uw).cpu().numpy()
            hpos = (robot.data.body_pos_w[:, hand_ids] - org[:, None] - root_w).cpu().numpy()
            hquat = robot.data.body_quat_w[:, hand_ids].cpu().numpy()
            succ_now = success_now(uw).cpu().numpy()
            images = {}
            if args.record is not None:
                for c in cams:
                    images[c] = read_camera(c)
                    if stepped % 50 == 0:  # image sanity: shape and brightness (an all-black frame = broken camera)
                        im = images[c]
                        cam_stats.setdefault(c, []).append({"step": stepped, "shape": list(im.shape),
                                                            "mean": round(float(im[..., :3].mean()), 2),
                                                            "std": round(float(im[..., :3].std()), 2)})
            act = np.tile(motor_rest10, (n_env, 1))
            tau = np.zeros((n_env, 10))
            for i in range(n_env):
                if not active[i]:
                    continue
                q_meas = sem_measured(q22_all[i])
                info = {}
                if warm[i] > 0:
                    q_cmd = q_rest.copy()
                elif args.policy == "scripted":
                    w, x, y, z = bquat[i]
                    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    q_cmd, info = pols[i].act(bpos[i], yaw, q_meas)
                else:
                    k = kb[i].tele_step
                    if args.policy == "teleop_keys" and k in key_script:
                        for ch in key_script[k]:
                            kb[i].feed(ch)
                    kb[i].tele_step += 1
                    tgt = kb[i].get(t_ep[i])
                    if "start" in tgt.events or "toggle" in tgt.events:
                        kb[i].tracking = True
                    if kb[i].tracking:
                        res = tele_ik[i].solve(tgt.left, tgt.right)
                        q_cmd = np.clip(res.q_sem, q_cmd_prev[i] - 0.08, q_cmd_prev[i] + 0.08)  # 1.6 rad/s
                        info = {"ik_err": max(a.pos_err_m for a in res.arms.values())}
                    else:
                        q_cmd = q_rest.copy()
                m10 = iks[i].to_motor10(q_cmd)
                act[i] = m10
                if args.gravity_ff == "on":
                    tau[i] = grav.motor_torque(q_cmd, m10)
                q_cmd_prev[i] = q_cmd
                stats["ik_err_max"] = max(stats["ik_err_max"], float(info.get("ik_err", 0.0)))
                if warm[i] > 0:
                    warm[i] -= 1
                    continue
                if True:  # rows are always kept (summary statistics); images only with --record
                    row = {
                        "observation.state": q_meas, "action": q_cmd, "observation.motor_q": q22_all[i],
                        "observation.motor_dq": dq22_all[i], "action.motor_q": m10, "action.tau_ff": tau[i],
                        "observation.block_pose": np.concatenate([bpos[i], bquat[i]]), "observation.zone_pos": zpos[i],
                        "observation.left_hand_pose": np.concatenate([hpos[i, 0], hquat[i, 0]]),
                        "observation.right_hand_pose": np.concatenate([hpos[i, 1], hquat[i, 1]]),
                        "policy.tool_cmd": info.get("tool_cmd", np.full(3, np.nan)),
                        "policy.phase": info.get("phase_id", -1), "policy.ik_err": info.get("ik_err", np.nan),
                        "task.success_now": float(succ_now[i]), "time.sim_s": t_ep[i],
                    }
                    if args.record is not None and "head" in cams and len(bufs[i]) == 0:
                        # render check: the zone's green must appear where the zone pose projects (first frame; the
                        # hand is still beside the body). Guards against visuals that do not follow the sim state.
                        exp = project_head(zpos[i])
                        got, n_green = green_centroid(images["head"][i])
                        zone_check[i] = {"expected_px": None if exp is None else [round(float(v), 1) for v in exp],
                                         "green_px": None if got is None else [round(float(v), 1) for v in got],
                                         "green_pixels": n_green,
                                         "err_px": None if (exp is None or got is None)
                                         else round(float(np.linalg.norm(exp - got)), 1)}
                    bufs[i].add(row, {c: images[c][i] for c in cams} if args.record is not None else None)
                    t_ep[i] += step_dt
            robot.set_joint_effort_target(torch.tensor(tau, dtype=torch.float32, device=dev), joint_ids=arm_ids)
            # velocity feed-forward dq* = finite difference of the motor targets (the teleop controller's dq*,
            # docs/TELEOP.md 2.1): the PD then damps the tracking error, not the motion itself
            if args.dq_ff == "on":
                dq_ff = (act - act_prev) / step_dt
                robot.set_joint_velocity_target(torch.tensor(dq_ff, dtype=torch.float32, device=dev), joint_ids=arm_ids)
            act_prev = act.copy()
            obs, rew, term, trunc, extras = env.step(torch.tensor(act, dtype=torch.float32, device=dev))
            stepped += 1
            stats["step_wall_s"].append(time.perf_counter() - ts)
            done = (term | trunc).cpu().numpy()
            if done.any():
                tm = uw.termination_manager
                t_succ = tm.get_term("success").cpu().numpy()
                t_drop = tm.get_term("block_dropped").cpu().numpy()
                t_out = tm.get_term("time_out").cpu().numpy()
                q22_new = robot.data.joint_pos[:, motor_ids].cpu().numpy().astype(np.float64)
                for i in np.nonzero(done)[0].tolist():
                    if not active[i]:
                        continue
                    # feed.current[i] was ALREADY advanced by the reset inside env.step -> use the record we keep
                    p = ep_placement[i]
                    reason = "success" if t_succ[i] else ("block_dropped" if t_drop[i] else ("time_out" if t_out[i]
                                                                                          else "other"))
                    arr = bufs[i].arrays()
                    fb = arr["observation.block_pose"][-1, :2] if len(bufs[i]) else np.full(2, np.nan)
                    rec = {"pid": p.pid, "split": p.split, "side": p.side, "success": bool(t_succ[i]),
                           "termination": reason, "frames": len(bufs[i]), "episode_s": round(t_ep[i], 3),
                           "wall_s": round(time.perf_counter() - wall_ep[i], 2),
                           "final_block_to_zone_m": float(np.linalg.norm(fb - np.asarray(p.zone_xy))),
                           "initial_block_to_zone_m": float(np.linalg.norm(np.asarray(p.block_xy) - np.asarray(p.zone_xy))),
                           "policy": args.policy, "blank_frames": dict(blank_frames[i]), "zone_check": zone_check[i]}
                    if args.policy == "scripted":
                        rec.update(n_reapproach=pols[i].n_reapproach, phase_events=pols[i].events,
                                   final_phase=pols[i].phase)
                    results.append(rec)
                    print(f"[episode] env {i} {p.pid} side={p.side} success={rec['success']} ({reason}) "
                          f"frames={rec['frames']} final_dist={rec['final_block_to_zone_m']:.3f} m", flush=True)
                    if args.record is not None:
                        meta = dict(rec)
                        meta.update(placement=p.to_dict(), instruction=LAYOUT.instruction, fps=LAYOUT.control_hz,
                                    provenance=provenance, schema="dropbear-tabletop-raw-episode-v1")
                        ep_dir = args.record / args.split / f"{p.pid}_{args.policy}"
                        eio.write_episode(ep_dir, bufs[i], meta, LAYOUT.control_hz, video)
                    # the env already reset env i with the next placement (or the default when exhausted)
                    nxt = feed.current.get(i)
                    if nxt is None or nxt.pid == "default":
                        active[i] = False
                    else:
                        start_episode(i, q22_new[i])
            if stepped % 100 == 0:
                sw = stats["step_wall_s"][-100:]
                print(f"[collect] step {stepped} done {len(results)}/{len(placements)} "
                      f"succ {sum(r['success'] for r in results)} step_wall p50 {1e3 * float(np.median(sw)):.0f} ms "
                      f"video queue {video.pending() if video else 0}", flush=True)
    finally:
        if video is not None:
            print("[collect] waiting for video encoding ...", flush=True)
            video.close()
    n = len(results)
    ns = sum(r["success"] for r in results)
    sw = np.asarray(stats["step_wall_s"])
    summary = {
        "schema": "dropbear-tabletop-collect-summary-v1", "provenance": provenance,
        "episodes": n, "successes": ns, "success_rate": (ns / n) if n else None,
        "by_side": {s: {"n": sum(1 for r in results if r["side"] == s),
                        "success": sum(1 for r in results if r["side"] == s and r["success"])} for s in ("left", "right")},
        "terminations": {k: sum(1 for r in results if r["termination"] == k)
                         for k in ("success", "block_dropped", "time_out", "other")},
        "episode_s_success_mean": float(np.mean([r["episode_s"] for r in results if r["success"]])) if ns else None,
        "ik_err_max_m": stats["ik_err_max"], "steps": stepped,
        "step_wall_ms": {"p50": float(np.median(sw) * 1e3), "p95": float(np.percentile(sw, 95) * 1e3)} if len(sw) else None,
        "wall_s": round(time.perf_counter() - t_start, 1), "num_envs": n_env,
        "videos": {"encoded": len(video.reports) if video else 0, "errors": video.errors if video else [],
                   "reports": video.reports if video else []},
        "incomplete": sum(active) > 0,
        "reset_info": getattr(uw, "tabletop_reset_info", None),
        "blank_frames": blank_stats,
        "camera_stats": cam_stats,
        "results": results,
    }
    out = args.summary or (_REPO / "logs" / "tabletop" / f"collect_{args.policy}_{args.split}_{run_stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[collect] SUMMARY {ns}/{n} success ({summary['success_rate']}) -> {out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        from dropbear_wbc.isaac.launch import close_app_and_exit

        close_app_and_exit(app, rc)
