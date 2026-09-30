"""Play (and optionally export / record) a Dropbear tracking policy.

Adapted from ``whole_body_tracking/scripts/rsl_rl/play.py`` (MIT): local checkpoint + motion file (no WandB),
bounded ``--steps`` for headless runs, ``--video`` records an mp4 headless (enables cameras), ``--export``
writes ``policy.pt`` / ``policy.onnx`` / ``policy_motion.onnx`` / ``policy.json`` next to the checkpoint.
``--clean_video`` (demo_eval, 2026-09-24) records demo-quality mp4s instead: no debug markers, neutral light, a
tracking 3/4 camera plus fixed front camera from the SAME rollout (``dropbear_wbc.tasks.tracking.demo_render``).
A JSON summary (tracking errors, terminations, finiteness) is written to ``<run>/play_<stamp>.json``.

    python tools/gpu_lock_run.py --log logs/robot_task/play_x.log -- C:/isaac-sim/python.bat -u scripts/play.py \
        --motion_file data/motions/<source>/<clip>.npz --load_run <run_dir_name> --export --video --headless
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

parser = argparse.ArgumentParser(description="Play / export / record a Dropbear tracking policy.")
parser.add_argument("--task", default="Dropbear-Tracking-Flat-Play-v0")
parser.add_argument("--motion_file", type=Path, default=None, help="contract NPZ (single-clip tasks)")
parser.add_argument("--motion_library", type=Path, default=None,
                    help="motion-library manifest (library tasks, Dropbear-Tracking-Library-Play-v0 / -Future-Play-v0; "
                    "added 2026-09-24 by multiclip): env i plays clip i %% N from frame 0; the summary has per-clip results; "
                    "--export writes a runtime-reference export (export_library.py, no policy_motion.onnx)")
parser.add_argument("--play_clips", default=None, help="library: comma list of clip names to play (env i -> i %% len)")
parser.add_argument("--checkpoint", type=Path, default=None, help="explicit model_*.pt path")
parser.add_argument("--load_run", default=None, help="run dir name under <log_root>/<experiment> (default: latest)")
parser.add_argument("--log_root", type=Path, default=_REPO / "logs" / "rsl_rl")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=500, help="policy steps to run (50 Hz)")
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, metavar=("POS", "VEL"))
parser.add_argument("--telemetry", type=Path, default=None,
                    help="per-motor telemetry NPZ of env 0 (tools/render_actuator_dashboard.py); with --clean_video one "
                         "row per video frame")
parser.add_argument("--actuator_profile", default=None,
                    help="legacy / hw_v1 / hw_v1_cad; default: the run's run_info.json value (legacy if absent)")
parser.add_argument("--export", action="store_true", help="write ONNX / TorchScript / JSON sidecar")
parser.add_argument("--export_dir", type=Path, default=None, help="default: <checkpoint dir>/exported")
parser.add_argument("--video", action="store_true", help="record an mp4 (headless: sets --enable_cameras)")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--clean_video", action="store_true",
                    help="demo render: all debug markers off, neutral lighting, own cameras (see --video_cams), one mp4 "
                    "per camera of env 0 for the first --video_length steps (sets --enable_cameras; use --num_envs 1)")
parser.add_argument("--video_cams", default="track,front", help="clean video cameras: track, front, side (comma list)")
parser.add_argument("--video_res", default="1280x720", help="clean video resolution WxH (e.g. 1920x1080)")
parser.add_argument("--video_dir", type=Path, default=None, help="clean video output dir (default <run>/videos/clean)")
parser.add_argument("--video_tag", default=None, help="clean video file prefix (default <clip>_<checkpoint>)")
parser.add_argument("--video_caption", default=None,
                    help="burned-in caption; a literal '\\n' separates up to 3 lines (default: clip, run, solver, eval "
                    "mode)")
parser.add_argument("--record_rollout", type=Path, default=None,
                    help="save env 0's SIMULATED state after every policy step (plus the initial state) as a contract "
                    "motion NPZ (all joints, all body link poses, COM velocities; meta.source 'policy rollout', "
                    "meta.not_a_reference) for a kinematic re-render with scripts/render_reference.py -- the glitch-free "
                    "render path (demo_eval, 2026-09-24). Store it under logs/, never under data/motions/")
parser.add_argument("--live_script", default=None,
                    help="LIVE reference (dropbear_wbc.isaac.live_reference): comma list 't_s:clip.npz'; each clip is "
                    "spliced into the running reference at that sim time (aligned, cross-faded); the pose is held after "
                    "the last clip. The --motion_file clip plays first. Single-clip tasks only")
parser.add_argument("--live_dir", type=Path, default=None,
                    help="LIVE reference: watch this folder and splice every new *.npz dropped into it (text-to-motion "
                    "hook)")
parser.add_argument("--live_blend_s", type=float, default=0.4, help="LIVE: cross-fade time into a new clip [s]")
parser.add_argument("--realtime", action="store_true",
                    help="pace the loop to wall-clock time (live viewing: sleeps when the sim runs ahead; it cannot "
                    "catch up when slower). The summary always reports the measured real-time factor ('timing'). One "
                    "robot runs about 5x faster with '--device cpu' than on the GPU pipeline (docs/TEXT_TO_MOTION.md)")
parser.add_argument("--continuous_loop", action="store_true",
                    help="evaluation: at the clip end only the reference clock wraps; the robot state is NOT rewritten "
                    "(default: the clip end writes the NPZ frame-0 state, i.e. every loop restarts from the reference)")
parser.add_argument("--perturbed", action="store_true",
                    help="evaluation under the TRAINING randomization: pushes, friction, base CoM, default-pose offsets "
                    "and observation noise (the reference still starts at frame 0 without RSI noise)")
parser.add_argument("--seed", type=int, default=None, help="env seed (randomization draws differ per env)")
parser.add_argument("--allow_rejected_motion", action="store_true",
                    help="play a motion whose verdict is 'rejected' (default: allowed only if the run's "
                    "motion_acceptance.json accepts these motion bytes)")
parser.add_argument("--lean", action="store_true",
                    help="preview speed: drop the reward terms (never read by play), compute only the actor's "
                    "observations, and the tracking metrics on every 5th step (mean_metrics become a 1-in-5 sample). "
                    "Physics and policy inputs are unchanged. Implied by --realtime")
parser.add_argument("--state_out", type=Path, default=None,
                    help="publish env 0's root pose + joint positions every policy step to this shared-memory file "
                    "(dropbear_wbc.isaac.state_share) for scripts/live_viewer.py, the separate GUI process")
parser.add_argument("--profile_out", type=Path, default=None,
                    help="cProfile the policy-step loop only and write the stats here (pstats format)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video and args.clean_video:
    parser.error("--video and --clean_video are exclusive")
if args.video or args.clean_video:
    args.enable_cameras = True
app = AppLauncher(args).app


def main() -> int:  # noqa: C901
    from dropbear_wbc.isaac.launch import assert_vendored_rsl_rl

    assert_vendored_rsl_rl()

    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.export import export_tracking_policy, latest_checkpoint, resolve_run_dir

    register()
    agent_cfg = load_entry_point(args.task, "rsl_rl_cfg_entry_point")
    if args.checkpoint is not None:
        ckpt = (args.checkpoint if args.checkpoint.is_absolute() else _REPO / args.checkpoint).resolve()
    else:
        ckpt = latest_checkpoint(resolve_run_dir(args.log_root / agent_cfg.experiment_name, args.load_run))
    run_dir = ckpt.parent
    print(f"[INFO] checkpoint: {ckpt}", flush=True)
    # Use the default pose (action offset) the policy was TRAINED with: the run's pinned calibration file, or the
    # legacy pose if the run had none. An explicit $DROPBEAR_CALIBRATION_JSON wins.
    pose_note = "env default (no run_info.json)"
    if "DROPBEAR_CALIBRATION_JSON" in os.environ:
        pose_note = f"explicit $DROPBEAR_CALIBRATION_JSON={os.environ['DROPBEAR_CALIBRATION_JSON']}"
    elif (run_dir / "run_info.json").is_file():
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        src = str(info.get("default_pose_source", ""))
        path = (info.get("default_pose_info") or {}).get("path")
        if src.startswith("legacy"):
            os.environ["DROPBEAR_CALIBRATION_JSON"] = "none"
            pose_note = "run trained with the legacy pose"
        elif path and Path(path).is_file():
            os.environ["DROPBEAR_CALIBRATION_JSON"] = path
            pose_note = f"run's calibration {path}"
        else:
            raise FileNotFoundError(f"run was trained with calibration {path!r} ({src}) which no longer exists")
    print(f"[INFO] default pose: {pose_note}", flush=True)
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    is_library = hasattr(env_cfg.commands.motion, "manifest")
    if is_library != (args.motion_library is not None) or (not is_library and args.motion_file is None):
        raise ValueError(f"{args.task}: library tasks need --motion_library, single-clip tasks --motion_file")
    src = args.motion_library if is_library else args.motion_file
    motion = (src if src.is_absolute() else _REPO / src).resolve()
    if is_library:
        env_cfg.commands.motion.manifest = str(motion)
        if args.play_clips:
            env_cfg.commands.motion.play_clips = [c.strip() for c in args.play_clips.split(",") if c.strip()]
    else:
        env_cfg.commands.motion.motion_file = str(motion)
    if args.num_envs is not None:
        env_cfg.set_num_envs(args.num_envs)
    if args.solver_iters is not None:
        env_cfg.set_solver_iterations(*args.solver_iters)
    profile = args.actuator_profile
    if profile is None:
        info_path = run_dir / "run_info.json"
        profile = (json.loads(info_path.read_text(encoding="utf-8")).get("actuator_profile") if info_path.is_file()
                   else None) or "legacy"
    env_cfg.set_actuator_profile(profile)
    print(f"[INFO] actuator profile: {profile}", flush=True)
    # replay with the target clamp the run was trained with (docs/ISSUES.md #25)
    info_path = run_dir / "run_info.json"
    clamp = json.loads(info_path.read_text(encoding="utf-8")).get("target_clamp_margin_deg") if info_path.is_file() else None
    if clamp is not None:
        env_cfg.clamp_targets_to_limits(float(clamp))
        print(f"[INFO] motor targets clamped to the joint limits +- {float(clamp):g} deg", flush=True)
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    from dropbear_wbc.isaac.launch import sha256_file

    if is_library:
        from dropbear_wbc.tasks.tracking.motion_library import manifest_fingerprint

        motion_sha = manifest_fingerprint(motion)["sha256"]
    else:
        motion_sha = sha256_file(motion)
    acceptance = None
    acc_file = run_dir / "motion_acceptance.json"
    if acc_file.is_file():
        acc = json.loads(acc_file.read_text(encoding="utf-8"))
        if acc.get("motion_sha256") == motion_sha and acc.get("allow_rejected_motion"):
            acceptance = acc
    if args.allow_rejected_motion or acceptance is not None:
        env_cfg.commands.motion.allow_rejected_motion = True
    env_cfg.commands.motion.continuous_loop = bool(args.continuous_loop)
    if args.seed is not None:
        env_cfg.seed = args.seed
    randomization = "none (play config: no pushes, no friction/CoM/default-pose randomization, no obs noise)"
    if args.perturbed:
        train_cfg = load_entry_point(args.task.replace("-Play", ""), "env_cfg_entry_point")
        for name in ("push_robot", "physics_material", "add_joint_default_pos", "base_com"):
            setattr(env_cfg.events, name, getattr(train_cfg.events, name))
        env_cfg.observations.policy.enable_corruption = train_cfg.observations.policy.enable_corruption
        randomization = ("training-level: " + ", ".join(
            n for n in ("push_robot", "physics_material", "add_joint_default_pos", "base_com")
            if getattr(env_cfg.events, n) is not None) + f", obs corruption={env_cfg.observations.policy.enable_corruption}")

    if args.lean or args.realtime:
        for name, term in list(vars(env_cfg.rewards).items()):
            if term is not None and not name.startswith("_"):
                setattr(env_cfg.rewards, name, None)
    clean_render_changes = None
    if args.clean_video:
        from dropbear_wbc.tasks.tracking.demo_render import apply_clean_render

        clean_render_changes = apply_clean_render(env_cfg)
    if args.live_script or args.live_dir is not None:
        # live control: a tracking-error termination would reset the robot to the START of the live timeline; keep the
        # fall check (anchor orientation) and the time-out only
        env_cfg.terminations.ee_body_pos = None
        env_cfg.terminations.anchor_pos = None
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    video_dir = run_dir / "videos" / "play"
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
            name_prefix=f"play_{ckpt.stem}",
        )
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args.lean or args.realtime:
        # preview speed: only the actor's observation group (the critic group is training-only; the runner above was
        # built with it, so the checkpoint loads unchanged) and the tracking metrics on every 5th policy step
        om = env.unwrapped.observation_manager
        om.compute = lambda update_history=False: {"policy": om.compute_group("policy", update_history=update_history)}
        _mterm = env.unwrapped.command_manager.get_term("motion")
        _metrics_full, _metric_calls = _mterm._update_metrics, [0]

        def _sampled_metrics():
            if _metric_calls[0] % 5 == 0:
                _metrics_full()
            _metric_calls[0] += 1

        _mterm._update_metrics = _sampled_metrics

    summary: dict = {"checkpoint": str(ckpt), "motion_file": str(motion), "task": args.task, "num_envs": env.num_envs,
                     "default_pose_source": env_cfg.default_pose_source, "default_pose_note": pose_note}
    cmd_term = env.unwrapped.command_manager.get_term("motion")
    if is_library:  # per-env clip (env i -> play clip i % N); T per env
        env_clip = cmd_term._play_clip_ids.clone()
        env_T = cmd_term.motion.lengths[env_clip]
        T = int(env_T.max())
        m = cmd_term.motion
        wrap_jump = float(max((m.joint_pos[s + n - 1] - m.joint_pos[s]).abs().max()
                              for s, n in zip(m.starts.tolist(), m.lengths.tolist())))
    else:
        T = int(cmd_term.motion.time_step_total)
        wrap_jump = float((cmd_term.motion.joint_pos[-1] - cmd_term.motion.joint_pos[0]).abs().max())
    summary["eval_design"] = {
        "policy": "deterministic (inference policy, mean action)",
        "randomization": randomization,
        "seed": env_cfg.seed,
        "start": "frame 0 (start_at_zero), no RSI noise",
        "loop_mode": ("continuous: reference clock wraps, robot state carried over" if args.continuous_loop else
                      "reset_at_wrap: the clip end writes the NPZ frame-0 joint+root state (each loop is a fresh start)"),
        "clip_frames": T,
        "wrap_reference_motor_jump_rad": wrap_jump,
        "independent_samples_note": ("without randomization every env runs the same deterministic trajectory: the "
                                     "env count is NOT a sample size (N = 1 nominal rollout)" if not args.perturbed else
                                     "envs differ only through the randomization draws"),
        "motion_validation": getattr(cmd_term.motion, "validation", None),
        "allow_rejected_motion": bool(env_cfg.commands.motion.allow_rejected_motion),
        "motion_acceptance": acceptance,
    }
    if args.export:
        out = args.export_dir or (run_dir / "exported")
        if is_library:
            from dropbear_wbc.tasks.tracking.export_library import export_library_policy

            sidecar = export_library_policy(env.unwrapped, runner, out, task=args.task, checkpoint=ckpt, manifest=motion)
        else:
            sidecar = export_tracking_policy(env.unwrapped, runner, out, task=args.task, checkpoint=ckpt,
                                             motion_file=motion)
        summary["export"] = {"dir": str(out), "parity": sidecar["dropbear_tracking"]["parity"],
                             "sha256": sidecar["dropbear_tracking"]["sha256"], "obs_dim": sidecar["obs_dim"],
                             "action_dim": len(sidecar["joint_names"]), "schema": sidecar["schema"]}
        print(json.dumps({"export": summary["export"]}), flush=True)

    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("motion")
    live = None
    if args.live_script or args.live_dir is not None:
        if is_library:
            raise SystemExit("--live_script/--live_dir need a single-clip task (play a library policy with the matching "
                             "single-clip task, e.g. Dropbear-Tracking-Flat-NoState-Play-v0)")
        from dropbear_wbc.isaac.live_reference import LiveReference, parse_live_script

        live = LiveReference(command, motion, blend_s=args.live_blend_s,
                             script=parse_live_script(args.live_script) if args.live_script else None,
                             watch_dir=args.live_dir)
    term_names = unwrapped.termination_manager.active_terms
    term_counts = {n: 0 for n in term_names}
    metric_keys = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot", "error_joint_pos")
    sums = {k: 0.0 for k in metric_keys}
    env_sums = {k: torch.zeros(unwrapped.num_envs, device=unwrapped.device) for k in metric_keys}
    finite = True
    # loop-closure gaps (27 excluded joints) and survival while the policy runs
    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures

    robot = unwrapped.scene["robot"]
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), unwrapped.device)
    gap_hist = []
    alive_steps = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    ended_lengths = []
    falls: list[dict] = []  # terminations that are not time-outs: env, global step, episode step, loop index, terms
    obs, _ = env.get_observations()
    steps = max(args.steps, args.video_length if (args.video or args.clean_video) else 0)
    recorder = None
    if args.clean_video:
        from dropbear_wbc.tasks.tracking.demo_render import DemoRecorder, parse_resolution

        solver = (env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                  env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count)
        mode = ("perturbed (training randomization)" if args.perturbed else "nominal") + (
            ", continuous loop" if args.continuous_loop else "")
        caption = args.video_caption if args.video_caption is not None else (
            f"Dropbear | {motion.stem} | policy {run_dir.name} ({ckpt.stem})\n"
            f"Isaac Lab PhysX {solver[0]}/{solver[1]} | {mode} | deterministic policy")
        recorder = DemoRecorder(
            env=unwrapped, out_dir=args.video_dir or (run_dir / "videos" / "clean"),
            tag=args.video_tag or f"{motion.stem}_{ckpt.stem}",
            cams=tuple(c.strip() for c in args.video_cams.split(",") if c.strip()),
            resolution=parse_resolution(args.video_res), caption=caption.replace("\\n", "\n"),
            reference_xy=(command.motion.root_pos_w[:, :2].detach().cpu().numpy()
                          if hasattr(command.motion, "root_pos_w") else None),
        )
    tel = None
    if args.telemetry is not None:
        from dropbear_wbc.isaac.telemetry import MotorTelemetry
        from dropbear_wbc.robots import dropbear_names as _N

        tel = MotorTelemetry(robot, env_id=0, anchor_body=_N.ANCHOR_BODY)
        _cs = unwrapped.scene.sensors.get("contact_forces") if hasattr(unwrapped.scene, "sensors") else None
        _fid = ([_cs.body_names.index(b) for b in _N.FOOT_BODIES] if _cs is not None and
                all(b in _cs.body_names for b in _N.FOOT_BODIES) else None)

        def _feet_contact():
            """(left, right) foot in contact (> 1 N on the sole plate or ankle cross) for env 0, or None."""
            if _fid is None:
                return None
            f = _cs.data.net_forces_w[0, _fid].norm(dim=-1) > 1.0
            return torch.stack([f[0] | f[1], f[2] | f[3]])
    rollout_obs, rollout_act = [], []
    roll: dict[str, list] | None = None
    if args.record_rollout is not None:
        roll = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                                "body_ang_vel_w", "closure_residual_m")}
        origin0 = unwrapped.scene.env_origins[0].detach().cpu()

        def _record_state() -> None:
            d = robot.data
            roll["joint_pos"].append(d.joint_pos[0].detach().cpu().clone())
            roll["joint_vel"].append(d.joint_vel[0].detach().cpu().clone())
            roll["body_pos_w"].append(d.body_link_pos_w[0].detach().cpu() - origin0)
            roll["body_quat_w"].append(d.body_link_quat_w[0].detach().cpu().clone())
            roll["body_lin_vel_w"].append(d.body_com_lin_vel_w[0].detach().cpu().clone())
            roll["body_ang_vel_w"].append(d.body_com_ang_vel_w[0].detach().cpu().clone())
            roll["closure_residual_m"].append(float(mon.worst(d.body_link_pos_w, d.body_link_quat_w)[0]))

        _record_state()
    # wall-clock accounting: PhysX (sim.step) vs the rest of env.step (actuator models, managers) vs policy vs script
    timing = {"physics_s": 0.0, "policy_s": 0.0, "env_step_s": 0.0, "sleep_s": 0.0}
    _sim_step = unwrapped.sim.step

    def _timed_sim_step(*a, **k):
        t = time.perf_counter()
        r = _sim_step(*a, **k)
        timing["physics_s"] += time.perf_counter() - t
        return r

    unwrapped.sim.step = _timed_sim_step
    proc_hints = None
    if args.realtime:
        from dropbear_wbc.isaac.launch import prefer_performance_cores

        proc_hints = prefer_performance_cores()
        print(f"[INFO] realtime: process hints {proc_hints}", flush=True)
    share = None
    if args.state_out is not None:
        from dropbear_wbc.isaac.state_share import StateWriter

        share = StateWriter(args.state_out if args.state_out.is_absolute() else _REPO / args.state_out,
                            list(robot.joint_names), list(robot.body_names))
        _origin0 = unwrapped.scene.env_origins[0]

        def _publish(step_n: int) -> None:
            d = robot.data
            share.write(step_n * unwrapped.step_dt, (d.root_pos_w[0] - _origin0).cpu().numpy(),
                        d.root_quat_w[0].cpu().numpy(), d.joint_pos[0].cpu().numpy(),
                        (d.body_link_pos_w[0] - _origin0).cpu().numpy(), d.body_link_quat_w[0].cpu().numpy())

        _publish(0)
    prof = None
    if args.profile_out is not None:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
    t_loop = time.perf_counter()
    for step_i in range(steps):
        if args.realtime:
            ahead = step_i * unwrapped.step_dt - (time.perf_counter() - t_loop)
            if ahead > 0:
                time.sleep(ahead)
                timing["sleep_s"] += ahead
        if live is not None:
            live.poll(step_i * unwrapped.step_dt)
        with torch.inference_mode():
            t0 = time.perf_counter()
            actions = policy(obs)
            if args.export and len(rollout_obs) < 64:  # real (sim) observations for the offline parity check
                rollout_obs.append(obs[:1].detach().cpu().clone())
                rollout_act.append(actions[:1].detach().cpu().clone())
            t1 = time.perf_counter()
            obs, _, dones, _ = env.step(actions)
            timing["policy_s"] += t1 - t0
            timing["env_step_s"] += time.perf_counter() - t1
        if share is not None:
            _publish(step_i + 1)
        if recorder is not None and step_i < args.video_length:
            recorder.capture()
        if tel is not None and (not args.clean_video or step_i < args.video_length):
            tel.record(contact=_feet_contact())
        if roll is not None:
            _record_state()
        finite &= bool(torch.isfinite(obs).all()) and bool(torch.isfinite(actions).all())
        gap_hist.append(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w).cpu())
        alive_steps += 1
        d = dones.bool()
        if live is not None and bool(d[0]):
            live.restart("fall" if bool(unwrapped.termination_manager.terminated[0]) else "time-out")
        if bool(d.any()):
            ended_lengths += alive_steps[d].tolist()
            fell = d & unwrapped.termination_manager.terminated.bool()
            for e in torch.nonzero(fell).flatten().tolist():
                a = int(alive_steps[e])
                t_env = int(env_T[e]) if is_library else T
                falls.append({"env": e, "step": len(gap_hist), "episode_step": a, "loop": (a - 1) // t_env,
                              "terms": [n for n in term_names if bool(unwrapped.termination_manager.get_term(n)[e])]})
            alive_steps[d] = 0
        for n in term_names:
            term_counts[n] += int((unwrapped.termination_manager.get_term(n) & dones.bool()).sum())
        for k in metric_keys:
            sums[k] += float(command.metrics[k].mean())
            if is_library:
                env_sums[k] += command.metrics[k].detach()
    wall = time.perf_counter() - t_loop
    if prof is not None:
        prof.disable()
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        prof.dump_stats(str(args.profile_out))
    unwrapped.sim.step = _sim_step
    sim_s = steps * unwrapped.step_dt
    summary["timing"] = {
        "device": str(unwrapped.device), "realtime_paced": bool(args.realtime), "lean": bool(args.lean or args.realtime),
        "process_hints": proc_hints, "sim_s": round(sim_s, 3),
        "wall_s": round(wall, 3), "real_time_factor": round(sim_s / max(wall - timing["sleep_s"], 1e-9), 3),
        "ms_per_policy_step": {k.removesuffix("_s"): round(1e3 * v / max(steps, 1), 2) for k, v in timing.items()}
        | {"total_unpaced": round(1e3 * (wall - timing["sleep_s"]) / max(steps, 1), 2),
           "budget": round(1e3 * unwrapped.step_dt, 2)},
        "note": "physics = sim.step calls (PhysX only, all decimation substeps); env_step includes physics plus the "
                "actuator models, scene updates, observations/rewards/commands; the remainder is this script's bookkeeping (closure monitor, "
                "metrics, telemetry, recording). real_time_factor excludes --realtime sleeps",
    }
    print(json.dumps({"timing": summary["timing"]}), flush=True)
    if roll is not None:
        import numpy as np

        out_npz = args.record_rollout if args.record_rollout.is_absolute() else _REPO / args.record_rollout
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        ref_meta = dict(getattr(command.motion, "arrays_meta", {}) or {})
        meta = {
            "schema": "dropbear-motion-npz-v1", "status": "ok", "source": "policy rollout (scripts/play.py --record_rollout)",
            "not_a_reference": True, "note": "simulated state of env 0 after each policy step (frame 0 = initial state); "
            "for kinematic re-rendering only, never for training",
            "checkpoint": str(ckpt), "reference_motion_file": str(motion), "task": args.task,
            "solver_iterations": [env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                                  env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count],
            "continuous_loop": bool(args.continuous_loop), "perturbed": bool(args.perturbed), "seed": env_cfg.seed,
            "usd_sha256": ref_meta.get("usd_sha256"), "authored_ankle_tierods": ref_meta.get("authored_ankle_tierods"),
            "falls_env0": [f for f in falls if f["env"] == 0][:50],
        }
        np.savez_compressed(
            out_npz, fps=np.float32(1.0 / unwrapped.step_dt),
            **{k: torch.stack(v).numpy().astype(np.float32) for k, v in roll.items() if k != "closure_residual_m"},
            closure_residual_m=np.asarray(roll["closure_residual_m"], dtype=np.float32),
            joint_names=np.asarray(robot.joint_names), body_names=np.asarray(robot.body_names),
            motor_names=np.asarray(list(command.cfg.joint_names)),
            meta=np.asarray(json.dumps(meta)))
        summary["record_rollout"] = {"npz": str(out_npz), "frames": len(roll["joint_pos"])}
    if live is not None:
        summary["live"] = {"events": live.events, "blend_s": live.blend_s, "lead_frames": live.lead}
    if args.export and rollout_obs:
        out = args.export_dir or (run_dir / "exported")
        torch.save({"obs": torch.cat(rollout_obs), "actions": torch.cat(rollout_act),
                    "note": "env 0 policy observations/actions from the play rollout (live inference policy)"},
                   out / "parity_rollout.pt")
        summary["export"]["parity_rollout_samples"] = len(rollout_obs)
    gaps = torch.stack(gap_hist).flatten() if gap_hist else torch.zeros(1)
    q = torch.quantile(gaps.float(), torch.tensor([0.5, 0.95, 0.99]))
    summary.update(
        steps=steps,
        finite=finite,
        solver_iterations=[
            env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
        ],
        closure_gap_m={"median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(gaps.max()),
                       "frac_over_3mm": float((gaps > 0.003).float().mean()),
                       "note": "worst of the 27 loop-closure anchor gaps per env and policy step (includes resets)"},
        episodes_ended=len(ended_lengths),
        ended_episode_len_steps=ended_lengths[:50],
        envs_never_reset=int((alive_steps == steps).sum()),
        termination_counts=term_counts,
        mean_metrics={k: v / steps for k, v in sums.items()},
        falls={
            "total": len(falls),
            "one_step_episodes": sum(1 for f in falls if f["episode_step"] <= 1),
            "per_env": {str(e): sum(1 for f in falls if f["env"] == e) for e in range(unwrapped.num_envs)},
            "by_loop": {str(k): sum(1 for f in falls if f["loop"] == k) for k in sorted({f["loop"] for f in falls})},
            "envs_with_a_fall": len({f["env"] for f in falls}),
            "first_fall_episode_step_per_env": {str(e): min(f["episode_step"] for f in falls if f["env"] == e)
                                                 for e in sorted({f["env"] for f in falls})},
            "events": falls[:200],
        },
        note="Tracking quality numbers only; finite simulation is not success.",
    )
    if is_library:  # per-clip breakdown (env i played clip env_clip[i] from frame 0)
        per_clip = {}
        for c, name in enumerate(cmd_term.clip_names):
            envs = [e for e in range(unwrapped.num_envs) if int(env_clip[e]) == c]
            if not envs:
                continue
            ev = [f for f in falls if f["env"] in envs]
            per_clip[name] = {
                "envs": envs, "frames": int(cmd_term.motion.lengths[c]),
                "mean_metrics": {k: float(env_sums[k][envs].mean()) / steps for k in metric_keys},
                "falls": len(ev), "one_step_falls": sum(1 for f in ev if f["episode_step"] <= 1),
                "first_fall_episode_step": min((f["episode_step"] for f in ev), default=None)}
        summary["library"] = {k: cmd_term.library_info.get(k) for k in ("name", "sha256", "num_clips", "all_accepted")}
        summary["per_clip"] = per_clip
    if args.video:
        env.close()
        summary["videos"] = [str(p) for p in sorted(video_dir.glob("*.mp4"))]
    if recorder is not None:
        summary["clean_video"] = recorder.describe() | {"outputs": recorder.close(), "render_cfg": clean_render_changes,
                                                          "recorded_steps": min(steps, args.video_length)}
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if tel is not None:
        tp = args.telemetry if args.telemetry.is_absolute() else _REPO / args.telemetry
        tel.save(tp, dt=unwrapped.step_dt, meta={"actuator_profile": profile, "checkpoint": str(ckpt),
                                                 "task": args.task, "motion": str(motion)})
        summary["telemetry"] = str(tp)
    out_json = run_dir / f"play_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"play_summary": summary, "summary_file": str(out_json)}), flush=True)
    if not args.video:
        env.close()
    ok = finite and all(math.isfinite(v) for v in summary["mean_metrics"].values())
    return 0 if ok else 2


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
