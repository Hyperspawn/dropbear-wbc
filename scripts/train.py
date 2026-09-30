"""Train a Dropbear motion-tracking policy (BeyondMimic port) with the vendored RSL-RL 2.3.3.

Adapted from ``whole_body_tracking/scripts/rsl_rl/train.py`` (MIT): local ``--motion_file`` instead of the
WandB registry, logs under ``logs/rsl_rl/<experiment>/<timestamp>_<run_name>``,
no ``isaaclab_tasks`` import, Windows-safe paths. Run under the GPU lock::

    python tools/gpu_lock_run.py --log logs/robot_task/train_x.log -- C:/isaac-sim/python.bat -u scripts/train.py \
        --task Dropbear-Tracking-Flat-v0 --motion_file data/motions/<source>/<clip>.npz --num_envs 1024 --headless
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train a Dropbear tracking policy with RSL-RL 2.3.3.")
parser.add_argument("--task", default="Dropbear-Tracking-Flat-v0")
parser.add_argument("--motion_file", type=Path, default=None, help="contract NPZ (dropbear-motion-npz-v1); single-clip tasks")
parser.add_argument("--motion_library", type=Path, default=None,
                    help="motion-library manifest (dropbear-motion-library-v1; added 2026-09-24 by multiclip) for the "
                    "library tasks Dropbear-Tracking-Library-v0 / -Future-v0 (docs/CONTRACTS.md 5.3)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", default="")
parser.add_argument("--experiment_name", default=None)
parser.add_argument("--log_root", type=Path, default=_REPO / "logs" / "rsl_rl")
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, metavar=("POS", "VEL"))
parser.add_argument("--hw_regularizers", action="store_true",
                    help="hw_* profiles: over-rated-torque (thermal) + torque-rate penalties (enable_hw_regularizers)")
parser.add_argument("--knee_stop_penalty", type=float, default=0.0,
                    help="with --hw_regularizers: weight (< 0) on the knee cranks sitting within 3 deg of a hard stop "
                         "(docs/ISSUES.md #19); pass as --knee_stop_penalty=-10")
parser.add_argument("--target_clamp_margin_deg", type=float, default=None,
                    help="clip the motor position targets to the hard limits +- this [deg] (0 recommended; default "
                         "unclipped as every earlier run; docs/ISSUES.md #25)")
parser.add_argument("--action_rate_weight", type=float, default=None,
                    help="override the action_rate_l2 weight (default -0.1): smoother 50 Hz targets, less motor jitter "
                         "(docs/ISSUES.md #28)")
parser.add_argument("--feet_slide_penalty", type=float, default=0.0,
                    help="with --hw_regularizers: weight (< 0) on sole planar speed while in ground contact (touchdown "
                         "skid, toe scuff; docs/ISSUES.md #20); pass as --feet_slide_penalty=-1")
parser.add_argument("--actuator_profile", default=None,
                    help="legacy (default) or a real-actuator twin profile hw_v1 / hw_v1_cad (docs/ACTUATORS.md 11)")
parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
parser.add_argument("--resume", action="store_true", help="resume from --load_run/--checkpoint")
parser.add_argument("--load_run", default=None, help="run dir name under <log_root>/<experiment>")
parser.add_argument("--checkpoint", default=None, help="checkpoint file name (default: latest model_*.pt)")
parser.add_argument("--continue_run", action="store_true",
                    help="with --resume: keep writing into the loaded run directory (chunked training: metrics.jsonl "
                    "appends, checkpoints continue the iteration count)")
parser.add_argument("--allow_rejected_motion", action="store_true",
                    help="EXPLORATORY: train on a motion whose meta.status or <clip>.validation.json verdict is 'rejected'. "
                    "Written to <run>/motion_acceptance.json (resumed chunks inherit it) and to run_info.json")
parser.add_argument("--accept_reason", default="", help="why the rejected motion is used (with --allow_rejected_motion)")
parser.add_argument("--restore_train_state", choices=("auto", "on", "off"), default="auto",
                    help="restore the learning rate and the adaptive-sampling statistics saved in the checkpoint on "
                    "--resume. auto = on, except for a --continue_run into a run whose first chunk predates the "
                    "persistence (its run_info.json has no train_state_persistence), which keeps its original behaviour")
parser.add_argument("--video", action="store_true", help="record training videos (needs --enable_cameras)")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.motion_file is None) == (args.motion_library is None):
    parser.error("give exactly one of --motion_file (single-clip task) or --motion_library (library task)")
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app


def main() -> int:
    from dropbear_wbc.isaac.launch import assert_vendored_rsl_rl, sha256_file

    rsl_rl_path = assert_vendored_rsl_rl()

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.export import latest_checkpoint
    from dropbear_wbc.tasks.tracking.runner import DropbearOnPolicyRunner

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    register()
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    agent_cfg = load_entry_point(args.task, "rsl_rl_cfg_entry_point")
    is_library = hasattr(env_cfg.commands.motion, "manifest")
    if is_library != (args.motion_library is not None):
        raise ValueError(f"{args.task}: library tasks need --motion_library, single-clip tasks --motion_file")
    src = args.motion_library if is_library else args.motion_file
    motion = (src if src.is_absolute() else (_REPO / src)).resolve()
    if is_library:
        env_cfg.commands.motion.manifest = str(motion)
    else:
        env_cfg.commands.motion.motion_file = str(motion)
    if args.num_envs is not None:
        env_cfg.set_num_envs(args.num_envs)
    if args.solver_iters is not None:
        env_cfg.set_solver_iterations(*args.solver_iters)
    if args.actuator_profile is not None:
        env_cfg.set_actuator_profile(args.actuator_profile)
    if args.target_clamp_margin_deg is not None:
        env_cfg.clamp_targets_to_limits(args.target_clamp_margin_deg)
    if args.action_rate_weight is not None:
        env_cfg.rewards.action_rate_l2.weight = float(args.action_rate_weight)
    if args.hw_regularizers:
        env_cfg.enable_hw_regularizers(knee_stop=float(args.knee_stop_penalty), feet_slide=float(args.feet_slide_penalty))
    elif args.knee_stop_penalty or args.feet_slide_penalty:
        raise SystemExit("--knee_stop_penalty / --feet_slide_penalty need --hw_regularizers")
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.experiment_name:
        agent_cfg.experiment_name = args.experiment_name
    if args.logger:
        agent_cfg.logger = args.logger
    agent_cfg.run_name = args.run_name
    agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device

    log_root = (args.log_root / agent_cfg.experiment_name).resolve()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.continue_run:
        if not (args.resume and args.load_run):
            raise ValueError("--continue_run needs --resume --load_run")
        log_dir = log_root / args.load_run
    else:
        log_dir = log_root / (f"{stamp}_{args.run_name}" if args.run_name else stamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] log dir: {log_dir}", flush=True)

    # -- rejected-motion acceptance (fail closed unless explicit; a resumed run inherits its recorded decision)
    if is_library:  # content identity of the library (clip bytes, names, weights), see motion_library.manifest_fingerprint
        from dropbear_wbc.tasks.tracking.motion_library import manifest_fingerprint

        motion_sha = manifest_fingerprint(motion)["sha256"]
    else:
        motion_sha = sha256_file(motion)
    acceptance_file = log_dir / "motion_acceptance.json"
    acceptance = None
    if args.continue_run and acceptance_file.is_file():
        acceptance = json.loads(acceptance_file.read_text(encoding="utf-8"))
        if acceptance.get("motion_sha256") != motion_sha:
            raise ValueError(f"{acceptance_file} accepts motion sha {str(acceptance.get('motion_sha256'))[:12]}..., "
                             f"but {motion} has sha {motion_sha[:12]}...")
    if args.allow_rejected_motion and acceptance is None:
        acceptance = {"allow_rejected_motion": True, "source": "scripts/train.py --allow_rejected_motion",
                      "reason": args.accept_reason, "motion_file": str(motion), "motion_sha256": motion_sha,
                      "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        acceptance_file.write_text(json.dumps(acceptance, indent=1) + "\n", encoding="utf-8")
    if acceptance is not None and acceptance.get("allow_rejected_motion"):
        env_cfg.commands.motion.allow_rejected_motion = True
    # -- train-state restore decision (learning rate + adaptive sampling across chunk restarts)
    first_info = log_dir / "run_info.json"
    restore = args.restore_train_state == "on"
    if args.restore_train_state == "auto":
        restore = True
        if args.continue_run and first_info.is_file():
            restore = bool(json.loads(first_info.read_text(encoding="utf-8")).get("train_state_persistence", False))

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(log_dir / "videos" / "train"),
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = DropbearOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.restore_train_state = restore
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        resume_dir = log_root / args.load_run if args.load_run else None
        if resume_dir is None:
            raise ValueError("--resume needs --load_run")
        ckpt = resume_dir / args.checkpoint if args.checkpoint else latest_checkpoint(resume_dir)
        print(f"[INFO] resuming from {ckpt}", flush=True)
        runner.load(str(ckpt))

    params_tag = f"_resume_{stamp}" if args.continue_run else ""
    dump_yaml(str(log_dir / "params" / f"env{params_tag}.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / f"agent{params_tag}.yaml"), agent_cfg)
    dump_pickle(str(log_dir / "params" / f"env{params_tag}.pkl"), env_cfg)
    dump_pickle(str(log_dir / "params" / f"agent{params_tag}.pkl"), agent_cfg)
    unwrapped = env.unwrapped
    motion_term = unwrapped.command_manager.get_term("motion")
    run_info = {
        "task": args.task,
        "motion_file": str(motion),
        "motion_sha256": motion_sha,
        "motion_meta_status": motion_term.motion.arrays_meta.get("status"),
        "motion_validation": getattr(motion_term.motion, "validation", None),
        "allow_rejected_motion": bool(env_cfg.commands.motion.allow_rejected_motion),
        "motion_acceptance": acceptance,
        "motion_library": getattr(motion_term, "library_info", None),
        "train_state_persistence": restore,
        "restore_train_state": restore,
        "train_state_restore": runner.train_state_report,
        "usd_path": env_cfg.scene.robot.spawn.usd_path,
        "default_pose_source": env_cfg.default_pose_source,
        "actuator_profile": getattr(env_cfg, "actuator_profile", "legacy"),
        "hw_regularizers": bool(args.hw_regularizers),
        "knee_stop_penalty": float(args.knee_stop_penalty),
        "feet_slide_penalty": float(args.feet_slide_penalty),
        "target_clamp_margin_deg": args.target_clamp_margin_deg,
        "action_rate_weight": args.action_rate_weight,
        "default_pose_info": dict(env_cfg.default_pose_info),
        "default_motor_pos": {n: env_cfg.scene.robot.init_state.joint_pos[n] for n in MOTOR_NAMES},
        "num_envs": env_cfg.scene.num_envs,
        "solver_iterations": [
            env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
        ],
        "obs_dim_policy": int(unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "obs_dim_critic": int(unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "action_dim": int(unwrapped.action_manager.total_action_dim),
        "npz_default_pose_max_abs_diff": unwrapped.command_manager.get_term("motion").npz_default_pose_max_abs_diff,
        "rsl_rl_path": rsl_rl_path,
        "max_iterations": agent_cfg.max_iterations,
    }
    if args.continue_run:
        run_info["resumed_from"] = str(ckpt)
        run_info["resume_start_iteration"] = int(runner.current_learning_iteration)
    (log_dir / f"run_info{params_tag}.json").write_text(json.dumps(run_info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"run_info": run_info}), flush=True)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(json.dumps({"status": "trained", "log_dir": str(log_dir), "last_checkpoint": str(latest_checkpoint(log_dir))}), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
