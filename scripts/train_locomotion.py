"""Train a Dropbear H1-style velocity-command walking policy (``Dropbear-Velocity-Flat-v0``) with RSL-RL 2.3.3.

Counterpart of ``scripts/train.py`` (tracking) for tasks without a motion command. Logs under
``logs/rsl_rl/dropbear_velocity/<timestamp>_<run_name>``. Run under the GPU lock::

    python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/train_x.log -- C:/isaac-sim/python.bat -u \
        scripts/train_locomotion.py --num_envs 2048 --max_iterations 20 --run_name smoke --headless

Chunked long runs: ``tools/run_chunked_locomotion.py`` (``--resume --load_run <dir> --continue_run`` per chunk; the
learning rate and the command-curriculum ranges are restored from the checkpoint).
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

parser = argparse.ArgumentParser(description="Train a Dropbear velocity-command walking policy with RSL-RL 2.3.3.")
parser.add_argument("--task", default="Dropbear-Velocity-Flat-v0")
parser.add_argument("--stand_npz", type=Path, default=None, help="reset NPZ (default data/motions/smoke/dropbear_static_stand.npz)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", default="")
parser.add_argument("--experiment_name", default=None)
parser.add_argument("--log_root", type=Path, default=_REPO / "logs" / "rsl_rl")
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, metavar=("POS", "VEL"))
parser.add_argument("--pushes", action="store_true", help="enable interval pushes (robustness stage)")
parser.add_argument("--actuator_profile", default=None,
                    help="flat_env_cfg.ACTUATOR_PROFILES key (default: the task default); recorded in run_info.json")
parser.add_argument("--gait_v2", action="store_true", help="--gait_shaping + the stance-knee extension term (-1.0)")
parser.add_argument("--gait_v3", action="store_true",
                    help="--gait_v2 + min feet width (-2), near-hard-limit (-0.5), hip roll/yaw deviation -0.5")
parser.add_argument("--feet_width_weight", type=float, default=None,
                    help="override the gait-v3 feet_lateral weight (-2): crossing on rough terrain (docs/ISSUES.md #29)")
parser.add_argument("--action_rate_weight", type=float, default=None,
                    help="override the action_rate_l2 weight (-0.01): target jitter (docs/ISSUES.md #29)")
parser.add_argument("--terrain", choices=("flat", "rough"), default="flat",
                    help="rough: blind curriculum terrain sized for Dropbear (flat_env_cfg.enable_rough_terrain)")
parser.add_argument("--target_clamp_margin_deg", type=float, default=3.0,
                    help="gait-shaped runs: motor targets clipped to the hard limits +- this [deg]; 3 pressed the knees "
                         "into their stops at 26 N*m (docs/ISSUES.md #23), use 0 for new runs")
parser.add_argument("--gait_shaping", action="store_true",
                    help="knee flexion in swing + swing clearance + torque-rate terms (flat_env_cfg.enable_gait_shaping)")
parser.add_argument("--thermal_penalty", type=float, default=None,
                    help="hw_* profiles: weight of sum relu(|tau| - rated)^2 (e.g. -2e-4; docs/ACTUATORS.md 11)")
parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
parser.add_argument("--resume", action="store_true", help="resume from --load_run/--checkpoint")
parser.add_argument("--load_run", default=None, help="run dir name under <log_root>/<experiment>")
parser.add_argument("--checkpoint", default=None, help="checkpoint file name (default: latest model_*.pt)")
parser.add_argument("--continue_run", action="store_true",
                    help="with --resume: keep writing into the loaded run directory (chunked training)")
parser.add_argument("--restore_train_state", choices=("on", "off"), default="on",
                    help="restore learning rate + command-curriculum ranges from the checkpoint on --resume")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main() -> int:
    from dropbear_wbc.isaac.launch import assert_vendored_rsl_rl, sha256_file

    rsl_rl_path = assert_vendored_rsl_rl()

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401  (registers the gym ids)
    from dropbear_wbc.tasks.locomotion.mdp.curriculums import command_ranges_dict
    from dropbear_wbc.tasks.locomotion.runner import LocomotionOnPolicyRunner
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.export import latest_checkpoint

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    agent_cfg = load_entry_point(args.task, "rsl_rl_cfg_entry_point")
    if args.stand_npz is not None:
        env_cfg.set_stand_npz(args.stand_npz if args.stand_npz.is_absolute() else _REPO / args.stand_npz)
    if args.num_envs is not None:
        env_cfg.set_num_envs(args.num_envs)
    if args.solver_iters is not None:
        env_cfg.set_solver_iterations(*args.solver_iters)
    if args.pushes:
        env_cfg.enable_pushes()
    if args.actuator_profile is not None:
        env_cfg.set_actuator_profile(args.actuator_profile)
    if args.thermal_penalty is not None:
        env_cfg.enable_thermal_penalty(args.thermal_penalty)
    if args.gait_v3:
        # weights sized so the new penalties stay well below the tracking reward (at -20 / -5 / -1.0 the warm-started
        # run learned that falling early is cheaper than walking: episode length 990 -> 49 in 70 iterations)
        env_cfg.enable_gait_shaping(knee_stance=-1.0, feet_width=-2.0, near_limit=-0.5, hip_deviation=-0.5,
                                    clamp_margin_deg=args.target_clamp_margin_deg)
    elif args.gait_shaping or args.gait_v2:
        env_cfg.enable_gait_shaping(knee_stance=-1.0 if args.gait_v2 else 0.0, clamp_margin_deg=args.target_clamp_margin_deg)
    if args.terrain == "rough":
        env_cfg.enable_rough_terrain()  # after gait shaping: re-points the swing-height term to the ground scan
    if args.feet_width_weight is not None:
        if getattr(env_cfg.rewards, "feet_lateral", None) is None:
            raise SystemExit("--feet_width_weight needs --gait_v3 (the feet_lateral term)")
        env_cfg.rewards.feet_lateral.weight = float(args.feet_width_weight)
    if args.action_rate_weight is not None:
        env_cfg.rewards.action_rate_l2.weight = float(args.action_rate_weight)
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

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = LocomotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.restore_train_state = args.restore_train_state == "on"
    runner.add_git_repo_to_log(__file__)
    ckpt = None
    if args.resume:
        if not args.load_run:
            raise ValueError("--resume needs --load_run")
        resume_dir = log_root / args.load_run
        ckpt = resume_dir / args.checkpoint if args.checkpoint else latest_checkpoint(resume_dir)
        print(f"[INFO] resuming from {ckpt}", flush=True)
        runner.load(str(ckpt))

    params_tag = f"_resume_{stamp}" if args.continue_run else ""
    dump_yaml(str(log_dir / "params" / f"env{params_tag}.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / f"agent{params_tag}.yaml"), agent_cfg)
    dump_pickle(str(log_dir / "params" / f"env{params_tag}.pkl"), env_cfg)
    dump_pickle(str(log_dir / "params" / f"agent{params_tag}.pkl"), agent_cfg)
    uw = env.unwrapped
    reset_term = uw.event_manager.get_term_cfg("reset_robot").func
    action_term = uw.action_manager.get_term("joint_pos")
    run_info = {
        "task": args.task,
        "stand_npz": env_cfg.stand_info.get("path"),
        "stand_npz_sha256": sha256_file(env_cfg.stand_info["path"]),
        "stand_info": dict(env_cfg.stand_info),
        "stand_validation": getattr(reset_term, "validation", None),
        "npz_default_pose_max_abs_diff": getattr(reset_term, "npz_default_pose_max_abs_diff", None),
        "usd_path": env_cfg.scene.robot.spawn.usd_path,
        "default_pose_source": env_cfg.default_pose_source,
        "default_pose_info": dict(env_cfg.default_pose_info),
        "default_motor_pos": {n: env_cfg.scene.robot.init_state.joint_pos[n] for n in MOTOR_NAMES},
        "actuator_profile": env_cfg.actuator_profile,
        "thermal_penalty": args.thermal_penalty,
        "gait_shaping": bool(args.gait_shaping or args.gait_v2 or args.gait_v3),
        "gait_v2": bool(args.gait_v2),
        "gait_v3": bool(args.gait_v3),
        "target_clamp_margin_deg": float(args.target_clamp_margin_deg),
        "terrain": args.terrain,
        "feet_width_weight": args.feet_width_weight,
        "action_rate_weight": args.action_rate_weight,
        "actuator_gains_sim": {n: {"kp": float(uw.scene["robot"].data.joint_stiffness[0, j]),
                                   "kd": float(uw.scene["robot"].data.joint_damping[0, j]),
                                   "effort": float(uw.scene["robot"].data.joint_effort_limits[0, j])}
                               for n, j in zip(MOTOR_NAMES, uw.scene["robot"].find_joints(list(MOTOR_NAMES), preserve_order=True)[0])},
        "action_joint_names": list(action_term._joint_names),
        "action_scale": [float(x) for x in action_term._scale[0].tolist()] if hasattr(action_term._scale, "tolist") else action_term._scale,
        "num_envs": env_cfg.scene.num_envs,
        "solver_iterations": [
            env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
        ],
        "pushes": env_cfg.events.push_robot is not None,
        "obs_dim_policy": int(uw.observation_manager.group_obs_dim["policy"][0]),
        "obs_dim_critic": int(uw.observation_manager.group_obs_dim["critic"][0]),
        "obs_terms_policy": list(uw.observation_manager.active_terms["policy"]),
        "action_dim": int(uw.action_manager.total_action_dim),
        "command_ranges_start": command_ranges_dict(uw),
        "command_limit_ranges": {k: list(getattr(env_cfg.commands.base_velocity.limit_ranges, k))
                                 for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "reward_weights": {n: float(uw.reward_manager.get_term_cfg(n).weight) for n in uw.reward_manager.active_terms},
        "termination_terms": list(uw.termination_manager.active_terms),
        "train_state_restore": runner.train_state_report,
        "rsl_rl_path": rsl_rl_path,
        "max_iterations": agent_cfg.max_iterations,
        "seed": agent_cfg.seed,
    }
    if args.continue_run:
        run_info["resumed_from"] = str(ckpt)
        run_info["resume_start_iteration"] = int(runner.current_learning_iteration)
    (log_dir / f"run_info{params_tag}.json").write_text(json.dumps(run_info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"run_info": run_info}), flush=True)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(json.dumps({"status": "trained", "log_dir": str(log_dir), "last_checkpoint": str(latest_checkpoint(log_dir)),
                      "command_ranges_end": command_ranges_dict(uw)}), flush=True)
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
