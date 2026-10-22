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
parser.add_argument("--push_velocity", type=float, default=None,
                    help="push strength: each push adds a uniform random velocity in +-this [m/s] per horizontal axis to "
                         "the whole robot (implies --pushes; default 0.3)")
parser.add_argument("--push_interval", type=float, nargs=2, default=None, metavar=("MIN_S", "MAX_S"),
                    help="seconds between pushes (implies --pushes; default 8 12)")
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
parser.add_argument("--stair_max_m", type=float, default=None, help="rough terrain: hardest step height [m] (default 0.10)")
parser.add_argument("--arm_action_scale", type=float, default=None,
                    help="arm action scale [rad per unit action] (default 0.1; legs 0.25-0.375): let the arms help balance")
parser.add_argument("--arm_swing_weight", type=float, default=None,
                    help="reward on natural counter-phase arm swing while moving (e.g. 0.5; enable_arm_swing)")
parser.add_argument("--arm_swing_k", type=float, default=0.45, help="shoulder / opposite-hip pitch ratio (human 0.40-0.55)")
parser.add_argument("--arm_gate_tilt_rate", type=float, default=None,
                    help="arm-style terms off while the base roll/pitch rate exceeds this [rad/s] and for "
                         "--arm_gate_hold_s after: natural when calm, free when shoved (e.g. 0.8)")
parser.add_argument("--arm_gate_hold_s", type=float, default=1.0)
parser.add_argument("--arm_abduction_weight", type=float, default=0.0,
                    help="L1 pull of the shoulder-roll motors to the standing pose (< 0, e.g. -0.3)")
parser.add_argument("--amp_weight", type=float, default=None,
                    help="adversarial motion prior: style reward from a discriminator trained against retargeted human "
                         "walking/stepping/balance clips (tasks/locomotion/amp.py; e.g. 1.0). Saved as amp_disc.pt in the run dir")
parser.add_argument("--amp_grad_penalty", type=float, default=10.0)
parser.add_argument("--symmetry", action="store_true",
                    help="left/right mirror data augmentation in PPO (tasks/locomotion/symmetry.py, calibration-derived "
                         "motor mirror): what the policy learns on one side it also learns on the other")
parser.add_argument("--clip_excess_arms", type=float, default=0.0,
                    help="weight (<= 0) on arm motor targets beyond the target clamp [rad] (mdp.target_beyond_clip): the session-10 arms sat past their stops on every step; pass as --clip_excess_arms=-1")
parser.add_argument("--clip_excess_legs", type=float, default=0.0, help="the same for the 12 leg motors")
parser.add_argument("--symmetry_mirror_loss", type=float, default=0.0,
                    help="with --symmetry: weight of rsl-rl's mirror loss, MSE(pi(mirror o), mirror pi(o)) (e.g. 1.0). "
                         "Augmentation alone keeps a symmetric policy symmetric but cannot repair an asymmetric warm "
                         "start: the mirrored samples reuse the original action's old log-prob and get clipped")
parser.add_argument("--arm_deviation_weight", type=float, default=None,
                    help="joint_deviation_arms weight (default -0.2; 0 removes it); pass as --arm_deviation_weight=-0.02")
parser.add_argument("--slope_max", type=float, default=None, help="rough terrain: steepest slope, rise/run (default 0.2)")
parser.add_argument("--ankle_roll_bind_deg", type=float, default=0.0,
                    help="> 0: clamp each ankle's calf-motor targets to semantic |roll| <= this - --ankle_roll_bind_margin_deg "
                         "(pinned tie rods bind at about 8; kinematics.ankle_bind). Applied after the per-motor clamp.")
parser.add_argument("--ankle_roll_bind_margin_deg", type=float, default=1.0, help="margin inside --ankle_roll_bind_deg")
parser.add_argument("--target_clamp_margin_deg", type=float, default=3.0,
                    help="gait-shaped runs: motor targets clipped to the hard limits +- this [deg]; 3 pressed the knees "
                         "into their stops at 26 N*m (docs/ISSUES.md #23), use 0 for new runs")
parser.add_argument("--gait_shaping", action="store_true",
                    help="knee flexion in swing + swing clearance + torque-rate terms (flat_env_cfg.enable_gait_shaping)")
parser.add_argument("--stop_model", action="store_true",
                    help="model the motor joints' end stops explicitly (stop loads measured, needed by --stop_load_penalty)")
parser.add_argument("--stop_load_penalty", type=float, default=0.0,
                    help="weight (< 0) on end-stop torque above a safe value (knee 40 N*m); pass as --stop_load_penalty=-5")
parser.add_argument("--impact_penalty", type=float, default=0.0,
                    help="weight (< 0) on foot contact force above 900 N per foot body; pass as --impact_penalty=-0.005")
parser.add_argument("--knee_band", default=None,
                    help="'stiffness,engage_deg': an elastic beside each knee motor (N*m/rad on the crank, crank angle)")
parser.add_argument("--knee_stop_penalty", type=float, default=0.0,
                    help="knee-crank hard-stop term (< 0, e.g. -10; pass as --knee_stop_penalty=-10): the terrain walkers "
                    "rested the knee on its flexion stop (docs/RESULTS.md, session 3)")
parser.add_argument("--thermal_penalty", type=float, default=None,
                    help="hw_* profiles: weight of sum relu(|tau| - rated)^2 (e.g. -2e-4; docs/archive/ACTUATORS_HISTORY.md 11)")
parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
parser.add_argument("--resume", action="store_true", help="resume from --load_run/--checkpoint")
parser.add_argument("--load_run", default=None, help="run dir name under <log_root>/<experiment>")
parser.add_argument("--checkpoint", default=None, help="checkpoint file name (default: latest model_*.pt)")
parser.add_argument("--continue_run", action="store_true",
                    help="with --resume: keep writing into the loaded run directory (chunked training)")
parser.add_argument("--robot_mass_kg", type=float, default=None,
                    help="measured total robot mass [kg]: re-centres the fixed plant mass and the add_base_mass band on "
                    "it. Default: unchanged. Recorded in run_info; --continue_run inherits it when not given")
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
    if args.pushes or args.push_velocity is not None or args.push_interval is not None:
        env_cfg.enable_pushes(**({} if args.push_velocity is None else {"velocity_xy": float(args.push_velocity)}),
                              **({} if args.push_interval is None else {"interval_s": tuple(args.push_interval)}))
    if args.actuator_profile is not None:
        env_cfg.set_actuator_profile(args.actuator_profile)
    robot_mass_kg = args.robot_mass_kg
    if robot_mass_kg is None and args.continue_run and args.load_run:  # a chunk keeps the plant mass of its run
        from dropbear_wbc.robots.mass_dr import run_robot_mass_kg

        prev = sorted(((args.log_root / (args.experiment_name or agent_cfg.experiment_name) / args.load_run)
                       .resolve()).glob("run_info*.json"))
        robot_mass_kg = run_robot_mass_kg(json.loads(f.read_text(encoding="utf-8")) for f in prev)
    if robot_mass_kg is not None:
        env_cfg.set_robot_mass(robot_mass_kg)
        print(f"[INFO] robot mass: {robot_mass_kg:.2f} kg", flush=True)
    if args.thermal_penalty is not None:
        env_cfg.enable_thermal_penalty(args.thermal_penalty)
    if args.arm_action_scale is not None or args.arm_deviation_weight is not None:
        env_cfg.set_arm_freedom(args.arm_action_scale, args.arm_deviation_weight)
    if args.arm_swing_weight:
        env_cfg.enable_arm_swing(args.arm_swing_weight, k=args.arm_swing_k, abduction=args.arm_abduction_weight,
                                 gate_tilt_rate=args.arm_gate_tilt_rate, gate_hold_s=args.arm_gate_hold_s)
    if args.gait_v3:
        # weights sized so the new penalties stay well below the tracking reward (at -20 / -5 / -1.0 the warm-started
        # run learned that falling early is cheaper than walking: episode length 990 -> 49 in 70 iterations)
        env_cfg.enable_gait_shaping(knee_stance=-1.0, feet_width=-2.0, near_limit=-0.5, hip_deviation=-0.5,
                                    clamp_margin_deg=args.target_clamp_margin_deg)
    elif args.gait_shaping or args.gait_v2:
        env_cfg.enable_gait_shaping(knee_stance=-1.0 if args.gait_v2 else 0.0, clamp_margin_deg=args.target_clamp_margin_deg)
    if args.clip_excess_arms or args.clip_excess_legs:  # after gait shaping, which sets the clamp
        env_cfg.enable_clip_excess_penalty(arms=args.clip_excess_arms, legs=args.clip_excess_legs)
    if args.ankle_roll_bind_deg > 0:  # after gait shaping, which sets the per-motor clamp
        env_cfg.enable_ankle_roll_bind(args.ankle_roll_bind_deg, args.ankle_roll_bind_margin_deg)
    if args.terrain == "rough":
        env_cfg.enable_rough_terrain(  # after gait shaping: re-points the swing-height term to the ground scan
            **({} if args.stair_max_m is None else {"stair_max_m": float(args.stair_max_m)}),
            **({} if args.slope_max is None else {"slope_max": float(args.slope_max)}))
    if args.knee_stop_penalty:
        env_cfg.enable_knee_stop_penalty(float(args.knee_stop_penalty))
    if args.stop_model or args.stop_load_penalty:
        env_cfg.enable_stop_model()
    if args.knee_band:
        k, q0 = (float(x) for x in args.knee_band.split(","))
        env_cfg.enable_knee_band(k, q0)
    if args.stop_load_penalty or args.impact_penalty:
        env_cfg.enable_safety_penalties(stop_load=float(args.stop_load_penalty), impact=float(args.impact_penalty))
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
    if args.symmetry:
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=args.symmetry_mirror_loss > 0,
            mirror_loss_coeff=float(args.symmetry_mirror_loss),
            data_augmentation_func="dropbear_wbc.tasks.locomotion.symmetry:compute_symmetric_states")
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
    if args.amp_weight:  # needs the run dir: the discriminator lives beside the policy and carries over between chunks
        env_cfg.enable_amp(args.amp_weight, ckpt_path=str(log_dir / "amp_disc.pt"), grad_penalty=args.amp_grad_penalty,
                           mirror=args.symmetry)  # symmetry augmentation needs a left/right symmetric style reward

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
        "robot_mass_kg": env_cfg.robot_mass_kg,
        "thermal_penalty": args.thermal_penalty,
        "knee_stop_penalty": float(args.knee_stop_penalty),
        "stop_model": bool(args.stop_model or args.stop_load_penalty), "stop_load_penalty": float(args.stop_load_penalty),
        "impact_penalty": float(args.impact_penalty), "knee_band": args.knee_band,
        "gait_shaping": bool(args.gait_shaping or args.gait_v2 or args.gait_v3),
        "gait_v2": bool(args.gait_v2),
        "gait_v3": bool(args.gait_v3),
        "target_clamp_margin_deg": float(args.target_clamp_margin_deg),
        "terrain": args.terrain, "stair_max_m": args.stair_max_m, "slope_max": args.slope_max,
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
        "push_velocity": args.push_velocity, "push_interval": args.push_interval,
        "arm_action_scale": args.arm_action_scale, "arm_deviation_weight": args.arm_deviation_weight,
        "arm_swing_weight": args.arm_swing_weight, "arm_swing_k": args.arm_swing_k,
        "arm_abduction_weight": args.arm_abduction_weight, "arm_gate_tilt_rate": args.arm_gate_tilt_rate,
        "arm_gate_hold_s": args.arm_gate_hold_s,
        "amp_weight": args.amp_weight, "amp_grad_penalty": args.amp_grad_penalty, "symmetry": bool(args.symmetry),
        "symmetry_mirror_loss": float(args.symmetry_mirror_loss),
        "clip_excess_arms": float(args.clip_excess_arms), "clip_excess_legs": float(args.clip_excess_legs),
        "ankle_roll_bind": ({"roll_limit_deg": float(args.ankle_roll_bind_deg), "margin_deg": float(args.ankle_roll_bind_margin_deg)}
                            if args.ankle_roll_bind_deg > 0 else None),
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
