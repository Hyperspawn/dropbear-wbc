"""Evaluate / export / record a Dropbear velocity-command walking policy (``Dropbear-Velocity-Flat-Play-v0``).

Evaluation design: every env gets a FIXED command for the whole rollout (no resampling, no standing envs), assigned
round-robin from ``--scenarios`` (default: stand, forward 0.3 / 0.5, backward 0.2, sideways +-0.2, turn +-0.5,
forward+turn). The nominal Play config has no randomization and the policy is deterministic, so envs with the same
scenario are identical (N = 1 per scenario); ``--perturbed`` keeps the training randomization (friction, base mass,
reset yaw/xy + serial-motor noise, observation noise) so envs differ (``--seed``). Per scenario (after a
``--warmup_s`` window, only while the env has not fallen): falls (any non-timeout termination; the first one ends the
env's evaluation), time of first fall, measured mean planar CoM velocity in the yaw frame and yaw rate vs command,
touchdowns per foot per second, single/double-support and flight fractions, anchor height, worst loop-closure gap.

``--export`` writes ``policy.pt/.onnx/.json`` (``dropbear_wbc.tasks.locomotion.export``) to ``<run>/exported``;
``--video`` records an mp4 of env 0 following ``--video_script`` (stand -> forward -> turn -> sideways -> faster).

    python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/play_x.log -- C:/isaac-sim/python.bat -u \
        scripts/play_locomotion.py --load_run <run dir> --steps 750 --export --headless
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

DEFAULT_SCENARIOS = ("stand:0,0,0;fwd03:0.3,0,0;fwd05:0.5,0,0;back02:-0.2,0,0;left02:0,0.2,0;right02:0,-0.2,0;"
                     "turnL05:0,0,0.5;turnR05:0,0,-0.5;fwd03_turn03:0.3,0,0.3")
DEFAULT_VIDEO_SCRIPT = "0:0,0,0;3:0.3,0,0;8:0.3,0,0.4;11:0,0.2,0;14:0.5,0,0;18:0,0,0"

parser = argparse.ArgumentParser(description="Evaluate / export / record a Dropbear velocity policy.")
parser.add_argument("--task", default="Dropbear-Velocity-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, default=None, help="explicit model_*.pt path")
parser.add_argument("--load_run", default=None, help="run dir name under <log_root>/<experiment> (default: latest)")
parser.add_argument("--log_root", type=Path, default=_REPO / "logs" / "rsl_rl")
parser.add_argument("--experiment_name", default="dropbear_velocity")
parser.add_argument("--num_envs", type=int, default=None, help="default: 2 per scenario")
parser.add_argument("--steps", type=int, default=750, help="policy steps (50 Hz)")
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="'name:vx,vy,wz;...'")
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, metavar=("POS", "VEL"))
parser.add_argument("--perturbed", action="store_true", help="keep the training randomization")
parser.add_argument("--actuator_profile", default=None,
                    help="ACTUATOR_PROFILES key; default: the loaded run's run_info.json value ('legacy' if absent)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--export", action="store_true")
parser.add_argument("--export_dir", type=Path, default=None, help="default: <run>/exported")
parser.add_argument("--video", action="store_true", help="record env 0 following --video_script (sets --enable_cameras)")
parser.add_argument("--video_script", default=DEFAULT_VIDEO_SCRIPT, help="'t_s:vx,vy,wz;...' (piecewise constant)")
parser.add_argument("--clean_video", action="store_true",
                    help="demo-quality mp4 of env 0 following --video_script with demo_eval's renderer "
                         "(tasks/tracking/demo_render.py: no debug markers, neutral light, tracking camera)")
parser.add_argument("--video_cams", default="track", help="clean video cameras: track, front, side, feet (comma list)")
parser.add_argument("--video_res", default="1280x720", help="clean video resolution WxH")
parser.add_argument("--video_caption", default=None, help="clean video caption (default: policy + solver + profile)")
parser.add_argument("--out", type=Path, default=None, help="summary JSON (default <run>/play_<stamp>.json)")
parser.add_argument("--telemetry", type=Path, default=None,
                    help="per-motor telemetry NPZ of --telemetry_env (dashboard: tools/render_actuator_dashboard.py)")
parser.add_argument("--telemetry_env", type=int, default=0)
parser.add_argument("--terrain", choices=("auto", "flat", "rough"), default="auto",
                    help="auto (default): the terrain the run was trained on (run_info 'terrain'); flat / rough override "
                    "(docs/TERRAIN.md)")
parser.add_argument("--clamp_targets", action="store_true",
                    help="clamp motor targets to the joint limits (automatic for runs trained with --gait_shaping)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video and args.clean_video:
    parser.error("--video and --clean_video are exclusive")
if args.video or args.clean_video:
    args.enable_cameras = True
app = AppLauncher(args).app


def _parse_scenarios(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    out = []
    for item in text.split(";"):
        if item.strip():
            name, vals = item.split(":")
            v = tuple(float(x) for x in vals.split(","))
            out.append((name.strip(), (v[0], v[1], v[2])))
    return out


def _parse_script(text: str) -> list[tuple[float, tuple[float, float, float]]]:
    return sorted((float(t), v) for t, v in _parse_scenarios(text))


def main() -> int:  # noqa: C901
    from dropbear_wbc.isaac.launch import assert_vendored_rsl_rl, sha256_file

    assert_vendored_rsl_rl()

    import gymnasium as gym
    import torch

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401
    from dropbear_wbc.tasks.locomotion.mdp.rewards import foot_groups
    from dropbear_wbc.tasks.locomotion.runner import LocomotionOnPolicyRunner
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.export import latest_checkpoint, resolve_run_dir

    scenarios = _parse_scenarios(args.scenarios)
    script = _parse_script(args.video_script) if (args.video or args.clean_video) else None
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    agent_cfg = load_entry_point(args.task, "rsl_rl_cfg_entry_point")
    run_dir = resolve_run_dir(args.log_root / args.experiment_name, args.load_run)
    ckpt = args.checkpoint or latest_checkpoint(run_dir)
    run_dir = Path(ckpt).parent
    profile = args.actuator_profile
    if profile is None:
        infos = sorted(run_dir.glob("run_info*.json"))
        profile = "legacy"
        if infos:
            profile = json.loads(infos[0].read_text(encoding="utf-8")).get("actuator_profile") or "legacy"
    env_cfg.set_actuator_profile(profile)
    print(f"[play_locomotion] actuator profile: {profile}", flush=True)
    # a run trained with gait shaping clamps the motor targets to the joint limits: evaluate with the same clamp
    infos = sorted(run_dir.glob("run_info*.json"))
    trained_terrain = next((json.loads(i.read_text(encoding="utf-8")).get("terrain") for i in infos
                            if json.loads(i.read_text(encoding="utf-8")).get("terrain")), "flat")
    terrain = trained_terrain if args.terrain == "auto" else args.terrain
    if terrain == "rough":
        env_cfg.enable_rough_terrain()
        print("[play_locomotion] rough terrain (docs/TERRAIN.md)", flush=True)
    trained_clamped = any(json.loads(i.read_text(encoding="utf-8")).get("gait_shaping") for i in infos)
    if trained_clamped or args.clamp_targets:
        margins = [json.loads(i.read_text(encoding="utf-8")).get("target_clamp_margin_deg") for i in infos]
        margin = float(next((m for m in margins if m is not None), 3.0))  # runs before the option used 3 deg
        env_cfg.clamp_targets_to_limits(margin)
        print(f"[play_locomotion] motor targets clamped to the joint limits +- {margin:g} deg", flush=True)
    num_envs = args.num_envs or (1 if (args.video or args.clean_video) else 2 * len(scenarios))
    env_cfg.set_num_envs(num_envs)
    if args.solver_iters is not None:
        env_cfg.set_solver_iterations(*args.solver_iters)
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    if args.perturbed:
        train_cfg = load_entry_point("Dropbear-Velocity-Flat-v0", "env_cfg_entry_point")
        env_cfg.events.physics_material = train_cfg.events.physics_material
        env_cfg.events.add_base_mass = train_cfg.events.add_base_mass
        env_cfg.events.reset_robot.params.update({k: train_cfg.events.reset_robot.params[k]
                                                  for k in ("frame_mode", "pose_range", "leg_serial_noise", "arm_serial_noise")})
        env_cfg.observations.policy.enable_corruption = True
    if args.video or args.clean_video:
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.steps * 0.02 + 5.0)
    clean_render_changes = None
    if args.clean_video:
        from types import SimpleNamespace

        from dropbear_wbc.tasks.tracking.demo_render import apply_clean_render

        shim = not hasattr(env_cfg.commands, "motion")  # the renderer expects the tracking task's motion command term
        if shim:
            env_cfg.commands.motion = SimpleNamespace(debug_vis=False)
        clean_render_changes = apply_clean_render(env_cfg)
        if shim:
            delattr(env_cfg.commands, "motion")
    if args.seed is not None:
        env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    video_dir = run_dir / "videos" / "play"
    if args.video:
        env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), step_trigger=lambda s: s == 0,
                                       video_length=args.steps, disable_logger=True,
                                       name_prefix=f"velocity_{Path(ckpt).stem}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = LocomotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.restore_train_state = False
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    sensor = uw.scene["contact_forces"]
    cmd = uw.command_manager.get_term("base_velocity")
    dev = uw.device

    recorder = None
    if args.clean_video:
        from dropbear_wbc.tasks.tracking.demo_render import DemoRecorder, parse_resolution

        sol = (env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
               env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count)
        caption = args.video_caption if args.video_caption is not None else (
            f"Dropbear | velocity walking | policy {run_dir.name} ({Path(ckpt).stem})\n"
            f"Isaac Lab PhysX {sol[0]}/{sol[1]} | actuator profile {profile} | deterministic policy")
        recorder = DemoRecorder(env=uw, out_dir=run_dir / "videos" / "clean", tag=f"velocity_{Path(ckpt).stem}",
                                cams=tuple(c.strip() for c in args.video_cams.split(",") if c.strip()),
                                resolution=parse_resolution(args.video_res), caption=caption.replace("\\n", "\n"))
    summary: dict = {"tool": "scripts/play_locomotion.py", "task": args.task, "checkpoint": str(ckpt),
                     "checkpoint_sha256": sha256_file(ckpt), "num_envs": num_envs, "steps": args.steps,
                     "actuator_profile": profile,
                     "solver_iterations": [env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
                                           env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count],
                     "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    if args.export:
        from dropbear_wbc.tasks.locomotion.export import export_velocity_policy

        out_dir = args.export_dir or (run_dir / "exported")
        side = export_velocity_policy(uw, runner, out_dir, task="Dropbear-Velocity-Flat-v0", checkpoint=ckpt)
        summary["export"] = {"dir": str(out_dir), "parity": side["dropbear_velocity"]["parity"],
                             "obs_dim": side["obs_dim"]}
        print(json.dumps({"export": summary["export"]}), flush=True)

    # per-env fixed command
    if script is None:
        env_scn = [i % len(scenarios) for i in range(num_envs)]
        fixed = torch.tensor([scenarios[s][1] for s in env_scn], dtype=torch.float32, device=dev)
    else:
        env_scn = [0] * num_envs
        fixed = None

    def command_at(step: int) -> torch.Tensor:
        if fixed is not None:
            return fixed
        t = step * uw.step_dt
        v = script[0][1]
        for t0, val in script:
            if t >= t0:
                v = val
        return torch.tensor([v], dtype=torch.float32, device=dev).expand(num_envs, 3)

    groups = foot_groups(uw.reward_manager.get_term_cfg("feet_air_time").params["sensor_cfg"], 2).to(dev)
    anchor = robot.body_names.index(N.ANCHOR_BODY)
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), dev)
    warm = int(round(args.warmup_s / uw.step_dt))
    alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
    fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
    fall_term: list[str | None] = [None] * num_envs
    acc = {k: torch.zeros(num_envs, device=dev) for k in ("n", "vx", "vy", "wz", "evx", "evy", "ewz", "single", "double",
                                                          "flight", "anchor_z", "td_l", "td_r")}
    anchor_min = torch.full((num_envs,), 9.0, device=dev)
    start_xy = robot.data.root_pos_w[:, :2].clone()
    end_xy = start_xy.clone()
    worst_gap = 0.0
    prev_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=dev)
    finite = True
    # per-motor torque / speed samples (last physics substep of each policy step, live envs after warm-up):
    # "clipped" = the actuator model cut the PD torque (effort limit or, for hw_* profiles, the torque-speed line)
    motor_cols = [(N.MOTOR_NAMES.index(name), act, k) for act in robot.actuators.values()
                  for k, name in enumerate(act.joint_names) if name in N.MOTOR_NAMES]
    motor_ids = robot.find_joints(list(N.MOTOR_NAMES), preserve_order=True)[0]
    tq_samples, clip_samples, vel_samples = [], [], []
    peak_lim = torch.zeros(len(N.MOTOR_NAMES), device=dev)
    for i, act, k in motor_cols:
        peak_lim[i] = act.effort_limit[0, k]
    tel = None
    if args.telemetry is not None:
        from dropbear_wbc.isaac.telemetry import MotorTelemetry

        tel = MotorTelemetry(robot, env_id=args.telemetry_env, anchor_body=N.ANCHOR_BODY)
    obs, _ = env.get_observations()
    trace = []
    for step in range(args.steps):
        c = command_at(step)
        cmd.vel_command_b[:] = c
        obs = env.get_observations()[0] if step == 0 else obs
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        cmd.vel_command_b[:] = c
        if recorder is not None:
            recorder.capture()
        finite &= bool(torch.isfinite(obs).all()) and bool(torch.isfinite(robot.data.root_state_w).all())
        terminated = uw.termination_manager.terminated
        newly = terminated & alive
        for e in torch.nonzero(newly).flatten().tolist():
            fall_step[e] = step
            fall_term[e] = next((n for n in uw.termination_manager.active_terms
                                 if bool(uw.termination_manager.get_term(n)[e])), "?")
        # stats for envs still alive (before this step's fall)
        live = alive & ~terminated
        vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
        wz = robot.data.root_ang_vel_w[:, 2]
        ct = sensor.data.current_contact_time[:, groups]
        in_contact = (ct > 0.0).any(dim=-1)
        n_c = in_contact.sum(dim=1)
        touchdown = in_contact & ~prev_contact
        prev_contact = in_contact
        if tel is not None:
            tel.record(cmd=c[args.telemetry_env], contact=in_contact[args.telemetry_env])
        if step >= warm:
            m = live.float()
            acc["n"] += m
            acc["vx"] += m * vel_yaw[:, 0]
            acc["vy"] += m * vel_yaw[:, 1]
            acc["wz"] += m * wz
            acc["evx"] += m * (vel_yaw[:, 0] - c[:, 0]).abs()
            acc["evy"] += m * (vel_yaw[:, 1] - c[:, 1]).abs()
            acc["ewz"] += m * (wz - c[:, 2]).abs()
            acc["single"] += m * (n_c == 1).float()
            acc["double"] += m * (n_c == 2).float()
            acc["flight"] += m * (n_c == 0).float()
            acc["anchor_z"] += m * robot.data.body_link_pos_w[:, anchor, 2]
            acc["td_l"] += m * touchdown[:, 0].float()
            acc["td_r"] += m * touchdown[:, 1].float()
            anchor_min = torch.where(live, torch.minimum(anchor_min, robot.data.body_link_pos_w[:, anchor, 2]), anchor_min)
            worst_gap = max(worst_gap, float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)[live].max())
                            if bool(live.any()) else 0.0)
            if bool(live.any()):
                tau = torch.zeros(num_envs, len(N.MOTOR_NAMES), device=dev)
                clip = torch.zeros_like(tau, dtype=torch.bool)
                for i, act, k in motor_cols:
                    tau[:, i] = act.applied_effort[:, k]
                    clip[:, i] = (act.computed_effort[:, k] - act.applied_effort[:, k]).abs() > 0.01 * peak_lim[i]
                tq_samples.append(tau[live].abs())
                clip_samples.append(clip[live])
                vel_samples.append(robot.data.joint_vel[live][:, motor_ids].abs())
        end_xy = torch.where(live.unsqueeze(-1), robot.data.root_pos_w[:, :2], end_xy)
        alive &= ~terminated
        if step % 50 == 0 or step == args.steps - 1:
            trace.append({"t_s": round((step + 1) * uw.step_dt, 2), "alive": int(alive.sum()),
                          "cmd0": [round(float(x), 2) for x in c[0]],
                          "vel_yaw0": [round(float(x), 3) for x in vel_yaw[0, :2]] + [round(float(wz[0]), 3)],
                          "anchor_z0": round(float(robot.data.body_link_pos_w[0, anchor, 2]), 3)})

    dt = uw.step_dt
    per_env = []
    for e in range(num_envs):
        n = float(acc["n"][e])
        d = {"env": e, "scenario": scenarios[env_scn[e]][0] if script is None else "video_script",
             "command": [round(float(x), 3) for x in (fixed[e] if fixed is not None else command_at(0)[e])],
             "fell": bool(fall_step[e] >= 0), "fall_t_s": round(float(fall_step[e] + 1) * dt, 2) if fall_step[e] >= 0 else None,
             "fall_term": fall_term[e], "eval_s": round(n * dt, 2),
             "distance_m": round(float((end_xy[e] - start_xy[e]).norm()), 3)}
        if n > 0:
            d.update({k: round(float(acc[k][e]) / n, 4) for k in ("vx", "vy", "wz", "evx", "evy", "ewz", "single", "double",
                                                                   "flight", "anchor_z")})
            d["touchdowns_per_s"] = [round(float(acc["td_l"][e]) / (n * dt), 3), round(float(acc["td_r"][e]) / (n * dt), 3)]
            d["anchor_z_min"] = round(float(anchor_min[e]), 4)
        per_env.append(d)
    by_scn: dict = {}
    for d in per_env:
        g = by_scn.setdefault(d["scenario"], {"command": d["command"], "envs": 0, "falls": 0, "fall_t_s": []})
        g["envs"] += 1
        g["falls"] += int(d["fell"])
        if d["fell"]:
            g["fall_t_s"].append(d["fall_t_s"])
        for k in ("vx", "vy", "wz", "evx", "evy", "ewz", "single", "double", "flight", "anchor_z"):
            if k in d:
                g.setdefault(k, []).append(d[k])
    for g in by_scn.values():
        for k in ("vx", "vy", "wz", "evx", "evy", "ewz", "single", "double", "flight", "anchor_z"):
            if k in g:
                g[k] = round(sum(g[k]) / len(g[k]), 4)
    summary.update({
        "eval_design": {"perturbed": bool(args.perturbed), "seed": args.seed, "warmup_s": args.warmup_s,
                        "note": ("nominal: no randomization + deterministic policy -> envs with the same scenario are "
                                 "identical (N = 1 per scenario)") if not args.perturbed else "training randomization",
                        "fall": "any non-timeout termination (anchor height/tilt, non-foot contact); first one ends the "
                                "env's evaluation"},
        "finite": finite, "falls_total": int((fall_step >= 0).sum()), "worst_closure_gap_m": worst_gap,
        "by_scenario": by_scn, "per_env": per_env, "trace_env0": trace,
    })
    if tq_samples:
        tq, cl, vl = torch.cat(tq_samples), torch.cat(clip_samples).float(), torch.cat(vel_samples)
        summary["motor_torque"] = {
            "note": "|applied motor torque| and |joint speed| at the last physics substep of each policy step, live envs "
                    "after warm-up, all scenarios pooled; clip_frac = share of samples where the actuator model cut the "
                    "PD torque (peak limit, or the torque-speed line for hw_* profiles)",
            "samples": int(tq.shape[0]),
            "per_motor": {name: {"peak_limit_Nm": round(float(peak_lim[i]), 2), "max_Nm": round(float(tq[:, i].max()), 2),
                                 "p95_Nm": round(float(torch.quantile(tq[:, i], 0.95)), 2),
                                 "rms_Nm": round(float(tq[:, i].pow(2).mean().sqrt()), 2),
                                 "clip_frac": round(float(cl[:, i].mean()), 4),
                                 "max_speed_rad_s": round(float(vl[:, i].max()), 2),
                                 "p95_speed_rad_s": round(float(torch.quantile(vl[:, i], 0.95)), 2)}
                          for i, name in enumerate(N.MOTOR_NAMES)},
        }
    if args.video:
        summary["video_dir"] = str(video_dir)
        summary["video_script"] = args.video_script
    if recorder is not None:
        from dropbear_wbc.tasks.tracking.demo_render import contact_sheet

        outputs = recorder.close()
        sheets = {}
        for cam, o in outputs.items():
            try:
                sheets[cam] = str(contact_sheet(Path(o["mp4"]), Path(o["mp4"]).with_suffix(".contact.png")))
            except Exception as exc:  # noqa: BLE001
                sheets[cam] = f"failed: {exc}"
        summary["clean_video"] = recorder.describe() | {"outputs": outputs, "contact_sheets": sheets,
                                                          "render_cfg": clean_render_changes,
                                                          "video_script": args.video_script}
    if tel is not None:
        tp = args.telemetry if args.telemetry.is_absolute() else _REPO / args.telemetry
        tel.save(tp, dt=uw.step_dt, meta={"actuator_profile": profile, "checkpoint": str(ckpt), "task": args.task,
                                         "video_script": args.video_script if script is not None else None,
                                         "scenario": scenarios[env_scn[args.telemetry_env]][0] if script is None else
                                         "video_script", "env": args.telemetry_env})
        summary["telemetry"] = str(tp)
    out = args.out or (run_dir / f"play_{_dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    out = out if out.is_absolute() else _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out), "falls_total": summary["falls_total"], "finite": finite,
                      "by_scenario": by_scn}), flush=True)
    env.close()
    return 0 if finite else 2


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
