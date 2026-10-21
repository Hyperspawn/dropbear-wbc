"""Push-recovery and terrain benchmark for a Dropbear velocity walker (docs/ROADMAP.md gate 4).

Two modes, one Isaac process each; every env runs one deterministic test case, and a case ends at its first fall
(any non-timeout termination: anchor height / tilt, non-foot contact).

* ``--mode push`` (flat ground). Cases = phase (``--phases``, e.g. standing and walking 0.5 m/s) x 8 push directions
  (robot yaw frame; 0 deg = from behind, 90 deg = from the robot's right, pushing it left) x push strength
  (``--push_dv``: the velocity change the push would give the whole robot, so it is size-independent; impulse =
  total mass x dv) x ``--offsets`` (push times inside the gait cycle). The push is a horizontal force on the base
  body's centre of mass for ``--push_s`` seconds. Survival = no fall within ``--after_s`` after the push.
* ``--mode terrain``. A grid of 8 m tiles: columns = flat, rough (random 0.45 m blocks), uphill, downhill, stairs up,
  stairs down; rows = ``--levels`` difficulty bands (stairs 2 -> ``--stair_max_m``, slopes 0 -> ``--slope_max``, blocks
  up to ``--block_max_m``). Every robot starts at its tile centre and walks forward at ``--speed`` for ``--walk_s``
  (8 s x 0.5 m/s = the 4 m to the tile edge). Survival = no fall; progress = distance along the command.

The policy is blind (no height scan), as in training. Output: one JSON (``--out``) with every case and a summary.

    python tools/gpu_lock_run.py --owner bench --log logs/bench/push.log -- C:/isaac-sim/python.bat -u \
        scripts/robustness_bench.py --mode push --checkpoint <model_*.pt> --stop_model --out logs/bench/push.json --headless
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Push-recovery and terrain benchmark for a Dropbear velocity walker.")
parser.add_argument("--mode", choices=("push", "terrain"), required=True)
parser.add_argument("--task", default="Dropbear-Velocity-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="model_*.pt of a velocity walker")
parser.add_argument("--actuator_profile", default=None, help="default: the run's run_info")
parser.add_argument("--robot_mass_kg", type=float, default=None,
                    help="measured total robot mass [kg] (see train.py); default: the run's run_info value (none: CAD + motors)")
parser.add_argument("--stop_model", action="store_true", help="explicit end-stop springs (policies trained with it)")
parser.add_argument("--knee_band", default=None, help="'stiffness,engage_deg' knee elastic (see train.py)")
parser.add_argument("--solver_iters", type=int, nargs=2, default=[32, 4], metavar=("POS", "VEL"))
parser.add_argument("--out", type=Path, required=True, help="result JSON")
# push mode
parser.add_argument("--phases", default="stand:0,0,0;walk:0.5,0,0", help="'name:vx,vy,wz;...'")
parser.add_argument("--directions", type=int, default=8)
parser.add_argument("--push_dv", default="0.2,0.4,0.6,0.8,1.0,1.2", help="push strengths as whole-body dv [m/s]")
parser.add_argument("--push_s", type=float, default=0.1, help="push duration [s]")
parser.add_argument("--push_at_s", type=float, default=3.0)
parser.add_argument("--offsets", default="0,0.25", help="extra push delays [s] (different gait phases)")
parser.add_argument("--after_s", type=float, default=5.0, help="survival window after the push [s]")
parser.add_argument("--repeats", type=int, default=1,
                    help="run every push cell N times, the push moved by k * --repeat_jitter_s (k = 0..N-1); the "
                    "summary adds majority max dv and the share of cells whose repeats disagree")
parser.add_argument("--repeat_jitter_s", type=float, default=0.02, help="push-time step between repeats [s] (1 policy step)")
# terrain mode
parser.add_argument("--levels", type=int, default=6)
parser.add_argument("--stair_max_m", type=float, default=0.20)
parser.add_argument("--slope_max", type=float, default=0.5, help="rise / run (0.5 = 26.6 deg)")
parser.add_argument("--block_max_m", type=float, default=0.12)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--walk_s", type=float, default=8.0)
# video: film one or a few cases in the live scene (terrain visible); falls are not terminated so they play out
parser.add_argument("--only", default=None,
                    help="keep only matching cases: push 'phase:dir_deg:dv:offset_s' (e.g. stand:0:0.6:0), terrain "
                         "'name:level' (e.g. stairs_down:1, level 0-based); several separated by ';'")
parser.add_argument("--clean_video", action="store_true", help="record env 0 (see --only) with the demo renderer")
parser.add_argument("--video_dir", type=Path, default=_REPO / "logs" / "reels" / "bench")
parser.add_argument("--video_tag", default="bench")
parser.add_argument("--video_cams", default="track", help="track, front, side, feet (comma list)")
parser.add_argument("--video_res", default="1280x720")
parser.add_argument("--caption", default="", help="burned-in caption (a literal \\n starts a new line)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.clean_video:
    args.enable_cameras = True
app = AppLauncher(args).app

TERRAIN_COLS = ("flat", "rough", "uphill", "downhill", "stairs_up", "stairs_down")


def _parse_cmds(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    out = []
    for item in text.split(";"):
        if item.strip():
            name, vals = item.split(":")
            v = [float(x) for x in vals.split(",")]
            out.append((name.strip(), (v[0], v[1], v[2])))
    return out


def _bench_terrain(levels: int, stair_max: float, slope_max: float, block_max: float):
    """One column per terrain type (curriculum rows = difficulty bands). Isaac Lab pyramids are highest at the
    centre, so a robot walking out from the centre goes DOWN a pyramid and UP an inverted one."""
    import isaaclab.terrains as tg
    from isaaclab.terrains import TerrainGeneratorCfg

    p = 1.0 / len(TERRAIN_COLS)
    subs = {
        "flat": tg.MeshPlaneTerrainCfg(proportion=p),
        "rough": tg.MeshRandomGridTerrainCfg(proportion=p, grid_width=0.45, grid_height_range=(0.01, block_max),
                                             platform_width=1.0),
        "uphill": tg.HfInvertedPyramidSlopedTerrainCfg(proportion=p, slope_range=(0.0, slope_max), platform_width=1.0,
                                                       border_width=0.25),
        "downhill": tg.HfPyramidSlopedTerrainCfg(proportion=p, slope_range=(0.0, slope_max), platform_width=1.0,
                                                 border_width=0.25),
        "stairs_up": tg.MeshInvertedPyramidStairsTerrainCfg(proportion=p, step_height_range=(0.02, stair_max),
                                                            step_width=0.35, platform_width=1.0, border_width=0.5),
        "stairs_down": tg.MeshPyramidStairsTerrainCfg(proportion=p, step_height_range=(0.02, stair_max), step_width=0.35,
                                                      platform_width=1.0, border_width=0.5),
    }
    return TerrainGeneratorCfg(size=(8.0, 8.0), border_width=20.0, num_rows=levels, num_cols=len(TERRAIN_COLS),
                               horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
                               curriculum=True, sub_terrains=subs)


def _band(lo: float, hi: float, level: int, levels: int) -> tuple[float, float]:
    """The parameter range of a curriculum row (difficulty in [level, level + 1) / levels)."""
    return lo + (hi - lo) * level / levels, lo + (hi - lo) * (level + 1) / levels


def main() -> int:  # noqa: C901
    from dropbear_wbc.isaac.launch import assert_vendored_rsl_rl, sha256_file

    assert_vendored_rsl_rl()

    import gymnasium as gym
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401
    from dropbear_wbc.tasks.locomotion.runner import LocomotionOnPolicyRunner
    from dropbear_wbc.tasks.registry import load_entry_point

    ckpt = args.checkpoint.resolve()
    run_dir = ckpt.parent
    # newest first: run_info.json is the run's first chunk; resumed chunks (run_info_resume_<stamp>.json) can change the
    # actuator profile, e.g. a run created on hw_v2i and continued on hw_v4
    infos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("run_info*.json"))][::-1]
    profile = args.actuator_profile or next((i.get("actuator_profile") for i in infos if i.get("actuator_profile")), "legacy")
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    agent_cfg = load_entry_point(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.set_actuator_profile(profile)
    from dropbear_wbc.robots.mass_dr import run_robot_mass_kg

    robot_mass_kg = args.robot_mass_kg if args.robot_mass_kg is not None else run_robot_mass_kg(infos[::-1])
    env_cfg.set_robot_mass(robot_mass_kg)
    if args.stop_model:
        env_cfg.enable_stop_model()
    if args.knee_band:
        env_cfg.enable_knee_band(*(float(x) for x in args.knee_band.split(",")))
    if any(i.get("gait_shaping") for i in infos):
        margin = float(next((i["target_clamp_margin_deg"] for i in infos if i.get("target_clamp_margin_deg") is not None), 3.0))
        env_cfg.clamp_targets_to_limits(margin)
    env_cfg.set_solver_iterations(*args.solver_iters)
    arm = {k: next((i[k] for i in infos if i.get(k) is not None), None) for k in ("arm_action_scale", "arm_deviation_weight")}
    if any(v is not None for v in arm.values()):  # the arm freedom the policy was trained with
        env_cfg.set_arm_freedom(arm["arm_action_scale"], arm["arm_deviation_weight"])

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    cases: list[dict] = []
    if args.mode == "push":
        from dropbear_wbc.tasks.locomotion.push_bench import push_cases

        phases = _parse_cmds(args.phases)
        dvs = [float(x) for x in args.push_dv.split(",")]
        offs = [float(x) for x in args.offsets.split(",")]
        cases = push_cases(phases, args.directions, dvs, offs, args.repeats, args.repeat_jitter_s)
        last = max(c["offset_s"] for c in cases)
        steps = int(round((args.push_at_s + last + args.push_s + args.after_s) / step_dt))
        env_cfg.scene.env_spacing = max(float(env_cfg.scene.env_spacing), 5.0)
    else:
        env_cfg.enable_rough_terrain()
        env_cfg.scene.terrain.terrain_generator = _bench_terrain(args.levels, args.stair_max_m, args.slope_max,
                                                                 args.block_max_m)
        env_cfg.curriculum.terrain_levels = None
        for col, name in enumerate(TERRAIN_COLS):
            for lv in range(args.levels):
                cases.append({"terrain": name, "col": col, "level": lv, "cmd": (args.speed, 0.0, 0.0)})
        steps = int(round(args.walk_s / step_dt))
    if args.only:
        keys = [k.strip() for k in args.only.split(";") if k.strip()]

        def _key(c):
            if args.mode == "push":
                from dropbear_wbc.tasks.locomotion.push_bench import case_key

                return case_key(c)
            return f"{c['terrain']}:{c['level']}"
        cases = [c for c in cases if _key(c) in keys]
        if not cases:
            raise SystemExit(f"--only matched no case: {args.only}")
    num_envs = len(cases)
    env_cfg.set_num_envs(num_envs)
    clean_render_changes = None
    if args.clean_video:  # let falls play out on camera; the clean demo lighting and floor
        from types import SimpleNamespace

        from dropbear_wbc.tasks.tracking.demo_render import apply_clean_render

        for term in ("anchor_height", "anchor_tilt", "non_foot_contact"):
            if getattr(env_cfg.terminations, term, None) is not None:
                setattr(env_cfg.terminations, term, None)
        env_cfg.commands.motion = SimpleNamespace(debug_vis=False)  # the renderer expects the tracking command term
        # terrain: a darker floor and softer light, or the white steps disappear on camera
        clean_render_changes = (apply_clean_render(env_cfg) if args.mode == "push" else
                                apply_clean_render(env_cfg, floor_rgb=(0.30, 0.33, 0.38), key_intensity=1500.0,
                                                   dome_intensity=450.0))
        delattr(env_cfg.commands, "motion")
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, steps * step_dt + 10.0)
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    env_cfg.sim.device = args.device
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = LocomotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.restore_train_state = False
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uw = env.unwrapped
    dev = uw.device
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("base_velocity")
    fixed = torch.tensor([c["cmd"] for c in cases], dtype=torch.float32, device=dev)

    if args.mode == "terrain":  # put every env on its own tile, then restart all envs there
        terr = uw.scene.terrain
        terr.terrain_levels[:] = torch.tensor([c["level"] for c in cases], device=dev)
        terr.terrain_types[:] = torch.tensor([c["col"] for c in cases], device=dev)
        terr.env_origins[:] = terr.terrain_origins[terr.terrain_levels, terr.terrain_types]
        env.reset()

    recorder = None
    if args.clean_video:
        from dropbear_wbc.tasks.tracking.demo_render import DemoRecorder, parse_resolution

        recorder = DemoRecorder(env=uw, out_dir=args.video_dir, tag=args.video_tag,
                                cams=tuple(c.strip() for c in args.video_cams.split(",") if c.strip()),
                                resolution=parse_resolution(args.video_res),
                                caption=args.caption.replace("\\n", "\n"))
    base = robot.body_names.index("world")  # the articulation root (pelvis chassis, 19.5 kg)
    mass = float(robot.root_physx_view.get_masses()[0].sum())
    push_steps = max(1, int(round(args.push_s / step_dt)))
    alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
    fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
    fall_term: list[str | None] = [None] * num_envs
    start_xy = robot.data.root_pos_w[:, :2].clone()
    progress = torch.zeros(num_envs, device=dev)
    tilt_max = torch.zeros(num_envs, device=dev)
    if args.mode == "push":
        p_start = torch.tensor([int(round((args.push_at_s + c["offset_s"]) / step_dt)) for c in cases], device=dev)
        f_mag = torch.tensor([mass * c["dv"] / (push_steps * step_dt) for c in cases], device=dev)
        ang = torch.tensor([math.radians(c["dir_deg"]) for c in cases], device=dev)
        # 0 deg = from behind (force along +x of the robot), 90 deg = from the right (force along +y, to the left)
        f_local = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * f_mag[:, None]
    pushing = False
    obs = env.get_observations()[0]
    for step in range(steps):
        cmd.vel_command_b[:] = fixed
        if args.mode == "push":
            on = (step >= p_start) & (step < p_start + push_steps) & (f_mag > 0)
            if bool(on.any()) or pushing:  # touch the wrench buffers only while some push is on (and once after)
                w = torch.zeros(num_envs, 3, device=dev)
                w[on] = quat_apply(yaw_quat(robot.data.root_quat_w)[on], f_local[on])
                robot.set_external_force_and_torque(w[:, None, :], torch.zeros(num_envs, 1, 3, device=dev),
                                                    body_ids=[base], is_global=True)
                pushing = bool(on.any())
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        cmd.vel_command_b[:] = fixed
        if recorder is not None:
            recorder.capture()
        terminated = uw.termination_manager.terminated
        newly = terminated & alive
        for e in torch.nonzero(newly).flatten().tolist():
            fall_step[e] = step
            fall_term[e] = next((n for n in uw.termination_manager.active_terms
                                 if bool(uw.termination_manager.get_term(n)[e])), "?")
        alive &= ~terminated
        g = quat_apply_inverse(robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(num_envs, 3))
        tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
        tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
        if args.mode == "terrain":
            d = robot.data.root_pos_w[:, :2] - start_xy
            progress = torch.where(alive, d[:, 0], progress)  # tiles start at yaw 0: the command is along +x

    if recorder is not None:
        print(json.dumps({"video": recorder.close()}, default=str), flush=True)
    results = []
    for e, c in enumerate(cases):
        r = dict(c, fell=bool(fall_step[e] >= 0), fall_s=float(fall_step[e]) * step_dt if fall_step[e] >= 0 else None,
                 fall_term=fall_term[e], tilt_max_deg=float(tilt_max[e]))
        if args.mode == "push":
            ps = (args.push_at_s + c["offset_s"])
            r["fell_before_push"] = bool(r["fell"] and r["fall_s"] < ps)
            r["survived"] = not r["fell"]
            r["impulse_ns"] = mass * c["dv"]
        else:
            r["progress_m"] = float(progress[e])
            r["survived"] = not r["fell"]
            band = {"stairs_up": (0.02, args.stair_max_m), "stairs_down": (0.02, args.stair_max_m),
                    "uphill": (0.0, args.slope_max), "downhill": (0.0, args.slope_max), "rough": (0.01, args.block_max_m)}
            if c["terrain"] in band:
                r["param_band"] = _band(*band[c["terrain"]], c["level"], args.levels)
        results.append(r)

    summary: dict = {}
    if args.mode == "push":
        from dropbear_wbc.tasks.locomotion.push_bench import push_summary

        summary = push_summary(results, [ph for ph, _ in _parse_cmds(args.phases)], args.directions)
        print(json.dumps({"mass_kg": mass, **summary}, indent=1), flush=True)
    else:
        for name in TERRAIN_COLS:
            rows = sorted((r for r in results if r["terrain"] == name), key=lambda r: r["level"])
            summary[name] = [{"level": r["level"], "band": r.get("param_band"), "survived": r["survived"],
                              "progress_m": round(r["progress_m"], 2)} for r in rows]
            print(name, " ".join(f"L{r['level']}:{'ok' if r['survived'] else 'FALL'}({r['progress_m']:.1f}m)" for r in rows),
                  flush=True)

    out = {"tool": "scripts/robustness_bench.py", "mode": args.mode, "checkpoint": str(ckpt),
           "checkpoint_sha256": sha256_file(ckpt), "actuator_profile": profile, "robot_mass_kg": robot_mass_kg,
           "stop_model": bool(args.stop_model),
           "knee_band": args.knee_band, "solver_iterations": args.solver_iters, "mass_kg": mass,
           "repeats": args.repeats, "repeat_jitter_s": args.repeat_jitter_s,
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "args": {k: str(v) for k, v in vars(args).items()},
           "summary": summary, "cases": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[robustness_bench] wrote {args.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        app.close()
    sys.exit(rc)
