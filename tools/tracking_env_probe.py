"""Create the Dropbear tracking env and either smoke-test it (zero actions) or measure throughput.

Modes:
  smoke       N zero-action policy steps; per-step finiteness of observations / rewards / robot state,
              termination counts per term, reward-term means, worst loop-closure gap, feet contact at reset.
  throughput  warm-up, then time N zero-action steps; env-steps/s and GPU memory (torch + nvidia-smi).

Zero actions mean "hold the default pose" (JointPositionAction with use_default_offset). Finite
simulation is NOT success; this tool only checks the plumbing and the cost.

    python tools/gpu_lock_run.py --log logs/robot_task/smoke_env16.log -- C:/isaac-sim/python.bat -u \
        tools/tracking_env_probe.py --mode smoke --num_envs 16 --steps 200 --headless
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()


def gpu_used_mib() -> float | None:
    """Total used memory of GPU 0 [MiB] from nvidia-smi (includes other processes)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


GPU_BASELINE_MIB = gpu_used_mib()  # before Kit starts

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=("smoke", "throughput"), default="smoke")
parser.add_argument("--task", default="Dropbear-Tracking-Flat-v0")
parser.add_argument("--motion_file", type=Path, default=_REPO / "data" / "motions" / "smoke" / "dropbear_static_stand.npz")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--solver_iters", type=int, nargs=2, default=(32, 4), metavar=("POS", "VEL"))
parser.add_argument("--hold_steps", type=int, default=50, help="throughput mode: standing-hold closure stats after timing (0 = off)")
parser.add_argument("--out", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main(report: dict) -> int:  # noqa: C901
    import gymnasium as gym
    import torch

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point

    register()
    cfg = load_entry_point(args.task, "env_cfg_entry_point")
    cfg.commands.motion.motion_file = str((args.motion_file if args.motion_file.is_absolute() else _REPO / args.motion_file).resolve())
    cfg.set_num_envs(args.num_envs)
    cfg.set_solver_iterations(*args.solver_iters)
    cfg.sim.device = args.device
    report.update(mode=args.mode, task=args.task, num_envs=args.num_envs, solver_iters=list(args.solver_iters),
                  motion_file=cfg.commands.motion.motion_file, default_pose_source=cfg.default_pose_source,
                  gpu_baseline_mib=GPU_BASELINE_MIB)
    t0 = time.time()
    env = gym.make(args.task, cfg=cfg)
    uw = env.unwrapped
    report["env_create_s"] = round(time.time() - t0, 1)
    obs, _ = env.reset()
    robot = uw.scene["robot"]
    contact = uw.scene["contact_forces"]
    report["obs_dims"] = {k: list(v.shape) for k, v in obs.items()}
    report["action_dim"] = int(uw.action_manager.total_action_dim)
    report["action_joints"] = list(uw.action_manager.get_term("joint_pos")._joint_names)
    report["num_bodies"] = robot.num_bodies
    report["num_joints"] = robot.num_joints
    report["contact_sensor_bodies"] = len(contact.body_names)
    print(json.dumps({"phase": "created", **{k: report[k] for k in ("env_create_s", "obs_dims", "action_dim", "num_bodies", "num_joints", "contact_sensor_bodies")}}), flush=True)
    actions = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=uw.device)

    # default (action-offset) motor pose actually used by the articulation vs the config's resolved pose
    motor_ids, _ = robot.find_joints(list(N.MOTOR_NAMES), preserve_order=True)
    cfg_pose = torch.tensor([cfg.scene.robot.init_state.joint_pos[n] for n in N.MOTOR_NAMES], device=uw.device)
    live_default = robot.data.default_joint_pos[:, motor_ids]
    offset = uw.action_manager.get_term("joint_pos")._offset
    report["default_pose"] = {
        "source": cfg.default_pose_source,
        "cfg_motor_pos": [round(float(v), 6) for v in cfg_pose],
        "max_abs_live_default_minus_cfg": float((live_default - cfg_pose).abs().max()),
        "max_abs_action_offset_minus_live_default": float((offset - live_default).abs().max()),
        "note": "train config adds U(-0.01, 0.01) per env/motor at startup (add_joint_default_pos); play adds none",
    }
    motion_meta = uw.command_manager.get_term("motion").motion.arrays_meta
    report["motion_meta"] = {k: motion_meta.get(k) for k in ("schema", "tool", "status", "usd_sha256", "default_pose_source", "purpose")}
    report["motion_meta"]["npz_default_pose_max_abs_diff"] = uw.command_manager.get_term("motion").npz_default_pose_max_abs_diff

    if args.mode == "smoke":
        mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), uw.device)
        g0 = mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)
        report["gap_after_reset_m"] = {"median": float(g0.median()), "max": float(g0.max()),
                                       "npz_closure_residual_max_m": float(uw.command_manager.get_term("motion").motion.arrays_closure_max)}
        feet_ids = [contact.body_names.index(n) for n in N.FOOT_BODIES]
        f0 = contact.data.net_forces_w[:, feet_ids].norm(dim=-1)
        report["reset_feet_contact_force_N_mean"] = float(f0.mean())
        terms = uw.termination_manager.active_terms
        term_counts = {n: 0 for n in terms}
        rew_names = uw.reward_manager.active_terms
        rew_sums = {n: 0.0 for n in rew_names}
        finite_all, worst_gap, gap_trace, first_bad = True, 0.0, [], None
        resets = 0
        for step in range(args.steps):
            obs, rew, terminated, truncated, info = env.step(actions)
            ok = all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(rew).all())
            ok &= bool(torch.isfinite(robot.data.root_state_w).all()) and bool(torch.isfinite(robot.data.body_link_pos_w).all())
            if not ok and first_bad is None:
                first_bad = step
            finite_all &= ok
            done = terminated | truncated
            resets += int(done.sum())
            for n in terms:
                term_counts[n] += int((uw.termination_manager.get_term(n) & done).sum())
            for i, n in enumerate(rew_names):
                rew_sums[n] += float(uw.reward_manager._step_reward[:, i].mean())
            gap = float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w).max())
            worst_gap = max(worst_gap, gap)
            if step % 25 == 0 or step == args.steps - 1:
                gap_trace.append({"step": step, "worst_closure_gap_m": round(gap, 6),
                                  "anchor_z_mean": round(float(robot.data.body_link_pos_w[:, robot.body_names.index(N.ANCHOR_BODY), 2].mean()), 4),
                                  "resets_so_far": resets})
        report.update(
            steps=args.steps,
            finite=finite_all,
            first_non_finite_step=first_bad,
            episode_resets=resets,
            termination_counts=term_counts,
            mean_step_reward_terms={n: v / args.steps for n, v in rew_sums.items()},
            worst_closure_gap_m=worst_gap,
            trace=gap_trace,
            note="zero actions (hold default pose) with pushes/RSI noise of the training config; finite != success",
        )
        return 0 if finite_all else 2

    # throughput
    for _ in range(args.warmup):
        env.step(actions)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    used = gpu_used_mib()
    report.update(
        steps=args.steps,
        wall_s=dt,
        env_steps_per_s=args.num_envs * args.steps / dt,
        policy_steps_per_s=args.steps / dt,
        physics_steps_per_s=args.num_envs * args.steps * uw.cfg.decimation / dt,
        gpu_used_mib_during=used,
        gpu_process_delta_mib=(used - GPU_BASELINE_MIB) if (used is not None and GPU_BASELINE_MIB is not None) else None,
        torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        torch_reserved_mib=torch.cuda.memory_reserved() / 2**20,
        finite=bool(torch.isfinite(robot.data.root_state_w).all()),
    )
    print(json.dumps({"phase": "throughput", "num_envs": args.num_envs, "solver_iters": list(args.solver_iters),
                      "env_steps_per_s": round(report["env_steps_per_s"], 1), "gpu_used_mib": used}), flush=True)
    if args.hold_steps > 0:
        report["standing_hold"] = standing_hold(env, robot, actions, args.hold_steps)
        print(json.dumps({"phase": "standing_hold", **{k: v for k, v in report["standing_hold"].items() if k != "per_closure_max_m"}}), flush=True)
    return 0


def standing_hold(env, robot, actions, steps: int) -> dict:
    """Closure-gap statistics while holding the reference standing pose (after the timed window).

    Pushes are disabled and the RSI noise zeroed, all envs are reset to motion frame 0, then ``steps``
    zero-action policy steps are run. A sample (env, step) counts as *standing* while that env has not
    been reset since the hold began and its anchor height is within 5 cm of the reference. Gaps are the
    anchor distances of the 27 excluded (loop-closing) joints (``dropbear_wbc.isaac.closures``).
    """
    import torch

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.robots import dropbear_names as N

    uw = env.unwrapped
    try:
        uw.event_manager.get_term_cfg("push_robot").func = lambda *a, **k: None
    except Exception:  # noqa: BLE001  (play config has no push term)
        pass
    cmd = uw.command_manager.get_term("motion")
    cmd.cfg.pose_range = {}
    cmd.cfg.velocity_range = {}
    cmd.cfg.joint_position_range = (0.0, 0.0)
    cmd.cfg.closure_joint_position_range = (0.0, 0.0)
    cmd.cfg.start_at_zero = True
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), uw.device)
    env.reset()
    a_idx = robot.body_names.index(N.ANCHOR_BODY)
    n = uw.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=uw.device)
    gap0 = mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)
    samples, per_closure = [], torch.zeros(len(mon.names), device=uw.device)
    standing_frac = []
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        alive &= ~(terminated | truncated)
        dz = (robot.data.body_link_pos_w[:, a_idx, 2] - cmd.anchor_pos_w[:, 2]).abs()
        standing = alive & (dz < 0.05)
        standing_frac.append(float(standing.float().mean()))
        if bool(standing.any()):
            g = mon.gaps(robot.data.body_link_pos_w, robot.data.body_link_quat_w)[standing]
            samples.append(g.max(dim=-1).values)
            per_closure = torch.maximum(per_closure, g.max(dim=0).values)
    out = {
        "steps": steps,
        "rule": "pushes off, RSI noise 0, reset to frame 0, zero actions; sample counted while env not reset and |anchor_z - ref| < 5 cm",
        "gap_after_reset_m": {"median": float(gap0.median()), "max": float(gap0.max())},
        "standing_fraction_first": standing_frac[0] if standing_frac else None,
        "standing_fraction_last": standing_frac[-1] if standing_frac else None,
        "standing_fraction_mean": sum(standing_frac) / max(len(standing_frac), 1),
    }
    if samples:
        s = torch.cat(samples)
        q = torch.quantile(s.float(), torch.tensor([0.5, 0.95, 0.99], device=s.device))
        out.update(
            samples=int(s.numel()),
            worst_gap_m={"median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(s.max())},
            frac_samples_over_3mm=float((s > 0.003).float().mean()),
            per_closure_max_m={name: round(float(v), 6) for name, v in zip(mon.names, per_closure)},
            worst_closure=mon.names[int(per_closure.argmax())],
        )
    return out


if __name__ == "__main__":
    rc = 1
    result: dict = {"status": "starting"}
    try:
        rc = main(result)
        result["status"] = "executed"
    except Exception:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        out = args.out or (_REPO / "logs" / "robot_task" / f"{args.mode}_n{args.num_envs}_s{args.solver_iters[0]}-{args.solver_iters[1]}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("trace",)}, default=str), flush=True)
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
