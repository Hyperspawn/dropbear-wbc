"""Policy-free stand test: can Dropbear hold its standing pose with motor PD alone, and how much torque does it take?

Builds ``Dropbear-Velocity-Flat-Play-v0`` with an actuator profile, applies ZERO actions (motor targets = the calibrated
standing pose) for ``--seconds`` and reports, per motor: mean/max |motor torque| over the last ``--window_s``, the share
of steps where the actuator model clipped the PD torque, and whether/when the robot fell (anchor-height termination).
``--unlimited`` scales every hw_* group's peak torque and torque-speed line by 20 to MEASURE the torque the pose needs.

    python tools/gpu_lock_run.py --owner hw_twin --log logs/hw_twin/stand_hw_v1.log -- C:/isaac-sim/python.bat -u \
        tools/hw_stand_test.py --headless --actuator_profile hw_v1 --out logs/hw_twin/stand_hw_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--actuator_profile", default="hw_v1")
parser.add_argument("--unlimited", action="store_true", help="x20 peak torque / torque-speed line (measure the need)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--window_s", type=float, default=2.0)
parser.add_argument("--solver_iters", type=int, nargs=2, default=(32, 4))
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main() -> int:
    import gymnasium as gym
    import torch

    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401
    from dropbear_wbc.tasks.registry import load_entry_point

    task = "Dropbear-Velocity-Flat-Play-v0"
    env_cfg = load_entry_point(task, "env_cfg_entry_point")
    env_cfg.set_actuator_profile(args.actuator_profile)
    if args.unlimited:
        for key, act in env_cfg.scene.robot.actuators.items():
            if key.startswith("hw_"):
                act.effort_limit = {j: 20.0 * v for j, v in act.effort_limit.items()}
                act.saturation_effort = {j: 20.0 * v for j, v in act.saturation_effort.items()}
    env_cfg.set_num_envs(args.num_envs)
    env_cfg.set_solver_iterations(*args.solver_iters)
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.sim.device = args.device
    env = gym.make(task, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    dev = uw.device
    anchor = robot.body_names.index(N.ANCHOR_BODY)
    cols = [(N.MOTOR_NAMES.index(n), act, k) for act in robot.actuators.values()
            for k, n in enumerate(act.joint_names) if n in N.MOTOR_NAMES]
    lim = torch.zeros(len(N.MOTOR_NAMES), device=dev)
    for i, act, k in cols:
        lim[i] = act.effort_limit[0, k]
    env.reset()
    steps = int(round(args.seconds / uw.step_dt))
    win = int(round(args.window_s / uw.step_dt))
    zero = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=dev)
    alive = torch.ones(uw.num_envs, dtype=torch.bool, device=dev)
    fall_t = [None] * uw.num_envs
    z0 = robot.data.body_link_pos_w[:, anchor, 2].clone()
    tq, cl, trace = [], [], []
    for s in range(steps):
        env.step(zero)
        term = uw.termination_manager.terminated & alive
        for e in torch.nonzero(term).flatten().tolist():
            fall_t[e] = round((s + 1) * uw.step_dt, 2)
        alive &= ~uw.termination_manager.terminated
        if s >= steps - win and bool(alive.any()):
            t = torch.zeros(uw.num_envs, len(N.MOTOR_NAMES), device=dev)
            c = torch.zeros_like(t, dtype=torch.bool)
            for i, act, k in cols:
                t[:, i] = act.applied_effort[:, k]
                c[:, i] = (act.computed_effort[:, k] - act.applied_effort[:, k]).abs() > 0.01 * lim[i]
            tq.append(t[alive].abs())
            cl.append(c[alive])
        if s % 25 == 0:
            trace.append({"t_s": round((s + 1) * uw.step_dt, 2), "alive": int(alive.sum()),
                          "anchor_z0": round(float(robot.data.body_link_pos_w[0, anchor, 2]), 4)})
    out = {"tool": "tools/hw_stand_test.py", "actuator_profile": args.actuator_profile, "unlimited": args.unlimited,
           "solver_iterations": list(args.solver_iters), "seconds": args.seconds, "num_envs": uw.num_envs,
           "falls": sum(f is not None for f in fall_t), "fall_t_s": fall_t,
           "anchor_drop_m": [round(float(z0[e] - robot.data.body_link_pos_w[e, anchor, 2]), 4) for e in range(uw.num_envs)],
           "trace_env0": trace}
    if tq:
        T, C = torch.cat(tq), torch.cat(cl).float()
        out["per_motor_last_window"] = {
            n: {"peak_limit_Nm": round(float(lim[i]), 1), "mean_Nm": round(float(T[:, i].mean()), 2),
                "max_Nm": round(float(T[:, i].max()), 2), "clip_frac": round(float(C[:, i].mean()), 3)}
            for i, n in enumerate(N.MOTOR_NAMES)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "falls": out["falls"], "fall_t_s": fall_t}), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
