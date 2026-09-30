"""Diagnose what happens right after an RSI reset in the tracking env (GPU, Isaac Lab 2.2).

Creates the Play env (no noise/pushes, motion starts at frame 0) with ``--motion_file``, then steps with
``--actions zero`` (hold the default pose) or ``--actions ref`` (open-loop: q* = reference motor positions) and
records, per policy step: anchor height (robot vs reference), lowest foot-body z, max |joint velocity| (+ the joints
responsible), worst loop-closure gap, and the first step at which each termination term fires.

    python tools/gpu_lock_run.py --owner gpu_pipeline --log logs/gpu_pipeline/diag_reset_x.log -- \
        C:/isaac-sim/python.bat -u tools/diag_reset_stability.py --motion_file data/motions/synthetic/wave_right.npz \
        --num_envs 8 --steps 60 --out logs/gpu_pipeline/diag_reset_x.json --headless
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Dropbear-Tracking-Flat-Play-v0")
parser.add_argument("--motion_file", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--solver_iters", type=int, nargs=2, default=(8, 4))
parser.add_argument("--actions", choices=["zero", "ref"], default="zero")
parser.add_argument("--allow_rejected", action="store_true")
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def main() -> dict:
    import gymnasium as gym
    import torch

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.robots.dropbear_names import ANCHOR_BODY, FOOT_BODIES
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point

    register()
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    env_cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    env_cfg.commands.motion.allow_rejected_motion = args.allow_rejected
    env_cfg.set_num_envs(args.num_envs)
    env_cfg.set_solver_iterations(*args.solver_iters)
    env_cfg.commands.motion.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    act = uw.action_manager.get_term("joint_pos")
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), uw.device)
    closure_names = [c.name for c in mon.closures] if hasattr(mon, "closures") else None
    ia = robot.body_names.index(ANCHOR_BODY)
    feet = [robot.body_names.index(b) for b in FOOT_BODIES]
    obs, _ = env.reset()
    jn = list(robot.joint_names)
    rep: dict = {"motion_file": str(args.motion_file), "task": args.task, "actions": args.actions,
                 "solver_iters": list(args.solver_iters), "num_envs": args.num_envs,
                 "spherical_joint_overrides": list(env_cfg.scene.robot.spawn.spherical_joint_overrides),
                 "steps": []}
    # state right after reset vs the reference frame
    q_ref0 = cmd.motion.full_joint_pos[cmd.time_steps]
    rep["reset_joint_pos_err_max"] = float((robot.data.joint_pos - q_ref0).abs().max())
    rep["reset_gap_max_m"] = float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w).max())
    first_term: dict = {}
    scale = act._scale if torch.is_tensor(act._scale) else torch.tensor(act._scale, device=uw.device)
    offset = act._offset
    for k in range(args.steps):
        if args.actions == "zero":
            a = torch.zeros(uw.num_envs, act.action_dim, device=uw.device)
        else:
            a = (cmd.joint_pos - offset) / scale
        obs, rew, term, trunc, info = env.step(a)
        jv = robot.data.joint_vel.abs()
        vmax, imax = jv.max(dim=1)
        top = torch.topk(jv[0], 4)
        gaps = mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)
        row = {"k": k, "anchor_z": float(robot.data.body_link_pos_w[0, ia, 2]),
               "ref_anchor_z": float(cmd.anchor_pos_w[0, 2]),
               "feet_z_min": float(robot.data.body_link_pos_w[0, feet, 2].min()),
               "joint_vel_max_env0": float(vmax[0]), "top_joints_env0": {jn[int(i)]: round(float(v), 2) for v, i in zip(top.values, top.indices)},
               "gap_max_mm": 1e3 * float(gaps.max()), "done_envs": int((term | trunc).sum()),
               "err_joint_pos": float(cmd.metrics["error_joint_pos"].mean()), "err_body_pos": float(cmd.metrics["error_body_pos"].mean())}
        for n in uw.termination_manager.active_terms:
            fired = uw.termination_manager.get_term(n)
            if bool(fired.any()) and n not in first_term:
                first_term[n] = k
        rep["steps"].append(row)
        if k < 12 or k % 10 == 0:
            print(json.dumps(row), flush=True)
    rep["first_termination_step"] = first_term
    env.close()
    return rep


if __name__ == "__main__":
    rc, result = 1, {}
    try:
        result = main()
        rc = 0
    except Exception:  # noqa: BLE001
        result = {"error": traceback.format_exc()}
        print(result["error"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"first_termination_step": result.get("first_termination_step"),
                      "reset_joint_pos_err_max": result.get("reset_joint_pos_err_max"),
                      "reset_gap_max_m": result.get("reset_gap_max_m")}), flush=True)
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
