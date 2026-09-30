"""Smoke-test the Dropbear velocity env: build, reset state, zero-action episodes. Finite simulation is NOT success.

Checks (all written to ``--out``):
  created     obs/action dims and term names, action joints + scales, command ranges, reset NPZ provenance;
  reset       feet on the ground: per-foot-body contact force, NON-foot bodies in contact (must be none), lowest
              non-foot body height, loop-closure gaps (``dropbear_wbc.isaac.closures``) vs the NPZ residual, joint
              row vs the NPZ, anchor height vs the NPZ, anchor tilt, root vs anchor projected gravity;
  zero_action ``--steps`` zero-action policy steps (= hold the default pose): terminations per term, first
              termination step per env, anchor-height trace, worst closure gap, finiteness, reward-term means.

    python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/probe_smoke.log --timeout 900 -- \
        C:/isaac-sim/python.bat -u tools/locomotion_env_probe.py --num_envs 64 --steps 250 --headless \
        --out logs/locomotion/probe_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Dropbear-Velocity-Flat-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--solver_iters", type=int, nargs=2, default=None, metavar=("POS", "VEL"))
parser.add_argument("--zero_command", action="store_true", help="force the velocity command to 0 for the episode")
parser.add_argument("--out", type=Path, default=_REPO / "logs" / "locomotion" / "probe_smoke.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main(report: dict) -> int:  # noqa: C901
    import gymnasium as gym
    import torch

    from isaaclab.utils.math import quat_apply_inverse, quat_mul

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401
    from dropbear_wbc.tasks.locomotion.mdp.curriculums import command_ranges_dict
    from dropbear_wbc.tasks.registry import load_entry_point

    cfg = load_entry_point(args.task, "env_cfg_entry_point")
    cfg.set_num_envs(args.num_envs)
    if args.solver_iters is not None:
        cfg.set_solver_iterations(*args.solver_iters)
    cfg.sim.device = args.device
    props = cfg.scene.robot.spawn.articulation_props
    report.update(task=args.task, num_envs=args.num_envs, stand_info=dict(cfg.stand_info),
                  solver_iters=[props.solver_position_iteration_count, props.solver_velocity_iteration_count],
                  default_pose_source=cfg.default_pose_source, default_pose_info=dict(cfg.default_pose_info))
    t0 = time.time()
    env = gym.make(args.task, cfg=cfg)
    uw = env.unwrapped
    report["env_create_s"] = round(time.time() - t0, 1)
    obs, _ = env.reset()
    robot = uw.scene["robot"]
    contact = uw.scene["contact_forces"]
    act = uw.action_manager.get_term("joint_pos")
    reset_term = uw.event_manager.get_term_cfg("reset_robot").func
    report["created"] = {
        "obs_dims": {k: list(v.shape) for k, v in obs.items()},
        "obs_terms": {g: list(uw.observation_manager.active_terms[g]) for g in uw.observation_manager.active_terms},
        "action_dim": int(uw.action_manager.total_action_dim),
        "action_joints": list(act._joint_names),
        "action_scale": [round(float(x), 4) for x in act._scale[0].tolist()],
        "num_bodies": robot.num_bodies, "num_joints": robot.num_joints,
        "command_ranges": command_ranges_dict(uw),
        "reward_terms": list(uw.reward_manager.active_terms),
        "termination_terms": list(uw.termination_manager.active_terms),
        "reset_npz": reset_term.npz_path, "reset_npz_validation": reset_term.validation,
        "npz_default_pose_max_abs_diff": reset_term.npz_default_pose_max_abs_diff,
    }
    print(json.dumps({"phase": "created", **report["created"]}, default=str), flush=True)

    # ------------------------------------------------------------------ reset state (after env.reset, before stepping)
    anchor = robot.body_names.index(N.ANCHOR_BODY)
    names = list(contact.body_names)
    foot_ids = [names.index(n) for n in N.FOOT_BODIES]
    # contact forces need a physics step: measure after ONE zero-action policy step (0.02 s after the reset);
    # the kinematic checks below (closure gaps, joint row, anchor) are taken before it, right after env.reset()
    kin = {"body_pos": robot.data.body_link_pos_w.clone(), "body_quat": robot.data.body_link_quat_w.clone(),
           "joint_pos": robot.data.joint_pos.clone(), "g_root": robot.data.projected_gravity_b.clone()}
    zero = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=uw.device)
    _, _, term0, trunc0, _ = env.step(zero)
    forces = contact.data.net_forces_w.norm(dim=-1)  # (N, B)
    non_foot = [i for i in range(len(names)) if i not in foot_ids]
    nf_contact = forces[:, non_foot] > 1.0
    nf_bodies = sorted({names[non_foot[j]] for j in torch.nonzero(nf_contact)[:, 1].tolist()})
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), uw.device)
    g0 = mon.worst(kin["body_pos"], kin["body_quat"])
    q_npz0 = reset_term.joint_pos[0]
    motor_ids, _ = robot.find_joints(list(N.MOTOR_NAMES), preserve_order=True)
    passive = torch.ones(robot.num_joints, dtype=torch.bool, device=uw.device)
    passive[motor_ids] = False
    jd = (kin["joint_pos"] - q_npz0).abs()
    off = torch.tensor(N.ANCHOR_FRAME_OFFSET_WXYZ, device=uw.device).expand(uw.num_envs, 4)
    g_anchor = quat_apply_inverse(quat_mul(kin["body_quat"][:, anchor], off), robot.data.GRAVITY_VEC_W)
    g_root = kin["g_root"]
    # lowest point proxy: link-origin heights of non-foot bodies (not collision geometry)
    body_z = kin["body_pos"][:, :, 2] - uw.scene.env_origins[:, 2:3]
    cnames = list(robot.body_names)
    nf_robot = [i for i, n in enumerate(cnames) if n not in N.FOOT_BODIES]
    low_i = int(body_z[:, nf_robot].mean(dim=0).argmin())
    report["reset"] = {
        "foot_body_force_N_mean": {n: round(float(forces[:, i].mean()), 1) for n, i in zip(N.FOOT_BODIES, foot_ids)},
        "envs_with_both_feet_loaded": int(((forces[:, foot_ids[0:2]].max(dim=1).values > 1.0)
                                           & (forces[:, foot_ids[2:4]].max(dim=1).values > 1.0)).sum()),
        "non_foot_bodies_in_contact": nf_bodies,
        "envs_with_non_foot_contact": int(nf_contact.any(dim=1).sum()),
        "total_robot_weight_N": round(float(robot.root_physx_view.get_masses().sum(dim=1).mean()) * 9.81, 1),
        "total_contact_force_N_mean": round(float(contact.data.net_forces_w[:, :, 2].sum(dim=1).mean()), 1),
        "closure_gap_m": {"median": float(g0.median()), "max": float(g0.max()), "npz_residual_max_m": reset_term.closure_max_m},
        "joint_row_vs_npz_max_abs": {"passive": float(jd[:, passive].max()), "motors": float(jd[:, motor_ids].max()),
                                      "note": "motors carry the serial-motor reset noise (legs +-0.02, arms +-0.1 rad)"},
        "anchor_z": {"mean": float(kin["body_pos"][:, anchor, 2].mean()),
                     "min": float(kin["body_pos"][:, anchor, 2].min()), "npz": cfg.stand_info["anchor_z"]},
        "contact_measured_after_first_step_s": uw.step_dt,
        "first_step_terminated_envs": int((term0 | trunc0).sum()),
        "anchor_gravity_mean": [round(float(x), 4) for x in g_anchor.mean(dim=0)],
        "root_projected_gravity_mean": [round(float(x), 4) for x in g_root.mean(dim=0)],
        "anchor_vs_root_gravity_max_abs": float((g_anchor - g_root).abs().max()),
        "lowest_non_foot_body_link_origin": {"body": cnames[nf_robot[low_i]],
                                             "z_mean": float(body_z[:, nf_robot[low_i]].mean())},
    }
    print(json.dumps({"phase": "reset", **report["reset"]}, default=str), flush=True)

    # ------------------------------------------------------------------ zero-action episodes
    actions = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=uw.device)
    cmd_term = uw.command_manager.get_term("base_velocity")
    terms = list(uw.termination_manager.active_terms)
    term_counts = {n: 0 for n in terms}
    rew_names = list(uw.reward_manager.active_terms)
    rew_sums = {n: 0.0 for n in rew_names}
    first_term = torch.full((uw.num_envs,), -1, dtype=torch.long, device=uw.device)
    finite_all, worst_gap, trace, first_bad, resets = True, 0.0, [], None, 0
    contact_bodies_at_termination: dict[str, int] = {}
    for step in range(args.steps):
        if args.zero_command:
            cmd_term.vel_command_b[:] = 0.0
        obs, rew, terminated, truncated, _ = env.step(actions)
        ok = all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(rew).all())
        ok &= bool(torch.isfinite(robot.data.root_state_w).all())
        if not ok and first_bad is None:
            first_bad = step
        finite_all &= ok
        done = terminated | truncated
        resets += int(done.sum())
        newly = done & (first_term < 0)
        first_term[newly] = step
        for n in terms:
            term_counts[n] += int((uw.termination_manager.get_term(n) & done).sum())
        if bool(uw.termination_manager.get_term("non_foot_contact").any()):
            hist = contact.data.net_forces_w_history.norm(dim=-1).max(dim=1).values  # (N, B)
            for e in torch.nonzero(uw.termination_manager.get_term("non_foot_contact")).flatten().tolist():
                for b in torch.nonzero(hist[e] > 1.0).flatten().tolist():
                    if names[b] not in N.FOOT_BODIES:
                        contact_bodies_at_termination[names[b]] = contact_bodies_at_termination.get(names[b], 0) + 1
        for i, n in enumerate(rew_names):
            rew_sums[n] += float(uw.reward_manager._step_reward[:, i].mean())
        gap = float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w).max())
        worst_gap = max(worst_gap, gap)
        if step % 10 == 0 or step == args.steps - 1:
            az = robot.data.body_link_pos_w[:, anchor, 2]
            trace.append({"step": step, "t_s": round((step + 1) * uw.step_dt, 2), "anchor_z_mean": round(float(az.mean()), 4),
                          "anchor_z_min": round(float(az.min()), 4), "worst_gap_m": round(gap, 6), "resets_so_far": resets})
    ft = first_term[first_term >= 0].float() * uw.step_dt
    report["zero_action"] = {
        "steps": args.steps, "zero_command": bool(args.zero_command), "finite": finite_all,
        "first_non_finite_step": first_bad, "episode_resets": resets, "termination_counts": term_counts,
        "envs_terminated": int((first_term >= 0).sum()),
        "first_termination_s": ({"min": float(ft.min()), "median": float(ft.median()), "max": float(ft.max())}
                                if len(ft) else None),
        "non_foot_contact_bodies_at_termination": dict(sorted(contact_bodies_at_termination.items(), key=lambda kv: -kv[1])),
        "mean_step_reward_terms": {n: v / args.steps for n, v in rew_sums.items()},
        "worst_closure_gap_m": worst_gap, "trace": trace,
        "note": "zero actions = hold the default (calibration standing) pose; no policy -> falls are expected; finite != success",
    }
    print(json.dumps({"phase": "zero_action", **{k: v for k, v in report["zero_action"].items() if k != "trace"}}), flush=True)
    return 0 if finite_all else 2


if __name__ == "__main__":
    result: dict = {"tool": "tools/locomotion_env_probe.py", "argv": sys.argv[1:],
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rc = 1
    try:
        rc = main(result)
        result["status"] = "executed"
    except Exception:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        out = args.out if args.out.is_absolute() else _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
