"""Knee (and hip-pitch) actuator authority probe for the velocity task: can the PD hold the settled stand?

The zero-action velocity env collapses at the knees in ~0.7 s with the legacy gains (kp 200 / kd 12 on the knee
CRANK, ~62 N*m/rad reflected at the knee through the four-bar, ``logs/locomotion/probe_smoke_v1.json``). This probe
runs ONE env with several actuator-gain groups side by side (env ``i`` gets group ``i % G``; gains are written per env
into PhysX and into the implicit-actuator model), zero actions (= hold the calibration standing pose), zero command,
deterministic reset (Play cfg: first NPZ frame, no pose/noise randomization), and reports per group:

  hold        fraction of envs without termination after ``--steps``; first-termination times; anchor-height trace
  knee        crank deflection from the default pose, the implied semantic knee angle (calibration LUT), and the PD
              torque the crank delivers (``kp*(q*-q) - kd*dq``, clipped to the group's effort) -- mean over the last
              second for envs still standing (= the static torque needed to hold the stand) and peak over the run
  hips/ankles same PD torque statistics (context: are they near their limits too?)

Group spec (repeatable ``--group``): ``name:knee_kp:knee_kd:knee_effort[:hip_pitch_kp:hip_pitch_kd]`` (crank units).

    python tools/gpu_lock_run.py --owner locomotion --log logs/locomotion/probe_knee_authority_v1.log --timeout 900 -- \
        C:/isaac-sim/python.bat -u tools/probe_knee_authority.py --num_envs 64 --steps 250 --solver_iters 8 4 \
        --headless --out logs/locomotion/probe_knee_authority_v1.json

Zero action is NOT walking; a stand that holds only shows the actuator can carry the static load with a policy-free PD.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

DEFAULT_GROUPS = [
    "legacy:200:12:300",
    "k400:400:17:300",
    "k600:600:21:300",
    "k800:800:24:300",
    "k1200:1200:30:300",
    "k600_e150:600:21:150",
    "k800_e150:800:24:150",
    "k600_hip300:600:21:300:300:7",
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Dropbear-Velocity-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--solver_iters", type=int, nargs=2, default=(8, 4), metavar=("POS", "VEL"))
parser.add_argument("--group", action="append", default=None, help="name:knee_kp:knee_kd:knee_effort[:hip_kp:hip_kd]")
parser.add_argument("--out", type=Path, default=_REPO / "logs" / "locomotion" / "probe_knee_authority.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def parse_group(spec: str) -> dict:
    p = spec.split(":")
    if len(p) not in (4, 6):
        raise ValueError(f"bad --group {spec!r}")
    g = {"name": p[0], "knee_kp": float(p[1]), "knee_kd": float(p[2]), "knee_effort": float(p[3])}
    if len(p) == 6:
        g.update(hip_pitch_kp=float(p[4]), hip_pitch_kd=float(p[5]))
    return g


def main(report: dict) -> int:  # noqa: C901
    import gymnasium as gym
    import numpy as np
    import torch

    from dropbear_wbc.robots import dropbear_names as N
    from dropbear_wbc.tasks.locomotion.config.dropbear import TASK_IDS  # noqa: F401
    from dropbear_wbc.tasks.registry import load_entry_point

    groups = [parse_group(s) for s in (args.group or DEFAULT_GROUPS)]
    G = len(groups)
    cfg = load_entry_point(args.task, "env_cfg_entry_point")
    cfg.set_num_envs(args.num_envs)
    cfg.set_solver_iterations(*args.solver_iters)
    cfg.sim.device = args.device
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * 0.02 + 5.0)
    cfg.commands.base_velocity.debug_vis = False
    report.update(task=args.task, num_envs=args.num_envs, steps=args.steps, solver_iters=list(args.solver_iters),
                  groups=groups, stand_info=dict(cfg.stand_info), default_pose_info=dict(cfg.default_pose_info))
    env = gym.make(args.task, cfg=cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    dev = uw.device
    n = uw.num_envs

    knee_ids, _ = robot.find_joints(list(N.KNEE_MOTORS), preserve_order=True)
    hip_pitch = ["LL_hip_joint", "RL_hip_joint"]
    hp_ids, _ = robot.find_joints(hip_pitch, preserve_order=True)
    hips_ids, hips_names = robot.find_joints(list(N.HIP_MOTORS), preserve_order=True)
    ank_ids, ank_names = robot.find_joints(list(N.ANKLE_MOTORS), preserve_order=True)
    gid = torch.arange(n, device=dev) % G

    # per-env gain tables (PhysX + implicit actuator model), all joints
    kp = robot.data.joint_stiffness.clone()
    kd = robot.data.joint_damping.clone()
    eff = robot.data.joint_effort_limits.clone() if hasattr(robot.data, "joint_effort_limits") else None
    legacy = {"knee_kp": float(kp[0, knee_ids[0]]), "knee_kd": float(kd[0, knee_ids[0]]),
              "hip_pitch_kp": float(kp[0, hp_ids[0]]), "hip_pitch_kd": float(kd[0, hp_ids[0]]),
              "knee_effort": float(eff[0, knee_ids[0]]) if eff is not None else None}
    report["sim_gains_before"] = legacy
    for g_i, g in enumerate(groups):
        ids = torch.nonzero(gid == g_i).flatten()
        kp[ids[:, None], torch.tensor(knee_ids, device=dev)] = g["knee_kp"]
        kd[ids[:, None], torch.tensor(knee_ids, device=dev)] = g["knee_kd"]
        if eff is not None:
            eff[ids[:, None], torch.tensor(knee_ids, device=dev)] = g["knee_effort"]
        if "hip_pitch_kp" in g:
            kp[ids[:, None], torch.tensor(hp_ids, device=dev)] = g["hip_pitch_kp"]
            kd[ids[:, None], torch.tensor(hp_ids, device=dev)] = g["hip_pitch_kd"]
    robot.write_joint_stiffness_to_sim(kp)
    robot.write_joint_damping_to_sim(kd)
    if eff is not None:
        robot.write_joint_effort_limit_to_sim(eff)
    for act in robot.actuators.values():  # keep the implicit-actuator model (applied_torque estimate) consistent
        jid = act.joint_indices
        jid = torch.arange(robot.num_joints, device=dev) if isinstance(jid, slice) else torch.as_tensor(jid, device=dev)
        act.stiffness[:] = kp[:, jid]
        act.damping[:] = kd[:, jid]
        if eff is not None:
            act.effort_limit[:] = eff[:, jid]
    # read back
    got_kp = robot.root_physx_view.get_dof_stiffnesses().to(dev)
    report["sim_gains_readback_knee_kp_per_group"] = [float(got_kp[int(torch.nonzero(gid == i)[0])][knee_ids[0]])
                                                       for i in range(G)]

    env.reset()
    cmd = uw.command_manager.get_term("base_velocity")
    anchor = robot.body_names.index(N.ANCHOR_BODY)
    actions = torch.zeros(n, uw.action_manager.total_action_dim, device=dev)
    calib = json.loads(Path(cfg.default_pose_info.get("path") or _REPO / "data/calibration/dropbear_semantic_calibration.json")
                       .read_text(encoding="utf-8"))
    luts = []
    for side in ("left", "right"):
        d = calib["dofs"][f"{side}_knee"]
        luts.append((np.array(d["motor_grid"]), np.array(d["semantic_values"])))
    q_def = robot.data.default_joint_pos.clone()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    first_term = torch.full((n,), -1, dtype=torch.long, device=dev)
    tail = max(1, int(round(1.0 / uw.step_dt)))
    acc = {k: torch.zeros(n, device=dev) for k in ("knee_tau", "knee_defl", "hp_tau")}
    acc_cnt = torch.zeros(n, device=dev)
    peak = {k: torch.zeros(n, device=dev) for k in ("knee_tau", "hp_tau", "hips_tau", "ank_tau")}
    sat_steps = torch.zeros(n, device=dev)
    trace = []
    t0 = time.time()
    for step in range(args.steps):
        cmd.vel_command_b[:] = 0.0
        _, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        newly = done & alive
        first_term[newly] = step
        alive &= ~done
        q = robot.data.joint_pos
        dq = robot.data.joint_vel
        tgt = robot.data.joint_pos_target
        tau = kp * (tgt - q) - kd * dq  # implicit PD (PhysX drive law), before the effort clip
        if eff is not None:
            tau = torch.maximum(torch.minimum(tau, eff), -eff)
        k_tau = tau[:, knee_ids].abs().amax(dim=1)
        peak["knee_tau"] = torch.where(alive, torch.maximum(peak["knee_tau"], k_tau), peak["knee_tau"])
        peak["hp_tau"] = torch.where(alive, torch.maximum(peak["hp_tau"], tau[:, hp_ids].abs().amax(dim=1)), peak["hp_tau"])
        peak["hips_tau"] = torch.where(alive, torch.maximum(peak["hips_tau"], tau[:, hips_ids].abs().amax(dim=1)), peak["hips_tau"])
        peak["ank_tau"] = torch.where(alive, torch.maximum(peak["ank_tau"], tau[:, ank_ids].abs().amax(dim=1)), peak["ank_tau"])
        if eff is not None:
            sat_steps += (alive & (k_tau >= 0.999 * eff[:, knee_ids[0]])).float()
        if step >= args.steps - tail:
            m = alive.float()
            acc["knee_tau"] += m * tau[:, knee_ids].mean(dim=1)
            acc["knee_defl"] += m * (q[:, knee_ids] - q_def[:, knee_ids]).mean(dim=1)
            acc["hp_tau"] += m * tau[:, hp_ids].mean(dim=1)
            acc_cnt += m
        if step % 10 == 0 or step == args.steps - 1:
            az = robot.data.body_link_pos_w[:, anchor, 2]
            row = {"step": step, "t_s": round((step + 1) * uw.step_dt, 2)}
            for i, g in enumerate(groups):
                sel = (gid == i)
                sa = sel & alive
                row[g["name"]] = {"alive": int(sa.sum()),
                                  "anchor_z_alive_mean": round(float(az[sa].mean()), 4) if bool(sa.any()) else None,
                                  "knee_crank_defl_deg": round(math.degrees(float((q[sa][:, knee_ids] - q_def[sa][:, knee_ids]).mean())), 2)
                                  if bool(sa.any()) else None}
            trace.append(row)
    wall = time.time() - t0
    res = {}
    for i, g in enumerate(groups):
        sel = gid == i
        ft = first_term[sel]
        ft_s = (ft[ft >= 0].float() + 1) * uw.step_dt
        held = sel & alive
        cnt = acc_cnt[held]
        r = {"envs": int(sel.sum()), "held_full_run": int(held.sum()),
             "first_termination_s": ({"min": round(float(ft_s.min()), 2), "median": round(float(ft_s.median()), 2),
                                      "max": round(float(ft_s.max()), 2)} if len(ft_s) else None),
             "peak_abs_tau_while_alive": {k: round(float(v[sel].max()), 1) for k, v in peak.items()},
             "knee_effort_saturated_steps_mean": round(float(sat_steps[sel].mean()), 1)}
        if bool(held.any()):
            defl = (acc["knee_defl"][held] / cnt).mean()
            q_crank = float((q_def[held][:, knee_ids].mean() + defl))
            knee_sem = [float(np.interp(q_crank, *luts[s])) for s in range(2)]
            slope = [float(np.interp(q_crank, luts[s][0], np.gradient(luts[s][1], luts[s][0]))) for s in range(2)]
            r.update(
                last_1s_mean_knee_crank_tau_Nm=round(float((acc["knee_tau"][held] / cnt).mean()), 1),
                last_1s_mean_knee_crank_deflection_deg=round(math.degrees(float(defl)), 2),
                knee_crank_deg=round(math.degrees(q_crank), 2),
                knee_semantic_deg=[round(math.degrees(k), 2) for k in knee_sem],
                knee_semantic_at_default_deg=round(math.degrees(float(np.interp(float(q_def[0, knee_ids[0]]), *luts[0]))), 2),
                crank_to_knee_ratio=[round(s, 3) for s in slope],
                equivalent_knee_tau_Nm=round(float((acc["knee_tau"][held] / cnt).mean()) / float(np.mean(slope)), 1),
                reflected_knee_kp_Nm_per_rad=round(g["knee_kp"] / float(np.mean(slope)) ** 2, 1),
                last_1s_mean_hip_pitch_tau_Nm=round(float((acc["hp_tau"][held] / cnt).mean()), 1),
                anchor_z_end_mean=round(float(robot.data.body_link_pos_w[held, anchor, 2].mean()), 4),
            )
        res[g["name"]] = r
        print(json.dumps({"group": g, **r}), flush=True)
    report.update(results=res, trace=trace, wall_s=round(wall, 1), stand_anchor_z=cfg.stand_info.get("anchor_z"),
                  joint_names={"hips": hips_names, "ankles": ank_names},
                  note="zero action + zero command, deterministic reset (first NPZ frame); PD torque = kp*(q*-q)-kd*dq "
                       "clipped to the effort limit (PhysX implicit drive law); 'held' = no termination in the run")
    return 0


if __name__ == "__main__":
    result: dict = {"tool": "tools/probe_knee_authority.py", "argv": sys.argv[1:], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
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
