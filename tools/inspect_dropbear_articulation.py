"""Inspect and verify the Dropbear articulation config in Isaac Lab 2.2 (GPU; run under tools/gpu_lock_run.py).

Checks (all results are written to ``--out`` JSON; the console log is the evidence):

1. Names: 91 joints / 93 bodies, the 22 motors + 6 neck screws resolve (preserve_order), the regex
   passive groups cover exactly the remaining 63 DOFs, applied stiffness/damping/effort per group.
2. Rest frames: key-body poses relative to the root and the anchor frame offset
   (``q_anchor_body * ANCHOR_FRAME_OFFSET`` must equal the root orientation).
3. Anchor rigidity: a free-floating robot is spun (|w| ~ 3.9 rad/s) and falls for 1 s; the anchor pose
   in the root frame must stay constant.
4. Angular-velocity cap: an identical robot with the legacy ``max_angular_velocity=50`` (deg/s) is spun
   the same way; we report whether PhysX clamps its link angular velocity.
5. Motor sweep (fixed root): each motor is moved by +/-0.2 rad from the default pose and every tracked
   body's displacement relative to the root is compared with ``EXPECTED_BODY_DRIVERS``.
6. Closure anchor gaps at spawn and after settling.

Usage::

    python tools/gpu_lock_run.py --log logs/robot_task/inspect_articulation.log -- \
        C:/isaac-sim/python.bat -u tools/inspect_dropbear_articulation.py --headless
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=_REPO / "logs" / "robot_task" / "inspect_articulation.json")
parser.add_argument("--solver_iters", type=int, nargs=2, default=(32, 4), metavar=("POS", "VEL"))
parser.add_argument("--sweep_delta", type=float, default=0.2, help="motor sweep amplitude [rad]")
parser.add_argument("--move_pos_mm", type=float, default=3.0, help="'moved' threshold, position [mm]")
parser.add_argument("--move_rot_deg", type=float, default=1.0, help="'moved' threshold, rotation [deg]")
parser.add_argument("--stiff_factor", type=float, default=20.0, help="motor gain multiplier for the kinematic sweep")
parser.add_argument("--joint_friction", default="0", help="joint friction for all joints, or 'usd' to keep USD values")
parser.add_argument("--passive_damping", type=float, default=50.0, help="passive DOF damping [N*m*s/rad]")
parser.add_argument("--keep_orphans", action="store_true", help="do NOT deactivate the 3 joint-less USD bodies")
parser.add_argument("--no_knee_axis_fix", action="store_true", help="keep LL_Revolute121 axis X as authored")
parser.add_argument("--no_inertia_fix", action="store_true", help="keep the zero *_bicep_1 inertia as authored")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main(report: dict) -> int:  # noqa: C901
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.isaac.launch import sha256_file
    from dropbear_wbc.robots import dropbear as D

    report.update({"status": "starting", "args": {k: str(v) for k, v in vars(args).items()}})
    usd = D.resolve_usd_path()
    report["usd_path"] = usd
    report["usd_sha256"] = sha256_file(usd)
    report["usd_sha_matches_contract"] = report["usd_sha256"] == D.USD_SHA256
    pos_it, vel_it = args.solver_iters
    friction = None if args.joint_friction == "usd" else float(args.joint_friction)
    common = dict(
        solver_position_iterations=pos_it,
        solver_velocity_iterations=vel_it,
        activate_contact_sensors=False,
        joint_friction=friction,
        passive_damping=args.passive_damping,
        remove_orphan_bodies=not args.keep_orphans,
        fix_left_knee_closure=not args.no_knee_axis_fix,
        fix_zero_inertia=not args.no_inertia_fix,
    )

    fixed_cfg = D.make_dropbear_cfg(prim_path="{ENV_REGEX_NS}/RobotFixed", fix_root_link=True, **common)
    fixed_cfg.init_state.pos = (0.0, 0.0, 0.6)
    free_cfg = D.make_dropbear_cfg(prim_path="{ENV_REGEX_NS}/RobotFree", **common)
    free_cfg.init_state.pos = (4.0, 0.0, 6.0)
    legacy_cfg = D.make_dropbear_cfg(
        prim_path="{ENV_REGEX_NS}/RobotLegacyVel", max_angular_velocity_deg_s=50.0, **common
    )
    legacy_cfg.init_state.pos = (-4.0, 0.0, 6.0)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))
        robot_fixed = fixed_cfg
        robot_free = free_cfg
        robot_legacy = legacy_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=20.0))
    t0 = time.time()
    sim.reset()
    report["sim_reset_s"] = round(time.time() - t0, 1)
    dt = sim.get_physics_dt()
    fixed: Articulation = scene["robot_fixed"]
    free: Articulation = scene["robot_free"]
    legacy: Articulation = scene["robot_legacy"]

    # ------------------------------------------------------------------ 1. names / groups / gains
    jn, bn = list(fixed.joint_names), list(fixed.body_names)
    motor_ids, motor_names = fixed.find_joints(list(D.MOTOR_NAMES), preserve_order=True)
    neck_ids, _ = fixed.find_joints(list(D.NECK_NAMES), preserve_order=True)
    groups = {name: list(act.joint_names) for name, act in fixed.actuators.items()}
    covered = [j for names in groups.values() for j in names]
    passive = groups["passive_legs"] + groups["passive_arms_head"]
    names_ok = (
        len(jn) == D.EXPECTED_NUM_JOINTS
        and len(bn) == D.EXPECTED_NUM_BODIES
        and tuple(motor_names) == D.MOTOR_NAMES
        and len(passive) == D.EXPECTED_NUM_PASSIVE
        and sorted(covered) == sorted(jn)
        and len(set(covered)) == len(covered)
        and bn[0] == D.ROOT_BODY
    )
    stiff = fixed.data.joint_stiffness[0].cpu()
    damp = fixed.data.joint_damping[0].cpu()
    eff = fixed.data.joint_effort_limits[0].cpu() if hasattr(fixed.data, "joint_effort_limits") else None
    arm = fixed.data.joint_armature[0].cpu()
    lim = fixed.data.joint_pos_limits[0].cpu()
    fric = fixed.data.joint_friction_coeff[0].cpu() if hasattr(fixed.data, "joint_friction_coeff") else None
    vmax = fixed.data.joint_vel_limits[0].cpu() if hasattr(fixed.data, "joint_vel_limits") else None
    group_gains = {}
    for gname, names in groups.items():
        ids = [jn.index(n) for n in names]
        group_gains[gname] = {
            "count": len(ids),
            "stiffness": sorted({round(float(stiff[i]), 4) for i in ids}),
            "damping": sorted({round(float(damp[i]), 4) for i in ids}),
            "effort_limit": sorted({round(float(eff[i]), 4) for i in ids}) if eff is not None else None,
            "armature": sorted({round(float(arm[i]), 5) for i in ids}),
            "friction": sorted({round(float(fric[i]), 4) for i in ids}) if fric is not None else None,
            "max_joint_vel": sorted({round(float(vmax[i]), 3) for i in ids}) if vmax is not None else None,
        }
    masses = fixed.root_physx_view.get_masses()[0].cpu()
    inertias = fixed.root_physx_view.get_inertias()[0].cpu().reshape(len(bn), -1)
    try:
        fprops = fixed.root_physx_view.get_dof_friction_properties()[0].cpu()
        physx_friction = {jn[i]: [round(float(v), 4) for v in fprops[i]] for i in range(len(jn)) if float(fprops[i].abs().max()) > 0}
    except Exception as exc:  # noqa: BLE001
        physx_friction = f"unavailable: {exc}"
    try:
        mm = fixed.root_physx_view.get_generalized_mass_matrices()[0].cpu()
        mass_matrix_diag = {n: round(float(mm[i, i]), 5) for n, i in zip(D.MOTOR_NAMES, fixed.find_joints(list(D.MOTOR_NAMES), preserve_order=True)[0])}
    except Exception as exc:  # noqa: BLE001
        mass_matrix_diag = f"unavailable: {exc}"
    bicep = {b: [round(float(v), 9) for v in inertias[bn.index(b)]] for b in ("LH_bicep_1", "RH_bicep_1") if b in bn}
    report["names"] = {
        "ok": names_ok,
        "num_joints": len(jn),
        "num_bodies": len(bn),
        "joint_names": jn,
        "body_names": bn,
        "motor_isaac_indices": [int(i) for i in motor_ids],
        "neck_isaac_indices": [int(i) for i in neck_ids],
        "num_passive": len(passive),
        "passive_joints": passive,
        "actuator_groups": group_gains,
        "motor_limits_deg": {
            n: [round(math.degrees(float(lim[i, 0])), 3), round(math.degrees(float(lim[i, 1])), 3)]
            for n, i in zip(motor_names, motor_ids)
        },
        "total_mass_kg": round(float(masses.sum()), 3),
        "root_mass_kg": round(float(masses[0]), 3),
        "max_shapes": int(getattr(fixed.root_physx_view, "max_shapes", -1)),
        "physx_nonzero_friction_props(static,dynamic,viscous)": physx_friction,
        "motor_mass_matrix_diag": mass_matrix_diag,
        "bicep_inertia_physx": bicep,
        "default_motor_pos": {n: float(fixed.data.default_joint_pos[0, i]) for n, i in zip(motor_names, motor_ids)},
    }
    print(json.dumps({"phase": "names", **{k: v for k, v in report["names"].items() if k not in ("joint_names", "body_names", "passive_joints")}}, indent=1), flush=True)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    report["orphans"] = {
        name: {"active": bool(stage.GetPrimAtPath(f"/World/envs/env_0/RobotFixed/{name}").IsActive())}
        for name in D.ORPHAN_BODIES
    }
    body_idx = {n: i for i, n in enumerate(bn)}
    anchor_i = body_idx[D.ANCHOR_BODY]
    tracked_i = [body_idx[n] for n in D.TRACKED_BODIES]
    offset = torch.tensor(D.ANCHOR_FRAME_OFFSET_WXYZ, device=sim.device).unsqueeze(0)

    def rel_pose(robot: Articulation, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Body poses in the root link frame: pos (len(ids),3), quat (len(ids),4)."""
        rp = robot.data.root_link_pos_w[0:1]
        rq = robot.data.root_link_quat_w[0:1]
        bp = robot.data.body_link_pos_w[0, ids]
        bq = robot.data.body_link_quat_w[0, ids]
        p = quat_apply_inverse(rq.expand(len(ids), 4), bp - rp)
        q = quat_mul(quat_inv(rq).expand(len(ids), 4), bq)
        return p, q

    # ------------------------------------------------------------------ 6a. closures at spawn
    closures = find_closures(fixed.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace(".*", "0"), bn)
    mon = ClosureMonitor(closures, sim.device)
    scene.update(dt)
    gap0 = mon.gaps(fixed.data.body_link_pos_w, fixed.data.body_link_quat_w)[0]
    report["closures"] = {
        "count": len(closures),
        "names": mon.names,
        "types": sorted({c.joint_type for c in closures}),
        "spawn_gap_max_m": float(gap0.max()),
        "spawn_gap_by_joint_m": {n: round(float(g), 6) for n, g in zip(mon.names, gap0)},
    }

    # ------------------------------------------------------------------ 2. rest frames (at spawn, fixed robot)
    p_rel, q_rel = rel_pose(fixed, [body_idx[n] for n in set(D.KEY_BODIES.values())])
    anchor_frame = quat_mul(fixed.data.body_link_quat_w[0:1, anchor_i], offset)
    anchor_vs_root_deg = math.degrees(float(quat_error_magnitude(anchor_frame, fixed.data.root_link_quat_w[0:1])[0]))
    report["rest_frames"] = {
        "key_body_pos_in_root": {
            n: [round(float(v), 4) for v in p_rel[k]] for k, n in enumerate(set(D.KEY_BODIES.values()))
        },
        "key_body_quat_in_root": {
            n: [round(float(v), 4) for v in q_rel[k]] for k, n in enumerate(set(D.KEY_BODIES.values()))
        },
        "anchor_frame_vs_root_deg": anchor_vs_root_deg,
    }
    print(json.dumps({"phase": "rest_frames", "anchor_frame_vs_root_deg": anchor_vs_root_deg,
                      "closure_spawn_gap_max_m": report["closures"]["spawn_gap_max_m"],
                      "closure_count": len(closures)}), flush=True)

    # ------------------------------------------------------------------ targets: hold default pose
    for robot in (fixed, free, legacy):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()

    # ------------------------------------------------------------------ 3/4. spin + fall (free & legacy)
    w_cmd = torch.tensor([[1.5, -2.0, 3.0]], device=sim.device)
    for robot in (free, legacy):
        vel = torch.zeros(1, 6, device=sim.device)
        vel[:, 3:] = w_cmd
        robot.write_root_com_velocity_to_sim(vel)
    scene.update(dt)
    p_a0, q_a0 = rel_pose(free, [anchor_i])
    max_dp_mm, max_dq_deg = 0.0, 0.0
    w_free, w_legacy, finite = [], [], True
    for step in range(200):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        p_a, q_a = rel_pose(free, [anchor_i])
        max_dp_mm = max(max_dp_mm, 1000.0 * float(torch.linalg.vector_norm(p_a - p_a0)))
        max_dq_deg = max(max_dq_deg, math.degrees(float(quat_error_magnitude(q_a, q_a0)[0])))
        if step % 20 == 0 or step == 199:
            w_free.append(round(float(torch.linalg.vector_norm(free.data.body_link_ang_vel_w[0, 0])), 4))
            w_legacy.append(round(float(torch.linalg.vector_norm(legacy.data.body_link_ang_vel_w[0, 0])), 4))
        finite &= bool(torch.isfinite(free.data.body_link_pos_w).all() and torch.isfinite(legacy.data.body_link_pos_w).all())
    anchor_ok = finite and max_dp_mm < 0.5 and max_dq_deg < 0.05
    report["anchor_rigidity"] = {
        "ok": anchor_ok,
        "commanded_root_ang_vel_rad_s": [1.5, -2.0, 3.0],
        "max_anchor_pos_dev_mm": max_dp_mm,
        "max_anchor_rot_dev_deg": max_dq_deg,
        "finite": finite,
    }
    report["angular_velocity_cap"] = {
        "root_ang_speed_rad_s_ours(2865deg/s cap)": w_free,
        "root_ang_speed_rad_s_legacy(50deg/s cap)": w_legacy,
        "legacy_cap_rad_s": math.radians(50.0),
        "legacy_value_clamps_link_speed": bool(max(w_legacy) <= math.radians(50.0) * 1.05),
    }
    print(json.dumps({"phase": "anchor_rigidity", **report["anchor_rigidity"]}), flush=True)
    print(json.dumps({"phase": "angular_velocity_cap", **report["angular_velocity_cap"]}), flush=True)

    # ------------------------------------------------------------------ 5. motor sweeps (fixed root)
    def run(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)

    default = fixed.data.default_joint_pos.clone()
    fixed.set_joint_position_target(default)
    settle_trace = {}
    for k in range(6):  # 3 s settle from the all-zero spawn state, sampled every 0.5 s
        run(100)
        settle_trace[f"{0.5 * (k + 1):.1f}s"] = {
            n: round(float(fixed.data.joint_pos[0, i]), 4) for n, i in zip(motor_names, motor_ids)
        }
    gap_settled = mon.gaps(fixed.data.body_link_pos_w, fixed.data.body_link_quat_w)[0]
    report["closures"]["settled_gap_max_m"] = float(gap_settled.max())
    report["closures"]["settled_gap_by_joint_m"] = {n: round(float(g), 6) for n, g in zip(mon.names, gap_settled)}
    report["settle_from_zero_nominal_gains"] = settle_trace
    report["settled_default_motor_pos"] = settle_trace["3.0s"]
    worst_settle = max(abs(settle_trace["3.0s"][n] - float(default[0, i])) for n, i in zip(motor_names, motor_ids))
    print(json.dumps({"phase": "settle", "closure_settled_gap_max_m": float(gap_settled.max()),
                      "worst_motor_error_after_3s_rad": worst_settle}), flush=True)

    def sweep_targets(m_id: int) -> torch.Tensor:
        lo_, hi_ = float(lim[m_id, 0]), float(lim[m_id, 1])
        q0 = float(default[0, m_id])
        delta = args.sweep_delta if (q0 + args.sweep_delta) <= hi_ - 0.02 else -args.sweep_delta
        target = default.clone()
        target[0, m_id] = min(max(q0 + delta, lo_), hi_)
        return target

    # 5a. nominal-gain step response (how fast does each motor follow a +/-0.2 rad step?)
    step_resp = {}
    sample_steps = {"0.1s": 20, "0.25s": 50, "0.5s": 100, "1.0s": 200, "2.0s": 400}
    for m_name, m_id in zip(motor_names, motor_ids):
        start_q = float(fixed.data.joint_pos[0, m_id])
        target = sweep_targets(m_id)
        fixed.set_joint_position_target(target)
        tgt = float(target[0, m_id])
        rec, done = {}, 0
        for label, n_steps in sample_steps.items():
            run(n_steps - done)
            done = n_steps
            q = float(fixed.data.joint_pos[0, m_id])
            rec[label] = round((q - start_q) / (tgt - start_q), 3) if abs(tgt - start_q) > 1e-6 else None
        step_resp[m_name] = {"start": round(start_q, 4), "target": round(tgt, 4), "fraction_reached": rec}
        fixed.set_joint_position_target(default)
        run(400)
        print(json.dumps({"phase": "step_response", "motor": m_name, **step_resp[m_name]}), flush=True)
    report["step_response_nominal_gains"] = step_resp

    # 5b. kinematic sweep with scaled motor gains (removes compliance / reaction-torque motion)
    stiff = fixed.data.joint_stiffness.clone()
    dampg = fixed.data.joint_damping.clone()
    mids = torch.as_tensor(motor_ids, dtype=torch.long, device=stiff.device)
    stiff[:, mids] *= args.stiff_factor
    dampg[:, mids] *= args.stiff_factor
    fixed.write_joint_stiffness_to_sim(stiff)
    fixed.write_joint_damping_to_sim(dampg)
    fixed.set_joint_position_target(default)
    run(400)
    base_p, base_q = rel_pose(fixed, tracked_i)
    matrix_pos, matrix_rot, failures, motor_track = {}, {}, [], {}
    for m_name, m_id in zip(motor_names, motor_ids):
        target = sweep_targets(m_id)
        fixed.set_joint_position_target(target)
        run(300)
        p, q = rel_pose(fixed, tracked_i)
        dp = 1000.0 * torch.linalg.vector_norm(p - base_p, dim=-1)
        dq = torch.rad2deg(quat_error_magnitude(q, base_q))
        motor_track[m_name] = {
            "target": round(float(target[0, m_id]), 4),
            "reached": round(float(fixed.data.joint_pos[0, m_id]), 4),
            "closure_gap_max_m": round(float(mon.worst(fixed.data.body_link_pos_w, fixed.data.body_link_quat_w)[0]), 6),
        }
        matrix_pos[m_name] = {b: round(float(v), 2) for b, v in zip(D.TRACKED_BODIES, dp)}
        matrix_rot[m_name] = {b: round(float(v), 2) for b, v in zip(D.TRACKED_BODIES, dq)}
        for b_name, dpi, dqi in zip(D.TRACKED_BODIES, dp, dq):
            moved = float(dpi) > args.move_pos_mm or float(dqi) > args.move_rot_deg
            expected = m_name in D.EXPECTED_BODY_DRIVERS[b_name]
            if moved != expected:
                failures.append({"motor": m_name, "body": b_name, "moved": moved, "expected": expected,
                                 "dpos_mm": round(float(dpi), 2), "drot_deg": round(float(dqi), 2)})
        fixed.set_joint_position_target(default)
        run(300)
        base_p, base_q = rel_pose(fixed, tracked_i)  # re-baseline (removes slow drift of passive mechanisms)
        print(json.dumps({"phase": "kinematic_sweep", "motor": m_name, **motor_track[m_name],
                          "moved_bodies": [b for b, v in matrix_pos[m_name].items()
                                           if v > args.move_pos_mm or matrix_rot[m_name][b] > args.move_rot_deg]}), flush=True)
    report["sweep"] = {
        "ok": not failures,
        "delta_rad": args.sweep_delta,
        "gain_factor": args.stiff_factor,
        "thresholds": {"pos_mm": args.move_pos_mm, "rot_deg": args.move_rot_deg},
        "failures": failures,
        "motor_tracking": motor_track,
        "dpos_mm": matrix_pos,
        "drot_deg": matrix_rot,
    }
    report["status"] = "executed"
    report["all_ok"] = bool(names_ok and anchor_ok and not failures)
    return 0 if report["all_ok"] else 2


if __name__ == "__main__":
    rc = 1
    result: dict = {}
    try:
        rc = main(result)
    except Exception:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"status": result.get("status"), "all_ok": result.get("all_ok"), "report": str(args.out)}), flush=True)
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
