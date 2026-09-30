"""SMOKE-ONLY: build a constant "standing still" contract NPZ (``dropbear-motion-npz-v1``) for the tracking env.

This is NOT the general settle tool (the calibration/settle builder owns that). It exists only so the
tracking task can be smoke-tested before real motions exist. Procedure (Isaac Lab 2.2, GPU):

1. Spawn Dropbear with the root ``world`` FIXED at ``--root_z`` (feet in the air), joints at the authored
   rest pose (all 0, loop closures consistent).
2. Ramp the motor PD targets from 0 to the default pose (calibration ``standing_motor_pos`` if present,
   else the legacy DROPBEAR_CFG pose) over ``--ramp_s``, then hold until the poses are stationary (max body
   displacement over 1 s < ``--drift_tol``; reported velocities keep a ~1e-2 floor from PhysX loop-closure bias)
   (PhysX solves the 27 excluded loop closures while the passive joints follow).
3. Compute the lowest collision-mesh vertex of all articulation bodies (USD collision meshes transformed
   by the live body poses) and shift the whole robot down so it sits exactly at z = 0.
4. Write T = fps * duration identical frames: settled joint row (all 91 joints), body link poses
   (ground at 0, env origin at (0, 0)), zero velocities, closure residual, metadata.

Usage::

    python tools/gpu_lock_run.py --log logs/robot_task/make_static_npz.log -- \
        C:/isaac-sim/python.bat -u tools/make_static_npz.py --headless
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=_REPO / "data" / "motions" / "smoke" / "dropbear_static_stand.npz")
parser.add_argument("--report", type=Path, default=_REPO / "logs" / "robot_task" / "make_static_npz.json",
                    help="JSON report path")
parser.add_argument("--fps", type=float, default=50.0)
parser.add_argument("--duration", type=float, default=4.0, help="clip length [s]")
parser.add_argument("--root_z", type=float, default=0.30, help="fixed root height while settling [m]")
parser.add_argument("--ramp_s", type=float, default=1.0)
parser.add_argument("--max_settle_s", type=float, default=12.0)
parser.add_argument("--motor_vel_tol", type=float, default=1e-2, help="settled: max |motor joint vel| < tol [rad/s]")
parser.add_argument("--drift_tol", type=float, default=5e-4, help="settled: max body displacement over 1 s < tol [m]")
parser.add_argument("--solver_iters", type=int, nargs=2, default=(32, 4), metavar=("POS", "VEL"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def collision_points_body_frame(robot_prim_path: str, body_names: list[str]):
    """Collision-mesh vertices of each body in its own link frame (numpy (N, 3) per body).

    Uses the authored USD transforms (mesh relative to body, so the spawn pose cancels out). Meshes are
    those below a ``collisions`` child of the body (URDF-importer convention used by this USD).
    """
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for name in body_names:
        body = stage.GetPrimAtPath(f"{robot_prim_path}/{name}")
        if not body.IsValid():
            raise ValueError(f"body prim {name} not found")
        inv_body = cache.GetLocalToWorldTransform(body).GetInverse()
        chunks = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions" not in str(prim.GetPath()):
                continue
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if pts is None or len(pts) == 0:
                continue
            m = np.array(cache.GetLocalToWorldTransform(prim) * inv_body, dtype=np.float64)  # row-vector convention
            p = np.asarray(pts, dtype=np.float64)
            chunks.append(p @ m[:3, :3] + m[3, :3])
        if chunks:
            out[name] = np.concatenate(chunks, axis=0)
    return out


def main(report: dict) -> int:  # noqa: C901
    import numpy as np
    import torch

    import isaaclab
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply

    from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
    from dropbear_wbc.isaac.launch import sha256_file
    from dropbear_wbc.robots import dropbear as D
    from dropbear_wbc.robots.defaults import resolve_default_pose
    from dropbear_wbc.tasks.tracking.motion_npz import SCHEMA, MotionArrays, save_motion_npz

    pose = resolve_default_pose()
    usd = D.resolve_usd_path()
    robot_cfg = D.make_dropbear_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        fix_root_link=True,
        default_pose=pose,
        solver_position_iterations=args.solver_iters[0],
        solver_velocity_iterations=args.solver_iters[1],
        activate_contact_sensors=False,
    )
    robot_cfg.init_state.pos = (0.0, 0.0, args.root_z)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))
        robot = robot_cfg

    physics_dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    motor_ids, _ = robot.find_joints(list(D.MOTOR_NAMES), preserve_order=True)
    mids = torch.as_tensor(motor_ids, dtype=torch.long, device=robot.device)
    mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), sim.device)

    # 1. authored rest pose (all zeros) -> consistent closures
    zeros = torch.zeros_like(robot.data.default_joint_pos)
    robot.write_joint_state_to_sim(zeros, zeros)
    default = robot.data.default_joint_pos.clone()

    def step(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(physics_dt)

    # 2. ramp targets 0 -> default, then hold until settled
    ramp_steps = max(1, int(args.ramp_s / physics_dt))
    for k in range(ramp_steps):
        robot.set_joint_position_target(default * (k + 1) / ramp_steps)
        step(1)
    robot.set_joint_position_target(default)
    # Settled = poses stationary: max body displacement over a 1 s window < drift_tol and motor speed < tol.
    # (Reported joint/body *velocities* keep a floor of ~1e-2 from PhysX's loop-closure velocity bias while
    # the poses do not move, so a pure velocity criterion never triggers; see logs/robot_task/make_static_npz_try1.)
    t_hold, settled, trace = 0.0, False, []
    window = int(round(1.0 / (20 * physics_dt)))
    history = []
    while t_hold < args.max_settle_s:
        step(20)
        t_hold += 20 * physics_dt
        history.append(robot.data.body_link_pos_w[0].clone())
        drift = float((history[-1] - history[-1 - window]).norm(dim=-1).max()) if len(history) > window else float("inf")
        vmax = float(robot.data.joint_vel.abs().max())
        vmot = float(robot.data.joint_vel[:, mids].abs().max())
        vbody = float(robot.data.body_link_lin_vel_w.norm(dim=-1).max())
        gap = float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)[0])
        trace.append({"t_s": round(args.ramp_s + t_hold, 2), "drift_1s_m": round(drift, 7), "max_joint_vel": round(vmax, 5),
                      "max_motor_vel": round(vmot, 5), "max_body_speed_m_s": round(vbody, 5), "closure_gap_m": round(gap, 6)})
        if drift < args.drift_tol and vmot < args.motor_vel_tol * 5 and t_hold >= 2.0:
            settled = True
            break
    qd = robot.data.joint_vel[0].abs()
    top = torch.topk(qd, 5)
    report["top_joint_speeds_rad_s"] = {robot.joint_names[int(i)]: round(float(v), 5) for v, i in zip(top.values, top.indices)}
    vb = robot.data.body_link_lin_vel_w[0].norm(dim=-1)
    topb = torch.topk(vb, 3)
    report["top_body_speeds_m_s"] = {robot.body_names[int(i)]: round(float(v), 6) for v, i in zip(topb.values, topb.indices)}
    report["settle_trace"] = trace[:: max(1, len(trace) // 20)] + trace[-1:]
    report["settled"] = settled
    if not settled:
        print(json.dumps({"phase": "settle", "warning": "did not reach tolerances", "last": trace[-1]}), flush=True)

    body_names = list(robot.body_names)
    pos = robot.data.body_link_pos_w[0].clone() - scene.env_origins[0]
    quat = robot.data.body_link_quat_w[0].clone()
    joint_pos = robot.data.joint_pos[0].clone()
    closure = float(mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w)[0])

    # 3. lowest collision vertex -> ground
    pts = collision_points_body_frame("/World/envs/env_0/Robot", body_names)
    lowest = {}
    for name, p in pts.items():
        i = body_names.index(name)
        pw = quat_apply(quat[i].expand(p.shape[0], 4), torch.as_tensor(p, dtype=torch.float32, device=pos.device)) + pos[i]
        lowest[name] = float(pw[:, 2].min())
    lowest_body = min(lowest, key=lowest.get)
    dz = -lowest[lowest_body]
    pos[:, 2] += dz
    feet_min = {n: round(lowest[n] + dz, 5) for n in D.FOOT_BODIES if n in lowest}
    report["lowest_body"] = lowest_body
    report["shift_dz_m"] = dz
    report["root_z_m"] = float(pos[0, 2])
    report["foot_body_min_z_m"] = feet_min
    report["num_bodies_with_collision_points"] = len(pts)

    # 4. constant clip
    t_frames = int(round(args.fps * args.duration))
    jn = list(robot.joint_names)
    motor_pos = {n: round(float(joint_pos[i]), 6) for n, i in zip(D.MOTOR_NAMES, motor_ids)}
    meta = {
        "schema": SCHEMA,
        "purpose": "SMOKE-ONLY constant standing clip for tracking-env tests (tools/make_static_npz.py); "
        "not a settled retarget, not for training real behaviours",
        "tool": "tools/make_static_npz.py",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "usd_path": usd,
        "usd_sha256": sha256_file(usd),
        "default_pose_source": pose.source,
        "default_pose_info": dict(pose.info),
        "default_motor_pos": pose.motor_pos,
        "settled_motor_pos": motor_pos,
        "settle": {"fixed_root_z_m": args.root_z, "ramp_s": args.ramp_s, "settled": settled, **trace[-1],
                   "criteria": {"drift_tol_m_per_s": args.drift_tol, "motor_vel_tol": args.motor_vel_tol * 5},
                   "top_joint_speeds_rad_s": report["top_joint_speeds_rad_s"]},
        "ground_shift": {"lowest_body": lowest_body, "dz_m": dz, "method": "min z of USD collision-mesh vertices"},
        "solver_iterations": list(args.solver_iters),
        "physics_dt": physics_dt,
        "joint_friction": 0.0,
        "orphan_bodies_deactivated": list(D.ORPHAN_BODIES),
        "authored_ankle_tierods": len(robot_cfg.spawn.spherical_joint_overrides) == 0,
        "spherical_joint_overrides": list(robot_cfg.spawn.spherical_joint_overrides),
        "isaaclab_version": getattr(isaaclab, "__version__", "2.2.0"),
        "body_lin_vel_w": "CoM linear velocity (Isaac Lab body_lin_vel_w); zero for this clip",
    }
    num_b = len(body_names)
    arrays = MotionArrays(
        fps=args.fps,
        joint_pos=np.repeat(joint_pos.cpu().numpy()[None], t_frames, axis=0),
        joint_vel=np.zeros((t_frames, len(jn)), dtype=np.float32),
        body_pos_w=np.repeat(pos.cpu().numpy()[None], t_frames, axis=0),
        body_quat_w=np.repeat(quat.cpu().numpy()[None], t_frames, axis=0),
        body_lin_vel_w=np.zeros((t_frames, num_b, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((t_frames, num_b, 3), dtype=np.float32),
        joint_names=jn,
        body_names=body_names,
        motor_names=list(D.MOTOR_NAMES),
        closure_residual_m=np.full((t_frames,), closure, dtype=np.float32),
        meta=meta,
    )
    out = save_motion_npz(args.out, arrays)
    report.update(
        status="executed",
        out=str(out),
        frames=t_frames,
        joints=len(jn),
        bodies=num_b,
        closure_residual_m=closure,
        settled_motor_pos=motor_pos,
        anchor_z_m=float(pos[body_names.index(D.ANCHOR_BODY), 2]),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "settle_trace"}, indent=1), flush=True)
    return 0 if settled else 3


if __name__ == "__main__":
    rc = 1
    result: dict = {"status": "starting"}
    try:
        rc = main(result)
    except Exception:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        side = args.report if args.report.is_absolute() else _REPO / args.report
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
