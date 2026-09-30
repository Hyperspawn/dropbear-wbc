"""Diagnostic: does the passive-joint damping of ``make_dropbear_cfg`` reach PhysX? (review finding, 2026-09-24)

The 63 passive DOFs get ``ImplicitActuatorCfg(stiffness=0, damping=passive_damping)``. Isaac Lab writes that
through ``root_physx_view.set_dof_dampings`` (a *drive* parameter), but none of the passive tree joints has an
authored ``UsdPhysics.DriveAPI`` (docs/usd_tree_45586414.json: the 28 drives are the motors and neck screws).
This probe measures whether the damping acts, instead of assuming it:

* one child process per plant variant (``--variants``):
  - ``nodrive``: the contract spawn (``make_dropbear_cfg``), as used by training, calibration and settle;
  - ``driveapi``: the same plus ``UsdPhysics.DriveAPI`` (stiffness 0) applied in memory to every movable passive
    tree joint before the physics parse (``angular`` on revolute joints, ``rotX/rotY/rotZ`` on the spherical
    ``*_Revolute115/117``), i.e. candidate fix (a) of the finding;
* root fixed, gravity OFF, 32/4 iterations, dt 5 ms; the robot starts at the authored rest (all joints 0);
* 8 envs = 2 tests x passive damping {0, 0.5, 5, 50} written per env with ``write_joint_damping_to_sim``;
* test ``free4bar`` (envs 0-3): the elbow and knee motors are made free (kp = kd = 0) and get an initial joint
  velocity of 2 rad/s toward the inside of their range; every other motor holds 0 with its legacy PD. With no
  gravity the only things that slow the free four-bar are passive damping, closure-solver losses and limits;
* test ``rodspin`` (envs 4-5-6-7): the tie-rod rod-end DOFs ``LL/RL_Revolute115:1`` get 3 rad/s;
* read-back of ``get_dof_dampings / stiffnesses / max_forces / armatures`` for the passive DOFs, per env.

Output: ``--out`` JSON with the read-back and velocity/position histories at 0.01..1.0 s; one line per case.

Run (parent, plain python) under the lock::

    python tools/gpu_lock_run.py --owner review_fixes --log logs/review_fixes/probe_passive_damping.log \
        --timeout 900 -- python tools/probe_passive_damping.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--variants", nargs="+", default=["nodrive", "driveapi"])
ap.add_argument("--child", default="")
ap.add_argument("--dampings", type=float, nargs=4, default=(0.0, 0.5, 5.0, 50.0))
ap.add_argument("--out-dir", type=Path, default=REPO / "logs/review_fixes/passive_damping")
ARGS, KIT = ap.parse_known_args()

CHECK_S = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
FREE_MOTORS = ("LH_elbow_joint", "RH_elbow_joint", "LL_knee_actuator_joint", "RL_knee_actuator_joint")
SPIN_DOFS = ("LL_Revolute115:1", "RL_Revolute115:1")


def _apply_passive_drive_api(stage, root_path: str, passive_names: set[str]) -> dict:
    """Apply DriveAPI(stiffness 0) to every movable passive tree joint below ``root_path`` (in memory)."""
    from pxr import Usd, UsdPhysics

    done = {"revolute": 0, "spherical": 0, "skipped": []}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        joint = UsdPhysics.Joint(prim)
        if joint.GetExcludeFromArticulationAttr().Get():
            continue
        tname = prim.GetTypeName()
        if tname == "PhysicsRevoluteJoint" and name in passive_names:
            instances = ("angular",)
            done["revolute"] += 1
        elif tname == "PhysicsSphericalJoint" and any(n.startswith(name + ":") for n in passive_names):
            instances = ("rotX", "rotY", "rotZ")
            done["spherical"] += 1
        else:
            if name in passive_names:
                done["skipped"].append(f"{name}:{tname}")
            continue
        for inst in instances:
            if prim.HasAPI(UsdPhysics.DriveAPI, inst):
                continue
            drive = UsdPhysics.DriveAPI.Apply(prim, inst)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            drive.CreateTypeAttr("force")
    return done


def child(variant: str) -> int:
    from dropbear_wbc.isaac.launch import close_app_and_exit, prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(KIT)
    a.headless = True
    app = AppLauncher(a).app
    report: dict = {"variant": variant, "dampings": list(ARGS.dampings), "gravity": False, "dt": 0.005}
    rc = 1
    try:
        import numpy as np
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        from dropbear_wbc.robots import dropbear as D

        cfg = D.make_dropbear_cfg(fix_root_link=True, activate_contact_sensors=False)
        cfg.init_state.pos = (0.0, 0.0, 1.0)
        cfg.init_state.joint_pos = {".*": 0.0}

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = cfg

        dt = 0.005
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, gravity=(0.0, 0.0, 0.0)))
        n = 8
        scene = InteractiveScene(SceneCfg(num_envs=n, env_spacing=4.0))
        if variant == "driveapi":
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            # passive names are only known after the articulation is built; use the regex-free rule instead:
            # every tree joint that is not a motor / neck screw and is revolute or spherical.
            motors = set(D.MOTOR_NAMES) | set(D.NECK_NAMES)
            from pxr import Usd, UsdPhysics

            names: set[str] = set()
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
                if prim.IsA(UsdPhysics.Joint) and prim.GetName() not in motors:
                    if prim.GetTypeName() == "PhysicsSphericalJoint":
                        names.update(f"{prim.GetName()}:{k}" for k in range(3))
                    else:
                        names.add(prim.GetName())
            report["drive_api_applied"] = {
                f"env_{e}": _apply_passive_drive_api(stage, f"/World/envs/env_{e}/Robot", names) for e in range(n)
            }["env_0"]
        sim.reset()
        robot = scene["robot"]
        jn = list(robot.joint_names)
        mids, _ = robot.find_joints(list(D.MOTOR_NAMES), preserve_order=True)
        motor_set = set(D.MOTOR_NAMES) | set(D.NECK_NAMES)
        passive = [i for i, nme in enumerate(jn) if nme not in motor_set]
        report["num_passive"] = len(passive)
        free_ids = [jn.index(m) for m in FREE_MOTORS]
        spin_ids = [jn.index(s) for s in SPIN_DOFS]
        spin_all = [i for i, nme in enumerate(jn) if nme.split(":")[0] in ("LL_Revolute115", "RL_Revolute115")]
        dev = sim.device

        # per-env passive damping: envs 0-3 free4bar, 4-7 rodspin; damping index = env % 4
        for e in range(n):
            d = float(ARGS.dampings[e % 4])
            robot.write_joint_damping_to_sim(torch.full((1, len(passive)), d, device=dev), joint_ids=passive,
                                             env_ids=torch.tensor([e], device=dev))

        view = robot.root_physx_view
        rb = {}
        for key, getter in (("damping", view.get_dof_dampings), ("stiffness", view.get_dof_stiffnesses),
                            ("max_force", view.get_dof_max_forces), ("armature", view.get_dof_armatures)):
            arr = getter().cpu().numpy()
            rb[key] = {f"env{e}": sorted(set(np.round(arr[e, passive], 6).tolist())) for e in range(n)}
        rb["isaaclab_joint_damping"] = {f"env{e}": sorted(set(np.round(
            robot.data.joint_damping[e, passive].cpu().numpy(), 6).tolist())) for e in range(n)}
        report["readback_passive"] = rb
        print("[pd] readback", json.dumps(rb), flush=True)

        def step(k: int) -> None:
            for _ in range(k):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt)

        zeros = torch.zeros_like(robot.data.joint_pos)
        robot.write_joint_state_to_sim(zeros, zeros)
        robot.set_joint_position_target(torch.zeros(n, len(mids), device=dev), joint_ids=mids)
        step(40)  # 0.2 s settle at the rest pose (gravity off: nothing should move)
        rest_vel = float(robot.data.joint_vel.abs().max())
        report["rest_max_joint_speed"] = rest_vel

        # test free4bar: free the elbow/knee motors in envs 0-3
        env_a = torch.arange(0, 4, device=dev)
        robot.write_joint_stiffness_to_sim(torch.zeros(4, 4, device=dev), joint_ids=free_ids, env_ids=env_a)
        robot.write_joint_damping_to_sim(torch.zeros(4, 4, device=dev), joint_ids=free_ids, env_ids=env_a)
        lim = robot.data.joint_pos_limits[0].cpu().numpy()
        pos = robot.data.joint_pos.clone()
        vel = robot.data.joint_vel.clone()
        vel[:] = 0.0
        signs = {}
        for j in free_ids:
            mid = 0.5 * (lim[j, 0] + lim[j, 1])
            s = 1.0 if mid >= float(pos[0, j]) else -1.0
            signs[jn[j]] = s
            vel[0:4, j] = 2.0 * s
        for j in spin_ids:
            vel[4:8, j] = 3.0
        report["initial_velocity_signs"] = signs
        robot.write_joint_state_to_sim(pos, vel)
        p0 = robot.data.joint_pos.clone()
        hist = {}
        t = 0.0
        for cs in CHECK_S:
            while t < cs - 1e-9:
                step(1)
                t += dt
            hist[cs] = (robot.data.joint_pos.clone(), robot.data.joint_vel.clone())
        cases = []
        for e in range(n):
            d = float(ARGS.dampings[e % 4])
            test = "free4bar" if e < 4 else "rodspin"
            ids = free_ids if e < 4 else spin_ids
            for j in ids:
                row = {"variant": variant, "test": test, "env": e, "passive_damping": d, "dof": jn[j],
                       "vel": {f"{cs}s": round(float(hist[cs][1][e, j]), 4) for cs in CHECK_S},
                       "disp": {f"{cs}s": round(float(hist[cs][0][e, j] - p0[e, j]), 4) for cs in CHECK_S}}
                if e >= 4:
                    row["rod_end_speed_norm"] = {
                        f"{cs}s": round(float(hist[cs][1][e, [k for k in spin_all if jn[k].startswith(jn[j][:2])]]
                                              .norm()), 4) for cs in CHECK_S}
                cases.append(row)
                print(json.dumps(row), flush=True)
        report["cases"] = cases
        report["status"] = "ok"
        rc = 0
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    (ARGS.out_dir / f"passive_damping_{variant}.json").write_text(json.dumps(report, indent=1))
    close_app_and_exit(app, rc)
    return rc


def parent() -> int:
    if not (REPO / ".locks/gpu.lock").exists():
        print("[pd] refusing to run: the GPU lock is not held (use tools/gpu_lock_run.py)", flush=True)
        return 3
    rcs = []
    for v in ARGS.variants:
        cmd = [_paths.isaac_python(), "-u", str(Path(__file__).relative_to(REPO)), "--child", v,
               "--dampings", *[str(d) for d in ARGS.dampings], "--out-dir", str(ARGS.out_dir), "--headless"]
        print(f"[pd] === {v}", flush=True)
        t = time.time()
        rcs.append(subprocess.call(cmd, cwd=str(REPO)))
        print(f"[pd] === {v} rc={rcs[-1]} wall={time.time() - t:.1f}s", flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(child(ARGS.child) if ARGS.child else parent())
