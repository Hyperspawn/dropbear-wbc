"""Diagnostic: why do the elbow and ``*_Revolute81`` motors reach only part of a step target?

Context: ``logs/robot_task/inspect_articulation_v3.log`` (robot/task component, legacy gains, passive damping 50,
gravity on) showed ``*_elbow_joint`` reaching 20-50 % of a 0.3 rad step and ``*_Revolute81`` stalling at ~0.45 of
a 0.2 rad step (from -0.075). Hypotheses: joint limit, loop binding (closure over-constraint), gain/damping.

This probe uses the contract robot config (``dropbear_wbc.robots.dropbear.make_dropbear_cfg``: legacy actuator
gains, contract 0.1 fixes, 32/4 solver iterations, root FIXED, no ground) and varies one factor per child
process (``--configs``): passive-joint damping ``pd50`` (legacy 50 N*m*s/rad) / ``pd0.5``, gravity ``g`` / ``nog``.
Each env holds the legacy default pose for 1.5 s, then one motor (env k -> motor k mod 22) gets a step of
``--step`` rad toward the inside of its range (``Revolute81``: also the robot_task case 0 -> 0.2 rad, and the
paired ``Revolute67`` + ``Revolute81`` common-mode step = pure ankle pitch). The fraction of the step reached at
0.1/0.25/0.5/1/2 s and the worst closure gap are reported.

Run (parent, plain python; the caller holds the GPU lock)::

    python tools/gpu_lock_run.py --owner calibrate_settle --log logs/calibrate_settle/probe_step_response.log \
        --timeout 1200 -- python tools/probe_step_response.py
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
ap.add_argument("--configs", nargs="+", default=["pd50_g", "pd50_nog", "pd0.5_g", "pd0.5_nog"])
ap.add_argument("--child", default="")
ap.add_argument("--step", type=float, default=0.3)
ap.add_argument("--out-dir", type=Path, default=REPO / "logs/calibrate_settle/step_response")
ARGS, KIT = ap.parse_known_args()

CHECK_S = (0.1, 0.25, 0.5, 1.0, 2.0)


def child(tag: str) -> int:
    from dropbear_wbc.isaac.launch import close_app_and_exit, prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(KIT)
    a.headless = True
    app = AppLauncher(a).app
    pd = float(tag.split("_")[0][2:])
    gravity = tag.split("_")[1] == "g"
    report: dict = {"tag": tag, "passive_damping": pd, "gravity": gravity, "step_rad": ARGS.step}
    rc = 1
    try:
        import numpy as np
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        from dropbear_wbc.isaac.closures import ClosureMonitor, find_closures
        from dropbear_wbc.robots import dropbear as D
        from dropbear_wbc.robots.defaults import DefaultPose

        pose = DefaultPose(motor_pos=dict(D.LEGACY_DEFAULT_MOTOR_POS), source="legacy")
        cfg = D.make_dropbear_cfg(fix_root_link=True, activate_contact_sensors=False, passive_damping=pd,
                                  default_pose=pose)
        cfg.init_state.pos = (0.0, 0.0, 1.0)
        cfg.init_state.joint_pos = {".*": 0.0}

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = cfg

        dt = 0.005
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, gravity=(0.0, 0.0, -9.81 if gravity else 0.0)))
        n = 64
        scene = InteractiveScene(SceneCfg(num_envs=n, env_spacing=4.0))
        sim.reset()
        robot = scene["robot"]
        mids, mnames = robot.find_joints(list(D.MOTOR_NAMES), preserve_order=True)
        mon = ClosureMonitor(find_closures("/World/envs/env_0/Robot", list(robot.body_names)), sim.device)
        lim = robot.data.joint_pos_limits[0, mids].cpu().numpy()
        default = torch.tensor([D.LEGACY_DEFAULT_MOTOR_POS[m] for m in D.MOTOR_NAMES], device=sim.device)

        def step(k: int) -> None:
            for _ in range(k):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt)

        zeros = torch.zeros_like(robot.data.joint_pos)
        robot.write_joint_state_to_sim(zeros, zeros)
        for k in range(1, 101):  # 0.5 s ramp to the legacy default, then 1 s hold
            robot.set_joint_position_target((default * k / 100).expand(n, 22), joint_ids=mids)
            step(1)
        step(200)
        start = robot.data.joint_pos[:, mids].clone()
        tgt = start.clone()
        cases = []
        for e in range(44):  # every motor twice: single-motor step toward the inside of its range
            j = e % 22
            s0 = float(start[e, j])
            dirn = 1.0 if s0 + ARGS.step <= lim[j, 1] - 1e-3 else -1.0
            tgt[e, j] = s0 + dirn * ARGS.step
            cases.append({"env": e, "j": j, "motor": D.MOTOR_NAMES[j], "kind": "single", "start": s0,
                          "target": float(tgt[e, j])})
        e = 44
        for side in ("LL", "RL"):  # robot_task case: Revolute81 alone to 0.2 rad
            j = D.MOTOR_NAMES.index(f"{side}_Revolute81")
            tgt[e, j] = 0.2
            cases.append({"env": e, "j": j, "motor": f"{side}_Revolute81", "kind": "abs_0.2",
                          "start": float(start[e, j]), "target": 0.2})
            e += 1
        for side in ("LL", "RL"):  # common-mode calf step (pure ankle pitch): both calf motors +step
            ja, jb = D.MOTOR_NAMES.index(f"{side}_Revolute67"), D.MOTOR_NAMES.index(f"{side}_Revolute81")
            tgt[e, ja] += ARGS.step
            tgt[e, jb] += ARGS.step
            cases.append({"env": e, "j": jb, "motor": f"{side}_Revolute81(+67 common mode)", "kind": "common_mode",
                          "start": float(start[e, jb]), "target": float(tgt[e, jb])})
            e += 1
        robot.set_joint_position_target(tgt, joint_ids=mids)
        hist = {}
        t = 0.0
        gap_max = torch.zeros(n, device=sim.device)
        for cs in CHECK_S:
            while t < cs - 1e-9:
                step(1)
                t += dt
                gap_max = torch.maximum(gap_max, mon.worst(robot.data.body_link_pos_w, robot.data.body_link_quat_w))
            hist[cs] = robot.data.joint_pos[:, mids].clone()
        rows = []
        for c in cases:
            j = c["j"]
            span = c["target"] - c["start"]
            frac = {f"{cs}s": round(float((hist[cs][c['env'], j] - c["start"]) / span), 3) for cs in CHECK_S}
            rows.append({**c, "fraction_reached": frac, "closure_gap_max_mm": round(1e3 * float(gap_max[c["env"]]), 3),
                         "limits_rad": lim[j].tolist()})
            print(json.dumps({"tag": tag, **{k: rows[-1][k] for k in ("motor", "kind", "start", "target",
                                                                     "fraction_reached", "closure_gap_max_mm")}}),
                  flush=True)
        report["cases"] = rows
        report["status"] = "ok"
        rc = 0
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    (ARGS.out_dir / f"step_{tag}.json").write_text(json.dumps(report, indent=1))
    close_app_and_exit(app, rc)
    return rc


def parent() -> int:
    if not (REPO / ".locks/gpu.lock").exists():
        print("[step] refusing to run: the GPU lock is not held (use tools/gpu_lock_run.py)", flush=True)
        return 3
    rcs = []
    for c in ARGS.configs:
        cmd = [_paths.isaac_python(), "-u", str(Path(__file__).relative_to(REPO)), "--child", c,
               "--step", str(ARGS.step), "--out-dir", str(ARGS.out_dir), "--headless"]
        print(f"[step] === {c}", flush=True)
        t = time.time()
        rcs.append(subprocess.call(cmd, cwd=str(REPO)))
        print(f"[step] === {c} rc={rcs[-1]} wall={time.time() - t:.1f}s", flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(child(ARGS.child) if ARGS.child else parent())
