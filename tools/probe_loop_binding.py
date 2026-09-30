"""Diagnostic: static 'loop binding' torque of the closed-loop motors, and whether passive damping reaches PhysX.

Gravity-free, root-fixed quasi-static articulation (``dropbear_wbc.isaac.quasistatic``, contract 0.1 fixes, 16
iterations) with very stiff motors (kp 1e4). Every env sweeps one closed-loop motor (or motor pair) slowly and
holds each sample for ``--hold`` steps; the static motor torque kp*(target - q) at the end of the hold is the
torque the loop needs to hold that pose without gravity, i.e. how hard the mechanism binds. A free, ideal
mechanism needs 0 N*m. Also reads back the per-DOF drive damping from PhysX
(``root_physx_view.get_dof_dampings``) next to what Isaac Lab believes it wrote (``data.joint_damping``).

Run under the lock::

    python tools/gpu_lock_run.py --owner calibrate_settle --log logs/calibrate_settle/probe_loop_binding.log -- \
        C:/isaac-sim/python.bat -u tools/probe_loop_binding.py --gpu-lock-held --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

ap = argparse.ArgumentParser()
ap.add_argument("--hold", type=int, default=30)
ap.add_argument("--kp", type=float, default=10000.0)
ap.add_argument("--kd", type=float, default=100.0)
ap.add_argument("--passive-damping", type=float, default=0.5)
ap.add_argument("--out", type=Path, default=REPO / "logs/calibrate_settle/probe_loop_binding.json")
ap.add_argument("--gpu-lock-held", action="store_true")
ARGS, KIT = ap.parse_known_args()


def main() -> int:
    from dropbear_wbc.isaac.launch import close_app_and_exit, prepare_kit_python

    prepare_kit_python()
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(KIT)
    a.headless = True
    app = AppLauncher(a).app
    report: dict = {"args": {k: str(v) for k, v in vars(ARGS).items()}}
    rc = 1
    try:
        import numpy as np
        import torch

        from dropbear_wbc.isaac.quasistatic import QuasiStaticCfg, QuasiStaticDropbear
        from dropbear_wbc.isaac.sweep_runner import run_programs
        from dropbear_wbc.kinematics.sweeps import Program, sweep1d
        from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES

        if not (REPO / ".locks/gpu.lock").exists():
            raise RuntimeError("GPU lock not held")
        qs = QuasiStaticDropbear(QuasiStaticCfg(num_envs=16, motor_kp=ARGS.kp, motor_kd=ARGS.kd,
                                                passive_damping=ARGS.passive_damping))
        # ---- damping read-back -------------------------------------------------------------------------
        phys = qs.robot.root_physx_view.get_dof_dampings()[0].cpu().numpy()
        lab = qs.robot.data.joint_damping[0].cpu().numpy()
        names = qs.joint_names
        mot = set(MOTOR_NAMES)
        passive = [i for i, n in enumerate(names) if n not in mot and not n.startswith("head_LeadScrew")]
        report["damping_readback"] = {
            "configured_passive": ARGS.passive_damping, "configured_motor": ARGS.kd,
            "isaaclab_passive_unique": sorted(set(np.round(lab[passive], 6).tolist())),
            "physx_passive_unique": sorted(set(np.round(phys[passive], 6).tolist())),
            "physx_motor_unique": sorted(set(np.round(phys[[names.index(m) for m in MOTOR_NAMES]], 6).tolist())),
        }
        print("[binding] damping", json.dumps(report["damping_readback"]), flush=True)
        # ---- programs ------------------------------------------------------------------------------------
        deg = np.pi / 180
        idx = {n: i for i, n in enumerate(MOTOR_NAMES)}
        progs = []
        for m in ("LH_elbow_joint", "RH_elbow_joint", "LL_knee_actuator_joint", "RL_knee_actuator_joint"):
            progs.append(sweep1d(idx[m], 0.25 * deg, 29.75 * deg, 1.5 * deg, 1.0 * deg, return_pass=True, name=m))
        for m in ("LL_Revolute81", "RL_Revolute81", "LL_Revolute67"):
            progs.append(sweep1d(idx[m], -24 * deg, 24 * deg, 2.0 * deg, 1.0 * deg, return_pass=False, name=m))
        for side in ("LL", "RL"):  # common mode = pure ankle pitch
            pr = Program(name=f"{side}_calf_common_mode")
            pr.targets.append(np.zeros(22)); pr.record.append(False); pr.sample.append(-1); pr.passno.append(-1)
            for v in np.linspace(-30, 30, 31) * deg:
                t = np.zeros(22)
                t[idx[f"{side}_Revolute67"]] = v
                t[idx[f"{side}_Revolute81"]] = v
                pr.move_to(t, 1.0 * deg, record_end=True, passno=0)
            progs.append(pr)
        rec = run_programs(qs, progs, ramp_steps=4, hold_steps=ARGS.hold, window=4, tag="binding")
        out = {}
        for pid, pr in enumerate(progs):
            sel = rec["program"] == pid
            m = pr.name.split("_calf")[0] + "_Revolute81" if "common" in pr.name else pr.name
            j = idx[m]
            tq = ARGS.kp * (rec["motor_target"][sel, j] - rec["motor_pos"][sel, j])
            if "common" in pr.name:
                ja = idx[pr.name.split("_calf")[0] + "_Revolute67"]
                tq_a = ARGS.kp * (rec["motor_target"][sel, ja] - rec["motor_pos"][sel, ja])
            else:
                tq_a = None
            out[pr.name] = {"motor_deg": np.degrees(rec["motor_pos"][sel, j]).round(2).tolist(),
                            "static_torque_nm": tq.round(3).tolist(),
                            "static_torque_abs_max_nm": float(np.abs(tq).max()),
                            "static_torque_calf_a_abs_max_nm": None if tq_a is None else float(np.abs(tq_a).max()),
                            "closure_gap_max_mm": float(1e3 * rec["gap"][sel].max())}
            print(f"[binding] {pr.name:28s} |tau|max {np.abs(tq).max():8.2f} N*m "
                  f"{'' if tq_a is None else f'(calf A {np.abs(tq_a).max():.2f})'} gap max {1e3 * rec['gap'][sel].max():.2f} mm "
                  f"| tau at sample 0/mid/end: {tq[0]:.2f} {tq[len(tq) // 2]:.2f} {tq[-1]:.2f}", flush=True)
        report["programs"] = out
        report["status"] = "ok"
        rc = 0
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    ARGS.out.write_text(json.dumps(report, indent=1))
    close_app_and_exit(app, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
