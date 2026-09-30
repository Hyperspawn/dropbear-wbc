"""Analyse a pose-sequence teleop probe (``tools/teleop_arm.py --scripted-kind poses``): per hold window (last
``--window-s`` of each hold), commanded vs measured semantic / motor angles, the forearm's actual elbow angle from
the simulated hand-plate orientation, wrist positions (target, FK of measured joints, simulated), and measured
motor torque vs the gravity model.

    .venv-teleop/Scripts/python.exe tools/analyze_teleop_probe.py data/teleop/<session> --hold-s 2 --move-s 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_wbc.kinematics.semantic import euler_yxz_matrix  # noqa: E402
from dropbear_wbc.motion.rotations import quat_to_matrix  # noqa: E402
from dropbear_wbc.teleop.arm_ik import ARM_MOTOR_SLOTS, SIDES, DropbearArmIK  # noqa: E402
from dropbear_wbc.teleop.gravity import ArmGravity  # noqa: E402


def actual_elbow(ik: DropbearArmIK, side: str, sim7: np.ndarray, q5: np.ndarray) -> float:
    """Semantic elbow angle of the simulated forearm: hand rotation relative to its authored (straight-arm, elbow =
    e_rest) reference, with the measured shoulder rotation removed."""
    r_ref = np.asarray(ik.calib["reference_rotations_root"][side]["hand"], dtype=float)
    r = quat_to_matrix(sim7[3:]) @ r_ref.T
    f = euler_yxz_matrix(np.asarray(q5[:3], dtype=float))
    m = f.T @ r
    return float(np.arctan2(m[0, 2], m[2, 2]) + ik.chains[side].info["elbow_rest_rad"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--forearm-model", choices=("table", "pivot"), default="table",
                    help="gravity model of the forearm path (teleop.gravity.ArmGravity)")
    ap.add_argument("--move-s", type=float, default=1.0)
    ap.add_argument("--window-s", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = np.load(a.session / "extras" / "episode_000000.npz")
    ik = DropbearArmIK()
    grav = ArmGravity(ik, forearm_model=a.forearm_model)
    t = d["timestamp"][:, 0]
    seg = a.hold_s + a.move_s
    n_pose = int(round(t[-1] / seg))
    rows = []
    for i in range(n_pose):
        t_end = (i + 1) * seg
        m = (t > t_end - a.window_s) & (t <= t_end)
        if m.sum() < 3:
            continue
        row = {"pose": i, "t_window": [float(t[m][0]), float(t[m][-1])]}
        for k, side in enumerate(SIDES):
            sl = slice(5 * k, 5 * k + 5)
            cmd = d["action"][m][:, sl].mean(0)
            meas = d["observation__state"][m][:, sl].mean(0)
            mslots = list(ARM_MOTOR_SLOTS)[5 * k:5 * k + 5]
            m_cmd = d["action__motor_q"][m][:, sl].mean(0)
            m_meas = d["observation__motor_q"][m][:, mslots].mean(0)
            tau = d["observation__motor_tau"][m][:, mslots].mean(0)
            ff = d["action__tau_ff"][m][:, sl].mean(0)
            q_meas10 = d["observation__state"][m].mean(0)
            g_meas = grav.motor_torque(q_meas10, d["observation__motor_q"][m][:, list(ARM_MOTOR_SLOTS)].mean(0))[sl]
            sim = d[f"sim__{side}_wrist"][m]
            tgt = d[f"target__{side}_wrist"][m][:, :3].mean(0)
            fk = d[f"fk__{side}_wrist"][m][:, :3].mean(0)
            e_act = np.mean([actual_elbow(ik, side, s7, q) for s7, q in zip(sim, d["observation__state"][m][:, sl])])
            vel = np.abs(np.diff(sim[:, :3], axis=0)).max() / max(np.median(np.diff(t[m])), 1e-9)
            row[side] = {
                "sem_cmd": cmd.round(4).tolist(), "sem_meas_lut": meas.round(4).tolist(),
                "elbow_actual_from_sim": round(e_act, 4),
                "motor_cmd": m_cmd.round(4).tolist(), "motor_meas": m_meas.round(4).tolist(),
                "tau_meas": tau.round(3).tolist(), "tau_ff_cmd": ff.round(3).tolist(),
                "tau_gravity_model_at_meas": g_meas.round(3).tolist(),
                "wrist_target_minus_sim_mm": (1e3 * (tgt - sim[:, :3].mean(0))).round(1).tolist(),
                "wrist_fk_minus_sim_mm": (1e3 * (fk - sim[:, :3].mean(0))).round(1).tolist(),
                "wrist_sim_speed_mps": round(float(vel), 4),
            }
        rows.append(row)
    out = {"session": str(a.session), "hold_s": a.hold_s, "move_s": a.move_s, "window_s": a.window_s, "poses": rows}
    txt = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(txt + "\n", encoding="utf-8")
    for r in rows:
        L = r["left"]
        print(f"pose {r['pose']}: elbow cmd {L['sem_cmd'][3]:+.3f} lut(meas motor) {L['sem_meas_lut'][3]:+.3f} "
              f"actual {L['elbow_actual_from_sim']:+.3f} | elbow motor cmd {L['motor_cmd'][3]:.3f} meas "
              f"{L['motor_meas'][3]:.3f} tau {L['tau_meas'][3]:+.2f} (grav model {L['tau_gravity_model_at_meas'][3]:+.2f}) "
              f"| target-sim {L['wrist_target_minus_sim_mm']} fk-sim {L['wrist_fk_minus_sim_mm']} "
              f"speed {L['wrist_sim_speed_mps']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
