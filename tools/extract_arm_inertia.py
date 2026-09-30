"""Extract the arm link inertials (mass, centre of mass) from the Newton plant for teleop gravity feed-forward.

xr_teleoperate computes ``tau_ff = pinocchio.rnea(model, q, 0, 0)`` from the URDF inertials. Dropbear has no URDF,
so this tool builds the contract plant exactly as the Newton bridge does (``DropbearNewtonPlant``: contract USD,
CONTRACTS 0.1 fixes, current default ankle variant) **on the CPU** (MuJoCo C, CUDA hidden, no GPU lock needed),
and writes every arm body's mass and world centre of mass at the authored configuration, expressed in the root
(``world`` body) frame, plus the semantic-chain segment each body moves with:

* segment 1: moves with shoulder pitch only (``LH_yaw`` rotor), 2: + shoulder roll (``LH_pitch``),
  3: + shoulder yaw (upper arm, ``LH_roll``), 4: + elbow (forearm body), 5: + wrist roll (hand plate);
* the small elbow four-bar links (hangers, bicep crank/coupler, forearm side link; about 0.11 kg per arm) are
  approximated as upper-arm bodies (segment 3) and flagged ``approx``.

Output: ``data/teleop/arm_inertia.json`` (schema ``dropbear-teleop-arm-inertia-v1``).

Run (about 30-60 s, CPU only)::

    set CUDA_VISIBLE_DEVICES=-1
    .venv-newton/Scripts/python.exe tools/extract_arm_inertia.py --out data/teleop/arm_inertia.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "third_party" / "pydeps")]

import numpy as np  # noqa: E402

SHOULDER = {"left": ("LH_yaw", "LH_pitch", "LH_roll"), "right": ("RH_yaw", "RH_pitch", "RH_roll")}
ELBOW_LOOP = {"left": ("LH_elbow_joint", "LH_Revolute32", "LH_Revolute33", "LH_Revolute41", "LH_Revolute42",
                       "LH_Revolute44", "LH_Revolute123"),
              "right": ("RH_elbow_joint", "RH_Revolute32", "RH_Revolute33", "RH_Revolute41", "RH_Revolute42",
                        "RH_Revolute44", "RH_Revolute123")}
FOREARM = {"left": "LH_6mm_bearing__4__1", "right": "RH_6mm_bearing__4__1"}
WRIST = {"left": "LH_wrist_roll", "right": "RH_wrist_roll"}


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "arm_inertia.json")
    ap.add_argument("--authored-ankle", action="store_true", help="plant variant (irrelevant for the arms)")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        print("refusing: set CUDA_VISIBLE_DEVICES=-1 (CPU-only tool; no GPU lock is taken)", flush=True)
        return 2
    from dropbear_wbc.newton_sim.plant import DropbearNewtonPlant, PlantConfig
    from dropbear_wbc.sdk import motors

    t0 = time.perf_counter()
    cfg = PlantConfig(fixed_base=True, device="cpu", mujoco_cpu=True, use_cuda_graph=False,
                      authored_ankle_tierods=True if a.authored_ankle else None)
    plant = DropbearNewtonPlant(cfg, log=lambda m: print(m, flush=True))
    print(f"[extract] plant built in {time.perf_counter() - t0:.1f} s", flush=True)
    model = plant.model
    body_q = plant.state_0.body_q.numpy()
    mass = model.body_mass.numpy()
    com_local = model.body_com.numpy()
    labels = [lab.rsplit("/", 1)[-1] for lab in plant.body_labels]
    jlabels = [lab.rsplit("/", 1)[-1] for lab in plant.joint_labels]
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    closure = set(plant.closure_joints)
    # tree parent joint of every body (closures excluded: they are equality constraints)
    tree_joint = {}
    for j in range(len(jlabels)):
        if j in closure:
            continue
        tree_joint[int(child[j])] = j
    root = plant.root_body
    rq = body_q[root]
    r_root = quat_xyzw_to_matrix(rq[3:7])
    p_root = rq[:3]
    r0 = plant.initial_readout()
    motor_q = r0.motor_q.tolist()

    def path_joints(b: int) -> list[str]:
        out = []
        while b != root and b in tree_joint:
            j = tree_joint[b]
            out.append(jlabels[j])
            b = int(parent[j])
        return out

    bodies = {}
    for b, name in enumerate(labels):
        pj = path_joints(b)
        side = None
        for s in ("left", "right"):
            if SHOULDER[s][0] in pj:
                side = s
        if side is None:
            continue
        n_sh = sum(m in pj for m in SHOULDER[side])
        approx = False
        if WRIST[side] in pj:
            seg = 5
        elif any(j in pj for j in ELBOW_LOOP[side]):
            if name == FOREARM[side]:
                seg = 4
            else:
                seg, approx = 3, True
        else:
            seg = n_sh
        tf = body_q[b]
        com_w = quat_xyzw_to_matrix(tf[3:7]) @ com_local[b] + tf[:3]
        com_root = r_root.T @ (com_w - p_root)
        origin_root = r_root.T @ (tf[:3] - p_root)
        bodies[name] = {"side": side, "segment": seg, "approx": approx, "mass_kg": float(mass[b]),
                        "com_root": com_root.tolist(), "origin_root": origin_root.tolist(), "tree_path": pj}
    per_side = {s: {"mass_kg": sum(v["mass_kg"] for v in bodies.values() if v["side"] == s),
                    "bodies": sorted(k for k, v in bodies.items() if v["side"] == s)} for s in ("left", "right")}
    out = {
        "schema": "dropbear-teleop-arm-inertia-v1",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "tools/extract_arm_inertia.py (Newton model of the contract plant, CPU)",
        "usd": plant.report.get("usd"), "usd_fixes_contract_0_1": plant.report.get("usd_fixes_contract_0_1"),
        "authored_ankle_tierods": plant.report.get("authored_ankle_tierods"),
        "frame": "root ('world' body) frame at the authored configuration; com_root = centre of mass",
        "motor_names": list(motors.MOTOR_NAMES), "motor_q_at_extraction": motor_q,
        "segments": {"1": "shoulder pitch (yaw-motor rotor)", "2": "+ shoulder roll", "3": "+ shoulder yaw (upper arm)",
                     "4": "+ elbow (forearm body)", "5": "+ wrist roll (hand plate)"},
        "per_side": per_side, "bodies": bodies,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for s in ("left", "right"):
        print(f"[extract] {s}: {per_side[s]['mass_kg']:.3f} kg in {len(per_side[s]['bodies'])} bodies", flush=True)
    for k, v in bodies.items():
        print(f"  {k:42s} {v['side']:5s} seg {v['segment']} {'approx' if v['approx'] else '      '} "
              f"m {v['mass_kg']:.4f} com {np.round(v['com_root'], 4).tolist()}", flush=True)
    print(f"[extract] motor q at extraction (arms): {np.round(motor_q[12:], 4).tolist()}", flush=True)
    print(f"[extract] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
