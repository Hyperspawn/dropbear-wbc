"""Export Dropbear semantic trajectories in the formats serial-tree motion tools consume (DERIVED serial model).

* **SONIC / PHC motion_lib PKL** (GR00T-WholeBodyControl ``gear_sonic`` motion_lib, ProtoMotions-style):
  ``{clip: {root_trans_offset (T,3), pose_aa (T, B, 3), dof (T, 22), root_rot (T, 4) xyzw, fps}}`` for
  ``data/robot/dropbear_serial_motionlib.xml`` (B = its 23 bodies: pelvis + 22 one-hinge links; ``pose_aa[:, 0]`` =
  root rotation vector, ``pose_aa[:, k] = axis_k * dof`` for the k-th body in document order, as
  ``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`` builds it for G1).
* **mjlab / BeyondMimic CSV** (``mjlab/scripts/csv_to_npz.py`` input): no header, per frame
  ``pelvis pos (3), pelvis quat xyzw (4), 22 joint angles`` in the MJCF joint order (written to ``joint_names.txt``).

Input: the ``semantic/<clip>.semantic.csv`` files of contract clips (pelvis pose + the 22 semantic angles that the
motors actually realise), e.g. ``data/motions/gmr_lafan1``. The PKL is checked by re-implementing the
``Humanoid_Batch`` forward kinematics (body offsets, body quats, integer joint axes) in numpy and comparing with
MuJoCo FK of the same MJCF (``--check``, default on).

    python tools/export_serial_motionlib.py data/motions/gmr_lafan1 --out logs/serial_mjcf_gmr/motionlib

Outputs inherit the source licenses (LAFAN1: CC BY-NC-ND 4.0, internal R&D only).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.motion.rotations import quat_to_matrix, xyzw_to_wxyz  # noqa: E402
from dropbear_wbc.kinematics.rigid import axis_angle_matrix, rotvec  # noqa: E402

MOTIONLIB_XML = REPO / "data/robot/dropbear_serial_motionlib.xml"
SEM_COLS = 7 + 22


def parse_humanoid_batch(xml: Path) -> dict:
    """Body tree as PHC ``Humanoid_Batch.from_mjcf`` reads it (document order)."""
    root = ET.parse(xml).getroot()
    body0 = root.find("worldbody").find("body")
    names, parents, offs, quats, axes, jnames = [], [], [], [], [], []

    def add(node, parent):
        idx = len(names)
        names.append(node.attrib["name"])
        parents.append(parent)
        offs.append(np.fromstring(node.attrib.get("pos", "0 0 0"), sep=" "))
        quats.append(np.fromstring(node.attrib.get("quat", "1 0 0 0"), sep=" "))
        js = node.findall("joint")
        if parent >= 0:
            if len(js) != 1:
                raise ValueError(f"{names[-1]}: Humanoid_Batch needs exactly one joint per non-root body")
            axes.append([int(v) for v in js[0].attrib["axis"].split(" ")])
            jnames.append(js[0].attrib["name"])
        for c in node.findall("body"):
            add(c, idx)

    add(body0, -1)
    return {"names": names, "parents": np.array(parents), "offsets": np.array(offs), "quats": np.array(quats),
            "dof_axis": np.array(axes, dtype=np.float64), "joint_names": jnames}


def phc_fk(tree: dict, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof: np.ndarray):
    """numpy re-implementation of Humanoid_Batch.forward_kinematics_batch (sequential version)."""
    t = len(dof)
    nb = len(tree["names"])
    local_rot = quat_to_matrix(tree["quats"])                                  # (B, 3, 3)
    joint_rot = axis_angle_matrix(np.broadcast_to(tree["dof_axis"], (t, nb - 1, 3)), dof)   # (T, B-1, 3, 3)
    pos = np.zeros((t, nb, 3))
    rot = np.zeros((t, nb, 3, 3))
    pos[:, 0] = root_pos
    rot[:, 0] = quat_to_matrix(root_quat_wxyz)
    for i in range(1, nb):
        p = tree["parents"][i]
        pos[:, i] = pos[:, p] + np.einsum("tij,j->ti", rot[:, p], tree["offsets"][i])
        rot[:, i] = rot[:, p] @ local_rot[i] @ joint_rot[:, i - 1]
    return pos, rot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_dir", type=Path, help="contract clip folder with semantic/<clip>.semantic.csv")
    ap.add_argument("--out", type=Path, default=REPO / "logs/serial_mjcf_gmr/motionlib")
    ap.add_argument("--xml", type=Path, default=MOTIONLIB_XML)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES

    tree = parse_humanoid_batch(args.xml)
    order = [SEMANTIC_NAMES.index(n) for n in tree["joint_names"]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mjlab_csv").mkdir(exist_ok=True)
    (args.out / "mjlab_csv" / "joint_names.txt").write_text("\n".join(tree["joint_names"]) + "\n", encoding="utf-8")
    lib, report = {}, {"xml": str(args.xml).replace("\\", "/"), "bodies": tree["names"], "clips": {}}
    for side_json in sorted(args.clip_dir.glob("*.json")):
        side = json.loads(side_json.read_text())
        if "semantic_trajectory" not in side or not side.get("semantic_trajectory"):
            continue
        clip = side["clip"]
        sem = np.loadtxt(args.clip_dir / side["semantic_trajectory"], delimiter=",", skiprows=1, ndmin=2)
        ppos, pquat = sem[:, 0:3], xyzw_to_wxyz(sem[:, 3:7])
        dof = sem[:, 7:SEM_COLS][:, order]
        t = len(dof)
        pose_aa = np.zeros((t, len(tree["names"]), 3), dtype=np.float32)
        pose_aa[:, 1:] = tree["dof_axis"][None] * dof[:, :, None]
        pose_aa[:, 0] = rotvec(quat_to_matrix(pquat))
        lib[clip] = {"root_trans_offset": ppos.astype(np.float32), "pose_aa": pose_aa, "dof": dof.astype(np.float32),
                     "root_rot": sem[:, 3:7].astype(np.float32), "fps": int(round(side["fps"]))}
        np.savetxt(args.out / "mjlab_csv" / f"{clip}.csv", np.concatenate([ppos, sem[:, 3:7], dof], axis=1),
                   delimiter=",", fmt="%.7f")
        entry = {"frames": t, "fps": side["fps"], "license": side["source_license"]}
        if not args.no_check:
            import mujoco

            m = mujoco.MjModel.from_xml_path(str(args.xml))
            d = mujoco.MjData(m)
            pos, rot = phc_fk(tree, ppos, pquat, dof)
            bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in tree["names"]]
            jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in tree["joint_names"]]
            err_p, err_r = 0.0, 0.0
            for k in range(0, t, max(1, t // 60)):
                d.qpos[:] = 0
                d.qpos[0:3], d.qpos[3:7] = ppos[k], pquat[k]
                d.qpos[jadr] = dof[k]
                mujoco.mj_kinematics(m, d)
                err_p = max(err_p, float(np.abs(d.xpos[bid] - pos[k]).max()))
                err_r = max(err_r, float(np.abs(d.xmat[bid].reshape(-1, 3, 3) - rot[k]).max()))
            entry["phc_fk_vs_mujoco_max_pos_err_m"] = err_p
            entry["phc_fk_vs_mujoco_max_rotmat_err"] = err_r
            if err_p > 1e-5 or err_r > 1e-5:
                raise SystemExit(f"{clip}: PHC-style FK disagrees with MuJoCo ({err_p:.2e} m, {err_r:.2e})")
        report["clips"][clip] = entry
        print(f"[export_serial_motionlib] {clip}: {t} frames; {json.dumps({k: v for k, v in entry.items() if 'err' in k})}")
    with open(args.out / "dropbear_motionlib.pkl", "wb") as f:
        pickle.dump(lib, f)
    report["pkl"] = str(args.out / "dropbear_motionlib.pkl").replace("\\", "/")
    report["note"] = ("SONIC motion_lib loads joblib; joblib.load reads plain pickles. Robot config: asset "
                      "dropbear_serial_motionlib.xml + extend_config from dropbear_serial.json 'motionlib_extend_config'.")
    (args.out / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
