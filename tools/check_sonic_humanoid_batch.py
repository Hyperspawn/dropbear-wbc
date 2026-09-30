"""Load the DERIVED serial Dropbear model with GR00T SONIC's real ``Humanoid_Batch`` and check its FK.

Imports ``gear_sonic.utils.motion_lib.torch_humanoid_batch.Humanoid_Batch`` from the read-only reference clone
``$DROPBEAR_UPSTREAM/GR00T-WholeBodyControl`` (nothing there is modified), builds it for
``data/robot/dropbear_serial_motionlib.xml`` with the ``extend_config`` from ``dropbear_serial.json``, runs
``fk_batch`` on the motion_lib PKL written by ``tools/export_serial_motionlib.py`` and compares every body
(and every extend point) with MuJoCo FK of the same MJCF. Also runs ``mesh_fk`` (the height-fix path, which needs
the MJCF ``<asset>`` meshes).

Needs ``.venv-gmr`` with torch + SONIC's import deps (easydict, loguru, lxml, omegaconf, hydra-core, open3d)::

    .venv-gmr/Scripts/python.exe tools/check_sonic_humanoid_batch.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
SONIC = _paths.upstream_dir() / "GR00T-WholeBodyControl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "logs/serial_mjcf_gmr/motionlib/dropbear_motionlib.pkl")
    ap.add_argument("--xml", type=Path, default=REPO / "data/robot/dropbear_serial_motionlib.xml")
    ap.add_argument("--out", type=Path, default=REPO / "logs/serial_mjcf_gmr/sonic_humanoid_batch_check.json")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(SONIC))
    import mujoco
    import torch
    from easydict import EasyDict

    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    meta = json.loads((REPO / "data/robot/dropbear_serial.json").read_text())
    ext = [EasyDict(joint_name=e["joint_name"], parent_name=e["parent_name"], pos=e["pos"], rot=e["rot"])
           for e in meta["motionlib_extend_config"]]
    cfg = EasyDict(asset=EasyDict(assetRoot=str(args.xml.parent), assetFileName=args.xml.name), extend_config=ext)
    hb = Humanoid_Batch(cfg)
    lib = pickle.load(open(args.pkl, "rb"))
    m = mujoco.MjModel.from_xml_path(str(args.xml))
    d = mujoco.MjData(m)
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)][1:]
    site_of = {e["joint_name"]: "usd_" + e["usd_body"] for e in meta["motionlib_extend_config"]}
    report = {"xml": str(args.xml).replace("\\", "/"), "sonic_clone": str(SONIC).replace("\\", "/"),
              "num_bodies": hb.num_bodies, "num_extend": hb.num_bodies_augment - hb.num_bodies,
              "num_dof": hb.num_dof, "body_names": hb.body_names, "clips": {}}
    worst = 0.0
    for clip, e in lib.items():
        pose = torch.from_numpy(np.asarray(e["pose_aa"], dtype=np.float32))[None]
        trans = torch.from_numpy(np.asarray(e["root_trans_offset"], dtype=np.float32))[None]
        res = hb.fk_batch(pose, trans, return_full=False)
        gt = res.global_translation_extend[0].double().numpy()      # (T, B + extend, 3)
        t = gt.shape[0]
        err_b, err_e = 0.0, 0.0
        for k in range(0, t, max(1, t // 60)):
            d.qpos[:] = 0
            d.qpos[0:3] = e["root_trans_offset"][k]
            rr = e["root_rot"][k]
            d.qpos[3:7] = [rr[3], rr[0], rr[1], rr[2]]
            for j, n in enumerate(jnames):
                d.qpos[m.jnt_qposadr[j + 1]] = e["dof"][k][j]
            mujoco.mj_kinematics(m, d)
            for bi, bn in enumerate(hb.body_names):
                err_b = max(err_b, float(np.abs(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)] - gt[k, bi]).max()))
            for xi, en in enumerate(hb.body_names_augment[hb.num_bodies:]):
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_of[en])
                err_e = max(err_e, float(np.abs(d.site_xpos[sid] - gt[k, hb.num_bodies + xi]).max()))
        worst = max(worst, err_b, err_e)
        report["clips"][clip] = {"frames": t, "max_body_pos_err_m": err_b, "max_extend_pos_err_m": err_e}
        print(f"[check_sonic_humanoid_batch] {clip}: SONIC Humanoid_Batch vs MuJoCo max err bodies {err_b:.2e} m, "
              f"extend points {err_e:.2e} m")
    mesh = hb.mesh_fk()
    report["mesh_fk_zero_pose"] = {"vertices": int(len(mesh.vertices)),
                                   "min_z_m": float(np.asarray(mesh.vertices)[:, 2].min()),
                                   "note": "zero pose with the root at the origin; min z = -(pelvis height above the lowest vertex)"}
    report["max_err_m"] = worst
    report["ok"] = bool(worst < 1e-4)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[check_sonic_humanoid_batch] bodies {hb.num_bodies}, extend {report['num_extend']}, dof {hb.num_dof}; "
          f"mesh_fk vertices {report['mesh_fk_zero_pose']['vertices']} min z {report['mesh_fk_zero_pose']['min_z_m']:.4f}; "
          f"max err {worst:.2e} m -> {'OK' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
