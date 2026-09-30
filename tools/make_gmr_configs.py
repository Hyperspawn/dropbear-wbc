"""Generate GMR IK configs for the DERIVED serial Dropbear model from the G1 / H1 templates.

    bvh_lafan1_to_dropbear.json   <- GMR ik_configs/bvh_lafan1_to_g1.json (LAFAN1 bone names)
    smplx_to_dropbear.json        <- GMR ik_configs/smplx_to_h1.json      (H1 is human-sized like Dropbear)

Method (all numbers are recorded in each file's ``_dropbear`` block):

* **Body mapping** template robot body -> Dropbear serial body (G1: identical link names except the hand
  ``left_wrist_yaw_link`` -> ``left_wrist_roll_link``; H1: ``left_hip_roll_link`` (H1's thigh-orientation
  target) -> ``left_hip_yaw_link`` (Dropbear thigh), ``left_shoulder_roll_link`` -> ``left_shoulder_yaw_link``
  (upper arm), ``left_ankle_link`` -> ``left_ankle_roll_link``).
* **Rotation offsets.** GMR drives ``R_body = R_human_bone * offset``. G1, H1 and the Dropbear serial model
  share the same anatomical zero pose (legs straight, arms hanging, forearms pointing forward), so
  ``offset_dropbear = offset_template * R_template_body(q=0)^T * R_dropbear_body(q=0)`` with both zero-pose body
  rotations from MuJoCo FK (Dropbear frames are world-aligned at q=0 except the knee/elbow/wrist links, which are
  rotated so their joint axes are principal axes).
* **Scale** (hip-height ratio). LAFAN1: ``s = H_dropbear / (H_lafan * 1.75 / 1.8)`` for every body, with
  ``H_dropbear`` = the serial model's pelvis height above the sole at the zero pose (serial model metadata;
  within a few mm of the calibration's ``standing_hip_height``, the difference is the knee four-bar fit's zero shift) and ``H_lafan`` = 95th percentile of the ``Hips`` height over the walking
  part of ``walk1_subject1`` (GMR's LAFAN loader; 1.75/1.8 is GMR's fixed LAFAN height ratio). SMPL-X (no data
  on disk): ``s = s_H1 * L_dropbear / L_H1`` with ``L`` = vertical pelvis-to-ankle distance at the zero pose.
* **Weights.** Copied from the template except: arm POSITION weights 0 (Dropbear's forearm is 0.105 m, so human
  hand/elbow positions are unreachable; the arms follow bone orientations) and hip-joint/knee POSITION weights 0
  (Dropbear's thigh:shank ratio is 0.59:0.32 m and its knee pivot is a fitted four-bar centre, so human knee
  positions are not meaningful targets; the legs follow thigh/shank orientations and foot poses).
* **Root policy (``--root-policy``, added 2026-09-24).** G1 has a 3-DoF waist between ``pelvis`` and ``torso_link``;
  in the Dropbear serial model ``torso_link`` is WELDED to the pelvis, so the template's torso (``Spine2``)
  rotation task (weight 100) and pelvis (``Hips``) rotation task (weight 10) act on one rigid body and the pelvis
  followed the human CHEST (Hips error p50 13-20 deg, pelvis pitched +9..+13 deg on walk/run). ``blend`` (default)
  gives both tasks the pelvis weight, so the rigid torso takes the least-squares compromise between the human pelvis
  and chest orientations; ``chest`` keeps the template weights (previous behaviour); ``pelvis`` keeps only a weak
  chest term (weight 1).

Run with ``.venv-gmr`` (needs mujoco + scipy)::

    .venv-gmr/Scripts/python.exe tools/make_gmr_configs.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from dropbear_wbc import paths as _paths  # noqa: E402
GMR = _paths.upstream_dir() / "GMR-dropbear"
OUT = REPO / "data/robot/gmr"

G1_MAP = {"pelvis": "pelvis", "torso_link": "torso_link",
          **{f"{s}_{a}": f"{s}_{b}" for s in ("left", "right") for a, b in (
              ("hip_yaw_link", "hip_yaw_link"), ("knee_link", "knee_link"), ("ankle_roll_link", "ankle_roll_link"),
              ("shoulder_yaw_link", "shoulder_yaw_link"), ("elbow_link", "elbow_link"),
              ("wrist_yaw_link", "wrist_roll_link"))}}
H1_MAP = {"pelvis": "pelvis", "torso_link": "torso_link",
          **{f"{s}_{a}": f"{s}_{b}" for s in ("left", "right") for a, b in (
              ("hip_roll_link", "hip_yaw_link"), ("knee_link", "knee_link"), ("ankle_link", "ankle_roll_link"),
              ("shoulder_roll_link", "shoulder_yaw_link"), ("elbow_link", "elbow_link"))}}
ARM_BODIES = ("shoulder", "elbow", "wrist")
LEG_JOINT_BODIES = ("hip_yaw_link", "knee_link")
LAFAN_WALK = _paths.upstream_dir() / "lafan1" / "bvh" / "walk1_subject1.bvh"
LAFAN_WALK_WINDOW_S = (5.0, 60.0)   # walking part (the clip starts with a T-pose)


def lafan_hip_height(bvh: Path = LAFAN_WALK) -> float:
    """95th percentile of the LAFAN Hips height [m] (GMR loader frame) over the walking window."""
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, _ = load_bvh_file(str(bvh))
    a, b = (int(30 * t) for t in LAFAN_WALK_WINDOW_S)
    z = np.array([f["Hips"][0][2] for f in frames[a:b]])
    return float(np.percentile(z, 95))


TEMPLATES = {
    "bvh_lafan1_to_dropbear.json": ("bvh_lafan1_to_g1.json", "unitree_g1/g1_mocap_29dof.xml", G1_MAP,
                                    ("left_hip_pitch_link", "left_ankle_pitch_link")),
    "smplx_to_dropbear.json": ("smplx_to_h1.json", "unitree_h1/h1.xml", H1_MAP,
                               ("left_hip_yaw_link", "left_ankle_link")),
}


def zero_pose_frames(xml: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)
    rot, pos = {}, {}
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        rot[n] = d.xmat[i].reshape(3, 3).copy()
        pos[n] = d.xpos[i].copy()
    return rot, pos


def main(argv: list[str] | None = None) -> int:
    from scipy.spatial.transform import Rotation as R

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmr", type=Path, default=GMR)
    ap.add_argument("--xml", type=Path, default=REPO / "data/robot/dropbear_serial.xml")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--lafan-hip-height", type=float, default=None, help="override the measured LAFAN Hips height [m]")
    ap.add_argument("--root-policy", choices=("blend", "chest", "pelvis"), default="blend",
                    help="waistless torso: which human frame the rigid pelvis+torso follows (see module doc)")
    args = ap.parse_args(argv)
    meta = json.loads(args.xml.with_suffix(".json").read_text())
    db_rot, db_pos = zero_pose_frames(args.xml)
    l_db = float(db_pos["pelvis"][2] - db_pos["left_ankle_pitch_link"][2])
    args.out.mkdir(parents=True, exist_ok=True)
    for out_name, (tmpl_name, xml_rel, bmap, (hip_b, ank_b)) in TEMPLATES.items():
        tmpl = json.loads((args.gmr / "general_motion_retargeting/ik_configs" / tmpl_name).read_text())
        t_rot, t_pos = zero_pose_frames(args.gmr / "assets" / xml_rel)
        l_t = float(t_pos["pelvis"][2] - t_pos[ank_b][2])
        cfg = copy.deepcopy(tmpl)
        ratio = l_db / l_t
        if out_name.startswith("bvh_lafan1"):
            h_human = lafan_hip_height() if args.lafan_hip_height is None else args.lafan_hip_height
            h_db = float(meta["frames"]["pelvis_height_above_sole_at_zero_m"])
            gmr_ratio = 1.75 / tmpl["human_height_assumption"]
            scale = h_db / (h_human * gmr_ratio)
            scale_info = {"method": "hip height ratio", "dropbear_pelvis_height_m": h_db,
                          "lafan_hips_height_p95_m": h_human, "lafan_source": f"{LAFAN_WALK.name} {LAFAN_WALK_WINDOW_S} s",
                          "gmr_lafan_height_ratio": gmr_ratio}
            cfg["human_scale_table"] = {k: round(scale, 4) for k in tmpl["human_scale_table"]}
        else:
            scale_info = {"method": "template scale * pelvis-to-ankle ratio", "ratio": ratio}
            cfg["human_scale_table"] = {k: round(v * ratio, 4) for k, v in tmpl["human_scale_table"].items()}
        changes = {}
        for table in ("ik_match_table1", "ik_match_table2"):
            new = {}
            for tb, entry in tmpl[table].items():
                if tb not in bmap:
                    raise KeyError(f"{tmpl_name}: no Dropbear body for template body {tb}")
                human, wp, wr, poff, roff = entry
                db_body = bmap[tb]
                if db_body not in db_rot:
                    raise KeyError(f"Dropbear MJCF lacks {db_body}")
                off = R.from_quat(roff, scalar_first=True) * R.from_matrix(t_rot[tb]).inv() * R.from_matrix(db_rot[db_body])
                q = off.as_quat(scalar_first=True)
                q = q if q[0] >= 0 else -q
                if any(a in db_body for a in ARM_BODIES) or db_body.endswith(LEG_JOINT_BODIES):
                    wp = 0
                if db_body == "torso_link":
                    pelvis_wr = tmpl[table][next(k for k, v in bmap.items() if v == "pelvis")][2]
                    wr = {"blend": pelvis_wr, "chest": wr, "pelvis": 1}[args.root_policy]
                new[db_body] = [human, wp, wr, list(poff), [round(float(x), 8) for x in q]]
                changes[f"{table}:{tb}"] = {"dropbear_body": db_body, "template_offset_wxyz": roff,
                                            "template_body_zero_quat_wxyz": np.round(
                                                R.from_matrix(t_rot[tb]).as_quat(scalar_first=True), 6).tolist(),
                                            "pos_weight": wp}
            cfg[table] = new
        cfg["robot_root_name"] = "pelvis"
        root_weights = {t: {"torso_link(Spine2)": cfg[t]["torso_link"][2], "pelvis(Hips)": cfg[t]["pelvis"][2]}
                        for t in ("ik_match_table1", "ik_match_table2") if "torso_link" in cfg[t]}
        cfg["_dropbear"] = {
            "generated_by": "tools/make_gmr_configs.py",
            "template": f"GMR ik_configs/{tmpl_name} ({xml_rel})",
            "serial_model": str(args.xml).replace("\\", "/"),
            "serial_model_calibration_sha256": meta["inputs"]["calibration_sha256"],
            "scale": scale_info, "pelvis_to_ankle_pitch_m": {"dropbear": l_db, "template": l_t},
            "rotation_offset_rule": "offset_dropbear = offset_template * R_template_body(q=0)^T * R_dropbear_body(q=0) "
                                    "(same anatomical zero pose; MuJoCo FK of both models)",
            "position_weights_zeroed": "arms (forearm 0.105 m: human hand positions unreachable) and hip-joint/knee "
                                       "(thigh:shank 0.59:0.32 m, knee = fitted four-bar pivot)",
            "body_changes": changes,
            "root_policy": {
                "policy": args.root_policy,
                "reason": "torso_link is welded to the pelvis in the serial model (G1 has a 3-DoF waist): the Spine2 and "
                          "Hips rotation tasks act on one rigid body",
                "rotation_weights": root_weights,
            },
            "status": "DERIVED serial model; not canonical",
        }
        (args.out / out_name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[make_gmr_configs] {out_name}: template {tmpl_name}, scale {json.dumps(scale_info)}, "
              f"scales {sorted(set(cfg['human_scale_table'].values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
