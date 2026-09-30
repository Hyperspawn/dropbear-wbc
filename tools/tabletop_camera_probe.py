"""GPU camera probe for the tabletop scene: head-camera pose candidates and blank-frame rate per camera class.

Why: the first GPU smoke (`logs/tabletop/smoke_v2.*`) showed (1) the head camera at root (0.10, -0.07, 1.78) sits inside
the head (visor) mesh, and (2) intermittent uniform frames (mean ~217, std ~1.4) on TiledCamera sensors, including a
static third-person camera. This probe runs the scripted pusher on the two smoke placements and records, per camera and
step, the image mean/std (a frame with std < 4 counts as BLANK), saving PNG frames at a few steps.

Cameras (all rigidly on the anchor body except the scene camera):
* ``tiled_<cand>``: TiledCamera at each head pose candidate (``HEAD_CANDIDATES``);
* ``cam_<cand>``: the standard (non-tiled) ``Camera`` at the same poses (``--camera_class both``);
* ``scene``: the static third-person TiledCamera of the smoke.

    python tools/gpu_lock_run.py --owner tabletop --log logs/tabletop/camera_probe_v1.log --timeout 900 -- \
        C:/isaac-sim/python.bat -u tools/tabletop_camera_probe.py --headless --out logs/tabletop/camera_probe_v1
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

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=_REPO / "logs/tabletop/camera_probe_v1")
ap.add_argument("--steps", type=int, default=160)
ap.add_argument("--camera_class", default="both", choices=["tiled", "camera", "both"])
ap.add_argument("--aa", default=None, choices=[None, "Off", "FXAA", "DLSS", "TAA", "DLAA"])
ap.add_argument("--candidates", nargs="+", default=None)
ap.add_argument("--save_steps", type=int, nargs="+", default=[0, 30, 60, 90, 120, 150])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

# name: (pos root frame, pitch down deg, focal mm)
HEAD_CANDIDATES = {
    "visor_hi": ((0.135, -0.0697, 1.88), 68.0, 10.0),   # 1.2 cm in front of the visor, eye height
    "visor_lo": ((0.135, -0.0697, 1.80), 68.0, 10.0),   # in front of the visor's lower edge
    "fwd": ((0.21, -0.0697, 1.84), 65.0, 10.0),         # 9 cm forward mount (clears the chest handle)
    "old": ((0.10, -0.0697, 1.78), 58.0, 10.0),         # the smoke_v2 pose (inside the visor), reference
}


def main() -> int:  # noqa: C901
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tabletop import cameras as cam
    from dropbear_wbc.tasks.tabletop.config import PUSH_ID
    from dropbear_wbc.tasks.tabletop.kinematics import TabletopArmIK
    from dropbear_wbc.tasks.tabletop.layout import LAYOUT, load_placements
    from dropbear_wbc.tasks.tabletop.mdp import PlacementFeed
    from dropbear_wbc.tasks.tabletop.scripted import ScriptedPushPolicy
    from dropbear_wbc.tasks.tabletop.tabletop_env_cfg import ARM_JOINTS, scene_camera_cfg
    from dropbear_wbc.teleop.arm_ik import DropbearArmIK
    from dropbear_wbc.teleop.gravity import ArmGravity
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    cands = {k: v for k, v in HEAD_CANDIDATES.items() if args.candidates is None or k in args.candidates}
    env_cfg = load_entry_point(PUSH_ID, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 2
    env_cfg.sim.device = args.device
    if args.aa:
        env_cfg.sim.render = sim_utils.RenderCfg(antialiasing_mode=args.aa)
    sensors = {}
    r_anchor = cam.matrix_from_quat(cam.ANCHOR_REST_QUAT_ROOT)
    for name, (pos, pitch, focal) in cands.items():
        p = math.radians(pitch)
        r_cam = cam.look_rotation(np.array([math.cos(p), 0.0, -math.sin(p)]))
        off_p = tuple(float(v) for v in r_anchor.T @ (np.asarray(pos) - cam.ANCHOR_REST_POS_ROOT))
        off_q = tuple(float(v) for v in cam.quat_from_matrix(r_anchor.T @ r_cam))
        spawn = sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=20.955,
                                           clipping_range=(0.02, 20.0))
        for cls_name, cls in (("tiled", TiledCameraCfg), ("cam", CameraCfg)):
            if args.camera_class not in (cls_name if cls_name == "tiled" else "camera", "both"):
                continue
            key = f"{cls_name}_{name}"
            sensors[key] = cls(prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam.ANCHOR_BODY}/{key}",
                               offset=CameraCfg.OffsetCfg(pos=off_p, rot=off_q, convention="world"),
                               data_types=["rgb"], spawn=spawn, width=640, height=480)
    sensors["scene"] = scene_camera_cfg()
    for k, v in sensors.items():
        setattr(env_cfg.scene, k, v)
    env = gym.make(PUSH_ID, cfg=env_cfg)
    uw = env.unwrapped
    placements, _ = load_placements(_REPO / "logs/tabletop/smoke_placements.json", "heldout")
    uw.tabletop_feed = PlacementFeed(list(placements), loop=True)
    robot = uw.scene["robot"]
    motor_ids, _ = robot.find_joints(list(MOTOR_NAMES), preserve_order=True)
    arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    base = DropbearArmIK()
    grav = ArmGravity(base)
    iks = [TabletopArmIK(base=base) for _ in range(2)]
    pols = [ScriptedPushPolicy(iks[i]) for i in range(2)]
    q_rest = base.rest_q()
    root_w = torch.tensor(LAYOUT.root_pos_w, device=uw.device)
    env.reset()
    q22 = robot.data.joint_pos[:, motor_ids].cpu().numpy().astype(np.float64)
    for i in range(2):
        p = uw.tabletop_feed.current[i]
        pols[i].reset(p.side, p.zone_xy, base.motor_to_semantic_arms_fast(q22[i]))
    stats = {k: [] for k in sensors}
    t0 = time.perf_counter()
    for step in range(args.steps):
        q22 = robot.data.joint_pos[:, motor_ids].cpu().numpy().astype(np.float64)
        bpos = (uw.scene["block"].data.root_pos_w - uw.scene.env_origins - root_w).cpu().numpy()
        bq = uw.scene["block"].data.root_quat_w.cpu().numpy()
        for k in sensors:
            im = uw.scene[k].data.output["rgb"].cpu().numpy()[..., :3]
            for e in range(im.shape[0]):
                s = float(im[e].std())
                stats[k].append({"step": step, "env": e, "mean": round(float(im[e].mean()), 2), "std": round(s, 2)})
                if step in args.save_steps:
                    Image.fromarray(np.ascontiguousarray(im[e], dtype=np.uint8)).save(
                        args.out / f"{k}_env{e}_s{step:03d}.png")
        act = np.zeros((2, 10))
        tau = np.zeros((2, 10))
        for i in range(2):
            qm = base.motor_to_semantic_arms_fast(q22[i])
            w, x, y, z = bq[i]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            q_cmd, _ = pols[i].act(bpos[i], yaw, qm) if step >= 6 else (q_rest, {})
            act[i] = iks[i].to_motor10(q_cmd)
            tau[i] = grav.motor_torque(q_cmd, act[i])
        robot.set_joint_effort_target(torch.tensor(tau, dtype=torch.float32, device=uw.device), joint_ids=arm_ids)
        env.step(torch.tensor(act, dtype=torch.float32, device=uw.device))
    wall = time.perf_counter() - t0
    summary = {"schema": "dropbear-tabletop-camera-probe-v1", "aa": args.aa, "steps": args.steps,
               "wall_s": round(wall, 1), "candidates": {k: {"pos_root": v[0], "pitch_deg": v[1], "focal_mm": v[2]}
                                                        for k, v in cands.items()},
               "blank_std_threshold": 4.0, "cameras": {}}
    for k, rows in stats.items():
        stds = np.array([r["std"] for r in rows])
        blank = stds < 4.0
        summary["cameras"][k] = {"frames": int(len(rows)), "blank": int(blank.sum()),
                                 "blank_steps_env0": [r["step"] for r in rows if r["env"] == 0 and r["std"] < 4.0][:40],
                                 "std_p50": float(np.median(stds)), "mean_p50": float(np.median([r["mean"] for r in rows]))}
        print(f"[probe] {k}: blank {int(blank.sum())}/{len(rows)} std p50 {np.median(stds):.1f}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (args.out / "per_frame.json").write_text(json.dumps(stats) + "\n", encoding="utf-8")
    print(f"[probe] wrote {args.out} ({wall:.0f} s)", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
    finally:
        from dropbear_wbc.isaac.launch import close_app_and_exit

        close_app_and_exit(app, rc)
