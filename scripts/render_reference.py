"""Clean render of a motion REFERENCE (kinematic replay of a settled NPZ; no policy, no physics integration).

demo_eval track (2026-09-24). Builds the tracking Play scene with ``demo_render.apply_clean_render`` (no markers,
neutral light), then for every NPZ frame writes the FULL settled joint row + the ``world`` root state of env 0 and
renders (``sim.render()`` -> ``forward()`` updates the articulation kinematics; the physics is never stepped, so the
robot is shown exactly where the reference puts it -- floating or penetrating feet included). Output: one mp4 per
camera (``demo_render.DemoRecorder``) + ``<tag>_reference_render.json``.

Use it to preview a new clip before training and to show what a policy was asked to track (e.g. the G1_Take_102
reference whose stance feet float up to ~14 cm above the ground: ``--video_cams track,feet``).

    python tools/gpu_lock_run.py --owner demo_eval --log logs/demo_eval/ref_x.log -- C:/isaac-sim/python.bat -u \
        scripts/render_reference.py --motion_file data/motions/synthetic/wave_right_v2.npz \
        --video_dir logs/demo_eval/media --video_tag wave_right_v2_reference --headless
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Clean kinematic render of a motion NPZ reference.")
parser.add_argument("--task", default="Dropbear-Tracking-Flat-Play-v0")
parser.add_argument("--motion_file", type=Path, required=True)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--end_frame", type=int, default=None, help="exclusive (default: clip end)")
parser.add_argument("--video_cams", default="track,front")
parser.add_argument("--video_res", default="1280x720")
parser.add_argument("--video_dir", type=Path, default=_REPO / "logs" / "demo_eval" / "media")
parser.add_argument("--video_tag", default=None, help="default <clip>_reference")
parser.add_argument("--video_caption", default=None, help="literal '\\n' separates lines")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.demo_render import DemoRecorder, apply_clean_render, parse_resolution

    register()
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    motion = (args.motion_file if args.motion_file.is_absolute() else _REPO / args.motion_file).resolve()
    env_cfg.commands.motion.motion_file = str(motion)
    env_cfg.commands.motion.allow_rejected_motion = True  # visualisation only: nothing is trained or evaluated
    env_cfg.set_num_envs(1)
    env_cfg.sim.device = args.device
    changes = apply_clean_render(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()
    cmd = uenv.command_manager.get_term("motion")
    mo = cmd.motion
    robot = uenv.scene["robot"]
    T = int(mo.time_step_total)
    f0, f1 = max(0, args.start_frame), min(T, args.end_frame if args.end_frame is not None else T)
    origin = uenv.scene.env_origins[0:1]
    ids = torch.tensor([0], dtype=torch.long, device=uenv.device)
    root_xy = mo.root_pos_w[:, :2].detach().cpu().numpy()
    cur = {"t": f0}
    validation = getattr(mo, "validation", None)
    verdict = (validation or {}).get("verdict") if isinstance(validation, dict) else None
    caption = args.video_caption if args.video_caption is not None else (
        f"Dropbear | REFERENCE {motion.stem} (settled NPZ; kinematic replay; no physics)\n"
        f"validation verdict: {verdict or 'n/a'}")
    rec = DemoRecorder(env=uenv, out_dir=args.video_dir, tag=args.video_tag or f"{motion.stem}_reference",
                       cams=tuple(c.strip() for c in args.video_cams.split(",") if c.strip()),
                       resolution=parse_resolution(args.video_res), caption=caption.replace("\\n", "\n"),
                       reference_xy=root_xy, root_xy_fn=lambda: root_xy[cur["t"]])
    for t in range(f0, f1):
        cur["t"] = t
        tt = torch.tensor([t], dtype=torch.long, device=uenv.device)
        root = torch.cat([mo.root_pos_w[tt] + origin, mo.root_quat_w[tt], mo.root_lin_vel_w[tt], mo.root_ang_vel_w[tt]], -1)
        robot.write_root_state_to_sim(root, env_ids=ids)
        robot.write_joint_state_to_sim(mo.full_joint_pos[tt], mo.full_joint_vel[tt], env_ids=ids)
        rec.capture(force_render=True)
    outputs = rec.close()
    summary = {"motion_file": str(motion), "frames": [f0, f1], "fps": float(mo.fps), "validation": validation,
               "render": rec.describe() | {"outputs": outputs, "render_cfg": changes},
               "note": "kinematic replay: joint + root state written per frame, physics never stepped"}
    out_json = Path(args.video_dir) / f"{rec.tag}_reference_render.json"
    out_json.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"reference_render": summary["render"]["outputs"], "summary_file": str(out_json)}), flush=True)
    env.close()
    ok = all(o["ffmpeg_rc"] == 0 and o["frames"] == f1 - f0 for o in outputs.values())
    return 0 if ok and np.isfinite(root_xy).all() else 2


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
