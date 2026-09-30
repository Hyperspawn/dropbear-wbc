"""Live GUI viewer of a running motor-twin simulation (kinematic copy; this process never steps physics).

The physics process is ``scripts/play.py ... --device cpu --realtime --state_out <file>`` (headless, ~1.1x real time
on the laptop CPU); it publishes env 0's root pose and all joint positions every policy step
(``dropbear_wbc.isaac.state_share``). This process builds the same Play scene with the clean-render look, and in a
loop poses its robot from the newest published state and renders the viewport (``sim.render()`` only; the viewport
camera follows the anchor body per the task's ``ViewerCfg``). Rendering costs ~28 ms per frame here, so it runs at
~30 fps without slowing the physics down. What you see is the SIMULATED robot (motor twin, closures, contacts), not
the reference.

    C:/isaac-sim/python.bat -u scripts/live_viewer.py --state logs/live/state.bin --device cpu

Stops when the window is closed, or ``--idle_exit_s`` after the physics process stops publishing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Live GUI viewer of a running motor-twin simulation.")
parser.add_argument("--state", type=Path, default=_REPO / "logs" / "live" / "state.bin")
parser.add_argument("--task", default="Dropbear-Tracking-Flat-NoState-Play-v0")
parser.add_argument("--scene_motion", type=Path, default=_REPO / "data" / "motions_v6ts" / "synthetic" / "stand.npz",
                    help="any accepted clip: only needed to build the tracking scene (never played)")
parser.add_argument("--idle_exit_s", type=float, default=20.0,
                    help="exit this long after the last new state (0 = never)")
parser.add_argument("--wait_s", type=float, default=600.0, help="how long to wait for the state file to appear")
parser.add_argument("--max_fps", type=float, default=30.0,
                    help="frame-rate cap: leaves CPU time to the physics process (0 = uncapped)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main() -> int:
    import gymnasium as gym
    import torch

    from dropbear_wbc.isaac.state_share import StateReader
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point
    from dropbear_wbc.tasks.tracking.demo_render import apply_clean_render

    register()
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    motion = args.scene_motion if args.scene_motion.is_absolute() else _REPO / args.scene_motion
    env_cfg.commands.motion.motion_file = str(motion.resolve())
    env_cfg.commands.motion.allow_rejected_motion = True  # scene building only
    env_cfg.set_num_envs(1)
    env_cfg.sim.device = args.device
    apply_clean_render(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()
    robot = uenv.scene["robot"]
    dev = uenv.device
    ids = torch.tensor([0], dtype=torch.long, device=dev)
    origin = uenv.scene.env_origins[0]
    print(f"[viewer] scene ready; waiting for {args.state}", flush=True)
    reader = StateReader(args.state if args.state.is_absolute() else _REPO / args.state, wait_s=args.wait_s)
    if reader.joint_names != list(robot.joint_names):
        raise SystemExit("joint order of the state file differs from this scene's robot (different USD?)")
    zeros6 = torch.zeros(1, 6, device=dev)
    zeros_j = torch.zeros(1, reader.n, device=dev)
    frames, shown, t_last_new, t_rep = 0, 0, time.time(), time.time()
    first_sim_t, first_wall = None, None
    stats = {"frames": 0, "states_shown": 0}
    print(f"[viewer] following {reader.path}", flush=True)
    t_frame = time.time()
    while app.is_running():
        if args.max_fps > 0:
            rest = 1.0 / args.max_fps - (time.time() - t_frame)
            if rest > 0:
                time.sleep(rest)
        t_frame = time.time()
        st = reader.read()
        now = time.time()
        if st is not None:
            if first_sim_t is None:
                first_sim_t, first_wall = st["sim_t"], now
            pos = torch.as_tensor(st["root_pos"], dtype=torch.float32, device=dev) + origin
            quat = torch.as_tensor(st["root_quat_wxyz"], dtype=torch.float32, device=dev)
            robot.write_root_state_to_sim(torch.cat([pos.unsqueeze(0), quat.unsqueeze(0), zeros6], -1), env_ids=ids)
            jp = torch.as_tensor(st["joint_pos"], dtype=torch.float32, device=dev).unsqueeze(0)
            robot.write_joint_state_to_sim(jp, zeros_j, env_ids=ids)
            shown += 1
            t_last_new = now
        uenv.sim.render()
        frames += 1
        if now - t_rep >= 5.0:
            sim_rate = ((st or {}).get("sim_t", 0.0) - first_sim_t) / max(now - first_wall, 1e-9) if (
                st is not None and first_sim_t is not None) else None
            print(json.dumps({"viewer": {"fps": round(frames / (now - t_rep), 1), "new_states": shown,
                                         "physics_rtf_seen": None if sim_rate is None else round(sim_rate, 3)}}),
                  flush=True)
            stats["frames"] += frames
            stats["states_shown"] += shown
            frames, shown, t_rep = 0, 0, now
        if args.idle_exit_s > 0 and first_sim_t is not None and now - t_last_new > args.idle_exit_s:
            print(f"[viewer] no new state for {args.idle_exit_s:.0f} s: exiting", flush=True)
            break
    print(json.dumps({"viewer_summary": stats}), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
