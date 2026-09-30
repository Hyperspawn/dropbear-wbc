"""GPU probe of the motion-LIBRARY tracking env (Isaac Lab 2.2): builds the env on a manifest and checks the command
term against the NPZ files directly. One Isaac process, one env (default: the -Future training config, which exercises
everything: adaptive sampling, RSI, the plain and the future command slices).

Checks (all written to ``--out``):
 1. build: library fingerprint, obs dims (policy/critic), action dim, bins, memory of the concatenated tensors;
 2. RSI sampling after ``env.reset``: clip histogram vs p(c), local times inside each clip;
 3. command slices == reference motor pos/vel at (clip, t), (clip, t+5), (clip, t+10) read from the NPZ with numpy
    (future frames clamped to the clip end), for every env; the reference tracked-body poses == NPZ + env origin;
 4. zero-action rollout (``--steps``): finiteness, terminations and failures per clip, clip hazard / probabilities and
    per-clip metrics logged by the command (they must react: clips that fail get more probability);
 5. play-mode reset (start_at_zero, all RSI noise off): env i on clip i % N frame 0 and the written joint + root state
    equal the NPZ frame-0 row (all 91 joints).

    python tools/gpu_lock_run.py --owner multiclip --log logs/multiclip/library_env_probe.log --wait-minutes 120 -- \
        C:/isaac-sim/python.bat -u tools/library_env_probe.py --motion_library data/motions/libraries/accepted_v0.json \
        --num_envs 256 --steps 150 --out logs/multiclip/library_env_probe.json --headless
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "third_party" / "pydeps"), str(_REPO / "source")]
from dropbear_wbc.isaac.launch import prepare_kit_python  # noqa: E402

prepare_kit_python()
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Dropbear-Tracking-Library-Future-v0")
parser.add_argument("--motion_library", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--solver_iters", type=int, nargs=2, default=(8, 4))
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def main() -> dict:  # noqa: C901
    import gymnasium as gym
    import numpy as np
    import torch

    from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES, TRACKED_BODIES
    from dropbear_wbc.tasks import register
    from dropbear_wbc.tasks.registry import load_entry_point

    register()
    env_cfg = load_entry_point(args.task, "env_cfg_entry_point")
    manifest = (args.motion_library if args.motion_library.is_absolute() else _REPO / args.motion_library).resolve()
    env_cfg.commands.motion.manifest = str(manifest)
    env_cfg.set_num_envs(args.num_envs)
    env_cfg.set_solver_iterations(*args.solver_iters)
    env_cfg.commands.motion.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    lib = cmd.motion
    future = tuple(cmd.future_steps)
    rep: dict = {"task": args.task, "manifest": str(manifest), "num_envs": uw.num_envs, "solver_iters": list(args.solver_iters),
                 "library": {k: v for k, v in cmd.library_info.items() if k != "clips"},
                 "clips": [{k: c[k] for k in ("name", "num_frames", "verdict", "sha256")} for c in cmd.library_info["clips"]],
                 "obs_dim_policy": int(uw.observation_manager.group_obs_dim["policy"][0]),
                 "obs_dim_critic": int(uw.observation_manager.group_obs_dim["critic"][0]),
                 "obs_terms_policy": list(uw.observation_manager.active_terms["policy"]),
                 "action_dim": int(uw.action_manager.total_action_dim), "future_steps": list(future),
                 "bin_count": int(cmd.bin_count), "bins_per_clip": cmd.sampler.bins_per_clip.tolist(),
                 "checks": {}}
    chk = rep["checks"]
    chk["obs_dim_expected"] = rep["obs_dim_policy"] == 125 + 44 * len(future)

    # -- NPZ ground truth (numpy, independent of the library code path)
    npz = []
    for c in cmd.library_info["clips"]:
        d = np.load(c["npz"], allow_pickle=False)
        jn = [str(j) for j in d["joint_names"]]
        bn = [str(b) for b in d["body_names"]]
        npz.append({"jp": np.asarray(d["joint_pos"]), "jv": np.asarray(d["joint_vel"]),
                    "motor_cols": [jn.index(m) for m in MOTOR_NAMES], "body_cols": [bn.index(b) for b in TRACKED_BODIES],
                    "root_col": bn.index("world"), "bp": np.asarray(d["body_pos_w"]), "bq": np.asarray(d["body_quat_w"])})

    def check_reference(tag: str) -> dict:
        obs = uw.observation_manager.compute()["policy"]
        cid = cmd.clip_ids.cpu().numpy()
        t = cmd.time_steps.cpu().numpy()
        origins = uw.scene.env_origins.cpu().numpy()
        body = cmd.body_pos_w.cpu().numpy()
        err_cmd, err_fut, err_body = 0.0, 0.0, 0.0
        for e in range(uw.num_envs):
            n = npz[cid[e]]
            T = n["jp"].shape[0]
            k = min(int(t[e]), T - 1)
            ref = np.concatenate([n["jp"][k, n["motor_cols"]], n["jv"][k, n["motor_cols"]]])
            err_cmd = max(err_cmd, float(np.abs(obs[e, :44].cpu().numpy() - ref).max()))
            for j, off in enumerate(future):
                kf = min(k + off, T - 1)
                fut = np.concatenate([n["jp"][kf, n["motor_cols"]], n["jv"][kf, n["motor_cols"]]])
                err_fut = max(err_fut, float(np.abs(obs[e, 44 * (j + 1):44 * (j + 2)].cpu().numpy() - fut).max()))
            err_body = max(err_body, float(np.abs(body[e] - (n["bp"][k, n["body_cols"]] + origins[e])).max()))
        hist = np.bincount(cid, minlength=cmd.num_clips)
        lengths = lib.lengths.cpu().numpy()
        return {"tag": tag, "command_max_abs_err": err_cmd, "future_max_abs_err": err_fut if future else None,
                "body_pos_max_abs_err_m": err_body, "clip_histogram": hist.tolist(),
                "time_inside_clip": bool(np.all(t < lengths[cid])) and bool(np.all(t >= 0)),
                "envs_near_clip_end": int(np.sum(t + (max(future) if future else 0) >= lengths[cid]))}

    obs, _ = env.reset()
    chk["after_reset"] = check_reference("after_reset")
    chk["after_reset"]["clip_probs"] = cmd.sampler.clip_probs().tolist()

    # -- zero-action rollout (default pose): which clips fail, does the sampler react
    act = torch.zeros(uw.num_envs, rep["action_dim"], device=uw.device)
    finite = True
    fails = np.zeros(cmd.num_clips, dtype=np.int64)
    trace = []
    for k in range(args.steps):
        clip_before = cmd.clip_ids.clone()
        obs, rew, term, trunc, extras = env.step(act)
        finite &= bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
        f = term.bool()
        if bool(f.any()):
            fails += np.bincount(clip_before[f].cpu().numpy(), minlength=cmd.num_clips)
        if k % 25 == 0 or k == args.steps - 1:
            trace.append({"step": k, "clip_probs": [round(x, 4) for x in cmd.sampler.clip_probs().tolist()],
                          "clip_fail_per_s": [round(x * lib.fps, 4) for x in cmd.sampler.clip_hazard().tolist()],
                          "clip_err_joint": [round(x, 4) for x in cmd.clip_err_joint_ema.tolist()],
                          "sampling_entropy": float(cmd.metrics["sampling_entropy"][0]),
                          "terminated_this_step": int(f.sum())})
    rep["zero_action_rollout"] = {"steps": args.steps, "finite": finite, "failures_per_clip": dict(zip(cmd.clip_names, fails.tolist())),
                                  "trace": trace,
                                  "logged_metric_keys": sorted(k for k in cmd.metrics if k.startswith("clip_"))}
    probs = np.asarray(cmd.sampler.clip_probs().tolist())
    prior = np.asarray(cmd.sampler.clip_prior().tolist())
    haz = np.asarray(cmd.sampler.clip_hazard().tolist())
    # the clip distribution moved away from the prior towards the clips that failed (None: nothing failed)
    chk["sampler_reacts"] = (bool(haz.sum() > 0 and probs[int(np.argmax(haz))] > prior[int(np.argmax(haz))])
                             if haz.sum() > 0 else None)
    chk["clip_prior"], chk["clip_probs_final"] = prior.round(4).tolist(), probs.round(4).tolist()
    chk["metrics_finite"] = all(math.isfinite(float(v.float().mean())) for v in cmd.metrics.values())
    chk["after_rollout"] = check_reference("after_rollout")

    # -- play-mode reset: env i -> clip i % N at frame 0, no noise; the written state is the NPZ frame-0 row
    cmd.cfg.start_at_zero = True
    cmd.cfg.pose_range, cmd.cfg.velocity_range = {}, {}
    cmd.cfg.joint_position_range = (0.0, 0.0)
    cmd.cfg.closure_joint_position_range = (0.0, 0.0)
    all_ids = torch.arange(uw.num_envs, device=uw.device)
    cmd._resample_command(all_ids)
    cid = cmd.clip_ids.cpu().numpy()
    jp = robot.data.joint_pos.cpu().numpy()
    rp = robot.data.root_link_pos_w.cpu().numpy() - uw.scene.env_origins.cpu().numpy()
    jerr, rerr = 0.0, 0.0
    for e in range(uw.num_envs):
        n = npz[cid[e]]
        jerr = max(jerr, float(np.abs(jp[e] - n["jp"][0]).max()))
        rerr = max(rerr, float(np.abs(rp[e] - n["bp"][0, n["root_col"]]).max()))
    chk["play_reset"] = {"assignment_ok": bool(np.all(cid == np.arange(uw.num_envs) % cmd.num_clips)),
                         "time_zero": bool((cmd.time_steps == 0).all()), "joint_pos_max_abs_err": jerr,
                         "root_pos_max_abs_err_m": rerr}
    chk["play_reset_reference"] = check_reference("play_reset")
    ok = (chk["obs_dim_expected"] and chk["after_reset"]["command_max_abs_err"] < 1e-5
          and (not future or chk["after_reset"]["future_max_abs_err"] < 1e-5)
          and chk["after_reset"]["body_pos_max_abs_err_m"] < 1e-4 and chk["after_reset"]["time_inside_clip"]
          and finite and chk["metrics_finite"] and chk["play_reset"]["assignment_ok"]
          and chk["play_reset"]["joint_pos_max_abs_err"] < 1e-4 and chk["play_reset"]["root_pos_max_abs_err_m"] < 1e-4
          and chk["after_rollout"]["command_max_abs_err"] < 1e-5)
    rep["ok"] = bool(ok)
    rep["note"] = "Interface/indexing checks and a zero-action rollout; not a tracking result."
    env.close()
    return rep


if __name__ == "__main__":
    rc = 1
    report: dict = {}
    try:
        report = main()
        rc = 0 if report.get("ok") else 2
    except Exception:  # noqa: BLE001
        report = {"error": traceback.format_exc()}
        traceback.print_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report.get("ok"), "out": str(args.out), "checks": report.get("checks")}, default=str)[:4000],
          flush=True)
    from dropbear_wbc.isaac.launch import close_app_and_exit

    close_app_and_exit(app, rc)
