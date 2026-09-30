"""Live text -> Dropbear reference service: Kimodo (warm) -> G1->Dropbear retarget -> kinematic clip -> live inbox.

Runs in the WSL venv ``.venv-kimodo-wsl`` (GPU) with the text encoder on Windows (``TEXT_ENCODER_MODE=file``,
``tools/kimodo_text_embed_server.py``). Everything is loaded once; then every prompt file dropped into ``--prompts``
(``<name>.txt`` containing ``prompt`` or ``duration|prompt``) becomes a contract-layout NPZ in ``--inbox``, which a
running ``scripts/play.py --live_dir`` (Isaac, motor twin) or ``tools/policy_runner.py --live_dir`` (deploy) splices
into its reference (``dropbear_wbc.motion.stream``). Per-stage timings go to ``--inbox/../live_service.jsonl``.

The clip is KINEMATIC (not physics-settled, unlike the library clips): motors and root pose from the retarget, the
anchor carried rigidly with the root (exact: it is rigid to the torso), passive joints and the other bodies from a
settled template frame. That is everything the deployable (NoState) tracker observes (reference motor positions /
velocities and the anchor orientation); body-position metrics of such a clip are meaningless. ``meta.kinematic_live``
marks it.

usage (WSL, from the repo root):
  HF_HOME=$DROPBEAR_HF_CACHE HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=file PYTHONPATH=source \\
    .venv-kimodo-wsl/bin/python tools/kimodo_live_service.py --prompts logs/live/prompts --inbox logs/live/inbox
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "data/motions_v6ts/synthetic/stand.npz"
TIME_SCALE = 1.19  # Froude-consistent G1 -> Dropbear (docs/ISSUES.md #27)


def _quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _rotate(q, v):
    w, x, y, z = np.moveaxis(q, -1, 0)
    r = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    return np.einsum("...ij,...j->...i", r, v)


def kinematic_clip(tpl: dict, motor_q: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, fps: float,
                   meta: dict) -> dict:
    """Contract-layout arrays from a retarget result (see module doc)."""
    jn, bn, mn = list(tpl["joint_names"]), list(tpl["body_names"]), list(tpl["motor_names"])
    T = motor_q.shape[0]
    jp = np.repeat(tpl["joint_pos"][:1], T, axis=0).astype(np.float64)
    for k, m in enumerate(mn):
        jp[:, jn.index(m)] = motor_q[:, k]
    r = bn.index("world")
    p0, q0 = tpl["body_pos_w"][0], tpl["body_quat_w"][0]
    q0_inv = q0[r] * np.array([1.0, -1.0, -1.0, -1.0])
    rel_p = _rotate(np.broadcast_to(q0_inv, q0.shape), p0 - p0[r])  # body positions in the root frame
    rel_q = _quat_mul(np.broadcast_to(q0_inv, q0.shape), q0)
    rq = root_quat_wxyz / np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True)
    bp = root_pos[:, None, :] + _rotate(rq[:, None, :].repeat(len(bn), 1), rel_p[None].repeat(T, 0))
    bq = _quat_mul(rq[:, None, :].repeat(len(bn), 1), rel_q[None].repeat(T, 0))
    dt = 1.0 / fps
    jv = np.gradient(jp, dt, axis=0)
    lv = np.gradient(bp, dt, axis=0)
    av = np.zeros_like(lv)  # angular velocities are recomputed by the live stream at the splice
    return {"fps": np.float32(fps), "joint_pos": jp.astype(np.float32), "joint_vel": jv.astype(np.float32),
            "body_pos_w": bp.astype(np.float32), "body_quat_w": bq.astype(np.float32),
            "body_lin_vel_w": lv.astype(np.float32), "body_ang_vel_w": av.astype(np.float32),
            "joint_names": np.asarray(jn), "body_names": np.asarray(bn), "motor_names": np.asarray(mn),
            "closure_residual_m": np.zeros(T, np.float32),
            "meta": np.asarray(json.dumps({**json.loads(str(tpl["meta"])), **meta}))}


def _stretch(x: np.ndarray, s: float) -> np.ndarray:
    """Resample (T, ...) frames to be ``s`` times longer at the same frame rate (linear)."""
    T = x.shape[0]
    u = np.arange(int(round((T - 1) * s)) + 1) / s
    i0 = np.clip(np.floor(u).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (u - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i0] + w * x[i1]


def speed_gate(motor_q: np.ndarray, motor_names: list[str], fps: float, no_load: dict, max_stretch: float) -> dict:
    """Hardware speed screen of a generated clip (same law as tools/hw_motion_feasibility.py): the 99th-percentile
    reference speed of every motor vs its no-load speed. Returns the slow-down factor needed (1.0 = fine), or
    ``refuse`` when more than ``max_stretch`` would be needed."""
    qd = np.abs(np.gradient(motor_q, 1.0 / fps, axis=0))
    ratios = {m: float(np.percentile(qd[:, k], 99) / no_load[m]) for k, m in enumerate(motor_names) if m in no_load}
    worst = max(ratios, key=lambda m: ratios[m])
    need = max(1.0, ratios[worst] * 1.05)  # 5 % margin under the no-load speed
    return {"worst_motor": worst, "p99_over_noload": round(ratios[worst], 3), "stretch": round(need, 3),
            "refuse": need > max_stretch}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--prompts", type=Path, default=REPO / "logs/live/prompts")
    ap.add_argument("--inbox", type=Path, default=REPO / "logs/live/inbox")
    ap.add_argument("--steps", type=int, default=50, help="diffusion steps (100 = kimodo_gen default)")
    ap.add_argument("--default_duration", type=float, default=4.0)
    ap.add_argument("--once", action="store_true", help="process the prompts present now, then exit")
    ap.add_argument("--profile", default="hw_v1i", help="motor map for the speed gate (robots/hw_motor_specs.py)")
    ap.add_argument("--max_stretch", type=float, default=1.6,
                    help="slow a too-fast clip down by at most this factor; beyond it the clip is refused")
    args = ap.parse_args()
    from dropbear_wbc.motion.calibration_view import load_calibration
    from dropbear_wbc.motion.foot_contact import FootContactParams
    from dropbear_wbc.motion.g1_sources import load_g1_motion
    from dropbear_wbc.motion.g1_to_dropbear import RetargetOptions, retarget_g1_motion
    from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.model.load_model import load_model

    t0 = time.perf_counter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model("Kimodo-G1-RP-v1", device=device, default_family="Kimodo")
    conv = MujocoQposConverter(model.skeleton)
    cal = load_calibration(None)
    opts = RetargetOptions(output_fps=50.0, foot_contact=FootContactParams(), max_motor_step_rad=0.18)
    with np.load(TEMPLATE, allow_pickle=False) as d:
        tpl = {k: d[k] for k in d.files}
    no_load = {m: float(v["no_load_speed"]) for m, v in joint_hw_params(HW_PROFILE_MAPS[args.profile]).items()}
    args.prompts.mkdir(parents=True, exist_ok=True)
    args.inbox.mkdir(parents=True, exist_ok=True)
    log = args.inbox.parent / "live_service.jsonl"
    print(f"[live] ready in {time.perf_counter() - t0:.1f} s; prompts {args.prompts} -> inbox {args.inbox}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="kimodo_live_"))
    while True:
        todo = sorted(args.prompts.glob("*.txt"))
        for f in todo:
            raw = f.read_text(encoding="utf-8").strip()
            f.unlink(missing_ok=True)
            dur, prompt = (raw.split("|", 1) if "|" in raw else (args.default_duration, raw))
            dur = float(dur)
            rec = {"name": f.stem, "prompt": prompt, "duration_s": dur}
            ta = time.perf_counter()
            out = model([prompt], [int(dur * model.fps)], constraint_lst=[], num_denoising_steps=args.steps,
                        num_samples=1, multi_prompt=True, num_transition_frames=5, post_processing=False,
                        return_numpy=True)
            csv = tmp / f"{f.stem}.csv"
            conv.save_csv(conv.dict_to_qpos(out, device), str(csv))
            tb = time.perf_counter()
            import dataclasses

            motion = load_g1_motion(csv, "kimodo_g1")
            motion = dataclasses.replace(motion, fps=motion.fps / TIME_SCALE)
            res = retarget_g1_motion(motion, cal, opts)
            tc = time.perf_counter()
            mq, rp, rq = np.asarray(res.motor_q), np.asarray(res.root_pos), np.asarray(res.root_quat_wxyz)
            gate = speed_gate(mq, [str(m) for m in tpl["motor_names"]], float(res.fps), no_load, args.max_stretch)
            rec["speed_gate"] = gate
            if gate["refuse"]:
                rec.update(refused=True, total_s=round(time.perf_counter() - ta, 2))
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[live] {f.stem}: {prompt!r} REFUSED: {gate['worst_motor']} would need "
                      f"{gate['p99_over_noload']:.2f}x its no-load speed (max slow-down {args.max_stretch})", flush=True)
                continue
            if gate["stretch"] > 1.0:  # slow it down to what the real motors can do
                flip = np.concatenate([[1.0], np.where(np.sum(rq[1:] * rq[:-1], axis=-1) < 0, -1.0, 1.0)]).cumprod()
                rq = rq * flip[:, None]  # one hemisphere, so the linear blend below is a valid nlerp
                mq, rp, rq = (_stretch(x, gate["stretch"]) for x in (mq, rp, rq))
                rq = rq / np.linalg.norm(rq, axis=-1, keepdims=True)
            arrays = kinematic_clip(tpl, mq, rp, rq, float(res.fps),
                                    {"kinematic_live": True, "prompt": prompt, "source": "kimodo-g1 live service",
                                     "time_scale": TIME_SCALE, "speed_gate": gate, "motor_profile": args.profile})
            part = args.inbox / f"{f.stem}.npz.part"
            with open(part, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(part, args.inbox / f"{time.strftime('%H%M%S')}_{f.stem}.npz")
            td = time.perf_counter()
            rec.update(generate_s=round(tb - ta, 2), retarget_s=round(tc - tb, 2), write_s=round(td - tc, 2),
                       total_s=round(td - ta, 2), frames=int(arrays["joint_pos"].shape[0]))
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[live] {f.stem}: {prompt!r} -> {rec['frames']} frames in {rec['total_s']} s "
                  f"(gen {rec['generate_s']}, retarget {rec['retarget_s']})", flush=True)
        if args.once and not todo:
            return 0
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
