"""Run motor-target programs (:mod:`dropbear_wbc.kinematics.sweeps`) on the quasi-static articulation.

Every env follows one program. Programs are packed longest-first into batches of ``num_envs``; within a
batch all envs advance stage by stage in lock step. A stage moves each env's targets from the previous
stage's value to the new one with a linear ramp of ``ramp_steps`` physics steps (programs keep every
stage increment <= their ``max_step``, so linkages stay on their assembly branch) and, if any env records
at that stage, holds for ``hold_steps``. Fixed step counts: no velocity-based settle loop.

For every recorded (env, stage) the full state is stored together with the per-stage quality metrics of
:meth:`dropbear_wbc.isaac.quasistatic.QuasiStaticDropbear.move` (closure gaps, joint position change over
the last ``window`` steps, motor tracking error). Must run inside an Isaac Lab app.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from dropbear_wbc.kinematics.sweeps import Program, pack


def run_programs(qs, programs: list[Program], ramp_steps: int, hold_steps: int, window: int = 4,
                 log_every: int = 25, tag: str = "sweep") -> dict[str, np.ndarray]:
    """Run ``programs`` on ``qs`` (a ``QuasiStaticDropbear``); return stacked numpy records.

    Keys (R = number of records): ``program`` ``sample`` ``passno`` ``stage`` (R,), ``motor_target``
    ``motor_pos`` (R, 22), ``joint_pos`` (R, J), ``body_pos`` (R, B, 3), ``body_quat`` (R, B, 4),
    ``gap`` (R,) worst closure gap [m], ``gaps`` (R, C), ``dq_window`` (R,), ``motor_err`` (R,), plus scalars
    ``wall_s``, ``physics_steps``, ``ms_per_step``.
    """
    keys = ("program", "sample", "passno", "stage", "motor_target", "motor_pos", "joint_pos", "body_pos",
            "body_quat", "gap", "gaps", "dq_window", "motor_err")
    rows: dict[str, list] = {k: [] for k in keys}
    batches = pack(programs, qs.num_envs)
    t0 = time.time()
    steps0 = qs.physics_steps
    for bi, batch in enumerate(batches):
        zero = torch.zeros(qs.num_envs, len(qs.joint_names), device=qs.device)
        qs.write_joint_state(zero)
        qs.step(4)
        current = torch.zeros(qs.num_envs, 22, device=qs.device)
        s_count = batch.targets.shape[0]
        for s in range(s_count):
            tg = torch.tensor(batch.targets[s], dtype=torch.float32, device=qs.device)
            rec = batch.record[s] & (batch.program_ids >= 0)
            info = qs.move(current, tg, ramp_steps, hold_steps if rec.any() else 0, window)
            current = tg
            if rec.any():
                st = qs.read_state()
                gap = info["gap"].cpu().numpy().astype(np.float64)
                dq = info["dq_window"].cpu().numpy().astype(np.float64)
                me = info["motor_err"].cpu().numpy().astype(np.float64)
                for e in np.nonzero(rec)[0]:
                    rows["program"].append(int(batch.program_ids[e]))
                    rows["sample"].append(int(batch.sample[s, e]))
                    rows["passno"].append(int(batch.passno[s, e]))
                    rows["stage"].append(s)
                    rows["motor_target"].append(batch.targets[s, e])
                    rows["motor_pos"].append(st["motor_pos"][e])
                    rows["joint_pos"].append(st["joint_pos"][e])
                    rows["body_pos"].append(st["body_pos"][e])
                    rows["body_quat"].append(st["body_quat"][e])
                    rows["gap"].append(gap[e])
                    rows["gaps"].append(st["gaps"][e])
                    rows["dq_window"].append(dq[e])
                    rows["motor_err"].append(me[e])
            if s % log_every == 0 or s == s_count - 1:
                act = batch.program_ids >= 0
                g = info["gap"][torch.as_tensor(act, device=qs.device)]
                print(f"[{tag}] batch {bi + 1}/{len(batches)} stage {s + 1}/{s_count} "
                      f"worst_gap={1e3 * float(g.max()):.3f} mm median_gap={1e3 * float(g.median()):.3f} mm "
                      f"max_dq={float(info['dq_window'].max()):.2e} records={len(rows['program'])} "
                      f"wall={time.time() - t0:.1f}s", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    wall = time.time() - t0
    steps = qs.physics_steps - steps0
    out["wall_s"] = np.asarray(wall)
    out["physics_steps"] = np.asarray(steps)
    out["ms_per_step"] = np.asarray(1e3 * wall / max(steps, 1))
    return out
