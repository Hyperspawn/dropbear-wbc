"""Per-motor torque / speed statistics of a play rollout (the ``motor_torque`` summary block).

Shared by ``scripts/play_locomotion.py`` and ``scripts/play.py`` so locomotion and tracking summaries have the same
block, computed the same way, for ``tools/scoreboard.py``'s hardware gate. Isaac-free (torch only): the actuator
objects are duck-typed (``joint_names``, ``effort_limit``, ``applied_effort``, ``computed_effort``), so the
aggregation is unit-tested on CPU (tests/test_motor_torque.py).

Sample = one env at the last physics substep of one policy step:

* ``|applied_effort|``: the motor torque the actuator model delivered (for hw_* profiles after the torque-speed
  envelope, before gearbox friction; :mod:`dropbear_wbc.robots.hw_actuators`);
* clipped = ``|computed_effort - applied_effort| > clip_tol * peak``: the model cut the PD torque (peak limit or the
  torque-speed line);
* ``|joint_vel|`` of the motor joint.

The caller decides which envs are valid for a step (``mask``); envs whose joint state was rewritten in that step
(reset, reference wrap) must be masked out, because ``joint_vel`` then holds the written state, not the rollout.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from dropbear_wbc.robots.hw_introspect import actuator_effort_limit

KEYS = ("peak_limit_Nm", "max_Nm", "p95_Nm", "rms_Nm", "clip_frac", "max_speed_rad_s", "p95_speed_rad_s")
CLIP_TOL = 0.01


def motor_columns(actuators, motor_names: Sequence[str]) -> list[tuple[int, object, int]]:
    """(motor index in ``motor_names``, actuator, column in that actuator) for every motor. Raises if a motor is in
    no actuator group (it would otherwise be reported as 0 N*m and look clean)."""
    names = list(motor_names)
    cols = [(names.index(n), act, k) for act in actuators.values() for k, n in enumerate(act.joint_names) if n in names]
    seen = [i for i, _, _ in cols]
    missing = [names[i] for i in range(len(names)) if i not in seen]
    dup = sorted({names[i] for i in seen if seen.count(i) > 1})
    if missing or dup:
        raise ValueError(f"motor_torque: motors in no actuator group {missing}, in several groups {dup}")
    return cols


def tracking_sample_mask(dones: torch.Tensor, ts_prev: torch.Tensor, ts_now: torch.Tensor,
                         wrap_rewrites: bool) -> torch.Tensor:
    """Envs of one tracking policy step whose post-step joint state is the rollout's.

    Excluded: envs reset in the step (``dones``: Isaac Lab wrote the reset state before ``env.step`` returned) and, when
    the clip end rewrites the robot (``reset_at_wrap``, i.e. not ``--continuous_loop``), envs whose reference clock did
    not advance by exactly one frame (``ts_now != ts_prev + 1``: the command update wrapped and wrote frame 0)."""
    ok = ~dones.to(torch.bool)
    if wrap_rewrites:
        ok &= ts_now == ts_prev + 1
    return ok


class MotorTorqueSampler:
    """Collect masked per-step samples on the device; :meth:`block` reduces them to the summary dict."""

    def __init__(self, actuators, motor_names: Sequence[str], num_envs: int, device, clip_tol: float = CLIP_TOL):
        self.names = tuple(motor_names)
        self.cols = motor_columns(actuators, self.names)
        self.num_envs, self.device, self.clip_tol = int(num_envs), device, float(clip_tol)
        self.peak_lim = torch.zeros(len(self.names), device=device)
        for i, act, k in self.cols:
            self.peak_lim[i] = actuator_effort_limit(act)[0, k]
        self._tq: list[torch.Tensor] = []
        self._cl: list[torch.Tensor] = []
        self._vl: list[torch.Tensor] = []
        self._grp: list[torch.Tensor] = []

    def efforts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(applied, computed) motor torques [num_envs, num_motors] of the last physics substep."""
        applied = torch.zeros(self.num_envs, len(self.names), device=self.device)
        computed = torch.zeros_like(applied)
        for i, act, k in self.cols:
            applied[:, i] = act.applied_effort[:, k]
            computed[:, i] = act.computed_effort[:, k]
        return applied, computed

    def add(self, joint_vel_motors: torch.Tensor, mask: torch.Tensor, groups: torch.Tensor | None = None,
            efforts: tuple[torch.Tensor, torch.Tensor] | None = None) -> int:
        """Record the valid envs of one policy step. ``joint_vel_motors``: [num_envs, num_motors] in ``motor_names``
        order; ``mask``: [num_envs] bool; ``groups``: optional [num_envs] long (e.g. library clip id). Returns the
        number of samples added."""
        mask = mask.to(torch.bool)
        if not bool(mask.any()):
            return 0
        applied, computed = efforts if efforts is not None else self.efforts()
        self._tq.append(applied[mask].abs())
        self._cl.append((computed[mask] - applied[mask]).abs() > self.clip_tol * self.peak_lim)
        self._vl.append(joint_vel_motors[mask].abs())
        if groups is not None:
            self._grp.append(groups[mask].to(torch.long))
        return int(mask.sum())

    @property
    def samples(self) -> int:
        return sum(int(t.shape[0]) for t in self._tq)

    def _per_motor(self, tq: torch.Tensor, cl: torch.Tensor, vl: torch.Tensor) -> dict:
        # identical reductions / rounding to play_locomotion.py before the refactor (tests pin it)
        return {name: {"peak_limit_Nm": round(float(self.peak_lim[i]), 2), "max_Nm": round(float(tq[:, i].max()), 2),
                       "p95_Nm": round(float(torch.quantile(tq[:, i], 0.95)), 2),
                       "rms_Nm": round(float(tq[:, i].pow(2).mean().sqrt()), 2),
                       "clip_frac": round(float(cl[:, i].mean()), 4),
                       "max_speed_rad_s": round(float(vl[:, i].max()), 2),
                       "p95_speed_rad_s": round(float(torch.quantile(vl[:, i], 0.95)), 2)}
                for i, name in enumerate(self.names)}

    def _cat(self):
        return torch.cat(self._tq), torch.cat(self._cl).float(), torch.cat(self._vl)

    def block(self, note: str, **extra) -> dict | None:
        """``{"note", "samples", "per_motor", **extra}`` over all samples, or None if nothing was recorded (a missing
        block is what the scoreboard reports as torque UNKNOWN)."""
        if not self._tq:
            return None
        tq, cl, vl = self._cat()
        return {"note": note, **extra, "samples": int(tq.shape[0]), "per_motor": self._per_motor(tq, cl, vl)}

    def group_blocks(self) -> dict[int, dict]:
        """Per-group ``{"samples", "per_motor"}`` (groups passed to :meth:`add`; groups without samples are absent)."""
        if not self._tq or not self._grp:
            return {}
        if len(self._grp) != len(self._tq):
            raise ValueError("motor_torque: groups were passed to add() for some steps only")
        tq, cl, vl = self._cat()
        g = torch.cat(self._grp)
        out = {}
        for gid in torch.unique(g).tolist():
            sel = g == gid
            out[int(gid)] = {"samples": int(sel.sum()), "per_motor": self._per_motor(tq[sel], cl[sel], vl[sel])}
        return out
