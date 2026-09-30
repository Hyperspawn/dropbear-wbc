"""Per-motor telemetry of one env: the data behind the actuator dashboard (``tools/render_actuator_dashboard.py``).

Per policy step (one row per rendered video frame at 50 Hz) for the 22 body motors, in ``MOTOR_NAMES`` order:
position, velocity, position target (what motion mode would send as ``p_des``), motor torque after the actuator model
(``applied_effort``: what the motor delivers; for ``hw_*`` profiles after the torque-speed envelope), the PD torque
before clipping, and -- if ``substeps`` -- the motor torque of EVERY physics step (200 Hz), so torque spikes between
policy steps are visible. Base velocity / anchor height / feet contact + height ride along, plus the world positions of
the two soles and the root and the projected gravity (for foot-slip / sway checks in ``tools/telemetry_issue_scan.py``).

Import only after the Isaac Sim app is running.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dropbear_wbc.robots.dropbear_names import FOOT_BODIES, MOTOR_NAMES
from dropbear_wbc.robots.hw_motor_specs import MOTOR_CAN_IDS


class MotorTelemetry:
    """Record env ``env_id`` of ``robot``; call :meth:`record` after every ``env.step`` and :meth:`save` at the end."""

    def __init__(self, robot, env_id: int = 0, substeps: bool = True, anchor_body: str | None = None):
        self.robot, self.e = robot, int(env_id)
        self.names = list(MOTOR_NAMES)
        self.ids = robot.find_joints(self.names, preserve_order=True)[0]
        self.cols = [(self.names.index(n), act, k) for act in robot.actuators.values()
                     for k, n in enumerate(act.joint_names) if n in self.names]
        self.peak = np.zeros(len(self.names))
        for i, act, k in self.cols:
            self.peak[i] = float(act.effort_limit[self.e, k])
        self.foot_ids = [robot.body_names.index(b) for b in FOOT_BODIES if b in robot.body_names]
        self.anchor = robot.body_names.index(anchor_body) if anchor_body else None
        self.rows: dict[str, list] = {k: [] for k in ("q", "qd", "q_target", "tau", "tau_pd", "tau_sub", "base_vel_b",
                                                      "yaw_rate", "anchor_z", "cmd", "contact", "feet_z",
                                                      "feet_lateral", "sole_pos_w", "root_pos_w", "proj_grav")}
        self._sub: list[np.ndarray] = []
        self._cur = np.zeros(len(self.names))
        if substeps:
            self._wrap_actuators()

    def _wrap_actuators(self) -> None:
        motor_acts = []
        for _, act, _ in self.cols:
            if act not in motor_acts:
                motor_acts.append(act)
        order = [a for a in self.robot.actuators.values() if a in motor_acts]
        last = order[-1]
        for act in order:
            cols = [(i, k) for i, a, k in self.cols if a is act]
            orig = act.compute

            def wrapped(*args, _orig=orig, _act=act, _cols=cols, **kwargs):
                out = _orig(*args, **kwargs)
                row = _act.applied_effort[self.e].detach().cpu().numpy()
                for i, k in _cols:
                    self._cur[i] = row[k]
                if _act is last:
                    self._sub.append(self._cur.copy())
                return out

            act.compute = wrapped

    def record(self, cmd=None, contact=None) -> None:
        r, e = self.robot, self.e
        tau = np.zeros(len(self.names))
        pd = np.zeros(len(self.names))
        for i, act, k in self.cols:
            tau[i] = float(act.applied_effort[e, k])
            pd[i] = float(act.computed_effort[e, k])
        self.rows["q"].append(r.data.joint_pos[e, self.ids].cpu().numpy())
        self.rows["qd"].append(r.data.joint_vel[e, self.ids].cpu().numpy())
        self.rows["q_target"].append(r.data.joint_pos_target[e, self.ids].cpu().numpy())
        self.rows["tau"].append(tau)
        self.rows["tau_pd"].append(pd)
        sub = np.array(self._sub[-8:]) if self._sub else tau[None]
        self.rows["tau_sub"].append(sub)
        self._sub.clear()
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        v = quat_apply_inverse(yaw_quat(r.data.root_quat_w[e:e + 1]), r.data.root_lin_vel_w[e:e + 1])[0]
        self.rows["base_vel_b"].append(v.cpu().numpy())
        self.rows["yaw_rate"].append(float(r.data.root_ang_vel_w[e, 2]))
        self.rows["anchor_z"].append(float(r.data.body_link_pos_w[e, self.anchor, 2]) if self.anchor is not None else 0.0)
        self.rows["cmd"].append(np.asarray(cmd.detach().cpu() if isinstance(cmd, torch.Tensor) else (cmd or [0, 0, 0]),
                                           dtype=float))
        self.rows["contact"].append(np.asarray(contact.detach().cpu() if isinstance(contact, torch.Tensor)
                                               else (contact or [0, 0]), dtype=bool))
        self.rows["feet_z"].append(r.data.body_link_pos_w[e, self.foot_ids, 2].cpu().numpy())
        # sole separation along the root yaw-frame y axis (left - right; < 0 = crossed legs)
        from dropbear_wbc.robots.dropbear_names import FOOT_EE_BODIES

        s = [r.body_names.index(b) for b in FOOT_EE_BODIES]
        dp = (r.data.body_link_pos_w[e, s[0]] - r.data.body_link_pos_w[e, s[1]])[None]
        self.rows["feet_lateral"].append(float(quat_apply_inverse(yaw_quat(r.data.root_quat_w[e:e + 1]), dp)[0, 1]))
        self.rows["sole_pos_w"].append(r.data.body_link_pos_w[e, s].cpu().numpy())  # (left, right) soles
        self.rows["root_pos_w"].append(r.data.root_link_pos_w[e].cpu().numpy())
        self.rows["proj_grav"].append(r.data.projected_gravity_b[e].cpu().numpy())

    def save(self, path: Path, dt: float, meta: dict | None = None) -> Path:
        from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = max((len(s) for s in self.rows["tau_sub"]), default=1)
        tau_sub = np.full((len(self.rows["tau_sub"]), n, len(self.names)), np.nan)
        for t, s in enumerate(self.rows["tau_sub"]):
            tau_sub[t, :len(s)] = s
        meta = dict(meta or {})
        profile = meta.get("actuator_profile", "")
        spec = {}
        if profile in HW_PROFILE_MAPS:
            p = joint_hw_params(HW_PROFILE_MAPS[profile])
            spec = {k: np.array([p[m][k] for m in self.names]) for k in
                    ("rated_torque", "no_load_speed", "saturation_effort")}
            spec["model"] = np.array([p[m]["model"] for m in self.names])
        arrays = {k: np.array(v) for k, v in self.rows.items() if k != "tau_sub"}
        spec["joint_limits"] = self.robot.data.joint_pos_limits[self.e, self.ids].cpu().numpy()
        np.savez_compressed(path, dt=dt, motor_names=np.array(self.names),
                            can_ids=np.array([MOTOR_CAN_IDS[m] for m in self.names]), peak_torque=self.peak,
                            tau_sub=tau_sub, meta=np.array(__import__("json").dumps(meta)), **arrays, **spec)
        return path
