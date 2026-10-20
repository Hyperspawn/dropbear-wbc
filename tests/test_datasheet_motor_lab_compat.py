"""DatasheetMotor runs on both Isaac Lab actuator APIs (notes/15 B, backlog item 66), without Isaac Sim.

Two minimal fake ``isaaclab`` packages reproduce exactly the API points that differ between 2.2 and 3.0:
2.2 has ``ActuatorBase._parse_joint_parameter`` + ``effort_limit``/``velocity_limit`` and calls ``_clip_effort(effort)``;
3.0 has module-level ``resolve_joint_parameter`` + ``actuator_effort_limit``/``actuator_velocity_limit`` (the old names
are DeprecationWarning aliases) and calls ``_clip_effort(effort, joint_vel)``. The motor law must match
``hw_motor_specs.hw_motor_torque`` on both, and the cfg must use the running version's canonical limit names."""
from __future__ import annotations

import dataclasses
import importlib
import re
import sys
import types
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _dense(value, names):
    if isinstance(value, (int, float)):
        return [float(value)] * len(names)
    out = []
    for n in names:
        hits = [v for k, v in value.items() if re.fullmatch(k, n)]
        out.append(float(hits[0]) if hits else 0.0)
    return out


def _param(value, default, names, num_envs, device):
    p = torch.zeros(num_envs, len(names), device=device)
    if value is not None:
        p[:] = torch.tensor(_dense(value, names))
    elif isinstance(default, torch.Tensor):
        p[:] = default
    elif default is not None:
        p[:] = float(default)
    else:
        raise ValueError("no value")
    return p


class _DelayBuffer:  # identity (randomize=False keeps max_delay 0, so compute never calls it)
    def __init__(self, *a, **k):
        pass

    def set_time_lag(self, *a):
        pass

    def reset(self, *a):
        pass

    def compute(self, x):
        return x


@dataclasses.dataclass
class _Actions:
    joint_positions: object = None
    joint_velocities: object = None
    joint_efforts: object = None


def _make_lab(version: str) -> dict[str, types.ModuleType]:
    lab3 = version == "3.0"

    @dataclasses.dataclass(kw_only=True)
    class IdealPDActuatorCfg:
        joint_names_expr: list
        stiffness: object = None
        damping: object = None
        armature: object = None
        friction: object = None
        class_type: type = None
        if lab3:
            actuator_effort_limit: object = None
            actuator_velocity_limit: object = None
            joint_effort_limit: object = None
            joint_velocity_limit: object = None
        effort_limit: object = None  # 3.0: deprecated alias
        velocity_limit: object = None
        effort_limit_sim: object = None
        velocity_limit_sim: object = None

        def __post_init__(self):
            if lab3:
                for new, old in (("actuator_effort_limit", "effort_limit"), ("actuator_velocity_limit", "velocity_limit"),
                                 ("joint_effort_limit", "effort_limit_sim"), ("joint_velocity_limit", "velocity_limit_sim")):
                    if getattr(self, old) is not None:
                        warnings.warn(f"deprecated '{old}'", DeprecationWarning)
                        setattr(self, new, getattr(self, old))
                        setattr(self, old, None)

    @dataclasses.dataclass(kw_only=True)
    class ImplicitActuatorCfg(IdealPDActuatorCfg):
        pass

    class IdealPDActuator:
        is_implicit_model = False

        def __init__(self, cfg, joint_names, joint_ids, num_envs, device):
            self.cfg, self._joint_names, self._num_envs, self._device = cfg, list(joint_names), num_envs, device
            self.computed_effort = torch.zeros(num_envs, len(joint_names))
            self.applied_effort = torch.zeros_like(self.computed_effort)
            if lab3:
                self.actuator_effort_limit = _param(cfg.actuator_effort_limit, torch.inf, joint_names, num_envs, device)
                self.actuator_velocity_limit = _param(cfg.actuator_velocity_limit, torch.inf, joint_names, num_envs, device)
                self.stiffness = _param(cfg.stiffness, 0.0, joint_names, num_envs, device)
                self.damping = _param(cfg.damping, 0.0, joint_names, num_envs, device)
            else:
                self.effort_limit = self._parse_joint_parameter(cfg.effort_limit, torch.inf)
                self.velocity_limit = self._parse_joint_parameter(cfg.velocity_limit, torch.inf)
                self.stiffness = self._parse_joint_parameter(cfg.stiffness, 0.0)
                self.damping = self._parse_joint_parameter(cfg.damping, 0.0)

        if not lab3:
            def _parse_joint_parameter(self, value, default):
                return _param(value, default, self._joint_names, self._num_envs, self._device)
        else:
            @property
            def effort_limit(self):
                warnings.warn("ActuatorBase.effort_limit is deprecated", DeprecationWarning)
                return self.actuator_effort_limit

            @property
            def velocity_limit(self):
                warnings.warn("ActuatorBase.velocity_limit is deprecated", DeprecationWarning)
                return self.actuator_velocity_limit

        @property
        def joint_names(self):
            return self._joint_names

        def reset(self, env_ids):
            pass

        def _clip_effort(self, effort, *args):
            lim = self.actuator_effort_limit if lab3 else self.effort_limit
            return torch.clip(effort, -lim, lim)

        def compute(self, control_action, joint_pos, joint_vel):
            self.computed_effort = (self.stiffness * (control_action.joint_positions - joint_pos)
                                    + self.damping * (control_action.joint_velocities - joint_vel)
                                    + control_action.joint_efforts)
            if lab3:
                self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)
            else:
                self.applied_effort = self._clip_effort(self.computed_effort)
            control_action.joint_efforts = self.applied_effort
            return control_action

    class ImplicitActuator(IdealPDActuator):
        is_implicit_model = True

    ImplicitActuatorCfg.class_type = ImplicitActuator
    mods = {n: types.ModuleType(n) for n in ("isaaclab", "isaaclab.actuators", "isaaclab.actuators.actuator_base",
                                              "isaaclab.utils", "isaaclab.utils.types")}
    act = mods["isaaclab.actuators"]
    act.IdealPDActuator, act.IdealPDActuatorCfg, act.ImplicitActuatorCfg = IdealPDActuator, IdealPDActuatorCfg, ImplicitActuatorCfg
    if lab3:
        mods["isaaclab.actuators.actuator_base"].resolve_joint_parameter = _param
    mods["isaaclab.utils"].DelayBuffer = _DelayBuffer
    mods["isaaclab.utils"].configclass = lambda c: dataclasses.dataclass(kw_only=True)(c)
    mods["isaaclab.utils.types"].ArticulationActions = _Actions
    return mods


@pytest.fixture(params=["2.2", "3.0"])
def lab(request, monkeypatch):
    for name, mod in _make_lab(request.param).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "dropbear_wbc.robots.hw_actuators", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        H = importlib.import_module("dropbear_wbc.robots.hw_actuators")
    yield request.param, H
    sys.modules.pop("dropbear_wbc.robots.hw_actuators", None)


def test_limit_fields_follow_the_running_lab(lab):
    version, H = lab
    expect = "actuator_effort_limit" if version == "3.0" else "effort_limit"
    assert H.LIMIT_FIELDS["effort"] == expect
    for cfg in H.make_hw_profile_groups("hw_v1").values():
        assert getattr(cfg, expect) is not None
    legacy = H.legacy_motor_groups()
    assert all(getattr(c, H.LIMIT_FIELDS["effort_sim"]) is not None for c in legacy.values())


def test_datasheet_motor_matches_numpy_law_without_deprecations(lab):
    version, H = lab
    from dropbear_wbc.robots.hw_motor_specs import hw_motor_torque

    g = torch.Generator().manual_seed(0)
    n_checked = n_clipped = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # no 3.0 alias reads/writes on the hot path
        for cfg in H.make_hw_motor_groups("user_2026_09_25", randomize=False).values():
            joints = list(cfg.joint_names_expr)
            act = cfg.class_type(cfg, joints, slice(None), 32, "cpu")
            act.reset(None)
            v0 = H.velocity_lim(act)
            q = torch.randn(32, len(joints), generator=g) * 0.3
            dq = (torch.rand(32, len(joints), generator=g) * 2.6 - 1.3) * v0
            tgt = q + torch.randn(32, len(joints), generator=g) * 0.6
            out = act.compute(_Actions(tgt, torch.zeros_like(q), torch.zeros_like(q)), q, dq)
            ref, clipped = hw_motor_torque(act.computed_effort.numpy(), dq.numpy(), H.effort_lim(act).numpy(),
                                           act._saturation.numpy(), v0.numpy(), act._coulomb.numpy(),
                                           act._viscous.numpy())
            np.testing.assert_allclose(act.applied_effort.numpy(), clipped, atol=1e-4)
            np.testing.assert_allclose(out.joint_efforts.numpy(), ref, atol=1e-4)
            n_checked += clipped.size
            n_clipped += int(np.sum(np.abs(clipped - act.computed_effort.numpy()) > 1e-6))
    assert n_clipped > 0.2 * n_checked  # the envelope is actually exercised


def test_clip_effort_accepts_explicit_joint_vel(lab):
    _, H = lab
    cfg = next(iter(H.make_hw_motor_groups("user_2026_09_25", randomize=False).values()))
    act = cfg.class_type(cfg, list(cfg.joint_names_expr), slice(None), 2, "cpu")
    big = torch.full((2, len(cfg.joint_names_expr)), 1e4)
    fast = H.velocity_lim(act).clone()  # at no-load speed the forward torque bound is 0
    assert torch.allclose(act._clip_effort(big, fast), torch.zeros_like(big), atol=1e-5)
    act._joint_vel[:] = 0.0  # the Lab 2.2 path (no joint_vel argument) uses the value cached by compute()
    assert torch.allclose(act._clip_effort(big), H.effort_lim(act))


def test_stop_model_and_knee_band_build_and_act_on_both_labs(lab):
    """The end-stop spring and the knee band resolve their per-joint parameters without the removed 3.0 API."""
    _, H = lab
    groups = H.make_hw_profile_groups("hw_v4", stop_model=True, knee_band=(150.0, 15.0))
    knee_groups = [c for c in groups.values() if any("knee_actuator" in j for j in c.joint_names_expr)]
    assert knee_groups
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for cfg in knee_groups:
            cfg.min_delay = cfg.max_delay = 0
            joints = list(cfg.joint_names_expr)
            act = cfg.class_type(cfg, joints, slice(None), 2, "cpu")
            act.reset(None)
            k = joints.index(next(j for j in joints if "knee_actuator" in j))
            q = act._stop_hi.clone()
            q[:, k] += 0.01  # 0.01 rad past the knee's upper stop
            zero = torch.zeros_like(q)
            act.compute(_Actions(q.clone(), zero.clone(), zero.clone()), q, zero)
            assert act.stop_effort[0, k] < 0.0 and abs(float(act.stop_effort[0, k]) + 0.01 * float(act._stop_k[0, k])) < 1e-3
            assert act.spring_effort[0, k] < 0.0  # past 15 deg of crank the band pulls back
