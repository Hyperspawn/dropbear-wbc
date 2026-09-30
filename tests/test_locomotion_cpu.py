"""CPU tests of the velocity-task pieces that do not need Isaac (``dropbear_wbc.tasks.locomotion``).

* the standing reset NPZ summary (anchor height = the torso target; root frame below the soles);
* the command-range curriculum (widening, clamping, success gating, the episode-boundary condition);
* the export's observation layout is consumable by the deploy runner (every term has a deploy function, the
  velocity command enters the observation, the sim-only term is refused without --allow-privileged).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

ROOT = sdk_test_paths.ROOT
NPZ = ROOT / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"

# the export's policy layout (tasks/locomotion/velocity_env_cfg.py PolicyCfg, export.OBS_FUNC)
VELOCITY_OBS = [
    {"name": "base_lin_vel", "func": "base_lin_vel", "dim": 3},
    {"name": "base_ang_vel", "func": "base_ang_vel", "dim": 3},
    {"name": "projected_gravity", "func": "projected_gravity", "dim": 3},
    {"name": "velocity_commands", "func": "velocity_commands", "dim": 3},
    {"name": "joint_pos", "func": "joint_pos_rel", "dim": 22},
    {"name": "joint_vel", "func": "joint_vel_rel", "dim": 22},
    {"name": "actions", "func": "last_action", "dim": 22},
]


@pytest.mark.skipif(not NPZ.is_file(), reason="standing NPZ not present")
def test_stand_summary_uses_anchor_not_root():
    from dropbear_wbc.tasks.locomotion.stand import summarize_stand_npz

    s = summarize_stand_npz(NPZ)
    assert 1.3 < s.anchor_z < 1.7  # chest height of the settled stand
    assert s.root_z < 0.0 < s.sole_z  # root frame origin below the ground, sole link frames above it
    assert s.max_joint_speed == 0.0 and s.closure_max_m < 1e-3


def _fake_env(lin_sum: float, ang_sum: float, step: int, ep_len: int = 1000, weight: float = 1.0):
    torch = pytest.importorskip("torch")
    ranges = SimpleNamespace(lin_vel_x=(0.0, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5))
    limits = SimpleNamespace(lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0))
    term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges, limit_ranges=limits))
    sums = {"track_lin_vel_xy_exp": torch.full((4,), lin_sum), "track_ang_vel_z_exp": torch.full((4,), ang_sum)}
    rm = SimpleNamespace(_episode_sums=sums, get_term_cfg=lambda n: SimpleNamespace(weight=weight))
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: term), reward_manager=rm,
                           common_step_counter=step, max_episode_length=ep_len, max_episode_length_s=20.0,
                           device="cpu"), ranges


def _curriculums():
    """Load mdp/curriculums.py by path (the mdp package __init__ imports isaaclab, absent on CPU)."""
    import importlib.util

    path = ROOT / "source" / "dropbear_wbc" / "tasks" / "locomotion" / "mdp" / "curriculums.py"
    spec = importlib.util.spec_from_file_location("_loco_curriculums", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_curriculum_widens_and_clamps():
    pytest.importorskip("torch")
    cur = _curriculums()
    _widen, velocity_cmd_levels = cur._widen, cur.velocity_cmd_levels

    assert _widen((0.0, 0.5), 0.1, (-0.3, 1.0)) == pytest.approx((-0.1, 0.6))
    assert _widen((-0.3, 1.0), 0.1, (-0.3, 1.0)) == pytest.approx((-0.3, 1.0))
    # success (per-second tracking reward 0.9 > 0.8 * weight) at an episode boundary: all three ranges widen
    env, r = _fake_env(lin_sum=0.9 * 20.0, ang_sum=0.9 * 20.0, step=2000)
    out = velocity_cmd_levels(env, [0, 1, 2, 3])
    assert r.lin_vel_x == pytest.approx((-0.1, 0.6)) and r.lin_vel_y == pytest.approx((-0.3, 0.3))
    assert r.ang_vel_z == pytest.approx((-0.6, 0.6)) and float(out) == pytest.approx(0.6)
    # not at an episode boundary: unchanged
    env, r = _fake_env(lin_sum=18.0, ang_sum=18.0, step=2001)
    velocity_cmd_levels(env, [0, 1])
    assert r.lin_vel_x == (0.0, 0.5)
    # failure (0.5 per second): unchanged; linear success alone does not widen yaw
    env, r = _fake_env(lin_sum=10.0, ang_sum=10.0, step=3000)
    velocity_cmd_levels(env, [0])
    assert r.lin_vel_x == (0.0, 0.5) and r.ang_vel_z == (-0.5, 0.5)
    env, r = _fake_env(lin_sum=18.0, ang_sum=10.0, step=3000)
    velocity_cmd_levels(env, [0])
    assert r.lin_vel_x == pytest.approx((-0.1, 0.6)) and r.ang_vel_z == (-0.5, 0.5)


def test_velocity_sidecar_layout_is_deployable(tmp_path):
    from dropbear_wbc.deploy.config import load_sidecar
    from dropbear_wbc.deploy.observations import (
        TERMS,
        ObsContext,
        ObservationBuilder,
        PrivilegedObservationError,
    )
    from dropbear_wbc.sdk import motors
    from deploy_fixtures import low_state

    assert all(t["func"] in TERMS for t in VELOCITY_OBS)
    side = {
        "schema": "dropbear-policy-sidecar-v1", "policy_onnx": "policy.onnx",
        "onnx_inputs": {"obs": "obs", "time_step": None}, "step_dt": 0.02,
        "joint_names": list(motors.MOTOR_NAMES), "default_joint_pos": list(motors.DEFAULT_POS),
        "joint_stiffness": list(motors.DEFAULT_KP), "joint_damping": list(motors.DEFAULT_KD),
        "action_scale": 0.25, "observations": [dict(t, scale=1.0, clip=None, history_length=1, params={})
                                               for t in VELOCITY_OBS],
        "obs_dim": sum(t["dim"] for t in VELOCITY_OBS), "motion": None,
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(side), encoding="utf-8")
    cfg = load_sidecar(p)
    assert cfg.motion is None and cfg.onnx_time_input is None and cfg.obs_dim == 78
    with pytest.raises(PrivilegedObservationError):
        ObservationBuilder(cfg.observations, allow_privileged=False)
    b = ObservationBuilder(cfg.observations, allow_privileged=True)
    ctx = ObsContext(state=low_state(1), cfg=cfg, last_action=np.zeros(22),
                     velocity_command=np.array([0.3, -0.1, 0.4]), allow_privileged=True)
    obs = b.compute(ctx)
    assert obs.shape == (78,)
    np.testing.assert_allclose(obs[9:12], [0.3, -0.1, 0.4], atol=1e-6)  # velocity_commands slice
    np.testing.assert_allclose(obs[6:9], [0.0, 0.0, -1.0], atol=1e-6)  # projected gravity, upright root
