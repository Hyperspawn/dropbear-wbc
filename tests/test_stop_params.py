"""Item 353: per-joint explicit end-stop springs stay discretely stable at the 5 ms step (notes/223 F2).

The stop spring (k, d) and the role PD (kp, kd) act explicitly over one physics step on the joint's inertia. The
discrete spectral radius of that loop must be < 1 for every stopped motor joint of every hw map at its smallest
plausible inertia (armature + ``STOP_LINK_INERTIA_MIN``); the calves (light foot behind the ankle linkage) at the
armature alone. The pre-353 generic calf stop (omega 250 / zeta 0.7) fails that check."""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dropbear_wbc.robots import hw_motor_specs as S

DT = 0.005
MAPS = sorted(set(S.HW_PROFILE_MAPS.values()))
CALVES = [f"{p}_Revolute{n}" for p in ("LL", "RL") for n in (67, 81)]


def _numpy_rho(k, d, kp, kd, inertia, dt=DT):
    K, C = (k + kp) / inertia, (d + kd) / inertia
    a = np.array([[1 - K * dt * dt, dt * (1 - C * dt)], [-K * dt, 1 - C * dt]])
    return float(max(abs(np.linalg.eigvals(a))))


def _generic(arm, w=None, z=None):
    w = S.STOP_MODEL["omega"] if w is None else w
    z = S.STOP_MODEL["zeta"] if z is None else z
    k = w * w * arm
    return k, 2 * z * math.sqrt(k * arm)


def test_spectral_radius_matches_eigenvalues():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k, d, kp, kd = rng.uniform(0, 9000), rng.uniform(0, 30), rng.uniform(0, 500), rng.uniform(0, 6)
        inertia = 10 ** rng.uniform(-2.5, 0.5)
        assert S.stop_spectral_radius(k, d, kp, kd, inertia, DT) == pytest.approx(_numpy_rho(k, d, kp, kd, inertia), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("map_name", MAPS)
def test_every_stopped_joint_is_stable_at_its_smallest_inertia(map_name):
    params = S.joint_hw_params(map_name)
    table = S.stop_param_table(map_name)
    n = 0
    for j, sp in table.items():
        if sp["source"] == "none":
            assert sp["k"] == 0.0 and sp["d"] == 0.0
            continue
        p = params[j]
        inertia = float(p["armature"]) + S.STOP_LINK_INERTIA_MIN[j]
        rho = S.stop_spectral_radius(sp["k"], sp["d"], float(p["kp"]), float(p["kd_sim"]), inertia, DT)
        assert rho < 0.99, (map_name, j, sp, inertia, rho)
        n += 1
    assert n == 16  # hips 6, knees 2, calves 4, shoulder rolls (LH_pitch) 2, elbows 2
    # 360 deg motors (arm rolls, wrist rolls, shoulder pitch) have no stop
    assert {j for j, v in table.items() if v["source"] == "none"} >= {"LH_roll", "RH_roll", "LH_wrist_roll", "LH_yaw"}


def test_calf_stop_is_stable_on_armature_alone_with_margin():
    for map_name in MAPS:
        params = S.joint_hw_params(map_name)
        for j in CALVES:
            sp = S.stop_param_table(map_name)[j]
            arm, kp, kd = float(params[j]["armature"]), float(params[j]["kp"]), float(params[j]["kd_sim"])
            for ratio in (1.0, 1.06, 1.2, 1.5, 2.0, 3.0, 5.0):
                assert S.stop_spectral_radius(sp["k"], sp["d"], kp, kd, ratio * arm, DT) < 0.95, (map_name, j, ratio)


def test_pre_353_generic_calf_stop_is_unstable():
    """The bug this item fixes: generic omega 250 / zeta 0.7 + PD 80/4 on the X8 Pro armature (ratio 1.0-1.5)."""
    params = S.joint_hw_params("user_2026_09_25")
    arm = float(params["LL_Revolute67"]["armature"])
    k, d = _generic(arm)
    for ratio in (1.0, 1.06, 1.19, 1.5):
        assert S.stop_spectral_radius(k, d, 80.0, 4.0, ratio * arm, DT) > 1.0
    assert S.stop_spectral_radius(k, d, 80.0, 4.0, 1.7 * arm, DT) < 1.0


def test_calf_role_params_and_other_joints_unchanged():
    assert S.STOP_ROLE_PARAMS == {"calf": {"omega": 210.0, "zeta": 0.3}}
    for map_name in MAPS:
        params = S.joint_hw_params(map_name)
        for j, sp in S.stop_param_table(map_name).items():
            arm = float(params[j]["armature"])
            if sp["source"] == "none":
                continue
            if "knee" in j:
                assert (sp["k"], sp["d"], sp["source"]) == (8000.0, 20.0, "knee")
            elif j in CALVES:
                k, d = _generic(arm, 210.0, 0.3)
                assert sp["source"] == "calf" and sp["k"] == pytest.approx(k) and sp["d"] == pytest.approx(d)
            else:
                k, d = _generic(arm)
                assert sp["source"] == "generic" and sp["k"] == pytest.approx(k) and sp["d"] == pytest.approx(d)
    x8 = S.stop_param_table("user_2026_09_25")["LL_Revolute67"]
    assert x8["k"] == pytest.approx(1214.5, abs=0.1) and x8["d"] == pytest.approx(3.470, abs=1e-3)


def test_calf_penetration_at_20_nm_stays_small():
    sp = S.stop_param_table("user_2026_09_25")["LL_Revolute81"]
    assert math.degrees(20.0 / sp["k"]) < 1.0


def test_run_info_record():
    assert S.stop_params_run_info("hw_v1", False) is None
    assert S.stop_params_run_info("legacy", True) is None
    info = S.stop_params_run_info("hw_cem6", True)
    assert info["map"] == "cem6_2026_10_16" and info["role_params"]["calf"] == {"omega": 210.0, "zeta": 0.3}
    assert info["joints"]["RL_Revolute81"]["source"] == "calf"
    assert info["joints"]["RL_Revolute81"]["k"] == pytest.approx(1214.5, abs=0.1)
    assert info["joints"]["LH_wrist_roll"] == {"k": 0.0, "d": 0.0, "source": "none"}
    for path in ("scripts/train.py", "scripts/train_locomotion.py"):
        text = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
        assert '"stop_params": stop_params_run_info(' in text, path


def test_actuator_groups_use_the_table(monkeypatch):
    """make_hw_motor_groups(stop_model=True) builds exactly stop_param_table (fake isaaclab, as in the lab-compat test)."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        compat = importlib.import_module("test_datasheet_motor_lab_compat")
    finally:
        sys.path.pop(0)
    for name, mod in compat._make_lab("2.2").items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "dropbear_wbc.robots.hw_actuators", raising=False)
    try:
        H = importlib.import_module("dropbear_wbc.robots.hw_actuators")
        for profile in ("hw_v1", "hw_cem6", "hw_v4_arms"):
            table = S.stop_param_table(S.HW_PROFILE_MAPS[profile])
            seen = set()
            for cfg in H.make_hw_profile_groups(profile, stop_model=True).values():
                if cfg.stop_stiffness is None:
                    continue
                for j, k in cfg.stop_stiffness.items():
                    assert k == pytest.approx(table[j]["k"]) and cfg.stop_damping[j] == pytest.approx(table[j]["d"]), (profile, j)
                    seen.add(j)
            assert set(CALVES) <= seen
    finally:
        sys.modules.pop("dropbear_wbc.robots.hw_actuators", None)
