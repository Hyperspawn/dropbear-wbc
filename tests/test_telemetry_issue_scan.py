"""``tools/telemetry_issue_scan.py`` on synthetic telemetry with planted defects (docs/ISSUES.md #19-#22)."""
from __future__ import annotations

import json
import sys

import numpy as np

import sdk_test_paths  # noqa: F401  (adds source/)

sys.path.insert(0, str(sdk_test_paths.ROOT / "tools"))
from telemetry_issue_scan import scan  # noqa: E402

from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES  # noqa: E402

T, DT = 500, 0.02
N = len(MOTOR_NAMES)
KNEE = MOTOR_NAMES.index("RL_knee_actuator_joint")


def _gait(t: np.ndarray, phase: float) -> np.ndarray:
    """Stance 60 % / swing 40 % of a 1 s cycle."""
    return ((t / 1.0 + phase) % 1.0) < 0.6


def _write(path, *, knee_on_stop=False, skid=False, scuff=False):
    t = np.arange(T) * DT
    contact = np.stack([_gait(t, 0.0), _gait(t, 0.5)], axis=1)
    lim = np.tile(np.radians([-60.0, 60.0]), (N, 1))
    lim[KNEE] = np.radians([0.0, 30.0])
    q = np.zeros((T, N))
    tau = np.zeros((T, N))
    peak = np.full(N, 25.0)
    peak[KNEE] = 60.0
    if knee_on_stop:  # right stance: knee pinned on its flexion stop while the motor pulls off it at peak
        st = contact[:, 1]
        q[st, KNEE] = lim[KNEE, 1]
        tau[st, KNEE] = -60.0
    sole = np.zeros((T, 2, 3))
    for k in range(2):  # the planted foot stays put; the swing foot moves 1.2 m/s forward
        x = 0.0
        for i in range(T):
            if not contact[i, k]:
                x += 1.2 * DT
            sole[i, k, 0] = x
    if skid:  # right foot: keeps sliding for 5 steps after every touchdown (0.15 m)
        c = contact[:, 1]
        for i in range(1, T):
            if c[i] and not c[i - 1]:
                sole[i:i + 5, 1, 0] += np.linspace(0.03, 0.15, 5)[: len(sole[i:i + 5])]
                sole[i + 5:, 1, 0] += 0.15
    if scuff:  # left foot: a one-step touch in the middle of every swing
        c = contact[:, 0].copy()
        sw = np.where(~c)[0]
        for i in sw:
            if (t[i] % 1.0) > 0.79 and (t[i] % 1.0) < 0.81:
                contact[i, 0] = True
    np.savez_compressed(
        path, dt=DT, motor_names=np.array(MOTOR_NAMES), can_ids=np.arange(N), peak_torque=peak,
        tau_sub=np.repeat(tau[:, None], 4, axis=1), meta=np.array(json.dumps({"actuator_profile": "hw_v1"})),
        q=q, qd=np.zeros((T, N)), q_target=q.copy(), tau=tau, tau_pd=tau.copy(), base_vel_b=np.zeros((T, 3)),
        yaw_rate=np.zeros(T), anchor_z=np.full(T, 1.4), cmd=np.zeros((T, 3)), contact=contact,
        feet_z=np.zeros((T, 4)), feet_lateral=np.full(T, 0.2), sole_pos_w=sole, root_pos_w=np.zeros((T, 3)),
        proj_grav=np.tile([0.0, 0.0, -1.0], (T, 1)), rated_torque=peak / 2.0, no_load_speed=np.full(N, 10.0),
        saturation_effort=peak * 2.0, model=np.array(["m"] * N), joint_limits=lim)
    return path


def _checks(r, severity="HIGH"):
    return {(f["check"], f["motor"]) for f in r["findings"] if f["severity"] == severity}


def test_clean_rollout_passes(tmp_path):
    r = scan(_write(tmp_path / "clean.npz"))
    assert r["hw_gate"]["pass"], r["hw_gate"]
    assert r["body"]["falls"] == 0


def test_knee_resting_on_its_stop_fails_the_gate(tmp_path):
    r = scan(_write(tmp_path / "stop.npz", knee_on_stop=True))
    assert ("stop_load", "RL_knee_actuator_joint") in _checks(r)
    assert not r["hw_gate"]["pass"] and "stop_load:RL_knee_actuator_joint" in r["hw_gate"]["fails"]


def test_touchdown_skid_is_measured_per_step(tmp_path):
    r = scan(_write(tmp_path / "skid.npz", skid=True))
    skid = [f for f in r["findings"] if f["check"] == "touchdown_skid" and f["motor"] == "right foot"]
    assert skid and abs(skid[0]["value"] - 0.15) < 0.04, skid
    assert not any(f["check"] == "touchdown_skid" and f["motor"] == "left foot" for f in r["findings"])


def test_mid_swing_touch_is_a_scuff(tmp_path):
    r = scan(_write(tmp_path / "scuff.npz", scuff=True))
    sc = [f for f in r["findings"] if f["check"] == "scuff" and f["motor"] == "left foot"]
    assert sc and sc[0]["value"] > 0.5, r["findings"]
