"""Item 322: recorded torque carries motor.tau_valid so clipped 0x500 values are identifiable. CPU only.

The policy never reads ``tau_est``; its only consumers are the teleop dataset column ``observation.motor_tau``
and the runner ``--log`` JSONL ``tau_est``. On hardware both knees sit at the +-24 N*m 0x500 rail for the
whole stance (notes/31), so without the validity bit those logs hold 24 where the joint carries 79-166 N*m.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import deploy_fixtures as fx
import sdk_test_paths  # noqa: F401
from dropbear_wbc.sdk import motors
from dropbear_wbc.sdk.transport import Endpoints
from dropbear_wbc.sdk.types import MOTOR_AGE_UNKNOWN
from dropbear_wbc.teleop.recorder import EXTRA_COLUMNS, SessionRecorder

import test_runner_loopback as rl

KNEES = [i for i, n in enumerate(motors.MOTOR_NAMES) if "knee" in n]


def test_knee_slots_exist():
    assert len(KNEES) == 2, motors.MOTOR_NAMES


def test_recorder_writes_motor_tau_valid(tmp_path):
    assert EXTRA_COLUMNS["observation.motor_tau_valid"][0] == 22
    rec = SessionRecorder(tmp_path / "s", fps=50.0, task="t")
    valid = np.ones(22)
    valid[KNEES] = 0
    for k in range(5):
        row = {"observation.state": np.zeros(10), "action": np.zeros(10), "timestamp": k / 50.0}
        for name, (w, _) in EXTRA_COLUMNS.items():
            row[name] = np.zeros(w)
        row["observation.motor_tau"] = np.where(valid > 0, 1.0, 24.0)
        row["observation.motor_tau_valid"] = valid
        rec.add(**row)
    rec.close({})
    z = np.load(tmp_path / "s" / "extras" / "episode_000000.npz")
    v = z["observation__motor_tau_valid"]
    assert v.shape == (5, 22) and (v[:, KNEES] == 0).all() and v.sum() == 5 * 20
    info = json.loads((tmp_path / "s" / "meta" / "info.json").read_text())
    assert info["features"]["observation.motor_tau_valid"]["shape"] == [22]


def test_recorder_requires_the_column(tmp_path):
    rec = SessionRecorder(tmp_path / "s", fps=50.0, task="t")
    row = {"observation.state": np.zeros(10), "action": np.zeros(10), "timestamp": 0.0}
    for name, (w, _) in EXTRA_COLUMNS.items():
        if name != "observation.motor_tau_valid":
            row[name] = np.zeros(w)
    with pytest.raises(KeyError, match="motor_tau_valid"):
        rec.add(**row)


def test_runner_log_records_tau_valid_and_age(tmp_path):
    ep = Endpoints(cmd_port=27597, state_port=27598)
    valid = np.ones(22, np.uint8)
    valid[KNEES] = 0
    age = np.full(22, 3000, np.uint32)
    age[0] = MOTOR_AGE_UNKNOWN
    robot = rl.FakeRobot(ep, tau_valid=valid, age_us=age)
    robot.start()
    try:
        runner = rl._load_runner()
        rc = runner.main(["--mode", "hold", "--duration", "0.6", "--passive-s", "0.1", "--move-s", "0.2",
                          "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(tmp_path / "summary.json"),
                          "--log", str(tmp_path / "steps.jsonl")])
    finally:
        robot.running = False
        robot.join(timeout=5)
    assert rc == 0, json.loads((tmp_path / "summary.json").read_text()).get("error")
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(steps) >= 10
    for s in steps:
        assert len(s["tau_valid"]) == 22 and len(s["motor_age_us"]) == 22
        assert [s["tau_valid"][i] for i in KNEES] == [0, 0] and sum(s["tau_valid"]) == 20
        assert s["motor_age_us"][0] is None and s["motor_age_us"][1] == 3000
