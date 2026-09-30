"""Seqlock state channel between the physics process and the live viewer (dropbear_wbc.isaac.state_share)."""
from __future__ import annotations

import numpy as np

import sdk_test_paths  # noqa: F401  (adds source/)

from dropbear_wbc.isaac.state_share import StateReader, StateWriter


def test_reader_sees_only_new_complete_states(tmp_path):
    names = ["a", "b", "c"]
    w = StateWriter(tmp_path / "state.bin", names)
    r = StateReader(tmp_path / "state.bin", wait_s=1.0)
    assert r.joint_names == names
    assert r.read() is None  # nothing written yet
    w.write(0.02, [1.0, 2.0, 0.9], [1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    st = r.read()
    assert st is not None and st["sim_t"] == 0.02
    np.testing.assert_allclose(st["root_pos"], [1.0, 2.0, 0.9])
    np.testing.assert_allclose(st["joint_pos"], [0.1, 0.2, 0.3])
    assert r.read() is None  # same state is not reported twice
    w.write(0.04, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
    w.write(0.06, [0, 0, 0], [1, 0, 0, 0], [0.5, 0.5, 0.5])
    st = r.read()
    assert st["sim_t"] == 0.06  # a slow reader skips to the newest state
    w.buf[0] = w.seq + 1  # simulate a write in progress (odd seq)
    assert r.read() is None


def test_body_poses_round_trip(tmp_path):
    w = StateWriter(tmp_path / "s.bin", ["j0", "j1"], ["torso", "foot"])
    r = StateReader(tmp_path / "s.bin", wait_s=1.0)
    assert r.body_names == ["torso", "foot"]
    bp = np.array([[0.0, 0.0, 1.0], [0.1, -0.1, 0.05]])
    bq = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    w.write(0.5, bp[0], bq[0], [0.2, -0.2], bp, bq)
    st = r.read()
    np.testing.assert_allclose(st["body_pos"], bp)
    np.testing.assert_allclose(st["body_quat_wxyz"], bq)
    np.testing.assert_allclose(st["joint_pos"], [0.2, -0.2])
