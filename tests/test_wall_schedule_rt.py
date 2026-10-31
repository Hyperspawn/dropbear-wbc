"""Item 292: the wall-clock deploy loop must not burst missed steps after an overrun, must count late steps, and
the RT setup must report (not raise) what it could not apply."""
import os
import sys

import pytest

import sdk_test_paths  # noqa: F401
from dropbear_wbc.deploy.rt import WallSchedule, apply_rt


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def _run(stalls, n=50, period=0.02):
    """Loop of ``n`` steps; ``stalls[i]`` seconds of extra compute after step i. Returns step start times."""
    c = _Clock()
    s = WallSchedule(period, t0=0.0, clock=c, sleep=c.sleep)
    starts = []
    for i in range(n):
        s.wait()
        starts.append(c.t)
        c.t += 0.001 + stalls.get(i, 0.0)
    return s, starts


def test_no_stall_is_on_grid():
    s, starts = _run({})
    assert s.overrun_steps == 0 and s.skipped_slots == 0
    assert all(abs(t - 0.02 * i) < 1e-9 for i, t in enumerate(starts))


@pytest.mark.parametrize("stall_s,skipped", [(0.025, 0), (0.045, 1), (0.065, 2), (0.105, 4)])
def test_overrun_skips_instead_of_bursting(stall_s, skipped):
    s, starts = _run({10: stall_s})
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    # Base (deadline += period) ran the missed steps back-to-back: gaps of ~1 ms after a stall.
    assert min(gaps) > 0.01, f"burst after a {stall_s * 1e3:.0f} ms stall: min gap {min(gaps) * 1e3:.2f} ms"
    assert s.skipped_slots == skipped
    assert s.overrun_steps == 1
    assert s.summary()["late_ms"]["max"] == pytest.approx((0.001 + stall_s - 0.02) * 1e3, abs=0.01)
    # Every step after the late one starts exactly on the original 20 ms grid, one per slot.
    on_grid = [t for t in starts if abs((t / 0.02) - round(t / 0.02)) < 1e-6]
    assert len(on_grid) == len(starts) - (1 if stall_s + 0.001 > 0.02 else 0)
    slots = [round(t / 0.02) for t in on_grid]
    assert slots == sorted(set(slots))


def test_late_ms_reports_full_lateness():
    s, _ = _run({10: 0.105})
    assert s.summary()["late_ms"]["max"] == pytest.approx(86.0, abs=0.1)  # 1 + 105 ms compute from slot 10 -> 86 ms past slot 11


def test_apply_rt_reports_unsupported_or_denied():
    rep = apply_rt(cpu=10_000, prio=99)
    assert rep["requested"] == {"cpu": 10_000, "prio": 99, "lock_memory": False, "freeze_gc": False}
    if hasattr(os, "sched_setaffinity"):
        assert any("affinity" in e for e in rep["errors"])
    if not hasattr(os, "sched_setscheduler") or os.geteuid() != 0:
        assert any("SCHED_FIFO" in e for e in rep["errors"]) or rep["policy"] == "SCHED_FIFO"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux affinity API")
def test_apply_rt_pins_cpu():
    before = os.sched_getaffinity(0)
    try:
        cpu = min(before)
        rep = apply_rt(cpu=cpu)
        assert rep["cpu_affinity"] == [cpu] and not rep["errors"]
    finally:
        os.sched_setaffinity(0, before)


def _run_runner_with_stall(tmp_path, stall_steps, stall_s, extra=()):
    import importlib.util
    import json
    import time
    from pathlib import Path

    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import deploy_fixtures as fx
    from test_state_freshness_gate import _wall_robot

    from dropbear_wbc.deploy.fsm import DeployController
    from dropbear_wbc.sdk.transport import Endpoints

    root = Path(__file__).resolve().parents[1]
    fx.write_beyondmimic_onnx(tmp_path / "policy.onnx", obs_dim=sum(t["dim"] for t in fx.OBS_TERMS_FULL))
    fx.write_sidecar(tmp_path / "policy.json", "policy.onnx", terms=fx.OBS_TERMS_FULL)
    spec = importlib.util.spec_from_file_location("policy_runner", root / "tools" / "policy_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    n = {"i": 0}
    orig = DeployController.step

    def step(self, *a, **k):  # controller overrun injected at chosen steps
        n["i"] += 1
        if n["i"] in stall_steps:
            time.sleep(stall_s)
        return orig(self, *a, **k)

    ep = Endpoints(cmd_port=27961, state_port=27962)
    robot = _wall_robot(ep)
    robot.start()
    try:
        DeployController.step = step
        rc = runner.main(["--mode", "policy", "--sidecar", str(tmp_path / "policy.json"), "--allow-privileged",
                          "--clock", "wall", "--duration", "2.0", "--passive-s", "0.1", "--move-s", "0.4",
                          "--hold-s", "0.2", "--cmd-port", str(ep.cmd_port), "--state-port", str(ep.state_port),
                          "--connect-timeout-s", "20", "--summary", str(tmp_path / "s.json"),
                          "--log", str(tmp_path / "steps.jsonl"), *extra])
    finally:
        DeployController.step = orig
        robot.running = False
        robot.join(timeout=5)
    steps = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    return rc, json.loads((tmp_path / "s.json").read_text(encoding="utf-8")), steps


def test_runner_overrun_no_catch_up_burst(tmp_path):
    rc, s, steps = _run_runner_with_stall(tmp_path, (50, 75), 0.105)
    dt_ms = [1e3 * (b["t"] - a["t"]) for a, b in zip(steps, steps[1:])]
    # Base: 4 back-to-back steps (0.2-1.6 ms apart) after each 105 ms stall, 5-7 of them re-using one LowState tick.
    assert min(dt_ms) > 10.0, sorted(dt_ms)[:6]
    lt = s["loop_timing"]
    assert lt["overrun_steps"] >= 2 and lt["skipped_slots"] >= 8, lt
    assert lt["late_ms"]["max"] > 80.0, lt
    assert s["rt"]["requested"]["cpu"] is None and rc == 0 and s["status"] == "executed"


def test_runner_max_overrun_steps_fails_run(tmp_path):
    rc, s, _ = _run_runner_with_stall(tmp_path, (50,), 0.045, extra=("--max-overrun-steps", "0"))
    assert rc == 1 and s["status"] == "failed" and "not real-time" in s["error"], s.get("error")
