"""Item 290: deploy_fixtures.unexplained_state_faults also explains a fault when the runner itself read the faulted
tick late (controller step > state_max_age_s / 2 past its step_dt slot), and only then; synthetic runner logs, no sockets."""
import json

import deploy_fixtures as fx

DT = 0.020  # runner step
PUB = {k: 0.002 * k for k in range(0, 2000)}  # robot published every 2 ms: no robot-side gap anywhere


def _log(tmp_path, rows):
    p = tmp_path / "steps.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _rows(n, late_at=None, late_s=0.0395, fault_at=(), reuse_after_late=False):
    rows, t, tick = [], 0.0, 0
    for j in range(n):
        step = late_s if j == late_at else DT
        t += step if j else 0.0
        if not (reuse_after_late and late_at is not None and j == late_at + 1):
            tick += max(1, round(step / 0.002))
        rows.append({"t": t, "tick": tick, "state_fault": "state_stale" if j in fault_at else None,
                     "state_age_ms": None})
    return rows


def test_late_runner_read_explains_fault(tmp_path):
    assert fx.unexplained_state_faults(_log(tmp_path, _rows(50, late_at=20, fault_at={20})), PUB) == []


def test_reused_step_of_late_read_tick_is_explained(tmp_path):
    rows = _rows(50, late_at=20, fault_at={20, 21}, reuse_after_late=True)
    assert rows[21]["tick"] == rows[20]["tick"]
    assert fx.unexplained_state_faults(_log(tmp_path, rows), PUB) == []


def test_on_time_runner_fault_stays_unexplained(tmp_path):
    bad = fx.unexplained_state_faults(_log(tmp_path, _rows(50, fault_at={20})), PUB)
    assert [b["tick"] for b in bad] == [_rows(50)[20]["tick"]] and abs(bad[0]["runner_read_late_ms"]) < 1.0


def test_earlier_late_read_does_not_cover_a_later_fresh_tick(tmp_path):
    bad = fx.unexplained_state_faults(_log(tmp_path, _rows(50, late_at=20, fault_at={22})), PUB)
    assert len(bad) == 1 and abs(bad[0]["runner_read_late_ms"]) < 1.0


def test_late_threshold_is_half_max_age(tmp_path):
    assert fx.unexplained_state_faults(_log(tmp_path, _rows(50, late_at=20, late_s=0.034, fault_at={20})), PUB)
    assert not fx.unexplained_state_faults(_log(tmp_path, _rows(50, late_at=20, late_s=0.036, fault_at={20})), PUB)
