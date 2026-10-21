"""robustness_bench --repeats: case plan and summary (dropbear_wbc.tasks.locomotion.push_bench). CPU only."""
from __future__ import annotations

import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.tasks.locomotion.push_bench import case_key, push_cases, push_summary

PH = [("stand", (0.0, 0.0, 0.0))]


def test_one_repeat_is_the_old_plan():
    cases = push_cases(PH, 8, [0.2, 0.4], [0.0, 0.25])
    assert len(cases) == 8 * 2 * 2
    assert all(c["offset_s"] == c["base_offset_s"] and c["repeat"] == 0 for c in cases)


def test_repeats_shift_the_push_time():
    cases = push_cases(PH, 2, [0.4], [0.25], repeats=3, jitter_s=0.02)
    assert len(cases) == 2 * 3
    assert sorted({c["offset_s"] for c in cases}) == [0.25, 0.27, 0.29]
    assert {case_key(c) for c in cases} == {"stand:0:0.4:0.25", "stand:180:0.4:0.25"}


def test_bad_repeat_settings():
    with pytest.raises(ValueError):
        push_cases(PH, 2, [0.4], [0.0], repeats=0)
    with pytest.raises(ValueError):
        push_cases(PH, 2, [0.4], [0.0], repeats=2, jitter_s=0.0)


def _results(outcome):
    cases = push_cases(PH, 1, [0.2, 0.4, 0.6], [0.0], repeats=4, jitter_s=0.02)
    return [dict(c, survived=outcome(c)) for c in cases]


def test_summary_strict_vs_majority_and_mixed_cells():
    # dv 0.2: 4/4, dv 0.4: 3/4 (mixed), dv 0.6: 1/4 (mixed)
    surv = {0.2: 4, 0.4: 3, 0.6: 1}
    s = push_summary(_results(lambda c: c["repeat"] < surv[c["dv"]]), ["stand"], 1)["stand"]
    assert s["max_dv_survived_by_dir"] == {"0": 0.2}
    assert s["max_dv_majority_by_dir"] == {"0": 0.4}
    assert s["cells"] == 3 and s["mixed_cells"] == 2 and s["mixed_cell_rate"] == pytest.approx(2 / 3)
    assert s["survival_rate"] == pytest.approx(8 / 12)


def test_summary_matches_the_old_rule_with_one_repeat():
    cases = push_cases(PH, 2, [0.2, 0.4, 0.6], [0.0])
    res = [dict(c, survived=not (c["dir_deg"] == 180.0 and c["dv"] >= 0.4)) for c in cases]
    s = push_summary(res, ["stand"], 2)["stand"]
    assert s["max_dv_survived_by_dir"] == {"0": 0.6, "180": 0.2} and s["worst_dir_dv"] == 0.2
    assert s["mixed_cells"] == 0
