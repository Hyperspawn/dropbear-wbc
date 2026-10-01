from benchmarks.g1.planner import DOC, MockExecutor, build_prompt, make_llm_planner, rule_planner, score


def test_reference_plans_execute_in_mock():
    ex = MockExecutor()
    for t in DOC["tasks"]:
        ok, log = ex.run(t["plan"])
        assert ok, (t["id"], log)


def test_mock_rejects_bad_plans():
    ex = MockExecutor()
    assert not ex.run(["fly_to(table)"])[0]
    assert not ex.run(["walk_to(moon)"])[0]


def test_llm_planner_wraps_completion_and_drops_chatter():
    plan = make_llm_planner(lambda p: "Sure!\nwalk_to(table)\nstand(3)\n")("Walk to the table and stop.")
    assert plan == ["walk_to(table)", "stand(3)"]
    assert "walk_to" in build_prompt("x")


def test_rule_baseline_runs():
    r = score(rule_planner)
    assert len(r) == 20 and sum(x["executes"] for x in r) >= 15
