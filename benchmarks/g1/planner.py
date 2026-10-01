"""G1 planner skeleton: goal text -> skill-call plan -> executor, runnable without Isaac.

``Planner`` is any callable ``goal -> list[str]`` of ``skill(arg, ...)`` steps. Two are provided:
``rule_planner`` (keyword baseline) and ``make_llm_planner`` (wraps any ``complete(prompt) -> str``
function, so no SDK dependency lives here). ``MockExecutor`` checks each step against the scene and the
skill list and tracks the robot's location; swap it for the real FSM/SDK runner later.

    python -m benchmarks.g1.planner            # score the rule baseline on the 20 tasks
"""
import json
import re
from pathlib import Path

DOC = json.loads(Path(__file__).with_name("tasks.json").read_text())
CALL = re.compile(r"^([a-z_]+)\((.*)\)$")
ENTITIES = DOC["scene"]["entities"]
EXTRA_PLACES = {"slope_top", "doorway", "room_b_center"}  # named in tasks, not scene entities


def parse(step):
    m = CALL.match(step.strip())
    if not m:
        raise ValueError(f"bad step {step!r}")
    return m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]


def build_prompt(goal):
    return (
        "Plan for a humanoid robot. Reply with one skill call per line, nothing else.\n"
        f"Skills: {', '.join(DOC['skills_available'])}. Format: skill(arg, ...).\n"
        f"Scene: {DOC['scene']['description']} Entities: {', '.join(ENTITIES + sorted(EXTRA_PLACES))}.\n"
        f"Goal: {goal}\n"
    )


def make_llm_planner(complete):
    def plan(goal):
        text = complete(build_prompt(goal))
        return [ln.strip() for ln in text.splitlines() if CALL.match(ln.strip())]
    return plan


def rule_planner(goal):
    """Keyword baseline; deliberately dumb, so an LLM planner has something to beat."""
    g = goal.lower()
    steps = []
    if "get up" in g or "fallen" in g:
        steps.append("get_up()")
    if "walk to" in g or "go to" in g:
        m = re.search(r"(?:walk|go) to the ([a-z ]+?)(?: and|[.,]|$)", g)
        if m:
            steps.append(f"walk_to({m.group(1).strip().replace(' ', '_')})")
    return steps or ["stand(3)"]


class MockExecutor:
    """Validates and 'runs' a plan against the scene; returns (ok, log)."""

    def __init__(self):
        self.places = set(ENTITIES) | EXTRA_PLACES
        self.skills = set(DOC["skills_available"])

    def run(self, plan):
        log, where = [], "start_pose"
        for step in plan:
            try:
                name, args = parse(step)
            except ValueError as e:
                return False, log + [str(e)]
            if name not in self.skills:
                return False, log + [f"unknown skill {name}"]
            for a in args:
                if not a.isdigit() and a not in self.places and name != "play_motion":
                    return False, log + [f"unknown target {a} in {step}"]
            if name in ("walk_to", "step_over", "turn_to") and args:
                where = args[0]
            log.append(f"{step} @ {where}")
        return True, log


def score(planner, tasks=None):
    """Fraction of tasks whose plan executes in the mock and equals the reference plan."""
    ex, results = MockExecutor(), []
    for t in tasks or DOC["tasks"]:
        plan = planner(t["goal"])
        ok, _ = ex.run(plan)
        results.append({"id": t["id"], "executes": ok, "matches_reference": plan == t["plan"]})
    return results


if __name__ == "__main__":
    r = score(rule_planner)
    print(f"executes {sum(x['executes'] for x in r)}/{len(r)}, matches reference {sum(x['matches_reference'] for x in r)}/{len(r)}")
