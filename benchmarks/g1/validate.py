"""Validate benchmarks/g1/tasks.json: schema, known skills, unique ids, plan steps use declared skills."""
import json
import re
import sys
from pathlib import Path

PATH = Path(__file__).with_name("tasks.json")
CALL = re.compile(r"^([a-z_]+)\((.*)\)$")


def validate(doc):
    errors = []
    skills = set(doc["skills_available"])
    entities = set(doc["scene"]["entities"])
    seen = set()
    for t in doc["tasks"]:
        tid = t.get("id", "?")
        if tid in seen:
            errors.append(f"{tid}: duplicate id")
        seen.add(tid)
        for key in ("goal", "plan", "success"):
            if not t.get(key):
                errors.append(f"{tid}: missing {key}")
        for step in t.get("plan", []):
            m = CALL.match(step)
            if not m:
                errors.append(f"{tid}: bad step {step!r}")
            elif m.group(1) not in skills:
                errors.append(f"{tid}: unknown skill {m.group(1)!r}")
    n = doc["pass_criteria"]["tasks"]
    if len(doc["tasks"]) != n:
        errors.append(f"expected {n} tasks, found {len(doc['tasks'])}")
    if not entities:
        errors.append("scene has no entities")
    return errors


if __name__ == "__main__":
    errs = validate(json.loads(PATH.read_text()))
    print("\n".join(errs) or "ok")
    sys.exit(1 if errs else 0)
