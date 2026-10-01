"""G0 scoreboard: one success table per skill from the play_*.json summaries the play scripts already write.

CPU only. Reads a registry (skills.json: name -> kind + summary glob), takes the latest summary per skill and
applies the soft gate: 0 falls, motor torque <= 1.5x rated (locomotion summaries), finite sim.

    python tools/scoreboard.py --registry tools/skills.json [--json out.json]

Exit code 1 if any registered skill is missing or fails, so it can gate a run.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# peak torque limit [N*m] -> rated torque [N*m] (docs/ACTUATORS.md section 0)
RATED_BY_PEAK = {100.0: 50.0, 40.0: 15.0, 25.0: 10.0}
TORQUE_FACTOR = 1.5


def _rated(peak: float) -> float | None:
    for p, r in RATED_BY_PEAK.items():
        if abs(peak - p) < 1.0:
            return r
    return None


def torque_ratio(summary: dict) -> float | None:
    """Worst max_Nm / rated over motors; None when the summary has no torque block or no known rated value."""
    worst = None
    for m in (summary.get("motor_torque") or {}).get("per_motor", {}).values():
        r = _rated(float(m["peak_limit_Nm"]))
        if r:
            x = float(m["max_Nm"]) / r
            worst = x if worst is None else max(worst, x)
    return worst


def evaluate(kind: str, s: dict) -> dict:
    """Return {"units", "falls", "torque_x", "pass", "why"} for one summary."""
    why = []
    if kind == "locomotion":
        units, falls = sum(g["envs"] for g in s["by_scenario"].values()), int(s["falls_total"])
    elif kind == "tracking":
        units = sum(len(c["envs"]) for c in s["per_clip"].values()) if "per_clip" in s else int(s.get("num_envs", 0))
        falls = int(s["falls"]["envs_with_a_fall"])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    tx = torque_ratio(s)
    if not s.get("finite", True):
        why.append("non-finite")
    if falls:
        why.append(f"{falls} falls")
    if tx is not None and tx > TORQUE_FACTOR:
        why.append(f"torque {tx:.2f}x rated")
    return {"units": units, "falls": falls, "torque_x": tx, "pass": not why, "why": why}


def latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern, recursive=True))
    return Path(files[-1]) if files else None


def build(registry: dict, root: Path) -> list[dict]:
    rows = []
    for name, spec in registry["skills"].items():
        p = latest(str(root / spec["summary_glob"]))
        row = {"skill": name, "kind": spec["kind"], "file": str(p) if p else None}
        if p is None:
            row.update(units=0, falls=None, torque_x=None, **{"pass": False, "why": ["no summary"]})
        else:
            row.update(evaluate(spec["kind"], json.loads(p.read_text(encoding="utf-8"))))
        rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    out = [f"{'skill':<28}{'envs':>6}{'falls':>7}{'torque':>9}  result"]
    for r in rows:
        tq = "-" if r["torque_x"] is None else f"{r['torque_x']:.2f}x"
        fl = "-" if r["falls"] is None else str(r["falls"])
        out.append(f"{r['skill']:<28}{r['units']:>6}{fl:>7}{tq:>9}  " + ("PASS" if r["pass"] else "FAIL: " + "; ".join(r["why"])))
    ok = sum(r["pass"] for r in rows)
    out.append(f"\n{ok}/{len(rows)} skills pass the soft gate (0 falls, torque <= {TORQUE_FACTOR}x rated)")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", type=Path, default=Path("tools/skills.json"))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = build(json.loads(a.registry.read_text(encoding="utf-8")), a.root)
    print(render(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
