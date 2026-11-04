"""Post-run loop timing check over a ``policy_runner.py --clock wall --log steps.jsonl`` file (item 308).

``summary.json["loop_timing"]`` only carries counts and late-ms quantiles of the whole run. This report re-derives the
timing from the per-step ``slot`` / ``late_ms`` fields (item 306) and checks it against limits, so a hardware run can be
judged after the fact and per FSM phase:

* ``late_ms``           p50 / p99 / max lateness of each step against its grid slot
* ``overrun_steps``     steps more than ``late_threshold`` (default 1/4 period, as ``WallSchedule``) late
* ``skipped_slots``     grid slots with no step (sum of slot gaps - 1); each one holds the previous LowCmd a period longer
* ``max_gap_slots``     the longest slot gap (1 = every slot ran); ``max_hole_ms`` the longest wall gap between steps
* ``dup_slots``         steps that share or go back to an earlier slot: a catch-up burst, i.e. a scheduler bug (must be 0)
* ``late_streak``       longest run of consecutive overrun steps
* ``by_fsm``            the same counts split by FSM state (an overrun in POLICY matters more than one in PASSIVE)

Default limits are PROVISIONAL first-power limits for 50 Hz (item 308): no skipped slot, no overrun, p99 late <= 2 ms
(10 % of the period), max late <= 5 ms (the overrun threshold). Revisit them once the deploy computer's RT setup is
measured (item 292). A sim-clock log (``late_ms`` is None) is reported as ``skipped``.

usage: python tools/loop_timing_report.py steps.jsonl [--period-ms 20] [--max-p99-late-ms 2] [--max-late-ms 5]
           [--max-skipped-slots 0] [--max-overrun-steps 0] [--json out.json] [--png out.png]
Exit code 0 = pass or skipped, 1 = a limit failed, 2 = bad input. Pure stdlib (matplotlib only for --png).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LIMITS = {"max_p99_late_ms": 2.0, "max_late_ms": 5.0, "max_skipped_slots": 0, "max_overrun_steps": 0,
                  "max_dup_slots": 0}


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]  # same nearest-rank pick as WallSchedule.summary


def load_rows(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i + 1}: not JSON ({e})") from None
    return rows


def _stats(rows: list[dict], period_ms: float, thr_ms: float) -> dict:
    late = [float(r["late_ms"]) for r in rows]
    skipped = dup = 0
    max_gap = 1 if rows else 0
    max_hole = 0.0
    streak = best = 0
    for r in rows:
        streak = streak + 1 if r["late_ms"] > thr_ms else 0
        best = max(best, streak)
    for a, b in zip(rows, rows[1:]):
        d = int(b["slot"]) - int(a["slot"])
        if d <= 0:
            dup += 1
        else:
            skipped += d - 1
            max_gap = max(max_gap, d)
        max_hole = max(max_hole, 1e3 * (float(b["t"]) - float(a["t"])))
    return {"steps": len(rows), "late_ms": {"p50": _q(late, 0.5), "p99": _q(late, 0.99), "max": max(late, default=None)},
            "overrun_steps": sum(x > thr_ms for x in late), "skipped_slots": skipped, "max_gap_slots": max_gap,
            "max_hole_ms": max_hole, "dup_slots": dup, "late_streak": best}


def check(rows: list[dict], period_ms: float = 20.0, limits: dict | None = None,
          late_threshold_ms: float | None = None) -> dict:
    """Timing report + pass/fail for parsed steps.jsonl rows (only rows with a slot and late_ms are wall-clock rows)."""
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    thr = 0.25 * period_ms if late_threshold_ms is None else float(late_threshold_ms)
    wall = [r for r in rows if r.get("slot") is not None and r.get("late_ms") is not None]
    rep = {"period_ms": period_ms, "late_threshold_ms": thr, "limits": lim, "rows": len(rows), "wall_rows": len(wall)}
    if not wall:
        rep.update(status="skipped", reason="no wall-clock rows (sim clock or pre-item-306 log without slot/late_ms)",
                   failures=[])
        return rep
    if len(wall) != len(rows):
        rep["mixed_rows"] = len(rows) - len(wall)
    rep.update(_stats(wall, period_ms, thr))
    phases: dict[str, list[dict]] = {}
    for r in wall:
        phases.setdefault(str(r.get("fsm")), []).append(r)
    # per-phase slot gaps are counted inside each phase's own consecutive rows; a phase change is not a gap
    rep["by_fsm"] = {k: {kk: v for kk, v in _stats(v, period_ms, thr).items() if kk != "max_hole_ms"}
                     for k, v in phases.items()}
    checks = [("max_p99_late_ms", rep["late_ms"]["p99"]), ("max_late_ms", rep["late_ms"]["max"]),
              ("max_skipped_slots", rep["skipped_slots"]), ("max_overrun_steps", rep["overrun_steps"]),
              ("max_dup_slots", rep["dup_slots"])]
    rep["failures"] = [f"{name[4:]}={val:.3g} > {lim[name]}" for name, val in checks
                       if lim.get(name) is not None and val > lim[name]]
    rep["status"] = "fail" if rep["failures"] else "pass"
    return rep


def plot(rows: list[dict], rep: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wall = [r for r in rows if r.get("slot") is not None and r.get("late_ms") is not None]
    t0 = wall[0]["t"]
    t = [r["t"] - t0 for r in wall]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    a.plot(t, [r["late_ms"] for r in wall], ".", ms=3)
    a.axhline(rep["late_threshold_ms"], color="r", ls="--", lw=1, label=f"overrun {rep['late_threshold_ms']:.1f} ms")
    a.axhline(rep["limits"]["max_p99_late_ms"], color="orange", ls=":", lw=1, label="p99 limit")
    a.set_ylabel("late vs slot (ms)")
    a.set_yscale("symlog", linthresh=1.0)
    a.set_ylim(0.0, 2.0 * max([r["late_ms"] for r in wall] + [rep["late_threshold_ms"], rep["limits"]["max_p99_late_ms"]]))
    a.legend(loc="upper right", fontsize=8)
    gaps = [(b_["t"] - t0, b_["slot"] - a_["slot"]) for a_, b_ in zip(wall, wall[1:])]
    b.step([g[0] for g in gaps], [g[1] for g in gaps], where="post")
    b.set_ylabel("slot gap (1 = on grid)")
    b.set_xlabel("run time (s)")
    lt = rep["late_ms"]
    fig.suptitle(f"{rep['status'].upper()}: p99 late {lt['p99']:.2f} ms, max {lt['max']:.1f} ms, "
                 f"{rep['overrun_steps']} overrun, {rep['skipped_slots']} skipped, {rep['dup_slots']} dup slots"
                 + (f"\nfailed: {'; '.join(rep['failures'])}" if rep["failures"] else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--period-ms", type=float, default=20.0)
    ap.add_argument("--late-threshold-ms", type=float, default=None, help="default 1/4 period (WallSchedule)")
    for k, v in DEFAULT_LIMITS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--png", type=Path)
    args = ap.parse_args(argv)
    try:
        rows = load_rows(args.log)
    except (OSError, ValueError) as e:
        print(f"[timing] {e}", file=sys.stderr)
        return 2
    rep = check(rows, args.period_ms, {k: getattr(args, k) for k in DEFAULT_LIMITS}, args.late_threshold_ms)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    if args.png and rep["status"] != "skipped":
        plot(rows, rep, args.png)
    if rep["status"] == "skipped":
        print(f"[timing] SKIPPED: {rep['reason']}")
        return 0
    lt = rep["late_ms"]
    print(f"[timing] {rep['status'].upper()} steps={rep['steps']} late p50/p99/max={lt['p50']:.2f}/{lt['p99']:.2f}/"
          f"{lt['max']:.2f} ms overrun={rep['overrun_steps']} skipped={rep['skipped_slots']} "
          f"max_gap={rep['max_gap_slots']} dup={rep['dup_slots']} streak={rep['late_streak']}")
    for f in rep["failures"]:
        print(f"[timing] FAIL {f}")
    return 1 if rep["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
