"""Which reference clips can Dropbear's REAL motors follow? Per-clip, per-motor speed and range check (numpy only).

For every clip of a library manifest (or explicit NPZs), each of the 22 motors is checked against the real-actuator map
of ``robots/hw_motor_specs.py`` (default ``hw_v1`` = the user's map):

* speed: the reference ``joint_vel`` of the motor vs the motor's no-load speed. ``over_noload_frac`` = share of frames
  above it (impossible at any load); ``over_peak_speed_frac`` = share above the speed where the motor still delivers
  its PEAK torque (``no_load * (1 - peak / saturation)``), i.e. where tracking under load degrades;
* range: share of frames within ``--limit_margin_deg`` of the motor's hard limit (``MOTOR_HARD_LIMITS_DEG``), where
  a tracking policy has no authority left in one direction.

This is a REFERENCE-only screen (no physics): it says which clips are kinematically out of reach of the real motors,
not that the rest is feasible (torque needs a policy rollout on the ``hw_*`` profile, docs/ACTUATORS.md section 11).

    python tools/hw_motion_feasibility.py --library data/motions/libraries/accepted_v2.json --out logs/hw_twin/motion_feasibility_hw_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG, MOTOR_NAMES  # noqa: E402
from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params  # noqa: E402


def clip_report(npz: Path, params: dict, margin_rad: float) -> dict:
    z = np.load(npz, allow_pickle=True)
    names = [str(n) for n in z["joint_names"]]
    idx = [names.index(m) for m in MOTOR_NAMES]
    q, qd = z["joint_pos"][:, idx], z["joint_vel"][:, idx]
    out = {"frames": int(q.shape[0]), "duration_s": round(q.shape[0] / float(z["fps"]), 2), "motors": {}}
    worst = {"over_noload_frac": 0.0, "over_peak_speed_frac": 0.0, "near_limit_frac": 0.0}
    for k, m in enumerate(MOTOR_NAMES):
        p = params[m]
        w_nl = p["no_load_speed"]
        w_pk = w_nl * (1.0 - p["peak_torque"] / p["saturation_effort"])
        lo, hi = (math.radians(v) for v in MOTOR_HARD_LIMITS_DEG[m])
        s = np.abs(qd[:, k])
        near = (q[:, k] < lo + margin_rad) | (q[:, k] > hi - margin_rad)
        rec = {"model": p["model"], "max_speed": round(float(s.max()), 2), "no_load": round(w_nl, 2),
               "peak_torque_speed": round(w_pk, 2), "over_noload_frac": round(float((s > w_nl).mean()), 4),
               "over_peak_speed_frac": round(float((s > w_pk).mean()), 4), "near_limit_frac": round(float(near.mean()), 4)}
        out["motors"][m] = rec
        for key in worst:
            worst[key] = max(worst[key], rec[key])
    out["worst"] = worst
    out["speed_feasible"] = worst["over_noload_frac"] == 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", type=Path, nargs="*")
    ap.add_argument("--library", type=Path, default=None, help="manifest JSON (clips[].npz relative to it)")
    ap.add_argument("--profile", default="hw_v1", choices=sorted(HW_PROFILE_MAPS))
    ap.add_argument("--limit_margin_deg", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    clips: list[tuple[str, Path]] = [(p.stem, p) for p in args.npz]
    if args.library:
        man = json.loads(args.library.read_text(encoding="utf-8"))
        clips += [(c["name"], (args.library.parent / c["npz"]).resolve()) for c in man["clips"]]
    params = joint_hw_params(HW_PROFILE_MAPS[args.profile])
    reports = {name: clip_report(p, params, math.radians(args.limit_margin_deg)) for name, p in clips}
    # per motor: how many clips exceed its no-load speed / its peak-torque speed at least once
    per_motor = {m: {"model": params[m]["model"],
                     "clips_over_noload": sum(r["motors"][m]["over_noload_frac"] > 0 for r in reports.values()),
                     "clips_over_peak_speed": sum(r["motors"][m]["over_peak_speed_frac"] > 0 for r in reports.values()),
                     "clips_near_limit_gt5pct": sum(r["motors"][m]["near_limit_frac"] > 0.05 for r in reports.values())}
                 for m in MOTOR_NAMES}
    summary = {"tool": "tools/hw_motion_feasibility.py", "profile": args.profile, "motor_map": HW_PROFILE_MAPS[args.profile],
               "clips": len(reports), "speed_feasible_clips": sum(r["speed_feasible"] for r in reports.values()),
               "per_motor": per_motor, "per_clip": reports}
    text = json.dumps(summary, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{args.profile}: {summary['speed_feasible_clips']}/{len(reports)} clips never exceed a motor's no-load speed")
    for m, v in per_motor.items():
        if v["clips_over_noload"] or v["clips_over_peak_speed"] or v["clips_near_limit_gt5pct"]:
            print(f"  {m:24s} {v['model']:20s} over no-load in {v['clips_over_noload']:2d} clips, over peak-torque "
                  f"speed in {v['clips_over_peak_speed']:2d}, >5% frames near a limit in {v['clips_near_limit_gt5pct']:2d}")
    bad = sorted(((r["worst"]["over_noload_frac"], n) for n, r in reports.items() if not r["speed_feasible"]), reverse=True)
    print("  worst clips (share of frames over a no-load speed):", [(n, round(f, 3)) for f, n in bad[:10]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
