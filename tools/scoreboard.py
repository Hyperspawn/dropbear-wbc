"""G0 scoreboard: one success table per skill from the play_*.json summaries the play scripts already write.

CPU only. Reads a registry (skills.json: name -> kind + summary glob), picks the newest eligible summary per skill and
applies the hardware gate. Every check is per motor in ``MOTOR_NAMES`` against the real-actuator spec of the motor map
the summary was simulated with (``robots.hw_motor_specs.HW_PROFILE_MAPS[actuator_profile]`` ->
``joint_hw_params``), never against the sim effort limit:

* ``actuator_profile`` is a hw_* profile (or listed in the registry's ``allow_profiles``); legacy / stiff_knee_hw
  summaries use sim limits (hips 200, calves 80 N*m) and cannot certify a skill
* perturbed play (``eval_design.perturbed``): nominal play is deterministic, N = 1 per scenario
* 0 falls, finite simulation
* a ``motor_torque`` block with every motor present; missing block or motor = UNKNOWN = FAIL (never skipped)
* per motor: rms <= rated (thermal proxy until a thermal model exists), max <= peak, clip_frac <= 1 %,
  max |joint speed| <= no-load speed, and the summary's sim peak limit equals the map's peak (catches a wrong map)

    python tools/scoreboard.py --registry tools/skills.json [--json out.json]

Exit code 1 if any registered skill is missing or fails, so it can gate a run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "source") not in sys.path:
    sys.path.insert(0, str(ROOT / "source"))

from dropbear_wbc.motion.names import MOTOR_NAMES  # noqa: E402
from dropbear_wbc.robots.hw_motor_specs import HW_PROFILE_MAPS, joint_hw_params  # noqa: E402

GATE = {"rms_over_rated": 1.0, "max_over_peak": 1.0, "clip_frac": 0.01, "speed_over_no_load": 1.0}
PEAK_MATCH_TOL = 0.02  # |sim peak limit - map peak| / map peak; summaries round to 0.01 N*m
_EPS = 1e-6


def motor_specs(profile: str) -> dict[str, dict]:
    """rated / peak / no-load per motor for a hw_* actuator profile (KeyError if the profile has no motor map)."""
    return {n: {"model": p["model"], "rated": float(p["rated_torque"]), "peak": float(p["peak_torque"]),
                "no_load": float(p["no_load_speed"])} for n, p in joint_hw_params(HW_PROFILE_MAPS[profile]).items()}


def torque_check(summary: dict, specs: dict[str, dict]) -> tuple[dict, list[str]]:
    """Worst per-motor ratios + failure reasons. Missing torque block / motor -> failure, never a skip."""
    per = (summary.get("motor_torque") or {}).get("per_motor")
    if not per:
        return {}, ["torque UNKNOWN (no motor_torque block)"]
    why: list[str] = []
    missing = [n for n in MOTOR_NAMES if n not in per]
    extra = sorted(set(per) - set(MOTOR_NAMES))
    if missing:
        why.append(f"torque UNKNOWN for {len(missing)} motors ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})")
    if extra:
        why.append(f"unknown motors in summary: {', '.join(extra[:3])}")
    worst = {k: (0.0, None) for k in GATE}
    mism = []
    for n in MOTOR_NAMES:
        m = per.get(n)
        if m is None:
            continue
        sp = specs[n]
        try:
            vals = {"rms_over_rated": float(m["rms_Nm"]) / sp["rated"], "max_over_peak": float(m["max_Nm"]) / sp["peak"],
                    "clip_frac": float(m["clip_frac"]), "speed_over_no_load": float(m["max_speed_rad_s"]) / sp["no_load"]}
        except (KeyError, TypeError, ValueError) as exc:
            why.append(f"{n}: torque block incomplete ({exc.__class__.__name__}: {exc})")
            continue
        for k, v in vals.items():
            if v > worst[k][0]:
                worst[k] = (v, n)
        lim = m.get("peak_limit_Nm")
        if lim is None or abs(float(lim) - sp["peak"]) > PEAK_MATCH_TOL * sp["peak"]:
            mism.append(f"{n} {lim} vs {sp['peak']:g}")
    if mism:
        why.append(f"sim peak limit != motor-map peak on {len(mism)} motors ({'; '.join(mism[:2])})")
    labels = {"rms_over_rated": "rms {v:.2f}x rated", "max_over_peak": "max {v:.2f}x peak",
              "clip_frac": "clip {v:.1%}", "speed_over_no_load": "speed {v:.2f}x no-load"}
    for k, lim in GATE.items():
        v, n = worst[k]
        if v > lim + _EPS:
            why.append(f"{labels[k].format(v=v)} ({n}, {specs[n]['model']})")
    return {k: {"value": round(v, 4), "motor": n} for k, (v, n) in worst.items()}, why


def _perturbed(s: dict) -> bool | None:
    ed = s.get("eval_design") or {}
    if "perturbed" in ed:
        return bool(ed["perturbed"])
    rnd = ed.get("randomization")  # play.py before the perturbed flag existed
    return None if rnd is None else str(rnd).startswith("training-level")


def evaluate(kind: str, s: dict, allow_profiles: tuple[str, ...] = ()) -> dict:
    """Return {"scenarios", "envs", "falls", "profile", "perturbed", "torque", "torque_x", "pass", "why"}."""
    why: list[str] = []
    if kind == "locomotion":
        scen = s.get("by_scenario") or {}
        scenarios, envs = len(scen), sum(int(g["envs"]) for g in scen.values())
        falls = int(s["falls_total"])
    elif kind == "tracking":
        clips = s.get("per_clip")
        envs = sum(len(c["envs"]) for c in clips.values()) if clips else int(s.get("num_envs", 0))
        scenarios = len(clips) if clips else 1
        falls = int(s["falls"]["envs_with_a_fall"])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    if not s.get("finite", False):
        why.append("non-finite" if "finite" in s else "finite flag missing")
    if falls:
        why.append(f"{falls} falls")
    pert = _perturbed(s)
    if not pert:
        why.append("nominal play (N = 1 per scenario); rerun with --perturbed" if pert is False else "perturbed flag missing")
    profile = s.get("actuator_profile")
    torque: dict = {}
    if profile is None:
        why.append("actuator_profile missing")
    elif profile not in HW_PROFILE_MAPS:
        why.append(f"actuator_profile {profile!r} has no motor map (sim limits, not datasheet)")
    elif allow_profiles and profile not in allow_profiles:
        why.append(f"actuator_profile {profile!r} not in allow_profiles")
    else:
        torque, twhy = torque_check(s, motor_specs(profile))
        why += twhy
    if not torque and not (s.get("motor_torque") or {}).get("per_motor"):
        if "torque UNKNOWN (no motor_torque block)" not in why:
            why.append("torque UNKNOWN (no motor_torque block)")
    tx = max((torque[k]["value"] for k in ("rms_over_rated", "max_over_peak")), default=None) if torque else None
    return {"scenarios": scenarios, "envs": envs, "falls": falls, "profile": profile, "perturbed": pert,
            "torque": torque, "torque_x": tx, "pass": not why, "why": why}


def _created(p: Path, s: dict) -> float:
    c = s.get("created")
    if c:
        try:
            return _dt.datetime.fromisoformat(c).timestamp()
        except ValueError:
            pass
    return os.path.getmtime(p)


def _eligible(s: dict, allow_profiles: tuple[str, ...]) -> bool:
    prof = s.get("actuator_profile")
    return bool(_perturbed(s)) and prof in HW_PROFILE_MAPS and (not allow_profiles or prof in allow_profiles)


def select(pattern: str, allow_profiles: tuple[str, ...] = ()) -> tuple[Path | None, dict | None, int]:
    """Newest summary by its `created` stamp (mtime fallback) among eligible ones (hw_* profile, perturbed); if none is
    eligible, the newest overall so the table shows why it fails. Returns (path, summary, n_candidates)."""
    cands = []
    for f in glob.glob(pattern, recursive=True):
        p = Path(f)
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cands.append((_created(p, s), p, s))
    if not cands:
        return None, None, 0
    ok = [c for c in cands if _eligible(c[2], allow_profiles)]
    _, p, s = max(ok or cands, key=lambda c: (c[0], str(c[1])))
    return p, s, len(cands)


def build(registry: dict, root: Path) -> list[dict]:
    rows = []
    allow = tuple(registry.get("allow_profiles", ()))
    for name, spec in registry["skills"].items():
        p, s, n = select(str(root / spec["summary_glob"]), allow)
        row = {"skill": name, "kind": spec["kind"], "file": str(p) if p else None, "candidates": n}
        if p is None:
            row.update(scenarios=0, envs=0, falls=None, profile=None, perturbed=None, torque={}, torque_x=None,
                       **{"pass": False, "why": ["no summary"]})
        else:
            row.update(evaluate(spec["kind"], s, allow))
        rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    out = [f"{'skill':<28}{'scen':>5}{'envs':>6}{'falls':>7}{'profile':>18}{'rms/rt':>8}{'max/pk':>8}  result"]
    for r in rows:
        t = r.get("torque") or {}
        f = lambda k: "-" if k not in t else f"{t[k]['value']:.2f}"  # noqa: E731
        fl = "-" if r["falls"] is None else str(r["falls"])
        out.append(f"{r['skill']:<28}{r['scenarios']:>5}{r['envs']:>6}{fl:>7}{str(r['profile'] or '-'):>18}"
                   f"{f('rms_over_rated'):>8}{f('max_over_peak'):>8}  "
                   + ("PASS" if r["pass"] else "FAIL: " + "; ".join(r["why"])))
    ok = sum(r["pass"] for r in rows)
    out.append(f"\n{ok}/{len(rows)} skills pass the hardware gate (hw_* profile, perturbed, 0 falls, per motor: "
               f"rms <= rated, max <= peak, clip <= {GATE['clip_frac']:.0%}, speed <= no-load)")
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
