"""Run-defining training settings of a tracking run, and how a resumed chunk inherits them. Pure Python (no Isaac).

``scripts/train.py --resume --load_run <run> --continue_run`` writes into an existing run. Before this module the
actuator profile and the hardware regularizers came only from that chunk's command line, so a chunk started without
``--actuator_profile`` silently fell back to the task default (``legacy`` for every tracking task) and dropped the
thermal / torque-rate / knee-stop / feet-slide terms and the target clamp, while the checkpoint kept its iteration
count. The same holds for the later run flags (explicit stop model, knee band, stop-load / impact penalties, tracking
strictness, head / foot height rewards): ``run_chunked_training.py`` forwards them only when they are passed again. ``tools/run_chunked_training.py --resume_run`` builds exactly such a command line when its own flags are left
out (``tools/run_chunked_locomotion.py`` already inherits the profile from run_info; the tracking runner did not).

:func:`resolve_run_settings` inherits every setting the chunk does not set and refuses a different value unless
``allow_change``; :func:`audit_run_dir` reports drift between the ``run_info*.json`` files of an existing run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RUN_SETTING_KEYS: tuple[str, ...] = (
    "actuator_profile",
    "hw_regularizers",
    "knee_stop_penalty",
    "feet_slide_penalty",
    "target_clamp_margin_deg",
    "action_rate_weight",
    "thermal_penalty",
    "stop_model",
    "stop_load_penalty",
    "impact_penalty",
    "knee_band",
    "body_pos_std",
    "joint_pos_weight",
    "end_threshold_m",
    "anchor_height_weight",
    "feet_height_weight",
    "robot_mass_kg",
)
"""Settings that define the plant / reward a run is trained on (all recorded in run_info.json by scripts/train.py)."""

CLI_DEFAULTS: dict[str, object] = {
    "actuator_profile": None,  # None = the task's own default profile
    "hw_regularizers": False,
    "knee_stop_penalty": 0.0,
    "feet_slide_penalty": 0.0,
    "target_clamp_margin_deg": None,
    "action_rate_weight": None,
    "thermal_penalty": None,  # None = the env's default motor_over_rated weight
    "stop_model": False,
    "stop_load_penalty": 0.0,
    "impact_penalty": 0.0,
    "knee_band": None,
    "body_pos_std": None,  # None = the task default tolerance
    "joint_pos_weight": 0.0,
    "end_threshold_m": None,
    "anchor_height_weight": 0.0,
    "feet_height_weight": 0.0,
    "robot_mass_kg": None,  # None = CAD + the profile's motor swap (mass_dr.plant_mass_offsets)
}
"""Value of a setting on a fresh run when the command line leaves it out."""

MISSING_IN_RUN_INFO: dict[str, object] = {**CLI_DEFAULTS, "actuator_profile": "legacy"}
"""What a run_info.json WITHOUT the key means: runs from before the key existed used the legacy groups and no
regularizers (play.py makes the same assumption for the profile)."""


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def recorded(info: dict, key: str):
    """The value ``info`` (a run_info dict) records for ``key`` (absent -> :data:`MISSING_IN_RUN_INFO`)."""
    return info[key] if key in info else MISSING_IN_RUN_INFO[key]


def run_info_files(run_dir: Path) -> list[Path]:
    """``run_info.json`` then ``run_info_resume_<stamp>.json`` in chunk order (stamps sort chronologically)."""
    run_dir = Path(run_dir)
    first = [run_dir / "run_info.json"] if (run_dir / "run_info.json").is_file() else []
    return first + sorted(run_dir.glob("run_info_resume_*.json"))


def latest_run_info(run_dir: Path) -> dict | None:
    """The run_info of the most recent chunk of ``run_dir`` (None if the run has none)."""
    files = run_info_files(run_dir)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def resolve_run_settings(cli: dict, prev_info: dict | None, *, continue_run: bool,
                         allow_change: bool = False) -> tuple[dict, list[str]]:
    """Settings for this chunk and human-readable notes.

    ``cli``: the parsed command line (``vars(args)``); a key that is None (or absent) was not given.
    ``prev_info``: the latest run_info of the run being continued (``continue_run``), else ignored.
    A fresh run / warm start (``--resume`` without ``--continue_run``) uses the command line + :data:`CLI_DEFAULTS`.
    A continued run inherits each setting the command line leaves out; a GIVEN value that differs from the recorded one
    raises ``ValueError`` unless ``allow_change`` (then it is applied and noted).
    """
    out: dict = {}
    notes: list[str] = []
    cont = bool(continue_run and prev_info is not None)
    for key in RUN_SETTING_KEYS:
        given = cli.get(key)
        if not cont:
            out[key] = CLI_DEFAULTS[key] if given is None else given
            continue
        rec = recorded(prev_info, key)
        if given is None:
            out[key] = rec
            notes.append(f"{key}: inherited {rec!r}")
        elif _same(given, rec):
            out[key] = given
        elif allow_change:
            out[key] = given
            notes.append(f"{key}: CHANGED {rec!r} -> {given!r} (--allow_settings_change)")
        else:
            raise ValueError(f"--continue_run: {key}={given!r} differs from the run's recorded {rec!r}; drop the flag to "
                             f"inherit it, or pass --allow_settings_change to change the plant/reward mid-run")
    if continue_run and prev_info is None:
        notes.append("continue_run without a previous run_info: nothing to inherit")
    return out, notes


def audit_run_dir(run_dir: Path) -> list[dict]:
    """Every chunk boundary of ``run_dir`` where a run-defining setting changed (empty = consistent run).
    Each record: ``{"file", "key", "before", "after", "flagged"}`` (``flagged`` = the chunk noted the change)."""
    files = run_info_files(run_dir)
    infos = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    drift: list[dict] = []
    for prev, (f, cur) in zip(infos, list(zip(files, infos))[1:]):
        flagged = any("CHANGED" in n for n in cur.get("run_settings_notes", []) or [])
        for key in RUN_SETTING_KEYS:
            a, b = recorded(prev, key), recorded(cur, key)
            if not _same(a, b):
                drift.append({"file": f.name, "key": key, "before": a, "after": b, "flagged": flagged})
    return drift


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Report run-defining setting drift across the chunks of tracking runs.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    bad = 0
    for d in a.run_dirs:
        drift = audit_run_dir(d)
        bad += sum(1 for r in drift if not r["flagged"])
        print(json.dumps({"run": str(d), "chunks": len(run_info_files(d)), "drift": drift}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
