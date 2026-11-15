"""Locomotion ``--continue_run`` continuity of the command-line training settings (pure Python, no Isaac).

``scripts/train_locomotion.py`` builds the reward / terrain / action set of a chunk from its own command line only.
A chunk started with ``--resume --load_run X --continue_run`` but without the run's flags (for example
``tools/run_chunked_locomotion.py --resume_run X`` with the training flags left out) silently trained the checkpoint
on another task: no thermal / gait-v3 / AMP / arm-swing / symmetry / clip-excess terms, flat instead of rough terrain.
Event / DR terms are checked separately (:mod:`dropbear_wbc.tasks.dr_record`); the robot mass is inherited.

This module records the settings in the same form as ``run_info.json`` and refuses a continued chunk whose effective
settings or reward weights differ from the run's latest recorded chunk unless ``--allow_settings_change``.
"""
from __future__ import annotations

import json

UNRECORDED = "UNRECORDED"

# run_info key -> command-line flag (for the error message)
SETTING_FLAGS: dict[str, str] = {
    "actuator_profile": "--actuator_profile",
    "thermal_penalty": "--thermal_penalty",
    "knee_stop_penalty": "--knee_stop_penalty",
    "stop_model": "--stop_model",
    "stop_load_penalty": "--stop_load_penalty",
    "impact_penalty": "--impact_penalty",
    "knee_band": "--knee_band",
    "gait_shaping": "--gait_shaping",
    "gait_v2": "--gait_v2",
    "gait_v3": "--gait_v3",
    "target_clamp_margin_deg": "--target_clamp_margin_deg",
    "terrain": "--terrain",
    "stair_max_m": "--stair_max_m",
    "slope_max": "--slope_max",
    "feet_width_weight": "--feet_width_weight",
    "action_rate_weight": "--action_rate_weight",
    "arm_action_scale": "--arm_action_scale",
    "arm_deviation_weight": "--arm_deviation_weight",
    "arm_swing_weight": "--arm_swing_weight",
    "arm_swing_k": "--arm_swing_k",
    "arm_abduction_weight": "--arm_abduction_weight",
    "arm_gate_tilt_rate": "--arm_gate_tilt_rate",
    "arm_gate_hold_s": "--arm_gate_hold_s",
    "amp_weight": "--amp_weight",
    "amp_grad_penalty": "--amp_grad_penalty",
    "symmetry": "--symmetry",
    "symmetry_mirror_loss": "--symmetry_mirror_loss",
    "clip_excess_arms": "--clip_excess_arms",
    "clip_excess_legs": "--clip_excess_legs",
    "ankle_roll_bind": "--ankle_roll_bind_deg",
}


def locomotion_settings(args, *, actuator_profile) -> dict:
    """The command-line training settings as recorded in run_info (the same normalisation as before item 182)."""
    return {
        "actuator_profile": actuator_profile,
        "thermal_penalty": args.thermal_penalty,
        "knee_stop_penalty": float(args.knee_stop_penalty),
        "stop_model": bool(args.stop_model or args.stop_load_penalty),
        "stop_load_penalty": float(args.stop_load_penalty),
        "impact_penalty": float(args.impact_penalty),
        "knee_band": args.knee_band,
        "gait_shaping": bool(args.gait_shaping or args.gait_v2 or args.gait_v3),
        "gait_v2": bool(args.gait_v2),
        "gait_v3": bool(args.gait_v3),
        "target_clamp_margin_deg": float(args.target_clamp_margin_deg),
        "terrain": args.terrain, "stair_max_m": args.stair_max_m, "slope_max": args.slope_max,
        "feet_width_weight": args.feet_width_weight,
        "action_rate_weight": args.action_rate_weight,
        "arm_action_scale": args.arm_action_scale, "arm_deviation_weight": args.arm_deviation_weight,
        "arm_swing_weight": args.arm_swing_weight, "arm_swing_k": args.arm_swing_k,
        "arm_abduction_weight": args.arm_abduction_weight, "arm_gate_tilt_rate": args.arm_gate_tilt_rate,
        "arm_gate_hold_s": args.arm_gate_hold_s,
        "amp_weight": args.amp_weight, "amp_grad_penalty": args.amp_grad_penalty, "symmetry": bool(args.symmetry),
        "symmetry_mirror_loss": float(args.symmetry_mirror_loss),
        "clip_excess_arms": float(args.clip_excess_arms), "clip_excess_legs": float(args.clip_excess_legs),
        "ankle_roll_bind": ({"roll_limit_deg": float(args.ankle_roll_bind_deg),
                             "margin_deg": float(args.ankle_roll_bind_margin_deg)}
                            if args.ankle_roll_bind_deg > 0 else None),
    }


def _canon(v):
    """JSON-equal form (tuple vs list, 1 vs 1.0)."""
    v = json.loads(json.dumps(v))
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in sorted(v.items())}
    return v


def effective_settings(s: dict) -> dict:
    """Settings that change the trained task: parameters of a disabled feature compare as None."""
    e = {k: _canon(s.get(k, UNRECORDED)) for k in SETTING_FLAGS}
    if not s.get("arm_swing_weight"):
        for k in ("arm_swing_k", "arm_abduction_weight", "arm_gate_tilt_rate", "arm_gate_hold_s"):
            e[k] = None
    if not s.get("amp_weight"):
        e["amp_grad_penalty"] = None
    if not s.get("symmetry"):
        e["symmetry_mirror_loss"] = None
    if s.get("terrain", "flat") != "rough":
        e["stair_max_m"] = e["slope_max"] = None
    if not s.get("gait_shaping"):
        e["target_clamp_margin_deg"] = None
    return e


def settings_diff(prev_info: dict, current: dict) -> tuple[list[str], list[str]]:
    """(changed, unrecorded): changed = ``key (flag): before -> after``; unrecorded = keys the old run_info lacks."""
    unrecorded = [k for k in SETTING_FLAGS if k not in prev_info]
    a = effective_settings({k: v for k, v in prev_info.items()})
    b = effective_settings(current)
    changed = [f"{k} ({SETTING_FLAGS[k]}): {json.dumps(a[k])} -> {json.dumps(b[k])}"
               for k in SETTING_FLAGS if k not in unrecorded and a[k] != b[k]]
    return changed, unrecorded


def check_settings_continuity(prev_info: dict | None, current: dict, *, continue_run: bool,
                              allow_change: bool = False) -> list[str]:
    """Notes for this chunk; a ``--continue_run`` chunk whose settings differ raises ``ValueError`` unless
    ``allow_change`` (then noted as CHANGED)."""
    if not continue_run or prev_info is None:
        return []
    changed, unrecorded = settings_diff(prev_info, current)
    notes = [f"training settings {UNRECORDED} in the run's previous chunk: {', '.join(unrecorded)}"] if unrecorded else []
    if not changed:
        return notes
    if not allow_change:
        raise ValueError("--continue_run: training settings differ from the run's latest recorded chunk: "
                         f"{'; '.join(changed)}. Pass the run's flags again, or --allow_settings_change to change "
                         "the task mid-run")
    return notes + [f"training settings CHANGED {'; '.join(changed)} (--allow_settings_change)"]


def check_reward_continuity(prev_info: dict | None, reward_weights: dict, *, continue_run: bool,
                            allow_change: bool = False) -> list[str]:
    """Same rule for the built reward manager (term set + weights): also catches a code-default change between
    chunks. Run after the env is built, before the checkpoint is loaded."""
    if not continue_run or prev_info is None:
        return []
    prev = prev_info.get("reward_weights")
    if prev is None:
        return [f"reward_weights {UNRECORDED} in the run's previous chunk"]
    out = [f"-{n} (was {prev[n]})" for n in sorted(set(prev) - set(reward_weights))]
    out += [f"+{n} ({reward_weights[n]})" for n in sorted(set(reward_weights) - set(prev))]
    out += [f"{n}: {prev[n]} -> {reward_weights[n]}" for n in sorted(set(prev) & set(reward_weights))
            if float(prev[n]) != float(reward_weights[n])]
    if not out:
        return []
    if not allow_change:
        raise ValueError(f"--continue_run: reward terms differ from the run's latest recorded chunk: {'; '.join(out)}. "
                         "Pass --allow_settings_change to change the reward mid-run")
    return [f"reward_weights CHANGED {'; '.join(out)} (--allow_settings_change)"]


def chunk_flags_from_run_info(info: dict) -> dict:
    """``tools/run_chunked_locomotion.py`` argument values that reproduce the training settings ``info`` (a run_info
    dict) records: ``{dest: value}`` for every recorded key (item 343). A key the run_info lacks is left out (the
    chunk then uses its own flag / default and train_locomotion notes it as UNRECORDED)."""
    out: dict = {}
    for key in SETTING_FLAGS:
        if key in info and key != "ankle_roll_bind":  # gait_shaping recorded True with v2/v3: same task either way
            out[key] = info[key]
    if "ankle_roll_bind" in info:
        bind = info["ankle_roll_bind"]
        out["ankle_roll_bind_deg"] = float(bind["roll_limit_deg"]) if bind else 0.0
        if bind:
            out["ankle_roll_bind_margin_deg"] = float(bind["margin_deg"])
    for key in ("task", "pushes", "push_velocity", "push_interval"):  # run_info keys outside SETTING_FLAGS
        if key in info:
            out[key] = info[key]
    return out
