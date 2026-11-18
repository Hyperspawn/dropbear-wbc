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

import hashlib
import json
from pathlib import Path

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
    "symmetric_clamp": "--symmetric_clamp",
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
        "symmetric_clamp": bool(args.symmetric_clamp),
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
        e["target_clamp_margin_deg"] = e["symmetric_clamp"] = None
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


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file (streamed; same as ``dropbear_wbc.isaac.launch.sha256_file``)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_stand_continuity(prev_info: dict | None, stand_path: str | Path, *, continue_run: bool,
                           allow_change: bool = False) -> list[str]:
    """The reset NPZ of a ``--continue_run`` chunk must be the run's (item 346): it sets the reset state, the anchor
    height target and the fall threshold. Compared by content (sha256), not path, so a run moved between machines
    still continues; a regenerated default clip or a missing ``--stand_npz`` is refused unless ``allow_change``."""
    if not continue_run or prev_info is None:
        return []
    prev_sha = prev_info.get("stand_npz_sha256")
    if prev_sha is None:
        return [f"stand_npz_sha256 {UNRECORDED} in the run's previous chunk"]
    sha = file_sha256(stand_path)
    if sha == prev_sha:
        return []
    msg = (f"stand NPZ {stand_path} (sha {sha[:12]}) differs from the run's {prev_info.get('stand_npz')} "
           f"(sha {prev_sha[:12]})")
    if not allow_change:
        raise ValueError(f"--continue_run: {msg}. Pass the run's --stand_npz, or --allow_settings_change")
    return [f"stand NPZ CHANGED: {msg} (--allow_settings_change)"]


def resolve_run_stand_npz(info: dict | None, repo_root: str | Path) -> tuple[str | None, str]:
    """``(path, note)``: a local file holding the stand NPZ a run recorded (item 346), for
    ``run_chunked_locomotion.py --resume_run``. Tries the recorded path, then its ``data/...`` part under
    ``repo_root`` (a run dir copied from another machine); only a file whose sha256 matches is returned."""
    if not info or not info.get("stand_npz") or not info.get("stand_npz_sha256"):
        return None, f"stand_npz {UNRECORDED} in run_info"
    rec = Path(str(info["stand_npz"]).replace("\\", "/"))
    cands = [rec]
    if "data" in rec.parts:
        i = len(rec.parts) - 1 - rec.parts[::-1].index("data")
        cands.append(Path(repo_root).joinpath(*rec.parts[i:]))
    for c in cands:
        if c.is_file() and file_sha256(c) == info["stand_npz_sha256"]:
            return str(c), "matched"
    return None, (f"no local file matches the run's stand NPZ {info['stand_npz']} "
                  f"(sha {str(info['stand_npz_sha256'])[:12]}); train_locomotion refuses unless the default matches")
