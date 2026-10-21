"""Mass domain randomization plan (notes/13 T4, backlog item 59). Pure Python + numpy: no Isaac/torch imports.

Two startup terms, used by both the tracking and the velocity task:

* ``add_base_mass`` -- Isaac Lab ``randomize_rigid_body_mass`` on the root body ``world`` (rigid torso + pelvis,
  19.5 kg CAD) with ``operation="add"``. Interim ranges until the robot is weighed (item 61): the CAD plant is
  56.23 kg, the user reports ~60 kg, plus an unweighed battery -> tracking ``(0, +6)`` kg; locomotion keeps its old
  ``-1`` floor and widens the top from ``+2`` to ``+6`` kg. Replace both with the measured number.
* ``scale_limb_mass`` -- :func:`dropbear_wbc.tasks.mass_events.randomize_limb_mass_scale`: ONE scale
  ``U(0.9, 1.1)`` per limb per env, applied to every body of that limb (CAD density/hardware errors are correlated
  within a limb: same print material, same fasteners, same motor batch). Isaac's ``randomize_rigid_body_mass`` with
  ``operation="scale"`` on a regex draws an independent scale per body, which averages out over a limb's ~10-16
  bodies (sigma of the limb total shrinks by ~sqrt(n)) and is not the T4 model.

Limb body groups (names of the plant USD ``45586414``, topology from ``docs/usd_tree_45586414.json``):

* ``left_leg``: every ``LL_*`` body + the two left hip bodies ``PG_RMD_X10_S2_MIR4__2_Rotor_1`` (child of
  ``PG_left_leg_pitch``) and ``PG_RMD_X10_V3Rotor_1`` (child of ``PG_left_leg_roll``);
* ``right_leg``: every ``RL_*`` body + ``PG_RMD_X10_S2_MIR4__1_Rotor_1`` / ``PG_RMD_X10_V3__1_Rotor_1``;
* ``left_arm`` / ``right_arm``: every ``LH_*`` / ``RH_*`` body.

Not scaled: ``world`` (covered by the root add), ``head_*`` (Stewart neck, fixed until weighed) and the two
``torso_RMD_X10*`` rotors. The latter are kinematically the shoulder-yaw output links (children of ``LH_yaw`` /
``RH_yaw``) despite the ``torso_`` prefix; they are an off-the-shelf RMD-X10 rotor (vendor mass, low uncertainty),
so they stay fixed by default (``include_shoulder_rotors=True`` adds them to the arms). The three
:data:`~dropbear_wbc.robots.dropbear_names.ORPHAN_BODIES` are deactivated at spawn and never part of the
articulation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dropbear_wbc.robots.dropbear_names import ORPHAN_BODIES, REPO_ROOT, ROOT_BODY, USD_SHA256

ROOT_MASS_ADD_TRACKING: tuple[float, float] = (0.0, 6.0)
"""Tracking ``add_base_mass`` on ``world`` [kg] (notes/13 T4 interim; item 61 weighing replaces it)."""
ROOT_MASS_ADD_LOCOMOTION: tuple[float, float] = (-1.0, 6.0)
"""Velocity ``add_base_mass`` on ``world`` [kg]; was ``(-1.0, 2.0)`` (does not reach the ~60 kg user plant)."""
LIMB_MASS_SCALE: tuple[float, float] = (0.9, 1.1)
"""Per-limb mass scale (one draw per limb per env, notes/13 T4)."""

# ---------------------------------------------------------------------------------------------
# Measured robot mass (opt-in: ``--robot_mass_kg`` on train / play / robustness_bench)
# ---------------------------------------------------------------------------------------------
# ``hw_motor_specs.motor_mass_offsets`` swaps the CAD's X10-S2 for each twin-v3 build's motors (hw_cem6 -4.62 kg), and
# the root ranges above were sized for the CAD plant (56.17 kg), so hw_cem6 samples 48.8-60.3 kg and play / eval
# (``add_base_mass`` off) run at the nominal 51.55 kg. Nothing here changes that by default. Given a measured (or
# agreed) total mass, the gap to "CAD + this profile's motor swap" goes on ``world`` as a fixed plant offset (kept in
# play, like the motor masses) and the root DR becomes a band around it.

ROOT_MASS_BAND_TRACKING: tuple[float, float] = (-3.0, 3.0)
"""Tracking root DR [kg] around a given ``robot_mass_kg`` (scale error, battery state, cabling)."""
ROOT_MASS_BAND_LOCOMOTION: tuple[float, float] = (-4.0, 3.0)
"""Velocity root DR [kg] around a given ``robot_mass_kg``; keeps locomotion's 1 kg lower floor relative to tracking."""
ROBOT_MASS_LIMITS_KG: tuple[float, float] = (40.0, 90.0)
"""Plausible total mass [kg]; a value outside fails (catches lb / g / typo inputs)."""


def cad_total_mass_kg() -> float:
    """Articulation mass of the plant USD [kg] (orphan bearings dropped): 56.17."""
    return float(sum(body_masses_from_json().values()))


def motor_swap_delta_kg(profile: str) -> float:
    """Total mass change [kg] of ``profile``'s motors vs the CAD (0 for profiles without a motor-mass swap)."""
    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets

    return float(sum(motor_mass_offsets(profile).values()))


def check_robot_mass_kg(robot_mass_kg: float | None) -> float | None:
    """``None`` passes through; a number must be finite and inside :data:`ROBOT_MASS_LIMITS_KG`."""
    if robot_mass_kg is None:
        return None
    m = float(robot_mass_kg)
    lo, hi = ROBOT_MASS_LIMITS_KG
    if not (lo <= m <= hi):  # also rejects nan
        raise ValueError(f"robot_mass_kg {robot_mass_kg!r} outside {ROBOT_MASS_LIMITS_KG} kg")
    return m


def unmodelled_mass_kg(profile: str, robot_mass_kg: float) -> float:
    """Mass the CAD + the profile's motor swap does not explain [kg]: ``robot_mass_kg`` - (CAD + swap delta)."""
    return float(check_robot_mass_kg(robot_mass_kg)) - cad_total_mass_kg() - motor_swap_delta_kg(profile)


def plant_mass_offsets(profile: str, robot_mass_kg: float | None = None) -> dict[str, float]:
    """Fixed per-body mass change [kg] of the plant: the profile's motor swap, plus (only when ``robot_mass_kg`` is
    given) the unmodelled mass on ``world``. Without ``robot_mass_kg`` this equals ``motor_mass_offsets(profile)``."""
    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets

    out = dict(motor_mass_offsets(profile))
    if robot_mass_kg is not None:
        out[ROOT_BODY] = round(out.get(ROOT_BODY, 0.0) + unmodelled_mass_kg(profile, robot_mass_kg), 4)
    return out


def predicted_mass_kg(profile: str, robot_mass_kg: float | None = None) -> float:
    """Nominal total mass [kg] of the profile's plant before DR."""
    return cad_total_mass_kg() + float(sum(plant_mass_offsets(profile, robot_mass_kg).values()))


def root_mass_add_range(task: str, robot_mass_kg: float | None = None) -> tuple[float, float]:
    """``add_base_mass`` range [kg] on ``world`` for ``task`` (``tracking`` / ``locomotion``): the unchanged defaults
    without ``robot_mass_kg``, the band around the measured mass with it."""
    if task not in ("tracking", "locomotion"):
        raise ValueError(f"unknown task {task!r}")
    if robot_mass_kg is None:
        return ROOT_MASS_ADD_TRACKING if task == "tracking" else ROOT_MASS_ADD_LOCOMOTION
    check_robot_mass_kg(robot_mass_kg)
    return ROOT_MASS_BAND_TRACKING if task == "tracking" else ROOT_MASS_BAND_LOCOMOTION


def run_robot_mass_kg(run_infos) -> float | None:
    """``robot_mass_kg`` a run was trained with: the newest chunk's non-null value of ``run_infos`` (oldest first), or
    ``None`` (runs before the option, or trained without it). Play / eval default to it so they replay that plant."""
    for info in reversed(list(run_infos)):
        if info.get("robot_mass_kg") is not None:
            return float(info["robot_mass_kg"])
    return None


LIMB_NAMES: tuple[str, ...] = ("left_leg", "right_leg", "left_arm", "right_arm")
LIMB_PREFIXES: dict[str, tuple[str, ...]] = {
    "left_leg": ("LL_",),
    "right_leg": ("RL_",),
    "left_arm": ("LH_",),
    "right_arm": ("RH_",),
}
HIP_GIRDLE_BODIES: dict[str, tuple[str, ...]] = {
    "left_leg": ("PG_RMD_X10_S2_MIR4__2_Rotor_1", "PG_RMD_X10_V3Rotor_1"),
    "right_leg": ("PG_RMD_X10_S2_MIR4__1_Rotor_1", "PG_RMD_X10_V3__1_Rotor_1"),
}
"""``PG_*`` bodies carry no side in their name; side from the USD joint tree (``PG_<side>_leg_pitch`` / ``_roll``
children). Any other ``PG_*`` body fails closed in :func:`limb_mass_groups`."""
SHOULDER_ROTOR_BODIES: dict[str, tuple[str, ...]] = {
    "left_arm": ("torso_RMD_X10__1_Rotor_1",),
    "right_arm": ("torso_RMD_X10Rotot_1",),
}
"""Shoulder-yaw output links (children of ``LH_yaw`` / ``RH_yaw``), 0.344 kg each; excluded by default."""
FIXED_PREFIXES: tuple[str, ...] = ("head_", "torso_")
"""Bodies never scaled by the limb term (``world`` is excluded explicitly)."""

BODY_PROPERTIES_JSON: Path = REPO_ROOT / "data" / "robot" / f"usd_body_properties_{USD_SHA256[:8]}.json"


def limb_mass_groups(body_names: Sequence[str], include_shoulder_rotors: bool = False) -> dict[str, list[str]]:
    """Limb -> body names (in ``body_names`` order) for the per-limb mass scale.

    Only names present in ``body_names`` are returned (an articulation's ``body_names`` already omits the orphan
    bearings; they are dropped here too). ``world``, ``head_*`` and (by default) ``torso_*`` are never included.
    Raises ``ValueError`` for a ``PG_*`` body whose side is unknown or if any limb resolves to no body.
    """
    pg_side = {b: limb for limb, bodies in HIP_GIRDLE_BODIES.items() for b in bodies}
    rotor_side = {b: limb for limb, bodies in SHOULDER_ROTOR_BODIES.items() for b in bodies}
    groups: dict[str, list[str]] = {limb: [] for limb in LIMB_NAMES}
    orphans = set(ORPHAN_BODIES)
    for name in body_names:
        if name == ROOT_BODY or name in orphans:
            continue
        if name.startswith("PG_"):
            if name not in pg_side:
                raise ValueError(f"PG body {name!r} has no known side (update HIP_GIRDLE_BODIES from the USD tree)")
            groups[pg_side[name]].append(name)
            continue
        if include_shoulder_rotors and name in rotor_side:
            groups[rotor_side[name]].append(name)
            continue
        if name.startswith(FIXED_PREFIXES):
            continue
        for limb, prefixes in LIMB_PREFIXES.items():
            if name.startswith(prefixes):
                groups[limb].append(name)
                break
    empty = [limb for limb, names in groups.items() if not names]
    if empty:
        raise ValueError(f"limb mass groups resolved to no bodies for {empty}")
    return groups


def limb_body_index_plan(body_names: Sequence[str], groups: Mapping[str, Sequence[str]] | None = None
                         ) -> tuple[list[int], list[int], list[str]]:
    """Flatten ``groups`` (default :func:`limb_mass_groups`) into (body ids, limb index per body, limb names).

    ``body_ids[k]`` is the articulation index of the k-th scaled body and ``limb_index[k]`` the column of its limb's
    scale; the event builds a per-body scale with ``scales[:, limb_index]`` (one draw per limb, shared by its bodies).
    """
    groups = limb_mass_groups(body_names) if groups is None else groups
    index = {n: i for i, n in enumerate(body_names)}
    limbs = list(groups)
    missing = [n for names in groups.values() for n in names if n not in index]
    if missing:
        raise ValueError(f"limb mass bodies not on the articulation: {missing}")
    seen: dict[str, str] = {}
    body_ids: list[int] = []
    limb_index: list[int] = []
    for li, limb in enumerate(limbs):
        for n in groups[limb]:
            if n in seen:
                raise ValueError(f"body {n!r} in two limbs ({seen[n]}, {limb})")
            seen[n] = limb
            body_ids.append(index[n])
            limb_index.append(li)
    return body_ids, limb_index, limbs


def sample_limb_scales(n_envs: int, rng: np.random.Generator | None = None,
                       scale_range: tuple[float, float] = LIMB_MASS_SCALE,
                       limbs: Sequence[str] = LIMB_NAMES) -> dict[str, np.ndarray]:
    """One uniform scale per limb per env: limb -> array (n_envs,). Reference sampler (the event uses torch)."""
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = scale_range
    draw = rng.uniform(lo, hi, size=(int(n_envs), len(limbs)))
    return {limb: draw[:, i] for i, limb in enumerate(limbs)}


# ---------------------------------------------------------------------------------------------
# Mass bookkeeping (CAD body table) for notes / tests
# ---------------------------------------------------------------------------------------------

def body_masses_from_json(path: str | Path | None = None, articulation_only: bool = True) -> dict[str, float]:
    """Body -> mass [kg] from ``usd_body_properties_<sha8>.json`` (orphans dropped when ``articulation_only``)."""
    doc = json.loads(Path(path or BODY_PROPERTIES_JSON).read_text(encoding="utf-8"))
    orphans = set(ORPHAN_BODIES) if articulation_only else set()
    return {k: float(v["mass"]) for k, v in doc["bodies"].items() if k not in orphans}


def mass_group_of(name: str) -> str:
    """Bookkeeping group: world / LL / RL / LH / RH / PG / head / torso_rotor / other."""
    if name == ROOT_BODY:
        return "world"
    for tag in ("LL", "RL", "LH", "RH", "PG"):
        if name.startswith(tag + "_"):
            return tag
    if name.startswith("head_"):
        return "head"
    if name.startswith("torso_"):
        return "torso_rotor"
    return "other"


def mass_group_breakdown(masses: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for n, m in masses.items():
        g = mass_group_of(n)
        out[g] = out.get(g, 0.0) + float(m)
    return out


def limb_masses(masses: Mapping[str, float], include_shoulder_rotors: bool = False) -> dict[str, float]:
    groups = limb_mass_groups(list(masses), include_shoulder_rotors=include_shoulder_rotors)
    return {limb: float(sum(masses[n] for n in names)) for limb, names in groups.items()}


def total_mass_envelope(masses: Mapping[str, float], root_add: tuple[float, float],
                        limb_scale: tuple[float, float] = LIMB_MASS_SCALE,
                        include_shoulder_rotors: bool = False) -> dict[str, float]:
    """Analytic total-mass bounds [kg] under ``root_add`` on ``world`` and a per-limb scale (all limbs at an end)."""
    nominal = float(sum(masses.values()))
    limb_total = float(sum(limb_masses(masses, include_shoulder_rotors).values()))
    lo_s, hi_s = limb_scale
    return {
        "nominal": nominal,
        "limb_total": limb_total,
        "root_only_min": nominal + root_add[0],
        "root_only_max": nominal + root_add[1],
        "limb_only_min": nominal + (lo_s - 1.0) * limb_total,
        "limb_only_max": nominal + (hi_s - 1.0) * limb_total,
        "min": nominal + root_add[0] + (lo_s - 1.0) * limb_total,
        "max": nominal + root_add[1] + (hi_s - 1.0) * limb_total,
    }


def monte_carlo_total_mass(masses: Mapping[str, float], root_add: tuple[float, float], n: int = 100_000,
                           rng: np.random.Generator | None = None,
                           limb_scale: tuple[float, float] = LIMB_MASS_SCALE,
                           per_body: bool = False) -> np.ndarray:
    """Sampled total mass [kg] (n,). ``per_body=True`` reproduces the WRONG independent-per-body scale (comparison)."""
    rng = np.random.default_rng(0) if rng is None else rng
    names = list(masses)
    m = np.array([masses[k] for k in names])
    groups = limb_mass_groups(names)
    total = np.full(n, m.sum()) + rng.uniform(root_add[0], root_add[1], size=n)
    idx = {k: i for i, k in enumerate(names)}
    for limb, bodies in groups.items():
        mb = np.array([m[idx[b]] for b in bodies])
        if per_body:
            s = rng.uniform(*limb_scale, size=(n, len(bodies)))
            total += ((s - 1.0) * mb).sum(axis=1)
        else:
            s = rng.uniform(*limb_scale, size=n)
            total += (s - 1.0) * mb.sum()
    return total
