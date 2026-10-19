"""Mass DR (notes/13 T4, item 59): root add on ``world`` + one scale per limb. CPU, Isaac-free.

The event itself is exercised against a fake articulation when torch is importable (skipped otherwise).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401

from dropbear_wbc.robots.dropbear_names import EXPECTED_NUM_BODIES, ORPHAN_BODIES, ROOT_BODY  # noqa: E402
from dropbear_wbc.robots.mass_dr import (  # noqa: E402
    HIP_GIRDLE_BODIES,
    LIMB_MASS_SCALE,
    LIMB_NAMES,
    ROOT_MASS_ADD_LOCOMOTION,
    ROOT_MASS_ADD_TRACKING,
    body_masses_from_json,
    limb_body_index_plan,
    limb_mass_groups,
    limb_masses,
    mass_group_breakdown,
    monte_carlo_total_mass,
    sample_limb_scales,
    total_mass_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "dropbear_wbc"


def _articulation_bodies() -> list[str]:
    return list(body_masses_from_json())  # USD order, orphans dropped (= the 90 articulation links)


def test_interim_ranges_match_notes_13_t4():
    assert ROOT_MASS_ADD_TRACKING == (0.0, 6.0)
    assert ROOT_MASS_ADD_LOCOMOTION == (-1.0, 6.0)
    assert LIMB_MASS_SCALE == (0.9, 1.1)


def test_limb_groups_cover_limb_prefixes_and_exclude_root_head_torso():
    names = _articulation_bodies()
    assert len(names) == EXPECTED_NUM_BODIES
    groups = limb_mass_groups(names)
    assert tuple(groups) == LIMB_NAMES
    flat = [n for v in groups.values() for n in v]
    assert len(flat) == len(set(flat)), "a body in two limbs"
    assert ROOT_BODY not in flat
    assert not any(n.startswith(("head_", "torso_")) for n in flat)
    assert not set(flat) & set(ORPHAN_BODIES)
    for prefix, limb in (("LL_", "left_leg"), ("RL_", "right_leg"), ("LH_", "left_arm"), ("RH_", "right_arm")):
        expected = {n for n in names if n.startswith(prefix)}
        assert expected <= set(groups[limb])
    pg = {n for n in names if n.startswith("PG_")}
    assert len(pg) == 4 and pg <= set(flat)
    assert {len(v) for v in groups.values()} == {17, 9}
    assert [len(groups[k]) for k in LIMB_NAMES] == [17, 17, 9, 9]
    # every articulation body is either scaled, the root, head or the two shoulder-yaw rotors
    rest = [n for n in names if n not in set(flat)]
    assert sorted({n.split("_")[0] for n in rest}) == ["head", "torso", "world"]
    with_rotors = limb_mass_groups(names, include_shoulder_rotors=True)
    assert len(with_rotors["left_arm"]) == 10 and len(with_rotors["right_arm"]) == 10


def test_hip_girdle_side_matches_usd_joint_tree():
    tree = json.loads((ROOT / "docs" / "usd_tree_45586414.json").read_text(encoding="utf-8"))
    child = {j["name"]: j["b1"][0] for j in tree["joints"] if j.get("b1")}
    assert set(HIP_GIRDLE_BODIES["left_leg"]) == {child["PG_left_leg_pitch"], child["PG_left_leg_roll"]}
    assert set(HIP_GIRDLE_BODIES["right_leg"]) == {child["PG_right_leg_pitch"], child["PG_right_leg_roll"]}
    # torso_* rotors are the shoulder-yaw output links (children of LH_yaw / RH_yaw)
    assert child["LH_yaw"].startswith("torso_") and child["RH_yaw"].startswith("torso_")


def test_unknown_pg_body_fails_closed_and_empty_limb_raises():
    names = _articulation_bodies()
    with pytest.raises(ValueError, match="no known side"):
        limb_mass_groups(names + ["PG_new_bracket_1"])
    with pytest.raises(ValueError, match="no bodies"):
        limb_mass_groups([n for n in names if not n.startswith("RH_")])


def test_index_plan_one_column_per_limb():
    names = _articulation_bodies()
    ids, limb_idx, limbs = limb_body_index_plan(names)
    assert limbs == list(LIMB_NAMES)
    assert len(ids) == len(set(ids)) == 52
    groups = limb_mass_groups(names)
    for i, li in zip(ids, limb_idx):
        assert names[i] in groups[limbs[li]]
    with pytest.raises(ValueError, match="not on the articulation"):
        limb_body_index_plan(names[:10], groups)


def test_sample_limb_scales_range_and_shape():
    s = sample_limb_scales(20_000, rng=np.random.default_rng(1))
    assert tuple(s) == LIMB_NAMES
    for v in s.values():
        assert v.shape == (20_000,)
        assert v.min() >= 0.9 and v.max() <= 1.1
        assert v.mean() == pytest.approx(1.0, abs=0.005)
    # independent across limbs
    assert abs(np.corrcoef(s["left_leg"], s["right_leg"])[0, 1]) < 0.03


def test_mass_breakdown_and_envelope():
    masses = body_masses_from_json()
    full = body_masses_from_json(articulation_only=False)
    assert sum(full.values()) == pytest.approx(56.229, abs=1e-3)
    assert sum(masses.values()) == pytest.approx(56.229 - 3 * 0.0181, abs=2e-3)
    g = mass_group_breakdown(masses)
    assert g["world"] == pytest.approx(19.505, abs=1e-3)
    assert g["PG"] == pytest.approx(4.671, abs=2e-3)
    limbs = limb_masses(masses)
    assert limbs["left_leg"] == pytest.approx(limbs["right_leg"], abs=0.05)
    assert limbs["left_arm"] == pytest.approx(limbs["right_arm"], abs=1e-6)
    limb_total = sum(limbs.values())
    assert 33.0 < limb_total < 34.0
    env_t = total_mass_envelope(masses, ROOT_MASS_ADD_TRACKING)
    env_l = total_mass_envelope(masses, ROOT_MASS_ADD_LOCOMOTION)
    assert env_t["limb_total"] == pytest.approx(limb_total)
    assert env_t["min"] == pytest.approx(env_t["nominal"] - 0.1 * limb_total)
    assert env_t["max"] == pytest.approx(env_t["nominal"] + 6.0 + 0.1 * limb_total)
    assert env_l["min"] == pytest.approx(env_t["min"] - 1.0)
    assert env_l["max"] == pytest.approx(env_t["max"])
    # covers the ~60 kg user plant with margin on both tasks; old loco (-1, +2) root-only did not
    for e in (env_t, env_l):
        assert e["root_only_max"] > 60.0 and e["max"] > 63.0
    assert total_mass_envelope(masses, (-1.0, 2.0))["root_only_max"] < 60.0
    mc = monte_carlo_total_mass(masses, ROOT_MASS_ADD_TRACKING, n=50_000)
    assert env_t["min"] <= mc.min() and mc.max() <= env_t["max"]
    # per-limb correlated draw keeps a wider spread than the (wrong) independent per-body draw
    mc_pb = monte_carlo_total_mass(masses, (0.0, 0.0), n=50_000, per_body=True)
    mc_pl = monte_carlo_total_mass(masses, (0.0, 0.0), n=50_000)
    assert mc_pl.std() > 1.5 * mc_pb.std()


# ---------------------------------------------------------------------------------------------
# Config wiring (AST: isaaclab is not importable on the CPU box)
# ---------------------------------------------------------------------------------------------

def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _event_terms(path: Path) -> dict[str, ast.Call]:
    cls = _class(path, "EventCfg")
    return {s.targets[0].id: s.value for s in cls.body if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call)}


def _kw(call: ast.Call, key: str):
    return next(k.value for k in call.keywords if k.arg == key)


def _params(call: ast.Call) -> dict[str, str]:
    d = _kw(call, "params")
    return {ast.literal_eval(k): ast.unparse(v) for k, v in zip(d.keys, d.values)}


@pytest.mark.parametrize("rel,const", [
    ("tasks/tracking/tracking_env_cfg.py", "ROOT_MASS_ADD_TRACKING"),
    ("tasks/locomotion/velocity_env_cfg.py", "ROOT_MASS_ADD_LOCOMOTION"),
])
def test_event_cfg_has_root_add_and_limb_scale(rel, const):
    terms = _event_terms(SRC / rel)
    base = terms["add_base_mass"]
    assert ast.unparse(_kw(base, "func")) == "mdp.randomize_rigid_body_mass"
    assert ast.literal_eval(_kw(base, "mode")) == "startup"
    p = _params(base)
    assert p["mass_distribution_params"] == const
    assert p["operation"] == "'add'"
    assert "ROOT_BODY" in p["asset_cfg"]
    limb = terms["scale_limb_mass"]
    assert ast.unparse(_kw(limb, "func")) == "mdp.randomize_limb_mass_scale"
    assert ast.literal_eval(_kw(limb, "mode")) == "startup"
    assert _params(limb)["scale_range"] == "LIMB_MASS_SCALE"


def test_mdp_packages_reexport_shared_event():
    for rel in ("tasks/tracking/mdp/events.py", "tasks/locomotion/mdp/events.py"):
        src = (SRC / rel).read_text(encoding="utf-8")
        assert "from dropbear_wbc.tasks.mass_events import randomize_limb_mass_scale" in src


@pytest.mark.parametrize("rel,cls", [
    ("tasks/tracking/config/dropbear/flat_env_cfg.py", "DropbearFlatPlayEnvCfg"),
    ("tasks/locomotion/config/dropbear/flat_env_cfg.py", "DropbearVelocityFlatPlayEnvCfg"),
])
def test_play_cfgs_clear_mass_dr(rel, cls):
    src = ast.unparse(_class(SRC / rel, cls))
    assert "self.events.add_base_mass = None" in src
    assert "self.events.scale_limb_mass = None" in src


def test_perturbed_play_restores_mass_dr():
    play = (ROOT / "scripts" / "play.py").read_text(encoding="utf-8")
    assert '"add_base_mass", "scale_limb_mass")' in play
    loco = (ROOT / "scripts" / "play_locomotion.py").read_text(encoding="utf-8")
    assert "env_cfg.events.scale_limb_mass = train_cfg.events.scale_limb_mass" in loco


# ---------------------------------------------------------------------------------------------
# Event on a fake articulation (torch only)
# ---------------------------------------------------------------------------------------------

def _fake_env(torch, names, masses, n_env=6):
    from types import SimpleNamespace

    m0 = torch.tensor([[masses[n] for n in names]] * n_env, dtype=torch.float32)
    I0 = torch.arange(1, 10, dtype=torch.float32).repeat(n_env, len(names), 1) * m0[..., None]

    class View:
        def __init__(self):
            self.m, self.I = m0.clone(), I0.clone()

        def get_masses(self):
            return self.m.clone()

        def set_masses(self, m, env_ids):
            self.m[env_ids] = m[env_ids]

        def get_inertias(self):
            return self.I.clone()

        def set_inertias(self, I, env_ids):
            self.I[env_ids] = I[env_ids]

    asset = SimpleNamespace(body_names=list(names), root_physx_view=View(),
                            data=SimpleNamespace(default_mass=m0.clone(), default_inertia=I0.clone()))
    scene = type("Scene", (), {"num_envs": n_env, "__getitem__": lambda self, k: asset})()
    env = SimpleNamespace(scene=scene)
    return env, asset, m0, I0


def test_event_one_draw_per_limb_on_fake_articulation():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from dropbear_wbc.tasks.mass_events import randomize_limb_mass_scale

    masses = body_masses_from_json()
    names = list(masses)
    env, asset, m0, I0 = _fake_env(torch, names, masses)
    cfg = SimpleNamespace(name="robot")
    torch.manual_seed(0)
    randomize_limb_mass_scale(env, None, cfg)
    m1 = asset.root_physx_view.m
    ratio = m1 / m0
    groups = limb_mass_groups(names)
    idx = {n: i for i, n in enumerate(names)}
    for li, limb in enumerate(LIMB_NAMES):
        cols = [idx[n] for n in groups[limb]]
        r = ratio[:, cols]
        assert torch.allclose(r, r[:, :1].expand_as(r), atol=1e-6), f"{limb}: bodies drew different scales"
        assert (r >= 0.9 - 1e-6).all() and (r <= 1.1 + 1e-6).all()
        assert torch.allclose(asset.data.limb_mass_scale[:, li], r[:, 0], atol=1e-6)
    fixed = [idx[n] for n in names if n == ROOT_BODY or n.startswith(("head_", "torso_"))]
    assert torch.equal(m1[:, fixed], m0[:, fixed])
    # inertia scaled by the same ratio (Isaac 2.2 recompute_inertia pattern)
    I1 = asset.root_physx_view.I
    assert torch.allclose(I1, I0 * ratio[..., None], rtol=1e-5)
    # repeated call starts from default (no compounding); env subset leaves the others alone
    before = asset.root_physx_view.m.clone()
    randomize_limb_mass_scale(env, torch.tensor([1, 3]), cfg)
    after = asset.root_physx_view.m
    keep = [0, 2, 4, 5]
    assert torch.equal(after[keep], before[keep])
    r2 = after[[1, 3]] / m0[[1, 3]]
    assert (r2 >= 0.9 - 1e-6).all() and (r2 <= 1.1 + 1e-6).all()
    assert asset.data.limb_mass_scale_names == list(LIMB_NAMES)


def test_motor_mass_offsets_event_moves_live_and_default_mass_once():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets
    from dropbear_wbc.tasks.mass_events import apply_body_mass_offsets, randomize_limb_mass_scale

    masses = body_masses_from_json()
    names = list(masses)
    env, asset, m0, I0 = _fake_env(torch, names, masses)
    cfg = SimpleNamespace(name="robot")
    off = motor_mass_offsets("hw_cem6")
    apply_body_mass_offsets(env, None, cfg, off)
    apply_body_mass_offsets(env, None, cfg, off)  # second call: no-op
    idx = {n: i for i, n in enumerate(names)}
    for b, d in off.items():
        assert torch.allclose(asset.root_physx_view.m[:, idx[b]], m0[:, idx[b]] + d, atol=1e-5)
        assert torch.allclose(asset.data.default_mass[:, idx[b]], m0[:, idx[b]] + d, atol=1e-5)
        r = (m0[:, idx[b]] + d) / m0[:, idx[b]]
        assert torch.allclose(asset.root_physx_view.I[:, idx[b]], I0[:, idx[b]] * r[:, None], rtol=1e-5)
    untouched = [i for n, i in idx.items() if n not in off]
    assert torch.equal(asset.root_physx_view.m[:, untouched], m0[:, untouched])
    assert abs(float(asset.root_physx_view.m[0].sum() - m0[0].sum()) - sum(off.values())) < 1e-3
    # the limb scale restarts from the moved defaults, so the thigh is randomized around the new plant
    torch.manual_seed(0)
    randomize_limb_mass_scale(env, None, cfg)
    thigh = idx["LL_RMD_X10_S2_MIR4__3_Stator_1"]
    r = asset.root_physx_view.m[:, thigh] / asset.data.default_mass[:, thigh]
    assert (r >= 0.9 - 1e-6).all() and (r <= 1.1 + 1e-6).all()
