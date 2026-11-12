"""Isaac Lab 3.0 deprecations (notes/334): our mass events keep their own default-plant snapshot instead of
``data.default_mass`` / ``default_inertia`` (lazy live snapshots in 3.0, removed in 4.0); the motor-mass offset runs
before the mass DR; PhysX-only diagnostics get the view from ``compat.physx_view`` (clear error on Newton)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import sdk_test_paths  # noqa: F401
from test_body_prop_writes import ROOT, _code_lines, _Lab3Asset

_DEFAULT = re.compile(r"\.default_(mass|inertia)\b")
DEFAULT_ALLOW = {"source/dropbear_wbc/isaac/compat.py"}
PHYSX_TOOLS = ("tools/probe_loop_binding.py", "tools/probe_passive_damping.py", "tools/probe_knee_authority.py",
               "tools/inspect_dropbear_articulation.py", "source/dropbear_wbc/isaac/telemetry.py")
CFGS = ("source/dropbear_wbc/tasks/tracking/tracking_env_cfg.py",
        "source/dropbear_wbc/tasks/locomotion/velocity_env_cfg.py")


def default_offenders(root: Path = ROOT) -> list[str]:
    out = []
    for sub in ("source/dropbear_wbc", "tools", "scripts"):
        for p in sorted((root / sub).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel in DEFAULT_ALLOW:
                continue
            for i, code in _code_lines(p.read_text(encoding="utf-8")):
                if _DEFAULT.search(code):
                    out.append(f"{rel}:{i}")
    return out


def test_no_default_mass_inertia_reads_outside_compat():
    assert default_offenders() == [], "use compat.body_prop_defaults (data.default_* deprecated in Lab 3.0)"


def test_default_gate_catches_planted_read(tmp_path):
    d = tmp_path / "source/dropbear_wbc/tasks"
    d.mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "scripts").mkdir()
    (d / "x.py").write_text("m = asset.data.default_mass\nfoo_default_mass = 1\n", encoding="utf-8")
    assert default_offenders(tmp_path) == ["source/dropbear_wbc/tasks/x.py:1"]


def _event_field_order(path: str) -> list[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
    names = []
    for st in cls.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names.append(st.target.id)
        elif isinstance(st, ast.Assign):
            names += [t.id for t in st.targets if isinstance(t, ast.Name)]
    return names


@pytest.mark.parametrize("path", CFGS, ids=lambda p: Path(p).stem)
def test_motor_mass_term_declared_before_mass_dr(path):
    order = _event_field_order(path)
    assert order[0] == "motor_mass", order
    assert order.index("motor_mass") < order.index("add_base_mass") < order.index("scale_limb_mass")


@pytest.mark.parametrize("path", PHYSX_TOOLS, ids=lambda p: Path(p).stem)
def test_physx_diagnostics_use_compat_view(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "physx_view(" in text


def _lab3_env(torch, n_env=4, newton=False):
    from dropbear_wbc.robots.mass_dr import body_masses_from_json

    masses = body_masses_from_json()
    names = list(masses)
    cls = _Lab3Asset
    if newton:
        cls = type("Articulation", (_Lab3Asset,), {"__module__": "isaaclab_newton.assets.articulation"})
    a = cls(torch, n_env, len(names))
    a.m[:] = torch.tensor([masses[n] for n in names], dtype=torch.float32)
    a.body_names = names
    scene = type("Scene", (), {"num_envs": n_env, "__getitem__": lambda self, k: a})()
    return SimpleNamespace(scene=scene), a, names


def test_lab3_events_never_touch_deprecated_defaults_and_compose_in_cfg_order():
    """motor_mass then limb scale (the cfg order) on a 3.0-shaped asset: thigh mass = s*(m0 + d) exactly,
    inertia = I0 * (m0 + d)/m0 * s, and data.default_mass/default_inertia are never read."""
    torch = pytest.importorskip("torch")
    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets
    from dropbear_wbc.tasks.mass_events import apply_body_mass_offsets, randomize_limb_mass_scale

    env, a, names = _lab3_env(torch)
    m0, I0 = a.m.clone(), a.I.clone()
    cfg = SimpleNamespace(name="robot")
    off = motor_mass_offsets("hw_cem6")
    apply_body_mass_offsets(env, None, cfg, off)
    torch.manual_seed(1)
    randomize_limb_mass_scale(env, None, cfg)
    randomize_limb_mass_scale(env, None, cfg)  # restarts from the snapshot: no compounding
    assert a._m0 is None and a._I0 is None, "deprecated data.default_mass/default_inertia was read"
    idx = {n: i for i, n in enumerate(names)}
    from dropbear_wbc.robots.mass_dr import limb_body_index_plan, limb_mass_groups

    body_ids, limb_index, _ = limb_body_index_plan(names, limb_mass_groups(names))
    col = {b: li for b, li in zip(body_ids, limb_index)}
    hit = 0
    for b, d in off.items():
        j = idx[b]
        if j not in col:
            continue
        s = a.data.limb_mass_scale[:, col[j]]
        assert torch.allclose(a.m[:, j], (m0[:, j] + d) * s, rtol=1e-5)
        r = (m0[:, j] + d) / m0[:, j]
        assert torch.allclose(a.I[:, j], I0[:, j] * (r * s)[:, None], rtol=1e-4)
        hit += 1
    assert hit > 0


def test_snapshot_taken_before_first_write_and_shared():
    torch = pytest.importorskip("torch")
    from dropbear_wbc.isaac import compat
    from dropbear_wbc.tasks.mass_events import randomize_limb_mass_scale

    env, a, _ = _lab3_env(torch)
    m0 = a.m.clone()
    randomize_limb_mass_scale(env, None, SimpleNamespace(name="robot"))
    d = compat.body_prop_defaults(a)
    assert torch.equal(d["mass"], m0) and d["mass"].data_ptr() != a.m.data_ptr()
    assert compat.body_prop_defaults(a) is d


def test_lab2_shift_moves_lab_defaults_too():
    torch = pytest.importorskip("torch")
    from dropbear_wbc.isaac import compat

    m0 = torch.rand(3, 4) + 1.0
    I0 = torch.rand(3, 4, 9)
    a = SimpleNamespace(root_physx_view=SimpleNamespace(), data=SimpleNamespace(default_mass=m0.clone(),
                                                                                  default_inertia=I0.clone()))
    new = m0[:, [2]] + 0.5
    compat.shift_body_mass_defaults(a, [2], new, (new / m0[:, [2]])[..., None])
    assert torch.allclose(a.data.default_mass[:, 2], m0[:, 2] + 0.5)
    assert torch.allclose(compat.body_prop_defaults(a)["mass"][:, 2], m0[:, 2] + 0.5)
    assert torch.allclose(a.data.default_inertia[:, 2], I0[:, 2] * ((m0[:, 2] + 0.5) / m0[:, 2])[:, None])


def test_physx_view_backends():
    torch = pytest.importorskip("torch")
    from dropbear_wbc.isaac import compat

    _, newton, _ = _lab3_env(torch, newton=True)
    with pytest.raises(compat.PhysXOnlyError, match="PhysX-only"):
        compat.physx_view(newton, "tools/probe_x.py")

    class _Physx3(_Lab3Asset):
        root_view = "physx3-view"

        @property
        def root_physx_view(self):  # deprecated alias in 3.0
            raise AssertionError("deprecated root_physx_view used")

    p3 = _Physx3(torch, 2, 3)
    assert compat.physx_view(p3, "x") == "physx3-view"
    p3.data.has_joint_ordering = True
    with pytest.raises(compat.PhysXOnlyError, match="joint_ordering"):
        compat.physx_view(p3, "x")
    lab2 = SimpleNamespace(root_physx_view="physx2-view", data=SimpleNamespace())
    assert compat.physx_view(lab2, "x") == "physx2-view"
