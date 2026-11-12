"""Body mass / inertia / CoM writes go through dropbear_wbc.isaac.compat (Isaac Lab 2.x view API vs 3.0
``set_*_index``). The Isaac Lab 3.0 Newton backend has no ``root_physx_view`` (notes/332)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import sdk_test_paths  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
# PhysX-only diagnostics (probes, inspect extras, telemetry joint loads) take the view from compat.physx_view,
# which raises PhysXOnlyError on Newton (notes/334).
VIEW_ALLOW = {"source/dropbear_wbc/isaac/compat.py"}
_VIEW = re.compile(r"\.root_physx_view\b")


def _code_lines(text: str):
    in_doc = False
    for i, line in enumerate(text.splitlines(), 1):
        q = line.count('"""')
        if in_doc:
            if q % 2 == 1:
                in_doc = False
            continue
        if q % 2 == 1:
            in_doc = True
            continue
        if q >= 2:
            continue
        yield i, line.split("#", 1)[0]


def view_offenders(root: Path = ROOT) -> list[str]:
    out = []
    for sub in ("source/dropbear_wbc", "tools", "scripts"):
        for p in sorted((root / sub).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel in VIEW_ALLOW:
                continue
            for i, code in _code_lines(p.read_text(encoding="utf-8")):
                if _VIEW.search(code):
                    out.append(f"{rel}:{i}")
    return out


def test_no_raw_root_physx_view_outside_compat():
    assert view_offenders() == [], "use compat.get/set_body_masses / _inertias / _com_pos (notes/332)"


def test_gate_catches_planted_view(tmp_path):
    d = tmp_path / "source/dropbear_wbc/tasks"
    d.mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "scripts").mkdir()
    (d / "x.py").write_text('"""doc root_physx_view"""\nm = asset.root_physx_view.get_masses()\n', encoding="utf-8")
    assert view_offenders(tmp_path) == ["source/dropbear_wbc/tasks/x.py:2"]


class _Proxy:
    def __init__(self, t):
        self.torch = t


class _Lab3Asset:
    """Isaac Lab 3.0 shaped asset: coherent .data buffers, partial-data set_*_index, no root_physx_view."""

    def __init__(self, torch, n_env, n_body):
        self.device = "cpu"
        self.m = torch.rand(n_env, n_body) + 0.5
        self.I = torch.rand(n_env, n_body, 9) + 0.1
        self.com = torch.rand(n_env, n_body, 7)
        self.com[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.calls = []
        self._m0, self._I0 = None, None
        outer = self

        class Data:
            @property
            def body_mass(self):
                return _Proxy(outer.m)

            @property
            def body_inertia(self):
                return _Proxy(outer.I)

            @property
            def body_com_pos_b(self):
                return _Proxy(outer.com[..., :3])

            @property
            def body_com_pose_b(self):
                return _Proxy(outer.com)

            @property
            def default_mass(self):  # 3.0: lazy snapshot of the live buffer on first access
                if outer._m0 is None:
                    outer._m0 = outer.m.clone()
                return _Proxy(outer._m0)

            @property
            def default_inertia(self):
                if outer._I0 is None:
                    outer._I0 = outer.I.clone()
                return _Proxy(outer._I0)

        self.data = Data()


    def set_masses_index(self, *, masses, body_ids=None, env_ids=None):
        b = body_ids if body_ids is not None else __import__('torch').arange(self.m.shape[1])
        assert masses.shape == (len(env_ids), self.m[:, b].shape[1])
        self.m[env_ids[:, None], b] = masses
        self.calls.append(("m", masses.shape))

    def set_inertias_index(self, *, inertias, body_ids=None, env_ids=None):
        b = body_ids if body_ids is not None else __import__('torch').arange(self.m.shape[1])
        assert inertias.shape == (len(env_ids), self.I[:, b].shape[1], 9)
        self.I[env_ids[:, None], b] = inertias
        self.calls.append(("I", inertias.shape))

    def set_coms_index(self, *, coms, body_ids=None, env_ids=None):
        b = body_ids if body_ids is not None else __import__('torch').arange(self.m.shape[1])
        k = coms.shape[-1]
        assert coms.shape == (len(env_ids), self.com[:, b].shape[1], k) and k in (3, 7)
        self.com[env_ids[:, None], b, :k] = coms
        self.calls.append(("com", tuple(coms.shape)))


def test_compat_lab3_mass_inertia_partial_writes():
    torch = pytest.importorskip("torch")
    from dropbear_wbc.isaac import compat

    a = _Lab3Asset(torch, 5, 4)
    m = compat.get_body_masses(a)
    assert m.data_ptr() != a.m.data_ptr()
    m[:, 2] = 9.0
    compat.set_body_masses(a, m, torch.tensor([1, 3]), torch.tensor([2]))
    assert a.calls[-1] == ("m", (2, 1))
    assert torch.equal(a.m[[1, 3], 2], torch.tensor([9.0, 9.0]))
    assert (a.m[[0, 2, 4], 2] != 9.0).all()
    I = compat.get_body_inertias(a)
    I[:] = 2.0
    compat.set_body_inertias(a, I, torch.arange(5))
    assert a.calls[-1] == ("I", (5, 4, 9)) and torch.equal(a.I, torch.full((5, 4, 9), 2.0))


def test_compat_com_keeps_orientation_physx3_and_pos_only_newton():
    torch = pytest.importorskip("torch")
    from dropbear_wbc.isaac import compat

    a = _Lab3Asset(torch, 3, 2)
    a.com[..., 3:] = torch.tensor([0.1, 0.2, 0.3, 0.927])
    p = compat.get_body_com_pos(a)
    p[1, 0, 0] += 0.01
    compat.set_body_com_pos(a, p, torch.tensor([1]), torch.tensor([0]))
    assert a.calls[-1] == ("com", (1, 1, 7))  # PhysX 3.0: pose, live quaternion re-sent
    assert torch.allclose(a.com[1, 0, 3:], torch.tensor([0.1, 0.2, 0.3, 0.927]))
    NewtonAsset = type("Articulation", (_Lab3Asset,), {"__module__": "isaaclab_newton.assets.articulation"})
    b = NewtonAsset(torch, 3, 2)
    p = compat.get_body_com_pos(b)
    p[2, 1, 2] += 0.02
    before = b.com.clone()
    compat.set_body_com_pos(b, p, torch.tensor([2]), torch.tensor([1]))
    assert b.calls[-1] == ("com", (1, 1, 3))
    assert torch.allclose(b.com[2, 1, :3], before[2, 1, :3] + torch.tensor([0, 0, 0.02]))


def test_mass_events_on_lab3_asset_scale_from_original_defaults():
    """Offsets then limb DR on a 3.0-shaped asset: defaults are snapshotted before any write, so inertia is
    scaled once (a late first read of default_inertia would capture the already-scaled plant)."""
    torch = pytest.importorskip("torch")
    from dropbear_wbc.robots.hw_motor_specs import motor_mass_offsets
    from dropbear_wbc.robots.mass_dr import body_masses_from_json
    from dropbear_wbc.tasks.mass_events import apply_body_mass_offsets, randomize_limb_mass_scale

    masses = body_masses_from_json()
    names = list(masses)
    a = _Lab3Asset(torch, 4, len(names))
    a.m[:] = torch.tensor([masses[n] for n in names], dtype=torch.float32)
    a.body_names = names
    m0, I0 = a.m.clone(), a.I.clone()
    scene = type("Scene", (), {"num_envs": 4, "__getitem__": lambda self, k: a})()
    env = SimpleNamespace(scene=scene)
    cfg = SimpleNamespace(name="robot")
    off = motor_mass_offsets("hw_cem6")
    apply_body_mass_offsets(env, None, cfg, off)
    from dropbear_wbc.isaac.compat import body_prop_defaults

    defaults = body_prop_defaults(a)  # our snapshot (3.0 data.default_* deprecated, notes/334)
    idx = {n: i for i, n in enumerate(names)}
    for b_, d in off.items():
        j = idx[b_]
        r = (m0[:, j] + d) / m0[:, j]
        assert torch.allclose(a.m[:, j], m0[:, j] + d, atol=1e-5)
        assert torch.allclose(a.I[:, j], I0[:, j] * r[:, None], rtol=1e-5)
        assert torch.allclose(defaults["inertia"][:, j], I0[:, j] * r[:, None], rtol=1e-5)  # not r**2
    torch.manual_seed(0)
    randomize_limb_mass_scale(env, None, cfg)
    ratio = a.m / defaults["mass"]
    assert ((ratio > 0.9 - 1e-6) & (ratio < 1.1 + 1e-6)).all()
    assert torch.allclose(a.I, defaults["inertia"] * ratio[..., None], rtol=1e-5)
    assert all(c[0] in ("m", "I") for c in a.calls)
