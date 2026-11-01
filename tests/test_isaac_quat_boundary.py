"""Isaac quaternion-order boundary (dropbear_wbc.isaac.compat): Lab 2.x wxyz vs Lab 3.0 xyzw.

Gate: no raw quaternion-bearing ``.data.*`` read and no raw root ``write_*_to_sim(`` call may appear in
``source/dropbear_wbc`` outside ``isaac/compat.py``; tools/ and scripts/ may not grow past the baseline.
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest

try:
    import torch
except ImportError:  # gate tests still run without torch
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "source"))

from dropbear_wbc.isaac import compat  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "source" / "dropbear_wbc"
COMPAT = PKG / "isaac" / "compat.py"

# quaternion-bearing Isaac reads (notes/15 scan list, Lab 2.2 + 3.0 names)
RAW_READ = re.compile(
    r"\.data\.(root_quat_w|body_quat_w|root_link_quat_w|body_link_quat_w|root_com_quat_w|body_com_quat_w"
    r"|root_state_w|root_pose_w|body_state_w|body_pose_w|default_root_state|root_link_state_w|root_link_pose_w"
    r"|body_link_state_w|body_link_pose_w|root_com_state_w|root_com_pose_w|body_com_state_w|body_com_pose_w)\b"
)
RAW_WRITE = re.compile(r"\.write_root_(state|pose|link_pose|link_state|com_pose|com_state)_to_sim(_index|_mask)?\(")
# tools/ + scripts/ raw reads at 9663167 (23 + 8); lower this as they are ported, never raise it
OUTSIDE_PKG_BASELINE = 31


def _hits(paths, rx):
    out = []
    for p in paths:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                out.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    return out


def _pkg_files():
    return [p for p in PKG.rglob("*.py") if p != COMPAT and "__pycache__" not in p.parts]


def test_no_raw_isaac_quat_read_in_package():
    hits = _hits(_pkg_files(), RAW_READ)
    assert not hits, "route through dropbear_wbc.isaac.compat.root_quat_wxyz/body_quat_wxyz:\n" + "\n".join(hits)


def test_no_raw_root_write_in_package():
    hits = _hits(_pkg_files(), RAW_WRITE)
    assert not hits, "route through dropbear_wbc.isaac.compat.write_root_state/write_root_pose:\n" + "\n".join(hits)


def test_tools_scripts_raw_reads_do_not_grow():
    files = [p for d in ("tools", "scripts") for p in (ROOT / d).rglob("*.py")]
    n = len(_hits(files, RAW_READ))
    assert n <= OUTSIDE_PKG_BASELINE, f"{n} raw Isaac quat reads in tools/+scripts/ (baseline {OUTSIDE_PKG_BASELINE})"


def test_gate_regex_catches_planted_reads():
    for line in ("q = robot.data.body_link_quat_w[:, 0]", "s = a.data.root_state_w", "a.data.default_root_state[ids]"):
        assert RAW_READ.search(line), line
    assert RAW_WRITE.search("robot.write_root_state_to_sim(x, env_ids=ids)")
    assert RAW_WRITE.search("robot.write_root_pose_to_sim_index(root_pose=x)")
    assert not RAW_READ.search("q = body_quat_wxyz(robot, link=True)")
    assert not RAW_WRITE.search("``write_root_state_to_sim`` expects")


# --------------------------------------------------------------------------- order conversion

def _rand_wxyz(n, seed=0):
    q = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_order_detection_and_override(monkeypatch):
    monkeypatch.delenv("DROPBEAR_ISAAC_QUAT_ORDER", raising=False)
    compat._detected_order.cache_clear()
    monkeypatch.setattr(compat, "_isaaclab_major", lambda: None)
    assert compat.isaac_quat_order() == "wxyz"
    compat._detected_order.cache_clear()
    monkeypatch.setattr(compat, "_isaaclab_major", lambda: 2)
    assert compat.isaac_quat_order() == "wxyz"
    compat._detected_order.cache_clear()
    monkeypatch.setattr(compat, "_isaaclab_major", lambda: 3)
    assert compat.isaac_quat_order() == "xyzw"
    compat._detected_order.cache_clear()
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "wxyz")
    assert compat.isaac_quat_order() == "wxyz"
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "zyx")
    with pytest.raises(ValueError):
        compat.isaac_quat_order()


@needs_torch
def test_lab2_is_zero_copy_identity(monkeypatch):
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "wxyz")
    q = _rand_wxyz(8)
    assert compat.wxyz_from_isaac(q) is q
    assert compat.isaac_from_wxyz(q) is q


@needs_torch
def test_lab3_reorder_known_rotation(monkeypatch):
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "xyzw")
    s = 0.5 ** 0.5
    z90_wxyz = torch.tensor([[s, 0.0, 0.0, s]])
    z90_xyzw = torch.tensor([[0.0, 0.0, s, s]])
    assert torch.equal(compat.isaac_from_wxyz(z90_wxyz), z90_xyzw)
    assert torch.equal(compat.wxyz_from_isaac(z90_xyzw), z90_wxyz)
    q = _rand_wxyz(1000, seed=1).reshape(10, 100, 4)
    assert torch.equal(compat.wxyz_from_isaac(compat.isaac_from_wxyz(q)), q)
    qn = q.numpy()
    assert np.array_equal(compat.wxyz_from_isaac(compat.isaac_from_wxyz(qn)), qn)
    with pytest.raises(ValueError):
        compat.wxyz_from_isaac(torch.zeros(3, 3))


# --------------------------------------------------------------------------- fake assets

class _Proxy:  # Lab 3.0 ProxyArray: zero-copy .torch property
    def __init__(self, t):
        self._t = t

    @property
    def torch(self):
        return self._t


class _Data:
    def __init__(self, root_q, body_q, proxy):
        w = _Proxy if proxy else (lambda t: t)
        self.root_quat_w = w(root_q)
        self.root_link_quat_w = w(root_q + 0)
        self.body_quat_w = w(body_q)
        self.body_link_quat_w = w(body_q + 0)


class _Lab2Asset:
    def __init__(self, root_q, body_q):
        self.data = _Data(root_q, body_q, proxy=False)
        self.calls = []

    def write_root_state_to_sim(self, state, env_ids=None):
        self.calls.append(("state", state.clone(), env_ids))

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.calls.append(("pose", pose.clone(), env_ids))


class _Lab3Asset:
    def __init__(self, root_q, body_q):
        self.data = _Data(root_q, body_q, proxy=True)
        self.calls = []

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None, skip_forward=False):
        self.calls.append(("pose_index", root_pose.clone(), env_ids))

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None, skip_forward=False):
        self.calls.append(("vel_index", root_velocity.clone(), env_ids))


@needs_torch
@pytest.mark.parametrize("lab", [2, 3])
def test_accessors_return_wxyz_on_both_versions(monkeypatch, lab):
    n, b = 4, 6
    root_wxyz = _rand_wxyz(n, seed=2)
    body_wxyz = _rand_wxyz(n * b, seed=3).reshape(n, b, 4)
    if lab == 2:
        monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "wxyz")
        asset = _Lab2Asset(root_wxyz, body_wxyz)
    else:
        monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "xyzw")
        asset = _Lab3Asset(torch.roll(root_wxyz, -1, -1), torch.roll(body_wxyz, -1, -1))
    assert torch.equal(compat.root_quat_wxyz(asset), root_wxyz)
    assert torch.equal(compat.root_quat_wxyz(asset, link=True), root_wxyz)
    assert torch.equal(compat.body_quat_wxyz(asset), body_wxyz)
    assert torch.equal(compat.body_quat_wxyz(asset, 2, link=True), body_wxyz[:, 2])
    ids = torch.tensor([0, 3, 5])
    assert torch.equal(compat.body_quat_wxyz(asset, ids), body_wxyz[:, ids])


@needs_torch
@pytest.mark.parametrize("lab", [2, 3])
def test_root_writes(monkeypatch, lab):
    n = 3
    pos, lin, ang = torch.randn(n, 3), torch.randn(n, 3), torch.randn(n, 3)
    q = _rand_wxyz(n, seed=4).float()
    ids = torch.tensor([0, 2, 5])
    if lab == 2:
        monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "wxyz")
        a = _Lab2Asset(q, q[:, None])
        compat.write_root_state(a, pos, q, lin, ang, env_ids=ids)
        compat.write_root_pose(a, pos, q, env_ids=ids)
        (k0, s, e0), (k1, p, e1) = a.calls
        assert (k0, k1) == ("state", "pose") and e0 is ids and e1 is ids
        assert torch.equal(s, torch.cat([pos, q, lin, ang], -1))  # byte-identical to the pre-shim call
        assert torch.equal(p, torch.cat([pos, q], -1))
    else:
        monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "xyzw")
        a = _Lab3Asset(q, q[:, None])
        compat.write_root_state(a, pos, q, lin, ang, env_ids=ids)
        compat.write_root_pose(a, pos, q, env_ids=ids)
        (k0, p0, _), (k1, v, _), (k2, p1, _) = a.calls
        assert (k0, k1, k2) == ("pose_index", "vel_index", "pose_index")
        xyzw = torch.roll(q, -1, -1)
        assert torch.equal(p0, torch.cat([pos, xyzw], -1)) and torch.equal(p1, p0)
        assert torch.equal(v, torch.cat([lin, ang], -1))


def test_import_does_not_pull_isaaclab():
    import subprocess
    import sys

    code = "import sys, dropbear_wbc.isaac.compat as c; c.isaac_quat_order(); print('isaaclab' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__('os').environ, "PYTHONPATH": str(ROOT / "source")})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
