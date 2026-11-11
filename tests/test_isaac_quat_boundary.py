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
# tools/ + scripts/ raw reads: 31 at 9663167 (23 + 8); 4 after item 300 (all order-free torch.isfinite checks).
# Lower this as they are ported, never raise it.
OUTSIDE_PKG_BASELINE = 4
# a raw read outside the package is allowed only if the line uses it order-free
ORDER_FREE = re.compile(r"torch\.isfinite\(")
RAW_MATH_IMPORT = re.compile(r"^\s*(from isaaclab\.utils\.math import|import isaaclab\.utils\.math|from isaaclab\.utils import math)")


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


def _outside_files():
    return [p for d in ("tools", "scripts") for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]


def test_tools_scripts_raw_reads_are_order_free():
    bad = [h for h in _hits(_outside_files(), RAW_READ) if not ORDER_FREE.search(h)]
    assert not bad, "quaternion consumer in tools/scripts: use compat.root_quat_wxyz/body_quat_wxyz:\n" + "\n".join(bad)


def test_tools_scripts_use_wxyz_qmath():
    # compat accessors return wxyz; isaaclab.utils.math is xyzw on Lab 3.0, so mixing them is the bug
    hits = _hits(_outside_files(), RAW_MATH_IMPORT)
    assert not hits, "import quaternion math from dropbear_wbc.isaac.qmath:\n" + "\n".join(hits)


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


# --------------------------------------------------------------------------- closure gaps on Lab 3.0

def _quat_apply_wxyz(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


@needs_torch
def test_closure_monitor_needs_wxyz_feed_on_lab3(monkeypatch):
    """Item 300: tools/scripts fed raw body_link_quat_w into ClosureMonitor (wxyz contract). On Lab 3.0
    that is xyzw, so every reported closure gap would be wrong; body_quat_wxyz keeps it exact."""
    from dropbear_wbc.isaac import qmath
    from dropbear_wbc.isaac.closures import Closure, ClosureMonitor

    monkeypatch.setitem(vars(qmath), "quat_apply", _quat_apply_wxyz)
    n, b = 64, 2
    g = torch.Generator().manual_seed(7)
    q = _rand_wxyz(n * b, seed=11).reshape(n, b, 4).float()
    l0, l1 = (0.05, 0.02, -0.03), (-0.04, 0.01, 0.06)
    p0 = torch.randn(n, 3, generator=g)
    p1 = p0 + _quat_apply_wxyz(q[:, 0], torch.tensor(l0).expand(n, 3)) - _quat_apply_wxyz(q[:, 1], torch.tensor(l1).expand(n, 3))
    pos = torch.stack([p0, p1], dim=1)
    mon = ClosureMonitor([Closure("c", "spherical", "a", "b", 0, 1, l0, l1)], "cpu")

    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "xyzw")
    lab3 = _Lab3Asset(torch.roll(q[:, 0], -1, -1), torch.roll(q, -1, -1))
    assert float(mon.worst(pos, compat.body_quat_wxyz(lab3, link=True)).max()) < 1e-6
    raw = lab3.data.body_link_quat_w.torch  # what the unported tools passed
    assert float(mon.worst(pos, raw).max()) > 0.02  # spurious gap of the order of the 5-7 cm anchors


# --------------------------------------------------------------------------- item 330: aliases + ProxyArray
# ``d = robot.data; d.root_quat_w`` evaded the regex gate above (amp.py, quasistatic.py, play.py at 0b98f9a).
# Isaac Lab 3.0 ``.data`` returns ProxyArray; raw use works only via its deprecation bridge (warns, to be removed).

import ast  # noqa: E402

QUAT_ATTR = re.compile(r"quat|_state_w$|_pose_w$|^default_root_state$")
NON_ISAAC_DATA = {"output", "xpos"}  # camera output dict / MuJoCo MjData


def _own_nodes(scope):
    """Nodes of ``scope`` excluding nested function bodies."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(n))


def _data_aliases(scope):
    out = set()
    for n in _own_nodes(scope):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Attribute) and n.value.attr == "data":
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


def _alias_scopes(tree):
    """id(node) -> aliases bound in its own function (``d = robot.data`` is function-local)."""
    m = {}
    for f in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        al = _data_aliases(f)
        for n in _own_nodes(f):
            m[id(n)] = al
    return m


def _raw_data_loads(src, *, quat):
    """(line, attr) of `.data.X` / `<alias>.X` loads; quat=True -> quaternion fields, False -> others not
    wrapped in as_tensor(...)."""
    tree = ast.parse(src)
    scopes = _alias_scopes(tree)
    wrapped = {id(c.args[0]) for c in ast.walk(tree) if isinstance(c, ast.Call) and c.args
               and getattr(c.func, "id", getattr(c.func, "attr", None)) == "as_tensor"}
    hits = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)):
            continue
        v = n.value
        via = (isinstance(v, ast.Attribute) and v.attr == "data") or (isinstance(v, ast.Name) and v.id in scopes.get(id(n), ()))
        if not via or n.attr in NON_ISAAC_DATA or n.attr.startswith("_"):
            continue
        if bool(QUAT_ATTR.search(n.attr)) != quat:
            continue
        if not quat and id(n) in wrapped:
            continue
        hits.append((n.lineno, n.attr))
    return hits


def _scan(paths, *, quat, alias_only=False):
    out = []
    for p in paths:
        src = p.read_text(encoding="utf-8")
        for ln, a in _raw_data_loads(src, quat=quat):
            line = src.splitlines()[ln - 1]
            if alias_only and re.search(rf"\.data\.{a}\b", line):
                continue
            out.append(f"{p.relative_to(ROOT)}:{ln}: {line.strip()}")
    return out


def test_no_alias_quat_read_anywhere():
    hits = _scan(_pkg_files() + _outside_files(), quat=True, alias_only=True)
    assert not hits, "`d = x.data; d.*quat*` bypasses compat; use root_quat_wxyz/body_quat_wxyz:\n" + "\n".join(hits)


def test_tools_scripts_wrap_isaac_data_in_as_tensor():
    hits = _scan(_outside_files(), quat=False)
    assert not hits, "wrap Isaac .data reads in compat.as_tensor(...) (Lab 3.0 ProxyArray):\n" + "\n".join(hits)


def test_ast_gate_catches_planted_reads():
    planted = ("def f(robot):\n    d = robot.data\n    return d.root_quat_w\n")
    assert _raw_data_loads(planted, quat=True) == [(3, "root_quat_w")]
    assert _raw_data_loads("x = robot.data.joint_pos[:, 0]\n", quat=False) == [(1, "joint_pos")]
    assert _raw_data_loads("x = as_tensor(robot.data.joint_pos)[:, 0]\n", quat=False) == []
    assert _raw_data_loads("robot.data.joint_pos = y\n", quat=False) == []
    assert _raw_data_loads("im = cam.data.output['rgb']\n", quat=False) == []


class _Proxy330:
    """Isaac Lab 3.0 ProxyArray stand-in: `.torch` view + indexing bridge (values as stored, xyzw for quats)."""

    def __init__(self, t):
        self._t = t

    @property
    def torch(self):
        return self._t

    def __getitem__(self, k):
        return self._t[k]


class _Data330:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, _Proxy330(v))


class _Asset330:
    def __init__(self, data):
        self.data = data


def _xyzw(q):
    return q[..., [1, 2, 3, 0]]


@needs_torch
def test_amp_state_is_order_invariant(monkeypatch):
    from dropbear_wbc.tasks.locomotion import amp

    n, j, b = 5, 7, 6
    g = torch.Generator().manual_seed(330)
    rq = _rand_wxyz(n, seed=330).float()
    kw = dict(joint_pos=torch.randn(n, j, generator=g), joint_vel=torch.randn(n, j, generator=g),
              root_pos_w=torch.randn(n, 3, generator=g), root_lin_vel_w=torch.randn(n, 3, generator=g),
              root_ang_vel_w=torch.randn(n, 3, generator=g), body_link_pos_w=torch.randn(n, b, 3, generator=g))
    term = object.__new__(amp.AmpStyleReward)
    term._motor_ids, term._feet, term._hands = [0, 2, 4], [1, 3], [2, 5]
    term._default = torch.zeros(3)

    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "wxyz")
    term._robot = _Asset330(_Data330(root_quat_w=rq, **kw))
    ref = term._state()
    monkeypatch.setenv("DROPBEAR_ISAAC_QUAT_ORDER", "xyzw")
    term._robot = _Asset330(_Data330(root_quat_w=_xyzw(rq), **kw))
    got = term._state()
    assert torch.allclose(ref, got, atol=1e-6), float((ref - got).abs().max())


# ---------------------------------------------------------------- item 331: package-wide ProxyArray wraps, gravity
NON_ISAAC_PKG = {"teleop/gravity.py", "kinematics/serial_model.py"}  # JSON dict ``self.data`` / MuJoCo MjData


def _isaac_pkg_files():
    return [p for p in _pkg_files() if p.relative_to(PKG).as_posix() not in NON_ISAAC_PKG]


def test_package_wraps_isaac_data_in_as_tensor():
    hits = _scan(_isaac_pkg_files(), quat=False)
    assert not hits, "wrap Isaac .data reads in compat.as_tensor(...) (Lab 3.0 ProxyArray):\n" + "\n".join(hits)


def test_no_raw_gravity_vec_outside_compat():
    """Newton (Lab 3.0) binds GRAVITY_VEC_W to model.gravity in m/s^2; only compat.gravity_dir_w may read it."""
    hits = _hits(_pkg_files() + _outside_files(), re.compile(r"\bGRAVITY_VEC_W\b"))
    assert not hits, "use compat.gravity_dir_w(asset) (unit) instead of raw GRAVITY_VEC_W:\n" + "\n".join(hits)


def test_camera_output_wrapped():
    """Lab 3.0 CameraData.output is dict[str, ProxyArray]; index the key, then as_tensor."""
    rx = re.compile(r"\.data\.output\[")
    bad = [h for h in _hits(_pkg_files() + _outside_files(), rx) if not re.search(r"as_tensor\([^)]*\.data\.output\[", h)]
    assert not bad, "wrap camera .data.output[key] in as_tensor(...):\n" + "\n".join(bad)


def _cached_data_views(src):
    """``self.x = <...>.data.y`` / ``self.x = as_tensor(<...>.data.y)`` (optionally indexed) stored without a copy.
    On Newton the ProxyArray (and its .torch view) is replaced on full reset; a stored view goes stale."""
    tree = ast.parse(src)
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                                  and t.value.id == "self" for t in n.targets)):
            continue
        v = n.value
        while isinstance(v, ast.Subscript):
            v = v.value
        if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "as_tensor" and v.args:
            v = v.args[0]
            while isinstance(v, ast.Subscript):
                v = v.value
        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Attribute) and v.value.attr == "data":
            out.append(n.lineno)
    return out


def test_no_data_view_cached_on_self():
    hits = []
    for p in _isaac_pkg_files() + _outside_files():
        src = p.read_text(encoding="utf-8")
        hits += [f"{p.relative_to(ROOT)}:{ln}" for ln in _cached_data_views(src)]
    assert not hits, "copy (.clone()) Isaac .data views before storing them on self (Newton reset):\n" + "\n".join(hits)


def test_cache_gate_catches_planted():
    assert _cached_data_views("class A:\n    def f(s):\n        self.q = self.robot.data.joint_pos\n") == [3]
    assert _cached_data_views("self.q = as_tensor(r.data.joint_pos)[:, ids]\n") == [1]
    assert _cached_data_views("self.q = as_tensor(r.data.joint_pos)[:, ids].clone()\n") == []
    assert _cached_data_views("q = r.data.joint_pos\n") == []


class _GravData:
    def __init__(self, g):
        self.GRAVITY_VEC_W = _Proxy330(g)


class _GravAsset:
    def __init__(self, g):
        self.data = _GravData(g)


@needs_torch
def test_gravity_dir_w_normalizes_newton_magnitude():
    g = torch.tensor([[0.0, 0.0, -9.81], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -9.81]])
    d = compat.gravity_dir_w(_GravAsset(g))
    assert torch.allclose(d[:2], torch.tensor([[0.0, 0.0, -1.0]] * 2))
    assert torch.equal(d[2], torch.zeros(3))  # gravity disabled stays zero
    assert torch.allclose(d.norm(dim=-1)[[0, 1, 3]], torch.ones(3))


def test_gravity_dir_w_numpy():
    class _A:
        class data:  # noqa: N801
            GRAVITY_VEC_W = np.array([[0.0, 0.0, -9.81]])
    assert np.allclose(compat.gravity_dir_w(_A), [[0.0, 0.0, -1.0]])
