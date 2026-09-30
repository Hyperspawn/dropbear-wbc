"""CPU tests for the CONTRACTS 0.2 adoption (spherical ankle tie-rod closures) and related tooling.

* ``dropbear_names.authored_ankle_requested`` / ``spherical_joint_fixes`` (flag beats env; env ``1`` opts out);
* the Newton in-memory fix ``newton_sim.usd_fixes.apply_contract_fixes`` on a synthetic USD stage (pxr only): the
  four closures are retyped, a tree joint with the same name is refused, the opt-out keeps them revolute;
* ``motion_npz.validate_provenance(expected_authored_ankle=...)`` fails closed on a variant mismatch, and NPZs
  without the key count as authored;
* ``tools/gpu_lock_run.py`` releases the lock when the child exits although a detached grandchild still holds the
  output pipe (the Omniverse ``hub.exe`` incident, logs/gpu_pipeline/PROGRESS.md).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots import dropbear_names as N  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import (  # noqa: E402
    MotionArrays,
    MotionFormatError,
    npz_authored_ankle,
    validate_provenance,
)


def test_flag_and_env(monkeypatch):
    monkeypatch.delenv(N.AUTHORED_ANKLE_ENV, raising=False)
    assert N.authored_ankle_requested() is False
    assert N.spherical_joint_fixes() == N.SPHERICAL_JOINT_FIXES
    assert set(N.SPHERICAL_JOINT_FIXES) == {"LL_Revolute111", "LL_Revolute112", "RL_Revolute111", "RL_Revolute112"}
    monkeypatch.setenv(N.AUTHORED_ANKLE_ENV, "1")
    assert N.authored_ankle_requested() is True
    assert N.spherical_joint_fixes() == ()
    assert N.spherical_joint_fixes(False) == N.SPHERICAL_JOINT_FIXES  # explicit flag beats the env
    monkeypatch.setenv(N.AUTHORED_ANKLE_ENV, "0")
    assert N.authored_ankle_requested() is False
    assert N.authored_ankle_requested(True) is True


def _need_usd() -> None:
    """Skip unless a working USD (pxr with Usd/UsdPhysics/Gf) is importable (system Python has a partial pxr)."""
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: F401
    except ImportError as e:
        pytest.skip(f"no working pxr: {e}")


def _stage(tree_joint_named_111: bool = False):
    """Minimal stage: /humanoid with bodies world, a, b + 4 excluded revolute closures + one tree joint."""
    _need_usd()
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/humanoid")
    bodies = ["world", "a", "b"] + list(N.ORPHAN_BODIES)
    for b in bodies:
        x = UsdGeom.Xform.Define(stage, f"/humanoid/{b}")
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
    tree = UsdPhysics.RevoluteJoint.Define(stage, "/humanoid/tree_a")
    tree.CreateBody0Rel().SetTargets(["/humanoid/world"])
    tree.CreateBody1Rel().SetTargets(["/humanoid/a"])
    tree.CreateAxisAttr("Z")
    for i, name in enumerate(N.SPHERICAL_JOINT_FIXES):
        path = f"/humanoid/{name}" if not (tree_joint_named_111 and i == 0) else f"/humanoid/{name}"
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets(["/humanoid/a"])
        j.CreateBody1Rel().SetTargets(["/humanoid/b"])
        j.CreateAxisAttr("Y")
        j.CreateExcludeFromArticulationAttr(not (tree_joint_named_111 and i == 0))
    # the other 0.1 fixes expect these joints to exist
    for name in N.JOINT_AXIS_FIXES:
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/humanoid/{name}")
        j.CreateBody0Rel().SetTargets(["/humanoid/a"])
        j.CreateBody1Rel().SetTargets(["/humanoid/b"])
        j.CreateAxisAttr("X")
        j.CreateExcludeFromArticulationAttr(True)
    return stage


def _types(stage):
    from pxr import Usd, UsdPhysics

    return {p.GetName(): p.GetTypeName() for p in Usd.PrimRange(stage.GetPrimAtPath("/humanoid"))
            if p.IsA(UsdPhysics.Joint)}


def test_newton_usd_fixes_retype(monkeypatch):
    _need_usd()
    from dropbear_wbc.newton_sim import usd_fixes

    monkeypatch.setattr(usd_fixes, "find_orphan_bodies",
                        lambda stage, root: [{"path": f"/humanoid/{b}", "mass_kg": 0.018} for b in N.ORPHAN_BODIES])
    monkeypatch.delenv(N.AUTHORED_ANKLE_ENV, raising=False)
    stage = _stage()
    changed = usd_fixes.apply_contract_fixes(stage, "/humanoid", "/humanoid/world")
    t = _types(stage)
    assert all(t[n] == "PhysicsSphericalJoint" for n in N.SPHERICAL_JOINT_FIXES)
    assert t["tree_a"] == "PhysicsRevoluteJoint"
    assert changed["authored_ankle_tierods"] is False and len(changed["spherical"]) == 4

    stage = _stage()
    changed = usd_fixes.apply_contract_fixes(stage, "/humanoid", "/humanoid/world", authored_ankle_tierods=True)
    assert all(_types(stage)[n] == "PhysicsRevoluteJoint" for n in N.SPHERICAL_JOINT_FIXES)
    assert changed["spherical"] == [] and changed["authored_ankle_tierods"] is True

    monkeypatch.setenv(N.AUTHORED_ANKLE_ENV, "1")
    stage = _stage()
    assert usd_fixes.apply_contract_fixes(stage, "/humanoid", "/humanoid/world")["spherical"] == []


def test_newton_usd_fixes_refuses_tree_joint(monkeypatch):
    _need_usd()
    from dropbear_wbc.newton_sim import usd_fixes

    monkeypatch.setattr(usd_fixes, "find_orphan_bodies",
                        lambda stage, root: [{"path": f"/humanoid/{b}", "mass_kg": 0.018} for b in N.ORPHAN_BODIES])
    monkeypatch.delenv(N.AUTHORED_ANKLE_ENV, raising=False)
    with pytest.raises(ValueError, match="not a loop closure"):
        usd_fixes.apply_contract_fixes(_stage(tree_joint_named_111=True), "/humanoid", "/humanoid/world")


def _arrays(meta: dict) -> MotionArrays:
    t, names = 3, list(N.MOTOR_NAMES)
    q = np.zeros((t, 1, 4), np.float32)
    q[..., 0] = 1
    return MotionArrays(fps=50.0, joint_pos=np.zeros((t, 22), np.float32), joint_vel=np.zeros((t, 22), np.float32),
                        body_pos_w=np.zeros((t, 1, 3), np.float32), body_quat_w=q,
                        body_lin_vel_w=np.zeros((t, 1, 3), np.float32), body_ang_vel_w=np.zeros((t, 1, 3), np.float32),
                        joint_names=names, body_names=["world"], motor_names=names,
                        closure_residual_m=np.zeros(t, np.float32), meta=meta)


def test_npz_ankle_variant_fail_closed():
    legacy = _arrays({"schema": "dropbear-motion-npz-v1"})
    spherical = _arrays({"schema": "dropbear-motion-npz-v1", "authored_ankle_tierods": False})
    authored = _arrays({"schema": "dropbear-motion-npz-v1", "authored_ankle_tierods": True})
    assert npz_authored_ankle(legacy.meta) is True  # predates the 0.2 adoption
    validate_provenance(spherical, expected_authored_ankle=False)
    validate_provenance(authored, expected_authored_ankle=True)
    validate_provenance(legacy)  # not checked unless requested
    for m, exp in ((legacy, False), (authored, False), (spherical, True)):
        with pytest.raises(MotionFormatError, match="ankle tie rods"):
            validate_provenance(m, expected_authored_ankle=exp)


def test_gpu_lock_run_releases_with_pipe_holding_grandchild(tmp_path):
    lock = REPO / ".locks" / "gpu.lock"
    if lock.exists():
        pytest.skip("GPU lock held by someone else; not touching it")
    log = tmp_path / "wrap.log"
    child = ("import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
             "print('child done', flush=True)")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(REPO / "tools" / "gpu_lock_run.py"), "--owner", "pytest_gpu_lock_run",
                        "--log", str(log), "--timeout", "60", "--drain-s", "1", "--wait-minutes", "0.1", "--",
                        sys.executable, "-c", child], capture_output=True, text=True, timeout=60)
    dt = time.time() - t0
    if r.returncode == 125:
        pytest.skip("GPU lock became busy during the test")
    assert r.returncode == 0, r.stdout + r.stderr
    assert dt < 15, dt
    text = log.read_text(encoding="utf-8")
    assert "child done" in text and "end=" in text and "not waiting" in text
    assert not lock.exists() or "pytest_gpu_lock_run" not in lock.read_text(encoding="utf-8")
