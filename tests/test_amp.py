"""CPU tests of the AMP style-reward pieces that do not need Isaac (``dropbear_wbc.tasks.locomotion.amp``).

* the state features are invariant to where the robot stands and which way it faces (yaw), and see the body frame;
* the reference transitions load from the accepted library (every listed clip present, finite, chained in time);
* the discriminator separates two distributions with the least-squares loss used in training.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

torch = pytest.importorskip("torch")

from dropbear_wbc.robots.dropbear_names import MOTOR_NAMES  # noqa: E402
from dropbear_wbc.tasks.locomotion.amp import (  # noqa: E402
    AMP_CLIPS, DEFAULT_LIBRARY, Discriminator, amp_features, load_reference_transitions)

ROOT = sdk_test_paths.ROOT


def _yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _rot_z(yaw: float, v: torch.Tensor) -> torch.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    m = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return v @ m.T


def _sample(n: int = 5, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"dq": r(n, 22), "qd": r(n, 22), "root_pos": r(n, 3) + torch.tensor([0.0, 0.0, 0.9]),
            "root_quat": torch.nn.functional.normalize(r(n, 4), dim=1), "root_lin_vel_w": r(n, 3),
            "root_ang_vel_w": r(n, 3), "feet_pos": r(n, 2, 3) * 0.2, "hands_pos": r(n, 2, 3) * 0.3 + 1.0}


def test_features_shape_and_height():
    s = _sample()
    f = amp_features(**s)
    assert f.shape == (5, 66)
    height = s["root_pos"][:, 2] - s["feet_pos"][:, :, 2].min(dim=1).values
    torch.testing.assert_close(f[:, 44], height)


def test_features_ignore_position_and_heading():
    s = _sample(seed=1)
    yaw, shift = 1.1, torch.tensor([3.0, -2.0, 0.0])
    q_yaw = _yaw_quat(yaw).expand(5, 4)
    w, xyz = q_yaw[:, :1], q_yaw[:, 1:]
    qw, qxyz = s["root_quat"][:, :1], s["root_quat"][:, 1:]  # q_yaw * q
    quat = torch.cat([w * qw - (xyz * qxyz).sum(1, keepdim=True),
                      w * qxyz + qw * xyz + torch.cross(xyz, qxyz, dim=1)], dim=1)
    moved = {**s, "root_quat": quat,
             "root_pos": _rot_z(yaw, s["root_pos"]) + shift,
             "root_lin_vel_w": _rot_z(yaw, s["root_lin_vel_w"]), "root_ang_vel_w": _rot_z(yaw, s["root_ang_vel_w"]),
             "feet_pos": _rot_z(yaw, s["feet_pos"]) + shift, "hands_pos": _rot_z(yaw, s["hands_pos"]) + shift}
    torch.testing.assert_close(amp_features(**moved), amp_features(**s), atol=1e-5, rtol=1e-5)


def test_features_are_in_the_body_frame():
    s = _sample(n=1, seed=2)
    s["root_quat"] = _yaw_quat(math.pi / 2)[None]  # facing +y
    s["root_lin_vel_w"] = torch.tensor([[0.0, 1.0, 0.0]])  # walking forward
    f = amp_features(**s)
    torch.testing.assert_close(f[0, 45:48], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0)  # upright
    torch.testing.assert_close(f[0, 48:51], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0)  # forward = body +x


LIBRARY = ROOT / DEFAULT_LIBRARY


@pytest.mark.skipif(not LIBRARY.exists(), reason="motion library not present")
def test_reference_transitions_load_every_clip():
    default = np.zeros(len(MOTOR_NAMES), dtype=np.float32)
    try:
        ref, report = load_reference_transitions(list(MOTOR_NAMES), default, LIBRARY)
    except FileNotFoundError:
        pytest.skip("motion clips not present (only the manifests are in git)")
    assert sorted(report["clips"]) == sorted(AMP_CLIPS), set(AMP_CLIPS) - set(report["clips"])
    assert ref.shape == (report["transitions"], 132)
    assert torch.isfinite(ref).all()
    # within a clip, transition t ends where transition t+1 starts
    n0 = next(iter(report["clips"].values()))
    torch.testing.assert_close(ref[: n0 - 2, 66:], ref[1 : n0 - 1, :66])
    # root height above the lower foot body: the USD root frame sits below the soles (tasks/locomotion/stand.py), so it
    # is negative; it is the change with crouching / stepping / leaning that carries the style
    h = ref[:, 44]
    assert -0.4 < float(h.median()) < 0.0 and float(h.std()) < 0.2


def test_discriminator_separates_distributions():
    torch.manual_seed(0)
    real, fake = torch.randn(512, 8) + 2.0, torch.randn(512, 8) - 2.0
    d = Discriminator(8, hidden=(32, 32))
    opt = torch.optim.Adam(d.parameters(), lr=1e-2)
    for _ in range(100):
        loss = torch.mean((d(real) - 1.0) ** 2) + torch.mean((d(fake) + 1.0) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    style = lambda x: torch.clamp(1.0 - 0.25 * (d(x) - 1.0) ** 2, min=0.0).mean()  # noqa: E731
    assert float(style(real)) > 0.9 and float(style(fake)) < 0.1
