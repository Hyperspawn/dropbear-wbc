"""Settled standing reset state for the velocity task (numpy only; no Isaac imports).

The velocity task resets every episode from a physically settled, closure-consistent standing state recorded in a
contract motion NPZ (``dropbear-motion-npz-v1``): the FULL joint row (22 motors + 63 passive DOFs + 6 neck screws)
plus the root (``world``) state. This replaces the airborne CAD reset of every earlier Dropbear walking attempt
(dropbear-research AGENTS.md rule 32: the CAD reset falls for ~0.16 s before touchdown).

Default clip: ``data/motions/smoke/dropbear_static_stand.npz`` (``tools/make_static_npz.py``: 200 identical frames at
the calibration standing pose, settled on the contract plant incl. the spherical ankle tie rods; closure 0.18 mm,
anchor z 1.498 m, validator verdict ``accepted``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropbear_wbc.robots.dropbear_names import ANCHOR_BODY, FOOT_EE_BODIES, REPO_ROOT, ROOT_BODY

DEFAULT_STAND_NPZ: Path = REPO_ROOT / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"


@dataclass(frozen=True)
class StandSummary:
    """Scalar facts of a standing NPZ that the env config needs before the sim exists.

    Attributes:
        path: absolute NPZ path.
        anchor_z: mean anchor (chest) body height over the frames [m] -- the reset-calibrated, loaded torso height.
        root_z: mean root (``world``) frame height [m] (negative: the frame origin sits below the soles).
        sole_z: mean height of the two sole-plate link frames [m] (link origins, not contact points).
        max_joint_speed: max |joint_vel| over all frames and joints [rad/s, m/s] (0 for a static clip).
        closure_max_m: worst loop-closure residual of the clip [m].
        num_frames: T.
    """

    path: str
    anchor_z: float
    root_z: float
    sole_z: float
    max_joint_speed: float
    closure_max_m: float
    num_frames: int


def summarize_stand_npz(path: str | Path = DEFAULT_STAND_NPZ) -> StandSummary:
    """Read the scalars the config needs (fails closed on a malformed NPZ via ``load_motion_npz``)."""
    from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz

    p = Path(path).resolve()
    m = load_motion_npz(p)
    bn = list(m.body_names)
    anchor = bn.index(ANCHOR_BODY)
    root = bn.index(ROOT_BODY)
    soles = [bn.index(n) for n in FOOT_EE_BODIES]
    return StandSummary(
        path=str(p),
        anchor_z=float(m.body_pos_w[:, anchor, 2].mean()),
        root_z=float(m.body_pos_w[:, root, 2].mean()),
        sole_z=float(m.body_pos_w[:, soles, 2].mean()),
        max_joint_speed=float(np.abs(m.joint_vel).max()),
        closure_max_m=float(m.closure_residual_m.max()),
        num_frames=m.num_frames,
    )
