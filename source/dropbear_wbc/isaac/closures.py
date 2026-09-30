"""Loop-closure residuals of the Dropbear articulation (anchor gaps of excluded joints).

The USD has 27 joints with ``physics:excludeFromArticulation = true`` that PhysX solves as
maximal-coordinate constraints (the parallel-mechanism closures). The residual of one closure is the
distance [m] between its two joint anchors expressed in world frame::

    gap = | (p0 + R0 * localPos0) - (p1 + R1 * localPos1) |

where (p, R) are the *link frame* poses of body0/body1 from the simulation. Only positional gaps are
measured (revolute/spherical closures dominate; fixed closures would also have an angular part).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Closure:
    """One excluded (loop-closing) joint."""

    name: str
    joint_type: str
    body0: str
    body1: str
    body0_index: int
    body1_index: int
    local_pos0: tuple[float, float, float]
    local_pos1: tuple[float, float, float]


def find_closures(robot_prim_path: str, body_names: list[str]) -> list[Closure]:
    """Read excluded joints of the robot at ``robot_prim_path`` from the live USD stage.

    Args:
        robot_prim_path: e.g. ``/World/envs/env_0/Robot``.
        body_names: articulation body names (Isaac order) used to map joint targets to indices.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root.IsValid():
        raise ValueError(f"robot prim {robot_prim_path} not found")
    index = {name: i for i, name in enumerate(body_names)}
    closures: list[Closure] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        excluded = joint.GetExcludeFromArticulationAttr().Get()
        enabled = joint.GetJointEnabledAttr().Get()
        if not excluded or enabled is False:
            continue
        t0, t1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(t0) != 1 or len(t1) != 1:
            continue  # e.g. the 'debugging_joint' to the world
        b0, b1 = t0[0].name, t1[0].name
        if b0 not in index or b1 not in index:
            continue
        lp0 = joint.GetLocalPos0Attr().Get()
        lp1 = joint.GetLocalPos1Attr().Get()
        closures.append(
            Closure(
                name=prim.GetName(),
                joint_type=prim.GetTypeName(),
                body0=b0,
                body1=b1,
                body0_index=index[b0],
                body1_index=index[b1],
                local_pos0=tuple(float(v) for v in lp0),
                local_pos1=tuple(float(v) for v in lp1),
            )
        )
    return closures


class ClosureMonitor:
    """Vectorised closure-gap evaluation for (num_envs, num_bodies) body pose tensors."""

    def __init__(self, closures: list[Closure], device: str | torch.device):
        if not closures:
            raise ValueError("no closures found")
        self.closures = closures
        self.names = [c.name for c in closures]
        self.i0 = torch.tensor([c.body0_index for c in closures], dtype=torch.long, device=device)
        self.i1 = torch.tensor([c.body1_index for c in closures], dtype=torch.long, device=device)
        self.l0 = torch.tensor([c.local_pos0 for c in closures], dtype=torch.float32, device=device)
        self.l1 = torch.tensor([c.local_pos1 for c in closures], dtype=torch.float32, device=device)

    def gaps(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
        """Anchor gaps [m], shape (num_envs, num_closures). Quaternions are wxyz."""
        from isaaclab.utils.math import quat_apply

        n = body_pos_w.shape[0]
        c = self.i0.numel()
        q0 = body_quat_w[:, self.i0].reshape(-1, 4)
        q1 = body_quat_w[:, self.i1].reshape(-1, 4)
        a0 = body_pos_w[:, self.i0] + quat_apply(q0, self.l0.expand(n, c, 3).reshape(-1, 3)).reshape(n, c, 3)
        a1 = body_pos_w[:, self.i1] + quat_apply(q1, self.l1.expand(n, c, 3).reshape(-1, 3)).reshape(n, c, 3)
        return torch.linalg.vector_norm(a0 - a1, dim=-1)

    def worst(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
        """Worst gap per env [m], shape (num_envs,)."""
        return self.gaps(body_pos_w, body_quat_w).max(dim=-1).values
