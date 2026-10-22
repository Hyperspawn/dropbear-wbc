"""Joint-position action with the semantic ankle-roll clamp (``kinematics.ankle_bind``) after the per-motor clip.

The pinned tie-rod ankle binds at about +-8 deg of roll; the per-motor clamp box cannot express that limit (it is
diagonal in calf-motor space). This term applies the tabulated :class:`AnkleRollBind` correction to the processed
targets, so the policy trains on the same target law the deploy FSM applies (sidecar ``ankle_roll_bind``).
"""
from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from dropbear_wbc.kinematics.ankle_bind import AnkleRollBind, apply_lut_torch, lut_tables_torch


class AnkleBindJointPositionAction(JointPositionAction):
    cfg: "AnkleBindJointPositionActionCfg"

    def __init__(self, cfg: "AnkleBindJointPositionActionCfg", env):
        super().__init__(cfg, env)
        bind = AnkleRollBind(roll_limit_deg=cfg.ankle_roll_limit_deg, margin_deg=cfg.ankle_roll_margin_deg)
        luts = bind.lut()
        names = list(self._joint_names)
        missing = [m for L in luts for m in L.motors if m not in names]
        if missing:
            raise ValueError(f"ankle_roll_bind: action term has no {missing}")
        self._ankle_cols = [(names.index(L.motors[0]), names.index(L.motors[1])) for L in luts]
        self._ankle_tabs = lut_tables_torch(luts, self.device, torch.float32)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = apply_lut_torch(self._processed_actions, None, self._ankle_cols, self._ankle_tabs)


@configclass
class AnkleBindJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = AnkleBindJointPositionAction
    ankle_roll_limit_deg: float = 8.0
    ankle_roll_margin_deg: float = 1.0
