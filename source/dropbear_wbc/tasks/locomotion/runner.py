"""``DropbearOnPolicyRunner`` + persistence of the velocity-command curriculum across chunked restarts.

``tools/run_chunked_locomotion.py`` restarts Isaac every chunk; the curriculum's widened command ranges live only in
the command term's cfg, so without this every chunk would restart from the initial ranges. :meth:`train_state` adds
``velocity_command_ranges`` to the checkpoint's ``dropbear_train_state``; :meth:`load` restores them (when
``restore_train_state``), together with the learning rate handled by the parent.
"""
from __future__ import annotations

import json

from dropbear_wbc.tasks.tracking.runner import TRAIN_STATE_KEY, DropbearOnPolicyRunner

COMMAND_NAME = "base_velocity"


class LocomotionOnPolicyRunner(DropbearOnPolicyRunner):
    def _command_cfg(self):
        try:
            return self.env.unwrapped.command_manager.get_term(COMMAND_NAME).cfg
        except Exception:  # noqa: BLE001
            return None

    def train_state(self) -> dict:
        state = super().train_state()
        cfg = self._command_cfg()
        if cfg is not None:
            state["velocity_command_ranges"] = {
                k: [float(v) for v in getattr(cfg.ranges, k)] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            }
        return state

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer)
        saved = infos.get(TRAIN_STATE_KEY) if isinstance(infos, dict) else None
        ranges = (saved or {}).get("velocity_command_ranges")
        cfg = self._command_cfg()
        report = self.train_state_report if isinstance(self.train_state_report, dict) else {}
        if ranges and cfg is not None and self.restore_train_state:
            for k, v in ranges.items():
                setattr(cfg.ranges, k, (float(v[0]), float(v[1])))
            report.setdefault("restored", []).append("velocity_command_ranges")
            report["velocity_command_ranges"] = ranges
            print(f"[LocomotionOnPolicyRunner] restored command ranges: {json.dumps(ranges)}", flush=True)
        self.train_state_report = report
        return infos

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # expose the current command ranges in the episode infos -> metrics.jsonl "episode" dict
        cfg = self._command_cfg()
        if cfg is not None and locs.get("ep_infos"):
            locs["ep_infos"][0] = dict(locs["ep_infos"][0])
            locs["ep_infos"][0]["Curriculum/cmd_vx_max"] = float(cfg.ranges.lin_vel_x[1])
            locs["ep_infos"][0]["Curriculum/cmd_vx_min"] = float(cfg.ranges.lin_vel_x[0])
            locs["ep_infos"][0]["Curriculum/cmd_vy_max"] = float(cfg.ranges.lin_vel_y[1])
            locs["ep_infos"][0]["Curriculum/cmd_wz_max"] = float(cfg.ranges.ang_vel_z[1])
        super().log(locs, width, pad)
