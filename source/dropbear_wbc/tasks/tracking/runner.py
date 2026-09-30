"""RSL-RL 2.3.3 ``OnPolicyRunner`` that also appends one JSON line of training metrics per iteration and carries the
training state that rsl-rl does not checkpoint across chunked restarts.

``<log_dir>/metrics.jsonl`` fields: it, mean_reward, mean_episode_length, losses{...}, action_std,
steps_per_s, collection_s, learn_s, finite (all numbers finite), episode{reward-term means}.

Train state (review finding 2026-09-24, ``tools/run_chunked_training.py`` restarts Isaac every chunk): rsl-rl 2.3.3
saves neither ``alg.learning_rate`` (the adaptive-KL learning rate restarts at the config value, 1e-3) nor the motion
command's adaptive-sampling statistics (``MotionCommand.bin_failed_count``, an EMA with alpha 0.001 that restarts
uniform). :meth:`save` stores both in the checkpoint's ``infos["dropbear_train_state"]``; :meth:`load` restores them
when :attr:`restore_train_state` is set (``scripts/train.py --restore_train_state``) and the checkpoint has them.
"""
from __future__ import annotations

import json
import math
import os
import statistics

from rsl_rl.runners import OnPolicyRunner

TRAIN_STATE_KEY = "dropbear_train_state"


class DropbearOnPolicyRunner(OnPolicyRunner):
    """``OnPolicyRunner`` + ``metrics.jsonl`` + persisted learning rate / adaptive-sampling state."""

    restore_train_state: bool = False
    """Restore :data:`TRAIN_STATE_KEY` on :meth:`load` (set by ``scripts/train.py``)."""
    train_state_report: dict | None = None
    """What :meth:`load` found / restored (for run_info.json)."""

    def _motion_term(self):
        try:
            return self.env.unwrapped.command_manager.get_term("motion")
        except Exception:  # noqa: BLE001  (other tasks / play without a motion term)
            return None

    def train_state(self) -> dict:
        state: dict = {"learning_rate": float(self.alg.learning_rate)}
        term = self._motion_term()
        if term is not None and hasattr(term, "bin_failed_count"):
            state["bin_count"] = int(term.bin_count)
            state["bin_failed_count"] = [float(x) for x in term.bin_failed_count.detach().cpu().tolist()]
        if term is not None and hasattr(term, "extra_train_state"):  # motion-library command: clip-level sampler state
            state["command_extra"] = term.extra_train_state()
        return state

    def save(self, path: str, infos=None):
        infos = dict(infos) if isinstance(infos, dict) else ({} if infos is None else {"infos": infos})
        infos[TRAIN_STATE_KEY] = self.train_state()
        super().save(path, infos)

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer)
        saved = infos.get(TRAIN_STATE_KEY) if isinstance(infos, dict) else None
        report: dict = {"checkpoint": str(path), "found": saved is not None, "restore_requested": self.restore_train_state,
                        "restored": []}
        if saved is not None and self.restore_train_state:
            lr = saved.get("learning_rate")
            if lr is not None and math.isfinite(float(lr)) and float(lr) > 0:
                self.alg.learning_rate = float(lr)
                for group in self.alg.optimizer.param_groups:
                    group["lr"] = float(lr)
                report["restored"].append("learning_rate")
                report["learning_rate"] = float(lr)
            term = self._motion_term()
            bins = saved.get("bin_failed_count")
            extra = saved.get("command_extra")
            if (term is not None and bins is not None and hasattr(term, "load_extra_train_state")
                    and (extra is None or extra.get("library_sha256") != term.extra_train_state().get("library_sha256"))):
                # motion-library command (multiclip, 2026-09-24): the per-bin failure EMA belongs to one library; a
                # checkpoint of another library (or of a single-clip run) with the same bin count must not seed it
                bins = None
                report["bin_failed_count_skipped"] = "other motion library (sha) or no library state in the checkpoint"
            if term is not None and bins is not None:
                if len(bins) == int(term.bin_count):
                    import torch

                    term.bin_failed_count[:] = torch.as_tensor(bins, dtype=term.bin_failed_count.dtype,
                                                               device=term.bin_failed_count.device)
                    report["restored"].append("bin_failed_count")
                    report["bin_failed_count_sum"] = float(sum(bins))
                else:
                    report["bin_count_mismatch"] = [len(bins), int(term.bin_count)]
            if term is not None and extra is not None and hasattr(term, "load_extra_train_state"):
                if term.load_extra_train_state(extra):
                    report["restored"].append("command_extra")
                else:
                    report["command_extra_mismatch"] = True  # other library (sha) or bin layout: not restored
        self.train_state_report = report
        print(f"[DropbearOnPolicyRunner] train state: {json.dumps(report)}", flush=True)
        return infos

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # snapshot episode infos before the parent consumes them
        ep_means: dict[str, float] = {}
        for key in locs["ep_infos"][0] if locs["ep_infos"] else []:
            vals = []
            for info in locs["ep_infos"]:
                if key in info:
                    v = info[key]
                    vals.append(float(v.float().mean()) if hasattr(v, "float") else float(v))
            if vals:
                ep_means[key] = sum(vals) / len(vals)
        super().log(locs, width, pad)
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        it_time = locs["collection_time"] + locs["learn_time"]
        record = {
            "it": int(locs["it"]),
            "mean_reward": statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else None,
            "mean_episode_length": statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else None,
            "losses": {k: float(v) for k, v in locs["loss_dict"].items()},
            "action_std": float(self.alg.policy.action_std.mean()),
            "learning_rate": float(self.alg.learning_rate),
            "steps_per_s": collection_size / it_time if it_time > 0 else None,
            "collection_s": locs["collection_time"],
            "learn_s": locs["learn_time"],
            "episode": ep_means,
        }
        numbers = [v for v in record["losses"].values()] + [record["action_std"]] + list(ep_means.values())
        numbers += [x for x in (record["mean_reward"], record["mean_episode_length"]) if x is not None]
        record["finite"] = all(math.isfinite(x) for x in numbers)
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
