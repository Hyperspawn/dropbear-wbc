"""Teleop session recorder: a LeRobot-v2.1-like dataset with a GR00T ``modality.json`` draft.

Layout (one session directory, one episode per teleop run)::

    data/teleop/<session>/
      meta/info.json            LeRobot v2.1 style (codebase_version, fps, features, paths, splits)
      meta/episodes.jsonl       {"episode_index", "tasks", "length"}
      meta/tasks.jsonl          {"task_index", "task"}
      meta/modality.json        GR00T draft: state/action = left_arm[0:5], right_arm[5:10]; no video yet
      meta/stats.json           mean/std/min/max/q01/q99 of observation.state and action
      meta/teleop_session.json  full provenance: config, calibration SHA, IK/gravity/mapping info, gains, summary
      data/chunk-000/episode_000000.parquet   (needs pyarrow; skipped with a note otherwise)
      extras/episode_000000.npz               every column as float64 arrays (no pyarrow needed)

Columns: ``observation.state`` = measured semantic arm angles (10, G1 names, left then right), ``action`` =
commanded semantic arm angles (10), plus motor-space and diagnostic columns (see :data:`EXTRA_COLUMNS`).

Timing (review fix 2026-09-24): LeRobot v2.1 checks ``timestamp`` against ``frame_index / fps`` (tolerance 1e-4 s), and
GR00T's ``delta_indices`` assume uniform frames. The recorder therefore writes ``timestamp = frame_index / fps``; the
real clock of each frame stays in ``time.sim_s`` / ``time.wall_s`` and control periods the loop missed before a frame
(an overrunning sim-clock loop skips ticks) are counted in ``time.skipped_steps``. :func:`check_timing` reports both the
LeRobot check and the real-clock drift; ``meta/episodes_stats.jsonl`` (v2.1) is written next to the v2.0-style
``stats.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arm_ik import ARM_MOTOR_NAMES, ARM_SEMANTIC_NAMES

STATE_NAMES = list(ARM_SEMANTIC_NAMES["left"] + ARM_SEMANTIC_NAMES["right"])
ARM_MOTORS = list(ARM_MOTOR_NAMES["left"] + ARM_MOTOR_NAMES["right"])
POSE7 = ["x", "y", "z", "qw", "qx", "qy", "qz"]

EXTRA_COLUMNS = {
    # name: (width, description)
    "observation.motor_q": (22, "measured motor angles, SDK slot order [rad]"),
    "observation.motor_dq": (22, "measured motor velocities [rad/s]"),
    "observation.motor_tau": (22, "applied motor torque (tau_est) [N*m]"),
    "action.motor_q": (10, "commanded arm motor targets, SDK slots 12..21 [rad]"),
    "action.tau_ff": (10, "arm gravity feed-forward torque [N*m]"),
    "action.ik_q": (10, "raw IK solution before the joint-velocity limit (semantic) [rad]"),
    "target.left_wrist": (7, "device target, torso frame, xyz + quat wxyz"),
    "target.right_wrist": (7, "device target, torso frame, xyz + quat wxyz"),
    "fk.left_wrist": (7, "FK of the measured semantic arm angles, torso frame"),
    "fk.right_wrist": (7, "FK of the measured semantic arm angles, torso frame"),
    "fk_cmd.left_wrist": (7, "FK of the commanded semantic arm angles, torso frame"),
    "fk_cmd.right_wrist": (7, "FK of the commanded semantic arm angles, torso frame"),
    "sim.left_wrist": (7, "ground-truth hand body pose from LowState.sim (NaN if absent), torso frame"),
    "sim.right_wrist": (7, "ground-truth hand body pose from LowState.sim (NaN if absent), torso frame"),
    "ik.pos_err_m": (2, "IK wrist position residual, left/right [m]"),
    "ik.rot_err_rad": (2, "IK wrist orientation residual, left/right [rad]"),
    "time.sim_s": (1, "simulated time of the LowState used [s]"),
    "time.wall_s": (1, "wall time since the teleop phase started [s]"),
    "time.tick": (1, "LowState tick"),
    "time.skipped_steps": (1, "control periods the loop missed right before this frame (0 = on schedule)"),
    "teleop.tracking": (1, "1 while the arms follow the device (started, device data valid)"),
    "latency.compute_ms": (1, "LowState received -> LowCmd sent [ms]"),
    "latency.state_age_ms": (1, "LowState publish -> received [ms]"),
    "latency.device_age_ms": (1, "device sample stamp -> LowCmd sent [ms]"),
}


LEROBOT_TOLERANCE_S = 1e-4


def check_timing_arrays(arr: dict[str, np.ndarray], fps: float) -> dict:
    """LeRobot timestamp check (``|timestamp - frame_index/fps| <= 1e-4``, on the written uniform timestamps) and the
    real-clock drift of the recorded frames (``time.sim_s`` relative to ``frame_index/fps``, skipped control steps)."""
    ts_col = arr.get("timestamp")
    n = int(len(ts_col)) if ts_col is not None else 0
    out: dict = {"frames": n, "fps": float(fps), "lerobot_tolerance_s": LEROBOT_TOLERANCE_S,
                 "timestamp_rule": "frame_index / fps"}
    if n == 0:
        out.update(lerobot_timestamps_ok=True, real_clock_uniform=True)
        return out
    k = np.arange(n) / fps
    ts_written = k.astype(np.float32).astype(np.float64)
    out["lerobot_max_abs_err_s"] = float(np.abs(ts_written - k).max())
    out["lerobot_timestamps_ok"] = bool(out["lerobot_max_abs_err_s"] <= LEROBOT_TOLERANCE_S)
    sim = np.asarray(arr.get("time.sim_s", np.zeros((n, 1))), dtype=np.float64).reshape(n)
    drift = (sim - sim[0]) - k
    out["sim_clock_drift_max_abs_s"] = float(np.abs(drift).max())
    dts = np.diff(sim)
    out["sim_frame_dt_s"] = {"min": float(dts.min()), "max": float(dts.max())} if n > 1 else None
    skipped = np.asarray(arr.get("time.skipped_steps", np.zeros((n, 1))), dtype=np.float64).reshape(n)
    out["skipped_steps_total"] = int(np.nansum(skipped))
    out["frames_after_a_skip"] = int((skipped > 0).sum())
    out["real_clock_uniform"] = bool(out["sim_clock_drift_max_abs_s"] <= 0.5 / fps and out["skipped_steps_total"] == 0)
    return out


def check_timing(root: str | Path) -> dict:
    """:func:`check_timing_arrays` for a written session. Legacy sessions (before 2026-09-24) stored the loop clock as
    ``timestamp``: their LeRobot check is recomputed from the parquet timestamps when pyarrow is available."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    d = np.load(root / "extras" / "episode_000000.npz", allow_pickle=False)
    arr = {k.replace("__", "."): d[k] for k in d.files if k not in ("state_names", "arm_motor_names")}
    out = check_timing_arrays(arr, fps)
    pq_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    out["timestamp_source"] = "recomputed (frame_index / fps)"
    if pq_path.is_file():
        try:
            import pyarrow.parquet as pq

            t = pq.read_table(pq_path, columns=["timestamp", "frame_index"])
            ts = np.asarray(t.column("timestamp").to_numpy(), dtype=np.float64)
            fi = np.asarray(t.column("frame_index").to_numpy(), dtype=np.float64)
            err = np.abs(ts - fi / fps)
            out["lerobot_max_abs_err_s"] = float(err.max())
            out["lerobot_timestamps_ok"] = bool(err.max() <= LEROBOT_TOLERANCE_S)
            out["timestamp_source"] = "parquet"
        except ImportError:
            pass
    out["episodes_stats_jsonl"] = (root / "meta" / "episodes_stats.jsonl").is_file()
    out["lerobot_v21_valid"] = bool(out["lerobot_timestamps_ok"] and out["episodes_stats_jsonl"])
    return out


def pose7(t: np.ndarray | None) -> np.ndarray:
    """4x4 -> (x, y, z, qw, qx, qy, qz); NaN for None."""
    if t is None:
        return np.full(7, np.nan)
    from dropbear_wbc.motion.rotations import matrix_to_quat

    return np.concatenate([np.asarray(t, float)[:3, 3], matrix_to_quat(np.asarray(t, float)[:3, :3])])


class SessionRecorder:
    """Collects per-frame rows in memory and writes the dataset at :meth:`close`."""

    def __init__(self, root: str | Path, fps: float, task: str, robot_type: str = "dropbear_arms_fixed_base"):
        self.root = Path(root)
        self.fps = float(fps)
        self.task = task
        self.robot_type = robot_type
        self.rows: dict[str, list] = {"observation.state": [], "action": [], "timestamp": []}
        for k in EXTRA_COLUMNS:
            self.rows[k] = []

    def add(self, **cols) -> None:
        for k in self.rows:
            if k not in cols:
                raise KeyError(f"missing column {k!r}")
            self.rows[k].append(np.atleast_1d(np.asarray(cols[k], dtype=np.float64)))

    def __len__(self) -> int:
        return len(self.rows["timestamp"])

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) if v else np.zeros((0,)) for k, v in self.rows.items()}

    def close(self, session_meta: dict) -> dict:
        """Write everything; returns a dict of written paths (and notes)."""
        n = len(self)
        arr = self.arrays()
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (self.root / "extras").mkdir(parents=True, exist_ok=True)
        written: dict = {}
        npz = self.root / "extras" / "episode_000000.npz"
        np.savez_compressed(npz, **{k.replace(".", "__"): v for k, v in arr.items()},
                            state_names=np.array(STATE_NAMES), arm_motor_names=np.array(ARM_MOTORS))
        written["npz"] = str(npz)
        # parquet (LeRobot v2 data file)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            cols = {
                "observation.state": pa.array(arr["observation.state"].astype(np.float32).tolist(), pa.list_(pa.float32())),
                "action": pa.array(arr["action"].astype(np.float32).tolist(), pa.list_(pa.float32())),
                # uniform LeRobot timestamps; the real clock is in time.sim_s / time.wall_s (see module doc)
                "timestamp": pa.array((np.arange(n, dtype=np.float64) / self.fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.zeros(n, dtype=np.int64)),
                "index": pa.array(np.arange(n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            }
            for k, (w, _) in EXTRA_COLUMNS.items():
                v = arr[k]
                cols[k] = pa.array(v.astype(np.float32).tolist(), pa.list_(pa.float32())) if w > 1 else \
                    pa.array(v[:, 0].astype(np.float64))
            path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(cols), path)
            written["parquet"] = str(path)
        except ImportError:
            written["parquet"] = None
            written["note"] = "pyarrow not installed: parquet skipped (the npz has every column)"
        features = {
            "observation.state": {"dtype": "float32", "shape": [10], "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [10], "names": STATE_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for k, (w, desc) in EXTRA_COLUMNS.items():
            names = None
            if w == 7:
                names = POSE7
            elif k.startswith("observation.motor"):
                from dropbear_wbc.kinematics.semantic import MOTOR_NAMES
                names = list(MOTOR_NAMES)
            elif w == 10 and k.startswith("action.motor") or k == "action.tau_ff":
                names = ARM_MOTORS
            elif k == "action.ik_q":
                names = STATE_NAMES
            features[k] = {"dtype": "float32" if w > 1 else "float64", "shape": [w], "names": names,
                           "description": desc}
        info = {
            "codebase_version": "v2.1", "robot_type": self.robot_type, "total_episodes": 1, "total_frames": n,
            "total_tasks": 1, "total_videos": 0, "total_chunks": 1, "chunks_size": 1000, "fps": self.fps,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None, "features": features,
            "notes": "dropbear-wbc arm teleop (simulation). observation.state / action are semantic (G1-named) arm "
                     "angles; no camera stream yet (the Newton bridge renders none).",
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": [self.task], "length": n}) + "\n",
                                             encoding="utf-8")
        (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": self.task}) + "\n", encoding="utf-8")
        modality = {
            "state": {"left_arm": {"start": 0, "end": 5}, "right_arm": {"start": 5, "end": 10}},
            "action": {"left_arm": {"start": 0, "end": 5}, "right_arm": {"start": 5, "end": 10}},
            "video": {},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
            "_draft_notes": [
                "DRAFT for a GR00T N1.7 NEW_EMBODIMENT fine-tune (docs: upstream Isaac-GR00T "
                "getting_started/finetune_new_embodiment.md). Register a modality config with state keys "
                "[left_arm, right_arm] and action keys [left_arm, right_arm] (NON_EEF joint space; RELATIVE is the "
                "SO100 example's choice for arms). See data/teleop/gr00t/dropbear_arms_config_draft.py.",
                "Missing before a real fine-tune: at least one video key (e.g. video.ego_view -> "
                "observation.images.ego_view); the Newton bridge has no camera yet.",
                "Units: radians, semantic (G1-named) joint space; motor-space columns are extras.",
            ],
        }
        (meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
        stats = {}
        for k in ("observation.state", "action"):
            v = arr[k]
            if len(v):
                stats[k] = {"mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "min": v.min(0).tolist(),
                            "max": v.max(0).tolist(), "q01": np.quantile(v, 0.01, axis=0).tolist(),
                            "q99": np.quantile(v, 0.99, axis=0).tolist()}
        (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        # LeRobot v2.1 per-episode statistics (one line per episode)
        ep_stats = {}
        numeric = {"timestamp": (np.arange(n, dtype=np.float64) / self.fps)[:, None],
                   "frame_index": np.arange(n, dtype=np.float64)[:, None], "episode_index": np.zeros((n, 1)),
                   "index": np.arange(n, dtype=np.float64)[:, None], "task_index": np.zeros((n, 1))}
        merged = {k: v for k, v in arr.items() if k != "timestamp"}
        merged.update(numeric)
        for k, v in merged.items():
            v = np.asarray(v, dtype=np.float64).reshape(n, -1) if n else np.zeros((0, 1))
            if not len(v):
                continue
            ep_stats[k] = {"min": np.nanmin(v, 0).tolist(), "max": np.nanmax(v, 0).tolist(),
                           "mean": np.nanmean(v, 0).tolist(), "std": np.nanstd(v, 0).tolist(), "count": [int(n)]}
        (meta / "episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": ep_stats}) + "\n",
                                                   encoding="utf-8")
        timing = check_timing_arrays(arr, self.fps)
        (meta / "timing.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
        written["timing"] = timing
        (meta / "teleop_session.json").write_text(json.dumps(session_meta, indent=1, default=str) + "\n",
                                                  encoding="utf-8")
        written["meta"] = str(meta)
        return written
