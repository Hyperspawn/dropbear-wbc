"""Reference motion for tracking policies (BeyondMimic / unitree_rl_lab ``Mimic``).

Sources:
    * contract NPZ ``dropbear-motion-npz-v1`` (CONTRACTS section 4): all joints in
      Isaac order with ``joint_names``; bodies with ``body_names``; wxyz quaternions.
    * CSV ``dropbear-motion-csv-v1`` (header row; root xyz + xyzw quaternion + 22
      motor columns) or Unitree's header-less CSV (same layout, motor order).
    * ONNX: a BeyondMimic-style export whose outputs include the reference at an
      input ``time_step`` (``joint_pos``, ``joint_vel``, ``body_pos_w``, ``body_quat_w``).

The reference is sampled per policy step ``k`` at time ``time_start + k*step_dt``
(nearest frame; BeyondMimic trains with integer time steps at the policy rate).
When the policy starts, :meth:`align` rotates (yaw) and translates (xy) the
reference so its anchor matches the robot's current anchor, as unitree_rl_lab's
``State_Mimic`` does with ``init_quat``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..sdk import motors
from . import quat as Q
from .config import MotionCfg
from .observations import ReferenceFrame

ROOT_BODY = "world"


@dataclass
class RawSample:
    """Unaligned reference sample: policy-order joints and anchor/root poses (world, wxyz)."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


class MotionReference:
    """Base class: subclasses implement :meth:`raw` and :attr:`num_frames`/:attr:`fps`."""

    fps: float
    num_frames: int
    anchor_offset_pos: np.ndarray
    anchor_offset_quat: np.ndarray

    def __init__(self, cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float):
        self.cfg, self.joint_names, self.step_dt = cfg, joint_names, step_dt
        self._align_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._align_p_ref = np.zeros(3)
        self._align_p_rob = np.zeros(3)

    # -- to implement
    def raw(self, frame: int) -> RawSample:
        raise NotImplementedError

    # -- common
    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def frame_index(self, step: int) -> int:
        t = self.cfg.time_start + step * self.step_dt
        return int(np.clip(round(t * self.fps), 0, self.num_frames - 1))

    def done(self, step: int) -> bool:
        t = self.cfg.time_start + step * self.step_dt
        end = self.duration_s if self.cfg.time_end is None else min(self.cfg.time_end, self.duration_s)
        return t >= end

    def align(self, robot_anchor_quat_w: np.ndarray, robot_anchor_pos_w: np.ndarray | None) -> None:
        """Align the reference at step 0 to the robot anchor (mode ``cfg.align``)."""
        s = self.raw(self.frame_index(0))
        mode = self.cfg.align
        if mode == "none":
            self._align_q = np.array([1.0, 0.0, 0.0, 0.0])
            self._align_p_ref = np.zeros(3)
            self._align_p_rob = np.zeros(3)
            return
        self._align_q = Q.from_yaw(Q.yaw(robot_anchor_quat_w) - Q.yaw(s.anchor_quat_w))
        if mode == "yaw_xy" and robot_anchor_pos_w is not None:
            self._align_p_ref = np.array([s.anchor_pos_w[0], s.anchor_pos_w[1], 0.0])
            self._align_p_rob = np.array([robot_anchor_pos_w[0], robot_anchor_pos_w[1], 0.0])
        else:
            self._align_p_ref = np.zeros(3)
            self._align_p_rob = np.zeros(3)

    def sample(self, step: int) -> ReferenceFrame:
        s = self.raw(self.frame_index(step))
        pos = Q.rotate(self._align_q, s.anchor_pos_w - self._align_p_ref) + self._align_p_rob
        return ReferenceFrame(joint_pos=s.joint_pos, joint_vel=s.joint_vel, anchor_pos_w=pos,
                              anchor_quat_w=Q.mul(self._align_q, s.anchor_quat_w))

    def _set_offset(self, root_p: np.ndarray | None, root_q: np.ndarray | None, anchor_p: np.ndarray,
                    anchor_q: np.ndarray) -> None:
        """Anchor pose in the root frame: config value, else derived from one frame, else identity.

        Identity is only correct when the anchor *is* the root body. A non-root anchor without a
        configured offset or a root pose to derive it from fails closed. The robot anchor would
        otherwise silently be the root frame (for example a tracking export without
        ``motion.anchor_offset_pos/quat``, whose 14 reference bodies do not include ``world``).
        """
        if self.cfg.anchor_offset_quat is not None:
            self.anchor_offset_quat = Q.normalize(np.asarray(self.cfg.anchor_offset_quat, float))
            self.anchor_offset_pos = np.asarray(self.cfg.anchor_offset_pos or (0.0, 0.0, 0.0), float)
        elif root_p is not None and root_q is not None:
            self.anchor_offset_quat = Q.normalize(Q.mul(Q.conj(root_q), anchor_q))
            self.anchor_offset_pos = Q.rotate_inv(root_q, anchor_p - root_p)
        elif (self.cfg.anchor_body or ROOT_BODY) != ROOT_BODY:
            raise ValueError(f"anchor body {self.cfg.anchor_body!r} is not the root {ROOT_BODY!r}: give "
                             "motion.anchor_offset_pos/anchor_offset_quat (the anchor pose in the root link frame) "
                             f"or a reference that includes the {ROOT_BODY!r} body")
        else:
            self.anchor_offset_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.anchor_offset_pos = np.zeros(3)


class NpzMotion(MotionReference):
    """Contract NPZ reference (``dropbear-motion-npz-v1``)."""

    def __init__(self, cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float):
        super().__init__(cfg, joint_names, step_dt)
        data = np.load(cfg.file, allow_pickle=False)
        self.fps = float(data["fps"])
        all_joints = [str(j) for j in data["joint_names"]]
        missing = [j for j in joint_names if j not in all_joints]
        if missing:
            raise KeyError(f"{cfg.file}: joints {missing} not in joint_names")
        cols = [all_joints.index(j) for j in joint_names]
        self._jp = np.asarray(data["joint_pos"], float)[:, cols]
        self._jv = np.asarray(data["joint_vel"], float)[:, cols]
        bodies = [str(b) for b in data["body_names"]]
        anchor = cfg.anchor_body or ROOT_BODY
        if anchor not in bodies:
            raise KeyError(f"{cfg.file}: anchor body {anchor!r} not in body_names")
        a = bodies.index(anchor)
        self._ap = np.asarray(data["body_pos_w"], float)[:, a]
        self._aq = np.asarray(data["body_quat_w"], float)[:, a]
        self.num_frames = self._jp.shape[0]
        self.body_names = bodies
        self.meta = json.loads(str(data["meta"])) if "meta" in data else {}
        if ROOT_BODY in bodies:
            r = bodies.index(ROOT_BODY)
            self._set_offset(np.asarray(data["body_pos_w"][0, r], float), np.asarray(data["body_quat_w"][0, r], float),
                             self._ap[0], self._aq[0])
        else:
            self._set_offset(None, None, self._ap[0], self._aq[0])

    def raw(self, frame: int) -> RawSample:
        return RawSample(self._jp[frame], self._jv[frame], self._ap[frame], self._aq[frame])


class CsvMotion(MotionReference):
    """CSV reference: ``root_x,y,z, qx,qy,qz,qw`` then motor angles. Anchor = root * offset."""

    def __init__(self, cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float):
        super().__init__(cfg, joint_names, step_dt)
        if cfg.fps is None:
            sidecar = Path(cfg.file).with_suffix(".json")
            if not sidecar.exists():
                raise ValueError(f"{cfg.file}: fps missing (give motion.fps or a {sidecar.name} sidecar)")
            self.fps = float(json.loads(sidecar.read_text(encoding="utf-8"))["fps"])
        else:
            self.fps = float(cfg.fps)
        with open(cfg.file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        try:
            float(rows[0][0])
            header = ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw", *motors.MOTOR_NAMES]
        except ValueError:
            header, rows = [h.strip() for h in rows[0]], rows[1:]
        data = np.asarray(rows, dtype=float)
        col = {h: i for i, h in enumerate(header)}
        missing = [j for j in joint_names if j not in col]
        if missing:
            raise KeyError(f"{cfg.file}: missing joint columns {missing}")
        self._jp = data[:, [col[j] for j in joint_names]]
        self._jv = np.gradient(self._jp, 1.0 / self.fps, axis=0) if len(data) > 1 else np.zeros_like(self._jp)
        self._rp = data[:, [col["root_x"], col["root_y"], col["root_z"]]]
        xyzw = data[:, [col["root_qx"], col["root_qy"], col["root_qz"], col["root_qw"]]]
        self._rq = Q.normalize(xyzw[:, [3, 0, 1, 2]])
        self.num_frames = len(data)
        self._set_offset(None, None, self._rp[0], self._rq[0])  # identity unless configured

    def raw(self, frame: int) -> RawSample:
        q = Q.mul(self._rq[frame], self.anchor_offset_quat)
        p = self._rp[frame] + Q.rotate(self._rq[frame], self.anchor_offset_pos)
        return RawSample(self._jp[frame], self._jv[frame], p, q)


class OnnxMotion(MotionReference):
    """Reference embedded in a BeyondMimic-style ONNX export, queried by ``time_step``.

    Args:
        query: ``query(frame) -> dict`` returning ``joint_pos`` (J_all,), ``joint_vel`` (J_all,),
            ``body_pos_w`` (B, 3), ``body_quat_w`` (B, 4) for that motion frame.
        reference_joint_names: names of the J_all joint columns (Isaac order).
        body_names: names of the B bodies (``MotionCommandCfg.body_names``).
        num_frames: number of frames (``time_step`` is clamped by the export).
    """

    def __init__(self, cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float,
                 query: Callable[[int], dict], reference_joint_names: list[str], body_names: list[str],
                 num_frames: int):
        super().__init__(cfg, joint_names, step_dt)
        self.fps = 1.0 / step_dt
        self.num_frames = num_frames
        self._query = query
        missing = [j for j in joint_names if j not in reference_joint_names]
        if missing:
            raise KeyError(f"ONNX reference joints lack {missing}")
        self._cols = [reference_joint_names.index(j) for j in joint_names]
        anchor = cfg.anchor_body or ROOT_BODY
        if anchor not in body_names:
            raise KeyError(f"anchor body {anchor!r} not among ONNX reference bodies {body_names}")
        self._a = body_names.index(anchor)
        out0 = query(0)
        bp, bq = np.asarray(out0["body_pos_w"], float), np.asarray(out0["body_quat_w"], float)
        if ROOT_BODY in body_names:
            r = body_names.index(ROOT_BODY)
            self._set_offset(bp[r], bq[r], bp[self._a], bq[self._a])
        else:
            self._set_offset(None, None, bp[self._a], bq[self._a])
        self._cache: dict[int, RawSample] = {}

    def raw(self, frame: int) -> RawSample:
        if frame not in self._cache:
            o = self._query(frame)
            self._cache[frame] = RawSample(np.asarray(o["joint_pos"], float)[self._cols],
                                           np.asarray(o["joint_vel"], float)[self._cols],
                                           np.asarray(o["body_pos_w"], float)[self._a],
                                           np.asarray(o["body_quat_w"], float)[self._a])
        return self._cache[frame]


def load_motion(cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float) -> MotionReference:
    """Build the file-based reference for ``cfg`` (``source`` ``npz`` or ``csv``)."""
    if cfg.source == "npz":
        return NpzMotion(cfg, joint_names, step_dt)
    if cfg.source == "csv":
        return CsvMotion(cfg, joint_names, step_dt)
    if cfg.source == "runtime":
        raise ValueError("this policy embeds no reference (motion.source='runtime', a motion-library export): feed one at "
                         "runtime, e.g. tools/policy_runner.py --motion <clip.npz|clip.csv>")
    raise ValueError(f"motion source {cfg.source!r} needs the ONNX policy (use OnnxMotion)")


class RuntimeReferenceError(ValueError):
    """A reference fed at runtime does not meet the export's ``motion.requirements`` (fail closed)."""


def load_runtime_motion(cfg: MotionCfg, path: Path, joint_names: tuple[str, ...], step_dt: float, *,
                        allow_rejected: bool = False, fps: float | None = None) -> tuple[MotionReference, dict]:
    """Reference from ``path`` (contract NPZ or motion CSV) for a policy, checked against ``cfg.requirements``.

    The file replaces whatever reference the export named (for ``source="runtime"`` exports -- motion-library policies
    -- it is the only reference). The export's anchor body/offset, alignment and time window are kept. Fails closed
    (:class:`RuntimeReferenceError`) when the reference was produced on another plant USD or ankle variant, has another
    frame rate than the policy (NPZ; a CSV is sampled at the nearest frame), or was REJECTED by
    ``tools/validate_motion_npz.py`` / its producer (unless ``allow_rejected``). A clip without a verdict is used and
    reported as ``unvalidated``.

    Returns:
        (reference, report) where report records the file, its SHA-256, the checks and their outcome.
    """
    import dataclasses
    import hashlib

    from ..tasks.tracking.motion_npz import load_validation_verdict, npz_authored_ankle

    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeReferenceError(f"runtime reference {path} not found")
    req = dict(cfg.requirements or {})
    policy_fps = float(req.get("fps") or 1.0 / step_dt)
    is_csv = path.suffix.lower() == ".csv"
    report: dict = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "format": "csv" if is_csv else "npz", "requirements": req, "checks": {}}
    meta: dict = {}
    if is_csv:
        side = path.with_suffix(".json")
        side_d = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
        cal = side_d.get("calibration") or {}
        if "authored_ankle_tierods" in cal:
            meta["authored_ankle_tierods"] = cal["authored_ankle_tierods"]
        file_fps = float(fps if fps is not None else side_d.get("fps", 0.0) or 0.0) or None
        if file_fps is None:
            raise RuntimeReferenceError(f"{path}: fps unknown (give --motion-fps or a {side.name} sidecar with 'fps')")
        report["checks"]["fps"] = {"file": file_fps, "policy": policy_fps,
                                   "note": None if abs(file_fps - policy_fps) < 1e-6 else "nearest-frame sampling"}
    else:
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
            file_fps = float(d["fps"])
        if abs(file_fps - policy_fps) > 1e-6:
            raise RuntimeReferenceError(f"{path}: fps {file_fps} != policy rate {policy_fps} (contract NPZs are at the "
                                        "policy rate; resample the clip)")
        report["checks"]["fps"] = {"file": file_fps, "policy": policy_fps, "note": None}
        if meta.get("status") == "rejected" and not allow_rejected:
            raise RuntimeReferenceError(f"{path}: meta.status='rejected' ({meta.get('status_reasons')}); pass "
                                        "--allow-rejected-motion only for experiments")
    usd = meta.get("usd_sha256")
    if req.get("usd_sha256") and usd and usd != req["usd_sha256"]:
        raise RuntimeReferenceError(f"{path}: produced on USD {usd[:12]}..., the policy's plant is "
                                    f"{str(req['usd_sha256'])[:12]}...")
    report["checks"]["usd_sha256"] = usd or "not recorded"
    if "authored_ankle_tierods" in req and (not is_csv or "authored_ankle_tierods" in meta):
        variant = npz_authored_ankle(meta)
        if variant != bool(req["authored_ankle_tierods"]):
            raise RuntimeReferenceError(
                f"{path}: produced on the {'authored revolute' if variant else 'spherical'} ankle tie rods, the policy "
                f"was trained on the {'authored revolute' if req['authored_ankle_tierods'] else 'spherical'} ones "
                "(docs/CONTRACTS.md 0.2)")
        report["checks"]["authored_ankle_tierods"] = variant
    verdict = None if is_csv else load_validation_verdict(path)
    if verdict and verdict.get("verdict") == "rejected" and not allow_rejected:
        raise RuntimeReferenceError(f"{path}: REJECTED by tools/validate_motion_npz.py: {verdict.get('reasons')}; pass "
                                    "--allow-rejected-motion only for experiments")
    report["checks"]["validation"] = (verdict or {}).get("verdict", "unvalidated")
    report["checks"]["validation_stale"] = (verdict or {}).get("stale")
    report["allow_rejected_motion"] = bool(allow_rejected)
    new_cfg = dataclasses.replace(cfg, file=path, source="csv" if is_csv else "npz",
                                  fps=file_fps if is_csv else cfg.fps)
    ref = CsvMotion(new_cfg, joint_names, step_dt) if is_csv else NpzMotion(new_cfg, joint_names, step_dt)
    report["num_frames"] = int(ref.num_frames)
    report["duration_s"] = float(ref.duration_s)
    return ref, report


class LiveMotion(MotionReference):
    """Live reference for real-time control (text -> motion, a behaviour scheduler): a
    :class:`~dropbear_wbc.motion.stream.MotionStream` seeded with ``cfg.file``. Every new contract NPZ dropped into
    ``watch_dir`` (written atomically, e.g. by ``tools/text_to_motion.py``) is checked like a runtime reference
    (:func:`load_runtime_motion`: plant, ankle variant, fps, validator verdict; fail closed per clip), spliced
    ``lead_s`` ahead of the playhead (aligned in x, y, yaw, cross-faded over ``blend_s``) and followed by the idle clip
    (default: the first clip) blended over ``idle_blend_s``. Never ``done``: the controller calls :meth:`poll` once per
    policy step. The anchor offset and start alignment are the export's, as for any reference."""

    def __init__(self, cfg: MotionCfg, joint_names: tuple[str, ...], step_dt: float, watch_dir: Path,
                 idle_file: Path | None = None, lead_s: float = 0.1, blend_s: float = 0.4, idle_blend_s: float = 0.8,
                 capacity_s: float = 600.0, allow_rejected: bool = False):
        super().__init__(cfg, joint_names, step_dt)
        from ..motion.stream import MotionStream

        first = self._load(Path(cfg.file))
        self.fps = first["fps"]
        self._joint_names, self._body_names = first["joint_names"], first["body_names"]
        missing = [j for j in joint_names if j not in self._joint_names]
        if missing:
            raise KeyError(f"{cfg.file}: joints {missing} not in joint_names")
        self._cols = [self._joint_names.index(j) for j in joint_names]
        anchor = cfg.anchor_body or ROOT_BODY
        if anchor not in self._body_names or ROOT_BODY not in self._body_names:
            raise KeyError(f"{cfg.file}: live references need the {ROOT_BODY!r} and anchor {anchor!r} bodies")
        self._a, root = self._body_names.index(anchor), self._body_names.index(ROOT_BODY)
        self.stream = MotionStream(self.fps, first["clip"], root, capacity_s=capacity_s)
        self.num_frames = self.stream.capacity
        self._idle = self._load(Path(idle_file or cfg.file))["clip"]
        self.lead, self.blend_s, self.idle_blend_s = int(round(lead_s * self.fps)), blend_s, idle_blend_s
        self.watch_dir, self.allow_rejected = Path(watch_dir), allow_rejected
        self._seen = {p.name for p in self.watch_dir.glob("*.npz")} if self.watch_dir.is_dir() else set()
        self.events: list[dict] = []
        s = self.stream
        self._set_offset(s.body_pos_w[0, root], s.body_quat_w[0, root], s.body_pos_w[0, self._a], s.body_quat_w[0, self._a])

    @staticmethod
    def _load(path: Path) -> dict:
        from ..motion.stream import Clip

        with np.load(path, allow_pickle=False) as d:
            return {"fps": float(d["fps"]), "joint_names": [str(j) for j in d["joint_names"]],
                    "body_names": [str(b) for b in d["body_names"]],
                    "clip": Clip(np.asarray(d["joint_pos"], float), np.asarray(d["body_pos_w"], float),
                                 np.asarray(d["body_quat_w"], float))}

    def raw(self, frame: int) -> RawSample:
        s = self.stream
        return RawSample(s.joint_pos[frame, self._cols], s.joint_vel[frame, self._cols], s.body_pos_w[frame, self._a],
                         s.body_quat_w[frame, self._a])

    def done(self, step: int) -> bool:
        return False

    def play(self, path: Path, step: int) -> dict:
        """Check ``path`` like a runtime reference, then splice it at the playhead (+ lead) followed by the idle clip."""
        _, report = load_runtime_motion(self.cfg, path, self.joint_names, self.step_dt, allow_rejected=self.allow_rejected)
        new = self._load(path)
        if new["joint_names"] != self._joint_names or new["body_names"] != self._body_names:
            raise RuntimeReferenceError(f"{path}: joint/body layout differs from the live stream's")
        start, end = self.stream.splice(new["clip"], self.frame_index(step) + self.lead, blend_s=self.blend_s)
        self.stream.splice(self._idle, end, blend_s=self.idle_blend_s)
        ev = {"file": Path(path).name, "frame": start, "end": end, "verdict": report["checks"].get("validation")}
        self.events.append(ev)
        return ev

    def poll(self, step: int) -> list[dict]:
        """Splice every new ``*.npz`` in the watched folder (name order); a rejected clip is reported and skipped."""
        out = []
        if not self.watch_dir.is_dir():
            return out
        for p in sorted(self.watch_dir.glob("*.npz")):
            if p.name in self._seen:
                continue
            self._seen.add(p.name)
            try:
                out.append(self.play(p, step))
            except (RuntimeReferenceError, KeyError, ValueError) as exc:
                out.append({"file": p.name, "rejected": f"{type(exc).__name__}: {exc}"})
                self.events.append(out[-1])
        return out
