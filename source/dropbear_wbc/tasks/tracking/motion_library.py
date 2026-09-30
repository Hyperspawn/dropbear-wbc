"""Motion LIBRARY for multi-clip tracking (contract ``dropbear-motion-library-v1``; Isaac-free: numpy + torch).

BeyondMimic trains one policy per clip. A library policy (the step towards a SONIC-style general tracker) trains on N
contract NPZs at once. This module holds everything that does not need Isaac, so it is unit-tested on CPU:

* :func:`load_manifest` -- the library manifest (JSON listing NPZ paths, optional per-clip weights).
* :func:`build_library` -- loads every clip, fails closed on anything that makes the clips non-interchangeable
  (names/order, fps, plant USD, ankle variant, calibration) or not accepted by the motion validator, and concatenates
  the frames. Only the root body and the tracked bodies are kept (memory: ~1.5 KB/frame instead of ~5 KB/frame).
* :class:`MotionLibrary` -- the concatenated arrays as torch tensors plus per-clip ``starts``/``lengths`` (offsets).
  Frame ``t`` of clip ``c`` is global row ``starts[c] + t``.
* :class:`LibrarySampler` -- reference-state-initialisation sampling over (clip, time-bin):
  BeyondMimic's adaptive bins *per clip* (1 s bins, failure EMA, uniform floor, optional look-ahead kernel) and a
  clip-level mixture of a prior (per-clip weight x duration, or uniform) and the clip's measured failure hazard.

Manifest (paths relative to the manifest file unless absolute)::

    {"schema": "dropbear-motion-library-v1", "name": "accepted_v0",
     "clips": [{"name": "stand", "npz": "../synthetic/stand.npz", "weight": 1.0}, ...]}

Sampling model (all tensors on the sim device; see :class:`LibrarySampler`):

    p(c, b) = p(c) * p(b | c)
    p(b | c)  ~ sum_j kernel_j * (F_{min(b+j, last_c)} + uniform_ratio / nb_c)        (BeyondMimic, per clip)
    p(c)      = (1 - rho) * prior_c + rho * h_c / sum h                              (rho = clip_adaptive_ratio)
    prior_c   ~ weight_c * T_c ("duration") or weight_c ("uniform")
    h_c       = EMA(failures on clip c per step) / EMA(envs on clip c per step)     (failure hazard per env-step)

With one clip and the defaults this reduces exactly to the single-clip BeyondMimic sampler
(``mdp/commands.py``; checked by ``tests/test_motion_library.py``).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .motion_npz import (
    MotionArrays,
    MotionFormatError,
    load_motion_npz,
    load_validation_verdict,
    npz_authored_ankle,
    validate_against_articulation,
    validate_provenance,
)

try:  # torch is needed for MotionLibrary / LibrarySampler only (manifest + build_library are numpy-only)
    import torch
except ImportError:  # pragma: no cover  (.venv-newton has no torch)
    torch = None  # type: ignore[assignment]

LIBRARY_SCHEMA = "dropbear-motion-library-v1"
CLIP_WEIGHTING = ("duration", "uniform")


class LibraryFormatError(MotionFormatError):
    """The manifest or one of its clips violates the library contract (fail closed)."""


# ---------------------------------------------------------------------------------------------------- manifest
@dataclass
class LibraryClipSpec:
    """One manifest entry."""

    name: str
    npz: Path
    weight: float = 1.0
    sha256: str | None = None
    """Optional pin (``tools/build_motion_library_manifest.py`` writes it): the NPZ bytes must still hash to this, so a
    re-settled clip cannot silently change a library (regenerate the manifest instead)."""


@dataclass
class LibraryManifest:
    """Parsed ``dropbear-motion-library-v1`` manifest."""

    name: str
    clips: list[LibraryClipSpec]
    path: Path | None = None
    sha256: str | None = None
    raw: dict = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_manifest(data: dict, base_dir: str | Path | None = None, path: Path | None = None,
                   sha256: str | None = None) -> LibraryManifest:
    """Validate a manifest dict (schema, non-empty unique clip names and paths, positive finite weights)."""
    if not isinstance(data, dict) or data.get("schema") != LIBRARY_SCHEMA:
        raise LibraryFormatError(f"{path or 'manifest'}: schema {data.get('schema') if isinstance(data, dict) else None!r}"
                                 f" != {LIBRARY_SCHEMA!r}")
    entries = data.get("clips")
    if not isinstance(entries, list) or not entries:
        raise LibraryFormatError(f"{path or 'manifest'}: 'clips' must be a non-empty list")
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    clips: list[LibraryClipSpec] = []
    for i, e in enumerate(entries):
        if isinstance(e, str):
            e = {"npz": e}
        if not isinstance(e, dict) or not e.get("npz"):
            raise LibraryFormatError(f"{path or 'manifest'}: clip {i} needs an 'npz' path")
        p = Path(e["npz"])
        p = (p if p.is_absolute() else base / p).resolve()
        weight = float(e.get("weight", 1.0))
        if not (math.isfinite(weight) and weight > 0):
            raise LibraryFormatError(f"{path or 'manifest'}: clip {i} weight {weight} must be finite and > 0")
        pin = e.get("sha256")
        if pin is not None:
            pin = str(pin).lower()
            if len(pin) != 64 or any(ch not in "0123456789abcdef" for ch in pin):
                raise LibraryFormatError(f"{path or 'manifest'}: clip {i} sha256 {e.get('sha256')!r} is not a hex SHA-256")
        clips.append(LibraryClipSpec(name=str(e.get("name") or p.stem), npz=p, weight=weight, sha256=pin))
    names = [c.name for c in clips]
    if len(set(names)) != len(names):
        raise LibraryFormatError(f"{path or 'manifest'}: duplicate clip names {sorted({n for n in names if names.count(n) > 1})}")
    paths = [str(c.npz).lower() for c in clips]
    if len(set(paths)) != len(paths):
        raise LibraryFormatError(f"{path or 'manifest'}: the same NPZ is listed twice (use 'weight' instead)")
    return LibraryManifest(name=str(data.get("name") or (path.stem if path else "library")), clips=clips, path=path,
                           sha256=sha256, raw=data)


def load_manifest(path: str | Path) -> LibraryManifest:
    """Read and validate a manifest file. Relative NPZ paths are resolved against the manifest's directory."""
    p = Path(path).resolve()
    if not p.is_file():
        raise LibraryFormatError(f"motion library manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LibraryFormatError(f"{p}: not JSON: {exc}") from exc
    return parse_manifest(data, base_dir=p.parent, path=p, sha256=_sha256_file(p))


# ---------------------------------------------------------------------------------------------------- building
def npz_calibration_sha256(meta: dict) -> str | None:
    """SHA-256 of the calibration whose SemanticMap produced the clip, if the NPZ records it.

    Settle v2.2 writes ``meta.source_calibration.calibration_sha256``; the retarget sidecar (``meta.sidecar.calibration``)
    may carry it too. Older clips (settle <= v2.1, static stands) record none -> ``None`` ("unknown").
    """
    for block in (meta.get("source_calibration"), (meta.get("sidecar") or {}).get("calibration")):
        if isinstance(block, dict) and block.get("calibration_sha256"):
            return str(block["calibration_sha256"])
    return None


@dataclass
class ClipInfo:
    """Per-clip provenance and extent inside the concatenated arrays."""

    name: str
    path: str
    sha256: str
    start: int
    num_frames: int
    weight: float
    fps: float
    meta_status: str | None
    validation: dict | None
    usd_sha256: str | None
    authored_ankle_tierods: bool
    calibration_sha256: str | None
    closure_max_m: float
    default_motor_pos: dict | None = None

    @property
    def duration_s(self) -> float:
        return (self.num_frames - 1) / self.fps

    def summary(self) -> dict[str, Any]:
        v = self.validation or {}
        return {"name": self.name, "npz": self.path, "sha256": self.sha256, "start": self.start,
                "num_frames": self.num_frames, "duration_s": round(self.duration_s, 3), "weight": self.weight,
                "meta_status": self.meta_status, "verdict": v.get("verdict"), "verdict_stale": v.get("stale"),
                "usd_sha256": self.usd_sha256, "authored_ankle_tierods": self.authored_ankle_tierods,
                "calibration_sha256": self.calibration_sha256, "closure_max_m": self.closure_max_m}


@dataclass
class LibraryData:
    """Concatenated library (numpy). Row ``starts[c] + t`` is frame ``t`` of clip ``c``.

    ``body_*`` arrays hold only ``kept_body_names`` (root first, then the requested tracked bodies, no duplicates).
    """

    manifest: LibraryManifest
    clips: list[ClipInfo]
    fps: float
    joint_names: list[str]
    body_names: list[str]
    motor_names: list[str]
    kept_body_names: list[str]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    closure_residual_m: np.ndarray
    fingerprint: dict = field(default_factory=dict)

    @property
    def starts(self) -> np.ndarray:
        return np.asarray([c.start for c in self.clips], dtype=np.int64)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([c.num_frames for c in self.clips], dtype=np.int64)

    @property
    def num_frames_total(self) -> int:
        return int(self.joint_pos.shape[0])

    def clip_slice(self, c: int) -> slice:
        return slice(self.clips[c].start, self.clips[c].start + self.clips[c].num_frames)


def library_fingerprint(manifest: LibraryManifest, clip_shas: Sequence[str]) -> dict[str, Any]:
    """Content identity of a library: manifest name + (clip name, npz sha256, weight) in order.

    ``sha256`` changes when a clip's bytes, the clip order/names or a weight change; it does not depend on where the
    files live (portable between Windows and a Linux cloud box).
    """
    rows = [{"name": c.name, "sha256": s, "weight": c.weight} for c, s in zip(manifest.clips, clip_shas)]
    blob = json.dumps({"schema": LIBRARY_SCHEMA, "clips": rows}, sort_keys=True).encode()
    return {"schema": LIBRARY_SCHEMA, "name": manifest.name, "sha256": hashlib.sha256(blob).hexdigest(),
            "manifest_path": str(manifest.path) if manifest.path else None, "manifest_file_sha256": manifest.sha256,
            "num_clips": len(rows)}


def manifest_fingerprint(manifest: LibraryManifest | str | Path) -> dict[str, Any]:
    """:func:`library_fingerprint` from the manifest and the clip files' bytes only (no array loading, no checks) --
    what ``scripts/train.py`` records as the run's motion SHA-256 (``motion_acceptance.json``, resume checks)."""
    if not isinstance(manifest, LibraryManifest):
        manifest = load_manifest(manifest)
    missing = [str(c.npz) for c in manifest.clips if not c.npz.is_file()]
    if missing:
        raise LibraryFormatError(f"library {manifest.name!r}: clip files not found: {missing}")
    shas = [_sha256_file(c.npz) for c in manifest.clips]
    changed = [c.name for c, s in zip(manifest.clips, shas) if c.sha256 is not None and c.sha256 != s]
    if changed:
        raise LibraryFormatError(f"library {manifest.name!r}: clips {changed} no longer match their manifest sha256 pins "
                                 "(regenerate the manifest with tools/build_motion_library_manifest.py)")
    return library_fingerprint(manifest, shas)


def build_library(
    manifest: LibraryManifest | str | Path,
    *,
    joint_names: Sequence[str] | None = None,
    body_names: Sequence[str] | None = None,
    motor_names: Sequence[str] | None = None,
    keep_body_names: Sequence[str] | None = None,
    root_body_name: str = "world",
    expected_fps: float | None = None,
    expected_usd_sha256: str | None = None,
    expected_authored_ankle: bool | None = None,
    allow_rejected: bool = False,
    require_accepted: bool = True,
) -> LibraryData:
    """Load, check and concatenate every clip of ``manifest`` (fail closed).

    Per clip: the contract self-checks (:func:`load_motion_npz`), names/order vs the live articulation when
    ``joint_names``/``body_names``/``motor_names`` are given (else vs the first clip), ``fps`` vs ``expected_fps``, and
    :func:`validate_provenance` (schema, producer status, validator verdict, USD SHA, ankle variant).
    ``require_accepted`` (default) additionally needs a sibling ``<clip>.validation.json`` whose verdict is
    ``accepted`` and not stale; ``allow_rejected`` (exploratory runs only) lifts both the rejection and this check.
    Across clips: identical joint/body/motor names, fps, USD SHA-256 (where recorded), ankle variant and calibration
    SHA-256 (where recorded; unknown is allowed and reported).

    Raises:
        LibraryFormatError / MotionFormatError: on any violation, naming the clip.
    """
    if not isinstance(manifest, LibraryManifest):
        manifest = load_manifest(manifest)
    arrays: list[MotionArrays] = []
    infos: list[ClipInfo] = []
    ref_names: tuple[list[str], list[str], list[str]] | None = None
    start = 0
    for spec in manifest.clips:
        where = f"library {manifest.name!r} clip {spec.name!r} ({spec.npz})"
        clip_sha = _sha256_file(spec.npz) if spec.npz.is_file() else None
        if spec.sha256 is not None and clip_sha is not None and clip_sha != spec.sha256:
            raise LibraryFormatError(f"{where}: NPZ sha256 {clip_sha[:12]}... != manifest pin {spec.sha256[:12]}... (the "
                                     "clip changed after the manifest was built; regenerate it with "
                                     "tools/build_motion_library_manifest.py)")
        try:
            m = load_motion_npz(spec.npz)
            if joint_names is not None and body_names is not None and motor_names is not None:
                validate_against_articulation(m, joint_names, body_names, motor_names, expected_fps=expected_fps,
                                              fps_tol=1e-3)
            elif expected_fps is not None and abs(m.fps - expected_fps) > 1e-3:
                raise MotionFormatError(f"fps {m.fps} != policy rate {expected_fps}")
            verdict = load_validation_verdict(spec.npz)
            validate_provenance(m, expected_usd_sha256=expected_usd_sha256, allow_rejected=allow_rejected,
                                expected_authored_ankle=expected_authored_ankle, validation=verdict)
        except MotionFormatError as exc:
            raise LibraryFormatError(f"{where}: {exc}") from exc
        if require_accepted and not allow_rejected:
            if verdict is None:
                raise LibraryFormatError(f"{where}: no <clip>.validation.json verdict; run tools/validate_motion_npz.py "
                                         "--write-verdicts (a library only takes validated 'accepted' clips)")
            if verdict.get("verdict") != "accepted" or verdict.get("stale"):
                raise LibraryFormatError(f"{where}: verdict {verdict.get('verdict')!r}"
                                         f"{' (STALE: written for other NPZ bytes)' if verdict.get('stale') else ''}; "
                                         "a library only takes validated 'accepted' clips")
        names = (list(m.joint_names), list(m.body_names), list(m.motor_names))
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            which = [k for k, a, b in zip(("joint_names", "body_names", "motor_names"), names, ref_names) if a != b]
            raise LibraryFormatError(f"{where}: {which} differ from clip {manifest.clips[0].name!r} (order matters)")
        meta = m.meta or {}
        dmp = meta.get("default_motor_pos")
        infos.append(ClipInfo(
            name=spec.name, path=str(spec.npz).replace("\\", "/"), sha256=clip_sha or _sha256_file(spec.npz), start=start,
            num_frames=m.num_frames, weight=spec.weight, fps=float(m.fps), meta_status=meta.get("status"),
            validation=verdict, usd_sha256=meta.get("usd_sha256"), authored_ankle_tierods=npz_authored_ankle(meta),
            calibration_sha256=npz_calibration_sha256(meta), closure_max_m=float(np.max(m.closure_residual_m)),
            default_motor_pos=dict(dmp) if isinstance(dmp, dict) else None))
        arrays.append(m)
        start += m.num_frames

    first = infos[0]
    for key in ("fps", "authored_ankle_tierods"):
        vals = {getattr(c, key) for c in infos}
        if len(vals) > 1:
            raise LibraryFormatError(f"library {manifest.name!r}: clips differ in {key}: "
                                     f"{ {c.name: getattr(c, key) for c in infos} }")
    for key in ("usd_sha256", "calibration_sha256"):
        vals = {getattr(c, key) for c in infos if getattr(c, key)}
        if len(vals) > 1:
            raise LibraryFormatError(f"library {manifest.name!r}: clips were produced on different {key}: "
                                     f"{ {c.name: getattr(c, key) for c in infos} }")

    assert ref_names is not None
    j_names, b_names, m_names = ref_names
    keep = [root_body_name] + [b for b in (keep_body_names or []) if b != root_body_name]
    missing = [b for b in keep if b not in b_names]
    if missing:
        raise LibraryFormatError(f"library {manifest.name!r}: bodies {missing} not in the clips' body_names")
    if len(set(keep)) != len(keep):
        raise LibraryFormatError(f"duplicate kept bodies {keep}")
    kb = [b_names.index(b) for b in keep]
    cat = lambda key, sel=None: np.concatenate(  # noqa: E731
        [getattr(a, key) if sel is None else getattr(a, key)[:, sel] for a in arrays], axis=0)
    data = LibraryData(
        manifest=manifest, clips=infos, fps=first.fps, joint_names=j_names, body_names=b_names, motor_names=m_names,
        kept_body_names=keep, joint_pos=cat("joint_pos"), joint_vel=cat("joint_vel"),
        body_pos_w=cat("body_pos_w", kb), body_quat_w=cat("body_quat_w", kb), body_lin_vel_w=cat("body_lin_vel_w", kb),
        body_ang_vel_w=cat("body_ang_vel_w", kb), closure_residual_m=cat("closure_residual_m"))
    data.fingerprint = library_fingerprint(manifest, [c.sha256 for c in infos])
    return data


def library_report(data: LibraryData) -> dict[str, Any]:
    """JSON-able summary: fingerprint, per-clip provenance, totals and the provenance consensus."""
    cal = {c.calibration_sha256 for c in data.clips}
    return {**data.fingerprint, "fps": data.fps, "num_frames_total": data.num_frames_total,
            "duration_s_total": round(sum(c.duration_s for c in data.clips), 3),
            "kept_body_names": data.kept_body_names, "num_joints": len(data.joint_names),
            "num_bodies_articulation": len(data.body_names),
            "bytes_float32": int(sum(a.nbytes for a in (data.joint_pos, data.joint_vel, data.body_pos_w, data.body_quat_w,
                                                         data.body_lin_vel_w, data.body_ang_vel_w))),
            "usd_sha256": next((c.usd_sha256 for c in data.clips if c.usd_sha256), None),
            "authored_ankle_tierods": data.clips[0].authored_ankle_tierods,
            "calibration_sha256": sorted(x for x in cal if x) or None,
            "calibration_unknown_clips": [c.name for c in data.clips if not c.calibration_sha256],
            "all_accepted": all((c.validation or {}).get("verdict") == "accepted" and not (c.validation or {}).get("stale")
                                for c in data.clips),
            "clips": [c.summary() for c in data.clips]}


# ---------------------------------------------------------------------------------------------------- torch side
def _require_torch():
    if torch is None:
        raise ImportError("torch is required for MotionLibrary / LibrarySampler")


class MotionLibrary:
    """Concatenated library on a torch device (the tracking command's ``self.motion`` for library tasks).

    Attributes (F = total frames, J = all joints, M = motors, K = tracked bodies):
        full_joint_pos, full_joint_vel: (F, J) RSI rows.
        joint_pos, joint_vel: (F, M) motor columns (command / metrics).
        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w: (F, K, .) tracked bodies (``tracked_body_names``).
        root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w: (F, .) root body.
        starts, lengths: (N,) long, clip offsets / frame counts.
    """

    def __init__(self, data: LibraryData, motor_names: Sequence[str], tracked_body_names: Sequence[str],
                 device: str = "cpu"):
        _require_torch()
        self.data = data
        self.fps = float(data.fps)
        self.clips = data.clips
        self.clip_names = [c.name for c in data.clips]
        self.num_clips = len(data.clips)
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)  # noqa: E731
        motor_idx = [data.joint_names.index(n) for n in motor_names]
        self.full_joint_pos = t(data.joint_pos)
        self.full_joint_vel = t(data.joint_vel)
        mi = torch.as_tensor(motor_idx, dtype=torch.long, device=device)
        self.joint_pos = self.full_joint_pos[:, mi].contiguous()
        self.joint_vel = self.full_joint_vel[:, mi].contiguous()
        missing = [b for b in tracked_body_names if b not in data.kept_body_names]
        if missing:
            raise LibraryFormatError(f"tracked bodies {missing} were not kept when building the library")
        ki = [data.kept_body_names.index(b) for b in tracked_body_names]
        self.body_pos_w = t(data.body_pos_w[:, ki])
        self.body_quat_w = t(data.body_quat_w[:, ki])
        self.body_lin_vel_w = t(data.body_lin_vel_w[:, ki])
        self.body_ang_vel_w = t(data.body_ang_vel_w[:, ki])
        self.root_pos_w = t(data.body_pos_w[:, 0])
        self.root_quat_w = t(data.body_quat_w[:, 0])
        self.root_lin_vel_w = t(data.body_lin_vel_w[:, 0])
        self.root_ang_vel_w = t(data.body_ang_vel_w[:, 0])
        self.starts = torch.as_tensor(data.starts, dtype=torch.long, device=device)
        self.lengths = torch.as_tensor(data.lengths, dtype=torch.long, device=device)
        self.num_frames_total = data.num_frames_total
        # single-clip compatible provenance attributes (scripts/train.py / play.py / export read them)
        statuses = {c.meta_status for c in data.clips}
        self.arrays_meta = {"status": statuses.pop() if len(statuses) == 1 else "mixed",
                            "library": data.fingerprint}
        verdicts = {(c.validation or {}).get("verdict") for c in data.clips}
        self.validation = {"verdict": verdicts.pop() if len(verdicts) == 1 else "mixed",
                           "stale": any((c.validation or {}).get("stale") for c in data.clips),
                           "clips": {c.name: (c.validation or {}).get("verdict") for c in data.clips}}
        self.arrays_closure_max = max(c.closure_max_m for c in data.clips)

    def global_index(self, clip_ids, time_steps):
        """Row of frame ``time_steps`` (clamped to the clip) of clip ``clip_ids``."""
        last = self.lengths[clip_ids] - 1
        return self.starts[clip_ids] + torch.minimum(torch.clamp(time_steps, min=0), last)


class LibrarySampler:
    """Adaptive RSI sampling over (clip, time-bin); see the module docstring for the model.

    State tensors (persisted across chunked restarts by ``runner.DropbearOnPolicyRunner``):
        bin_failed_count: (B,) EMA of per-step failures per bin (BeyondMimic's ``bin_failed_count``, all clips).
        clip_fail_ema, clip_active_ema: (N,) EMA of per-step failures / envs on each clip.

    Args:
        lengths: frames per clip (N,).
        fps: frame rate (bins are ~1 s: ``nb_c = T_c // fps + 1`` like upstream).
        weights: per-clip prior weights (N,), default 1.
        clip_weighting: ``"duration"`` (prior ~ weight x frames: time-uniform like one long clip) or ``"uniform"``.
        clip_adaptive_ratio: rho, share of p(c) that follows the failure hazard (0 = prior only).
        adaptive_kernel_size, adaptive_lambda, adaptive_uniform_ratio, adaptive_alpha: BeyondMimic per-bin parameters.
        clip_alpha: EMA rate of the clip hazard statistics (per env step).
    """

    def __init__(self, lengths: Sequence[int], fps: float, weights: Sequence[float] | None = None, *,
                 clip_weighting: str = "duration", clip_adaptive_ratio: float = 0.5, adaptive_kernel_size: int = 1,
                 adaptive_lambda: float = 0.8, adaptive_uniform_ratio: float = 0.1, adaptive_alpha: float = 0.001,
                 clip_alpha: float = 0.001, device: str = "cpu"):
        _require_torch()
        if clip_weighting not in CLIP_WEIGHTING:
            raise ValueError(f"clip_weighting {clip_weighting!r} not in {CLIP_WEIGHTING}")
        if not 0.0 <= clip_adaptive_ratio <= 1.0:
            raise ValueError("clip_adaptive_ratio must be in [0, 1]")
        self.device = device
        self.lengths = torch.as_tensor(list(lengths), dtype=torch.long, device=device)
        if self.lengths.numel() == 0 or bool((self.lengths < 2).any()):
            raise ValueError("every clip needs >= 2 frames")
        n = int(self.lengths.numel())
        self.num_clips = n
        self.fps = float(fps)
        w = torch.ones(n, device=device) if weights is None else torch.as_tensor(list(weights), dtype=torch.float32,
                                                                                device=device)
        if w.shape != (n,) or not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
            raise ValueError("weights must be N finite positive values")
        self.weights = w
        self.clip_weighting = clip_weighting
        self.clip_adaptive_ratio = float(clip_adaptive_ratio)
        # bins: nb_c = T_c // fps + 1 (upstream: int(time_step_total // policy_rate) + 1)
        self.bins_per_clip = (self.lengths.double() // self.fps).long() + 1
        self.bin_offsets = torch.zeros(n + 1, dtype=torch.long, device=device)
        self.bin_offsets[1:] = torch.cumsum(self.bins_per_clip, 0)
        self.bin_count = int(self.bin_offsets[-1])
        self.bin_clip = torch.repeat_interleave(torch.arange(n, device=device), self.bins_per_clip)
        self.bin_local = torch.arange(self.bin_count, device=device) - self.bin_offsets[self.bin_clip]
        self.bin_last = self.bin_offsets[self.bin_clip + 1] - 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self.clip_fail_ema = torch.zeros(n, dtype=torch.float, device=device)
        self.clip_active_ema = torch.zeros(n, dtype=torch.float, device=device)
        self._current_clip_failed = torch.zeros(n, dtype=torch.float, device=device)
        self._current_clip_active = torch.zeros(n, dtype=torch.float, device=device)
        k = max(int(adaptive_kernel_size), 1)
        kern = torch.tensor([adaptive_lambda ** i for i in range(k)], dtype=torch.float, device=device)
        self.kernel = kern / kern.sum()
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_alpha = float(adaptive_alpha)
        self.clip_alpha = float(clip_alpha)

    # -- bookkeeping
    def time_bin(self, clip_ids, time_steps):
        """Global bin of (clip, local frame) (upstream: ``t * bin_count // time_step_total``, per clip)."""
        nb = self.bins_per_clip[clip_ids]
        local = torch.clamp((time_steps * nb) // torch.clamp(self.lengths[clip_ids], min=1), torch.zeros_like(nb), nb - 1)
        return self.bin_offsets[clip_ids] + local

    def record_failures(self, clip_ids, time_steps, failed) -> None:
        """Count the failed episodes (bool mask) at their (clip, time-bin) for this step."""
        if not bool(failed.any()):
            return
        c, t = clip_ids[failed], time_steps[failed]
        self._current_bin_failed += torch.bincount(self.time_bin(c, t), minlength=self.bin_count).float()
        self._current_clip_failed += torch.bincount(c, minlength=self.num_clips).float()

    def record_exposure(self, clip_ids) -> None:
        """Count the envs currently on each clip (once per env step)."""
        self._current_clip_active += torch.bincount(clip_ids, minlength=self.num_clips).float()

    def step(self) -> None:
        """EMA update (once per env step, like upstream ``_update_command``)."""
        a = self.adaptive_alpha
        self.bin_failed_count = a * self._current_bin_failed + (1 - a) * self.bin_failed_count
        self._current_bin_failed.zero_()
        b = self.clip_alpha
        self.clip_fail_ema = b * self._current_clip_failed + (1 - b) * self.clip_fail_ema
        self.clip_active_ema = b * self._current_clip_active + (1 - b) * self.clip_active_ema
        self._current_clip_failed.zero_()
        self._current_clip_active.zero_()

    # -- distributions
    def clip_prior(self):
        p = self.weights * (self.lengths.float() if self.clip_weighting == "duration" else 1.0)
        return p / p.sum()

    def clip_hazard(self):
        """Failures per env-step on each clip (0 where the clip has had no exposure yet)."""
        return torch.where(self.clip_active_ema > 1e-9, self.clip_fail_ema / self.clip_active_ema.clamp(min=1e-9),
                           torch.zeros_like(self.clip_fail_ema))

    def clip_probs(self):
        prior = self.clip_prior()
        h = self.clip_hazard()
        if self.clip_adaptive_ratio <= 0.0 or float(h.sum()) <= 0.0:
            return prior
        return (1.0 - self.clip_adaptive_ratio) * prior + self.clip_adaptive_ratio * h / h.sum()

    def bin_probs_within_clip(self):
        """p(b | c) for every global bin (sums to 1 within each clip)."""
        base = self.bin_failed_count + self.adaptive_uniform_ratio / self.bins_per_clip[self.bin_clip].float()
        idx = torch.arange(self.bin_count, device=self.device)
        q = torch.zeros_like(base)
        for j in range(int(self.kernel.numel())):  # replicate padding at each clip's last bin
            q = q + self.kernel[j] * base[torch.minimum(idx + j, self.bin_last)]
        sums = torch.zeros(self.num_clips, dtype=q.dtype, device=self.device).index_add_(0, self.bin_clip, q)
        return q / sums[self.bin_clip]

    def joint_probs(self):
        """p(c, b) over all global bins (sums to 1)."""
        return self.clip_probs()[self.bin_clip] * self.bin_probs_within_clip()

    def sample(self, n: int, generator=None):
        """Draw ``n`` (clip_ids, local time_steps) from p(c, b), uniform inside the bin (upstream formula)."""
        probs = self.joint_probs()
        bins = torch.multinomial(probs, n, replacement=True, generator=generator)
        clips = self.bin_clip[bins]
        nb = self.bins_per_clip[clips].float()
        u = torch.rand(n, device=self.device, generator=generator)
        t = ((self.bin_local[bins].float() + u) / nb * (self.lengths[clips] - 1).float()).long()
        return clips, t

    def stats(self) -> dict[str, Any]:
        """Sampling-distribution summary (entropy normalised by log(B), top bin, per-clip probabilities)."""
        p = self.joint_probs()
        ent = float(-(p * (p + 1e-12).log()).sum())
        pmax, imax = p.max(dim=0)
        return {"entropy": ent / math.log(self.bin_count) if self.bin_count > 1 else 0.0, "top1_prob": float(pmax),
                "top1_clip": int(self.bin_clip[imax]), "top1_bin_local": int(self.bin_local[imax]),
                "clip_probs": self.clip_probs().tolist(), "clip_hazard": self.clip_hazard().tolist()}

    # -- persistence (runner.DropbearOnPolicyRunner train state)
    def state_dict(self) -> dict[str, Any]:
        return {"bins_per_clip": self.bins_per_clip.tolist(), "bin_failed_count": self.bin_failed_count.tolist(),
                "clip_fail_ema": self.clip_fail_ema.tolist(), "clip_active_ema": self.clip_active_ema.tolist()}

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore if the bin layout matches; returns whether it did."""
        if list(state.get("bins_per_clip", [])) != self.bins_per_clip.tolist():
            return False
        f = lambda k, ref: torch.as_tensor(state[k], dtype=ref.dtype, device=ref.device)  # noqa: E731
        self.bin_failed_count[:] = f("bin_failed_count", self.bin_failed_count)
        for k in ("clip_fail_ema", "clip_active_ema"):
            if k in state:
                getattr(self, k)[:] = f(k, getattr(self, k))
        return True


def play_assignment(num_envs: int, num_clips: int, clip_names: Sequence[str], play_clips: Sequence[str] | None = None,
                    device: str = "cpu"):
    """Fixed evaluation assignment: env ``i`` plays clip ``order[i % len(order)]`` (all clips, or ``play_clips``)."""
    _require_torch()
    if play_clips:
        unknown = [c for c in play_clips if c not in clip_names]
        if unknown:
            raise ValueError(f"play_clips {unknown} not in the library {list(clip_names)}")
        order = [list(clip_names).index(c) for c in play_clips]
    else:
        order = list(range(num_clips))
    order_t = torch.as_tensor(order, dtype=torch.long, device=device)
    return order_t[torch.arange(num_envs, device=device) % len(order)]


def future_indices(library: MotionLibrary, clip_ids, time_steps, future_steps: Sequence[int]):
    """(E, K) global rows of frames ``t + k`` for each k in ``future_steps``, clamped to the clip's last frame
    (the deploy runner's :meth:`MotionReference.frame_index` clamps the same way)."""
    if not future_steps:
        return None
    ks = torch.as_tensor(list(future_steps), dtype=torch.long, device=time_steps.device)
    return library.global_index(clip_ids[:, None].expand(-1, len(ks)), time_steps[:, None] + ks[None, :])
