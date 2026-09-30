"""Semantic (G1-named) joint space <-> Dropbear motor space (``dropbear-semantic-v1``).

The map is measured by ``tools/calibrate_semantics.py`` from the USD physics and stored in
``data/calibration/dropbear_semantic_calibration.json`` (schema ``dropbear-semantic-calibration-v1``).

Semantic DOFs (order :data:`SEMANTIC_NAMES`) use G1 names and G1 sign conventions in the pelvis frame
(x forward, y left, z up): pitch about +y, roll about +x, yaw about +z; knee flexion positive. The
**elbow uses the exact G1 joint convention**: ``elbow = 0`` means forearm pointing forward (90 deg
flexion), ``elbow = +pi/2`` a straight arm hanging down, positive = extension (G1 ``*_elbow_joint``
rotates about +y). See the calibration JSON ``conventions`` block and docs/CONTRACTS.md section 2.

Per-DOF map types (all vectorised over leading batch dimensions):

* ``linear``: ``q_sem = scale * (m - offset)``;
* ``lut1d``:  monotonic table ``q_sem = f(m)`` (knee crank, elbow four-bar), inverse by table swap;
* ``fixed``:  a DOF the plant cannot move (measured range ~0); constant value, motor held;
* ``lut2d``:  the two calf motors ``(a, b)`` -> ``(ankle_pitch, ankle_roll)`` on a regular motor grid
  (bilinear), inverse from a regular semantic grid refined by Newton steps on the forward table;
* ``serial3``: three serial motors (hip or shoulder) <-> the three YXZ Euler angles (pitch, roll, yaw) of
  the segment's zero-referenced orientation, exactly, through the measured screw axes (``serial_groups``).
  Needed because Dropbear's hip chain is roll -> yaw -> pitch while the semantic (G1) order is
  pitch -> roll -> yaw, and because the measured axes are not exactly the idealised ones. Each DOF entry
  also carries the linear approximation (``scale``/``offset``) used as the Newton initial guess and to pick
  the Euler branch.

Every conversion clips to the measured valid range and reports saturation (:class:`SaturationReport`).
Units: radians.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "dropbear-semantic-calibration-v1"
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[3] / "data/calibration/dropbear_semantic_calibration.json"

MOTOR_NAMES: tuple[str, ...] = (
    "PG_left_leg_pitch", "PG_left_leg_roll", "LL_hip_joint", "LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81",
    "PG_right_leg_pitch", "PG_right_leg_roll", "RL_hip_joint", "RL_knee_actuator_joint", "RL_Revolute67", "RL_Revolute81",
    "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
    "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
)
"""``dropbear-wbc-motors-v1`` order (SDK slots)."""

SEMANTIC_NAMES: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
)
"""``dropbear-semantic-v1`` order."""

NUM_DOF = 22


@dataclass
class SaturationReport:
    """Where a conversion had to clip.

    Attributes:
        requested: (..., 22) input values (semantic or motor, depending on the call) before clipping.
        used: (..., 22) values after clipping to the valid range.
        clipped: (..., 22) bool, ``|requested - used| > tol``.
        names: DOF names of the input space.
    """

    requested: np.ndarray
    used: np.ndarray
    clipped: np.ndarray
    names: tuple[str, ...]

    def any(self) -> bool:
        return bool(self.clipped.any())

    def excess(self) -> np.ndarray:
        """Signed amount [rad] by which the request exceeded the valid range (0 where not clipped)."""
        return self.requested - self.used

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-DOF clip fraction and max |excess| [rad] for DOFs that clipped at least once."""
        c = self.clipped.reshape(-1, NUM_DOF)
        e = np.abs(self.excess()).reshape(-1, NUM_DOF)
        return {n: {"fraction": float(c[:, i].mean()), "max_excess_rad": float(e[:, i].max())}
                for i, n in enumerate(self.names) if c[:, i].any()}


def _arr(x) -> np.ndarray:
    """JSON list (None = invalid) -> float array with NaN."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if len(x) and isinstance(x[0], (list, tuple, np.ndarray)):
        return np.array([[np.nan if v is None else v for v in row] for row in x], dtype=np.float64)
    return np.array([np.nan if v is None else v for v in x], dtype=np.float64)


def _bilinear(grid_x: np.ndarray, grid_y: np.ndarray, table: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Bilinear interpolation of ``table[ix, iy, ...]`` on a regular grid; returns (value, d/dx, d/dy)."""
    nx, ny = len(grid_x), len(grid_y)
    fx = (x - grid_x[0]) / (grid_x[-1] - grid_x[0]) * (nx - 1)
    fy = (y - grid_y[0]) / (grid_y[-1] - grid_y[0]) * (ny - 1)
    ix = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    tx = (fx - ix)[..., None] if table.ndim == 3 else fx - ix
    ty = (fy - iy)[..., None] if table.ndim == 3 else fy - iy
    v00, v10 = table[ix, iy], table[ix + 1, iy]
    v01, v11 = table[ix, iy + 1], table[ix + 1, iy + 1]
    val = (v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty)
    hx = (grid_x[-1] - grid_x[0]) / (nx - 1)
    hy = (grid_y[-1] - grid_y[0]) / (ny - 1)
    dvdx = ((v10 - v00) * (1 - ty) + (v11 - v01) * ty) / hx
    dvdy = ((v01 - v00) * (1 - tx) + (v11 - v10) * tx) / hy
    return val, dvdx, dvdy


class _Linear:
    def __init__(self, d: dict, motor_index: dict[str, int]):
        self.m = motor_index[d["motors"][0]]
        self.scale = float(d["scale"])
        self.offset = float(d["offset"])
        self.range = np.asarray(d["valid_range"], dtype=np.float64)

    def fwd(self, m: np.ndarray) -> np.ndarray:
        return self.scale * (m[..., self.m] - self.offset)

    def inv(self, s: np.ndarray) -> np.ndarray:
        return s / self.scale + self.offset


class _Lut1d:
    def __init__(self, d: dict, motor_index: dict[str, int]):
        self.m = motor_index[d["motors"][0]]
        self.mg = np.asarray(d["motor_grid"], dtype=np.float64)
        self.sv = np.asarray(d["semantic_values"], dtype=np.float64)
        if np.any(np.diff(self.mg) <= 0):
            raise ValueError(f"lut1d motor_grid must be increasing ({d['motors']})")
        ds = np.diff(self.sv)
        if not (np.all(ds > 0) or np.all(ds < 0)):
            raise ValueError(f"lut1d semantic_values must be strictly monotonic ({d['motors']})")
        order = np.argsort(self.sv)
        self.inv_s, self.inv_m = self.sv[order], self.mg[order]
        self.range = np.array([self.sv.min(), self.sv.max()])

    def fwd(self, m: np.ndarray) -> np.ndarray:
        return np.interp(m[..., self.m], self.mg, self.sv)

    def inv(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.inv_s, self.inv_m)


class _Fixed:
    """A DOF the mechanism cannot move (measured range ~0): constant semantic value, motor held."""

    def __init__(self, d: dict, motor_index: dict[str, int]):
        self.m = motor_index[d["motors"][0]]
        self.motor_value = float(d["motor_value"])
        self.value = float(d["semantic_value"])
        self.range = np.array([self.value, self.value])

    def fwd(self, m: np.ndarray) -> np.ndarray:
        return np.full(m.shape[:-1], self.value)

    def inv(self, s: np.ndarray) -> np.ndarray:
        return np.full(np.shape(s), self.motor_value)


class _Pair:
    """Two motors (a, b) <-> two semantic DOFs (pitch, roll) of one ankle."""

    def __init__(self, d: dict, motor_index: dict[str, int]):
        self.ma, self.mb = (motor_index[n] for n in d["motors"])
        self.ag = np.asarray(d["a_grid"], dtype=np.float64)
        self.bg = np.asarray(d["b_grid"], dtype=np.float64)
        self.table = np.stack([_arr(d["pitch"]), _arr(d["roll"])], axis=-1)  # (na, nb, 2), NaN-filled
        inv = d["inverse"]
        self.pg = np.asarray(inv["pitch_grid"], dtype=np.float64)
        self.rg = np.asarray(inv["roll_grid"], dtype=np.float64)
        self.inv_ab = np.stack([_arr(inv["a"]), _arr(inv["b"])], -1)
        self.inv_valid = np.asarray(inv["valid"], dtype=bool)
        self.a_lim = np.asarray(d["a_limits"], dtype=np.float64)
        self.b_lim = np.asarray(d["b_limits"], dtype=np.float64)
        vp = np.stack(np.meshgrid(self.pg, self.rg, indexing="ij"), -1)[self.inv_valid]
        self.valid_points = vp  # (K, 2) feasible (pitch, roll) grid nodes
        # Newton initial guess table: invalid inverse nodes take the nearest valid node's motors
        guess = self.inv_ab.copy()
        vi = np.argwhere(self.inv_valid)
        for i, j in np.argwhere(~self.inv_valid):
            k = np.argmin(((vi - [i, j]) ** 2).sum(-1))
            guess[i, j] = self.inv_ab[tuple(vi[k])]
        self.inv_guess = guess
        self.pitch_range = np.asarray(d["pitch_range"], dtype=np.float64)
        self.roll_range = np.asarray(d["roll_range"], dtype=np.float64)

    def fwd(self, m: np.ndarray) -> np.ndarray:
        a = np.clip(m[..., self.ma], self.ag[0], self.ag[-1])
        b = np.clip(m[..., self.mb], self.bg[0], self.bg[-1])
        val, _, _ = _bilinear(self.ag, self.bg, self.table, a, b)
        return val

    def project(self, pr: np.ndarray) -> np.ndarray:
        """Clip (pitch, roll) (..., 2) to the feasible set (nearest feasible grid node when outside)."""
        p = np.clip(pr[..., 0], self.pg[0], self.pg[-1])
        r = np.clip(pr[..., 1], self.rg[0], self.rg[-1])
        fp = (p - self.pg[0]) / (self.pg[-1] - self.pg[0]) * (len(self.pg) - 1)
        fr = (r - self.rg[0]) / (self.rg[-1] - self.rg[0]) * (len(self.rg) - 1)
        ip0 = np.clip(np.floor(fp).astype(np.int64), 0, len(self.pg) - 2)
        ir0 = np.clip(np.floor(fr).astype(np.int64), 0, len(self.rg) - 2)
        cell_ok = (self.inv_valid[ip0, ir0] & self.inv_valid[ip0 + 1, ir0] & self.inv_valid[ip0, ir0 + 1]
                   & self.inv_valid[ip0 + 1, ir0 + 1])
        out = np.stack([p, r], axis=-1)
        if (~cell_ok).any():
            q = out[~cell_ok]
            d2 = ((q[:, None, :] - self.valid_points[None, :, :]) ** 2).sum(-1)
            out[~cell_ok] = self.valid_points[np.argmin(d2, axis=1)]
        return out

    def inv(self, pr: np.ndarray, newton_iters: int = 12) -> np.ndarray:
        """(pitch, roll) (..., 2) -> motors (a, b) (..., 2). Input must already be projected."""
        guess, _, _ = _bilinear(self.pg, self.rg, self.inv_guess, pr[..., 0], pr[..., 1])
        a, b = guess[..., 0], guess[..., 1]
        for _ in range(newton_iters):
            a = np.clip(a, self.ag[0], self.ag[-1])
            b = np.clip(b, self.bg[0], self.bg[-1])
            val, da, db = _bilinear(self.ag, self.bg, self.table, a, b)
            err = pr - val
            j00, j10, j01, j11 = da[..., 0], da[..., 1], db[..., 0], db[..., 1]
            det = j00 * j11 - j01 * j10
            ok = np.abs(det) > 1e-9
            det = np.where(ok, det, 1.0)
            step_a = np.where(ok, (j11 * err[..., 0] - j01 * err[..., 1]) / det, 0.0)
            step_b = np.where(ok, (-j10 * err[..., 0] + j00 * err[..., 1]) / det, 0.0)
            shrink = np.maximum(1.0, np.hypot(step_a, step_b) / 0.1)  # trust region 0.1 rad per step
            a, b = a + step_a / shrink, b + step_b / shrink
        a = np.clip(a, max(self.ag[0], self.a_lim[0]), min(self.ag[-1], self.a_lim[1]))
        b = np.clip(b, max(self.bg[0], self.b_lim[0]), min(self.bg[-1], self.b_lim[1]))
        return np.stack([a, b], axis=-1)


def _rot(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation (..., 3, 3) about a fixed unit ``axis`` (3,) by ``angle`` (...)."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    m = np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y,
                  t * x * y + s * z, t * y * y + c, t * y * z - s * x,
                  t * x * z - s * y, t * y * z + s * x, t * z * z + c], axis=-1)
    return m.reshape(np.shape(angle) + (3, 3))


def _rotvec(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3)."""
    from dropbear_wbc.motion.rotations import rotation_log

    return rotation_log(m)


def _wrap(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def euler_yxz(m: np.ndarray) -> np.ndarray:
    """``R = Ry(a) Rx(b) Rz(c)`` -> (a, b, c) (..., 3), b in [-pi/2, pi/2]."""
    b = np.arcsin(np.clip(-m[..., 1, 2], -1.0, 1.0))
    a = np.arctan2(m[..., 0, 2], m[..., 2, 2])
    c = np.arctan2(m[..., 1, 0], m[..., 1, 1])
    return np.stack([a, b, c], axis=-1)


def euler_yxz_matrix(e: np.ndarray) -> np.ndarray:
    """(a, b, c) (..., 3) -> ``Ry(a) Rx(b) Rz(c)`` (..., 3, 3)."""
    ex, ey, ez = np.eye(3)
    return _rot(ey, e[..., 0]) @ _rot(ex, e[..., 1]) @ _rot(ez, e[..., 2])


class _Serial3:
    """Three serial motors <-> (pitch, roll, yaw) YXZ Euler angles of one segment (see module doc).

    Model: ``R_seg(m) = Rot(a0, k0 m0) Rot(a1, k1 m1) Rot(a2, k2 m2) R_rest`` (chain order, space-frame
    axes ``a_i`` measured at the all-zero configuration, slopes ``k_i``), ``F = R_seg R_ref^T`` and the
    semantic triple is the YXZ Euler decomposition of ``F`` on the branch closest to the linear estimate.
    """

    def __init__(self, g: dict, motor_index: dict[str, int]):
        self.chain = [motor_index[n] for n in g["chain_motors"]]
        self.sem = [SEMANTIC_NAMES.index(n) for n in g["semantic"]]
        self.axes = np.asarray(g["axes_root"], dtype=np.float64)
        self.axes /= np.linalg.norm(self.axes, axis=1, keepdims=True)
        self.slopes = np.asarray(g["slopes"], dtype=np.float64)
        self.c = np.asarray(g["R_rest"], dtype=np.float64) @ np.asarray(g["R_ref"], dtype=np.float64).T
        self.role = [motor_index[n] for n in g["role_motors"]]  # motor of each semantic DOF (linear guess)
        self.scale = np.asarray(g["linear_scale"], dtype=np.float64)
        self.offset = np.asarray(g["linear_offset"], dtype=np.float64)
        self.box = np.asarray(g["semantic_box"], dtype=np.float64)
        self.iters = int(g.get("newton_iters", 40))

    def _chain(self, th: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        rs = [_rot(self.axes[i], th[..., i]) for i in range(3)]
        return rs[0] @ rs[1] @ rs[2], rs

    def linear(self, m: np.ndarray) -> np.ndarray:
        return self.scale * (m[..., self.role] - self.offset)

    def fwd(self, m: np.ndarray) -> np.ndarray:
        """Motors (..., 22) -> semantic (..., 3) in (pitch, roll, yaw) order."""
        th = self.slopes * m[..., self.chain]
        g, _ = self._chain(th)
        e1 = euler_yxz(g @ self.c)
        e2 = np.stack([e1[..., 0] + np.pi, np.pi - e1[..., 1], e1[..., 2] + np.pi], axis=-1)
        lin = self.linear(m)
        d1 = np.abs(_wrap(e1 - lin)).sum(-1)
        d2 = np.abs(_wrap(e2 - lin)).sum(-1)
        e = np.where((d2 < d1)[..., None], e2, e1)
        return lin + _wrap(e - lin)

    def inv(self, s: np.ndarray) -> np.ndarray:
        """Semantic (..., 3) -> chain motor values (..., 3) (unclipped), damped Newton from the linear guess."""
        gt = euler_yxz_matrix(s) @ self.c.T
        m0 = np.zeros(s.shape[:-1] + (22,))
        m0[..., self.role] = s / self.scale + self.offset
        th = self.slopes * m0[..., self.chain]
        for _ in range(self.iters):
            g, rs = self._chain(th)
            err = _rotvec(gt @ np.swapaxes(g, -1, -2))
            w0 = np.broadcast_to(self.axes[0], err.shape)
            w1 = np.einsum("...ij,j->...i", rs[0], self.axes[1])
            w2 = np.einsum("...ij,j->...i", rs[0] @ rs[1], self.axes[2])
            jac = np.stack([w0, w1, w2], axis=-1)
            jtj = np.swapaxes(jac, -1, -2) @ jac + 1e-10 * np.eye(3)
            step = np.linalg.solve(jtj, np.einsum("...ji,...j->...i", jac, err)[..., None])[..., 0]
            n = np.linalg.norm(step, axis=-1, keepdims=True)
            th = th + step * np.minimum(1.0, 0.5 / np.maximum(n, 1e-300))
            if float(np.abs(err).max(initial=0.0)) < 1e-13:
                break
        return th / self.slopes

    def inv_sequence(self, s_seq: np.ndarray, max_step: float | None = None, damping: float = 1e-4,
                     iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Continuity-seeded inverse over a trajectory (T, 3) -> chain motors (T, 3) and orientation error (T,) [rad].

        Frame t starts its damped Newton at frame t-1's motor solution (frame 0: the linear guess), so the solution
        stays on the branch of the previous frame instead of the one closest to the per-frame linear guess (review
        fix 2026-09-24: near the YXZ singularity at +-90 deg abduction the semantic pitch/yaw swing by up to 2 rad
        while the segment orientation moves < 10 deg per frame). ``max_step`` [rad] bounds each motor's change per
        frame; the orientation then lags the target through the singular region (error returned) instead of the
        motors jumping.
        """
        s_seq = np.asarray(s_seq, dtype=np.float64)
        out = np.zeros_like(s_seq)
        err_out = np.zeros(len(s_seq))
        gt_all = euler_yxz_matrix(s_seq) @ self.c.T
        th_prev = None
        for t in range(len(s_seq)):
            if th_prev is None:
                m0 = np.zeros(22)
                m0[self.role] = s_seq[t] / self.scale + self.offset
                th = self.slopes * m0[self.chain]
            else:
                th = th_prev.copy()
            gt = gt_all[t]
            for _ in range(iters):
                g, rs = self._chain(th)
                err = _rotvec(gt @ g.T)
                w1 = rs[0] @ self.axes[1]
                w2 = rs[0] @ rs[1] @ self.axes[2]
                jac = np.stack([self.axes[0], w1, w2], axis=-1)
                step = np.linalg.solve(jac.T @ jac + damping * np.eye(3), jac.T @ err)
                n = np.linalg.norm(step)
                th = th + step * min(1.0, 0.5 / max(n, 1e-300))
                if n < 1e-12:
                    break
            if th_prev is not None and max_step is not None:
                lim = np.abs(self.slopes) * max_step
                th = th_prev + np.clip(th - th_prev, -lim, lim)
            g, _ = self._chain(th)
            err_out[t] = float(np.linalg.norm(_rotvec(gt @ g.T)))
            out[t] = th / self.slopes
            th_prev = th
        return out, err_out


class SemanticMap:
    """Measured semantic <-> motor map. Construct with :meth:`load`."""

    def __init__(self, calib: dict):
        if calib.get("schema") != SCHEMA:
            raise ValueError(f"unexpected schema {calib.get('schema')!r}, want {SCHEMA}")
        if tuple(calib["semantic_names"]) != SEMANTIC_NAMES or tuple(calib["motor_names"]) != MOTOR_NAMES:
            raise ValueError("calibration name tables differ from dropbear-semantic-v1 / dropbear-wbc-motors-v1")
        self.calib = calib
        mi = {n: i for i, n in enumerate(MOTOR_NAMES)}
        self.motor_limits = np.asarray(calib["motor_limits_rad"], dtype=np.float64)
        self._single: list[tuple[int, object]] = []
        self._pairs: list[tuple[int, int, _Pair]] = []
        for side, pd in calib.get("ankle_pairs", {}).items():
            pair = _Pair(pd, mi)
            self._pairs.append((SEMANTIC_NAMES.index(f"{side}_ankle_pitch"), SEMANTIC_NAMES.index(f"{side}_ankle_roll"),
                                pair))
        paired = {i for ip, ir, _ in self._pairs for i in (ip, ir)}
        self._serial: list[_Serial3] = [_Serial3(g, mi) for g in calib.get("serial_groups", {}).values()]
        in_serial = {i: (g, k) for g in self._serial for k, i in enumerate(g.sem)}
        limits = np.zeros((NUM_DOF, 2))
        for i, name in enumerate(SEMANTIC_NAMES):
            d = calib["dofs"][name]
            if i in paired:
                pair = next(p for ip, ir, p in self._pairs if i in (ip, ir))
                limits[i] = pair.pitch_range if name.endswith("pitch") else pair.roll_range
                continue
            if d["type"] == "serial3":
                if i not in in_serial:
                    raise ValueError(f"{name}: serial3 DOF without a serial_groups entry")
                g, k = in_serial[i]
                limits[i] = np.sort(g.box[k])
                continue
            kind = d["type"]
            obj = {"linear": _Linear, "lut1d": _Lut1d, "fixed": _Fixed}.get(kind)
            obj = obj(d, mi) if obj is not None else None
            if obj is None:
                raise ValueError(f"{name}: unsupported map type {kind!r}")
            self._single.append((i, obj))
            limits[i] = np.sort(obj.range)
        self.semantic_limits = limits
        self.semantic_zero_motor_pos = np.asarray(calib["semantic_zero_motor_pos"], dtype=np.float64)
        self.standing_motor_pos = np.asarray(calib["standing_motor_pos"], dtype=np.float64)
        self.last_report: SaturationReport | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION) -> "SemanticMap":
        return cls(json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------------------------------------
    def motor_to_semantic(self, q_motor: np.ndarray, clip: bool = True, return_report: bool = False):
        """Motor angles (..., 22) [rad] -> semantic angles (..., 22) [rad].

        Motor values are clipped to the authored motor limits (reported in ``last_report``)."""
        q = np.asarray(q_motor, dtype=np.float64)
        if q.shape[-1] != NUM_DOF:
            raise ValueError(f"expected (..., 22), got {q.shape}")
        m = np.clip(q, self.motor_limits[:, 0], self.motor_limits[:, 1]) if clip else q
        rep = SaturationReport(q, m, np.abs(q - m) > 1e-9, MOTOR_NAMES)
        out = np.zeros(q.shape)
        for i, obj in self._single:
            out[..., i] = obj.fwd(m)
        for ip, ir, pair in self._pairs:
            pr = pair.fwd(m)
            out[..., ip], out[..., ir] = pr[..., 0], pr[..., 1]
        for g in self._serial:
            out[..., g.sem] = g.fwd(m)
        self.last_report = rep
        return (out, rep) if return_report else out

    def semantic_to_motor_sequence(self, q_sem: np.ndarray, max_serial_step: float | None = None,
                                   clip: bool = True) -> tuple[np.ndarray, dict]:
        """Trajectory version of :meth:`semantic_to_motor` (``(T, 22)``): the serial3 groups (hips, shoulders) are solved
        with :meth:`_Serial3.inv_sequence` (continuity-seeded, optionally rate-limited to ``max_serial_step`` rad per
        frame); every other DOF is identical to :meth:`semantic_to_motor`. Returns ``(q_motor, info)`` with the
        per-group orientation error (the price of the rate limit) and the largest per-frame motor step."""
        s = np.asarray(q_sem, dtype=np.float64)
        if s.ndim != 2 or s.shape[-1] != NUM_DOF:
            raise ValueError(f"expected (T, 22), got {s.shape}")
        out = self.semantic_to_motor(s, clip=clip)
        used = np.clip(s, self.semantic_limits[:, 0], self.semantic_limits[:, 1]) if clip else s
        info: dict = {"serial_orientation_error_deg": {}, "max_serial_step": max_serial_step}
        for g in self._serial:
            m, err = g.inv_sequence(used[:, g.sem], max_step=max_serial_step)
            out[:, g.chain] = m
            name = f"{SEMANTIC_NAMES[g.sem[0]].rsplit('_', 1)[0]}"
            info["serial_orientation_error_deg"][name] = {"p95": float(np.degrees(np.percentile(err, 95))),
                                                          "max": float(np.degrees(err.max()))}
        out = np.clip(out, self.motor_limits[:, 0], self.motor_limits[:, 1])
        step = np.abs(np.diff(out, axis=0))
        info["max_motor_step_rad"] = float(step.max()) if step.size else 0.0
        info["max_motor_step_motor"] = MOTOR_NAMES[int(step.max(axis=0).argmax())] if step.size else None
        return out, info

    def semantic_to_motor(self, q_sem: np.ndarray, clip: bool = True, return_report: bool = False):
        """Semantic angles (..., 22) [rad] -> motor angles (..., 22) [rad].

        Requests outside the measured valid range are clipped (ankle pairs: projected to the nearest
        feasible (pitch, roll)); ``last_report`` (or the returned report) lists what was clipped."""
        s = np.asarray(q_sem, dtype=np.float64)
        if s.shape[-1] != NUM_DOF:
            raise ValueError(f"expected (..., 22), got {s.shape}")
        used = np.clip(s, self.semantic_limits[:, 0], self.semantic_limits[:, 1]) if clip else s.copy()
        out = np.zeros(s.shape)
        for ip, ir, pair in self._pairs:
            pr = pair.project(np.stack([s[..., ip], s[..., ir]], -1)) if clip else np.stack([s[..., ip], s[..., ir]], -1)
            used[..., ip], used[..., ir] = pr[..., 0], pr[..., 1]
            ab = pair.inv(pr)
            out[..., pair.ma], out[..., pair.mb] = ab[..., 0], ab[..., 1]
        for i, obj in self._single:
            out[..., obj.m] = obj.inv(used[..., i])
        for g in self._serial:
            out[..., g.chain] = g.inv(used[..., g.sem])
        out = np.clip(out, self.motor_limits[:, 0], self.motor_limits[:, 1])
        for g in self._serial:  # motor clipping changes the reachable segment orientation
            used[..., g.sem] = g.fwd(out)
        rep = SaturationReport(s, used, np.abs(s - used) > 1e-6, SEMANTIC_NAMES)
        self.last_report = rep
        return (out, rep) if return_report else out
