"""Build the ``lut2d`` ankle map: two calf motors (a, b) <-> (ankle_pitch, ankle_roll).

Forward table: measured semantic (pitch, roll) on the regular motor grid of the calibration sweep; grid
nodes where the physics did not reach the commanded motors, did not settle, or left a loop-closure gap
are *invalid* (kept in ``valid``) and filled with the nearest valid value so bilinear evaluation stays
defined. Inverse table: a regular (pitch, roll) grid; each node is solved for (a, b) by Newton steps on
the bilinear forward table and is valid only if it converges inside an all-valid motor cell.
Units: radians.
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.kinematics.semantic import _bilinear

DEG = np.pi / 180.0


def _fill_nearest(tab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = tab.copy()
    vi = np.argwhere(valid)
    for i, j in np.argwhere(~valid):
        k = np.argmin(((vi - [i, j]) ** 2).sum(-1))
        out[i, j] = tab[tuple(vi[k])]
    return out


def _newton(ag, bg, table, target, a0, b0, iters=30):
    a, b = a0.copy(), b0.copy()
    for _ in range(iters):
        a = np.clip(a, ag[0], ag[-1])
        b = np.clip(b, bg[0], bg[-1])
        val, da, db = _bilinear(ag, bg, table, a, b)
        err = target - val
        j00, j10, j01, j11 = da[..., 0], da[..., 1], db[..., 0], db[..., 1]
        det = j00 * j11 - j01 * j10
        ok = np.abs(det) > 1e-12
        det = np.where(ok, det, 1.0)
        sa = np.where(ok, (j11 * err[..., 0] - j01 * err[..., 1]) / det, 0.0)
        sb = np.where(ok, (-j10 * err[..., 0] + j00 * err[..., 1]) / det, 0.0)
        step = np.maximum(1.0, np.hypot(sa, sb) / (5 * DEG))  # trust region: <= 5 deg per step
        a, b = a + sa / step, b + sb / step
    a = np.clip(a, ag[0], ag[-1])
    b = np.clip(b, bg[0], bg[-1])
    val, _, _ = _bilinear(ag, bg, table, a, b)
    return a, b, np.abs(target - val).max(-1)


def build_pair(side: str, motors: tuple[str, str], a_grid: np.ndarray, b_grid: np.ndarray, pitch: np.ndarray,
               roll: np.ndarray, valid: np.ndarray, a_lim, b_lim, findings: list[str], n_inv: int = 41) -> dict:
    """Return the ``ankle_pairs[side]`` JSON block (see module doc)."""
    if valid.sum() < 4:
        raise RuntimeError(f"{side} ankle: fewer than 4 valid grid nodes")
    table = np.stack([_fill_nearest(pitch, valid), _fill_nearest(roll, valid)], -1)
    pr_valid = table[valid]
    p_rng = [float(pr_valid[:, 0].min()), float(pr_valid[:, 0].max())]
    r_rng = [float(pr_valid[:, 1].min()), float(pr_valid[:, 1].max())]
    pg = np.linspace(p_rng[0], p_rng[1], n_inv)
    rg = np.linspace(r_rng[0], r_rng[1], n_inv)
    tgt = np.stack(np.meshgrid(pg, rg, indexing="ij"), -1)  # (n, n, 2)
    # initial guesses: nearest valid forward node
    flat_t = tgt.reshape(-1, 2)
    vi = np.argwhere(valid)
    d2 = ((flat_t[:, None, :] - pr_valid[None, :, :]) ** 2).sum(-1)
    k = np.argmin(d2, axis=1)
    a0 = a_grid[vi[k, 0]]
    b0 = b_grid[vi[k, 1]]
    a, b, err = _newton(a_grid, b_grid, table, flat_t, a0, b0)
    # cell validity: all four corners of the motor cell containing (a, b) must be valid
    ia = np.clip(np.searchsorted(a_grid, a) - 1, 0, len(a_grid) - 2)
    ib = np.clip(np.searchsorted(b_grid, b) - 1, 0, len(b_grid) - 2)
    cell_ok = valid[ia, ib] & valid[ia + 1, ib] & valid[ia, ib + 1] & valid[ia + 1, ib + 1]
    ok = cell_ok & (err < 0.05 * DEG)
    inv_a = np.where(ok, a, np.nan).reshape(n_inv, n_inv)
    inv_b = np.where(ok, b, np.nan).reshape(n_inv, n_inv)
    inv_ok = ok.reshape(n_inv, n_inv)
    # local Jacobian at the grid centre-most valid node
    ci = vi[np.argmin(((vi - (np.array(valid.shape) - 1) / 2.0) ** 2).sum(-1))]
    _, da, db = _bilinear(a_grid, b_grid, table, a_grid[ci[0]], b_grid[ci[1]])
    jac = np.array([[da[0], db[0]], [da[1], db[1]]])
    info = {
        "grid_nodes": int(valid.size), "valid_nodes": int(valid.sum()),
        "invalid_fraction": float(1 - valid.mean()),
        "pitch_span_deg": float(np.degrees(p_rng[1] - p_rng[0])), "roll_span_deg": float(np.degrees(r_rng[1] - r_rng[0])),
        "inverse_valid_fraction": float(inv_ok.mean()),
        "jacobian_at_centre": {"motor_a": motors[0], "motor_b": motors[1], "a_rad": float(a_grid[ci[0]]),
                               "b_rad": float(b_grid[ci[1]]), "d_pitch_d_a": float(jac[0, 0]),
                               "d_pitch_d_b": float(jac[0, 1]), "d_roll_d_a": float(jac[1, 0]), "d_roll_d_b": float(jac[1, 1]),
                               "cond": float(np.linalg.cond(jac))},
    }
    if info["roll_span_deg"] < 2.0:
        findings.append(f"{side} ankle roll is essentially not controllable by the calf motors "
                        f"(roll span {info['roll_span_deg']:.2f} deg over the valid grid)")
    if info["invalid_fraction"] > 0.2:
        findings.append(f"{side} ankle: {100 * info['invalid_fraction']:.0f}% of the calf-motor grid is infeasible "
                        "(closure gap above the validity threshold, motor off target or still moving)")
    return {
        "motors": list(motors), "a_grid": a_grid.tolist(), "b_grid": b_grid.tolist(),
        "a_limits": [float(a_lim[0]), float(a_lim[1])], "b_limits": [float(b_lim[0]), float(b_lim[1])],
        "pitch": table[..., 0].tolist(), "roll": table[..., 1].tolist(), "valid": valid.tolist(),
        "pitch_range": p_rng, "roll_range": r_rng,
        "inverse": {"pitch_grid": pg.tolist(), "roll_grid": rg.tolist(),
                    "a": np.where(inv_ok, inv_a, np.nan).tolist(), "b": np.where(inv_ok, inv_b, np.nan).tolist(),
                    "valid": inv_ok.tolist()},
        "info": info,
    }
