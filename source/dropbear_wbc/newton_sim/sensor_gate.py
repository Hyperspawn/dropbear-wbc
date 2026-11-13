"""Policy-rate gate for :class:`~dropbear_wbc.newton_sim.sensor_model.SensorErrorModel`.

``SensorErrorModel`` ages are in POLICY ticks (default ``tick_dt=0.02`` = 50 Hz). Calling
``imu()`` / ``joints()`` once per Newton bridge tick (500 Hz) shrinks every age 10x. This
module advances the model once per policy step and reuses the last measured sensors between
those steps so sim2sim ages stay in the designed 7-34 ms band.
"""
from __future__ import annotations

import math

from dropbear_wbc.newton_sim.sensor_model import PROFILES, SensorErrorModel


def make_sensor_model(profile: str, seed: int | None = None, policy_hz: float = 50.0) -> SensorErrorModel:
    """Build a :class:`SensorErrorModel` for ``profile`` with ``tick_dt = 1/policy_hz``.

    Raises ``KeyError`` if ``profile`` is unknown (callers that want argparse-style errors
    should validate against ``sorted(PROFILES)`` first).
    """
    if policy_hz <= 0:
        raise ValueError(f"policy_hz must be > 0, got {policy_hz}")
    return SensorErrorModel(PROFILES[profile], tick_dt=1.0 / policy_hz, seed=seed)


def policy_steps_per_bridge(bridge_dt: float, policy_hz: float) -> int:
    """How many bridge ticks between model advances: ``round((1/policy_hz) / bridge_dt)``."""
    if bridge_dt <= 0 or policy_hz <= 0:
        raise ValueError(f"bridge_dt and policy_hz must be > 0, got {bridge_dt}, {policy_hz}")
    return max(1, round((1.0 / policy_hz) / bridge_dt))


def bridge_sensor_tick_error(sim_dt: float, substeps: int, policy_hz: float, rtol: float = 0.01) -> str | None:
    """CLI-time twin of the ``LowStateAssembler`` tick check (patch 0081), or ``None`` if consistent.

    The bridge tick is ``sim_dt * substeps``; the gated model advances every
    ``policy_steps_per_bridge(tick, policy_hz)`` ticks, so the held-sample age step is
    ``(1 / policy_hz) / steps_per_policy`` and must equal the tick within ``rtol``. Lets
    ``tools/newton_bridge.py`` reject e.g. ``--sim-dt 0.003`` (7 ticks = 21 ms != 20 ms) before
    the GPU lock and plant build instead of after them.
    """
    if not (sim_dt > 0 and math.isfinite(sim_dt)):
        return f"--sim-dt must be a finite value > 0, got {sim_dt}"
    if int(substeps) < 1:
        return f"--substeps must be >= 1, got {substeps}"
    if not policy_hz > 0:  # NaN fails the comparison; +inf gives a 0 ms age step and fails below
        return f"--sensor-policy-hz must be a finite value > 0, got {policy_hz}"
    tick = float(sim_dt) * int(substeps)
    spp = policy_steps_per_bridge(tick, policy_hz)
    eff = (1.0 / policy_hz) / spp
    if abs(eff - tick) > rtol * tick:
        return (f"--sim-dt {sim_dt} x --substeps {substeps} = {1e3 * tick:.3f} ms bridge tick does not divide the "
                f"{1e3 / policy_hz:.3f} ms sensor policy period (--sensor-policy-hz {policy_hz}): {spp} ticks = "
                f"{1e3 * spp * tick:.3f} ms, so every sensor age would be off by {100 * (eff / tick - 1):+.1f} %; "
                f"pick a tick that divides it (e.g. 2 ms or 4 ms at 50 Hz)")
    return None


class PolicyGatedSensorModel:
    """Adapter with the same ``imu`` / ``joints`` surface as ``SensorErrorModel``.

    Advances the wrapped model every ``steps_per_policy`` ``imu`` calls (one per
    ``LowStateAssembler.build``). Between advances, returns the last measured quat/gyro/accel
    and motor q/dq without pushing into the delay lines or walking bias. ``joints`` follows the
    advance decision made by the preceding ``imu`` in the same build.
    """

    def __init__(self, model: SensorErrorModel, steps_per_policy: int):
        if steps_per_policy < 1:
            raise ValueError(f"steps_per_policy must be >= 1, got {steps_per_policy}")
        self.model = model
        self.steps_per_policy = int(steps_per_policy)
        self._i = 0
        self._advance = True
        self._held_imu: tuple | None = None
        self._held_joints: tuple | None = None

    @property
    def ep(self):
        return self.model.ep

    def reset(self):
        out = self.model.reset()
        self._i = 0
        self._advance = True
        self._held_imu = None
        self._held_joints = None
        return out

    def imu(self, quat_wxyz, gyro_b, accel_b):
        self._advance = (self._i % self.steps_per_policy == 0) or self._held_imu is None
        if self._advance:
            self._held_imu = self.model.imu(quat_wxyz, gyro_b, accel_b)
        self._i += 1
        return self._held_imu

    def imu_age_s(self) -> float:
        """Modelled age of the held IMU output [s]: model lag + bridge ticks it has been held since the advance."""
        held = (self._i - 1) % self.steps_per_policy if self._i > 0 else 0
        return self.model.imu_age_s() + held * self.model.tick_dt / self.steps_per_policy

    def joints(self, q, dq):
        if self._advance or self._held_joints is None:
            self._held_joints = self.model.joints(q, dq)
        return self._held_joints
