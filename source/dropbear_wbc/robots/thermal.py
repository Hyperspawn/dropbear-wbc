"""Two-node (winding / housing) motor thermal model with an explicit ambient temperature. Pure numpy.

Replaces the ``rms <= rated`` thermal proxy of ``tools/scoreboard.py`` and ``tools/telemetry_issue_scan.py``.
Per motor, with torque ratio ``r = tau / rated``::

    P   = P_ref * r^2 * (1 + ALPHA_CU * (Tw - 25))          copper loss, resistance rises with winding temperature
    C_w dTw/dt = P - (Tw - Th) / R_wh
    C_h dTh/dt = (Tw - Th) / R_wh - (Th - ambient_C) / R_ha

``R_wh + R_ha = 1`` (normalised), ``C = tau / R``, and ``P_ref`` is set so that the steady state at ``r = 1`` and the
datasheet reference ambient (24 C) rises exactly ``rise_at_rated_K``. The trip is the latched winding temperature
``Tw >= t_trip_C``.

Steady state (closed form): ``Tw = (Ta + P_ref r^2 (1 - 25 ALPHA_CU)) / (1 - P_ref r^2 ALPHA_CU)`` (runaway when the
denominator is <= 0). Max continuous ratio (steady ``Tw`` = trip): ``r^2 = (T_trip - Ta) / (P_ref (1 + ALPHA_CU
(T_trip - 25)))``. With the defaults: 1.09 / 1.02 / 0.93 at 25 / 35 / 45 C; rated (1.0) is not continuous above
36.9 C ambient.

ASSUMED constants (no motor on the robot publishes them; replace per model once measured): ``tau_w_s`` 40 s,
``tau_h_s`` 900 s, ``split_wh`` 0.3, ``t_trip_C`` 100 C, and the 60 K rise at rated (MyActuator V3 datasheet
"thermal balance point", 24 C ambient) applied to every model. ``ambient_C`` has no default on purpose: every caller
states the room / enclosure temperature it assumes. ``DESIGN_AMBIENT_C`` (45 C) is the conservative default of the
command-line tools (Indian summer, enclosed hip), ASSUMED pending a design decision.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

ALPHA_CU = 0.00393  # 1/K, copper resistance tempco
REF_AMBIENT_C = 24.0  # datasheet thermal balance point ambient
DESIGN_AMBIENT_C = 45.0  # ASSUMED design ambient of the gates (pending decision)
THERMAL_ASSUMED = {"rise_at_rated_K": 60.0, "split_wh": 0.3, "tau_w_s": 40.0, "tau_h_s": 900.0, "t_trip_C": 100.0}
# Arm-duty screen ``sus`` HIGH (item 363): the trip time of 1.3x rated held from a soak at 25 C (17.35 min); at another
# ambient HIGH is the ratio with that same trip time. Lives here (not in tools/motion_duty_screen.py) since item 368 so
# the hw-library gate can recompute what a screen made now would record.
SUS_HIGH_REF = (1.3, 25.0)  # (ratio, ambient C) defining sus_high_trip_s()
SUS_TRIP_DT = 0.05


@dataclass
class ThermalModel:
    """Constants of the two-node law; ``ambient_C`` is required."""

    ambient_C: float
    rise_at_rated_K: float = THERMAL_ASSUMED["rise_at_rated_K"]
    split_wh: float = THERMAL_ASSUMED["split_wh"]
    tau_w_s: float = THERMAL_ASSUMED["tau_w_s"]
    tau_h_s: float = THERMAL_ASSUMED["tau_h_s"]
    t_trip_C: float = THERMAL_ASSUMED["t_trip_C"]
    P_ref: float = field(init=False)
    R_wh: float = field(init=False)
    R_ha: float = field(init=False)
    C_w: float = field(init=False)
    C_h: float = field(init=False)

    def __post_init__(self):
        if self.ambient_C is None or not math.isfinite(float(self.ambient_C)):
            raise ValueError("ambient_C must be a finite temperature in C")
        if not 0.0 < self.split_wh < 1.0:
            raise ValueError("split_wh must be in (0, 1)")
        self.ambient_C = float(self.ambient_C)
        t_ref = REF_AMBIENT_C + self.rise_at_rated_K
        self.P_ref = self.rise_at_rated_K / (1.0 + ALPHA_CU * (t_ref - 25.0))
        self.R_wh = self.split_wh
        self.R_ha = 1.0 - self.split_wh
        self.C_w = self.tau_w_s / self.R_wh
        self.C_h = self.tau_h_s / self.R_ha

    # ---- closed forms ----
    def steady_winding_C(self, ratio) -> np.ndarray:
        """Steady winding temperature for a constant (or cycle-RMS) torque ratio; inf = thermal runaway."""
        p = self.P_ref * np.asarray(ratio, float) ** 2
        den = 1.0 - p * ALPHA_CU
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = (self.ambient_C + p * (1.0 - 25.0 * ALPHA_CU)) / den
        return np.where(den > 0, tw, np.inf)

    def steady_housing_C(self, ratio) -> np.ndarray:
        tw = self.steady_winding_C(ratio)
        p = self.P_ref * np.asarray(ratio, float) ** 2 * (1.0 + ALPHA_CU * (tw - 25.0))
        return np.where(np.isfinite(tw), self.ambient_C + p * self.R_ha, np.inf)

    def max_continuous_ratio(self) -> float:
        """Largest constant torque / rated whose steady winding temperature stays below the trip (0 if ambient >= trip)."""
        num = self.t_trip_C - self.ambient_C
        if num <= 0:
            return 0.0
        return math.sqrt(num / (self.P_ref * (1.0 + ALPHA_CU * (self.t_trip_C - 25.0))))

    def ambient_for_continuous(self, ratio: float) -> float:
        """Highest ambient at which ``ratio`` x rated is still continuous (steady winding == trip)."""
        return self.t_trip_C - ratio ** 2 * self.P_ref * (1.0 + ALPHA_CU * (self.t_trip_C - 25.0))

    # ---- integration ----
    def step(self, Tw: np.ndarray, Th: np.ndarray, ratio: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """One explicit-Euler step (the notes/36 twin_v2 ThermalState law)."""
        P = self.P_ref * ratio * ratio * (1.0 + ALPHA_CU * (Tw - 25.0))
        q_wh = (Tw - Th) / self.R_wh
        Tw_n = Tw + dt * (P - q_wh) / self.C_w
        Th_n = Th + dt * (q_wh - (Th - self.ambient_C) / self.R_ha) / self.C_h
        return Tw_n, Th_n

    def simulate(self, ratio: np.ndarray, dt: float, *, T0_C=None, loop_s: float = 0.0, record: bool = True) -> dict:
        """Run a (T, n) torque-ratio trace from ``T0_C`` (default: soaked at ambient).

        ``loop_s`` > 0 repeats the trace end to end until that many seconds (the duty held indefinitely).
        Returns ``tw`` (recorded winding trace over the first pass, (T, n), if ``record``), ``tw_end`` / ``th_end``,
        ``tw_max`` and ``trip_s`` (first time ``Tw >= t_trip_C`` per motor, inf = no trip within the run)."""
        r = np.atleast_2d(np.asarray(ratio, float))
        if r.shape[0] == 1 and np.asarray(ratio).ndim == 1:
            r = r.reshape(-1, 1)
        T, n = r.shape
        if T == 0:
            raise ValueError("empty trace")
        T0 = self.ambient_C if T0_C is None else T0_C
        Tw = np.full(n, 1.0) * T0
        Th = np.full(n, 1.0) * T0
        trip = np.full(n, np.inf)
        tw_max = Tw.copy()
        rec = np.empty((T, n)) if record else None
        total = max(T, int(math.ceil(loop_s / dt))) if loop_s > 0 else T
        for k in range(total):
            Tw, Th = self.step(Tw, Th, r[k % T], dt)
            if rec is not None and k < T:
                rec[k] = Tw
            np.maximum(tw_max, Tw, out=tw_max)
            new = (Tw >= self.t_trip_C) & ~np.isfinite(trip)
            if new.any():
                trip[new] = (k + 1) * dt
        return {"tw": rec, "tw_end": Tw, "th_end": Th, "tw_max": tw_max, "trip_s": trip}

    def time_to_trip(self, ratio: float, *, T0_C=None, dt: float = 0.05, t_max: float = 3600.0,
                     interpolate: bool = False) -> float:
        """Constant ``ratio`` x rated from ``T0_C`` (default ambient): seconds until the trip, inf if none by ``t_max``.

        ``interpolate`` returns the linearly interpolated crossing inside the step (continuous and strictly monotone in
        ``ratio``, which :meth:`ratio_for_time_to_trip` needs) instead of the end of the step that crossed."""
        T0 = self.ambient_C if T0_C is None else T0_C
        if interpolate:
            return self._trip_scalar(float(ratio), float(T0), dt, t_max)
        Tw, Th = np.array([float(T0)]), np.array([float(T0)])
        r = np.array([float(ratio)])
        for k in range(int(math.ceil(t_max / dt))):
            Tw, Th = self.step(Tw, Th, r, dt)
            if Tw[0] >= self.t_trip_C:
                return (k + 1) * dt
        return math.inf

    def _trip_scalar(self, r: float, T0: float, dt: float, t_max: float) -> float:
        """Scalar copy of :meth:`step` (pure floats, ~50x faster) with an interpolated trip crossing."""
        Tw = Th = T0
        if Tw >= self.t_trip_C:
            return 0.0
        pr = self.P_ref * r * r
        for k in range(int(math.ceil(t_max / dt))):
            q_wh = (Tw - Th) / self.R_wh
            Tw_n = Tw + dt * (pr * (1.0 + ALPHA_CU * (Tw - 25.0)) - q_wh) / self.C_w
            Th = Th + dt * (q_wh - (Th - self.ambient_C) / self.R_ha) / self.C_h
            if Tw_n >= self.t_trip_C:
                return (k + (self.t_trip_C - Tw) / (Tw_n - Tw)) * dt
            Tw = Tw_n
        return math.inf

    def ratio_for_time_to_trip(self, t_s: float, *, T0_C=None, dt: float = 0.05, tol: float = 1e-7) -> float:
        """Constant torque ratio that reaches the trip in exactly ``t_s`` seconds from ``T0_C`` (default: soaked at
        ambient); bisection on the interpolated :meth:`time_to_trip`. Always above :meth:`max_continuous_ratio`."""
        if not (t_s > 0 and math.isfinite(t_s)):
            raise ValueError("t_s must be finite and > 0")
        T0 = self.ambient_C if T0_C is None else float(T0_C)
        t_max = 1.5 * t_s + dt

        def ttt(r):
            return self._trip_scalar(r, T0, dt, t_max)

        lo = self.max_continuous_ratio()
        hi = max(2.0 * lo, 0.1)
        while ttt(hi) > t_s:
            lo, hi = hi, 2.0 * hi
            if hi > 1e3:
                raise ValueError(f"no ratio trips within {t_s} s")
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if ttt(mid) > t_s:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def describe(self) -> dict:
        return {"ambient_C": self.ambient_C, "rise_at_rated_K": self.rise_at_rated_K, "ref_ambient_C": REF_AMBIENT_C,
                "split_wh": self.split_wh, "tau_w_s": self.tau_w_s, "tau_h_s": self.tau_h_s, "t_trip_C": self.t_trip_C,
                "max_continuous_ratio": round(self.max_continuous_ratio(), 4),
                "assumed": "tau_w/tau_h/split/trip and the 60 K rise are ASSUMED; ambient_C is the caller's choice"}


def thermal_trace_check(tau: np.ndarray, rated: np.ndarray, dt: float, ambient_C: float, *,
                        loop_s: float = 3600.0, **kw) -> dict:
    """Per-motor thermal verdict of a (T, n) motor torque trace held indefinitely at ``ambient_C``.

    ``value`` = (trace RMS / rated) / max continuous ratio: >= 1 means the duty repeated forever reaches the trip.
    ``trip_s`` loops the trace from ambient for ``loop_s`` seconds."""
    m = ThermalModel(ambient_C, **kw)
    tau = np.asarray(tau, float)
    rated = np.asarray(rated, float)
    ratio = tau / rated
    rr = np.sqrt((ratio ** 2).mean(0))
    lim = m.max_continuous_ratio()
    sim = m.simulate(ratio, dt, loop_s=loop_s, record=False)
    with np.errstate(divide="ignore"):
        value = rr / lim if lim > 0 else np.full_like(rr, np.inf)
    return {"model": m.describe(), "rms_over_rated": rr, "value": value, "steady_winding_C": m.steady_winding_C(rr),
            "trip_s": sim["trip_s"], "tw_max_C": sim["tw_max"], "loop_s": loop_s}


# ---------------------------------------------------------------- arm-duty screen thresholds (items 363, 368)
@functools.lru_cache(maxsize=None)
def sus_high_trip_s() -> float:
    """Seconds for ``SUS_HIGH_REF`` ratio held from a soak at its ambient to reach the trip (1041.0 s)."""
    r, amb = SUS_HIGH_REF
    return ThermalModel(amb).time_to_trip(r, dt=SUS_TRIP_DT, interpolate=True)


@functools.lru_cache(maxsize=None)
def duty_sus_thresholds(ambient_C: float) -> tuple[float, float]:
    """``(warn, HIGH)`` dynamic ``sus`` thresholds of the arm-duty screen at ``ambient_C`` (rounded to 1e-4):
    warn = max continuous ratio, HIGH = ratio that trips in :func:`sus_high_trip_s` from a soak at ``ambient_C``."""
    m = ThermalModel(float(ambient_C))
    warn = m.max_continuous_ratio()
    high = m.ratio_for_time_to_trip(sus_high_trip_s(), dt=SUS_TRIP_DT) if warn > 0 else 0.0
    return round(warn, 4), round(high, 4)


def duty_screen_thermal(ambient_C: float) -> dict:
    """The ``thermal`` block a ``motion_duty_screen`` result records at ``ambient_C`` (schema v2)."""
    return {**ThermalModel(float(ambient_C)).describe(), "sus_warn": "max continuous ratio at ambient",
            "sus_high": f"ratio tripping in sus_high_trip_s from a soak at ambient (= {SUS_HIGH_REF[0]}x at "
                        f"{SUS_HIGH_REF[1]:g} C)", "sus_high_trip_s": round(sus_high_trip_s(), 3)}
