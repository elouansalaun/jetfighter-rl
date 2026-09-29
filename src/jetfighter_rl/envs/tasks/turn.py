"""Task 8.3: coordinated turn at maximum **sustained** rate.

The sustained turn rate is the one that can be held without losing energy (thrust =
drag, in level flight). It peaks at a particular speed (≈ 250 m/s at 5,000 m for our
F-16). Pulling harder gives a higher instantaneous rate, but the aircraft slows down and
then can no longer hold altitude: over a 60 s episode, the best strategy is indeed the
optimal sustained turn.

Reference ω_ref(h): maximum over speed of ``performance.sustained_turn`` (point-mass
model, max thrust), cached in 250 m bands. It serves as the reference for both models.
The 6-DOF (aerodynamic tables different from the drag polar) holds about 90 % of this
reference with the reference pilot: its success thresholds are corrected by
``six_dof_factor`` (an approximation, lacking a 6-DOF turn trim).

Per-step reward: + turn rate / ω_ref (in the requested direction, capped at 1)
**multiplied** by a quality factor exp(−|Δh|/300 m − energy deficit/300 m), − energy loss
below the energy of the optimal turn (h₀ + V_opt²/2g), − altitude deviation,
− sideslip², − command jerks.

The multiplicative factor comes from an observed case of *reward hacking*: with purely
additive (and saturated) terms, the first trained agent had learned a **descending
spiral** (rate ×2.5, but 3 km of altitude lost) — once the penalties were saturated,
losing altitude cost nothing more.

Success: over the last ``window`` seconds, mean rate ≥ ``rate_fraction`` · ω_ref,
altitude within ± 200 m and energy at most 150 m below the reference energy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from jetsim.aircraft.dynamics_3dof import PointMassAircraft
from jetsim.aircraft.instruments import Instruments
from jetsim.aircraft.params import load_aircraft
from jetsim.aircraft.performance import sustained_turn
from jetsim.control.autopilot import Autopilot, AutopilotTargets
from jetsim.control.fbw import HighLevelCommand
from jetsim.core.constants import G0
from scipy.optimize import minimize_scalar

from jetfighter_rl.envs.rewards import RewardTerms, action_smoothness, angle_error, normalized_error
from jetfighter_rl.envs.tasks.base import DEG, FlightGeometry, InitialCondition, Task, Vec, clip


@dataclass(frozen=True)
class SustainedTurnReference:
    altitude: float
    turn_rate: float  # ω_ref [rad/s]
    speed: float  # V_opt [m/s]
    load_factor: float  # n at the optimal turn

    @property
    def bank(self) -> float:
        return math.acos(1.0 / self.load_factor)


@lru_cache(maxsize=1)
def _point_mass() -> PointMassAircraft:
    return PointMassAircraft(load_aircraft("f16"))


@lru_cache(maxsize=128)
def _reference_cached(altitude: float) -> SustainedTurnReference:
    ac = _point_mass()

    def neg_rate(v: float) -> float:
        perf = sustained_turn(ac, altitude, v)
        return 0.0 if perf is None else -perf.turn_rate

    speeds = np.arange(120.0, 361.0, 15.0)
    rates = [neg_rate(float(v)) for v in speeds]
    i = int(np.argmin(rates))
    lo, hi = speeds[max(i - 1, 0)], speeds[min(i + 1, len(speeds) - 1)]
    res = minimize_scalar(neg_rate, bounds=(lo, hi), method="bounded", options={"xatol": 0.5})
    v = float(res.x)
    perf = sustained_turn(ac, altitude, v)
    assert perf is not None
    return SustainedTurnReference(altitude, perf.turn_rate, v, perf.load_factor)


def sustained_turn_reference(altitude: float) -> SustainedTurnReference:
    """Optimal sustained turn at the given altitude (rounded to 250 m, cached)."""
    return _reference_cached(round(altitude / 250.0) * 250.0)


@dataclass
class SustainedTurnTask(Task):
    name: str = "sustained_turn"
    episode_time: float = 60.0
    window: float = 20.0  # evaluation window for the sustained rate [s]
    rate_fraction: float = 0.90
    altitude_tolerance: float = 200.0
    energy_tolerance: float = 150.0
    w_rate: float = 0.5
    w_energy: float = 0.3
    w_altitude: float = 0.3
    w_beta: float = 0.2
    w_smooth: float = 0.1
    six_dof_factor: float = 0.90  # 6-DOF sustained rate / point-mass reference (measured)
    six_dof_energy_tolerance: float = 250.0
    quality_scale: float = 300.0  # [m] decay of the rate gain with Δh and ΔE
    k_speed_bank: float = 0.2 * DEG  # reference pilot: bank per m/s of deviation
    n_features: int = 5
    direction: float = field(default=1.0, init=False)  # +1: right turn
    ref: SustainedTurnReference | None = field(default=None, init=False)
    h0: float = field(default=5000.0, init=False)
    turn_rate: float = field(default=0.0, init=False)
    six_dof: bool = field(default=False, init=False)
    _last: tuple[float, float] | None = field(default=None, init=False)
    _history: deque[tuple[float, float, float, float]] = field(default_factory=deque, init=False)

    @property
    def max_step_cost(self) -> float:
        return self.w_rate + self.w_energy + self.w_altitude + self.w_beta + self.w_smooth

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        h = float(rng.uniform(3000, 8000))
        v_opt = sustained_turn_reference(h).speed
        v_max = 260.0 if six_dof else 300.0
        return InitialCondition(
            altitude=h,
            airspeed=float(clip(rng.uniform(v_opt - 20, v_opt + 50), 150.0, v_max)),
            heading=float(rng.uniform(-math.pi, math.pi)),
        )

    @property
    def reference_energy(self) -> float:
        assert self.ref is not None
        return self.h0 + self.ref.speed**2 / (2 * G0)

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        self.direction = 1.0 if rng.random() < 0.5 else -1.0
        self.six_dof = six_dof
        self.h0 = ins.altitude
        self.ref = sustained_turn_reference(ins.altitude)
        self.turn_rate = 0.0
        self._last = (0.0, ins.course)
        self._history = deque()

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        assert self._last is not None
        t_prev, course_prev = self._last
        if t > t_prev:
            self.turn_rate = angle_error(ins.course - course_prev) / (t - t_prev)
        self._last = (t, ins.course)
        self._history.append(
            (t, self.direction * self.turn_rate, ins.altitude, ins.specific_energy)
        )
        while self._history and self._history[0][0] < t - self.window:
            self._history.popleft()

    def features(self, ins: Instruments) -> Vec:
        assert self.ref is not None
        return np.array(
            [
                clip(self.direction * self.turn_rate / self.ref.turn_rate, -2, 2),
                clip((ins.altitude - self.h0) / 300.0, -3, 3),
                clip((ins.specific_energy - self.reference_energy) / 300.0, -3, 3),
                clip((ins.tas - self.ref.speed) / 50.0, -3, 3),
                self.direction,
            ]
        )

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        assert self.ref is not None
        rate = clip(self.direction * self.turn_rate / self.ref.turn_rate, -1.0, 1.0)
        deficit = max(self.reference_energy - ins.specific_energy - 50.0, 0.0)
        dh = abs(ins.altitude - self.h0)
        # the turn rate only pays if it is "sustained": altitude and energy held
        quality = math.exp(-dh / self.quality_scale - deficit / self.quality_scale)
        return {
            "turn_rate": self.w_rate * rate * (quality if rate > 0 else 1.0),
            "energy": -self.w_energy * normalized_error(deficit, 2000.0),
            "altitude": -self.w_altitude * normalized_error(dh, 2000.0),
            "sideslip": -self.w_beta * min((ins.beta / (5 * DEG)) ** 2, 1.0),
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }

    def _window_stats(self) -> tuple[float, float, float]:
        """(mean rate, max altitude deviation, final energy deficit) over the window."""
        if not self._history:
            return 0.0, math.inf, math.inf
        rates = [h[1] for h in self._history]
        dh = max(abs(h[2] - self.h0) for h in self._history)
        deficit = self.reference_energy - self._history[-1][3]
        return float(np.mean(rates)), float(dh), float(deficit)

    def success(self, ins: Instruments) -> bool:
        assert self.ref is not None
        rate, dh, deficit = self._window_stats()
        fraction, e_tol = self.rate_fraction, self.energy_tolerance
        if self.six_dof:
            fraction, e_tol = fraction * self.six_dof_factor, self.six_dof_energy_tolerance
        return (
            rate >= fraction * self.ref.turn_rate
            and dh <= self.altitude_tolerance
            and deficit <= e_tol
        )

    def metrics(self) -> dict[str, float]:
        assert self.ref is not None
        rate, dh, deficit = self._window_stats()
        return {
            "turn_rate_deg_s": math.degrees(rate),
            "turn_rate_ratio": rate / self.ref.turn_rate,
            "max_altitude_error": dh,
            "energy_deficit": deficit,
        }

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        assert self.ref is not None
        return AutopilotTargets(
            altitude=self.h0, bank=self.direction * self.ref.bank, airspeed=self.ref.speed
        )

    def baseline_command(
        self, ins: Instruments, autopilot: Autopilot, dt: float
    ) -> HighLevelCommand:
        """Full throttle, altitude held with the load factor, speed held with the bank
        (slower than V_opt -> ease off the turn). The autopilot's load factor limit is
        raised to 8.8 g (the optimal turn needs up to ≈ 8 g at low altitude) and the bank
        is bounded accordingly."""
        assert self.ref is not None
        if autopilot.g.nz_max < 8.8:
            autopilot.g = replace(autopilot.g, nz_max=8.8)
        bank = clip(
            self.ref.bank + self.k_speed_bank * (ins.tas - self.ref.speed),
            45 * DEG,
            math.acos(1 / 8.3),
        )
        autopilot.targets = AutopilotTargets(
            altitude=self.h0, bank=self.direction * bank, throttle=1.0
        )
        return autopilot.high_level(ins, dt)

    def describe(self) -> dict[str, float]:
        assert self.ref is not None
        return {
            "direction": self.direction,
            "reference_turn_rate_deg_s": math.degrees(self.ref.turn_rate),
            "reference_speed": self.ref.speed,
            "reference_bank_deg": math.degrees(self.ref.bank),
        }
