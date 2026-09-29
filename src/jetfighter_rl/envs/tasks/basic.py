"""Tâches 8.1 (stabilisation) et 8.2 (cap, altitude, vitesse)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from jetsim.aircraft.instruments import Instruments
from jetsim.control.autopilot import AutopilotTargets

from jetfighter_rl.envs.rewards import RewardTerms, action_smoothness, angle_error, normalized_error
from jetfighter_rl.envs.tasks.base import DEG, FlightGeometry, InitialCondition, Task, Vec


# --------------------------------------------------------------------------
# 8.1 — Stabilisation
# --------------------------------------------------------------------------
@dataclass
class LevelFlightTask(Task):
    """Revenir en palier ailes horizontales depuis une attitude perturbée.

    La vitesse initiale doit être conservée (petit terme ``speed``) : sans lui, les
    premiers agents entraînés rétablissaient très vite puis laissaient la vitesse décroître
    indéfiniment, manette réduite (vu sur les trajectoires, pas sur la récompense).

    Critère de la roadmap : **retour en palier en moins de 10 s**. Le temps de
    rétablissement est l'instant où l'avion entre dans la tolérance (|μ| < 5°, |γ| < 2°)
    pour y rester au moins ``hold_time`` secondes. Réussite : rétabli en moins de
    ``max_recovery_time`` et toujours dans la tolérance à la fin de l'épisode.
    """

    name: str = "level"
    episode_time: float = 30.0
    max_initial_bank: float = 80 * DEG
    max_initial_gamma: float = 25 * DEG
    max_initial_roll_rate: float = 60 * DEG
    bank_tolerance: float = 5 * DEG
    gamma_tolerance: float = 2 * DEG
    hold_time: float = 2.0
    max_recovery_time: float = 10.0
    w_bank: float = 0.5
    w_gamma: float = 0.5
    w_speed: float = 0.1
    w_smooth: float = 0.1
    success_bonus: float = 0.1
    n_features: int = 0
    reference_speed: float = field(default=200.0, init=False)
    recovery_time: float = field(default=math.inf, init=False)
    _entered: float | None = field(default=None, init=False)

    @property
    def max_step_cost(self) -> float:
        return self.w_bank + self.w_gamma + self.w_speed + self.w_smooth

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        v_max = 260.0 if six_dof else 280.0
        return InitialCondition(
            altitude=float(rng.uniform(3000, 8000)),
            airspeed=float(rng.uniform(160, v_max)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            gamma=float(rng.uniform(-1, 1) * self.max_initial_gamma),
            bank=float(rng.uniform(-1, 1) * self.max_initial_bank),
            roll_rate=float(rng.uniform(-1, 1) * self.max_initial_roll_rate),
        )

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        self.reference_speed = ins.tas
        self.recovery_time = math.inf
        self._entered = 0.0 if self.success(ins) else None

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        if not self.success(ins):
            self._entered = None
            return
        if self._entered is None:
            self._entered = t
        if t - self._entered >= self.hold_time - 1e-9 and math.isinf(self.recovery_time):
            self.recovery_time = self._entered

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        terms = {
            "bank": -self.w_bank * abs(angle_error(ins.bank)) / math.pi,
            "gamma": -self.w_gamma * normalized_error(ins.gamma, 20 * DEG),
            "speed": -self.w_speed * normalized_error(ins.tas - self.reference_speed, 50.0),
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }
        terms["success"] = self.success_bonus if self.success(ins) else 0.0
        return terms

    def success(self, ins: Instruments) -> bool:
        return (abs(angle_error(ins.bank)) < self.bank_tolerance
                and abs(ins.gamma) < self.gamma_tolerance)  # fmt: skip

    def episode_success(self, ins: Instruments) -> bool:
        return self.recovery_time <= self.max_recovery_time and self.success(ins)

    def metrics(self) -> dict[str, float]:
        # non rétabli : on reporte la durée de l'épisode (valeur finie pour TensorBoard)
        rec = self.recovery_time if math.isfinite(self.recovery_time) else self.episode_time
        return {"recovery_time": rec}

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        # altitude = altitude courante -> vitesse verticale nulle -> pente nulle
        return AutopilotTargets(altitude=ins.altitude, bank=0.0, airspeed=self.reference_speed)


# --------------------------------------------------------------------------
# 8.2 — Cap, altitude, vitesse
# --------------------------------------------------------------------------
@dataclass
class HeadingAltitudeTask(Task):
    """Rejoindre un cap, une altitude et une vitesse tirés au hasard.

    Récompense : coûts proportionnels aux erreurs (ce qu'on veut minimiser) **plus** un
    terme de progression ΔΦ (réduction de l'erreur normalisée à chaque pas, *shaping*
    potentiel). Sans ce dernier, l'effet d'une action sur le coût est minuscule à l'échelle
    d'un pas : le premier agent entraîné n'avait rien appris en 300 000 pas.

    Critère de la roadmap : **erreur faible sans dépassement excessif**. Réussite : dans la
    tolérance à la fin (30 m, 3°, 5 m/s, soit ≈ 2–5 % des changements demandés), avec un
    dépassement d'altitude ≤ max(50 m, 10 % du changement) et de cap ≤ 5°.
    """

    name: str = "heading_altitude"
    episode_time: float = 90.0
    max_altitude_change: float = 1500.0
    altitude_tolerance: float = 30.0
    heading_tolerance: float = 3 * DEG
    speed_tolerance: float = 5.0
    max_heading_overshoot: float = 5 * DEG
    altitude_overshoot_ratio: float = 0.10
    min_altitude_overshoot: float = 50.0
    w_altitude: float = 0.4
    w_heading: float = 0.4
    w_speed: float = 0.2
    w_smooth: float = 0.1
    w_progress: float = 20.0  # shaping : réduction de l'erreur normalisée
    success_bonus: float = 0.2
    n_features: int = 4
    target_altitude: float = field(default=5000.0, init=False)
    target_heading: float = field(default=0.0, init=False)
    target_speed: float = field(default=220.0, init=False)
    _initial_errors: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _overshoot: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _settle_time: float = field(default=0.0, init=False)
    _potential: float = field(default=0.0, init=False)

    @property
    def max_step_cost(self) -> float:
        # la progression peut être négative : on s'éloigne d'au plus ≈ 0.02 par pas
        return (
            self.w_altitude + self.w_heading + self.w_speed + self.w_smooth + 0.02 * self.w_progress
        )

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        v_max = 260.0 if six_dof else 270.0
        return InitialCondition(
            altitude=float(rng.uniform(2000, 9000)),
            airspeed=float(rng.uniform(170, v_max)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            gamma=float(rng.uniform(-5, 5) * DEG),
            bank=float(rng.uniform(-10, 10) * DEG),
        )

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        dh = rng.uniform(-1, 1) * self.max_altitude_change
        self.target_altitude = float(min(max(ins.altitude + dh, 1000.0), 12_000.0))
        self.target_heading = float(rng.uniform(-math.pi, math.pi))
        self.target_speed = float(rng.uniform(180, 250 if six_dof else 270))
        self._initial_errors = self.errors(ins)
        self._overshoot = [0.0, 0.0]
        self._settle_time = 0.0
        self._potential = self.potential(ins)

    def errors(self, ins: Instruments) -> tuple[float, float, float]:
        """(altitude, cap, vitesse) : consigne − mesure."""
        return (self.target_altitude - ins.altitude,
                angle_error(self.target_heading - ins.course),
                self.target_speed - ins.tas)  # fmt: skip

    def potential(self, ins: Instruments) -> float:
        """Φ = −(erreurs normalisées) : la récompense de progression est ΔΦ."""
        eh, echi, ev = self.errors(ins)
        return -(min(abs(eh) / 1000.0, 3.0) + abs(echi) / math.pi + min(abs(ev) / 50.0, 3.0))

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        errors = self.errors(ins)
        for k in range(2):  # dépassement = erreur de signe opposé à l'erreur initiale
            e0 = self._initial_errors[k]
            if e0 != 0.0:
                self._overshoot[k] = max(self._overshoot[k], -math.copysign(1.0, e0) * errors[k])
        if not self.success(ins):
            self._settle_time = t

    def features(self, ins: Instruments) -> Vec:
        eh, echi, ev = self.errors(ins)
        return np.array([np.clip(eh / 1000.0, -3, 3), math.sin(echi), math.cos(echi),
                         np.clip(ev / 50.0, -3, 3)])  # fmt: skip

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        eh, echi, ev = self.errors(ins)
        terms = {
            "altitude": -self.w_altitude * normalized_error(eh, 1000.0),
            "heading": -self.w_heading * abs(echi) / math.pi,
            "speed": -self.w_speed * normalized_error(ev, 50.0),
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }
        phi = self.potential(ins)
        terms["progress"] = self.w_progress * (phi - self._potential)
        self._potential = phi
        terms["success"] = self.success_bonus if self.success(ins) else 0.0
        return terms

    def success(self, ins: Instruments) -> bool:
        eh, echi, ev = self.errors(ins)
        return (abs(eh) < self.altitude_tolerance and abs(echi) < self.heading_tolerance
                and abs(ev) < self.speed_tolerance)  # fmt: skip

    def episode_success(self, ins: Instruments) -> bool:
        max_alt = max(self.min_altitude_overshoot,
                      self.altitude_overshoot_ratio * abs(self._initial_errors[0]))  # fmt: skip
        return (self.success(ins) and self._overshoot[0] <= max_alt
                and self._overshoot[1] <= self.max_heading_overshoot)  # fmt: skip

    def metrics(self) -> dict[str, float]:
        return {
            "altitude_overshoot": self._overshoot[0],
            "heading_overshoot_deg": math.degrees(self._overshoot[1]),
            "settle_time": self._settle_time,
        }

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        return AutopilotTargets(altitude=self.target_altitude, heading=self.target_heading,
                                airspeed=self.target_speed)  # fmt: skip

    def describe(self) -> dict[str, float]:
        return {
            "target_altitude": self.target_altitude,
            "target_heading": self.target_heading,
            "target_speed": self.target_speed,
        }
