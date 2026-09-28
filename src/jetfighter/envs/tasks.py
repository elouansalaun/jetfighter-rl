"""Tâches d'apprentissage : conditions initiales, consignes, observations et récompenses.

Une tâche dit **quoi** apprendre ; l'environnement (``jet_env.py``) dit **comment** l'avion
vole. Chaque tâche fournit :

* ``sample_initial(rng)`` : conditions initiales aléatoires (altitude, vitesse, attitude…) ;
* ``reset(ins, rng)`` : tire les consignes de l'épisode ;
* ``features(ins)`` : observations propres à la tâche (erreurs **relatives** à la consigne,
  pour que la politique soit invariante par rotation autour de la verticale) ;
* ``reward(ins, action, previous_action)`` : termes de récompense nommés ;
* ``success(ins)`` : l'avion est-il dans la tolérance à cet instant ?
* ``autopilot_targets(ins)`` : les consignes équivalentes pour le pilote automatique
  (politique de référence, cf. ``baselines.py``).

Tâches disponibles (phase 8 de la roadmap) :

* ``"level"`` (8.1) — **stabilisation** : partir d'une attitude perturbée (inclinaison jusqu'à
  ±80°, pente ±25°, roulis en cours) et revenir en palier ailes horizontales ;
* ``"heading_altitude"`` (8.2) — **changement de cap, d'altitude et de vitesse** vers des
  consignes tirées au hasard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jetfighter.aircraft.instruments import Instruments
from jetfighter.control.autopilot import AutopilotTargets
from jetfighter.envs.rewards import RewardTerms, action_smoothness, angle_error, normalized_error

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180


@dataclass(frozen=True)
class InitialCondition:
    altitude: float  # [m]
    airspeed: float  # [m/s]
    heading: float = 0.0  # route χ [rad]
    gamma: float = 0.0  # pente [rad]
    bank: float = 0.0  # inclinaison μ [rad]
    roll_rate: float = 0.0  # [rad/s]


class Task:
    """Interface commune des tâches (voir le module)."""

    name: str = "task"
    episode_time: float = 30.0  # durée max d'un épisode [s]
    crash_penalty: float = -50.0
    n_features: int = 0

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        raise NotImplementedError

    def reset(self, ins: Instruments, rng: np.random.Generator, six_dof: bool) -> None:
        """Tire les consignes de l'épisode (après construction de l'état initial)."""

    def features(self, ins: Instruments) -> Vec:
        return np.zeros(0)

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        raise NotImplementedError

    def success(self, ins: Instruments) -> bool:
        raise NotImplementedError

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        raise NotImplementedError

    def describe(self) -> dict[str, float]:
        """Consignes de l'épisode (pour les journaux)."""
        return {}


# --------------------------------------------------------------------------
# 8.1 — Stabilisation
# --------------------------------------------------------------------------
@dataclass
class LevelFlightTask(Task):
    """Revenir en palier ailes horizontales depuis une attitude perturbée."""

    name: str = "level"
    episode_time: float = 30.0
    max_initial_bank: float = 80 * DEG
    max_initial_gamma: float = 25 * DEG
    max_initial_roll_rate: float = 60 * DEG
    bank_tolerance: float = 5 * DEG
    gamma_tolerance: float = 2 * DEG
    w_bank: float = 0.5
    w_gamma: float = 0.5
    w_smooth: float = 0.1
    success_bonus: float = 0.1
    n_features: int = 0
    reference_speed: float = field(default=200.0, init=False)

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

    def reset(self, ins: Instruments, rng: np.random.Generator, six_dof: bool) -> None:
        self.reference_speed = ins.tas

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        terms = {
            "bank": -self.w_bank * abs(angle_error(ins.bank)) / math.pi,
            "gamma": -self.w_gamma * normalized_error(ins.gamma, 20 * DEG),
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }
        terms["success"] = self.success_bonus if self.success(ins) else 0.0
        return terms

    def success(self, ins: Instruments) -> bool:
        return (abs(angle_error(ins.bank)) < self.bank_tolerance
                and abs(ins.gamma) < self.gamma_tolerance)  # fmt: skip

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        # altitude = altitude courante -> vitesse verticale nulle -> pente nulle
        return AutopilotTargets(altitude=ins.altitude, bank=0.0, airspeed=self.reference_speed)


# --------------------------------------------------------------------------
# 8.2 — Cap, altitude, vitesse
# --------------------------------------------------------------------------
@dataclass
class HeadingAltitudeTask(Task):
    """Rejoindre un cap, une altitude et une vitesse tirés au hasard."""

    name: str = "heading_altitude"
    episode_time: float = 90.0
    max_altitude_change: float = 1500.0
    altitude_tolerance: float = 30.0
    heading_tolerance: float = 3 * DEG
    speed_tolerance: float = 5.0
    w_altitude: float = 0.4
    w_heading: float = 0.4
    w_speed: float = 0.2
    w_smooth: float = 0.1
    success_bonus: float = 0.2
    n_features: int = 4
    target_altitude: float = field(default=5000.0, init=False)
    target_heading: float = field(default=0.0, init=False)
    target_speed: float = field(default=220.0, init=False)

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        v_max = 260.0 if six_dof else 270.0
        return InitialCondition(
            altitude=float(rng.uniform(2000, 9000)),
            airspeed=float(rng.uniform(170, v_max)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            gamma=float(rng.uniform(-5, 5) * DEG),
            bank=float(rng.uniform(-10, 10) * DEG),
        )

    def reset(self, ins: Instruments, rng: np.random.Generator, six_dof: bool) -> None:
        dh = rng.uniform(-1, 1) * self.max_altitude_change
        self.target_altitude = float(min(max(ins.altitude + dh, 1000.0), 12_000.0))
        self.target_heading = float(rng.uniform(-math.pi, math.pi))
        self.target_speed = float(rng.uniform(180, 250 if six_dof else 270))

    def errors(self, ins: Instruments) -> tuple[float, float, float]:
        """(altitude, cap, vitesse) : consigne − mesure."""
        return (self.target_altitude - ins.altitude,
                angle_error(self.target_heading - ins.course),
                self.target_speed - ins.tas)  # fmt: skip

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
        terms["success"] = self.success_bonus if self.success(ins) else 0.0
        return terms

    def success(self, ins: Instruments) -> bool:
        eh, echi, ev = self.errors(ins)
        return (abs(eh) < self.altitude_tolerance and abs(echi) < self.heading_tolerance
                and abs(ev) < self.speed_tolerance)  # fmt: skip

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        return AutopilotTargets(altitude=self.target_altitude, heading=self.target_heading,
                                airspeed=self.target_speed)  # fmt: skip

    def describe(self) -> dict[str, float]:
        return {
            "target_altitude": self.target_altitude,
            "target_heading": self.target_heading,
            "target_speed": self.target_speed,
        }


TASKS: dict[str, type[Task]] = {
    "level": LevelFlightTask,
    "heading_altitude": HeadingAltitudeTask,
}


def make_task(name: str, **kwargs: float) -> Task:
    if name not in TASKS:
        raise ValueError(f"Tâche inconnue : {name!r} (disponibles : {sorted(TASKS)})")
    return TASKS[name](**kwargs)
