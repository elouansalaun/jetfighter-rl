"""Interface commune des tâches d'apprentissage.

Une tâche dit **quoi** apprendre ; l'environnement (``jet_env.py``) dit **comment** l'avion
vole. Cycle de vie d'un épisode, vu de la tâche ::

    ic = task.sample_initial(rng, six_dof)        # conditions initiales
    task.reset(ins, geo, rng, six_dof)            # consignes de l'épisode
    boucle :
        features = task.features(mesures)         # observations propres à la tâche
        ... l'agent agit, l'avion vole pendant agent_dt ...
        task.update(ins, geo, t)                  # suivi (chronos, progression, waypoints…)
        terms = task.reward(ins, action, action_précédente)
        fin si task.done() (tâche accomplie) ou sortie d'enveloppe ou durée maximale
    info["is_success"] = task.episode_success(ins) ; info["metrics"] = task.metrics()

``ins`` : instruments **vrais** ; ``geo`` : position et repère vent (sans singularité),
vrais également. Seul ``features`` reçoit les **mesures** (capteurs éventuellement bruités).

La politique de référence (``baselines.AutopilotPolicy``) appelle ``baseline_command`` :
par défaut, le pilote automatique de la phase 6 avec les consignes ``autopilot_targets`` ;
les tâches de voltige la remplacent par un pilote scripté.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from jetsim.aircraft.instruments import Instruments
from jetsim.control.autopilot import Autopilot, AutopilotTargets
from jetsim.control.fbw import HighLevelCommand

from jetfighter_rl.envs.rewards import RewardTerms

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


@dataclass(frozen=True)
class FlightGeometry:
    """Position et orientation du repère vent, sans singularité (cf. ``wind_axes``)."""

    position: Vec  # [nord, est, bas] [m]
    c_nw: Vec  # repère vent -> NED

    @property
    def velocity_dir(self) -> Vec:
        return self.c_nw[:, 0]

    @property
    def wing_dir(self) -> Vec:
        return self.c_nw[:, 1]

    @property
    def lift_dir(self) -> Vec:
        return -self.c_nw[:, 2]


class Task:
    """Interface commune des tâches (voir le module)."""

    name: str = "task"
    episode_time: float = 30.0  # durée max d'un épisode [s]
    crash_penalty: float = -50.0
    n_features: int = 0
    # termes ponctuels (bonus, progression) : indépendants de la cadence de l'agent ; les
    # autres sont des coûts par pas, mis à l'échelle agent_dt / 0.1 s par l'environnement
    event_terms: frozenset[str] = frozenset({"progress", "waypoint"})

    @property
    def max_step_cost(self) -> float:
        """Borne du coût par pas (somme des poids des termes négatifs). Sert à rendre la
        pénalité de crash toujours pire que de continuer à voler (voir ``JetEnv.step``)."""
        return 1.0

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        raise NotImplementedError

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        """Tire les consignes de l'épisode (après construction de l'état initial)."""

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        """Suivi interne après chaque pas de l'agent (avant ``reward``)."""

    def features(self, ins: Instruments) -> Vec:
        return np.zeros(0)

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        raise NotImplementedError

    def success(self, ins: Instruments) -> bool:
        """L'avion est-il dans la tolérance à cet instant ?"""
        raise NotImplementedError

    def done(self) -> bool:
        """Tâche accomplie : l'épisode se termine (``terminated``) sans pénalité."""
        return False

    def episode_success(self, ins: Instruments) -> bool:
        """Critère de réussite de l'épisode (roadmap, phase 8), évalué à la fin."""
        return self.success(ins)

    def metrics(self) -> dict[str, float]:
        """Mesures de fin d'épisode (temps de rétablissement, dépassement…)."""
        return {}

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        raise NotImplementedError

    def baseline_command(
        self, ins: Instruments, autopilot: Autopilot, dt: float
    ) -> HighLevelCommand:
        """Commande de la politique de référence (pilote automatique par défaut)."""
        autopilot.targets = self.autopilot_targets(ins)
        return autopilot.high_level(ins, dt)

    def describe(self) -> dict[str, float]:
        """Consignes de l'épisode (pour les journaux)."""
        return {}


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
