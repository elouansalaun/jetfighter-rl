"""Common interface of the learning tasks.

A task says **what** to learn; the environment (``jet_env.py``) says **how** the aircraft
flies. Lifecycle of an episode, seen from the task ::

    ic = task.sample_initial(rng, six_dof)        # initial conditions
    task.reset(ins, geo, rng, six_dof)            # episode setpoints
    loop:
        features = task.features(measurements)    # task-specific observations
        ... the agent acts, the aircraft flies for agent_dt ...
        task.update(ins, geo, t)                  # tracking (timers, progress, waypoints…)
        terms = task.reward(ins, action, previous_action)
        end if task.done() (task complete) or envelope exit or maximum duration
    info["is_success"] = task.episode_success(ins); info["metrics"] = task.metrics()

``ins``: **true** instruments; ``geo``: position and wind frame (singularity-free), also
true. Only ``features`` receives the **measurements** (sensors, possibly noisy).

The reference policy (``baselines.AutopilotPolicy``) calls ``baseline_command``: by
default, the phase 6 autopilot with the ``autopilot_targets`` setpoints; the aerobatics
tasks replace it with a scripted pilot.
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
    heading: float = 0.0  # course χ [rad]
    gamma: float = 0.0  # flight-path angle [rad]
    bank: float = 0.0  # bank μ [rad]
    roll_rate: float = 0.0  # [rad/s]


@dataclass(frozen=True)
class FlightGeometry:
    """Position and orientation of the wind frame, singularity-free (cf. ``wind_axes``)."""

    position: Vec  # [north, east, down] [m]
    c_nw: Vec  # wind frame -> NED

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
    """Common task interface (see the module)."""

    name: str = "task"
    episode_time: float = 30.0  # max episode duration [s]
    crash_penalty: float = -50.0
    n_features: int = 0
    # one-off terms (bonuses, progress): independent of the agent's rate; the others are
    # per-step costs, scaled by agent_dt / 0.1 s by the environment
    event_terms: frozenset[str] = frozenset({"progress", "waypoint"})

    @property
    def max_step_cost(self) -> float:
        """Bound on the per-step cost (sum of the weights of the negative terms). Used to make
        the crash penalty always worse than continuing to fly (see ``JetEnv.step``)."""
        return 1.0

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        raise NotImplementedError

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        """Draws the episode setpoints (after the initial state is built)."""

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        """Internal tracking after each agent step (before ``reward``)."""

    def features(self, ins: Instruments) -> Vec:
        return np.zeros(0)

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        raise NotImplementedError

    def success(self, ins: Instruments) -> bool:
        """Is the aircraft within tolerance at this instant?"""
        raise NotImplementedError

    def done(self) -> bool:
        """Task complete: the episode ends (``terminated``) without penalty."""
        return False

    def episode_success(self, ins: Instruments) -> bool:
        """Episode success criterion (roadmap, phase 8), evaluated at the end."""
        return self.success(ins)

    def metrics(self) -> dict[str, float]:
        """End-of-episode measurements (recovery time, overshoot…)."""
        return {}

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        raise NotImplementedError

    def baseline_command(
        self, ins: Instruments, autopilot: Autopilot, dt: float
    ) -> HighLevelCommand:
        """Command of the reference policy (autopilot by default)."""
        autopilot.targets = self.autopilot_targets(ins)
        return autopilot.high_level(ins, dt)

    def describe(self) -> dict[str, float]:
        """Episode setpoints (for the logs)."""
        return {}


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
