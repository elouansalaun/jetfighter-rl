"""Politiques de référence et évaluation.

* ``AutopilotPolicy`` : le pilote automatique de la phase 6, branché sur l'espace d'action
  hiérarchique de l'environnement. C'est la **référence à battre** (et la preuve que la
  tâche est faisable avec cette récompense).
* ``RandomPolicy`` : actions uniformes ; donne le plancher de performance.
* ``evaluate`` : rendement moyen, taux de succès et de crash sur des épisodes à graines fixes.

Une politique est un appelable ``action = policy(observation)`` ; ``reset()`` est appelé après
chaque ``env.reset()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from jetfighter.control.autopilot import Autopilot
from jetfighter.envs.jet_env import JetEnv, command_to_action

Obs = npt.NDArray[np.float32]


class Policy(Protocol):
    def reset(self) -> None: ...
    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]: ...


class AutopilotPolicy:
    """Pilote automatique -> action hiérarchique normalisée (mesures vraies de l'env)."""

    def __init__(self, env: Any) -> None:
        self.env: JetEnv = env.unwrapped
        if not self.env.hierarchical:
            raise ValueError("La politique pilote automatique requiert le mode hiérarchique.")
        self.ap = Autopilot(self.env.model)

    def reset(self) -> None:
        assert self.env.instruments is not None
        self.ap.reset(self.env.x, self.env.instruments)

    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]:
        ins = self.env.instruments
        assert ins is not None
        self.ap.targets = self.env.task.autopilot_targets(ins)
        cmd = self.ap.high_level(ins, self.env.cfg.agent_dt)
        return command_to_action(cmd, self.env.cfg.max_roll_rate).astype(np.float32)


class RandomPolicy:
    def __init__(self, env: Any, seed: int = 0) -> None:
        self.space = env.action_space
        self.space.seed(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]:
        return np.asarray(self.space.sample(), dtype=np.float32)


@dataclass(frozen=True)
class Evaluation:
    mean_return: float
    std_return: float
    success_rate: float
    crash_rate: float
    mean_length: float  # [s]
    returns: tuple[float, ...]

    def __str__(self) -> str:
        return (f"rendement {self.mean_return:8.1f} ± {self.std_return:5.1f} | "
                f"succès {100 * self.success_rate:5.1f} % | crash {100 * self.crash_rate:5.1f} % | "
                f"durée {self.mean_length:5.1f} s")  # fmt: skip


def evaluate(
    env: Any,
    policy: Policy | Callable[[Obs], npt.NDArray[np.float32]],
    episodes: int = 10,
    seed: int = 1000,
) -> Evaluation:
    """Évalue une politique sur ``episodes`` épisodes (graines seed, seed+1, …)."""
    returns, successes, crashes, lengths = [], 0, 0, []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        if hasattr(policy, "reset"):
            policy.reset()
        done, ep_return = False, 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
        successes += int(info.get("is_success", False))
        crashes += int(terminated)
        lengths.append(info["t"])
    arr = np.array(returns)
    return Evaluation(float(arr.mean()), float(arr.std()), successes / episodes,
                      crashes / episodes, float(np.mean(lengths)), tuple(returns))  # fmt: skip
