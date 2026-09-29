"""Modular rewards: a sum of named, weighted terms, all of them logged.

Each task returns a ``{term name: value}`` dictionary. The environment sums it and stores
the breakdown in ``info["reward_terms"]``, which shows in TensorBoard which term
drives learning (and helps spot *reward hacking*).

Conventions: cost terms are **negative** and normalized roughly to [−1, 0] per decision
step; bonuses are small and positive; the crash penalty is one-off and large: a fixed
penalty plus the maximum cost of all remaining steps (``Task.max_step_cost``), so that
crashing is always worse than continuing to fly, even far from the target (otherwise the
agent would learn to cut episodes short).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

RewardTerms = dict[str, float]


def total(terms: Mapping[str, float]) -> float:
    return float(sum(terms.values()))


def normalized_error(error: float, scale: float) -> float:
    """|error| / scale, saturated at 1 (bounded cost, constant gradient far from the target)."""
    return min(abs(error) / scale, 1.0)


def angle_error(angle: float) -> float:
    """Angle wrapped to [−π, π[."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def action_smoothness(action: npt.NDArray[np.float64], previous: npt.NDArray[np.float64]) -> float:
    """Mean squared difference between two successive actions (actions in [−1, 1])."""
    diff = np.asarray(action) - np.asarray(previous)
    return float(np.mean(diff * diff)) / 4.0  # ∈ [0, 1]


def accumulate(sums: dict[str, float], terms: Mapping[str, float]) -> None:
    """Accumulates an episode's terms (for end-of-episode statistics)."""
    for k, v in terms.items():
        sums[k] = sums.get(k, 0.0) + v
