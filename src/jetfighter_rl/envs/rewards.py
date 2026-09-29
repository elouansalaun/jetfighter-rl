"""Récompenses modulaires : une somme de termes nommés et pondérés, tous journalisés.

Chaque tâche renvoie un dictionnaire ``{nom du terme: valeur}``. L'environnement en fait la
somme et range le détail dans ``info["reward_terms"]``, ce qui permet de voir dans
TensorBoard **quel terme** pilote l'apprentissage (et de repérer le *reward hacking*).

Conventions : les termes de coût sont **négatifs** et normalisés à peu près dans [−1, 0]
par pas de décision ; les bonus sont petits et positifs ; la pénalité de crash est
ponctuelle et grande : pénalité fixe **plus le coût maximal de tous les pas restants**
(``Task.max_step_cost``), pour que s'écraser soit toujours pire que de continuer à voler,
même loin de la consigne (sinon l'agent apprendrait à abréger les épisodes).
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
    """|erreur| / échelle, saturé à 1 (coût borné, gradient constant loin de la cible)."""
    return min(abs(error) / scale, 1.0)


def angle_error(angle: float) -> float:
    """Angle ramené dans [−π, π[."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def action_smoothness(action: npt.NDArray[np.float64], previous: npt.NDArray[np.float64]) -> float:
    """Écart quadratique moyen entre deux actions successives (actions dans [−1, 1])."""
    diff = np.asarray(action) - np.asarray(previous)
    return float(np.mean(diff * diff)) / 4.0  # ∈ [0, 1]


def accumulate(sums: dict[str, float], terms: Mapping[str, float]) -> None:
    """Cumule les termes d'un épisode (pour les statistiques de fin d'épisode)."""
    for k, v in terms.items():
        sums[k] = sums.get(k, 0.0) + v
