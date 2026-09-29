"""Initialisation d'une politique à partir d'une autre (curriculum, 3-DOF -> 6-DOF).

Les observations commencent toujours par les mêmes grandeurs (état propre de l'avion,
puis action précédente) ; seules les grandeurs propres à la tâche, à la fin, changent.
On copie donc :

* tous les paramètres de même forme (3-DOF -> 6-DOF sur la même tâche : copie intégrale) ;
* pour une première couche dont seul le nombre d'entrées diffère, les colonnes des
  entrées communes (les nouvelles entrées partent de poids nuls : au départ, la politique
  les ignore et se comporte comme l'ancienne) ;
* rien d'autre (les couches de sortie d'un autre espace d'action sont réinitialisées).

L'écart-type d'exploration (``log_std``) n'est **pas** copié : la politique source a
réduit son exploration pour sa propre tâche ; la reprendre telle quelle empêche de
découvrir la nouvelle (constaté sur le virage soutenu : l'agent issu de la stabilisation
n'osait plus incliner). On garde l'écart-type initial de la nouvelle expérience.

Les critiques de SAC, dont l'entrée est (observation, action) concaténées, ne sont copiés
que si les formes coïncident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class TransferReport:
    copied: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{len(self.copied)} tenseurs copiés, {len(self.partial)} partiellement "
            f"(entrées communes), {len(self.skipped)} réinitialisés"
        )


def transfer_weights(source: BaseAlgorithm, target: BaseAlgorithm) -> TransferReport:
    """Copie les poids compatibles de ``source`` dans ``target`` (en place)."""
    report = TransferReport()
    src = source.policy.state_dict()
    dst = target.policy.state_dict()
    for name, tensor in dst.items():
        if name not in src or name == "log_std":
            report.skipped.append(name)
            continue
        s = src[name]
        if s.shape == tensor.shape:
            dst[name] = s.clone()
            report.copied.append(name)
        elif (
            s.ndim == 2
            and tensor.ndim == 2
            and s.shape[0] == tensor.shape[0]
            and "critic" not in name
            and "qf" not in name
        ):
            n = min(s.shape[1], tensor.shape[1])
            new = torch.zeros_like(tensor)
            new[:, :n] = s[:, :n]
            dst[name] = new
            report.partial.append(name)
        else:
            report.skipped.append(name)
    target.policy.load_state_dict(dst)
    return report


def resolve_model_path(init_from: str, runs_dir: str | Path = "runs") -> Path:
    """``init_from`` : chemin d'un ``.zip``, ou nom d'une expérience déjà entraînée
    (on prend alors le meilleur modèle de la meilleure graine de sa dernière exécution)."""
    path = Path(init_from)
    if path.suffix == ".zip" or path.exists():
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    from jetfighter_rl.training.train import best_model_of  # import tardif (évite un cycle)

    return best_model_of(init_from, runs_dir)
