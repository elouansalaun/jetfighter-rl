"""Callbacks stable-baselines3 : journalisation détaillée et évaluation périodique.

* ``EpisodeStatsCallback`` : à chaque fin de collecte, moyenne sur les épisodes terminés
  de chaque **terme de récompense** (``terms/…``), des **métriques de tâche**
  (``metrics/…``), du taux de succès et de crash (``episode/…``). C'est ce qui permet de
  voir dans TensorBoard *quel* terme l'agent optimise (et de repérer le *reward hacking*).
* ``EvalCallback`` : toutes les ``freq`` étapes, évaluation déterministe sur des graines
  fixes, comparée à la politique de référence (pilote automatique) évaluée une fois au
  début sur les mêmes graines ; sauvegarde du meilleur modèle et d'un CSV d'historique.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from jetfighter.envs.baselines import Evaluation, RandomPolicy, evaluate, reference_policy
from jetfighter.envs.jet_env import EnvConfig, JetEnv

EVAL_SEED = 10_000  # graines d'évaluation, disjointes de celles de l'entraînement


class AgentPolicy:
    """Politique déterministe d'un modèle stable-baselines3 (interface de ``evaluate``)."""

    def __init__(self, model: BaseAlgorithm) -> None:
        self.model = model

    def reset(self) -> None:
        pass

    def __call__(self, obs: Any) -> Any:
        return self.model.predict(obs, deterministic=True)[0]


class EpisodeStatsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._values: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_terms" not in info:
                continue
            for name, value in info["episode_terms"].items():
                self._values[f"terms/{name}"].append(value)
            for name, value in info.get("metrics", {}).items():
                self._values[f"metrics/{name}"].append(value)
            self._values["episode/success_rate"].append(float(info.get("is_success", False)))
            self._values["episode/crash_rate"].append(float(bool(info.get("violation"))))
            self._values["episode/duration_s"].append(info["t"])
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._values.items():
            self.logger.record(key, float(np.mean(values)))
        self._values.clear()


class CriticWarmupCallback(BaseCallback):
    """Gèle la politique pendant les ``n_rollouts`` premières mises à jour (PPO).

    Après une imitation, la politique est bonne mais la fonction de valeur n'a rien
    appris : les premiers avantages estimés sont du bruit, et PPO dégrade la politique
    imitée (constaté : rendement 82 -> −28 en 50 000 pas). On laisse d'abord le critique
    apprendre la valeur de la politique imitée.
    """

    def __init__(self, n_rollouts: int = 5) -> None:
        super().__init__()
        self.n_rollouts = n_rollouts
        self._count = 0

    def _actor_parameters(self) -> list[Any]:
        policy = self.model.policy
        return [p for name, p in policy.named_parameters()
                if "value" not in name and "critic" not in name and "qf" not in name]  # fmt: skip

    def _set_frozen(self, frozen: bool) -> None:
        for p in self._actor_parameters():
            p.requires_grad_(not frozen)

    def _on_training_start(self) -> None:
        if self.n_rollouts > 0:
            self._set_frozen(True)

    def _on_rollout_start(self) -> None:
        if self._count == self.n_rollouts:
            self._set_frozen(False)
        self._count += 1

    def _on_step(self) -> bool:
        return True


def evaluation_row(ev: Evaluation) -> dict[str, float]:
    return {
        "mean_return": ev.mean_return,
        "std_return": ev.std_return,
        "success_rate": ev.success_rate,
        "crash_rate": ev.crash_rate,
        "mean_length": ev.mean_length,
        **{f"metric_{k}": v for k, v in ev.metrics.items()},
    }


class EvalCallback(BaseCallback):
    """Évaluation périodique + référence + meilleur modèle (voir le module)."""

    def __init__(
        self,
        env_config: EnvConfig,
        out_dir: Path,
        freq: int = 25_000,
        episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.env = JetEnv(env_config)
        self.out_dir = Path(out_dir)
        self.freq = freq
        self.episodes = episodes
        self.baseline: Evaluation | None = None
        self.random: Evaluation | None = None
        self.best: Evaluation | None = None
        self.history: list[dict[str, float]] = []
        self._last_eval = -math.inf

    def _evaluate(self, policy: Any) -> Evaluation:
        return evaluate(self.env, policy, episodes=self.episodes, seed=EVAL_SEED)

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.env.has_reference:
            self.baseline = self._evaluate(reference_policy(self.env))
        self.random = self._evaluate(RandomPolicy(self.env, seed=0))
        if self.verbose:
            if self.baseline is not None:
                print(f"    référence (pilote auto) : {self.baseline}")
            print(f"    aléatoire               : {self.random}")
        self.run_evaluation()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.freq:
            self.run_evaluation()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps > self._last_eval:
            self.run_evaluation()

    @staticmethod
    def _better(new: Evaluation, old: Evaluation | None) -> bool:
        if old is None:
            return True
        return (new.success_rate, new.mean_return) > (old.success_rate, old.mean_return)

    def run_evaluation(self) -> Evaluation:
        self._last_eval = self.num_timesteps
        ev = self._evaluate(AgentPolicy(self.model))
        row = {"timesteps": float(self.num_timesteps), **evaluation_row(ev)}
        self.history.append(row)
        for key, value in evaluation_row(ev).items():
            self.logger.record(f"eval/{key}", value)
        if self.baseline is not None:
            self.logger.record("eval/baseline_return", self.baseline.mean_return)
            self.logger.record("eval/baseline_success_rate", self.baseline.success_rate)
        self.logger.dump(self.num_timesteps)
        if self._better(ev, self.best):
            self.best = ev
            self.model.save(self.out_dir / "best_model")
        self._write_csv()
        if self.verbose:
            print(f"    {self.num_timesteps:>9} pas : {ev}")
        return ev

    def _write_csv(self) -> None:
        keys: list[str] = []
        for row in self.history:
            keys += [k for k in row if k not in keys]
        with open(self.out_dir / "evaluations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
