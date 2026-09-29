"""Imitation (clonage de comportement) de la politique de référence, avant le RL.

Piste prévue par la roadmap pour l'étape 8.6 (initialiser un agent bas niveau par
imitation de « agent hiérarchique + commandes de vol »), utile aussi en hiérarchique : un
agent initialisé par imitation du pilote automatique vole d'emblée correctement, et le RL
n'a plus qu'à l'améliorer.

* ``collect_demonstrations`` : épisodes pilotés par la référence ; l'action **exécutée**
  est bruitée (méthode DART) mais l'action **enregistrée** est celle de la référence : les
  démonstrations couvrent ainsi les petits écarts que l'agent commettra, et il apprend à
  les corriger (sans bruit, le clone dérive dès qu'il s'écarte de la trajectoire idéale).
* ``behavior_cloning`` : régression (erreur quadratique) de la moyenne de la politique
  PPO — ou de l'acteur SAC — sur les actions de la référence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from jetfighter.envs.baselines import reference_policy
from jetfighter.envs.jet_env import EnvConfig, JetEnv

Array = npt.NDArray[np.float32]


@dataclass
class Demonstrations:
    observations: Array
    actions: Array
    returns: list[float]

    def __len__(self) -> int:
        return len(self.actions)


def collect_demonstrations(
    env_config: EnvConfig, episodes: int = 50, noise: float = 0.1, seed: int = 20_000
) -> Demonstrations:
    """Épisodes de la référence avec bruit d'exécution (voir le module)."""
    env = JetEnv(env_config)
    policy = reference_policy(env)
    rng = np.random.default_rng(seed)
    obs_list: list[Array] = []
    act_list: list[Array] = []
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        policy.reset()
        done, ret = False, 0.0
        while not done:
            action = policy(obs)
            obs_list.append(obs)
            act_list.append(action)
            executed = np.clip(action + rng.normal(0.0, noise, action.shape), -1, 1)
            obs, reward, terminated, truncated, _ = env.step(executed.astype(np.float32))
            ret += reward
            done = terminated or truncated
        returns.append(ret)
    return Demonstrations(np.array(obs_list, np.float32), np.array(act_list, np.float32), returns)


def _predicted_mean(model: BaseAlgorithm, obs: torch.Tensor) -> torch.Tensor:
    if isinstance(model, PPO):
        dist: Any = model.policy.get_distribution(obs).distribution  # gaussienne diagonale
        return dist.mean
    if isinstance(model, SAC):
        return model.actor(obs, deterministic=True)
    raise TypeError(f"Algorithme non pris en charge : {type(model).__name__}")


def behavior_cloning(
    model: BaseAlgorithm,
    demos: Demonstrations,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Ajuste la politique sur les démonstrations ; renvoie la perte moyenne par époque."""
    policy: Any = model.policy
    device = policy.device
    obs = torch.as_tensor(demos.observations, device=device)
    target = torch.as_tensor(demos.actions, device=device)
    params = [p for name, p in policy.named_parameters() if "log_std" not in name]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    generator = torch.Generator().manual_seed(0)
    losses = []
    policy.set_training_mode(True)
    for _ in range(epochs):
        perm = torch.randperm(len(obs), generator=generator)
        total, n = 0.0, 0
        for i in range(0, len(obs), batch_size):
            idx = perm[i : i + batch_size]
            loss = torch.mean((_predicted_mean(model, obs[idx]) - target[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)
        losses.append(total / n)
    policy.set_training_mode(False)
    return losses
