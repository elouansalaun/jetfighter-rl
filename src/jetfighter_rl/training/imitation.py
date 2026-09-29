"""Imitation (behavior cloning) of the reference policy, before RL.

Lead planned in the roadmap for step 8.6 (initialize a low-level agent by imitating
"hierarchical agent + fly-by-wire"), also useful in hierarchical mode: an agent
initialized by imitating the autopilot flies correctly from the start, and RL only has
to improve it.

* ``collect_demonstrations``: episodes flown by the reference; the executed action is
  noisy (DART method) but the **recorded** action is the reference's: the demonstrations
  thus cover the small deviations the agent will make, and it learns to correct them
  (without noise, the clone drifts as soon as it leaves the ideal trajectory).
* ``behavior_cloning``: regression (squared error) of the mean of the PPO policy — or of
  the SAC actor — onto the reference's actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from jetfighter_rl.envs.baselines import reference_policy
from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv

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
    """Reference episodes with execution noise (see the module)."""
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
        dist: Any = model.policy.get_distribution(obs).distribution  # diagonal Gaussian
        return dist.mean
    if isinstance(model, SAC):
        return model.actor(obs, deterministic=True)
    raise TypeError(f"Unsupported algorithm: {type(model).__name__}")


def behavior_cloning(
    model: BaseAlgorithm,
    demos: Demonstrations,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Fits the policy to the demonstrations; returns the mean loss per epoch."""
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
