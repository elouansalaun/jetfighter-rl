"""Configuration of a training experiment (versioned YAML file).

Example (``configs/training/8_1_level.yaml``) ::

    name: 8_1_level
    description: Stabilization from a perturbed attitude (3-DOF, hierarchical)
    env:                      # EnvConfig parameters
      task: level
      model: 3dof
      action_mode: hierarchical
    algo: ppo                 # ppo | sac
    total_timesteps: 300000
    n_envs: 8
    seeds: [0, 1, 2]
    hyperparams:              # passed as is to PPO(...) / SAC(...)
      n_steps: 1024
      batch_size: 512
      policy_kwargs: {net_arch: [128, 128]}
      lr_final: 3.0e-5        # optional: linear decay of learning_rate down to lr_final
    eval: {freq: 25000, episodes: 10}
    init_from: null           # .zip model, or name of an already trained experiment
    pretrain: null            # or {episodes: 40, noise: 0.1, epochs: 30, critic_warmup: 5}:
                              # imitation of the reference, then critic only, then PPO

``init_from`` enables the curriculum and the 3-DOF -> 6-DOF transfer (cf. ``transfer.py``).

``normalize_reward`` (true by default) divides the training rewards by the running standard
deviation of the discounted returns (``VecNormalize``): value functions learn targets of
the order of −100 poorly (explained variance ≈ 0 on 8.2 without it). Observations are
not normalized (they already are), so saved models remain usable as is; evaluations use
the raw rewards.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jetfighter_rl.envs.jet_env import EnvConfig

ALGOS = ("ppo", "sac")

DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "ppo": {
        "n_steps": 1024,
        "batch_size": 512,
        "n_epochs": 10,
        "learning_rate": 3.0e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        # initial standard deviation 0.37 (not 1): random ±1 commands crash the aircraft
        # within seconds, so the first episodes teach nothing
        "policy_kwargs": {"net_arch": [128, 128], "log_std_init": -1.0},
    },
    "sac": {
        "learning_rate": 3.0e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_starts": 10_000,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
}


def linear_schedule(initial: float, final: float) -> Any:
    """Linearly decaying learning rate (``progress_remaining``: 1 -> 0)."""

    def schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return schedule


@dataclass
class TrainConfig:
    name: str
    env: dict[str, Any] = field(default_factory=dict)
    algo: str = "ppo"
    total_timesteps: int = 300_000
    n_envs: int = 8
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    hyperparams: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, int] = field(default_factory=lambda: {"freq": 25_000, "episodes": 10})
    init_from: str | None = None
    normalize_reward: bool = True  # VecNormalize (rewards only) during training
    pretrain: dict[str, Any] | None = None  # imitation of the reference before RL (imitation.py)
    description: str = ""

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, not {self.algo!r}.")
        EnvConfig(**self.env)  # key validation

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.setdefault("name", Path(path).stem)
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        cfg = cls(**data)
        return cfg.override(**overrides)

    def override(self, **overrides: Any) -> TrainConfig:
        """Replaces fields; ``env`` is merged key by key (``None`` ignored)."""
        values = {k: v for k, v in overrides.items() if v is not None}
        env = {**self.env, **values.pop("env", {})}
        return dataclasses.replace(self, env=env, **values)

    def env_config(self, **extra: Any) -> EnvConfig:
        """Environment configuration; ``discount`` follows the algorithm's γ."""
        values = {"discount": self.algo_kwargs()["gamma"], **self.env, **extra}
        return EnvConfig(**values)

    def algo_kwargs(self) -> dict[str, Any]:
        """Hyperparameters: algorithm defaults, overridden by the YAML
        (``policy_kwargs`` is merged key by key)."""
        defaults = DEFAULT_HYPERPARAMS[self.algo]
        kwargs = {**defaults, **self.hyperparams}
        kwargs["policy_kwargs"] = {
            **defaults.get("policy_kwargs", {}),
            **self.hyperparams.get("policy_kwargs", {}),
        }
        lr_final = kwargs.pop("lr_final", None)
        if lr_final is not None:  # linear learning-rate decay
            kwargs["learning_rate"] = linear_schedule(
                float(kwargs["learning_rate"]), float(lr_final)
            )
        if self.algo == "ppo":  # the batch must divide the rollout size
            rollout = kwargs["n_steps"] * self.n_envs
            kwargs["batch_size"] = min(kwargs["batch_size"], rollout)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return path
