"""Configuration d'une expérience d'entraînement (fichier YAML versionné).

Exemple (``configs/training/8_1_level.yaml``) ::

    name: 8_1_level
    description: Stabilisation depuis une attitude perturbée (3-DOF, hiérarchique)
    env:                      # paramètres de EnvConfig
      task: level
      model: 3dof
      action_mode: hierarchical
    algo: ppo                 # ppo | sac
    total_timesteps: 300000
    n_envs: 8
    seeds: [0, 1, 2]
    hyperparams:              # passés tels quels à PPO(...) / SAC(...)
      n_steps: 1024
      batch_size: 512
      policy_kwargs: {net_arch: [128, 128]}
      lr_final: 3.0e-5        # optionnel : décroissance linéaire de learning_rate jusqu'à lr_final
    eval: {freq: 25000, episodes: 10}
    init_from: null           # modèle .zip, ou nom d'une expérience déjà entraînée
    pretrain: null            # ou {episodes: 40, noise: 0.1, epochs: 30, critic_warmup: 5} :
                              # imitation de la référence, puis critique seul, puis PPO

``init_from`` permet le curriculum et le transfert 3-DOF -> 6-DOF (cf. ``transfer.py``).

``normalize_reward`` (vrai par défaut) divise les récompenses d'entraînement par l'écart-type
courant des retours actualisés (``VecNormalize``) : les fonctions de valeur apprennent
mal des cibles de l'ordre de −100 (variance expliquée ≈ 0 sur 8.2 sans elle). Les
observations ne sont pas normalisées (elles le sont déjà), les modèles sauvegardés
restent donc utilisables tels quels ; les évaluations se font sur les récompenses brutes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jetfighter.envs.jet_env import EnvConfig

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
        # écart-type initial 0.37 (et non 1) : des commandes aléatoires à ±1 font s'écraser
        # l'avion en quelques secondes, les premiers épisodes n'apprennent alors rien
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
    """Pas d'apprentissage décroissant linéairement (``progress_remaining`` : 1 -> 0)."""

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
    normalize_reward: bool = True  # VecNormalize (récompenses seulement) pendant l'entraînement
    pretrain: dict[str, Any] | None = None  # imitation de la référence avant le RL (imitation.py)
    description: str = ""

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo doit valoir l'un de {ALGOS}, pas {self.algo!r}.")
        EnvConfig(**self.env)  # validation des clés

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.setdefault("name", Path(path).stem)
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Clés inconnues dans {path} : {sorted(unknown)}")
        cfg = cls(**data)
        return cfg.override(**overrides)

    def override(self, **overrides: Any) -> TrainConfig:
        """Remplace des champs ; ``env`` est fusionné clé par clé (``None`` ignoré)."""
        values = {k: v for k, v in overrides.items() if v is not None}
        env = {**self.env, **values.pop("env", {})}
        return dataclasses.replace(self, env=env, **values)

    def env_config(self, **extra: Any) -> EnvConfig:
        """Configuration d'environnement ; ``discount`` suit le γ de l'algorithme."""
        values = {"discount": self.algo_kwargs()["gamma"], **self.env, **extra}
        return EnvConfig(**values)

    def algo_kwargs(self) -> dict[str, Any]:
        """Hyperparamètres : défauts de l'algorithme, surchargés par le YAML
        (``policy_kwargs`` est fusionné clé par clé)."""
        defaults = DEFAULT_HYPERPARAMS[self.algo]
        kwargs = {**defaults, **self.hyperparams}
        kwargs["policy_kwargs"] = {
            **defaults.get("policy_kwargs", {}),
            **self.hyperparams.get("policy_kwargs", {}),
        }
        lr_final = kwargs.pop("lr_final", None)
        if lr_final is not None:  # décroissance linéaire du pas d'apprentissage
            kwargs["learning_rate"] = linear_schedule(
                float(kwargs["learning_rate"]), float(lr_final)
            )
        if self.algo == "ppo":  # le lot doit diviser la taille du rollout
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
