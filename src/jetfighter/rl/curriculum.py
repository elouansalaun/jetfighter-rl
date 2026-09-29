"""Programme d'entraînement (curriculum) : une suite d'expériences décrite en YAML.

Chaque étape reprend un fichier de ``configs/training/`` et peut en surcharger ``model``,
``action_mode``, ``init_from``, ``timesteps``, ``n_envs`` et ``name``. Une étape dont le
modèle (ou le mode d'action) diffère de sa configuration prend par défaut le nom de
celle-ci suffixé (``_6dof``, ``_low_level``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jetfighter.rl.config import TrainConfig

STAGE_KEYS = {"config", "model", "action_mode", "init_from", "timesteps", "n_envs", "name"}


def stage_config(
    stage: dict[str, Any], base_dir: Path, seeds: list[int] | None, n_envs: int | None
) -> TrainConfig:
    unknown = set(stage) - STAGE_KEYS
    if unknown:
        raise ValueError(f"Clés inconnues dans l'étape {stage} : {sorted(unknown)}")
    cfg = TrainConfig.from_yaml(base_dir / stage["config"])
    env = {k: stage[k] for k in ("model", "action_mode") if k in stage}
    name = stage.get("name")
    for key, default in (("model", "3dof"), ("action_mode", "hierarchical")):
        if name is None and key in env and env[key] != cfg.env.get(key, default):
            name = f"{cfg.name}_{env[key]}"
    return cfg.override(
        env=env,
        name=name,
        init_from=stage.get("init_from"),
        total_timesteps=stage.get("timesteps"),
        seeds=seeds,
        n_envs=n_envs or stage.get("n_envs"),
    )


def load_curriculum(
    path: str | Path, seeds: list[int] | None = None, n_envs: int | None = None
) -> list[TrainConfig]:
    path = Path(path)
    stages = yaml.safe_load(path.read_text(encoding="utf-8"))["stages"]
    return [stage_config(s, path.parent, seeds, n_envs) for s in stages]
