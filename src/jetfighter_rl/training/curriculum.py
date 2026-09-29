"""Training curriculum: a sequence of experiments described in YAML.

Each stage reuses a file from ``configs/training/`` and can override its ``model``,
``action_mode``, ``init_from``, ``timesteps``, ``n_envs`` and ``name``. A stage whose
model (or action mode) differs from its configuration is named by default after that
configuration with a suffix (``_6dof``, ``_low_level``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jetfighter_rl.training.config import TrainConfig

STAGE_KEYS = {"config", "model", "action_mode", "init_from", "timesteps", "n_envs", "name"}


def stage_config(
    stage: dict[str, Any], base_dir: Path, seeds: list[int] | None, n_envs: int | None
) -> TrainConfig:
    unknown = set(stage) - STAGE_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in stage {stage}: {sorted(unknown)}")
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
