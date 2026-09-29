"""Training an agent (PPO or SAC) from a YAML configuration, over several seeds.

Output tree ::

    runs/<name>/<date_time>/
        config.yaml               # effective configuration (reproducibility)
        summary.json              # per-seed results + mean ± standard deviation
        seed0/
            tb/                   # TensorBoard logs
            best_model.zip        # best model in evaluation
            final_model.zip
            evaluations.csv       # evaluation history
        seed1/ …

Usage ::

    cfg = TrainConfig.from_yaml("configs/training/8_1_level.yaml")
    summary = train_seeds(cfg)

or from the command line: ``python scripts/train.py configs/training/8_1_level.yaml``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from jetfighter_rl.envs.baselines import Evaluation
from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv
from jetfighter_rl.training.callbacks import (
    CriticWarmupCallback,
    EpisodeStatsCallback,
    EvalCallback,
    evaluation_row,
)
from jetfighter_rl.training.config import TrainConfig
from jetfighter_rl.training.imitation import behavior_cloning, collect_demonstrations
from jetfighter_rl.training.transfer import resolve_model_path, transfer_weights

ALGO_CLASSES: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC}


@dataclass
class SeedResult:
    seed: int
    directory: Path
    best: Evaluation
    final: Evaluation
    baseline: Evaluation | None
    duration: float  # [s]
    timesteps: int


def make_env_factory(env_config: EnvConfig) -> Any:
    def factory() -> JetEnv:
        return JetEnv(env_config)

    return factory


def run_config_of(model_path: str | Path) -> TrainConfig | None:
    """Configuration of the run that produced this model (neighboring ``config.yaml``)."""
    path = Path(model_path)
    for directory in (path.parent, path.parent.parent):
        if (directory / "config.yaml").exists():
            cfg = TrainConfig.from_yaml(directory / "config.yaml")
            if "attitude_features" not in cfg.env:  # run predating 2026-09-28
                cfg = cfg.override(env={"attitude_features": "euler"})
            return cfg
    return None


_SCHEDULE_OBJECTS: dict[str, Any] = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
"""Replaces the saved optimization schedules on load: they are functions serialized by value
(bytecode), unreadable from one Python version to another, and useless for inference or
weight transfer."""


def load_model(path: str | Path, env: Any = None, device: str = "cpu") -> BaseAlgorithm:
    """Loads a PPO or SAC model (algorithm read from the run configuration, otherwise
    tried in order), for inference or weight transfer."""
    path = Path(path)
    cfg = run_config_of(path)
    classes: list[type[BaseAlgorithm]] = [ALGO_CLASSES[cfg.algo]] if cfg else [PPO, SAC]
    errors = []
    for cls in classes:
        try:
            return cls.load(path, env=env, device=device, custom_objects=_SCHEDULE_OBJECTS)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise ValueError(f"Cannot load {path}: {'; '.join(errors)}")


def train_one(
    cfg: TrainConfig,
    seed: int,
    out_dir: Path,
    runs_dir: str | Path = "runs",
    verbose: int = 1,
) -> SeedResult:
    # A single compute thread for torch: small networks gain nothing from more, and several
    # threads compete for cores with the environment processes (throughput ÷ 6 measured)
    torch.set_num_threads(1)
    env_config = cfg.env_config()
    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(
        make_env_factory(env_config), n_envs=cfg.n_envs, seed=seed, vec_env_cls=vec_cls
    )
    algo_kwargs = cfg.algo_kwargs()
    if cfg.normalize_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=algo_kwargs["gamma"])
    algo_cls = ALGO_CLASSES[cfg.algo]
    model = algo_cls(
        "MlpPolicy",
        venv,
        seed=seed,
        device="cpu",
        verbose=0,
        tensorboard_log=str(out_dir),
        **algo_kwargs,
    )
    if cfg.init_from:
        src_path = resolve_model_path(cfg.init_from, runs_dir)
        report = transfer_weights(load_model(src_path), model)
        if verbose:
            print(f"    initialized from {src_path}: {report}")
    if cfg.pretrain:
        pre = dict(cfg.pretrain)
        demos = collect_demonstrations(env_config, episodes=pre.get("episodes", 40),
                                       noise=pre.get("noise", 0.1))  # fmt: skip
        losses = behavior_cloning(model, demos, epochs=pre.get("epochs", 30))
        if verbose:
            print(f"    imitation of the reference: {len(demos)} samples, "
                  f"squared error {losses[0]:.4f} -> {losses[-1]:.5f}")  # fmt: skip

    ev_cb = EvalCallback(
        env_config,
        out_dir,
        freq=cfg.eval.get("freq", 25_000),
        episodes=cfg.eval.get("episodes", 10),
        verbose=verbose,
    )
    callbacks = [EpisodeStatsCallback(), ev_cb]
    if cfg.pretrain and cfg.algo == "ppo":
        callbacks.append(CriticWarmupCallback(cfg.pretrain.get("critic_warmup", 5)))
    t0 = time.perf_counter()
    try:
        model.learn(
            cfg.total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="tb",
        )
    finally:
        venv.close()
    duration = time.perf_counter() - t0
    model.save(out_dir / "final_model")
    assert ev_cb.best is not None
    return SeedResult(
        seed, out_dir, ev_cb.best, _last(ev_cb), ev_cb.baseline, duration, int(model.num_timesteps)
    )


def _last(cb: EvalCallback) -> Evaluation:
    """Last evaluation (rebuilt from the history)."""
    row = cb.history[-1]
    return Evaluation(
        row["mean_return"],
        row["std_return"],
        row["success_rate"],
        row["crash_rate"],
        row["mean_length"],
        (),
        {k[len("metric_") :]: v for k, v in row.items() if k.startswith("metric_")},
    )


def train_seeds(
    cfg: TrainConfig,
    runs_dir: str | Path = "runs",
    verbose: int = 1,
) -> dict[str, Any]:
    """Trains one seed after another; returns (and writes) the summary."""
    if cfg.init_from:  # fail right away if the starting policy does not exist
        resolve_model_path(cfg.init_from, runs_dir)
    root = Path(runs_dir) / cfg.name / time.strftime("%Y%m%d_%H%M%S")
    env_cfg = cfg.env_config()  # representation choices frozen in the saved configuration
    frozen = {
        k: getattr(env_cfg, k) for k in ("attitude_features", "nz_neutral", "action_exponent")
    }
    cfg.override(env=frozen).save(root / "config.yaml")
    results: list[SeedResult] = []
    for seed in cfg.seeds:
        if verbose:
            env = cfg.env
            print(
                f"\n=== {cfg.name} — seed {seed} ({cfg.algo.upper()}, "
                f"{env.get('task')}, {env.get('model', '3dof')}, "
                f"{env.get('action_mode', 'hierarchical')}, {cfg.total_timesteps} steps, "
                f"{cfg.n_envs} env) ==="
            )
        results.append(train_one(cfg, seed, root / f"seed{seed}", runs_dir, verbose))
        summary = summarize(cfg, root, results)
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if verbose:
        print(format_summary(summary))
    return summary


def summarize(cfg: TrainConfig, root: Path, results: list[SeedResult]) -> dict[str, Any]:
    def stats(key: str, which: str) -> dict[str, float]:
        values = [evaluation_row(getattr(r, which))[key] for r in results]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    baseline = next((r.baseline for r in results if r.baseline is not None), None)
    return {
        "name": cfg.name,
        "directory": str(root),
        "config": cfg.to_dict(),
        "baseline": evaluation_row(baseline) if baseline else None,
        "seeds": [
            {
                "seed": r.seed,
                "directory": str(r.directory),
                "timesteps": r.timesteps,
                "duration_s": r.duration,
                "best": evaluation_row(r.best),
                "final": evaluation_row(r.final),
            }
            for r in results
        ],
        "best": {k: stats(k, "best") for k in ("mean_return", "success_rate", "crash_rate")},
        "final": {k: stats(k, "final") for k in ("mean_return", "success_rate", "crash_rate")},
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f'\nSummary "{summary["name"]}" ({len(summary["seeds"])} seed(s)) — {summary["directory"]}'
    ]
    base = summary["baseline"]
    if base:
        lines.append(
            f"  reference   : return {base['mean_return']:8.1f} | "
            f"success {100 * base['success_rate']:5.1f} %"
        )
    for which, label in (("best", "best       "), ("final", "final      ")):
        s = summary[which]
        lines.append(
            f"  {label} : return {s['mean_return']['mean']:8.1f} ± "
            f"{s['mean_return']['std']:5.1f} | success {100 * s['success_rate']['mean']:5.1f} "
            f"± {100 * s['success_rate']['std']:4.1f} % | crash "
            f"{100 * s['crash_rate']['mean']:5.1f} %"
        )
    return "\n".join(lines)


def best_model_of(name: str, runs_dir: str | Path = "runs") -> Path:
    """Best model of the best seed of the latest run of ``name``."""
    runs = sorted(p for p in (Path(runs_dir) / name).glob("*/summary.json"))
    if not runs:
        raise FileNotFoundError(
            f'No finished run of "{name}" in {runs_dir}/: this experiment '
            f'starts from the "{name}" policy (init_from). Train that one first '
            f"(python scripts/train.py configs/training/{name}.yaml), or start from scratch "
            f"with --init-from none."
        )
    summary = json.loads(runs[-1].read_text(encoding="utf-8"))
    best = max(
        summary["seeds"], key=lambda s: (s["best"]["success_rate"], s["best"]["mean_return"])
    )
    return Path(best["directory"]) / "best_model.zip"
