"""Trains an agent from a YAML configuration (phase 8).

Usage:
    python scripts/train.py configs/training/8_1_level.yaml
    python scripts/train.py configs/training/8_1_level.yaml --seeds 0 --timesteps 100000
    python scripts/train.py configs/training/8_2_heading_altitude.yaml --model 6dof \\
        --init-from 8_2_heading_altitude          # 3-DOF -> 6-DOF transfer
    tensorboard --logdir runs                     # curves (detailed reward terms)

Results in ``runs/<name>/<date_time>/`` (cf. ``jetfighter_rl.training.train``). Then evaluate
with ``python scripts/evaluate_agent.py runs/<name>/<date_time>/seed0/best_model.zip``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

from jetfighter_rl.training.config import TrainConfig
from jetfighter_rl.training.train import train_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="YAML file (configs/training/…)")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--timesteps", type=int, help="total number of agent steps")
    parser.add_argument("--n-envs", type=int, help="environments in parallel")
    parser.add_argument("--model", choices=("3dof", "6dof"))
    parser.add_argument("--action-mode", choices=("hierarchical", "low_level"))
    parser.add_argument(
        "--init-from", help='.zip model, name of a trained experiment, or "none" (from scratch)'
    )
    parser.add_argument("--name", help="experiment name (default: the YAML's)")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    n_envs = args.n_envs
    cfg = TrainConfig.from_yaml(args.config)
    if n_envs is None and cfg.n_envs > (os.cpu_count() or 1):
        n_envs = max(1, (os.cpu_count() or 2) - 1)
        print(f"({cfg.n_envs} environments requested, {os.cpu_count()} cores: {n_envs})")
    env_overrides = {"model": args.model, "action_mode": args.action_mode}
    name = args.name
    if name is None and args.model and args.model != cfg.env.get("model", "3dof"):
        name = f"{cfg.name}_{args.model}"
    cfg = cfg.override(
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        n_envs=n_envs,
        init_from=args.init_from,
        name=name,
        env={k: v for k, v in env_overrides.items() if v is not None},
    )
    if args.init_from and args.init_from.lower() == "none":
        cfg = dataclasses.replace(cfg, init_from=None)
    try:
        train_seeds(cfg, runs_dir=args.runs_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
