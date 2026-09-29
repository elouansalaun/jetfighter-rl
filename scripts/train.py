"""Entraîne un agent à partir d'une configuration YAML (phase 8).

Usage :
    python scripts/train.py configs/training/8_1_level.yaml
    python scripts/train.py configs/training/8_1_level.yaml --seeds 0 --timesteps 100000
    python scripts/train.py configs/training/8_2_heading_altitude.yaml --model 6dof \\
        --init-from 8_2_heading_altitude          # transfert 3-DOF -> 6-DOF
    tensorboard --logdir runs                     # courbes (termes de récompense détaillés)

Résultats dans ``runs/<nom>/<date_heure>/`` (cf. ``jetfighter.rl.train``). Évaluer ensuite
avec ``python scripts/evaluate_agent.py runs/<nom>/<date_heure>/seed0/best_model.zip``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

from jetfighter.rl.config import TrainConfig
from jetfighter.rl.train import train_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="fichier YAML (configs/training/…)")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--timesteps", type=int, help="nombre total de pas d'agent")
    parser.add_argument("--n-envs", type=int, help="environnements en parallèle")
    parser.add_argument("--model", choices=("3dof", "6dof"))
    parser.add_argument("--action-mode", choices=("hierarchical", "low_level"))
    parser.add_argument(
        "--init-from", help="modèle .zip, nom d'expérience entraînée, ou « none » (partir de zéro)"
    )
    parser.add_argument("--name", help="nom de l'expérience (par défaut : celui du YAML)")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    n_envs = args.n_envs
    cfg = TrainConfig.from_yaml(args.config)
    if n_envs is None and cfg.n_envs > (os.cpu_count() or 1):
        n_envs = max(1, (os.cpu_count() or 2) - 1)
        print(f"({cfg.n_envs} environnements demandés, {os.cpu_count()} cœurs : {n_envs})")
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
        raise SystemExit(f"Erreur : {exc}") from None


if __name__ == "__main__":
    main()
