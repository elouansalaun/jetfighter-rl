"""Évalue un agent entraîné : comparaison à la référence, vol Tacview, tracés (phase 8).

Usage :
    python scripts/evaluate_agent.py runs/8_1_level/<date_heure>/seed0/best_model.zip
    python scripts/evaluate_agent.py <modèle.zip> --episodes 50 --model 6dof   # transfert
    python scripts/evaluate_agent.py <modèle.zip> --task-kwargs maneuvers=loop

Produit, à côté du modèle : ``eval_<graine>.acmi`` (agent en bleu, pilote auto en orange,
mêmes conditions initiales) et ``eval_<graine>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from jetfighter_rl.training.callbacks import EVAL_SEED
from jetfighter_rl.training.evaluate import (
    compare,
    env_config_for,
    export_side_by_side,
    format_comparison,
    plot_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", choices=("3dof", "6dof"))
    parser.add_argument("--sensors", help="None, realistic ou chemin d'un YAML")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="graine du vol exporté")
    parser.add_argument("--flights", type=int, default=1, help="nombre de vols exportés")
    parser.add_argument("--task-kwargs", nargs="*", default=[], metavar="CLÉ=VALEUR")
    args = parser.parse_args()

    env_cfg = env_config_for(args.model_path, model=args.model, sensors=args.sensors)
    if args.task_kwargs:
        extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.task_kwargs)}
        env_cfg.task_kwargs = {**env_cfg.task_kwargs, **extra}
    print(
        f"Agent {args.model_path}\n  tâche {env_cfg.task} {env_cfg.task_kwargs or ''}, "
        f"modèle {env_cfg.model}, mode {env_cfg.action_mode}, {args.episodes} épisodes"
    )
    print(format_comparison(compare(args.model_path, env_cfg, episodes=args.episodes)))

    out = Path(args.model_path).parent
    for k in range(args.flights):
        seed = args.seed + k
        acmi, recs = export_side_by_side(args.model_path, out / f"eval_{seed}.acmi", env_cfg, seed)
        fig = plot_comparison(recs, title=f"{env_cfg.task} ({env_cfg.model}) — graine {seed}")
        png = out / f"eval_{seed}.png"
        fig.savefig(png, dpi=130)
        print(f"  vol : {acmi}   tracés : {png}")


if __name__ == "__main__":
    main()
