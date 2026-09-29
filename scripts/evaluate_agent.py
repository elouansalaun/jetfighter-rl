"""Evaluates a trained agent: comparison with the reference, Tacview flight, plots (phase 8).

Usage:
    python scripts/evaluate_agent.py runs/8_1_level/<date_time>/seed0/best_model.zip
    python scripts/evaluate_agent.py <model.zip> --episodes 50 --model 6dof   # transfer
    python scripts/evaluate_agent.py <model.zip> --task-kwargs maneuvers=loop

Writes, next to the model: ``eval_<seed>.acmi`` (agent in blue, autopilot in orange,
same initial conditions) and ``eval_<seed>.png``.
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
    parser.add_argument("--sensors", help="None, realistic or path to a YAML")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="seed of the exported flight")
    parser.add_argument("--flights", type=int, default=1, help="number of exported flights")
    parser.add_argument("--task-kwargs", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    env_cfg = env_config_for(args.model_path, model=args.model, sensors=args.sensors)
    if args.task_kwargs:
        extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.task_kwargs)}
        env_cfg.task_kwargs = {**env_cfg.task_kwargs, **extra}
    print(
        f"Agent {args.model_path}\n  task {env_cfg.task} {env_cfg.task_kwargs or ''}, "
        f"model {env_cfg.model}, mode {env_cfg.action_mode}, {args.episodes} episodes"
    )
    print(format_comparison(compare(args.model_path, env_cfg, episodes=args.episodes)))

    out = Path(args.model_path).parent
    for k in range(args.flights):
        seed = args.seed + k
        acmi, recs = export_side_by_side(args.model_path, out / f"eval_{seed}.acmi", env_cfg, seed)
        fig = plot_comparison(recs, title=f"{env_cfg.task} ({env_cfg.model}) — seed {seed}")
        png = out / f"eval_{seed}.png"
        fig.savefig(png, dpi=130)
        print(f"  flight: {acmi}   plots: {png}")


if __name__ == "__main__":
    main()
