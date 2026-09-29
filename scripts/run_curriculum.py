"""Enchaîne les entraînements d'un programme (curriculum) décrit en YAML (phase 8).

Usage :
    python scripts/run_curriculum.py configs/training/curriculum.yaml
    python scripts/run_curriculum.py configs/training/curriculum.yaml --seeds 0 --from-stage 3
    python scripts/run_curriculum.py configs/training/curriculum.yaml --dry-run

Chaque étape reprend un fichier de ``configs/training/`` et peut en surcharger ``model``,
``action_mode``, ``init_from``, ``timesteps``, ``n_envs`` et ``name``. Une étape 6-DOF
prend par défaut le nom de sa configuration suffixé de ``_6dof``.
"""

from __future__ import annotations

import argparse
import os
import time

from jetfighter.rl.curriculum import load_curriculum
from jetfighter.rl.train import train_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("curriculum")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--from-stage", type=int, default=1, help="reprendre à l'étape k (1…)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true", help="affiche le programme")
    args = parser.parse_args()

    n_envs = args.n_envs or min(8, max(1, (os.cpu_count() or 2) - 1))
    cfgs = load_curriculum(args.curriculum, args.seeds, n_envs)
    for k, cfg in enumerate(cfgs, start=1):
        mark = "  " if k >= args.from_stage else "✓ "
        print(
            f"{mark}{k:2d}. {cfg.name:30s} {cfg.env.get('model', '3dof')}  "
            f"{cfg.total_timesteps:>9} pas × {len(cfg.seeds)} graine(s)"
            f"{'  <- ' + cfg.init_from if cfg.init_from else ''}"
        )
    if args.dry_run:
        return
    t0 = time.time()
    for k, cfg in enumerate(cfgs, start=1):
        if k < args.from_stage:
            continue
        print(
            f"\n##### Étape {k}/{len(cfgs)} : {cfg.name} "
            f"({(time.time() - t0) / 60:.0f} min écoulées) #####"
        )
        train_seeds(cfg, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
