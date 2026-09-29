"""Runs the training stages of a curriculum described in YAML, one after the other (phase 8).

Usage:
    python scripts/run_curriculum.py configs/training/curriculum.yaml
    python scripts/run_curriculum.py configs/training/curriculum.yaml --seeds 0 --from-stage 3
    python scripts/run_curriculum.py configs/training/curriculum.yaml --dry-run

Each stage reuses a file from ``configs/training/`` and can override its ``model``,
``action_mode``, ``init_from``, ``timesteps``, ``n_envs`` and ``name``. A 6-DOF stage is
named by default after its configuration with the ``_6dof`` suffix.
"""

from __future__ import annotations

import argparse
import os
import time

from jetfighter_rl.training.curriculum import load_curriculum
from jetfighter_rl.training.train import train_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("curriculum")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--from-stage", type=int, default=1, help="resume at stage k (1…)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true", help="print the curriculum")
    args = parser.parse_args()

    n_envs = args.n_envs or min(8, max(1, (os.cpu_count() or 2) - 1))
    cfgs = load_curriculum(args.curriculum, args.seeds, n_envs)
    for k, cfg in enumerate(cfgs, start=1):
        mark = "  " if k >= args.from_stage else "✓ "
        print(
            f"{mark}{k:2d}. {cfg.name:30s} {cfg.env.get('model', '3dof')}  "
            f"{cfg.total_timesteps:>9} steps × {len(cfg.seeds)} seed(s)"
            f"{'  <- ' + cfg.init_from if cfg.init_from else ''}"
        )
    if args.dry_run:
        return
    t0 = time.time()
    for k, cfg in enumerate(cfgs, start=1):
        if k < args.from_stage:
            continue
        print(
            f"\n##### Stage {k}/{len(cfgs)}: {cfg.name} "
            f"({(time.time() - t0) / 60:.0f} min elapsed) #####"
        )
        train_seeds(cfg, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
