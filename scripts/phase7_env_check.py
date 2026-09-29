"""Vérification de l'environnement d'apprentissage (phase 7) et essai d'entraînement PPO.

Étapes :
1. ``check_env`` de Gymnasium sur toutes les combinaisons tâche × modèle × mode d'action ;
2. débit (pas d'agent par seconde) avec 1 environnement puis ``--n-envs`` en parallèle ;
3. politiques de référence (pilote automatique, aléatoire) sur les deux tâches ;
4. optionnel (``--ppo-steps N``) : entraînement PPO en mode hiérarchique, suivi TensorBoard
   dans ``runs/``, modèle sauvegardé dans ``models/``, vol de l'agent exporté pour Tacview.

Installation : ``uv pip install -e ".[dev,rl]"``.

Usage :
    python scripts/phase7_env_check.py
    python scripts/phase7_env_check.py --ppo-steps 100000 --task level --n-envs 8
    tensorboard --logdir runs          # courbes d'apprentissage
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from jetsim.viz.tacview import export_recording

from jetfighter_rl.envs.baselines import AutopilotPolicy, RandomPolicy, evaluate
from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv


def throughput(cfg: EnvConfig, n_envs: int, steps: int) -> float:
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = make_vec_env(lambda: JetEnv(cfg), n_envs=n_envs, vec_env_cls=vec_cls, seed=0)
    venv.reset()
    rng = np.random.default_rng(0)
    t0, n = time.perf_counter(), 0
    while n < steps:
        venv.step(rng.uniform(-0.2, 0.2, size=(n_envs, venv.action_space.shape[0])))
        n += n_envs
    rate = n / (time.perf_counter() - t0)
    venv.close()
    return rate


def train_ppo(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    cfg = EnvConfig(task=args.task, model=args.model)
    venv = make_vec_env(lambda: JetEnv(cfg), n_envs=args.n_envs, seed=0,
                        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None)  # fmt: skip
    run = f"ppo_{args.task}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, seed=0, verbose=0,
                tensorboard_log="runs")  # fmt: skip
    eval_env = JetEnv(cfg)

    def policy(obs):
        return model.predict(obs, deterministic=True)[0]

    print(f"\nPPO sur « {args.task} » ({args.model}, hiérarchique), {args.n_envs} environnements")
    print(f"  avant        : {evaluate(eval_env, policy, episodes=args.eval_episodes)}")
    t0 = time.perf_counter()
    chunk = max(args.ppo_steps // 5, 1)
    while model.num_timesteps < args.ppo_steps:
        model.learn(chunk, reset_num_timesteps=False, tb_log_name=run)
        ev = evaluate(eval_env, policy, episodes=args.eval_episodes)
        print(f"  {model.num_timesteps:>8} pas ({time.perf_counter() - t0:5.0f} s) : {ev}")
    Path("models").mkdir(exist_ok=True)
    model.save(Path("models") / run)
    # Vol de l'agent enregistré pour Tacview
    rec_env = JetEnv(EnvConfig(task=args.task, model=args.model, record=True))
    obs, _ = rec_env.reset(seed=2026)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = rec_env.step(policy(obs))
        done = terminated or truncated
    rec = rec_env.recording()
    if rec is not None:
        path = export_recording(rec, Path("outputs/flights") / f"{run}.acmi", pilot="Agent PPO")
        print(f"  modèle : models/{run}.zip   vol Tacview : {path}")
    venv.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="level", choices=("level", "heading_altitude"))
    parser.add_argument("--model", default="3dof", choices=("3dof", "6dof"))
    parser.add_argument("--n-envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--ppo-steps", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_checks:
        print("1. check_env")
        for task in ("level", "heading_altitude"):
            for model in ("3dof", "6dof"):
                for mode in ("hierarchical", "low_level"):
                    env = JetEnv(EnvConfig(task=task, model=model, action_mode=mode))
                    check_env(env, skip_render_check=True)
                    print(f"   OK  {task:17s} {model}  {mode:12s} obs {env.observation_space.shape}"
                          f"  action {env.action_space.shape}")  # fmt: skip

        print("\n2. Débit (pas d'agent/s, 1 pas = 0.1 s de vol)")
        try:
            for model in ("3dof", "6dof"):
                cfg = EnvConfig(task="heading_altitude", model=model)
                steps = 600 if model == "6dof" else 2000
                one = throughput(cfg, 1, steps)
                many = throughput(cfg, args.n_envs, steps * 2) if args.n_envs > 1 else one
                par = f"{many:6.0f} ({args.n_envs} env en parallèle)"
                print(f"   {model} : {one:6.0f} (1 env)   {par}")
        except ImportError:
            print('   (stable-baselines3 absent : uv pip install -e ".[rl]")')

        print("\n3. Politiques de référence")
        for task in ("level", "heading_altitude"):
            env = JetEnv(EnvConfig(task=task, model=args.model))
            n = args.eval_episodes
            print(f"   {task:17s} pilote auto : {evaluate(env, AutopilotPolicy(env), episodes=n)}")
            print(f"   {'':17s} aléatoire   : {evaluate(env, RandomPolicy(env), episodes=n)}")

    if args.ppo_steps > 0:
        train_ppo(args)


if __name__ == "__main__":
    main()
