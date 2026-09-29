"""Entraînement des agents (phase 8) : configurations YAML, PPO/SAC, plusieurs graines,
journalisation détaillée, comparaison à la référence, transfert de politiques.

Nécessite l'extra ``[rl]`` (stable-baselines3, torch, tensorboard).
"""

from jetfighter_rl.training.config import TrainConfig
from jetfighter_rl.training.train import best_model_of, load_model, train_one, train_seeds

__all__ = ["TrainConfig", "best_model_of", "load_model", "train_one", "train_seeds"]
