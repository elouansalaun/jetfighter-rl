"""Agent training (phase 8): YAML configurations, PPO/SAC, multiple seeds,
detailed logging, comparison with the reference, policy transfer.

Requires the ``[rl]`` extra (stable-baselines3, torch, tensorboard).
"""

from jetfighter_rl.training.config import TrainConfig
from jetfighter_rl.training.train import best_model_of, load_model, train_one, train_seeds

__all__ = ["TrainConfig", "best_model_of", "load_model", "train_one", "train_seeds"]
