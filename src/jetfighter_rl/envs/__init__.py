"""Environnements Gymnasium pour l'apprentissage par renforcement (phases 7 à 10).

Identifiants enregistrés (paramètres surchargeables à la création) ::

    gymnasium.make("JetFighter/Level-v0")                          # stabilisation, 3-DOF
    gymnasium.make("JetFighter/HeadingAltitude-v0", model="6dof")  # cap/altitude/vitesse
    gymnasium.make("JetFighter/Level-v0", action_mode="low_level")

Nécessite l'extra ``[rl]`` (gymnasium).
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv, action_to_command, command_to_action
from jetfighter_rl.envs.tasks import TASKS, make_task

ENV_IDS = {
    "JetFighter/Level-v0": "level",
    "JetFighter/HeadingAltitude-v0": "heading_altitude",
}

for env_id, task in ENV_IDS.items():
    if env_id not in registry:
        register(id=env_id, entry_point="jetfighter_rl.envs.jet_env:JetEnv", kwargs={"task": task})

__all__ = [
    "ENV_IDS",
    "TASKS",
    "EnvConfig",
    "JetEnv",
    "action_to_command",
    "command_to_action",
    "make_task",
]
