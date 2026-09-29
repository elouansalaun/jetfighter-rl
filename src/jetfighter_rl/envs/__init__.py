"""Gymnasium environments for reinforcement learning (phases 7 to 10).

Registered identifiers (parameters can be overridden at creation) ::

    gymnasium.make("JetFighter/Level-v0")                          # stabilization, 3-DOF
    gymnasium.make("JetFighter/HeadingAltitude-v0", model="6dof")  # heading/altitude/speed
    gymnasium.make("JetFighter/Level-v0", action_mode="low_level")

Requires the ``[rl]`` extra (gymnasium).
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
