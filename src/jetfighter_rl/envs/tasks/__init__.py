"""Tâches d'apprentissage (phase 8 de la roadmap).

Une tâche fournit conditions initiales, consignes, observations propres, termes de
récompense, critère de réussite et politique de référence (voir ``base.py``).

| Nom                  | Étape | Contenu                                                  |
|----------------------|-------|----------------------------------------------------------|
| ``level``            | 8.1   | rattrapage d'assiette inusuelle, retour en palier < 10 s |
| ``heading_altitude`` | 8.2   | cap, altitude et vitesse tirés au hasard                 |
| ``sustained_turn``   | 8.3   | virage coordonné à taux maximal soutenu                  |
| ``waypoints``        | 8.4   | suite de points de passage 3D                            |
| ``aerobatics``       | 8.5   | looping, tonneau, Immelmann, Split-S                     |
"""

from __future__ import annotations

from typing import Any

from jetfighter_rl.envs.tasks.aerobatics import MANEUVERS, AerobaticsTask
from jetfighter_rl.envs.tasks.base import FlightGeometry, InitialCondition, Task
from jetfighter_rl.envs.tasks.basic import HeadingAltitudeTask, LevelFlightTask
from jetfighter_rl.envs.tasks.turn import SustainedTurnTask, sustained_turn_reference
from jetfighter_rl.envs.tasks.waypoints import WaypointsTask

TASKS: dict[str, type[Task]] = {
    "level": LevelFlightTask,
    "heading_altitude": HeadingAltitudeTask,
    "sustained_turn": SustainedTurnTask,
    "waypoints": WaypointsTask,
    "aerobatics": AerobaticsTask,
}


def make_task(name: str, **kwargs: Any) -> Task:
    if name not in TASKS:
        raise ValueError(f"Tâche inconnue : {name!r} (disponibles : {sorted(TASKS)})")
    if "maneuvers" in kwargs and isinstance(kwargs["maneuvers"], str):
        kwargs["maneuvers"] = (kwargs["maneuvers"],)
    return TASKS[name](**kwargs)


__all__ = [
    "MANEUVERS",
    "TASKS",
    "AerobaticsTask",
    "FlightGeometry",
    "HeadingAltitudeTask",
    "InitialCondition",
    "LevelFlightTask",
    "SustainedTurnTask",
    "Task",
    "WaypointsTask",
    "make_task",
    "sustained_turn_reference",
]
