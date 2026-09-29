"""Learning tasks (phase 8 of the roadmap).

A task provides initial conditions, setpoints, own observations, reward terms,
success criterion and reference policy (see ``base.py``).

| Name                 | Step  | Content                                                  |
|----------------------|-------|----------------------------------------------------------|
| ``level``            | 8.1   | unusual-attitude recovery, back to level flight < 10 s   |
| ``heading_altitude`` | 8.2   | randomly drawn heading, altitude and speed               |
| ``sustained_turn``   | 8.3   | coordinated turn at maximum sustained rate               |
| ``waypoints``        | 8.4   | sequence of 3D waypoints                                 |
| ``aerobatics``       | 8.5   | loop, roll, Immelmann, Split-S                           |
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
        raise ValueError(f"Unknown task: {name!r} (available: {sorted(TASKS)})")
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
