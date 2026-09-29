"""Task 8.4: following a sequence of 3D waypoints.

Each episode draws ``n_waypoints`` points: legs of 5 to 9 km (longer than the turn radius
at 60° of bank, ≈ 3 km at 230 m/s), direction change of up to ± ``max_turn`` from one leg
to the next, altitude change of up to ± 800 m. A waypoint is passed when the aircraft
flies within ``capture_radius`` horizontally and ``capture_height`` in altitude; the next
one then becomes the target.

Observations (in the **flight-path frame**: distances, bearing relative to the course,
height difference — invariant under rotation about the vertical) for the target waypoint
and the next one, plus the fraction of waypoints remaining.

Reward: progress toward the target waypoint (distance difference, potential-based
*reward shaping* that does not change the optimal policy), bonus per waypoint passed,
small per-step cost (go fast), command jerks. The episode ends (``terminated``, without
penalty) when the last waypoint is passed: that is success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from jetsim.aircraft.instruments import Instruments
from jetsim.control.autopilot import AutopilotTargets

from jetfighter_rl.envs.rewards import RewardTerms, action_smoothness, angle_error
from jetfighter_rl.envs.tasks.base import DEG, FlightGeometry, InitialCondition, Task, Vec, clip


@dataclass
class WaypointsTask(Task):
    name: str = "waypoints"
    n_waypoints: int = 4
    time_per_waypoint: float = 75.0
    leg_min: float = 5000.0
    leg_max: float = 9000.0
    max_turn: float = 90 * DEG
    max_climb: float = 800.0
    capture_radius: float = 400.0
    capture_height: float = 150.0
    vertical_weight: float = 2.0  # weight of the height difference in the progress distance
    cruise_speed: float = 230.0  # setpoint of the reference autopilot
    w_progress: float = 1.0  # per km of progress
    waypoint_bonus: float = 5.0
    final_bonus: float = 10.0
    w_time: float = 0.01
    w_smooth: float = 0.1
    n_features: int = 9
    waypoints: list[tuple[float, float, float]] = field(default_factory=list, init=False)
    index: int = field(default=0, init=False)
    _distance: float = field(default=0.0, init=False)
    _captured_now: bool = field(default=False, init=False)
    _position: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _t: float = field(default=0.0, init=False)
    _capture_times: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.episode_time = self.n_waypoints * self.time_per_waypoint

    @property
    def max_step_cost(self) -> float:
        # moving away at 400 m/s costs at most 0.04 w_progress per 0.1 s step
        return 0.05 * self.w_progress + self.w_time + self.w_smooth

    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        return InitialCondition(
            altitude=float(rng.uniform(2500, 8000)),
            airspeed=float(rng.uniform(180, 260)),
            heading=float(rng.uniform(-math.pi, math.pi)),
        )

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        n, e, h = ins.north, ins.east, ins.altitude
        bearing = ins.course + rng.uniform(-1, 1) * 60 * DEG
        self.waypoints = []
        for _ in range(self.n_waypoints):
            leg = rng.uniform(self.leg_min, self.leg_max)
            n, e = n + leg * math.cos(bearing), e + leg * math.sin(bearing)
            h = clip(h + rng.uniform(-1, 1) * self.max_climb, 1500.0, 10_000.0)
            self.waypoints.append((float(n), float(e), float(h)))
            bearing += rng.uniform(-1, 1) * self.max_turn
        self.index = 0
        self._position = (ins.north, ins.east, ins.altitude)
        self._distance = self._progress_distance(self.waypoints[0])
        self._captured_now = False
        self._t = 0.0
        self._capture_times = []

    # ------------------------------------------------------------------
    @property
    def target(self) -> tuple[float, float, float] | None:
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None

    def _relative(self, wp: tuple[float, float, float], ins: Instruments) -> tuple[float, ...]:
        """(horizontal distance, bearing relative to the course, height difference)."""
        dn, de = wp[0] - ins.north, wp[1] - ins.east
        rng_h = math.hypot(dn, de)
        return rng_h, angle_error(math.atan2(de, dn) - ins.course), wp[2] - ins.altitude

    def _progress_distance(self, wp: tuple[float, float, float]) -> float:
        n, e, h = self._position
        return math.hypot(wp[0] - n, wp[1] - e, self.vertical_weight * (wp[2] - h))

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        self._t = t
        self._position = (ins.north, ins.east, ins.altitude)
        self._captured_now = False
        wp = self.target
        if wp is None:
            return
        rng_h, _, dh = self._relative(wp, ins)
        if rng_h < self.capture_radius and abs(dh) < self.capture_height:
            self._captured_now = True
            self._capture_times.append(t)
            self.index += 1
            if self.target is not None:
                self._distance = self._progress_distance(self.target)

    def features(self, ins: Instruments) -> Vec:
        out: list[float] = []
        for k in (self.index, self.index + 1):
            if k < len(self.waypoints):
                rng_h, brg, dh = self._relative(self.waypoints[k], ins)
                out += [
                    clip(rng_h / 5000.0, 0.0, 4.0),
                    math.sin(brg),
                    math.cos(brg),
                    clip(dh / 1000.0, -3.0, 3.0),
                ]
            else:
                out += [0.0, 0.0, 0.0, 0.0]
        out.append((len(self.waypoints) - self.index) / self.n_waypoints)
        return np.array(out)

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        terms: RewardTerms = {"progress": 0.0, "waypoint": 0.0}
        wp = self.target
        if wp is not None and not self._captured_now:
            d = self._progress_distance(wp)
            terms["progress"] = self.w_progress * (self._distance - d) / 1000.0
            self._distance = d
        if self._captured_now:
            terms["waypoint"] = self.waypoint_bonus + (self.final_bonus if self.done() else 0.0)
        terms["time"] = -self.w_time
        terms["smoothness"] = -self.w_smooth * action_smoothness(action, previous_action)
        return terms

    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def success(self, ins: Instruments) -> bool:
        return self.done()

    def metrics(self) -> dict[str, float]:
        return {
            "waypoints_reached": float(self.index),
            "completion_time": self._capture_times[-1] if self.done() else self.episode_time,
        }

    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        wp = self.target or self.waypoints[-1]
        bearing = math.atan2(wp[1] - ins.east, wp[0] - ins.north)
        return AutopilotTargets(altitude=wp[2], heading=bearing, airspeed=self.cruise_speed)

    def describe(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, (n, e, h) in enumerate(self.waypoints):
            out |= {f"wp{k}_north": n, f"wp{k}_east": e, f"wp{k}_altitude": h}
        return out
