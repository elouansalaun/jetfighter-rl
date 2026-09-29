"""Task 8.5: aerobatics — loop, roll, Immelmann, Split-S.

Each maneuver is a sequence of reference **segments**, described in the maneuver frame
(frozen at the start: e₁ = initial course in the horizontal plane, e₂ = to the right,
e₃ = down):

* ``Pitch(Δθ)``: the velocity vector rotates by Δθ **in the vertical plane** (e₁, e₃),
  lift in that plane, toward the center of the loop (Δθ > 0: upward at the start);
* ``Roll(Δφ)``: the aircraft rotates by Δφ **about its velocity vector**, flight path
  unchanged.

| Maneuver   | Segments                          | Exit                            |
|------------|-----------------------------------|---------------------------------|
| loop       | Pitch(+360°)                      | same heading, wings level       |
| roll       | Roll(±360°)                       | same heading, wings level       |
| Immelmann  | Pitch(+180°), Roll(±180°)         | reversed heading, higher        |
| Split-S    | Roll(±180°), Pitch(−180°)         | reversed heading, lower         |

Angles are tracked **without singularity** from the wind frame (``FlightGeometry``): a
loop passes through the vertical, where bank μ and course χ are no longer defined.

Reward: progress along the segments (**new records** only, as a fraction of the total
angle: 40 points for the complete maneuver + 20 on exit), deviations from the reference
(leaving the loop plane, misoriented lift, flight path drifting during a roll), small
per-step cost, then, once the segments are done, return to wings level. The maneuver
succeeds (``done``) when all segments are done and the aircraft is stabilized
(|μ| < 15°, |γ| < 10°) on the right heading (± 30° of the initial heading, or of its
reverse for the Immelmann and the Split-S). Pitch only counts if the velocity stays close
to the maneuver plane (|v·e₂| < sin 30°).

Pitfalls met during training: with only 10 points of progress, the agent preferred not
to perform the maneuver (flying straight costs nothing during a pitch segment); with a
reversible progress (ΔΦ), a failed attempt (climb then fall back) paid nothing and the
agent did not try; with Euler angles in the observation, an agent climbed to the vertical
and stayed there to keep its progress; with tracking penalties of 0.2 per step, a failed
attempt left the aircraft out of the plane for the rest of the episode (down to −60
points) and the agent stopped trying.

Reference pilot (``baseline_command``): scripted segment tracking (constant load factor
and roll to keep the lift in the plane, or fixed-rate roll while compensating for
gravity), then autopilot in level flight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from jetsim.aircraft.instruments import Instruments
from jetsim.control.autopilot import Autopilot, AutopilotTargets
from jetsim.control.fbw import HighLevelCommand

from jetfighter_rl.envs.rewards import RewardTerms, action_smoothness, angle_error
from jetfighter_rl.envs.tasks.base import DEG, FlightGeometry, InitialCondition, Task, Vec, clip

PITCH, ROLL = "pitch", "roll"
MANEUVERS: tuple[str, ...] = ("loop", "roll", "immelmann", "split_s")

# Entry conditions: (min, max altitude), (min, max speed) [m, m/s]
ENTRY: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "loop": ((3000, 7000), (240, 300)),
    "roll": ((3000, 8000), (180, 280)),
    "immelmann": ((3000, 7000), (250, 300)),
    "split_s": ((5500, 8500), (150, 190)),
}


def segments_for(maneuver: str, roll_sign: float) -> list[tuple[str, float]]:
    if maneuver == "loop":
        return [(PITCH, 2 * math.pi)]
    if maneuver == "roll":
        return [(ROLL, roll_sign * 2 * math.pi)]
    if maneuver == "immelmann":
        return [(PITCH, math.pi), (ROLL, roll_sign * math.pi)]
    if maneuver == "split_s":
        return [(ROLL, roll_sign * math.pi), (PITCH, -math.pi)]
    raise ValueError(f"Unknown maneuver: {maneuver!r} (available: {MANEUVERS})")


def _signed_angle(a: Vec, b: Vec, axis: Vec) -> float:
    """Signed angle from ``a`` to ``b`` about ``axis`` (unit vectors)."""
    return math.atan2(float(np.cross(a, b) @ axis), float(a @ b))


@dataclass
class AerobaticsTask(Task):
    name: str = "aerobatics"
    maneuvers: tuple[str, ...] = MANEUVERS  # maneuvers drawn at random for each episode
    episode_time: float = 45.0
    segment_tolerance: float = 5 * DEG
    exit_bank: float = 15 * DEG
    exit_gamma: float = 10 * DEG
    exit_heading: float = 30 * DEG  # exit heading tolerance (initial heading or its reverse)
    max_out_of_plane: float = math.sin(30 * DEG)  # |v·e₂| beyond which pitch no longer counts
    w_progress: float = 40.0  # spread over the whole maneuver
    w_plane: float = 0.05  # small: penalties accumulated over 45 s -> the agent stopped trying
    w_lift: float = 0.05
    w_track: float = 0.1
    w_exit: float = 0.2
    w_time: float = 0.01
    w_smooth: float = 0.05
    completion_bonus: float = 20.0
    pull_load_factor: float = 5.0  # reference pilot
    n_features: int = 17
    event_terms: frozenset[str] = frozenset({"progress", "success"})
    maneuver: str = field(default="loop", init=False)
    segments: list[tuple[str, float]] = field(default_factory=list, init=False)
    index: int = field(default=0, init=False)
    seg_progress: float = field(default=0.0, init=False)  # angle covered (signed)
    frame: Vec = field(default_factory=lambda: np.eye(3), init=False)  # columns e1, e2, e3
    origin: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _geo: FlightGeometry | None = field(default=None, init=False)
    _prev_theta: float = field(default=0.0, init=False)
    _prev_lift: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _seg_start_velocity: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _last_delta: float = field(default=0.0, init=False)
    _seg_best: float = field(default=0.0, init=False)  # best progress within the segment
    _completed: bool = field(default=False, init=False)
    _finished_time: float = field(default=math.inf, init=False)
    _exit_state: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _pending_maneuver: str | None = field(default=None, init=False)
    _ap_engaged: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.maneuvers = tuple(self.maneuvers)
        for m in self.maneuvers:
            segments_for(m, 1.0)  # validation

    @property
    def max_step_cost(self) -> float:
        shaping = max(self.w_plane + self.w_lift, self.w_track, 2 * self.w_exit)
        # moving backward costs at most ≈ 0.05 rad per step, over a total angle ≥ π
        return shaping + self.w_time + self.w_smooth + 0.016 * self.w_progress

    # ------------------------------------------------------------------
    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        m = self.maneuvers[int(rng.integers(len(self.maneuvers)))]
        self._pending_maneuver = m
        (h_lo, h_hi), (v_lo, v_hi) = ENTRY[m]
        v_hi = min(v_hi, 260.0) if six_dof else v_hi
        return InitialCondition(
            altitude=float(rng.uniform(h_lo, h_hi)),
            airspeed=float(rng.uniform(min(v_lo, v_hi - 10), v_hi)),
            heading=float(rng.uniform(-math.pi, math.pi)),
        )

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        m = self._pending_maneuver or self.maneuvers[int(rng.integers(len(self.maneuvers)))]
        self._pending_maneuver = None
        self.maneuver = m
        self.segments = segments_for(m, 1.0 if rng.random() < 0.5 else -1.0)
        v = geo.velocity_dir
        e1 = np.array([v[0], v[1], 0.0])
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e3 = np.array([0.0, 0.0, 1.0])
        self.frame = np.column_stack([e1, np.cross(e3, e1), e3])
        self.origin = geo.position.copy()
        self.index = 0
        self._completed = False
        self._finished_time = math.inf
        self._ap_engaged = False
        self._start_segment(geo)
        self._geo = geo

    def _start_segment(self, geo: FlightGeometry) -> None:
        self.seg_progress = 0.0
        self._last_delta = 0.0
        self._seg_best = 0.0
        self._prev_theta = self._plane_angle(geo)
        self._prev_lift = geo.lift_dir.copy()
        self._seg_start_velocity = geo.velocity_dir.copy()

    def _plane_angle(self, geo: FlightGeometry) -> float:
        v = geo.velocity_dir
        return math.atan2(-float(v @ self.frame[:, 2]), float(v @ self.frame[:, 0]))

    # ------------------------------------------------------------------
    @property
    def total_angle(self) -> float:
        return sum(abs(a) for _, a in self.segments)

    @property
    def segment(self) -> tuple[str, float] | None:
        return self.segments[self.index] if self.index < len(self.segments) else None

    def progress_fraction(self) -> float:
        done = sum(abs(a) for _, a in self.segments[: self.index])
        seg = self.segment
        if seg is not None:
            done += clip(math.copysign(1.0, seg[1]) * self.seg_progress, 0.0, abs(seg[1]))
        return done / self.total_angle

    def desired_lift(self, geo: FlightGeometry, sign: float) -> Vec:
        """Desired lift in a pitch segment: in the plane, toward the center."""
        d: Vec = np.asarray(sign * np.cross(self.frame[:, 1], geo.velocity_dir), dtype=np.float64)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else geo.lift_dir

    def lift_error(self, geo: FlightGeometry) -> float:
        """Signed angle lift -> desired lift about the velocity (0 outside pitch segments)."""
        seg = self.segment
        if seg is None or seg[0] != PITCH:
            return 0.0
        d = self.desired_lift(geo, math.copysign(1.0, seg[1]))
        return _signed_angle(geo.lift_dir, d, geo.velocity_dir)

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        self._geo = geo
        self._last_delta = 0.0
        seg = self.segment
        if seg is not None:
            kind, angle = seg
            if kind == PITCH:
                theta = self._plane_angle(geo)
                delta = angle_error(theta - self._prev_theta)
                self._prev_theta = theta
                # out of the maneuver plane, the projected angle is meaningless: without this
                # guard, an agent turned 90° and then "wound up" the angle within a few
                # seconds with small motions around the lateral axis (a loop in 6 s)
                if abs(float(geo.velocity_dir @ self.frame[:, 1])) > self.max_out_of_plane:
                    delta = 0.0
            else:
                delta = _signed_angle(self._prev_lift, geo.lift_dir, geo.velocity_dir)
            self._prev_lift = geo.lift_dir.copy()
            self.seg_progress += delta
            sign = math.copysign(1.0, angle)
            after = clip(sign * self.seg_progress, 0.0, abs(angle))
            # only new progress pays (segment record); moving backward costs nothing:
            # a partial attempt still pays off, which encourages trying
            self._last_delta = max(after - self._seg_best, 0.0)
            self._seg_best = max(self._seg_best, after)
            if sign * self.seg_progress >= abs(angle) - self.segment_tolerance:
                self.index += 1
                if self.segment is not None:
                    self._start_segment(geo)
        if self.segment is None and not self._completed and self._exit_ok(ins, geo):
            self._completed = True
            self._finished_time = t
            self._exit_state = self._exit_errors(ins, geo)

    def _exit_ok(self, ins: Instruments, geo: FlightGeometry) -> bool:
        heading_error = abs(self._exit_errors(ins, geo)[0])
        return (abs(angle_error(ins.bank)) < self.exit_bank and abs(ins.gamma) < self.exit_gamma
                and heading_error < self.exit_heading)  # fmt: skip

    def _exit_errors(self, ins: Instruments, geo: FlightGeometry) -> tuple[float, float, float]:
        """(heading deviation from the expected exit [rad], lateral offset [m], Δh [m])."""
        reverse = self.maneuver in ("immelmann", "split_s")
        e1 = self.frame[:, 0]
        expected = math.atan2(e1[1], e1[0]) + (math.pi if reverse else 0.0)
        offset = geo.position - self.origin
        return (
            angle_error(ins.course - expected),
            float(offset @ self.frame[:, 1]),
            float(-offset @ self.frame[:, 2]),
        )

    # ------------------------------------------------------------------
    def features(self, ins: Instruments) -> Vec:
        geo = self._geo
        assert geo is not None
        one_hot = [1.0 if self.maneuver == m else 0.0 for m in MANEUVERS]
        seg = self.segment
        if seg is not None:
            remaining = (abs(seg[1]) - math.copysign(1.0, seg[1]) * self.seg_progress) / math.pi
            seg_feats = [
                1.0 if seg[0] == PITCH else -1.0,
                math.copysign(1.0, seg[1]),
                clip(remaining, -1.0, 2.0),
            ]
        else:
            seg_feats = [0.0, 0.0, 0.0]
        offset = geo.position - self.origin
        return np.array(
            [
                *one_hot,
                *seg_feats,
                self.progress_fraction(),
                *(self.frame.T @ geo.velocity_dir),
                *(self.frame.T @ geo.lift_dir),
                self.lift_error(geo) / math.pi,
                clip(float(offset @ self.frame[:, 1]) / 500.0, -3.0, 3.0),
                clip(-float(offset @ self.frame[:, 2]) / 1000.0, -3.0, 3.0),
            ]
        )

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        geo = self._geo
        assert geo is not None
        terms: RewardTerms = {
            "progress": self.w_progress * self._last_delta / self.total_angle,
            "plane": 0.0,
            "lift": 0.0,
            "track": 0.0,
            "exit": 0.0,
            "time": -self.w_time,
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }
        seg = self.segment
        if seg is not None and seg[0] == PITCH:
            lateral = abs(float(geo.velocity_dir @ self.frame[:, 1]))
            terms["plane"] = -self.w_plane * min(lateral / math.sin(20 * DEG), 1.0)
            terms["lift"] = -self.w_lift * abs(self.lift_error(geo)) / math.pi
        elif seg is not None:
            cos_dev = clip(float(geo.velocity_dir @ self._seg_start_velocity), -1.0, 1.0)
            terms["track"] = -self.w_track * min(math.acos(cos_dev) / (20 * DEG), 1.0)
        else:
            terms["exit"] = -self.w_exit * (
                abs(angle_error(ins.bank)) / math.pi + min(abs(ins.gamma) / (30 * DEG), 1.0)
            )
        terms["success"] = self.completion_bonus if self._completed else 0.0
        return terms

    def done(self) -> bool:
        return self._completed

    def success(self, ins: Instruments) -> bool:
        return self._completed

    def metrics(self) -> dict[str, float]:
        heading, lateral, dh = self._exit_state if self._completed else (math.pi, 0.0, 0.0)
        return {
            "progress": self.progress_fraction() if self.segment else 1.0,
            "completion_time": self._finished_time if self._completed else self.episode_time,
            "exit_heading_error_deg": abs(math.degrees(heading)),
            "exit_lateral_offset": abs(lateral),
            "exit_altitude_change": dh,
        }

    # ------------------------------------------------------------------
    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        return AutopilotTargets(altitude=ins.altitude, bank=0.0)

    def baseline_command(
        self, ins: Instruments, autopilot: Autopilot, dt: float
    ) -> HighLevelCommand:
        """Scripted pilot: follows the segments, then brings the aircraft back to level flight."""
        geo = self._geo
        assert geo is not None
        seg = self.segment
        if seg is None:
            if not self._ap_engaged:
                autopilot.gamma_pid.reset(0.0)
                autopilot.bank_pid.reset(0.0)
                autopilot.speed_pid.reset(1.0)
                self._ap_engaged = True
            autopilot.targets = AutopilotTargets(altitude=ins.altitude, bank=0.0, throttle=1.0)
            return autopilot.high_level(ins, dt)
        kind, angle = seg
        sign = math.copysign(1.0, angle)
        remaining = abs(angle) - sign * self.seg_progress
        if kind == PITCH:
            roll_rate = clip(3.0 * self.lift_error(geo), -90 * DEG, 90 * DEG)
            return HighLevelCommand(nz=self.pull_load_factor, roll_rate=roll_rate, throttle=1.0)
        # roll: fixed roll rate, slowed at the end of the segment; the load factor
        # compensates for the gravity component along the lift (straight flight path)
        rate = sign * clip(2.5 * remaining + 20 * DEG, 0.0, 150 * DEG)
        nz = -float(geo.lift_dir[2]) * 1.0
        return HighLevelCommand(nz=nz, roll_rate=rate, throttle=1.0)

    def describe(self) -> dict[str, float]:
        return {
            "maneuver_index": float(MANEUVERS.index(self.maneuver)),
            "roll_sign": next((math.copysign(1.0, a) for k, a in self.segments if k == ROLL), 0.0),
        }
