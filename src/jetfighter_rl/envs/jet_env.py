"""Gymnasium environment ``JetEnv``: the F-16 (3-DOF or 6-DOF) flown by an RL agent.

Two action modes ( hierarchical first, low-level second),
always in ``Box(−1, 1)``:

* ``"hierarchical"`` (default) — [n_z, roll rate, throttle], executed by the inner loop
  (fly-by-wire in 6-DOF); same action space for both models.
  0 on the pitch axis = the n_z that holds the flight path (cos γ / cos μ: 1 g in level
  flight, 2 g at 60° of bank); +1 = +9 g; −1 = −3 g. Without this compensation, the policy
  had to output 1/cos μ itself to within a hundredth to avoid climbing or descending in a
  turn: PPO did not learn the heading/altitude task (``nz_neutral="one_g"`` for the old
  behavior);
* ``"low_level"`` — 6-DOF: [throttle, δe, δa, δr] (direct control surfaces, no limiter);
  3-DOF: [throttle, commanded α, roll rate].

Rates: physics 100 Hz, inner loop and envelope monitoring 50 Hz, agent decisions 10 Hz
(adjustable in ``EnvConfig``).

Observations (``Box(−10, 10)``, float32): normalized own state (speed, altitude, flight-path
angle, bank and roll as sin/cos, α, β, p, q, r, n_z, Ps, power), last action, then the
task-specific quantities (errors relative to the target). They are computed from the
measurements (sensors, possibly noisy); the reward and episode terminations use the
true values.

Episode end: ``terminated`` if the envelope is violated (crash, overload, stall…,
penalty ``task.crash_penalty`` plus the maximum cost of the remaining steps, capped at the
agent's 1/(1−γ) horizon) or if the task is complete (``task.done()``: last waypoint
passed, aerobatic maneuver finished); ``truncated`` at the task's maximum duration. At the
end of an episode, ``info`` contains ``is_success`` (roadmap criterion), ``metrics``
(recovery time, overshoots…) and ``episode_terms`` (cumulative reward per term).

Usage ::

    env = JetEnv(EnvConfig(task="heading_altitude", model="3dof"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.envelope import EnvelopeMonitor
from jetsim.aircraft.instruments import (
    Instruments,
    position_ned,
    read_instruments,
    wind_axes,
)
from jetsim.aircraft.params import SENSORS_DIR, load_aircraft, load_envelope
from jetsim.aircraft.sensors import SensorSuite
from jetsim.control.fbw import HighLevelCommand, make_inner_loop
from jetsim.core.frames import dcm_from_euler, dcm_wind_to_body, quat_from_dcm
from jetsim.viz.recorder import FlightRecorder, FlightRecording

from jetfighter_rl.envs.rewards import accumulate, total
from jetfighter_rl.envs.tasks import FlightGeometry, InitialCondition, Task, make_task

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180
OBS_LIMIT = 10.0
N_OWN_SHIP = 19  # aircraft own-state quantities in the observation
REFERENCE_DT = 0.1  # [s] reference rate for per-step rewards


@dataclass
class EnvConfig:
    task: str = "level"
    model: str = "3dof"  # "3dof" | "6dof"
    action_mode: str = "hierarchical"  # "hierarchical" | "low_level"
    agent_dt: float = 0.1  # agent decision period [s]
    control_dt: float = 0.02  # inner loop period [s]
    physics_dt: float = 0.01  # integration step [s]
    episode_time: float | None = None  # None: the task's duration
    sensors: str | None = None  # None (perfect), "realistic" or path to a YAML
    xcg: float | None = None  # 6-DOF CG position (None: value from the YAML)
    max_roll_rate: float = 180 * DEG  # max commandable roll rate (hierarchical mode)
    action_exponent: float = 1.0  # "expo" curve for n_z and roll (1: linear)
    attitude_features: str = "gravity"  # "gravity" (continuous) | "euler" (singular when vertical)
    nz_neutral: str = "compensated"  # n_z at zero action: "compensated" (cos γ/cos μ) | "one_g"
    record: bool = False  # records the flights (cf. ``JetEnv.recording``)
    discount: float = 0.99  # agent's γ: sets the horizon covered by the crash penalty
    task_kwargs: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Normalized action <-> physical command mapping
# --------------------------------------------------------------------------
NZ_MAX, NZ_MIN = 9.0, -3.0


# --------------------------------------------------------------------------
# Low-level (step 8.6): no more fly-by-wire to protect the structure
# --------------------------------------------------------------------------
LIMIT_WEIGHTS = {"overload": 0.5, "sideslip": 0.2}
NZ_WARN_HIGH, NZ_WARN_LOW = 8.0, -2.0  # penalty onset (structural limits: +10/−4 g)


def limit_terms(ins: Instruments) -> dict[str, float]:
    """Penalties specific to low-level mode: approaching the structural limits and
    sideslip (no limiter or yaw damper takes care of them anymore)."""
    over = max(ins.nz - NZ_WARN_HIGH, NZ_WARN_LOW - ins.nz, 0.0)
    return {
        "overload": -LIMIT_WEIGHTS["overload"] * min(over / 2.0, 1.0),
        "sideslip": -LIMIT_WEIGHTS["sideslip"] * min((ins.beta / (5 * DEG)) ** 2, 1.0),
    }


def _expo(x: float, exponent: float) -> float:
    return math.copysign(abs(x) ** exponent, x)


def gravity_compensation(gamma: float, bank: float) -> float:
    """Load factor that holds the flight path in a turn: cos γ / cos μ.

    |cos μ| is bounded below by 0.25 (compensation of at most 4 g); inverted (|μ| > 90°),
    the compensation is negative (you "push" to hold the flight path). The result is
    clipped to [−2, 4] g, strictly within the command range.
    """
    c = math.cos(bank)
    c = math.copysign(max(abs(c), 0.25), c)
    return min(max(math.cos(gamma) / c, -2.0), 4.0)


def action_to_command(
    a: Vec, max_roll_rate: float, exponent: float = 1.0, nz_bias: float = 1.0
) -> HighLevelCommand:
    """Hierarchical action [−1, 1]³ -> (n_z, roll rate, throttle).

    ``nz_bias`` is the load factor commanded when a₀ = 0: 1 g (``"one_g"`` mode) or the
    gravity compensation cos γ / cos μ (``"compensated"`` mode, the environment's default:
    zero action = flight path held, even when banked). a₀ = ±1 always gives +9 / −3 g.

    Option ``exponent`` > 1: "expo" curve (|a|^exponent, as on RC transmitters) on the
    load factor and roll, finer around neutral. Tried with 3: no measurable gain for PPO,
    and imitating the autopilot becomes much less accurate (the inverse, a cube root, is
    very steep near zero); linear by default.
    """
    a0 = _expo(float(a[0]), exponent)
    nz = nz_bias + a0 * ((NZ_MAX - nz_bias) if a0 >= 0 else (nz_bias - NZ_MIN))
    return HighLevelCommand(nz=nz, roll_rate=_expo(float(a[1]), exponent) * max_roll_rate,
                            throttle=0.5 * (float(a[2]) + 1.0))  # fmt: skip


def command_to_action(
    cmd: HighLevelCommand, max_roll_rate: float, exponent: float = 1.0, nz_bias: float = 1.0
) -> Vec:
    """Inverse of ``action_to_command`` (clipped to [−1, 1])."""
    dn = cmd.nz - nz_bias
    a0 = dn / (NZ_MAX - nz_bias) if dn >= 0 else dn / (nz_bias - NZ_MIN)
    a1 = cmd.roll_rate / max_roll_rate
    inv = 1.0 / exponent
    a = np.array([_expo(min(max(a0, -1.0), 1.0), inv), _expo(min(max(a1, -1.0), 1.0), inv),
                  2.0 * cmd.throttle - 1.0])  # fmt: skip
    return np.clip(a, -1.0, 1.0)


class JetEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Learning environment (see the module)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}  # noqa: RUF012  (API Gymnasium)

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = replace(config or EnvConfig(), **overrides)
        if cfg.model not in ("3dof", "6dof"):
            raise ValueError("model must be '3dof' or '6dof'.")
        if cfg.attitude_features not in ("gravity", "euler"):
            raise ValueError("attitude_features must be 'gravity' or 'euler'.")
        if cfg.nz_neutral not in ("compensated", "one_g"):
            raise ValueError("nz_neutral must be 'compensated' or 'one_g'.")
        if cfg.action_mode not in ("hierarchical", "low_level"):
            raise ValueError("action_mode must be 'hierarchical' or 'low_level'.")
        self.cfg = cfg
        self.render_mode = render_mode
        self.six_dof = cfg.model == "6dof"
        self.hierarchical = cfg.action_mode == "hierarchical"

        params = load_aircraft("f16")
        self.model: d3.PointMassAircraft | d6.F16SixDof = (
            d6.F16SixDof(params, xcg=cfg.xcg) if self.six_dof else d3.PointMassAircraft(params)
        )
        self.params = params
        self.inner = make_inner_loop(self.model)
        self.monitor = EnvelopeMonitor(load_envelope("f16", six_dof=self.six_dof))
        self.task: Task = make_task(cfg.task, **cfg.task_kwargs)
        self.sensors = self._make_sensors(cfg.sensors)

        self.n_control = round(cfg.agent_dt / cfg.control_dt)
        self.n_physics = round(cfg.control_dt / cfg.physics_dt)
        if not (math.isclose(self.n_control * cfg.control_dt, cfg.agent_dt)
                and math.isclose(self.n_physics * cfg.physics_dt, cfg.control_dt)):  # fmt: skip
            raise ValueError("agent_dt, control_dt and physics_dt must be multiples of each other.")
        self.episode_time = cfg.episode_time or self.task.episode_time

        n_act = 3 if (self.hierarchical or not self.six_dof) else 4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        # 6-DOF low-level: actual control surface positions (δe, δa, δr) in the observation
        self.n_surfaces = 3 if (self.six_dof and not self.hierarchical) else 0
        # per-step costs scaled to the rate: returns stay comparable between a 10 Hz agent
        # (hierarchical) and a 50 Hz agent (direct control surfaces)
        self.reward_scale = cfg.agent_dt / REFERENCE_DT
        self.limit_cost = LIMIT_WEIGHTS["overload"] + LIMIT_WEIGHTS["sideslip"]
        if self.hierarchical:
            self.limit_cost = 0.0
        n_obs = N_OWN_SHIP + n_act + self.n_surfaces + self.task.n_features
        self.n_act, self.n_obs = n_act, n_obs
        self.observation_space = spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape=(n_obs,),
                                            dtype=np.float32)  # fmt: skip

        self.x: Vec = np.zeros(0)
        self.t = 0.0
        self.instruments: Instruments | None = None
        self.previous_action: Vec = np.zeros(n_act)
        self.episode_terms: dict[str, float] = {}
        self.last_violation: str = ""
        self._recorder: FlightRecorder | None = None
        self._last_recording: FlightRecording | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _make_sensors(spec: str | None) -> SensorSuite:
        if spec is None:
            return SensorSuite.perfect()
        path = SENSORS_DIR / f"{spec}.yaml" if "/" not in spec else spec
        return SensorSuite.from_yaml(path)

    def _initial_state(self, ic: InitialCondition) -> tuple[Vec, Vec]:
        """Initial state from initial conditions.

        Starts from trim at the requested flight-path angle γ; if there is none (dive too
        steep: the aircraft accelerates even at idle), from level trim, onto which the
        flight-path angle, bank and roll rate are then imposed. The starting state is then
        not a trim point, which is intended for the recovery tasks.
        Raises ``TrimError`` if even level trim is impossible (speed outside the envelope).
        """
        if isinstance(self.model, d6.F16SixDof):
            try:
                x, u = self.model.trim(ic.altitude, ic.airspeed, gamma=ic.gamma)
            except d6.TrimError:
                x, u = self.model.trim(ic.altitude, ic.airspeed)
            alpha = math.atan2(x[d6.W], x[d6.U])
            c_bw = dcm_wind_to_body(alpha, 0.0)
            c_nw = dcm_from_euler(ic.bank, ic.gamma, ic.heading)
            x[d6.QUAT] = quat_from_dcm(c_nw @ c_bw.T)
            x[d6.RATES] = c_bw @ np.array([ic.roll_rate, 0.0, 0.0])
            return x, u
        try:
            alpha, power = self.model.trim(ic.altitude, ic.airspeed, ic.gamma)
        except d3.TrimError:
            alpha, power = self.model.trim(ic.altitude, ic.airspeed)
        x = self.model.make_state(altitude=ic.altitude, airspeed=ic.airspeed, gamma=ic.gamma,
                                  heading=ic.heading, bank=ic.bank, alpha=alpha, power=power,
                                  roll_rate=ic.roll_rate)  # fmt: skip
        return x, np.array([power, alpha, 0.0])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        for _ in range(50):
            ic = options.get("initial_condition") or self.task.sample_initial(rng, self.six_dof)
            try:
                self.x, u0 = self._initial_state(ic)
                break
            except (d3.TrimError, d6.TrimError):
                if "initial_condition" in options:
                    raise
        else:  # pragma: no cover - practically impossible with the tasks' ranges
            raise RuntimeError("Could not find a trimmed initial state.")

        self.t = 0.0
        self.inner.reset(self.x)
        self.monitor.reset()
        self.sensors.reset(np.random.default_rng(rng.integers(2**32)))
        self.instruments = read_instruments(self.model, self.x)
        self.task.reset(self.instruments, self.geometry(), rng, self.six_dof)
        self.previous_action = self._neutral_action(u0)
        self.episode_terms = {}
        self.last_violation = ""
        if self.cfg.record:
            self._recorder = FlightRecorder(
                self.model, physics_dt=self.cfg.physics_dt, substeps=self.n_physics,
                metadata={"task": self.task.name, **self.task.describe()},
            )  # fmt: skip
        info = {"initial_condition": ic.__dict__, "targets": self.task.describe()}
        return self._observation(), info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.instruments is None:
            raise RuntimeError("Call reset() before step().")
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        u_direct = np.zeros(0) if self.hierarchical else self._low_level_controls(a)

        violation = None
        ins = self.instruments
        u: Vec = np.zeros(0)
        for _ in range(self.n_control):
            if self.hierarchical:
                command = action_to_command(a, self.cfg.max_roll_rate, self.cfg.action_exponent,
                                            self.nz_bias(ins))  # fmt: skip
                u_next = self.inner(ins, command, self.cfg.control_dt)
            else:
                u_next = u_direct
            assert u_next is not None
            u = u_next
            if self._recorder is not None:
                self._recorder.record(self.t, self.x, u)
            for _ in range(self.n_physics):
                self.x = self.model.step(self.x, u, self.cfg.physics_dt)
            self.t = round(self.t + self.cfg.control_dt, 9)
            ins = read_instruments(self.model, self.x)
            violation = self.monitor.check(ins, self.cfg.control_dt, state=self.x)
            if violation is not None:
                break
        self.instruments = ins
        if violation is None:
            self.task.update(ins, self.geometry(), self.t)

        terms = self.task.reward(ins, a, self.previous_action)
        if not self.hierarchical:
            terms |= limit_terms(ins)
        terms = {k: (v if k in self.task.event_terms else v * self.reward_scale)
                 for k, v in terms.items()}  # fmt: skip
        crashed = violation is not None
        terminated = crashed or self.task.done()
        if crashed:
            # fixed penalty + maximum cost of the remaining steps (capped at the agent's 1/(1−γ)
            # horizon): crashing is always worse than continuing to fly, even badly
            # (otherwise, far from the target, the agent would learn to cut the episode short)
            remaining = max(self.episode_time - self.t, 0.0) / self.cfg.agent_dt
            horizon = 1.0 / max(1.0 - self.cfg.discount, 1e-3)
            step_cost = (self.task.max_step_cost + self.limit_cost) * self.reward_scale
            terms["crash"] = self.task.crash_penalty - step_cost * min(remaining, horizon)
            self.last_violation = self.monitor.message
        truncated = not terminated and self.t >= self.episode_time - 1e-9
        reward = total(terms)
        accumulate(self.episode_terms, terms)
        self.previous_action = a

        info: dict[str, Any] = {"reward_terms": terms, "t": self.t}
        if terminated or truncated:
            info["episode_terms"] = dict(self.episode_terms)
            info["is_success"] = bool(not crashed and self.task.episode_success(ins))
            info["violation"] = self.last_violation
            info["metrics"] = self.task.metrics()
            if self._recorder is not None:
                if self.last_violation:
                    self._recorder.event(self.t, self.last_violation)
                self._recorder.record(self.t, self.x, u)
                self._last_recording = self._recorder.finish()
        obs = self._observation() if not crashed or np.all(np.isfinite(self.x)) else \
            np.zeros(self.n_obs, dtype=np.float32)  # fmt: skip
        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self.instruments is None:
            return None
        i = self.instruments
        return (f"t={self.t:6.1f}s h={i.altitude:6.0f}m V={i.tas:5.0f}m/s "
                f"μ={math.degrees(i.bank):+6.1f}° γ={math.degrees(i.gamma):+5.1f}° "
                f"n={i.nz:+5.2f}g α={math.degrees(i.alpha):+5.1f}°")  # fmt: skip

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _low_level_controls(self, a: Vec) -> Vec:
        throttle = 0.5 * (a[0] + 1.0)
        if isinstance(self.model, d6.F16SixDof):
            cs = self.params.control_surfaces
            return np.array([throttle, a[1] * cs["elevator"].max, a[2] * cs["aileron"].max,
                             a[3] * cs["rudder"].max])  # fmt: skip
        lim = self.params.limits
        alpha = a[1] * (lim.alpha_max if a[1] >= 0 else -lim.alpha_min)
        return np.array([throttle, alpha, a[2] * lim.roll_rate_max])

    @property
    def has_reference(self) -> bool:
        """A reference policy exists (hierarchical, or 6-DOF low-level)."""
        return self.hierarchical or self.six_dof

    def controls_to_action(self, u: Vec) -> Vec:
        """Inverse of ``_low_level_controls`` (6-DOF): model command -> action."""
        cs = self.params.control_surfaces
        a = np.array([2.0 * u[0] - 1.0, u[1] / cs["elevator"].max, u[2] / cs["aileron"].max,
                      u[3] / cs["rudder"].max])  # fmt: skip
        return np.clip(a, -1.0, 1.0)

    def _neutral_action(self, u0: Vec) -> Vec:
        """Action matching the initial trim (avoids an artificial smoothness penalty on the
        first step)."""
        if self.hierarchical:
            thr = float(u0[0])
            if isinstance(self.model, d6.F16SixDof):
                thr = self.model.engine.throttle_for_power(float(self.x[d6.POWER]))
            assert self.instruments is not None
            bias = self.nz_bias(self.instruments)
            return command_to_action(HighLevelCommand(bias, 0.0, min(max(thr, 0.0), 1.0)),
                                     self.cfg.max_roll_rate, self.cfg.action_exponent,
                                     bias)  # fmt: skip
        if self.six_dof:
            return self.controls_to_action(u0)
        return np.zeros(self.n_act)

    def nz_bias(self, ins: Instruments) -> float:
        """Load factor commanded at zero action (see ``action_to_command``)."""
        if self.cfg.nz_neutral == "compensated":
            return gravity_compensation(ins.gamma, ins.bank)
        return 1.0

    def _observation(self) -> npt.NDArray[np.float32]:
        assert self.instruments is not None
        m = self.sensors.measure(self.instruments)
        if self.cfg.attitude_features == "gravity":
            # gravity direction in wind then body axes: continuous at any attitude
            # (Euler angles flip by 180° when passing through the vertical)
            cg, cp = math.cos(m.gamma), math.cos(m.pitch)
            attitude = [-math.sin(m.gamma), math.sin(m.bank) * cg, math.cos(m.bank) * cg,
                        -math.sin(m.pitch), math.sin(m.roll) * cp, math.cos(m.roll) * cp,
                        ]  # fmt: skip
            last = []
        else:
            attitude = [math.sin(m.gamma), math.sin(m.bank), math.cos(m.bank),
                        math.sin(m.roll), math.cos(m.roll), math.sin(m.pitch)]  # fmt: skip
            last = [math.cos(m.pitch)]
        own = np.array([
            (m.tas - 200.0) / 100.0,
            (m.altitude - 6000.0) / 4000.0,
            (m.mach - 0.7) / 0.5,
            m.vertical_speed / 100.0,
            *attitude,
            m.alpha / 0.4,
            m.beta / 0.2,
            m.p / 3.0, m.q / 1.0, m.r / 1.0,
            (m.nz - 1.0) / 4.0,
            m.specific_excess_power / 100.0,
            2.0 * m.power - 1.0,
            *last,
        ])  # fmt: skip
        if len(own) < N_OWN_SHIP:  # same size in both modes
            own = np.concatenate([own, np.zeros(N_OWN_SHIP - len(own))])
        parts = [own, self.previous_action]
        if self.n_surfaces:
            cs = self.params.control_surfaces
            parts.append(np.array([self.x[d6.DE] / cs["elevator"].max,
                                   self.x[d6.DA] / cs["aileron"].max,
                                   self.x[d6.DR] / cs["rudder"].max]))  # fmt: skip
        obs = np.concatenate([*parts, self.task.features(m)])
        return np.clip(np.nan_to_num(obs), -OBS_LIMIT, OBS_LIMIT).astype(np.float32)

    def geometry(self) -> FlightGeometry:
        """True position and wind frame (for the navigation and aerobatics tasks)."""
        return FlightGeometry(position_ned(self.model, self.x), wind_axes(self.model, self.x))

    def recording(self) -> FlightRecording | None:
        """Recording of the last finished episode (if ``record=True``)."""
        return self._last_recording
