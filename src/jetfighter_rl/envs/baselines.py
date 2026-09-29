"""Reference policies and evaluation.

* ``AutopilotPolicy``: the phase 6 autopilot (or the task's scripted pilot, for
  aerobatics), plugged into the environment's hierarchical action space. It is the
  reference to beat (and the proof that the task is feasible with this reward).
* ``RandomPolicy``: uniform actions; gives the performance floor.
* ``evaluate``: mean return, success and crash rates over episodes with fixed seeds.

A policy is a callable ``action = policy(observation)``; ``reset()`` is called after each
``env.reset()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from jetsim.control.autopilot import Autopilot
from jetsim.control.fbw import FlyByWire

from jetfighter_rl.envs.jet_env import JetEnv, command_to_action

Obs = npt.NDArray[np.float32]


class Policy(Protocol):
    def reset(self) -> None: ...
    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]: ...


class AutopilotPolicy:
    """Autopilot -> normalized hierarchical action (true measurements from the env)."""

    def __init__(self, env: Any) -> None:
        self.env: JetEnv = env.unwrapped
        if not self.env.hierarchical:
            raise ValueError("The autopilot policy requires hierarchical mode.")
        self.ap = Autopilot(self.env.model)

    def reset(self) -> None:
        assert self.env.instruments is not None
        self.ap.reset(self.env.x, self.env.instruments)

    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]:
        ins = self.env.instruments
        assert ins is not None
        cmd = self.env.task.baseline_command(ins, self.ap, self.env.cfg.agent_dt)
        cfg = self.env.cfg
        bias = self.env.nz_bias(ins)
        action = command_to_action(cmd, cfg.max_roll_rate, cfg.action_exponent, bias)
        return action.astype(np.float32)


class LowLevelReferencePolicy:
    """Low-level mode reference (6-DOF): autopilot (or scripted pilot) **and** fly-by-wire,
    whose control surface commands are returned.

    This is the "hierarchical agent + fly-by-wire" that the roadmap proposes to imitate to
    kick off step 8.6. The fly-by-wire is tuned for 50 Hz: at 10 Hz it oscillates and the
    aircraft crashes, hence ``agent_dt = 0.02`` in the low-level configurations.
    """

    def __init__(self, env: Any) -> None:
        self.env: JetEnv = env.unwrapped
        if self.env.hierarchical or not self.env.six_dof:
            raise ValueError("Low-level reference: low_level mode on the 6-DOF only.")
        self.ap = Autopilot(self.env.model)
        self.fbw = FlyByWire(self.env.model)  # type: ignore[arg-type]

    def reset(self) -> None:
        assert self.env.instruments is not None
        self.ap.reset(self.env.x, self.env.instruments)
        self.fbw.reset(self.env.x)

    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]:
        ins = self.env.instruments
        assert ins is not None
        dt = self.env.cfg.agent_dt
        cmd = self.env.task.baseline_command(ins, self.ap, dt)
        u = self.fbw(ins, cmd, dt)
        return self.env.controls_to_action(u).astype(np.float32)


def reference_policy(env: Any) -> AutopilotPolicy | LowLevelReferencePolicy:
    """Reference policy matching the environment's action mode."""
    return AutopilotPolicy(env) if env.unwrapped.hierarchical else LowLevelReferencePolicy(env)


class RandomPolicy:
    def __init__(self, env: Any, seed: int = 0) -> None:
        self.space = env.action_space
        self.space.seed(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: Obs) -> npt.NDArray[np.float32]:
        return np.asarray(self.space.sample(), dtype=np.float32)


@dataclass(frozen=True)
class Evaluation:
    mean_return: float
    std_return: float
    success_rate: float
    crash_rate: float
    mean_length: float  # [s]
    returns: tuple[float, ...]
    metrics: dict[str, float] = field(default_factory=dict)  # means of the task metrics

    def __str__(self) -> str:
        return (f"return {self.mean_return:8.1f} ± {self.std_return:5.1f} | "
                f"success {100 * self.success_rate:5.1f} % | "
                f"crash {100 * self.crash_rate:5.1f} % | "
                f"length {self.mean_length:5.1f} s")  # fmt: skip


def evaluate(
    env: Any,
    policy: Policy | Callable[[Obs], npt.NDArray[np.float32]],
    episodes: int = 10,
    seed: int = 1000,
) -> Evaluation:
    """Evaluates a policy over ``episodes`` episodes (seeds seed, seed+1, …)."""
    returns, successes, crashes, lengths = [], 0, 0, []
    metrics: dict[str, list[float]] = {}
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        if hasattr(policy, "reset"):
            policy.reset()
        done, ep_return = False, 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
        successes += int(info.get("is_success", False))
        crashes += int(bool(info.get("violation")))
        lengths.append(info["t"])
        for name, value in info.get("metrics", {}).items():
            metrics.setdefault(name, []).append(float(value))
    arr = np.array(returns)
    return Evaluation(float(arr.mean()), float(arr.std()), successes / episodes,
                      crashes / episodes, float(np.mean(lengths)), tuple(returns),
                      {k: float(np.mean(v)) for k, v in metrics.items()})  # fmt: skip
