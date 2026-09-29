"""Phase 7 : environnement Gymnasium JetEnv."""

import math

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from jetsim.control.fbw import HighLevelCommand  # noqa: E402
from jetsim.viz.recorder import replay  # noqa: E402

import jetfighter_rl.envs  # noqa: E402,F401  (enregistre les identifiants)
from jetfighter_rl.envs.baselines import AutopilotPolicy, RandomPolicy, evaluate  # noqa: E402
from jetfighter_rl.envs.jet_env import (  # noqa: E402
    EnvConfig,
    JetEnv,
    action_to_command,
    command_to_action,
)
from jetfighter_rl.envs.tasks import InitialCondition, make_task  # noqa: E402

DEG = math.pi / 180
COMBOS = [(t, m, a) for t in ("level", "heading_altitude") for m in ("3dof", "6dof")
          for a in ("hierarchical", "low_level")]  # fmt: skip


@pytest.mark.parametrize(("task", "model", "mode"), COMBOS)
def test_check_env(task, model, mode):
    env = JetEnv(EnvConfig(task=task, model=model, action_mode=mode))
    check_env(env, skip_render_check=True)
    n_act = 4 if (model == "6dof" and mode == "low_level") else 3
    assert env.action_space.shape == (n_act,)


def test_registered_ids_and_overrides():
    env = gym.make("JetFighter/HeadingAltitude-v0", model="6dof")
    assert env.unwrapped.six_dof and env.unwrapped.task.name == "heading_altitude"
    env = gym.make("JetFighter/Level-v0", action_mode="low_level")
    assert not env.unwrapped.hierarchical


def test_invalid_config():
    with pytest.raises(ValueError):
        JetEnv(EnvConfig(model="2dof"))
    with pytest.raises(ValueError):
        JetEnv(EnvConfig(action_mode="direct"))
    with pytest.raises(ValueError):
        make_task("dogfight")


def test_action_command_mapping():
    cmd = action_to_command(np.zeros(3), 180 * DEG)
    assert (cmd.nz, cmd.roll_rate, cmd.throttle) == (1.0, 0.0, 0.5)
    assert action_to_command(np.array([1.0, -1.0, 1.0]), 180 * DEG).nz == 9.0
    assert action_to_command(np.array([-1.0, 0.0, -1.0]), 180 * DEG).nz == -3.0
    for c in (HighLevelCommand(4.0, 1.0, 0.8), HighLevelCommand(-1.0, -2.0, 0.1)):
        back = action_to_command(command_to_action(c, 180 * DEG), 180 * DEG)
        assert (back.nz, back.roll_rate, back.throttle) == pytest.approx((c.nz, c.roll_rate,
                                                                          c.throttle))  # fmt: skip


def test_reset_is_seeded():
    env = JetEnv(EnvConfig(task="heading_altitude"))
    o1, i1 = env.reset(seed=42)
    o2, i2 = env.reset(seed=42)
    o3, _ = env.reset(seed=43)
    np.testing.assert_array_equal(o1, o2)
    assert i1["targets"] == i2["targets"]
    assert not np.array_equal(o1, o3)


@pytest.mark.parametrize("model", ["3dof", "6dof"])
def test_initial_condition_is_honoured(model):
    env = JetEnv(EnvConfig(task="level", model=model))
    ic = InitialCondition(altitude=4000, airspeed=220, heading=1.0, gamma=10 * DEG,
                          bank=60 * DEG, roll_rate=0.3)  # fmt: skip
    env.reset(seed=0, options={"initial_condition": ic})
    ins = env.instruments
    assert ins.altitude == pytest.approx(4000)
    assert ins.tas == pytest.approx(220)
    assert ins.bank == pytest.approx(60 * DEG, abs=1e-6)
    assert ins.gamma == pytest.approx(10 * DEG, abs=1e-6)
    assert ins.course == pytest.approx(1.0, abs=1e-6)
    assert math.hypot(ins.p, ins.r) == pytest.approx(0.3, rel=0.05)  # roulis autour de la vitesse


def test_truncation_and_episode_info():
    env = JetEnv(EnvConfig(task="level", episode_time=1.0))
    env.reset(seed=0)
    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert not terminated
        assert set(info["reward_terms"]) >= {"bank", "gamma", "speed", "smoothness", "success"}
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert truncated and info["t"] == pytest.approx(1.0)
    assert "episode_terms" in info and "is_success" in info


def test_crash_terminates_with_penalty():
    env = JetEnv(EnvConfig(task="level"))
    ic = InitialCondition(altitude=800, airspeed=250, gamma=-20 * DEG)
    env.reset(seed=0, options={"initial_condition": ic})
    push = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # −3 g, plein gaz : piqué
    for _ in range(300):
        obs, _, terminated, truncated, info = env.step(push)
        if terminated or truncated:
            break
    remaining = (env.episode_time - info["t"]) / env.cfg.agent_dt
    horizon = 1 / (1 - env.cfg.discount)
    expected = env.task.crash_penalty - env.task.max_step_cost * min(remaining, horizon)
    assert terminated and info["reward_terms"]["crash"] == pytest.approx(expected)
    assert info["violation"] and not info["is_success"]
    assert np.all(np.isfinite(obs))


@pytest.mark.parametrize("model", ["3dof", "6dof"])
def test_observations_stay_bounded_under_random_actions(model):
    env = JetEnv(EnvConfig(task="heading_altitude", model=model))
    obs, _ = env.reset(seed=3)
    policy = RandomPolicy(env, seed=3)
    for _ in range(150):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        assert obs.dtype == np.float32 and env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()


def test_noisy_sensors_change_observation_not_reward():
    clean = JetEnv(EnvConfig(task="heading_altitude"))
    noisy = JetEnv(EnvConfig(task="heading_altitude", sensors="realistic"))
    o1, _ = clean.reset(seed=5)
    o2, _ = noisy.reset(seed=5)
    assert not np.allclose(o1, o2)
    a = np.zeros(3, dtype=np.float32)
    assert clean.step(a)[1] == pytest.approx(noisy.step(a)[1])


def test_low_level_6dof_mapping():
    env = JetEnv(EnvConfig(model="6dof", action_mode="low_level"))
    env.reset(seed=0)
    u = env._low_level_controls(np.array([1.0, -1.0, 0.5, 0.0]))
    cs = env.params.control_surfaces
    np.testing.assert_allclose(u, [1.0, -cs["elevator"].max, 0.5 * cs["aileron"].max, 0.0])


def test_recording_replays_exactly():
    env = JetEnv(EnvConfig(task="level", episode_time=2.0, record=True))
    env.reset(seed=1)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([0.2, 0.3, 0.0], dtype=np.float32))
        done = terminated or truncated
    rec = env.recording()
    assert rec is not None and rec.metadata["task"] == "level"
    assert rec.duration == pytest.approx(2.0)
    np.testing.assert_allclose(replay(rec), rec.states, atol=1e-9)


@pytest.mark.parametrize("task", ["level", "heading_altitude"])
def test_autopilot_baseline_solves_task_and_beats_random(task):
    env = JetEnv(EnvConfig(task=task, episode_time=60.0 if task != "level" else None))
    ap = evaluate(env, AutopilotPolicy(env), episodes=3, seed=10)
    rnd = evaluate(env, RandomPolicy(env), episodes=3, seed=10)
    assert ap.crash_rate == 0.0
    assert ap.mean_return > rnd.mean_return + 50
    if task == "level":
        assert ap.success_rate == 1.0


def test_autopilot_policy_requires_hierarchical_mode():
    with pytest.raises(ValueError):
        AutopilotPolicy(JetEnv(EnvConfig(action_mode="low_level")))


@pytest.mark.parametrize(("speed", "action", "limit"), [(290, -1.0, -3.6), (290, 1.0, 9.2)])
def test_fbw_load_factor_limiter_does_not_overshoot(speed, action, limit):
    """Échelon de n_z à pleine commande à grande vitesse : pas de sortie d'enveloppe."""
    env = JetEnv(EnvConfig(task="level", model="6dof", episode_time=3.0))
    env.reset(seed=0, options={"initial_condition": InitialCondition(5000, speed)})
    extreme = 1.0
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.array([action, 0, 0.5], np.float32))
        nz = env.instruments.nz
        extreme = min(extreme, nz) if action < 0 else max(extreme, nz)
        done = terminated or truncated
    assert not info["violation"]
    assert (extreme > limit) if action < 0 else (extreme < limit)


def test_expo_action_mapping_option():
    a = np.array([0.5, -0.5, 0.0])
    cmd = action_to_command(a, 180 * DEG, exponent=3.0)
    assert cmd.nz == pytest.approx(1 + 8 * 0.125) and cmd.roll_rate == pytest.approx(
        -0.125 * math.pi
    )
    np.testing.assert_allclose(command_to_action(cmd, 180 * DEG, exponent=3.0), a, atol=1e-12)
    env = JetEnv(EnvConfig(action_exponent=3.0))
    check_env(env, skip_render_check=True)


def test_attitude_features_are_continuous_through_the_vertical():
    """Passage de la verticale en looping : les angles d'Euler basculent de 180°, pas les
    composantes de la pesanteur utilisées dans l'observation."""
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": "loop"}))
    policy = AutopilotPolicy(env)
    obs, _ = env.reset(seed=3)
    policy.reset()
    prev, jumps_euler, jump_obs = None, 0, 0.0
    for _ in range(300):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        bank = env.instruments.bank
        if prev is not None:
            jumps_euler += abs(math.remainder(bank - prev[0], 2 * math.pi)) > 2.0
            jump_obs = max(jump_obs, float(np.max(np.abs(obs[4:10] - prev[1]))))
        prev = (bank, obs[4:10].copy())
        if terminated or truncated:
            break
    assert jumps_euler >= 1  # l'inclinaison μ bascule bien au sommet de la boucle
    assert jump_obs < 0.5  # mais pas l'observation


def test_old_runs_keep_euler_attitude_features(tmp_path):
    pytest.importorskip("stable_baselines3")
    from jetfighter_rl.training.config import TrainConfig
    from jetfighter_rl.training.train import run_config_of

    TrainConfig(name="old", env={"task": "level"}).save(tmp_path / "config.yaml")
    assert run_config_of(tmp_path / "seed0" / "best_model.zip").env["attitude_features"] == "euler"
