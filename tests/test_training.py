"""Phase 8 : outillage d'entraînement (configurations, entraînement, transfert, évaluation)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv
from jetfighter_rl.training.config import TrainConfig
from jetfighter_rl.training.evaluate import (
    compare,
    export_side_by_side,
    format_comparison,
    plot_comparison,
)
from jetfighter_rl.training.train import best_model_of, load_model, train_seeds
from jetfighter_rl.training.transfer import resolve_model_path, transfer_weights

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "training"


EXPERIMENTS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.stem.startswith("curriculum"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_training_configs_are_valid(path):
    cfg = TrainConfig.from_yaml(path)
    assert cfg.name == path.stem
    env = JetEnv(cfg.env_config())
    assert env.task.name == cfg.env["task"]
    kwargs = cfg.algo_kwargs()
    if cfg.algo == "ppo":
        assert (kwargs["n_steps"] * cfg.n_envs) % kwargs["batch_size"] == 0
    assert len(cfg.seeds) >= 3  # bonne pratique : 3 graines minimum


def test_curriculum_stages():
    from jetfighter_rl.training.curriculum import load_curriculum

    cfgs = load_curriculum(CONFIG_DIR / "curriculum.yaml", seeds=[5], n_envs=2)
    names = [c.name for c in cfgs]
    assert names[:2] == ["8_1_level", "8_1_level_6dof"]
    six = cfgs[1]
    assert six.env["model"] == "6dof" and six.init_from == "8_1_level"
    assert six.seeds == [5] and six.n_envs == 2
    # chaque transfert pointe vers une étape antérieure du programme
    for k, c in enumerate(cfgs):
        if c.init_from:
            assert c.init_from in names[:k]


def test_config_override_and_validation(tmp_path):
    cfg = TrainConfig(name="x", env={"task": "level"})
    cfg2 = cfg.override(env={"model": "6dof"}, total_timesteps=10, seeds=None)
    assert cfg2.env == {"task": "level", "model": "6dof"} and cfg2.total_timesteps == 10
    assert cfg2.seeds == cfg.seeds
    with pytest.raises(ValueError):
        TrainConfig(name="x", algo="dqn")
    with pytest.raises(TypeError):
        TrainConfig(name="x", env={"tsk": "level"})
    path = cfg2.save(tmp_path / "c.yaml")
    assert TrainConfig.from_yaml(path) == cfg2


def tiny(name: str, **kw) -> TrainConfig:
    base = {
        "name": name,
        "env": {"task": "level", "episode_time": 2.0},
        "total_timesteps": 128,
        "n_envs": 1,
        "seeds": [0],
        "hyperparams": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        "eval": {"freq": 10_000, "episodes": 1},
    }
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    summary = train_seeds(tiny("smoke"), runs_dir=runs, verbose=0)
    return runs, summary


def test_training_writes_models_logs_and_summary(trained_run):
    runs, summary = trained_run
    seed_dir = Path(summary["seeds"][0]["directory"])
    for name in ("best_model.zip", "final_model.zip", "evaluations.csv"):
        assert (seed_dir / name).exists()
    assert list(seed_dir.glob("tb_*/events.out.tfevents.*"))
    saved = json.loads((Path(summary["directory"]) / "summary.json").read_text())
    assert saved["baseline"]["success_rate"] >= 0 and saved["seeds"][0]["timesteps"] == 128
    assert (Path(summary["directory"]) / "config.yaml").exists()
    assert best_model_of("smoke", runs) == seed_dir / "best_model.zip"
    assert resolve_model_path("smoke", runs) == seed_dir / "best_model.zip"
    assert isinstance(load_model(seed_dir / "best_model.zip"), PPO)


def test_transfer_3dof_to_6dof_copies_everything(trained_run):
    runs, _ = trained_run
    src = load_model(best_model_of("smoke", runs))
    dst = PPO(
        "MlpPolicy",
        JetEnv(EnvConfig(task="level", model="6dof")),
        n_steps=64,
        policy_kwargs={"net_arch": [128, 128]},
    )
    report = transfer_weights(src, dst)
    assert not report.partial and report.skipped == ["log_std"]  # exploration réinitialisée
    obs = JetEnv(EnvConfig(task="level")).reset(seed=0)[0]
    np.testing.assert_allclose(
        src.predict(obs, deterministic=True)[0], dst.predict(obs, deterministic=True)[0]
    )


def test_transfer_to_a_task_with_more_observations(trained_run):
    runs, _ = trained_run
    src = load_model(best_model_of("smoke", runs))
    env = JetEnv(EnvConfig(task="heading_altitude"))
    dst = PPO("MlpPolicy", env, n_steps=64, policy_kwargs={"net_arch": [128, 128]})
    report = transfer_weights(src, dst)
    assert report.partial  # premières couches : entrées communes seulement
    obs = env.reset(seed=0)[0]
    n_common = src.observation_space.shape[0]
    obs[n_common:] = 0.0  # entrées nouvelles à zéro : même action que la source
    np.testing.assert_allclose(
        src.predict(obs[:n_common], deterministic=True)[0],
        dst.predict(obs, deterministic=True)[0],
        atol=1e-6,
    )


def test_curriculum_init_from_previous_run(trained_run):
    runs, _ = trained_run
    cfg = tiny(
        "smoke_6dof",
        env={"task": "level", "model": "6dof", "episode_time": 1.0},
        init_from="smoke",
        total_timesteps=64,
    )
    summary = train_seeds(cfg, runs_dir=runs, verbose=0)
    assert summary["seeds"][0]["timesteps"] == 64


def test_sac_smoke(tmp_path):
    cfg = tiny(
        "sac",
        algo="sac",
        total_timesteps=64,
        hyperparams={"learning_starts": 32, "batch_size": 16, "buffer_size": 1000},
    )
    summary = train_seeds(cfg, runs_dir=tmp_path, verbose=0)
    model = load_model(Path(summary["seeds"][0]["directory"]) / "final_model.zip")
    assert type(model).__name__ == "SAC"


def test_evaluation_tools(trained_run, tmp_path):
    runs, _ = trained_run
    path = best_model_of("smoke", runs)
    results = compare(path, EnvConfig(task="level", episode_time=2.0), episodes=2)
    assert set(results) == {"agent", "référence", "aléatoire"}
    assert "recovery_time" in format_comparison(results)
    acmi, recs = export_side_by_side(
        path, tmp_path / "vol.acmi", EnvConfig(task="level", episode_time=2.0)
    )
    text = acmi.read_text()
    assert "Pilot=Agent RL" in text and "Pilot=Pilote auto" in text
    assert set(recs) == {"Agent RL", "Pilote auto"}
    fig = plot_comparison(recs, title="test")
    assert len(fig.axes) == 6


def test_tacview_waypoints():
    from jetsim.viz.tacview import AcmiWriter, add_waypoints

    w = AcmiWriter()
    add_waypoints(w, [(1000.0, 0.0, 3000.0), (2000.0, 500.0, 3500.0)])
    text = w.text()
    assert "Name=WP1" in text and "Name=WP2" in text and "Navaid+Static+Waypoint" in text


def test_imitation_of_the_reference_policy():
    from jetfighter_rl.training.imitation import behavior_cloning, collect_demonstrations

    cfg = EnvConfig(task="heading_altitude", episode_time=5.0)
    demos = collect_demonstrations(cfg, episodes=3, noise=0.1)
    assert len(demos) == 150 and demos.observations.shape[1] == 26
    env = JetEnv(cfg)
    model = PPO("MlpPolicy", env, n_steps=64, policy_kwargs={"net_arch": [64, 64]})
    losses = behavior_cloning(model, demos, epochs=40, batch_size=64)
    assert losses[-1] < 0.2 * losses[0]
    # le clone reproduit (en moyenne) les actions de la référence sur les états vus
    pred = model.predict(demos.observations, deterministic=True)[0]
    assert np.mean(np.abs(pred - demos.actions)) < 0.1


def test_training_with_imitation_pretraining(tmp_path):
    cfg = tiny("bc", env={"task": "level", "episode_time": 2.0},
               pretrain={"episodes": 2, "epochs": 5})  # fmt: skip
    summary = train_seeds(cfg, runs_dir=tmp_path, verbose=0)
    assert summary["seeds"][0]["timesteps"] == 128


def test_missing_parent_run_fails_early_with_a_hint(tmp_path):
    cfg = tiny("child", init_from="no_such_run")
    with pytest.raises(FileNotFoundError, match="init-from none"):
        train_seeds(cfg, runs_dir=tmp_path, verbose=0)
    assert not (tmp_path / "child").exists()  # rien de créé


def test_linear_learning_rate_schedule():
    cfg = TrainConfig(name="x", env={"task": "level"},
                      hyperparams={"learning_rate": 3e-4, "lr_final": 3e-5})  # fmt: skip
    lr = cfg.algo_kwargs()["learning_rate"]
    assert lr(1.0) == pytest.approx(3e-4) and lr(0.0) == pytest.approx(3e-5)
    assert "lr_final" not in cfg.algo_kwargs()


def test_low_level_curriculum_and_reference():
    from jetfighter_rl.envs.baselines import evaluate, reference_policy
    from jetfighter_rl.training.curriculum import load_curriculum

    cfgs = load_curriculum(CONFIG_DIR / "curriculum_low_level.yaml")
    assert cfgs[0].name == "8_6_level"
    cfg = cfgs[0]
    env_cfg = cfg.env_config()
    assert env_cfg.action_mode == "low_level" and env_cfg.agent_dt == 0.02
    assert env_cfg.discount == pytest.approx(0.998)
    env = JetEnv(replace(env_cfg, episode_time=3.0))
    assert env.observation_space.shape == (19 + 4 + 3,)  # + position des gouvernes
    ev = evaluate(env, reference_policy(env), episodes=1, seed=0)
    assert ev.crash_rate == 0.0
