"""Évaluation d'un agent entraîné : comparaison à la référence, vols Tacview, tracés.

Bonne pratique de la roadmap : **toujours regarder les trajectoires**, pas seulement la
courbe de récompense. ``export_side_by_side`` met dans le même fichier Tacview le vol de
l'agent (bleu) et celui du pilote automatique (orange) sur la même graine, avec les
points de passage éventuels ; ``plot_comparison`` trace leurs séries temporelles.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from jetsim.viz.recorder import FlightRecording
from jetsim.viz.style import SERIES, apply_style
from jetsim.viz.tacview import AcmiWriter, add_waypoints, export_recording
from matplotlib.figure import Figure

from jetfighter_rl.envs.baselines import Evaluation, RandomPolicy, evaluate, reference_policy
from jetfighter_rl.envs.jet_env import EnvConfig, JetEnv
from jetfighter_rl.envs.tasks import WaypointsTask
from jetfighter_rl.training.callbacks import EVAL_SEED, AgentPolicy
from jetfighter_rl.training.train import load_model, run_config_of

DEG = 180.0 / math.pi


def env_config_for(model_path: str | Path, **overrides: Any) -> EnvConfig:
    """Configuration d'environnement de l'exécution qui a produit le modèle (surchargeable,
    par exemple ``model="6dof"`` pour tester un agent 3-DOF sur le 6-DOF)."""
    cfg = run_config_of(model_path)
    base = cfg.env_config() if cfg else EnvConfig()
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})


def compare(
    model_path: str | Path,
    env_config: EnvConfig | None = None,
    episodes: int = 20,
    seed: int = EVAL_SEED,
) -> dict[str, Evaluation]:
    """Agent, référence (si mode hiérarchique) et politique aléatoire, mêmes graines."""
    env_config = env_config or env_config_for(model_path)
    env = JetEnv(env_config)
    agent = AgentPolicy(load_model(model_path))
    out = {"agent": evaluate(env, agent, episodes=episodes, seed=seed)}
    if env.has_reference:
        out["référence"] = evaluate(env, reference_policy(env), episodes=episodes, seed=seed)
    out["aléatoire"] = evaluate(env, RandomPolicy(env), episodes=episodes, seed=seed)
    return out


def fly(
    env_config: EnvConfig, policy: Any | None, seed: int
) -> tuple[FlightRecording, dict[str, Any]]:
    """Un épisode enregistré (``policy=None`` : pilote automatique de référence) ;
    renvoie l'enregistrement et l'``info`` de fin."""
    env = JetEnv(replace(env_config, record=True))
    obs, _ = env.reset(seed=seed)
    if policy is None:
        policy = reference_policy(env)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    info: dict[str, Any] = {}
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
    rec = env.recording()
    assert rec is not None
    info["waypoints"] = list(env.task.waypoints) if isinstance(env.task, WaypointsTask) else []
    return rec, info


def export_side_by_side(
    model_path: str | Path,
    path: str | Path,
    env_config: EnvConfig | None = None,
    seed: int = EVAL_SEED,
) -> tuple[Path, dict[str, FlightRecording]]:
    env_config = env_config or env_config_for(model_path)
    agent = AgentPolicy(load_model(model_path))
    flights = {"Agent RL": fly(env_config, agent, seed)}
    if JetEnv(env_config).has_reference:
        flights["Pilote auto"] = fly(env_config, None, seed)
    writer = AcmiWriter(title=f"{env_config.task} — agent vs référence (graine {seed})")
    waypoints = flights["Agent RL"][1]["waypoints"]
    if waypoints:
        add_waypoints(writer, waypoints)
    colors = ("Blue", "Orange")
    for k, (label, (rec, _)) in enumerate(flights.items()):
        export_recording(
            rec, path, writer=writer, object_id=0x101 + k, pilot=label, color=colors[k], write=False
        )
    return writer.write(path), {label: rec for label, (rec, _) in flights.items()}


def plot_comparison(recordings: dict[str, FlightRecording], title: str = "") -> Figure:
    """Altitude, vitesse, inclinaison, facteur de charge, incidence et trace au sol."""
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), constrained_layout=True)
    panels = [
        ("altitude", 1.0, "altitude [m]"),
        ("tas", 1.0, "vitesse vraie [m/s]"),
        ("bank", DEG, "inclinaison μ [°]"),
        ("nz", 1.0, "facteur de charge [g]"),
        ("alpha", DEG, "incidence α [°]"),
    ]
    for ax, (key, scale, label) in zip(axes.flat, panels, strict=False):
        for k, (name, rec) in enumerate(recordings.items()):
            ax.plot(rec.t, np.asarray(rec.instruments[key]) * scale, color=SERIES[k], label=name)
        ax.set_ylabel(label)
        ax.set_xlabel("t [s]")
    ax = axes.flat[-1]
    for k, (name, rec) in enumerate(recordings.items()):
        ax.plot(
            np.asarray(rec.instruments["east"]) / 1000,
            np.asarray(rec.instruments["north"]) / 1000,
            color=SERIES[k],
            label=name,
        )
    ax.set_xlabel("est [km]")
    ax.set_ylabel("nord [km]")
    ax.set_aspect("equal", adjustable="datalim")
    axes.flat[0].legend()
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def format_comparison(results: dict[str, Evaluation]) -> str:
    lines = []
    for label, ev in results.items():
        lines.append(f"  {label:10s} : {ev}")
    metric_names = sorted({k for ev in results.values() for k in ev.metrics})
    if metric_names:
        header = "  " + " " * 12 + "".join(f"{m[:22]:>24s}" for m in metric_names)
        lines.append("\n  Métriques de tâche (moyennes)")
        lines.append(header)
        for label, ev in results.items():
            row = "".join(f"{ev.metrics.get(m, float('nan')):24.2f}" for m in metric_names)
            lines.append(f"  {label:10s}  {row}")
    return "\n".join(lines)
