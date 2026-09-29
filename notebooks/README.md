# Notebooks: a guided tour of jetfighter-rl

The learning half of the project: phases 7 and 8. Each notebook explains the choices made,
gives the key equations, and runs small examples on the real code, with plots.

| # | Notebook | Content |
|---|---|---|
| 0 | [00_project_overview](00_project_overview.ipynb) | Goal, architecture, design decisions, results at a glance |
| 7 | [07_gym_environment](07_gym_environment.ipynb) | Gymnasium environment: actions, observations, rewards, crash penalty, reference policies |
| 8 | [08_rl_maneuvers](08_rl_maneuvers.ipynb) | PPO, live training, curriculum results, reward hacking, imitation learning, transfer, direct surface control |

Notebooks 1–6 (physics, aircraft models, instruments, visualization, classical control) are in
[jetfighter-sim](https://github.com/elouansalaun/jetfighter-sim/tree/main/notebooks).

## Running them

```bash
uv venv && uv pip install -e ".[dev,rl,notebooks]"
uv run jupyter lab notebooks/
```

Notebooks 0 and 7 run in under a minute. Notebook 8 trains a small PPO agent live
(1–2 minutes) and reads the recorded runs and trained agents stored in `results/runs/`.
