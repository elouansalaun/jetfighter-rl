# Notebooks — a guided tour of JetFighter_RL

One notebook per phase of the project. Each one explains the choices made, gives the key
equations, and runs small examples on the real project code, with plots. They are meant to
be read in order, but each stands on its own.

| # | Notebook | Content |
|---|---|---|
| 0 | [00_project_overview](00_project_overview.ipynb) | Goal, method, architecture, conventions, results at a glance |
| 1 | [01_physics_foundations](01_physics_foundations.ipynb) | Reference frames, quaternions vs gimbal lock, ISA atmosphere, RK4 |
| 2 | [02_point_mass_3dof](02_point_mass_3dof.ipynb) | Point-mass F-16: drag polar, thrust, trim, flight envelope, turn performance, a loop |
| 3 | [03_rigid_body_6dof](03_rigid_body_6dof.ipynb) | Rigid-body F-16 with NASA tables: stability, CG, actuators, natural modes, control responses |
| 4 | [04_instruments_envelope](04_instruments_envelope.ipynb) | Instruments, TAS/CAS/EAS, energy-maneuverability diagram, noisy sensors, envelope monitor |
| 5 | [05_visualization](05_visualization.ipynb) | Flight recording, bit-exact replay, plots, Tacview export, manual flight |
| 6 | [06_classical_control](06_classical_control.ipynb) | PID, fly-by-wire, autopilot, LQR, evasion primitives |
| 7 | [07_gym_environment](07_gym_environment.ipynb) | Gymnasium environment: actions, observations, rewards, crash penalty, baselines |
| 8 | [08_rl_maneuvers](08_rl_maneuvers.ipynb) | PPO, live training, curriculum results, reward hacking, imitation learning, direct surface control |

## Running them

```bash
uv venv && uv pip install -e ".[dev,rl]" jupyterlab
uv run jupyter lab notebooks/
```

Every notebook runs in under a minute except notebook 8, which trains a small PPO agent live
(1–2 minutes). Notebook 8 also reads the recorded training runs in `runs/` (not versioned);
without them, those sections are skipped.
