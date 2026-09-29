# jetfighter-rl  Project roadmap (phases 7 to 11)

> **End goal**: use reinforcement learning (RL) to train a simulated fighter jet (parameters inspired by the F-16) to fly maneuvers, up to evading a homing missile.

> **Note**: the project is split across two repositories. This document covers the learning environment and RL (phases 7 to 11); the simulator (phases 0 to 6) lives in [jetfighter-sim](https://github.com/elouansalaun/jetfighter-sim/blob/main/project_roadmap.md).

---

## Decision log

| Decision | Reason | Consequences |
|---|---|---|
| Reference aircraft: **F-16** (instead of the F-22) | Public aerodynamic data (NASA TP-1538, Stevens & Lewis) | Real tables in the 6-DOF model (phase 3) |
| Default CG position **x_cg = 0.30** (option (a) of phase 3) | Naturally stable aircraft, simpler to start with | x_cg = 0.35 (unstable, like the real F-16) remains available; the phase 6 fly-by-wire stabilizes it |
| **Hierarchical agent first, low-level second** | Much faster learning; validates the whole RL chain (environment, rewards, curriculum) before tackling the hard problem | Phases 7, 8 and 10 in two stages: the agent first commands (n_z, roll rate, throttle) through the fly-by-wire, then the control surfaces directly. Direct control remains the project's **end goal** |
| Crash penalty = fixed **+ maximum cost of the remaining steps** (within the 1/(1−γ) horizon) | Far from the target, crashing cost less than continuing to fly: an incentive to cut episodes short | Crashing is always the worst outcome; each task declares its per-step cost bound (`max_step_cost`) |
| Hierarchical mode: **zero action = hold the flight path** (n_z = cos γ / cos μ) | With "zero action = 1 g", the policy had to output 1/cos μ to within a hundredth to avoid climbing or descending in a turn | Old behavior available (`nz_neutral: one_g`) |
| Fly-by-wire: n_z command rate-limited to 20 g/s + limiter anticipation from q | Abrupt command reversals (typical of an RL agent) went beyond −4 g (down to −5 g) | No more envelope exits on ±1 reversals; step responses unchanged |
| Training: normalized rewards (`VecNormalize`), torch on 1 thread, reduced initial exploration (σ = 0.37) | Critic explained variance ≈ 0 without normalization; throughput ÷ 6 with several threads; random commands at σ = 1 → crash within seconds | Defaults of `jetfighter_rl.training` |
| Imitation of the reference available **as an option** (`pretrain`), pure RL by default | On the Mac, PPO alone learns 8.2 → 8.4 in 3 to 5 M steps; imitation stays useful for aerobatics (the clone of the scripted pilot completes all 4 maneuvers) | `*_imitation.yaml` variants |
| 8.2 and 8.5: **start from imitation** in the main curriculum | 3 pure-RL runs of 8.2: 90 %, 60 %, 20 %; in aerobatics, pure RL only discovers the roll | Pure RL kept as a variant (`8_2_heading_altitude.yaml`, `8_5_aerobatics.yaml`) |
| Low-level (8.6) at **50 Hz**, per-step rewards × agent_dt / 0.1 s, warm start by imitating "autopilot + fly-by-wire" | The fly-by-wire cannot hold the aircraft at 10 Hz; returns comparable across rates | Episodes 5 times longer in agent steps |
| Attitude observation: **gravity direction in wind and body axes** (instead of Euler angles) | When passing through the vertical, μ and φ flip by 180°: the aerobatics agent climbed to the vertical and stopped there | Earlier models incompatible (automatically read in `euler` mode); curriculum to be rerun |
| Project split into **two repositories**: `jetfighter-sim` (package `jetsim`, phases 0–6) and `jetfighter-rl` (package `jetfighter_rl`, phases 7+) | The simulator is reusable on its own; the RL repository depends on it as a library (versioned git dependency) | Aircraft data shipped in the package (`jetsim/data/`); `jetfighter.rl` becomes `jetfighter_rl.training` |

---

## Overview

| Phase | Content | Key deliverable | 
|---|---|---|
| 0 | Scoping, tooling, repository structure | Initialized repository, test CI | 
| 1 | Physical foundations (frames, atmosphere, integrator) | Tested `core/` |
| 2 | 3-DOF point-mass model | "Fast" aircraft for prototyping RL | 
| 3 | 6-DOF rigid-body model with control surfaces | "Realistic" aircraft flown through its controls | 
| 4 | Instruments, sensors, flight envelope | Cockpit display + episode termination conditions | 
| 5 | Visualization and manual flight | Plots, Tacview export, keyboard flying |
| 6 | Classical controllers (baseline) | PID autopilot |
| 7 | Gymnasium environment | `JetEnv` passing `check_env`, hierarchical and low-level modes |
| 8 | RL maneuver curriculum | Hierarchical, then low-level agents: stabilization → turn → aerobatics |
| 9 | Missile model (generic) | Proportional-navigation missile |
| 10 | Missile-evasion RL | Evasion agent + survival map | 
| 11 | Extensions | Robustness, multiple threats, JSBSim, self-play | 


---

## Phase 7 — Gymnasium environment

-  `JetEnv(gymnasium.Env)` class: `reset(seed)`, `step(action)`, `render()`
-  Continuous **action space** `Box(-1, 1)` remapped to physical ranges, with **two modes** (`action_mode` parameter):
  - `"hierarchical"` — **default, to do first**: [n_z, roll rate, throttle] → `HighLevelCommand` → inner loop (`make_inner_loop`). Same action space in 3-DOF and 6-DOF;
  - `"low_level"` — **second stage**: 6-DOF [throttle, δe, δa, δr] (direct control surfaces, no limiters); 3-DOF [throttle, α, roll rate] (this model has no control surfaces);
  - rates: physics 100 Hz, inner loop 50 Hz, agent decisions 10 Hz
-  Normalized **observation space** (≈ [−1, 1]):
  - own state: V/V_ref, h/h_ref, sin/cos of angles (not raw angles → discontinuity at ±180°), bank μ, α, β, p, q, r, Nz, throttle
  -  in low-level mode: control surface positions as well ; for now, the previous action (= surface command); the actual actuator position remains to be added if the low-level agent needs it (step 8.6)
  -  error relative to the target: as relative deviations (Δh, sin/cos Δheading, ΔV), sufficient for these tasks; expressing it **in the aircraft frame** will be needed for 3D targets (pursuit, missile)
-  Reward in `rewards.py`, modular (sum of weighted terms, each term logged separately)
-  `terminated` (crash, success) / `truncated` (time limit) distinction
-  Randomized initial conditions (altitude, speed, heading, attitude)
-  `gymnasium.utils.env_checker.check_env` passes
-  Vectorized environments (`SubprocVecEnv`); measure steps/second (target: > 5,000 steps/s in 3-DOF) ; measured, target not reached per core (see below)
-  Perf option: `numba` on the dynamics if throughput is insufficient ; postponed to phase 8, if training turns out to be too slow

**Done** (`src/jetfighter_rl/envs/`: `jet_env.py`, `tasks.py`, `rewards.py`, `baselines.py`; check: `scripts/phase7_env_check.py`)
- **Identifiers** `gymnasium.make("JetFighter/Level-v0")` and `"JetFighter/HeadingAltitude-v0"`, overridable parameters (`model="6dof"`, `action_mode="low_level"`, `sensors="realistic"`, `record=True`…). `check_env` passes on all 8 task × model × mode combinations.
- **Tasks** (`tasks.py`, one class per task: initial conditions, own observations, reward terms, success criterion, autopilot setpoints):
  - `level` (30 s) ; unusual-attitude recovery (μ ±80°, γ ±25°, roll ±60°/s) → wings-level cruise;
  - `heading_altitude` (90 s) ; reach a randomly drawn heading, altitude (±1,500 m) and speed.
- **Observation**: 19 own-state quantities (read through the **sensors**, noisy if requested) + previous action + task errors, clipped to ±10, `float32`.
- **Reward**: weighted terms in [−1, 0] per step + success bonus, action smoothness penalty, −50 on envelope exit/crash; per-term breakdown in `info["reward_terms"]` and `info["episode_terms"]`.
- **Initial conditions**: trim at the requested flight-path angle, otherwise level trim onto which flight-path angle, bank and roll rate are imposed (steep dives have no trim).
- Optional episode **recording** (exact replay, Tacview export).
- **Baselines** (3-DOF, 6 episodes): autopilot on `level` return ≈ 19–20, 100 % success; `heading_altitude` ≈ 40–55, 100 % success; random policy ≈ −200 / −325 (100 % crash on `heading_altitude`). The autopilot also succeeds in 6-DOF.
- **Throughput** (2-core VM): 3-DOF ≈ 480 agent steps/s for 1 env, ≈ 590 with 2 envs; 6-DOF ≈ 120 / 190. One agent step = 5 control steps + 10 RK4 physics steps: ≈ 4,800 physics steps/s in 3-DOF. The bottleneck is the pure-Python dynamics; ≈ 4–5 k steps/s expected on 8 cores. `numba` postponed.
- **PPO trial** (stable-baselines3, `level`, 3-DOF, hierarchical, 2 envs): 100 % success from ≈ 25 k steps, return 20.5 at 49 k steps (≥ autopilot), ≈ 2 min 30 s. The full chain (env → PPO → TensorBoard → model → Tacview flight) works.

---

## Phase 8 — RL maneuver curriculum

**Algorithms** (via `stable-baselines3`):
- **PPO**: robust, good starting point
- **SAC**: more sample-efficient on continuous actions
- Tracking: TensorBoard (or Weights & Biases), **at least 3 seeds** per experiment, versioned YAML configs

**Progression** (each task reuses the previous policy as initialization when relevant):

| # | Task | Success criterion | Reward (idea) |
|---|---|---|---|
| 8.1 | Stabilization from a perturbed attitude | Back to level flight < 10 s | −‖attitude error‖ − control effort |
| 8.2 | Altitude / heading / speed change | Error < 5 % without excessive overshoot | −normalized error + reach bonus |
| 8.3 | Coordinated turn at maximum sustained rate | Rate close to optimum, β ≈ 0 | + turn rate − β² − energy loss |
| 8.4 | Waypoint following | Chain of 3D waypoints | − distance + bonus per waypoint |
| 8.5 | Aerobatics: loop, roll, Immelmann, Split-S | Trajectory matching a reference | reference trajectory tracking |
| 8.6 | **Switch to low-level**: redo 8.1 → 8.5 in `low_level` mode on the 6-DOF | ≥ 90 % of the hierarchical agent's performance, without exceeding structural limits (no more limiters to protect it) | same rewards + overload and sideslip penalties |

**8.6 — setup** (`action_mode: low_level`, `8_6_*.yaml` configurations, `curriculum_low_level.yaml` curriculum):
- the agent commands throttle, elevator, ailerons and rudder at **50 Hz**: the reference fly-by-wire, tuned for 50 Hz, oscillates and crashes at 10 Hz;
- per-step rewards scaled to the rate (× agent_dt / 0.1 s): returns stay comparable with the hierarchical agent; bonuses and progress unchanged;
- observation: actual control surface positions added; `overload` (beyond +8 / −2 g) and `sideslip` (β² beyond 5°) penalties;
- low-level reference = autopilot (or scripted pilot) **+ fly-by-wire**, whose surface commands are recorded: same performance as hierarchical (8.1: 19.4 vs 19.8; aerobatics 56.6 vs 56.6);
- warm start by **imitation** of this reference (roadmap lead), then critic only, then PPO.
- **local validation (8.1, 500,000 steps at 50 Hz)**: the clone flies at 15.4 / 100 % (reference 16.9 / 100 %). With the hierarchical settings, PPO degraded this clone (−45, 0 % success in 400,000 steps): 3.5° exploration noise on the elevator at 50 Hz and λ = 0.95, which only "sees" 0.4 s. With σ ≈ 0.7°, λ = 0.99 and a smaller step size, success stays at 100 % but the return does not improve yet (10.2): the "≥ 90 % of the hierarchical agent" criterion is **not yet met**, to be judged on the Mac's 5 M steps.
- **8.6 results on the Mac** (1 seed, 5 M steps at 50 Hz, best model):

  | Task | Low-level (imitation + PPO) | Low-level reference | Best 6-DOF hierarchical agent | Low-level / hierarchical ratio |
  |---|---|---|---|---|
  | 8.1 stabilization | 16.7 / 100 % | 16.9 / 100 % | 20.0 / 100 % | 84 % |
  | 8.2 heading / altitude | 89.5 / 100 % | 91.0 / 100 % | 123.7 / 90 % (1st run) | 72 % (but 100 % success vs 90 %) |
  | 8.3 sustained turn | **225.1 / 100 %** | 165.6 / 100 % | 266.5 / 100 % | 84 % |
  | 8.4 waypoints | 38.3 / 90 % | 38.9 / 90 % | 46.0 / 100 % | 83 % |
  | 8.5 aerobatics | **57.4 / 100 %** | 57.4 / 100 % | 55.3 / 100 % (pure RL, before fix) | ≈ 100 % |

  No envelope exits (`overload` term always zero), negligible sideslip. "≥ 90 % of the hierarchical agent" criterion: met in aerobatics, nearly in 8.1 (84 %), not yet elsewhere; only the sustained turn improves clearly beyond imitation (+36 %). On 8.2 and 8.4, PPO **degrades** the imitated policy after 1–2 M steps (8.2: 89.5 → −69.6 at the end of training) without the learning diagnostics (KL, explained variance, σ) flagging anything: the best evaluated model is kept. Leads for later (phase 10): regularization toward the imitated policy during RL, or early stopping on evaluation.

**Chosen order** : tasks 8.1 → 8.5 in **hierarchical mode** (3-DOF then 6-DOF), then 8.6 in **low-level**. Leads to make 8.6 easier:
- initialize the low-level policy by **imitation** (behavior cloning) of the "hierarchical agent + fly-by-wire" ensemble;
- or **residual RL**: the agent learns a correction added to the fly-by-wire output, which is gradually reduced down to direct control.

**Good practices**:
-  Start in 3-DOF, transfer to 6-DOF afterwards (`init_from`, weight copy; `configs/training/curriculum.yaml` curriculum)
-  Penalize abrupt command changes (|Δaction|) for smooth trajectories (`smoothness` term of each task)
-  Watch for *reward hacking*: always look at the trajectories (Tacview), not just the reward curve ; `scripts/evaluate_agent.py` overlays agent and autopilot (Tacview + plots); 4 cases found and fixed (see "Done")
-  Always compare to the PID baseline (reference evaluated on the same seeds at every training run, `eval/baseline_return` curve in TensorBoard)

**Progress**:
-  Training tooling (`src/jetfighter_rl/training/`): versioned YAML configurations, PPO and SAC, 3 seeds, detailed TensorBoard, periodic evaluation against the reference, best model, policy transfer, full curriculum, optional imitation
-  Tasks 8.1 → 8.5 and their reference policies, validated in 3-DOF and 6-DOF
-  8.1 → 8.4 learned by PPO in 3-DOF, **at autopilot level or better** (full curriculum on the Mac, 1 seed, ≈ 2 h 40)
-  6-DOF transfer: 8.1, 8.3 and 8.4 succeed; 8.2 erratic with pure RL → **start from imitation** adopted in the curriculum
-  8.5 aerobatics: **imitation of the scripted pilot, then PPO** (100 %, correct maneuvers). With pure RL, only the roll is discovered (35 %, 3rd run, fixed task)
-  8.6 low-level (direct control surfaces, 6-DOF, 50 Hz): 5 tasks at 90–100 % success (see "8.6")
-  3 seeds per experiment (done with 1 seed so far)

**Done** (`src/jetfighter_rl/envs/tasks/`, `src/jetfighter_rl/training/`, `configs/training/`, `scripts/train.py`, `scripts/evaluate_agent.py`, `scripts/run_curriculum.py`)
- **Tasks** (one class per task: initial conditions, relative observations, reward terms, roadmap success criterion, metrics, reference policy):

  | Task | Implemented success criterion | Reference (3-DOF) |
  |---|---|---|
  | 8.1 `level` | recovered (\|μ\| < 5°, \|γ\| < 2°, held 2 s) in under 10 s | 100 %, recovery ≈ 2.8 s |
  | 8.2 `heading_altitude` | within tolerance at the end (30 m, 3°, 5 m/s), overshoot ≤ max(50 m, 10 %) and ≤ 5° | 100 % |
  | 8.3 `sustained_turn` | mean rate ≥ 90 % of the optimal sustained rate over 20 s, altitude ± 200 m, no energy loss | 100 %, 1.02 × optimum |
  | 8.4 `waypoints` | 4 3D waypoints passed (400 m / 150 m) | 100 % |
  | 8.5 `aerobatics` | complete loop, roll, Immelmann or Split-S, then wings level | 100 % for all 4 maneuvers (scripted pilot) |

  All references also succeed in 6-DOF (sustained turn: threshold corrected by 10 %, the optimum being computed on the point-mass model).
- **Singularity-free geometry** for aerobatics (`wind_axes`): progress tracked on the velocity vector and lift, not on Euler angles (undefined at the vertical).
- **Training**: `python scripts/train.py configs/training/<task>.yaml`; results in `runs/<name>/<date>/seed<k>/` (TensorBoard, best model, CSV history, multi-seed summary).
- **8.1 result** (PPO, 3-DOF, 200,000 steps): 100 % success, return 20.3 vs 17.8 for the autopilot, **recovery in 1.4 s vs 2.8 s** (the agent pulls harder, up to ≈ 8 g).
- **Reward hacking and pitfalls found by looking at the trajectories**:
  1. 8.1: fast recovery, then speed left to decay indefinitely → speed-hold term;
  2. 8.3: **descending spiral** (rate ×2.5 but 3 km of altitude lost) → turn rate is only rewarded if sustained (factor exp(−Δh − ΔE));
  3. 8.5: the agent learned **not to perform the maneuver** (flying straight cost nothing) → dominant progress reward (40 points + 20 on exit);
  4. all: incentive to crash when far from the target → crash penalty covering the remaining cost.
- **Fly-by-wire limits revealed by the agent**: the 3-DOF agent transferred to the 6-DOF alternated +8 / −3 g and went beyond −4 g → rate limit on the command (see the decision log).
- **Results on the Mac** (full curriculum, 1 seed, 8 environments, Euler-angle observations) — best model in evaluation (10–20 episodes):

  | Task | Model | PPO agent: return / success | Autopilot: return / success |
  |---|---|---|---|
  | 8.1 stabilization | 3-DOF | **20.8 / 100 %** | 17.8 / 100 % |
  | 8.1 stabilization | 6-DOF (transfer, 200 k steps) | −3.9 / 20 % — 5 Hz roll oscillation | 17.3 / 100 % |
  | 8.2 heading/altitude/speed | 3-DOF (3 M steps) | **126.8 / 90 %** | 89.0 / 100 % |
  | 8.2 heading/altitude/speed | 6-DOF (transfer, 1 M steps) | **123.7 / 90 %** | 91.3 / 100 % |
  | 8.3 sustained turn | 3-DOF (3 M steps) | **289.7 / 100 %** | 274.2 / 100 % |
  | 8.3 sustained turn | 6-DOF (transfer, 1 M steps) | **265.7 / 100 %** | 165.6 / 100 % |
  | 8.4 waypoints | 3-DOF (5 M steps) | **45.1 / 100 %**, course in 111 s | 39.2 / 90 %, 143 s |
  | 8.4 waypoints | 6-DOF (transfer, 1.5 M steps) | 32.1 / 90 %, 10 % crashes | 39.1 / 90 % |
  | 8.5 aerobatics | 3-DOF (5 M steps) | 25.0 / 35 % (roll only) | 57.5 / 100 % |

  → **Milestone J3 reached** (8.2). On 8.2 the agent gets a better return than the reference but fails the strict criterion (overshoot) more often; its success rate swings between 30 and 90 % from one evaluation to the next.
- **2nd run on the Mac** (continuous attitude observations, 1 seed):

  | Task | PPO agent: return / success | Autopilot |
  |---|---|---|
  | 8.1, 3-DOF / 6-DOF | 18.5 / 100 % — **20.0 / 100 %** (the 6-DOF transfer goes from 20 % to 100 %) | 17.8 / 17.3 |
  | 8.2, 3-DOF / 6-DOF | 74.4 / 60 % — 60.3 / 50 % (best models; at the end of training, 0 %) | 89.0 / 91.3 |
  | 8.3, 3-DOF / 6-DOF | **290.9 / 100 %** — **266.5 / 100 %** | 274.2 / 165.6 |
  | 8.4, 3-DOF / 6-DOF | **48.6 / 100 %** — **46.0 / 100 %** | 39.2 / 39.1 (90 %) |
  | 8.5, 3-DOF / 6-DOF | 56.6 / 100 % — 55.3 / 100 %, but *reward hacking* | 57.9 / 56.9 |
  | 8.5 by imitation, 3-DOF | **58.2 / 100 %** (correct maneuvers) | 57.9 |

  8.2: learning oscillates (success 0 → 60 → 0 % from one evaluation to the next) → linear learning-rate decay (`lr_final`) on 8.2 → 8.5. The first run, identical except for the attitude representation, had reached 90 %: 3 seeds will be needed to conclude.
- **8.5, reward hacking #5**: with pure RL, a "loop" in 6 s instead of 25 s. Looking at the trajectory: the agent turned 90° and then made small motions around the lateral axis; the angle of the velocity **projected** onto the loop plane, ill-defined when the velocity is perpendicular to the plane, then "rotated" by 360° without any loop. Fixed: pitch no longer counts when the velocity leaves the plane by more than 30°, and the exit must be on the right heading (± 30°). Re-evaluated with the fixed task, this agent only completes the roll; the scripted pilot stays at 100 %.
- **8.5, diagnosis (1st run)**: the agent flew well up to the vertical, then stopped there (loop and Immelmann at 26 % progress, Split-S at 50 %). When passing through the vertical, the Euler angles in the observation flip by 180°: the agent "thought it was inverted". Fixes: attitude observed as gravity in wind and body axes (continuous), progress rewarded by records (a failed attempt no longer cancels the gain), reduced tracking penalties (accumulated over 45 s, they discouraged trying). Variant `8_5_aerobatics_imitation.yaml`: the clone of the scripted pilot completes all 4 maneuvers at 100 % before any RL.
- **6-DOF transfer of 8.1**: the 3-DOF agent alternates roll commands at every step (harmless in 3-DOF, where the response is instantaneous); in 6-DOF this becomes an oscillation. Budget raised to 1 M steps; to watch for 8.6 (stronger smoothness penalty if needed).
- **8.2, observation during local validation**: PPO (from scratch or from the 8.1 policy, with or without normalization, 2 or 8 environments, reduced noise, "expo" curve, compensated n_z) and SAC did not get beyond "fly straight" in 250,000 to 600,000 steps. The signal exists (imitation of the autopilot scores 97 vs 99); the hypothesis is simply a lack of samples (configuration budgets: 3 to 5 million steps, i.e. 15 to 30 min per seed on the Mac). If not: variant `8_2_heading_altitude_imitation.yaml` (imitation, then critic only, then cautious PPO).

---

## Phase 9 — Possible extensions

- **Missile evasion**: learn ability to evade an anti-aircraft missile
- **Robustness**: domain randomization (mass, aero coefficients ±10 %, sensor noise, wind)
- **Multiple threats**: two missiles, staggered launches
- **Higher fidelity**: plug in [JSBSim](https://github.com/JSBSim-Team/jsbsim) (F-16 model included) and compare with the in-house model
- **1v1 air combat** in *self-play*
- Simplified **countermeasures** (decoys modeled as a seeker disturbance)
- **Explainability**: policy analysis (which observations drive the decisions)

---

## Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Sign / convention bug in the dynamics | Phase 1 unit tests, Phase 5 manual flight |
| Numerical instability (NaN) | dt = 0.01 s, RK4, quaternion normalization, NaN detection |
| Model too hard for RL | Hierarchical commands, 3-DOF first, curriculum |
| Reward hacking | Visualize trajectories, reward terms logged separately |
| Training too slow | Vectorization, numba, 3-DOF for large campaigns |
| Non-reproducible results | Fixed seeds, versioned YAML configs, ≥ 3 seeds per experiment |

---

## Milestones (demo checkpoints)

1. **J1** : The 3-DOF aircraft flies level and turns correctly (end of Phase 2)
2. **J2** : The 6-DOF aircraft can be flown with the keyboard and is visible in Tacview (end of Phase 5)
3. **J3** : A PPO agent stabilizes the aircraft and reaches a heading / altitude (Phase 8.2) — ✅ reached on 2026-09-28 (3-DOF and 6-DOF)
4. **J4** : A (hierarchical) agent performs a loop and a max-rate turn (Phase 8.5) — ✅ 2026-09-28 (loop: imitation then PPO; turn: PPO alone, 106 % of the reference)
5. **J4b** : A **low-level** agent (direct control surfaces, 6-DOF) matches the hierarchical agent on tasks 8.1 → 8.5 (Phase 8.6) — 🟡 partial: 90–100 % success everywhere, 72–100 % of the hierarchical return


---

## Useful references

- Stevens, Lewis & Johnson — *Aircraft Control and Simulation* (complete F-16 model, 6-DOF equations)
- Nguyen et al. (1979) — NASA TP-1538, F-16 wind-tunnel aerodynamic data
- Zarchan — *Tactical and Strategic Missile Guidance* (proportional navigation)
- [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) documentation
- [JSBSim](https://github.com/JSBSim-Team/jsbsim) — open-source flight dynamics engine
- [Tacview](https://www.tacview.net/) ACMI format for visualization
- DARPA AlphaDogfight Trials (2020) — background on RL in simulated air combat
