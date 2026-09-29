# JetFighter_RL

Simulation simplifiée d'un avion de chasse (paramètres inspirés du **F-16**) et apprentissage par
renforcement de manœuvres de vol, jusqu'à l'évitement d'un missile à autodirecteur.

La feuille de route complète est dans [`project_roadmap.md`](project_roadmap.md).

## Installation

Avec [uv](https://docs.astral.sh/uv/) (recommandé) :

```bash
uv venv
uv pip install -e ".[dev]"        # cœur + outils de dev
uv pip install -e ".[dev,viz]"    # + pygame pour le pilotage manuel (phase 5)
uv pip install -e ".[dev,rl]"     # + gymnasium / stable-baselines3 / torch (à partir de la phase 7)
uv run pytest
```

Avec conda (Miniconda / Anaconda) :

```bash
conda create -n jetfighter python=3.11 -y
conda activate jetfighter
pip install -e ".[dev,viz]"       # viz = pygame, pour le pilotage manuel
pytest
```

Ou avec `venv` + `pip` :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Qualité de code : `ruff check . && ruff format .` (ou `pre-commit install` pour l'automatiser).

## Entraîner un agent (phase 8)

Installer les dépendances d'apprentissage : `uv pip install -e ".[dev,viz,rl]"`.

```bash
# une expérience (3 graines par défaut, voir configs/training/*.yaml)
python scripts/train.py configs/training/8_1_level.yaml
python scripts/train.py configs/training/8_2_heading_altitude.yaml --seeds 0

# transfert sur le 6-DOF d'une politique apprise en 3-DOF
python scripts/train.py configs/training/8_1_level.yaml --model 6dof --init-from 8_1_level

# étape 8.6 : commande directe des gouvernes (6-DOF, 50 Hz), imitation puis PPO
python scripts/run_curriculum.py configs/training/curriculum_low_level.yaml --seeds 0

# tout le programme 8.1 -> 8.5, 3-DOF puis 6-DOF (plusieurs heures)
python scripts/run_curriculum.py configs/training/curriculum.yaml --dry-run
python scripts/run_curriculum.py configs/training/curriculum.yaml

# suivre l'apprentissage (chaque terme de récompense, métriques de tâche, référence)
tensorboard --logdir runs

# évaluer : agent vs pilote automatique vs aléatoire, vol Tacview côte à côte, tracés
python scripts/evaluate_agent.py runs/8_1_level/<date_heure>/seed0/best_model.zip
```

Tâches disponibles (`task` dans la configuration) : `level` (8.1), `heading_altitude` (8.2),
`sustained_turn` (8.3), `waypoints` (8.4), `aerobatics` (8.5, `task_kwargs: {maneuvers: [loop]}`
pour une seule figure).

Bon à savoir : l'agent commande en mode hiérarchique (facteur de charge, taux de roulis, manette),
exécuté par les commandes de vol électriques ; une action nulle **maintient la trajectoire**.
Les variantes `*_sac.yaml` (SAC) et `*_imitation.yaml` (imitation du pilote automatique avant le
RL) servent de comparaison.

## Conventions (à respecter dans tout le code)

| Sujet | Convention |
|---|---|
| Unités | **SI partout** en interne (m, s, kg, N, Pa, K). Conversions uniquement aux frontières (fichiers de données, affichage) via `jetfighter.core.constants`. |
| Angles | **Radians** en interne. Degrés uniquement pour l'affichage et les fichiers de config lisibles. |
| Repère inertiel | **NED** (x Nord, y Est, z Bas), Terre plate, origine arbitraire. Altitude `h = -z`. |
| Repère corps | **FRD** : x vers le nez, y vers l'aile droite, z vers le bas (ventre). |
| Repère vent | x aligné sur la vitesse air ; passage corps ↔ vent par l'incidence α et le dérapage β. |
| Attitude | **Quaternion unitaire** `q = [q0, q1, q2, q3]` (scalaire en premier, convention de Hamilton) qui fait passer un vecteur **du repère corps au repère NED** : `v_ned = C_nb(q) @ v_body`. Renormalisé après chaque pas d'intégration. |
| Angles d'Euler | Séquence aéronautique **3-2-1** : cap ψ (lacet), assiette θ (tangage), gîte φ (roulis). Utilisés pour l'affichage et les observations, jamais pour intégrer. |
| Aérodynamique | α = atan2(w, u), β = asin(v / V), avec (u, v, w) la vitesse air en repère corps. |
| Pas de temps | Physique à pas fixe **dt = 0.01 s (100 Hz)**, intégrateur RK4. L'agent décide à 10–20 Hz (*frame skip*). |
| Notations matrices | `C_ab` transforme un vecteur exprimé dans le repère **b** vers le repère **a** : `v_a = C_ab @ v_b`. |

## Structure

```
src/jetfighter/
├── core/        # constantes, atmosphère ISA, repères/rotations, intégrateurs  (phase 1)
├── aircraft/    # paramètres, aéro, propulsion, actionneurs, dynamique 3/6-DOF (phases 2-4)
├── missile/     # missile générique à navigation proportionnelle              (phase 9)
├── control/     # PID, pilote automatique                                     (phase 6)
├── envs/        # environnements Gymnasium et tâches                          (phases 7-10)
├── rl/          # entraînement, transfert, imitation, évaluation              (phase 8)
└── viz/         # tracés, export Tacview                                      (phase 5)
configs/         # paramètres avion / missile / entraînement (YAML)
scripts/         # points d'entrée : vol manuel, entraînement, évaluation
tests/           # tests unitaires et de validation physique
notebooks/       # validations physiques et analyses
```

## Avancement

- [x] Phase 0 — Cadrage et outillage
- [x] Phase 1 — Fondations physiques (repères, atmosphère ISA, intégrateurs)
- [x] Phase 2 — Modèle point-masse 3-DOF (`python scripts/phase2_performance.py` pour le domaine de vol)
- [x] Phase 3 — Modèle 6-DOF avec tables F-16 Stevens & Lewis (`python scripts/phase3_validation.py`)
- [x] Phase 4 — Instruments, capteurs bruités, enveloppe de vol (`python scripts/phase4_dashboard.py`)
- [x] Phase 5 — Enregistrement/rejeu, courbes, export Tacview, pilotage manuel (`python scripts/demo_flight.py`, `python scripts/fly_manual.py`)
- [x] Phase 6 — Commandes de vol électriques, pilote automatique, LQR (`python scripts/phase6_autopilot.py`)
- [x] Phase 7 — Environnement Gymnasium `JetEnv`, tâches, références, essai PPO (`python scripts/phase7_env_check.py --ppo-steps 100000`)
- [x] Phase 8 — Manœuvres par RL : 8.1 → 8.5 en hiérarchique (3-DOF et 6-DOF), 8.6 en commande directe des gouvernes (90–100 % de réussite, 72–100 % du rendement hiérarchique)
