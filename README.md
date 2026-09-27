# JetFighter_RL

Simulation simplifiée d'un avion de chasse (paramètres inspirés du **F-16**) et apprentissage par
renforcement de manœuvres de vol, jusqu'à l'évitement d'un missile à autodirecteur.

La feuille de route complète est dans [`project_roadmap.md`](project_roadmap.md).

## Installation

Avec [uv](https://docs.astral.sh/uv/) (recommandé) :

```bash
uv venv
uv pip install -e ".[dev]"        # cœur + outils de dev
uv pip install -e ".[dev,rl]"     # + gymnasium / stable-baselines3 / torch (à partir de la phase 7)
uv run pytest
```

Avec conda (Miniconda / Anaconda) :

```bash
conda create -n jetfighter python=3.11 -y
conda activate jetfighter
pip install -e ".[dev]"
pytest
```

Ou avec `venv` + `pip` :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Qualité de code : `ruff check . && ruff format .` (ou `pre-commit install` pour l'automatiser).

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
├── envs/        # environnements Gymnasium                                    (phases 7-10)
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
