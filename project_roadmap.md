# JetFighter_RL — Roadmap du projet

> **Objectif final** : entraîner par apprentissage par renforcement (RL) un avion de chasse simulé (paramètres inspirés du F-16) à exécuter des manœuvres de vol, jusqu'à l'évitement d'un missile à autodirecteur.
>
> **Principe directeur** : on avance par couches. Chaque couche est **validée physiquement** avant d'être utilisée par la suivante. Un agent RL entraîné sur une physique fausse apprend à exploiter les bugs, pas à piloter.

---

## Vue d'ensemble

| Phase | Contenu | Livrable clé | Durée indicative* |
|---|---|---|---|
| 0 | Cadrage, outillage, structure du dépôt | Dépôt initialisé, CI de tests | 2–3 jours |
| 1 | Fondations physiques (repères, atmosphère, intégrateur) | `core/` testé | 1 semaine |
| 2 | Modèle point-masse 3-DOF | Avion « rapide » pour prototyper le RL | 1 semaine |
| 3 | Modèle corps rigide 6-DOF avec gouvernes | Avion « réaliste » piloté par ses commandes | 3–4 semaines |
| 4 | Instruments, capteurs, enveloppe de vol | Tableau de bord + conditions de fin d'épisode | 1 semaine |
| 5 | Visualisation et pilotage manuel | Tracés, export Tacview, pilotage clavier | 1 semaine |
| 6 | Contrôleurs classiques (baseline) | Pilote automatique PID | 1–2 semaines |
| 7 | Environnement Gymnasium | `JetEnv` conforme `check_env` | 1 semaine |
| 8 | Curriculum de manœuvres RL | Agents : stabilisation → virage → voltige | 3–5 semaines |
| 9 | Modèle de missile (générique) | Missile à navigation proportionnelle | 1–2 semaines |
| 10 | RL d'évitement de missile | Agent d'évasion + carte de survie | 3–5 semaines |
| 11 | Extensions | Robustesse, multi-menaces, JSBSim, self-play | ouvert |

\* Estimations pour un développeur seul, en parallèle d'autres activités.

---

## Phase 0 — Cadrage et outillage

**But** : un dépôt propre dès le départ, pour pouvoir itérer vite sans casser l'existant.

- [x] Environnement Python 3.10+ (`uv` ou `venv`) et `pyproject.toml`
- [x] Dépendances de base : `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`
- [x] Dépendances RL (plus tard) : `gymnasium`, `stable-baselines3`, `torch`, `tensorboard`
- [x] Qualité : `ruff` (lint + format), `mypy` optionnel, `pre-commit`
- [x] Git + `.gitignore` (exclure `runs/`, `models/`, `*.acmi`)
- [x] Conventions écrites dans le README : **unités SI partout**, radians en interne, degrés uniquement en affichage

**Structure de dépôt proposée**

```
JetFighter_RL/
├── pyproject.toml
├── README.md
├── project_roadmap.md
├── configs/
│   ├── aircraft/f16.yaml             # masse, géométrie, poussée, coefficients aéro
│   ├── missile/generic_pn.yaml
│   └── training/*.yaml               # hyperparamètres par tâche
├── src/jetfighter/
│   ├── core/
│   │   ├── constants.py              # g0, R, gamma air...
│   │   ├── atmosphere.py             # ISA : rho, T, P, a(h)
│   │   ├── frames.py                 # rotations, quaternions, Euler, NED <-> corps
│   │   └── integrators.py            # Euler, RK4
│   ├── aircraft/
│   │   ├── params.py                 # dataclass chargée depuis YAML
│   │   ├── aero.py                   # coefficients CL, CD, CY, Cl, Cm, Cn
│   │   ├── propulsion.py             # poussée(h, Mach, manette), post-combustion, lag moteur
│   │   ├── actuators.py              # limites de débattement / vitesse, retard 1er ordre
│   │   ├── dynamics_3dof.py          # modèle point-masse
│   │   ├── dynamics_6dof.py          # modèle corps rigide
│   │   └── sensors.py                # instruments + bruit
│   ├── missile/
│   │   ├── missile.py
│   │   ├── guidance.py               # navigation proportionnelle
│   │   └── seeker.py                 # champ de vue, limites de cardan
│   ├── control/
│   │   ├── pid.py
│   │   └── autopilot.py              # maintien altitude / cap / vitesse
│   ├── envs/
│   │   ├── base_env.py
│   │   ├── maneuver_envs.py
│   │   ├── evasion_env.py
│   │   └── rewards.py
│   └── viz/
│       ├── plots.py
│       └── tacview.py                # export .acmi
├── scripts/
│   ├── fly_manual.py
│   ├── train.py
│   └── evaluate.py
├── tests/
└── notebooks/                        # validation physique, analyses
```

---

## Phase 1 — Fondations physiques

**But** : les briques mathématiques dont tout le reste dépend. C'est ici que se cachent les bugs les plus sournois (signes, conventions d'angles).

### 1.1 Repères et conventions
- [x] Repère inertiel **NED** (North-East-Down), approximation Terre plate (suffisant à cette échelle)
- [x] Repère **corps** : x vers le nez, y aile droite, z vers le bas
- [x] Repère **aérodynamique / vent** (via α incidence et β dérapage)
- [x] Attitude en **quaternion** en interne (évite le blocage de cardan en voltige), conversion vers Euler (φ roulis, θ tangage, ψ cap) pour l'affichage et les observations
- [x] Normalisation du quaternion à chaque pas

### 1.2 Atmosphère standard (ISA)
- [x] Température, pression, masse volumique ρ(h), vitesse du son a(h) de 0 à 20 km (troposphère + début stratosphère)
- [x] Calcul du Mach et de la pression dynamique q̄ = ½ρV²

### 1.3 Intégrateur
- [x] RK4 à pas fixe, **dt physique = 0.01 s (100 Hz)**
- [x] Séparer la fréquence physique de la fréquence de décision de l'agent (ex. agent à 10–20 Hz → *frame skip* de 5 à 10)

### Tests de validation
- [x] Rotations : aller-retour Euler ↔ quaternion ↔ matrice (tolérance 1e-9)
- [x] ISA : comparer à des valeurs tabulées (ρ₀ = 1.225 kg/m³, a₀ = 340.3 m/s, ρ(11 km) ≈ 0.364 kg/m³)
- [x] Intégrateur : chute libre et oscillateur harmonique contre la solution analytique

---

## Phase 2 — Modèle point-masse 3-DOF (prototype rapide)

**But** : un modèle simple et très rapide pour prototyper les environnements et le RL pendant que le 6-DOF se construit. Il servira aussi pour les grandes campagnes d'évitement de missile.

**État** : position (x, y, h), vitesse V, pente γ, cap χ, masse m
**Commandes** : poussée T (ou manette), incidence α (ou facteur de charge n), angle de gîte μ

**Équations** :

```
V̇  = (T·cos α − D) / m − g·sin γ
γ̇  = [(L + T·sin α)·cos μ − m·g·cos γ] / (m·V)
χ̇  = (L + T·sin α)·sin μ / (m·V·cos γ)
ẋ  = V·cos γ·cos χ     ẏ = V·cos γ·sin χ     ḣ = V·sin γ
L  = q̄·S·CL(α, M)      D = q̄·S·(CD0(M) + k·CL²)
```

- [x] Implémentation + limites : α_max (décrochage), n ∈ [−3 g, +9 g], taux de variation de μ limité
- [x] Traînée : polaire parabolique CD = CD0(M) + k·CL², avec hausse de CD0 en transsonique
- [x] Tests : vol en palier stabilisé, virage à facteur de charge constant (taux de virage ω = g·√(n²−1)/V), montée, plafond

**Réalisé** (`src/jetfighter/aircraft/`, validation : `scripts/phase2_performance.py`)
- Formulation **sans singularité** : les équations ci-dessus deviennent singulières à γ = ±90°. Le modèle intègre donc un quaternion du repère vent (angles d'Euler = χ, γ, μ), ce qui permet les loopings.
- Commandes retenues : `[manette, α commandée, taux de roulis autour de la vitesse]`, avec réponse du 1er ordre (α : 0,25 s ; roulis : 0,2 s ; moteur : 1 s), **limiteur d'incidence et de facteur de charge** façon commandes de vol électriques.
- Propulsion : poussée ∝ (ρ/ρ0)^1,15 (calée sur les poussées statiques Stevens & Lewis), effet d'admission linéaire en Mach, borné à 1,3 × la poussée statique au sol.
- Modules : `params.py` (YAML → dataclasses), `aero_polar.py`, `propulsion.py`, `dynamics_3dof.py` (dynamique, trim, instruments), `performance.py` (domaine de vol, virage, plafond).
- Performances obtenues vs F-16 publié : Mach max 1,18 au sol (≈ 1,2) et 2,08 à 11 km (≈ 2,0) ; montée 275 m/s (≈ 250) ; virage soutenu 18,3 °/s à 3 km M0,8 (≈ 18–20) ; ≈ 10 600 pas physiques/s.
- Limites connues : pas de décrochage, donc vitesse mini optimiste (≈ 45 m/s au sol contre ≈ 60–65 m/s réels) ; plafond aérodynamique 17,7 km contre 15,2 km publiés (limite opérationnelle) ; masse constante.

---

## Phase 3 — Modèle corps rigide 6-DOF

**But** : le vrai « avion », piloté par ses **commandes de vol** et non par des consignes abstraites.

### 3.1 Paramètres « F-16-like »
Contrairement au F-22, le F-16 dispose de **données aérodynamiques publiées** (soufflerie NASA, reprises par Stevens & Lewis) : c'est ce qui rend ce choix pertinent. Approche retenue :
- **Masse, géométrie, poussée** (valeurs publiques approximatives, à sourcer dans le YAML)
  - Masse de référence du modèle Stevens & Lewis : 20 500 lb ≈ 9 300 kg (F-16C : ≈ 8 600 kg à vide, ≈ 19 200 kg max au décollage)
  - Envergure 9,14 m (30 ft), longueur ≈ 15,0 m, surface alaire 27,87 m² (300 ft²), corde moyenne 3,45 m (11,32 ft)
  - 1 moteur (F100-PW-229 ou F110-GE-129) : ≈ 76–79 kN à sec, ≈ 129–131 kN avec PC ; le modèle Stevens & Lewis utilise les tables de poussée d'un F100 plus ancien
  - Pas de vectorisation de poussée
  - Facteur de charge +9 g / −3 g ; Mach max ≈ 2,0 ; plafond ≈ 15 000 m
- **Coefficients aéro** : tables F-16 publiées (Stevens & Lewis ; Nguyen et al., NASA TP-1538), utilisées directement, sans mise à l'échelle
- **Inerties** (Stevens & Lewis) : Ixx = 9 496, Iyy = 55 814, Izz = 63 100, Ixz = 982 slug·ft² → ≈ 12 875 / 75 674 / 85 552 / 1 331 kg·m²
- [x] Tout dans `configs/aircraft/f16.yaml` → pouvoir changer d'avion sans toucher au code

### 3.2 État (13 variables + moteur)
- Position NED (3), vitesse corps u, v, w (3), quaternion (4), vitesses angulaires p, q, r (3)
- + état moteur (régime / poussée effective), + masse (consommation, optionnel)

### 3.3 Commandes de pilotage (entrées)
| Commande | Effet principal | Plage indicative |
|---|---|---|
| Manette des gaz δ_T (+ PC) | Poussée | 0 → 1 (PC au-delà d'un seuil) |
| Stabilisateurs monoblocs δ_e | Tangage (Cm) | ±25° |
| Ailerons / flaperons δ_a | Roulis (Cl) | ±20° |
| Gouvernes de direction δ_r | Lacet (Cn) | ±30° |

### 3.4 Forces et moments
- [x] Aéro : CL(α, δe, M), CD(α, M), CY(β, δr), Cl(β, δa, δr, p, r), Cm(α, δe, q), Cn(β, δa, δr, p, r)
  - Tables d'interpolation en α et Mach (`scipy.interpolate.RegularGridInterpolator`)
  - Dérivées d'amortissement (Cmq, Clp, Cnr) : indispensables, sinon l'avion oscille sans fin
- [x] Propulsion : T(h, M, δ_T) avec baisse en altitude (∝ ρ/ρ₀ environ), dynamique moteur du 1er ordre (constante de temps ~1 s, plus lente pour allumer la PC)
- [x] Gravité en repère NED projetée en corps
- [x] Équations de Newton-Euler (forces en corps + terme ω × V ; moments avec tenseur d'inertie incluant Ixz)

### 3.5 Actionneurs
- [x] Saturation en position **et** en vitesse (ex. 60°/s pour les gouvernes)
- [x] Retard du 1er ordre (τ ≈ 0,05 s)
- → Évite que l'agent RL apprenne des commandes « bang-bang » physiquement impossibles

### Tests de validation (dans `notebooks/`)
- [x] **Trim** : trouver (α, δe, δ_T) pour le vol en palier à plusieurs vitesses / altitudes (optimisation `scipy.optimize`)
- [x] **Modes propres** : linéariser autour du trim, vérifier la présence d'une oscillation d'incidence rapide (*short period*, ~1–3 s) et d'un phugoïde lent (~30–100 s)
- [x] **Réponses indicielles** : échelon de profondeur, d'ailerons, de direction — signes et ordres de grandeur cohérents
- [x] **Conservation** sans aéro ni poussée ni gravité : énergie de rotation et moment cinétique (moteur inclus) conservés
- [ ] **Cohérence 3-DOF vs 6-DOF** sur un virage stabilisé (fait en palier seulement ; le virage demande un pilote automatique, phase 6)

> ⚠️ Point d'attention : un F-16 réel est **instable par conception** et piloté via des commandes de vol électriques. Pour le RL, deux options : (a) rendre le modèle légèrement stable statiquement (plus facile), (b) garder l'instabilité et ajouter une loi de commande de stabilisation (Phase 6). Commencer par (a).

**Réalisé** (`dynamics_6dof.py`, `f16_aero.py`, `analysis.py`, tables dans `configs/aircraft/f16_sl_tables.yaml`, validation : `scripts/phase3_validation.py`)
- Tables aéro et moteur Stevens & Lewis complètes (données NASA TP-1538), réimplémentées (le code de référence AeroBench est sous GPL-3 : seules les valeurs numériques ont été reprises).
- Vérifications : dérivées identiques à l'implémentation de référence à 2·10⁻⁴ près sur 300 états aléatoires (écart = arrondi des constantes d'inertie du livre) ; cas de vérification publié retrouvé sur 11 dérivées sur 13 (ṗ et ṙ diffèrent de ≈ 0.001 sur Cl dans toutes les implémentations disponibles) ; trim publié retrouvé (manette 0.1385, δe −0.759°).
- **Centrage** : au nominal x_cg = 0.35, l'avion est instable en tangage (temps de doublement ≈ 7 s), comme le vrai F-16. Option (a) retenue : x_cg = 0.30 par défaut → avion stable dans tout le domaine subsonique (sauf divergence lente de second régime à α > 11°). Le 0.35 reste disponible pour la phase 6.
- Modes à 3000 m / 200 m/s : oscillation d'incidence ω = 2.1 rad/s ζ = 0.56 ; phugoïde 107 s ; roulis hollandais 1.8 s ζ = 0.12 ; roulis τ = 0.28 s ; spirale stable.
- Limites : tables sans effet du Mach (fiables jusqu'à ≈ Mach 0.6–0.8) ; ≈ 1 700–2 000 pas physiques/s en Python pur (à accélérer en phase 7) ; cohérence 3-DOF/6-DOF vérifiée sur l'incidence d'équilibre en palier (< 1°), pas encore en virage.

---

## Phase 4 — Instruments, capteurs et enveloppe de vol

### 4.1 Instruments (sorties observables)
- [x] Vitesses : TAS, CAS (approx.), Mach, vitesse verticale
- [x] Altitude, cap, attitude (φ, θ, ψ)
- [x] Incidence α, dérapage β
- [x] Vitesses angulaires p, q, r
- [x] Accélérations en corps, **facteur de charge Nz**
- [x] Énergie spécifique Es = h + V²/2g et excès de puissance Ps (très utile pour l'évitement de missile)
- [x] Option : bruit et biais capteurs (pour la robustesse, Phase 11)

### 4.2 Enveloppe de vol et fins d'épisode
- [x] Collision sol (h < 0 ou h < h_min de sécurité)
- [x] Dépassement de facteur de charge (limite structurelle)
- [x] Décrochage prolongé / vrille (α > α_max pendant > N s)
- [x] Sortie de domaine (altitude > plafond, vitesse < V_min)
- [x] Détection NaN / divergence numérique → fin d'épisode + log

**Réalisé** (`instruments.py`, `sensors.py`, `envelope.py`, limites dans `configs/aircraft/f16.yaml` section `envelope`, démo : `scripts/phase4_dashboard.py`)
- **Tableau de bord commun** aux modèles 3-DOF et 6-DOF (`read_instruments`) : 27 grandeurs (position, TAS/CAS/EAS, Mach, variomètre, γ/χ, attitude φ/θ/ψ, α/β, p/q/r, facteurs de charge nx/ny/nz, énergie E et Ps, poussée). Pour le 3-DOF, attitude et vitesses angulaires corps reconstruites à partir du repère vent et de α.
- **CAS** exacte (pression d'impact isentropique en subsonique, formule de Rayleigh en supersonique) et son inverse.
- **Capteurs** : bruit blanc + biais tiré à chaque épisode, par voie, reproductible (générateur fourni) ; parfaits par défaut ; exemple réaliste dans `configs/sensors/realistic.yaml`.
- **Fins d'épisode** : NaN, sol, surcharge (−4 / +10 g), plafond (17 km), vitesse mini (50 m/s), Mach max (2.0 en 3-DOF, 0.95 en 6-DOF), dérapage (30°), décrochage prolongé (α > 30° pendant > 2 s).

---

## Phase 5 — Visualisation et pilotage manuel

- [ ] Tracés matplotlib : trajectoire 3D, séries temporelles (V, h, α, Nz, commandes)
- [ ] **Export Tacview (.acmi)** : format texte simple, gratuit en lecture, rendu 3D très parlant pour analyser les manœuvres et les engagements missile
- [ ] `scripts/fly_manual.py` : pilotage clavier ou joystick (`pygame`) — le meilleur test de sanité : si un humain ne peut pas faire voler le modèle, un agent RL non plus
- [ ] Rejeu d'épisodes enregistrés (états + actions en `.npz`)

---

## Phase 6 — Contrôleurs classiques (baseline)

**But** : disposer d'une référence pour juger l'agent RL, et d'une couche basse optionnelle pour un RL hiérarchique.

- [ ] Amortisseurs de tangage / roulis / lacet (*stability augmentation*)
- [ ] PID : maintien de vitesse, d'altitude, de cap, virage coordonné (β → 0)
- [ ] Option : LQR sur le modèle linéarisé autour du trim
- [ ] Heuristiques d'évitement de missile scriptées (pour la Phase 10) : virage de mise en travers (*beam*), fuite (*drag*), virage serré de dernière seconde

**Décision d'architecture à prendre à cette étape** :

| Option | L'agent commande… | + | − |
|---|---|---|---|
| Bas niveau | directement les gouvernes et la manette | fidèle à l'objectif initial | apprentissage long et instable |
| Hiérarchique | des consignes (Nz, taux de roulis, poussée) suivies par un contrôleur | apprentissage beaucoup plus rapide | dépend de la qualité du contrôleur |

→ Recommandation : **commencer hiérarchique**, puis descendre au bas niveau une fois le pipeline validé.

---

## Phase 7 — Environnement Gymnasium

- [ ] Classe `JetEnv(gymnasium.Env)` : `reset(seed)`, `step(action)`, `render()`
- [ ] **Espace d'action** continu `Box(-1, 1)` remappé vers les plages physiques
- [ ] **Espace d'observation** normalisé (≈ [−1, 1]) :
  - états propres : V/V_ref, h/h_ref, sin/cos des angles (pas les angles bruts → discontinuité à ±180°), α, β, p, q, r, Nz, manette
  - erreur par rapport à la cible, exprimée **dans le repère avion** (invariance par rotation)
- [ ] Récompense dans `rewards.py`, modulaire (somme de termes pondérés, chaque terme loggé séparément)
- [ ] Distinction `terminated` (crash, succès) / `truncated` (limite de temps)
- [ ] Randomisation des conditions initiales (altitude, vitesse, cap, attitude)
- [ ] `gymnasium.utils.env_checker.check_env` passe
- [ ] Environnements vectorisés (`SubprocVecEnv`) ; mesurer les pas/seconde (objectif : > 5 000 pas/s en 3-DOF)
- [ ] Option perf : `numba` sur la dynamique si le débit est insuffisant

---

## Phase 8 — Curriculum de manœuvres par RL

**Algorithmes** (via `stable-baselines3`) :
- **PPO** : robuste, bon point de départ
- **SAC** : plus efficace en échantillons sur les actions continues
- Suivi : TensorBoard (ou Weights & Biases), **3 graines minimum** par expérience, configs YAML versionnées

**Progression** (chaque tâche réutilise la politique précédente comme initialisation si pertinent) :

| # | Tâche | Critère de réussite | Récompense (idée) |
|---|---|---|---|
| 8.1 | Stabilisation depuis une attitude perturbée | Retour en palier < 10 s | −‖écart attitude‖ − effort commande |
| 8.2 | Changement d'altitude / de cap / de vitesse | Erreur < 5 % sans dépassement excessif | −erreur normalisée + bonus d'atteinte |
| 8.3 | Virage coordonné à taux maximal soutenu | Taux proche de l'optimum, β ≈ 0 | + taux de virage − β² − perte d'énergie |
| 8.4 | Suivi de points de passage | Enchaînement de waypoints 3D | − distance + bonus par waypoint |
| 8.5 | Voltige : looping, tonneau, Immelmann, Split-S | Trajectoire conforme à une référence | suivi de trajectoire de référence |

**Bonnes pratiques** :
- [ ] Commencer en 3-DOF, transférer en 6-DOF ensuite
- [ ] Pénaliser les variations brusques de commande (|Δaction|) pour des trajectoires lisses
- [ ] Surveiller le *reward hacking* : toujours regarder les trajectoires (Tacview), pas seulement la courbe de récompense
- [ ] Comparer systématiquement à la baseline PID

---

## Phase 9 — Modèle de missile (générique et simplifié)

**But** : une menace plausible pour l'entraînement, construite uniquement à partir de modèles de manuel. Les paramètres sont **génériques**, pas ceux d'un missile réel.

- [ ] Dynamique point-masse 3-DOF
- [ ] Profil de propulsion : phase propulsée courte puis vol balistique (le missile **perd de l'énergie** avec la traînée → c'est ce que l'avion va exploiter)
- [ ] Limite de facteur de charge (dépendante de q̄ : un missile lent manœuvre moins)
- [ ] Guidage par **navigation proportionnelle** (manuel classique) : a_cmd = N · V_c · λ̇, N ≈ 3–5
- [ ] Autodirecteur : champ de vue et limite de cardan → perte d'accrochage si dépassés
- [ ] Fusée de proximité : « impact » si distance < rayon létal, sinon distance de passage (*miss distance*) enregistrée
- [ ] Autodestruction / fin si vitesse < seuil ou temps de vol max
- [ ] Tests : interception d'une cible non manœuvrante, d'une cible en virage constant ; vérifier la sensibilité à N

---

## Phase 10 — RL d'évitement de missile

### 10.1 Environnement `EvasionEnv`
- [ ] Conditions initiales randomisées : distance de tir, angle d'aspect, altitudes relatives, vitesses
- [ ] **Observation** (deux niveaux de difficulté) :
  - *Information complète* : position / vitesse relatives du missile en repère avion, distance, vitesse de rapprochement, angles de ligne de visée
  - *Information partielle* (plus réaliste) : uniquement azimut / élévation d'alerte + temps écoulé depuis le tir, éventuellement bruités
- [ ] Fin d'épisode : impact (échec), missile à court d'énergie ou perte d'accrochage (succès), crash (échec)

### 10.2 Récompense
- [ ] Gros bonus de survie / grosse pénalité d'impact
- [ ] Façonnage (*shaping*) : + distance de passage, + baisse de vitesse du missile, − perte d'altitude excessive, − sortie d'enveloppe
- [ ] Veiller à ce que le shaping ne domine pas l'objectif principal

### 10.3 Curriculum
1. Missile lent et peu manœuvrant, tir à longue distance
2. Augmentation progressive : vitesse, facteur de charge, N
3. Distances de tir plus courtes, aspects défavorables
4. Passage de l'information complète à l'information partielle
5. Passage du 3-DOF au 6-DOF

### 10.4 Évaluation
- [ ] Taux de survie sur un banc de tests fixe (graines figées)
- [ ] **Carte de survie** : heatmap du taux de survie en fonction (distance de tir × angle d'aspect), comparée aux heuristiques scriptées de la Phase 6 → c'est le résultat le plus parlant du projet
- [ ] Analyse qualitative des stratégies apprises dans Tacview (l'agent redécouvre-t-il la mise en travers ou la fuite ?)

---

## Phase 11 — Extensions possibles

- **Robustesse** : randomisation de domaine (masse, coefficients aéro ±10 %, bruit capteurs, vent)
- **Multi-menaces** : deux missiles, tirs décalés
- **Plus haute fidélité** : brancher [JSBSim](https://github.com/JSBSim-Team/jsbsim) (modèle F-16 inclus) et comparer au modèle maison
- **Combat aérien 1v1** en *self-play*
- **Contre-mesures** simplifiées (leurres modélisés comme perturbation de l'autodirecteur)
- **Explicabilité** : analyse de la politique (quelles observations pilotent les décisions)

---

## Risques principaux et parades

| Risque | Parade |
|---|---|
| Bug de signe / convention dans la dynamique | Tests unitaires Phase 1, pilotage manuel Phase 5 |
| Instabilité numérique (NaN) | dt = 0.01 s, RK4, normalisation du quaternion, détection NaN |
| Modèle trop difficile pour le RL | Commandes hiérarchiques, 3-DOF d'abord, curriculum |
| Reward hacking | Visualiser les trajectoires, termes de récompense loggés séparément |
| Entraînement trop lent | Vectorisation, numba, 3-DOF pour les grandes campagnes |
| Résultats non reproductibles | Graines fixées, configs YAML versionnées, ≥ 3 graines par expérience |

---

## Jalons (checkpoints de démonstration)

1. **J1** — L'avion 3-DOF vole en palier et vire correctement (fin Phase 2)
2. **J2** — L'avion 6-DOF est pilotable au clavier et visible dans Tacview (fin Phase 5)
3. **J3** — Un agent PPO stabilise l'avion et atteint un cap / une altitude (Phase 8.2)
4. **J4** — Un agent exécute un looping et un virage à taux max (Phase 8.5)
5. **J5** — Un agent évite un missile mieux que les heuristiques scriptées, carte de survie à l'appui (Phase 10)

---

## Références utiles

- Stevens, Lewis & Johnson — *Aircraft Control and Simulation* (modèle F-16 complet, équations 6-DOF)
- Nguyen et al. (1979) — NASA TP-1538, données aérodynamiques F-16 en soufflerie
- Zarchan — *Tactical and Strategic Missile Guidance* (navigation proportionnelle)
- Documentation [Gymnasium](https://gymnasium.farama.org/) et [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [JSBSim](https://github.com/JSBSim-Team/jsbsim) — moteur de dynamique de vol open source
- Format ACMI de [Tacview](https://www.tacview.net/) pour la visualisation
- DARPA AlphaDogfight Trials (2020) — contexte sur le RL en combat aérien simulé
