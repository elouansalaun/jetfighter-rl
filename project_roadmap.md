# jetfighter-rl — Roadmap du projet (phases 7 à 11)

> **Objectif final** : entraîner par apprentissage par renforcement (RL) un avion de chasse simulé (paramètres inspirés du F-16) à exécuter des manœuvres de vol, jusqu'à l'évitement d'un missile à autodirecteur.
>
> **Principe directeur** : on avance par couches. Chaque couche est **validée physiquement** avant d'être utilisée par la suivante. Un agent RL entraîné sur une physique fausse apprend à exploiter les bugs, pas à piloter.
>
> **Note** : le projet est réparti en deux dépôts. Ce document couvre l'environnement d'apprentissage et le RL (phases 7 à 11) ; le simulateur (phases 0 à 6) est dans [jetfighter-sim](https://github.com/elouansalaun/jetfighter-sim/blob/main/project_roadmap.md).

---

## Journal des décisions

| Date | Décision | Raison | Conséquences |
|---|---|---|---|
| 27/09/2026 | Avion de référence : **F-16** (au lieu du F-22) | Données aérodynamiques publiques (NASA TP-1538, Stevens & Lewis) | Tables réelles dans le modèle 6-DOF (phase 3) |
| 27/09/2026 | Centrage par défaut **x_cg = 0.30** (option (a) de la phase 3) | Avion naturellement stable, plus simple pour démarrer | x_cg = 0.35 (instable, comme le vrai F-16) reste disponible ; les commandes de vol électriques de la phase 6 le stabilisent |
| 27/09/2026 | Agent **hiérarchique d'abord, bas niveau ensuite** | Apprentissage bien plus rapide ; permet de valider toute la chaîne RL (environnement, récompenses, curriculum) avant d'attaquer le problème difficile | Phases 7, 8 et 10 en deux temps : l'agent commande d'abord (n_z, taux de roulis, manette) via les commandes de vol électriques, puis directement les gouvernes. La commande directe reste l'**objectif final** du projet |
| 28/09/2026 | Pénalité de crash = fixe **+ coût maximal des pas restants** (dans l'horizon 1/(1−γ)) | Loin de la consigne, s'écraser coûtait moins que continuer à voler : incitation à abréger les épisodes | S'écraser est toujours la pire issue ; chaque tâche déclare sa borne de coût par pas (`max_step_cost`) |
| 28/09/2026 | Mode hiérarchique : **action nulle = trajectoire maintenue** (n_z = cos γ / cos μ) | Avec « action nulle = 1 g », la politique devait produire 1/cos μ au centième près pour ne pas monter ou descendre en virage | Ancien comportement disponible (`nz_neutral: one_g`) |
| 28/09/2026 | Commandes de vol : consigne de n_z limitée à 20 g/s + anticipation du limiteur par q | Des inversions brutales de commande (typiques d'un agent RL) dépassaient −4 g (jusqu'à −5 g) | Plus aucune sortie d'enveloppe en inversion ±1 ; réponses aux échelons inchangées |
| 28/09/2026 | Entraînement : récompenses normalisées (`VecNormalize`), torch sur 1 fil, exploration initiale réduite (σ = 0.37) | Variance expliquée du critique ≈ 0 sans normalisation ; débit ÷ 6 avec plusieurs fils ; commandes aléatoires à σ = 1 → crash en quelques secondes | Réglages par défaut de `jetfighter_rl.training` |
| 28/09/2026 | Imitation de la référence disponible **en option** (`pretrain`), RL pur par défaut | Sur le Mac, PPO seul apprend 8.2 → 8.4 avec 3 à 5 M de pas ; l'imitation reste utile pour la voltige (le clone du pilote scripté réussit les 4 figures) | Variantes `*_imitation.yaml` |
| 28/09/2026 | 8.2 et 8.5 : **départ par imitation** dans le programme principal | 3 exécutions de 8.2 en RL pur : 90 %, 60 %, 20 % ; en voltige, RL pur ne découvre que le tonneau | RL pur conservé en variante (`8_2_heading_altitude.yaml`, `8_5_aerobatics.yaml`) |
| 28/09/2026 | Bas niveau (8.6) à **50 Hz**, récompenses par pas × agent_dt / 0.1 s, démarrage par imitation de « pilote auto + commandes de vol » | Les commandes de vol ne tiennent pas l'avion à 10 Hz ; rendements comparables entre cadences | Épisodes 5 fois plus longs en pas d'agent |
| 28/09/2026 | Observation d'attitude : **direction de la pesanteur en axes vent et corps** (au lieu des angles d'Euler) | Au passage de la verticale, μ et φ basculent de 180° : l'agent de voltige montait à la verticale puis s'y arrêtait | Modèles antérieurs incompatibles (lus en mode `euler` automatiquement) ; programme à relancer |
| 29/09/2026 | Projet réparti en **deux dépôts** : `jetfighter-sim` (paquet `jetsim`, phases 0–6) et `jetfighter-rl` (paquet `jetfighter_rl`, phases 7+) | Le simulateur est réutilisable seul ; le dépôt RL en dépend comme d'une bibliothèque (dépendance git versionnée) | Données avion livrées dans le paquet (`jetsim/data/`) ; `jetfighter.rl` devient `jetfighter_rl.training` |

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
| 7 | Environnement Gymnasium | `JetEnv` conforme `check_env`, modes hiérarchique et bas niveau | 1 semaine |
| 8 | Curriculum de manœuvres RL | Agents hiérarchiques, puis bas niveau : stabilisation → virage → voltige | 3–5 semaines |
| 9 | Modèle de missile (générique) | Missile à navigation proportionnelle | 1–2 semaines |
| 10 | RL d'évitement de missile | Agent d'évasion + carte de survie | 3–5 semaines |
| 11 | Extensions | Robustesse, multi-menaces, JSBSim, self-play | ouvert |

\* Estimations pour un développeur seul, en parallèle d'autres activités.

---

## Phase 7 — Environnement Gymnasium

- [x] Classe `JetEnv(gymnasium.Env)` : `reset(seed)`, `step(action)`, `render()`
- [x] **Espace d'action** continu `Box(-1, 1)` remappé vers les plages physiques, avec **deux modes** (paramètre `action_mode`) :
  - `"hierarchical"` — **par défaut, à faire en premier** : [n_z, taux de roulis, manette] → `HighLevelCommand` → boucle interne (`make_inner_loop`). Même espace d'action en 3-DOF et en 6-DOF ;
  - `"low_level"` — **dans un second temps** : 6-DOF [manette, δe, δa, δr] (gouvernes directes, sans limiteurs) ; 3-DOF [manette, α, taux de roulis] (ce modèle n'a pas de gouvernes) ;
  - fréquences : physique 100 Hz, boucle interne 50 Hz, décision de l'agent 10 Hz
- [x] **Espace d'observation** normalisé (≈ [−1, 1]) :
  - états propres : V/V_ref, h/h_ref, sin/cos des angles (pas les angles bruts → discontinuité à ±180°), inclinaison μ, α, β, p, q, r, Nz, manette
  - [~] en mode bas niveau : position des gouvernes en plus — pour l'instant, l'action précédente (= consigne des gouvernes) ; la position réelle des actionneurs reste à ajouter si l'agent bas niveau en a besoin (étape 8.6)
  - [~] erreur par rapport à la cible : en écarts relatifs (Δh, sin/cos Δcap, ΔV), suffisant pour ces tâches ; l'expression **dans le repère avion** sera nécessaire pour les cibles 3D (poursuite, missile)
- [x] Récompense dans `rewards.py`, modulaire (somme de termes pondérés, chaque terme loggé séparément)
- [x] Distinction `terminated` (crash, succès) / `truncated` (limite de temps)
- [x] Randomisation des conditions initiales (altitude, vitesse, cap, attitude)
- [x] `gymnasium.utils.env_checker.check_env` passe
- [~] Environnements vectorisés (`SubprocVecEnv`) ; mesurer les pas/seconde (objectif : > 5 000 pas/s en 3-DOF) — mesuré, objectif non atteint par cœur (voir ci-dessous)
- [ ] Option perf : `numba` sur la dynamique si le débit est insuffisant — reporté à la phase 8, si les entraînements s'avèrent trop longs

**Réalisé** (`src/jetfighter_rl/envs/` : `jet_env.py`, `tasks.py`, `rewards.py`, `baselines.py` ; vérification : `scripts/phase7_env_check.py`)
- **Identifiants** `gymnasium.make("JetFighter/Level-v0")` et `"JetFighter/HeadingAltitude-v0"`, paramètres surchargeables (`model="6dof"`, `action_mode="low_level"`, `sensors="realistic"`, `record=True`…). `check_env` passe sur les 8 combinaisons tâche × modèle × mode.
- **Tâches** (`tasks.py`, une classe par tâche : conditions initiales, observations propres, termes de récompense, critère de succès, consignes du pilote auto) :
  - `level` (30 s) — rattrapage d'assiette inusuelle (μ ±80°, γ ±25°, roulis ±60°/s) → palier ailes horizontales ;
  - `heading_altitude` (90 s) — rallier un cap, une altitude (±1 500 m) et une vitesse tirés au hasard.
- **Observation** : 19 grandeurs propres (lues via les **capteurs**, bruités si demandé) + action précédente + erreurs de la tâche, bornées à ±10, `float32`.
- **Récompense** : termes pondérés dans [−1, 0] par pas + bonus de succès, pénalité de lissage des actions, −50 en cas de sortie d'enveloppe/crash ; détail par terme dans `info["reward_terms"]` et `info["episode_terms"]`.
- **Conditions initiales** : équilibre à la pente demandée, sinon équilibre en palier auquel on impose pente, inclinaison et roulis (les piqués raides n'ont pas d'équilibre).
- **Enregistrement** optionnel de l'épisode (rejeu exact, export Tacview).
- **Références** (3-DOF, 6 épisodes) : pilote automatique `level` rendement ≈ 19–20, 100 % de succès ; `heading_altitude` ≈ 40–55, 100 % de succès ; politique aléatoire ≈ −200 / −325 (100 % de crash sur `heading_altitude`). Le pilote auto réussit aussi en 6-DOF.
- **Débit** (VM 2 cœurs) : 3-DOF ≈ 480 pas d'agent/s pour 1 env, ≈ 590 avec 2 env ; 6-DOF ≈ 120 / 190. Un pas d'agent = 5 pas de contrôle + 10 pas de physique RK4 : ≈ 4 800 pas de physique/s en 3-DOF. Le goulot est la dynamique en Python pur ; attendu ≈ 4–5 k pas/s sur 8 cœurs. `numba` reporté.
- **Essai PPO** (stable-baselines3, `level`, 3-DOF, hiérarchique, 2 env) : 100 % de succès dès ≈ 25 k pas, rendement 20.5 à 49 k pas (≥ pilote auto), ≈ 2 min 30 s. La chaîne complète (env → PPO → TensorBoard → modèle → vol Tacview) fonctionne.

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
| 8.6 | **Passage au bas niveau** : reprendre 8.1 → 8.5 en mode `low_level` sur le 6-DOF | ≥ 90 % des performances de l'agent hiérarchique, sans dépasser les limites structurales (plus de limiteurs pour le protéger) | mêmes récompenses + pénalité de surcharge et de dérapage |

**8.6 — mise en place** (`action_mode: low_level`, configurations `8_6_*.yaml`, programme `curriculum_low_level.yaml`) :
- l'agent commande manette, profondeur, ailerons et direction à **50 Hz** : les commandes de vol de référence, réglées pour 50 Hz, oscillent et s'écrasent à 10 Hz ;
- récompenses par pas mises à l'échelle de la cadence (× agent_dt / 0.1 s) : les rendements restent comparables avec l'agent hiérarchique ; bonus et progression inchangés ;
- observation : position réelle des gouvernes en plus ; pénalités `overload` (au-delà de +8 / −2 g) et `sideslip` (β² au-delà de 5°) ;
- référence bas niveau = pilote automatique (ou pilote scripté) **+ commandes de vol**, dont on relève les ordres aux gouvernes : mêmes performances qu'en hiérarchique (8.1 : 19.4 contre 19.8 ; voltige 56.6 contre 56.6) ;
- démarrage par **imitation** de cette référence (piste de la roadmap), puis critique seul, puis PPO.
- **validation locale (8.1, 500 000 pas à 50 Hz)** : le clone vole à 15.4 / 100 % (référence 16.9 / 100 %). Avec les réglages hiérarchiques, PPO dégradait ce clone (−45, 0 % de réussite en 400 000 pas) : bruit d'exploration de 3.5° sur la profondeur à 50 Hz et λ = 0.95 qui ne « voit » que 0.4 s. Avec σ ≈ 0.7°, λ = 0.99 et un pas plus petit, la réussite reste à 100 % mais le rendement ne progresse pas encore (10.2) : critère « ≥ 90 % de l'agent hiérarchique » **pas encore atteint**, à juger sur les 5 M de pas du Mac.
- **Résultats 8.6 sur le Mac** (1 graine, 5 M de pas à 50 Hz, meilleur modèle) :

  | Tâche | Bas niveau (imitation + PPO) | Référence bas niveau | Meilleur agent hiérarchique 6-DOF | Ratio bas niveau / hiérarchique |
  |---|---|---|---|---|
  | 8.1 stabilisation | 16.7 / 100 % | 16.9 / 100 % | 20.0 / 100 % | 84 % |
  | 8.2 cap / altitude | 89.5 / 100 % | 91.0 / 100 % | 123.7 / 90 % (1re exécution) | 72 % (mais 100 % de réussite contre 90 %) |
  | 8.3 virage soutenu | **225.1 / 100 %** | 165.6 / 100 % | 266.5 / 100 % | 84 % |
  | 8.4 points de passage | 38.3 / 90 % | 38.9 / 90 % | 46.0 / 100 % | 83 % |
  | 8.5 voltige | **57.4 / 100 %** | 57.4 / 100 % | 55.3 / 100 % (RL pur, avant correction) | ≈ 100 % |

  Aucune sortie d'enveloppe (terme `overload` toujours nul), dérapage négligeable. Critère « ≥ 90 % de l'agent hiérarchique » : atteint en voltige, presque en 8.1 (84 %), pas encore ailleurs ; seul le virage soutenu progresse nettement au-delà de l'imitation (+36 %). Sur 8.2 et 8.4, PPO **dégrade** la politique imitée après 1–2 M de pas (8.2 : 89.5 → −69.6 en fin d'entraînement) sans que les indicateurs d'apprentissage (KL, variance expliquée, σ) ne signalent rien : on conserve le meilleur modèle évalué. Pistes pour la suite (phase 10) : régularisation vers la politique imitée pendant le RL, ou arrêt anticipé sur l'évaluation.

**Ordre retenu** (décision du 27/09/2026) : tâches 8.1 → 8.5 en **mode hiérarchique** (3-DOF puis 6-DOF), puis 8.6 en **bas niveau**. Pistes pour faciliter 8.6 :
- initialiser la politique bas niveau par **imitation** (clonage de comportement) de l'ensemble « agent hiérarchique + commandes de vol » ;
- ou **RL résiduel** : l'agent apprend une correction ajoutée aux commandes de vol, qu'on réduit progressivement jusqu'à la commande directe.

**Bonnes pratiques** :
- [x] Commencer en 3-DOF, transférer en 6-DOF ensuite (`init_from`, copie des poids ; programme `configs/training/curriculum.yaml`)
- [x] Pénaliser les variations brusques de commande (|Δaction|) pour des trajectoires lisses (terme `smoothness` de chaque tâche)
- [x] Surveiller le *reward hacking* : toujours regarder les trajectoires (Tacview), pas seulement la courbe de récompense — `scripts/evaluate_agent.py` superpose agent et pilote auto (Tacview + tracés) ; 4 cas trouvés et corrigés (voir « Réalisé »)
- [x] Comparer systématiquement à la baseline PID (référence évaluée sur les mêmes graines à chaque entraînement, courbe `eval/baseline_return` dans TensorBoard)

**Avancement** :
- [x] Outillage d'entraînement (`src/jetfighter_rl/training/`) : configurations YAML versionnées, PPO et SAC, 3 graines, TensorBoard détaillé, évaluation périodique contre la référence, meilleur modèle, transfert de politique, programme complet, imitation optionnelle
- [x] Tâches 8.1 → 8.5 et leurs politiques de référence, validées en 3-DOF et en 6-DOF
- [x] 8.1 → 8.4 appris par PPO en 3-DOF, **au niveau du pilote automatique ou mieux** (programme complet sur le Mac, 1 graine, ≈ 2 h 40)
- [x] Transfert 6-DOF : 8.1, 8.3 et 8.4 réussis ; 8.2 irrégulier en RL pur → **départ par imitation** retenu dans le programme
- [x] 8.5 voltige : **par imitation du pilote scripté puis PPO** (100 %, figures conformes). En RL pur, seul le tonneau est découvert (35 %, 3e exécution, tâche corrigée)
- [x] 8.6 — bas niveau (gouvernes directes, 6-DOF, 50 Hz) : 5 tâches à 90–100 % de réussite (voir « 8.6 »)
- [ ] 3 graines par expérience (fait avec 1 graine pour l'instant)

**Réalisé** (`src/jetfighter_rl/envs/tasks/`, `src/jetfighter_rl/training/`, `configs/training/`, `scripts/train.py`, `scripts/evaluate_agent.py`, `scripts/run_curriculum.py`)
- **Tâches** (une classe par tâche : conditions initiales, observations relatives, termes de récompense, critère de réussite de la roadmap, métriques, politique de référence) :

  | Tâche | Critère de réussite implémenté | Référence (3-DOF) |
  |---|---|---|
  | 8.1 `level` | rétabli (\|μ\| < 5°, \|γ\| < 2°, tenu 2 s) en moins de 10 s | 100 %, rétablissement ≈ 2.8 s |
  | 8.2 `heading_altitude` | dans la tolérance à la fin (30 m, 3°, 5 m/s), dépassement ≤ max(50 m, 10 %) et ≤ 5° | 100 % |
  | 8.3 `sustained_turn` | taux moyen ≥ 90 % du taux soutenu optimal sur 20 s, altitude ± 200 m, pas de perte d'énergie | 100 %, 1.02 × l'optimum |
  | 8.4 `waypoints` | 4 points de passage 3D franchis (400 m / 150 m) | 100 % |
  | 8.5 `aerobatics` | looping, tonneau, Immelmann ou Split-S complet, puis ailes à plat | 100 % pour les 4 figures (pilote scripté) |

  Toutes les références réussissent aussi en 6-DOF (virage soutenu : seuil corrigé de 10 %, l'optimum étant calculé sur le modèle point-masse).
- **Géométrie sans singularité** pour la voltige (`wind_axes`) : progression suivie sur le vecteur vitesse et la portance, pas sur les angles d'Euler (indéfinis à la verticale).
- **Entraînement** : `python scripts/train.py configs/training/<tâche>.yaml` ; résultats dans `runs/<nom>/<date>/seed<k>/` (TensorBoard, meilleur modèle, historique CSV, résumé multi-graines).
- **Résultat 8.1** (PPO, 3-DOF, 200 000 pas) : 100 % de réussite, rendement 20.3 contre 17.8 pour le pilote auto, **rétablissement en 1.4 s contre 2.8 s** (l'agent tire plus fort, jusqu'à ≈ 8 g).
- **Reward hacking et pièges trouvés en regardant les trajectoires** :
  1. 8.1 : rétablissement rapide puis vitesse laissée décroître indéfiniment → terme de tenue de vitesse ;
  2. 8.3 : **spirale descendante** (taux ×2.5 mais 3 km d'altitude perdus) → le taux de virage n'est récompensé que s'il est soutenu (facteur exp(−Δh − ΔE)) ;
  3. 8.5 : l'agent apprenait à **ne pas faire la figure** (voler droit ne coûtait rien) → progression dominante (40 points + 20 à la sortie) ;
  4. tous : incitation à s'écraser loin de la consigne → pénalité de crash couvrant le coût restant.
- **Limites des commandes de vol révélées par l'agent** : l'agent 3-DOF transféré sur le 6-DOF alternait +8 / −3 g et dépassait −4 g → limitation de la vitesse de variation de la consigne (voir le journal des décisions).
- **Résultats sur le Mac** (programme complet, 1 graine, 8 environnements, observations en angles d'Euler) — meilleur modèle en évaluation (10–20 épisodes) :

  | Tâche | Modèle | Agent PPO : rendement / réussite | Pilote auto : rendement / réussite |
  |---|---|---|---|
  | 8.1 stabilisation | 3-DOF | **20.8 / 100 %** | 17.8 / 100 % |
  | 8.1 stabilisation | 6-DOF (transfert, 200 k pas) | −3.9 / 20 % — oscillation de roulis à 5 Hz | 17.3 / 100 % |
  | 8.2 cap/altitude/vitesse | 3-DOF (3 M pas) | **126.8 / 90 %** | 89.0 / 100 % |
  | 8.2 cap/altitude/vitesse | 6-DOF (transfert, 1 M pas) | **123.7 / 90 %** | 91.3 / 100 % |
  | 8.3 virage soutenu | 3-DOF (3 M pas) | **289.7 / 100 %** | 274.2 / 100 % |
  | 8.3 virage soutenu | 6-DOF (transfert, 1 M pas) | **265.7 / 100 %** | 165.6 / 100 % |
  | 8.4 points de passage | 3-DOF (5 M pas) | **45.1 / 100 %**, parcours en 111 s | 39.2 / 90 %, 143 s |
  | 8.4 points de passage | 6-DOF (transfert, 1.5 M pas) | 32.1 / 90 %, 10 % de crash | 39.1 / 90 % |
  | 8.5 voltige | 3-DOF (5 M pas) | 25.0 / 35 % (tonneau seulement) | 57.5 / 100 % |

  → **Jalon J3 atteint** (8.2). Sur 8.2 l'agent obtient un meilleur rendement que la référence mais échoue plus souvent au critère strict (dépassement), sa réussite oscille entre 30 et 90 % d'une évaluation à l'autre.
- **2e exécution sur le Mac** (observations d'attitude continues, 1 graine) :

  | Tâche | Agent PPO : rendement / réussite | Pilote auto |
  |---|---|---|
  | 8.1, 3-DOF / 6-DOF | 18.5 / 100 % — **20.0 / 100 %** (le transfert 6-DOF passe de 20 % à 100 %) | 17.8 / 17.3 |
  | 8.2, 3-DOF / 6-DOF | 74.4 / 60 % — 60.3 / 50 % (meilleurs modèles ; en fin d'entraînement, 0 %) | 89.0 / 91.3 |
  | 8.3, 3-DOF / 6-DOF | **290.9 / 100 %** — **266.5 / 100 %** | 274.2 / 165.6 |
  | 8.4, 3-DOF / 6-DOF | **48.6 / 100 %** — **46.0 / 100 %** | 39.2 / 39.1 (90 %) |
  | 8.5, 3-DOF / 6-DOF | 56.6 / 100 % — 55.3 / 100 %, mais *reward hacking* | 57.9 / 56.9 |
  | 8.5 par imitation, 3-DOF | **58.2 / 100 %** (figures conformes) | 57.9 |

  8.2 : l'apprentissage oscille (réussite 0 → 60 → 0 % d'une évaluation à l'autre) → décroissance linéaire du pas d'apprentissage (`lr_final`) sur 8.2 → 8.5. La première exécution, identique à la représentation de l'attitude près, avait atteint 90 % : 3 graines seront nécessaires pour conclure.
- **8.5, reward hacking n° 5** : en RL pur, « looping » en 6 s au lieu de 25 s. En regardant la trajectoire : l'agent virait de 90° puis faisait de petits mouvements autour de l'axe latéral ; l'angle de la vitesse **projetée** dans le plan de la boucle, mal défini quand la vitesse est perpendiculaire au plan, « tournait » alors de 360° sans looping. Corrigé : le tangage ne compte plus quand la vitesse sort du plan de plus de 30°, et la sortie doit se faire au bon cap (± 30°). Réévalué avec la tâche corrigée, cet agent ne réussit plus que le tonneau ; le pilote scripté reste à 100 %.
- **8.5, diagnostic (1re exécution)** : l'agent pilotait bien jusqu'à la verticale, puis s'y arrêtait (looping et Immelmann à 26 % de progression, Split-S à 50 %). Au passage de la verticale, les angles d'Euler de l'observation basculent de 180° : l'agent « se croyait sur le dos ». Corrections : attitude observée sous forme de pesanteur en axes vent et corps (continue), progression récompensée par records (une tentative ratée n'annule plus le gain), pénalités de suivi réduites (cumulées sur 45 s, elles dissuadaient d'essayer). Variante `8_5_aerobatics_imitation.yaml` : le clone du pilote scripté réussit les 4 figures à 100 % avant tout RL.
- **Transfert 6-DOF de 8.1** : l'agent 3-DOF alterne des commandes de roulis à chaque pas (sans conséquence en 3-DOF, où la réponse est instantanée) ; en 6-DOF cela devient une oscillation. Budget relevé à 1 M de pas ; à surveiller pour 8.6 (pénalité de lissage plus forte si besoin).
- **8.2, constat lors de la validation locale** : PPO (depuis zéro ou depuis la politique 8.1, avec ou sans normalisation, 2 ou 8 environnements, bruit réduit, courbe « expo », n_z compensé) et SAC n'ont pas dépassé le niveau « voler droit » en 250 000 à 600 000 pas. Le signal existe (l'imitation du pilote auto fait 97 contre 99) ; l'hypothèse est un simple manque d'échantillons (budgets des configurations : 3 à 5 millions de pas, soit 15 à 30 min par graine sur le Mac). Si ce n'est pas le cas : variante `8_2_heading_altitude_imitation.yaml` (imitation, puis critique seul, puis PPO prudent).

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
6. Passage du mode hiérarchique au mode bas niveau (même démarche qu'en 8.6)

### 10.4 Évaluation
- [ ] Taux de survie sur un banc de tests fixe (graines figées)
- [ ] **Carte de survie** : heatmap du taux de survie en fonction (distance de tir × angle d'aspect), comparée aux heuristiques scriptées de la Phase 6 → c'est le résultat le plus parlant du projet
- [ ] Analyse qualitative des stratégies apprises dans Tacview (l'agent redécouvre-t-il la mise en travers ou la fuite ?)
- [ ] Comparaison agent hiérarchique / agent bas niveau : la commande directe apporte-t-elle un gain (manœuvres au-delà des limiteurs) ou seulement de la difficulté ?

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
3. **J3** — Un agent PPO stabilise l'avion et atteint un cap / une altitude (Phase 8.2) — ✅ atteint le 28/09/2026 (3-DOF et 6-DOF)
4. **J4** — Un agent (hiérarchique) exécute un looping et un virage à taux max (Phase 8.5) — ✅ 28/09/2026 (looping : imitation puis PPO ; virage : PPO seul, 106 % de la référence)
5. **J4b** — Un agent **bas niveau** (gouvernes directes, 6-DOF) égale l'agent hiérarchique sur les tâches 8.1 → 8.5 (Phase 8.6) — 🟡 partiel : 90–100 % de réussite partout, 72–100 % du rendement hiérarchique
6. **J5** — Un agent évite un missile mieux que les heuristiques scriptées, carte de survie à l'appui — d'abord en hiérarchique, puis en bas niveau (Phase 10)

---

## Références utiles

- Stevens, Lewis & Johnson — *Aircraft Control and Simulation* (modèle F-16 complet, équations 6-DOF)
- Nguyen et al. (1979) — NASA TP-1538, données aérodynamiques F-16 en soufflerie
- Zarchan — *Tactical and Strategic Missile Guidance* (navigation proportionnelle)
- Documentation [Gymnasium](https://gymnasium.farama.org/) et [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [JSBSim](https://github.com/JSBSim-Team/jsbsim) — moteur de dynamique de vol open source
- Format ACMI de [Tacview](https://www.tacview.net/) pour la visualisation
- DARPA AlphaDogfight Trials (2020) — contexte sur le RL en combat aérien simulé
