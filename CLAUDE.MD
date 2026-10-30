# Contexte pour Claude Code — hackathon Rock Climbing Robots

Équipe d'étudiants (Master ASPIC, Bordeaux). Ce dépôt contient des copies de `microban/` (Rhoban, branche dev_v2)
et `simulator/` (simulateur officiel du hackathon), plus notre projet RL `climbing_rl/`. Machine : Windows 11 + WSL
Ubuntu, RTX 5070 (12 Go, sm_120). L'utilisateur écrit en français : répondre en français.

## Règles de travail
- Ne jamais modifier `microban/` ni `simulator/` (copies de dépôts externes ; on lit seulement leurs fichiers).
- Travailler dans `climbing_rl/`. Avant tout changement important, vérifier que git est propre (`git status`).
- Après toute modification de la physique, du robot ou de la scène : `cd climbing_rl && uv run check-scene --num-envs 64 --seconds 3`.
  Attendu : aucun `nefc overflow`, les 4 membres accrochés à 100 %, « OK : le robot tient au mur ».
- Un seul changement à la fois, puis test. Signaler l'effet sur les « Steps per second » de l'entraînement.
- Ne pas lancer d'entraînement long sans le demander (il occupe le GPU de l'utilisateur).

## climbing_rl : architecture
mjlab 1.3 (RL massivement parallèle sur GPU, API type Isaac Lab, physique MuJoCo Warp), PPO via rsl_rl 5.0.1.

| Fichier | Rôle |
|---|---|
| `src/climbing_rl/wall.py` | lit les murs JSON et la géométrie des prises dans `simulator/climbing_sim/` ; formules des positions des coupes (recopiées de `simulator/climbing_sim/wall.py`) |
| `src/climbing_rl/robot.py` | robot = `microban_model/mjcf/robot_climb.xml` ; collisions "light"/"full" ; servos BAM ; pose accrochée (méthode de `ClimbingSim.reset_hanging` : `hang.json` + recalage SVD + IK) |
| `src/climbing_rl/scene.py` | ajoute sol, mur, prises (jusqu'à `max_row`) via `SceneCfg.spec_fn` |
| `src/climbing_rl/tasks/mdp.py` | fonctions d'escalade : distances boule/coupe, accroché (< 6 mm), récompenses, terminaisons |
| `src/climbing_rl/tasks/climb_env_cfg.py` | config des tâches (`_base_cfg` commun, `hang_env_cfg`, `left_foot_env_cfg`) + `ppo_cfg` |
| `src/climbing_rl/tasks/__init__.py` | enregistre `Climb-Hang-Microban` et `Climb-LeftFoot-Microban` (point d'entrée `mjlab.tasks`) |
| `src/climbing_rl/check_scene.py` | test sans entraînement (`uv run check-scene`) |

Commandes : `uv run train <tâche> --env.scene.num-envs 8192`, `uv run play <tâche> --checkpoint-file logs/rsl_rl/<exp>/<date>/model_N.pt --viewer viser --num-envs 4`.

## Pièges connus (déjà résolus, ne pas défaire)
- **Versions** : BAM (`better-actuator-models[mjlab]`, commit e9a619d, même que le simulateur officiel) impose mjlab 1.3,
  MuJoCo 3.7, mujoco-warp 3.7, warp 1.12, Python 3.12. Le simulateur officiel exige MuJoCo ≥ 3.11 : **ne pas importer
  `climbing_sim` ni `microban_sim`** dans climbing_rl (on relit leurs fichiers de données).
- `wandb==0.22.3` épinglé : les versions récentes cassent rsl_rl 5.0.1 (`start_method`).
- `Rhoban/mjlab_microban` : NE PAS utiliser son robot (ancien Microban de marche, sans boules). Seule inspiration pour
  la structure de tâche et la config BAM.
- `njmax=400, nconmax=128` dans `SimulationCfg` : sinon `nefc overflow` → contacts ignorés → les mains tombent.
- Pas de physique **4 ms**, décimation 5 (20 ms par décision), validé par check-scene. Simulateur officiel : 2 ms.
- `terrain=None` : toutes les origines d'env à 0, le mur est au même endroit dans chaque monde (normal).
- Avertissement « Actuator config matched 19 joint(s) ... 8 site(s) » : sans gravité.
- Avertissements « solver/linesearch iterations limit reached » : fréquents en début d'entraînement, sans gravité.
  En revanche `nefc overflow` est grave.

## Modèle et fidélité au vrai robot
- Mur N1 : barreaux tous les 5 rangs (7,5 cm), 4 prises par barreau (colonnes 18, 21, 26, 29). Départ : mains (18,22)
  et (29,22), pieds (21,2) et (26,2). Les pieds partent à ~2,6 cm du sol.
- Servos BAM m6 XL330 : `kp_fw=125`, vin 7,2–7,8 V, chute de tension, `max_current=1.75`, latence 10–30 ms.
- Collisions "light" (défaut entraînement) : seules les 4 boules (`*_ball_coll`, sphères r = 12,5 mm) et les 7 boîtes
  des servos sont actives ; les 269 maillages convexes du corps sont désactivés (trop coûteux sur GPU).
- Prises en mode light : seulement les ~30 sphères de la coupe (`cup_spheres.json`) ; `hold_meshes=True` ajoute les
  14 maillages convexes (plaque, goussets, bloc arrière, peau sous la coupe).

## Pied / tige qui traversaient le mur (résolu)
- Le `robot_climb.xml` installé (microban_model GitHub) A des maillages de collision pour le pied (24 morceaux) et
  l'avant-bras : c'est le mode "light" qui les désactivait tous. Correctif : `LIGHT_BOXES` dans `robot.py`, 5 boîtes
  par pied et 4 par avant-bras (dont la tige de la boule), mesurées sur ces morceaux ; contype 1 / conaffinity 0
  (monde seulement, pas d'auto-collision). Vitesse inchangée (~49 000 pas d'env/s à 4096 envs).
- Prises : `wall_spec_fn(hold_meshes=[(col, row), ...])` maille seulement certaines prises ; LeftFoot maille les
  rangées 2 et 7 (`_base_cfg(mesh_holds=...)`). Coût : −19 % (~39 600 pas/s), identique avec 4 ou 8 prises.
- Effet connu : dans la pose de départ, la semelle pénètre le sol de ~6 mm (le vrai maillage aussi) → le robot
  repose sur ses pieds, le tronc ne descend plus (−2 mm au lieu de +16 mm), mains à ≤ 3 mm de la coupe.
- Les politiques entraînées avant ce correctif ne sont plus valables : réentraîner LeftFoot.

## Tâche LeftFoot (état actuel)
Pied gauche de (21,2) vers (18,7) (en haut à gauche ; la prise juste au-dessus (21,7) est hors de portée, comme
constaté à la main sur le vrai robot). Récompenses : `others_hooked` 1, `foot_reach` 2 (σ 4 cm), `foot_hooked` 5,
`action_rate` −0,01, `joint_limits` −1. Terminaisons : `limb_lost` (> 3 cm), `fell` (tronc −5 cm), 8 s.
Le pied devra zigzaguer (extérieur/intérieur) d'un barreau à l'autre.
