# climbing_rl — RL du Microban grimpeur avec mjlab (Léonard Bettcher)

Ce projet a été ma première expérience en apprentissage par renforcement (Reinforcement Learning).
J'ai utilisé une IA (Claude) pour assembler l'environnement mjlab à partir de trois dépôts existants (modèle du robot,
modèle des servos BAM, simulateur officiel) qui ne fonctionnaient pas ensemble tels quels, pour améliorer les
collisions du robot, puis pour apprendre comment fonctionnent PPO et la structure d'une tâche à partir d'un exemple.
J'ai mis en place l'environnement GPU sur mon ordinateur (RTX 5070), testé la simulation et repéré ses défauts
(contacts perdus, membres qui traversaient le mur, les prises ou l'autre jambe), puis créé et entraîné mes propres
tâches avec les connaissances apprises pendant le hackathon.

Tâches [mjlab](https://github.com/mujocolab/mjlab) (RL massivement parallèle sur GPU) pour le Microban grimpeur
(mains en boule) sur le mur du hackathon Rock Climbing Robots.*

## Dépôts utilisés

| Dépôt | Ce qu'on en prend |
|---|---|
| [Rhoban/microban_model](https://github.com/Rhoban/microban_model) | le modèle MJCF du robot grimpeur (`robot_climb.xml`, mains en boule), dépendance uv |
| [Rhoban/bam](https://github.com/Rhoban/bam) | les servos XL330 simulés sur GPU (`bam.mjlab`), dépendance uv, même commit que le simulateur officiel |
| [rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator) | le simulateur officiel du hackathon : murs, géométrie des prises et posture accrochée, relus comme fichiers de données (dossier `simulator/` du dépôt d'équipe) |

S'y ajoutent [mujocolab/mjlab](https://github.com/mujocolab/mjlab) (le framework RL, installé par BAM) et
[Rhoban/mjlab_microban](https://github.com/Rhoban/mjlab_microban), dont on a repris la structure de tâche et les
réglages PPO (son robot est l'ancien Microban de marche, sans boules).

## Installation

Ce dossier doit être dans le dépôt de l'équipe, à côté de `simulator/` : on y lit les murs, la géométrie des prises et
la posture accrochée. Il faut un GPU NVIDIA (MuJoCo Warp ne fonctionne qu'avec CUDA) et Git LFS (maillages du robot).

```bash
sudo apt install -y git-lfs && git lfs install
cd climbing_rl
uv sync --python 3.12
uv run check-scene --num-envs 64 --seconds 3     # le robot tient-il au mur sans politique ?
```

`check-scene` vérifie la pose de départ (erreur de l'IK, aucun contact entre membres), puis simule quelques secondes
avec action nulle : on attend les 4 boules accrochées dans 100 % des environnements et aucun `nefc overflow`.
Option `--task <tâche>` pour tester une autre tâche.

## Tâches

| Tâche | But |
|---|---|
| `Climb-Hang-Microban (exemple Claude)` | rester suspendu aux 4 prises de départ |
| `Climb-LeftFoot-Microban (exemple Claude)` | pied gauche de (21, 2) vers (18, 7), en haut à gauche, les 3 autres membres restant accrochés |
| `Climb-BothFeet-Microban (travail personel)` | les deux pieds montent d'un barreau (en haut à gauche et en haut à droite), les deux mains restant accrochées |

La prise juste au-dessus d'un pied, (21, 7), est hors de portée : les pieds montent en diagonale (zigzag).

```bash
# regarder une tâche sans politique
uv run play Climb-BothFeet-Microban --agent zero --viewer viser --num-envs 4

# entraîner (6144 robots tiennent dans les 12 Go d'une RTX 5070)
uv run train Climb-BothFeet-Microban --env.scene.num-envs 6144

# reprendre un entraînement depuis un checkpoint
uv run train Climb-BothFeet-Microban --env.scene.num-envs 6144 \
  --agent.resume True --agent.load-run <dossier du run> --agent.load-checkpoint model_<N>.pt

# regarder le résultat
uv run play Climb-BothFeet-Microban --checkpoint-file logs/rsl_rl/climb_both_feet/<date>/model_<N>.pt --viewer viser --num-envs 4
```

## Organisation

| Fichier | Rôle |
|---|---|
| `src/climbing_rl/wall.py` | lit le mur (JSON du simulateur), positions des coupes |
| `src/climbing_rl/robot.py` | MJCF grimpeur, collisions en boîtes, servos BAM, pose accrochée |
| `src/climbing_rl/scene.py` | ajoute le mur, les prises et le sol à la scène |
| `src/climbing_rl/tasks/mdp.py` | fonctions propres à l'escalade : observations, récompenses, terminaisons |
| `src/climbing_rl/tasks/climb_env_cfg.py` | configuration complète des tâches + réglages PPO |
| `src/climbing_rl/tasks/__init__.py` | enregistrement des tâches |
| `src/climbing_rl/check_scene.py` | vérification de la scène sans entraînement |

Les quatre premiers fichiers décrivent le monde physique et ne changent presque plus. Mon travail sur les tâches c'est fait
dans `climb_env_cfg.py` et `mdp.py`.

### `climb_env_cfg.py` : les règles du jeu

Ce fichier ne simule rien : ses fonctions **renvoient une description** de l'environnement, que mjlab lit pour
construire les milliers de robots sur le GPU. Chaque règle suit le même motif :

```python
RewardTermCfg(func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": LEFT_FOOT})
```

À chaque pas, mjlab appelle `func(env, **params)` pour tous les robots à la fois et multiplie le résultat par `weight`.

- **`_base_cfg`** : ce qui est commun à toutes les tâches.
  - *Scène* : le robot (`robot.py`) et le mur (`scene.py`), tous les robots au même endroit, chacun dans son monde.
  - *Actions* : 18 consignes d'angle (pas la tête), consigne = pose accrochée + 0,5 × action. BAM transforme ensuite
    la consigne en couple réaliste (couple limité, chute de tension, latence).
  - *Observations* : l'**actor** ne voit que ce que le vrai robot mesure (gyroscope, gravité dans le tronc, angles et
    vitesses des servos, dernière action ; 60 valeurs), avec bruit et retards. Le **critic** voit en plus la vérité
    terrain (position de chaque boule par rapport à sa coupe cible ; 79 valeurs), car il ne sert qu'à l'entraînement.
  - *Événements* : `bam_init` au démarrage (une copie des frottements des servos par robot), retour à la pose
    accrochée au début de chaque épisode.
  - *Terminaisons* : temps écoulé, chute (tronc descendu de plus de 5 cm), NaN.
  - *Pénalités* : saccades (`action_rate`), articulations en butée (`joint_limits`).
  - *Physique* : pas de 4 ms, une décision toutes les 20 ms comme le vrai robot.
- **Une fonction par tâche** (`hang_env_cfg`, `left_foot_env_cfg`, `both_feet_env_cfg`) : elle prend la base, choisit
  les prises cibles, ajoute ses récompenses et ses terminaisons.
- **`ppo_cfg`** : les réglages de PPO, repris de la tâche de marche de Rhoban (`mjlab_microban`).

Les récompenses suivent le schéma classique contre les récompenses rares : une **récompense dense** qui guide
(`foot_reach`, une cloche selon la distance à la cible), un **bonus** quand le but est atteint (`foot_hooked`, boule à
moins de 6 mm du centre de la coupe), une **contrainte douce** sur les appuis (`others_hooked`) et une **contrainte
dure** qui coupe l'épisode s'ils lâchent (`limb_lost`, plus de 3 cm).

### `mdp.py` : la boîte à outils de l'escalade

De petites fonctions de quelques lignes, chacune calculant une seule chose pour tous les robots à la fois (tenseurs
PyTorch sur le GPU). Les membres sont désignés par leur numéro : 0 main gauche, 1 main droite, 2 pied gauche,
3 pied droit. « Accroché » a le même sens que `hooked()` du simulateur officiel : centre de la boule à moins de 6 mm
du centre de la coupe.

| Fonction | Rôle | Utilisée comme |
|---|---|---|
| `ball_positions`, `ball_distances` | positions des 4 boules, distance de chacune à sa coupe cible | brique de base |
| `ball_offsets` | vecteur boule → coupe cible pour les 4 membres (12 valeurs) | observation du critic |
| `hooked_flags` | chaque boule accrochée ou non (4 valeurs) | observation du critic |
| `balls_near_sockets` | note de 0 à 1 par boule selon sa distance (cloche de largeur σ), moyenne sur les membres choisis | récompense (guidage, maintien) |
| `balls_hooked` | fraction des membres choisis accrochés | récompense (bonus) |
| `limb_reach` | cloche de distance pour un seul membre vers sa cible | récompense (guidage) |
| `limb_hooked` | 1 si ce membre est accroché à sa cible | récompense (bonus) |
| `trunk_dropped` | le tronc est descendu sous sa hauteur de départ | terminaison (chute) |
| `limbs_lost` | un membre d'appui s'est éloigné de sa coupe | terminaison (appui perdu) |

On ajoute une fonction ici seulement quand aucune ne mesure ce qu'on veut récompenser ; on l'utilise ensuite comme
règle dans `climb_env_cfg.py`.

## Fidélité au vrai robot

- Servos : BAM, modèle m6 du XL330 identifié au banc par Rhoban (mêmes réglages que le simulateur officiel, avec un
  peu de randomisation).
- Collisions : pour la vitesse sur GPU, les 269 maillages du corps sont remplacés par des boîtes mesurées sur ces
  maillages, qui touchent le mur, les prises, le sol et les autres membres. Les prises proches du membre en mouvement
  ont leurs maillages de collision complets.
- Écarts connus : pas de 4 ms au lieu de 2 ms, MuJoCo Warp au lieu de MuJoCo CPU. Une politique doit donc être
  revérifiée dans le simulateur officiel avant le vrai robot.

## Versions

mjlab 1.3 + MuJoCo 3.7 + Warp 1.12, imposés par BAM (`better-actuator-models[mjlab]`). Ce projet ne peut donc pas
importer `climbing_sim` (MuJoCo ≥ 3.11) : on relit ses fichiers de données. `wandb` est épinglé en 0.22.3 (les
versions récentes cassent rsl_rl 5.0.1).

README écrit avec l'aide de Claude.