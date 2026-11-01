# climbing_rl — RL du Microban grimpeur avec mjlab (Léonard Bettcher)

Ce projet a été pour moi ma premièr expérience en Renforcement Learning.
J’ai utilisé une IA (Claude) pour assembler l’environnement mjlab à partir de trois dépôts existants (modèle du robot, modèle des servos BAM, simulateur officiel) qui ne fonctionnaient pas ensemble tels quels, puis pour apprendre comment fonctionnent PPO et la structure d’une tâche à partir d’un exemple.
J’ai mis en place l’environnement GPU sur mon ordinateur (rtx 5070) et testé la simulation, puis crée et entrainé mes propres taches avec les connaissances apprises pendant le hackaton.

Tâches [mjlab](https://github.com/mujocolab/mjlab) (RL massivement parallèle sur GPU) pour le Microban
grimpeur (mains en boule) sur le mur du hackathon Rock Climbing Robots.

## Installation

Ce dossier doit être dans le dépôt de l'équipe, à côté de `simulator/` (on y lit les murs et la géométrie des prises).

```bash
cd climbing_rl
uv sync --python 3.12
uv run check-scene            # le robot tient-il au mur sans politique ?
```

## Tâches

| Tâche | But |
|---|---|
| `Climb-Hang-Microban` | rester suspendu aux 4 prises de départ |
| `Climb-LeftFoot-Microban` | monter le pied gauche d'une prise, les 3 autres membres restant accrochés |

```bash
uv run play Climb-Hang-Microban --agent zero --viewer viser --num-envs 4
uv run train Climb-Hang-Microban --env.scene.num-envs 4096
uv run play Climb-Hang-Microban --wandb-run-path <entité>/climbing_rl/<id> --viewer viser
```

## Organisation

| Fichier | Rôle |
|---|---|
| `src/climbing_rl/wall.py` | lit le mur (JSON du simulateur), positions des coupes |
| `src/climbing_rl/robot.py` | MJCF grimpeur, collisions allégées, servos BAM, pose accrochée |
| `src/climbing_rl/scene.py` | ajoute le mur, les prises et le sol à la scène |
| `src/climbing_rl/tasks/mdp.py` | observations, récompenses, terminaisons propres à l'escalade |
| `src/climbing_rl/tasks/climb_env_cfg.py` | configuration complète des tâches + réglages PPO |
| `src/climbing_rl/tasks/__init__.py` | enregistrement des tâches |
| `src/climbing_rl/check_scene.py` | vérification de la scène sans entraînement |

Versions : mjlab 1.3 + MuJoCo 3.7 + Warp 1.12, imposés par BAM (`better-actuator-models[mjlab]`). Ce projet ne peut
donc pas importer `climbing_sim` (MuJoCo ≥ 3.11) : on relit ses fichiers de données.
