from __future__ import annotations

# Import de toutes les fonctions de la bibliotheque mjlab necessaire
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  base_ang_vel,
  base_lin_vel,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  nan_detection,
  projected_gravity,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# BAM permet de simuler realistiquemet les servomoteurs, comme dans un robot reel
from bam.mjlab import bam_init

# Import de nos fonctions utilitaires, permettant la mise en place de la scène (généré par Claude)
from climbing_rl.robot import ACTUATED, climber_cfg, hang_pose
from climbing_rl.scene import wall_spec_fn
from climbing_rl.wall import LIMBS, Hold, Wall

# Bibliotheque de fonctions utilitaires, utilisés pour les rewards et les observations
from climbing_rl.tasks import mdp

# Constantes 
WALL = "N1"
TIMESTEP = 0.004  # La physique est mise a jour tout les 4ms
DECIMATION = 5  # 5 x 4 ms = une décision de la politique toutes les 20 ms, comme le vrai robot
LEFT_FOOT = LIMBS.index("left_foot")
RIGHT_FOOT = LIMBS.index("right_foot")
OTHERS_THAN_LEFT_FOOT = tuple(i for i in range(4) if i != LEFT_FOOT)
OTHERS_THAN_BOTH_FEET = tuple(i for i in range(4) if i != LEFT_FOOT and i != RIGHT_FOOT)

# Methode servant de base au taches (configuration de la scene, des actions, des observations pour le critic et l'actor, des events, des terminaisons et des rewards)
def _base_cfg(play: bool, collisions: str = "light", max_row: int = 32, mesh_holds: tuple[Hold, ...] = (),) -> ManagerBasedRlEnvCfg:

  # On récupere les prises de base du robot
  wall = Wall.load(WALL)
  start_sockets = [s.tolist() for s in wall.start_sockets()]
  robot = SceneEntityCfg("robot", joint_names=(ACTUATED,))

  # Configuration de la scène qui sera simulé parallelement
  scene = SceneCfg(
    terrain=None,  # Pas de terrain mjlab, tout les robot seront au même endroit (mais dans leur scene individuel)
    num_envs=1, # Nombre de scène simulé en meme temps, sera écrasé par l'option de lancement --env.scene.num-envs 8192
    entities={"robot": climber_cfg(WALL, collisions=collisions, timestep=TIMESTEP)}, # La scène contiendra une entité (le robot), donné par climber_cfg(...) dans robot.py
    spec_fn=wall_spec_fn(
      WALL, max_row=max_row, hold_meshes=collisions == "full" or [(h.col, h.row) for h in mesh_holds],
    ), # On ajoute le mur, les prises (avec maillage pour celle précisé dans mesh_holds) et le sol à la scène (donné par la fonction wall_spec_fn(...) de scene.py) 
  )

  # Creation des actions possible sur le robot, les 18 servomoteurs parametrables (sans la tete,  inutile)
  # cible = pose accrochée + 0,5 x action (rad) 
  # l'ordre à zero corresponds à la pose de départ
  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(ACTUATED,), scale=0.5, use_default_offset=True,
    ),
  }

  # Creation des observations, ce que l'actor et le critic peuvent voir, ce qu'il récupere comme info avant chaque pas
  # Deja, l'actor recoit comme info uniquement ce que le robot peut vrm observer (60 données en tout)
  # On rajoute du bruit/délai à ces données pour simuler les erreurs qui auront forcement lieu sur le vrai robot
  actor = {
    # Vitesse de rotation du tronc, gyroscope (3 dimension pour x y z)
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03), delay_min_lag=0, delay_max_lag=3,
    ),
    # Direction de la gravité en fonction du tronc (3 dimension pour x y z)
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01), delay_min_lag=0, delay_max_lag=3,
    ),
    # Angle de chaque servo, en fonction de celui de la pose de base (18 dimension pour 18 servo)
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, params={"asset_cfg": robot}, noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    # Vitesse de chaque servo (18 dimension pour 18 servo)
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, params={"asset_cfg": robot}, noise=Unoise(n_min=-0.25, n_max=0.25), delay_max_lag=1,
    ),
    # Les actions du pas d'avant (18 dimension pour 18 actions possibles)
    "actions": ObservationTermCfg(func=last_action),
  }
  # Le critic recupere les meme 60 donnée (non bruité) plus 19 autres que le robot ne connait pas, donnée par la simulation
  # Le critic peut tricher car il ne sert que pendant l'entrainement, pour juger les actions de l'actor (donc autant qu'il ai toutes les infos possibles)
  critic = {
    # Les données de l'actor (60 dimension)
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()},
    # La vitesse de deplacement du tronc dans l'espace (3 dimension pour x y z)
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    # Un vecteur boule vers prise cible pour chaque boule (dimension 12 pour les 3 x y z des 4 vecteurs) (les prises du départ uniquement dans cette tache de base)
    "ball_offsets": ObservationTermCfg(func=mdp.ball_offsets, params={"sockets": start_sockets}),
    # Si chaque boule est ancré dans la prise cible (4 dimensions pour 4 boules du robot)
    "hooked": ObservationTermCfg(func=mdp.hooked_flags, params={"sockets": start_sockets}),
  }
  observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=not play), # On garde le bruit uniquement pendant l'entrainement
    "critic": ObservationGroupCfg(critic),
  }

  # Creation des évenements qui prenerons place durant la vie d'une scène (épisodes)
  events = {
    # Creer une copie des frottements des servos par monde (BAM en écrit des différents pour chaque robot, aléatoire) (une seule fois, au démarrage de la tache)
    "bam_init": EventTermCfg(func=bam_init, mode="startup"),
    # A chaque début d'épisode, on remet le robor dans la pose de départ accrochée (init_state)
    "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
  }

  # Creation des terminaisons possible d'un épisode
  start_z = hang_pose(WALL).trunk_pos[2]
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True), # Le chrono est terminé ("time_out=True" car pour dire que la simu n'a pas échoué, juste time out)
    "fell": TerminationTermCfg(func=mdp.trunk_dropped, params={"start_z": start_z, "max_drop": 0.05}), # Le robot est tombé
    "nan": TerminationTermCfg(func=nan_detection), # Valeur de physique invalide (sécurité)
  }

  # Creation des reward commune a toutes les taches. Penalité uniquement dans cette tache de base, on punni :
  rewards = {
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01), # Les gros changement d'action d'un pas à l'autre => saccade
    "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0, params={"asset_cfg": robot}), # Articulations proche de leurs angle max 
  }

  # On termine par la creation de la classe correspondant a une tache, avec les classes crées et d'autre parametres :
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
      # Tampons par monde : en simulation on reste sous ~45 contacts / ~190 lignes de contrainte, même avec des
      # actions aléatoires fortes. Les agrandir sans besoin coûte ~15 % de vitesse.
      njmax=400,
      nconmax=128,
      mujoco=MujocoCfg(
        timestep=TIMESTEP,
        integrator="implicitfast",
        cone="pyramidal",
        iterations=20,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="trunk",
      distance=1.0,
      elevation=-10.0,
      azimuth=90.0,  # caméra côté grimpeur (y < 0), regardant le mur
    ),
    decimation=DECIMATION,
    episode_length_s=10.0, # Duree d'un épisode de 10sec par defaut
  )


# Tache basique hang, rester accroché (exemple donné par Claude)
def hang_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play) # On recupere la tache de base vu plus tot
  sockets = [s.tolist() for s in Wall.load(WALL).start_sockets()] # Tableau des positions des prises de base sur le mur du robot

  # On creer les rewards uniques à cette tache (attribué a chaque pas de l'épisode)
  # Schema classique contre le "sparse reward problem" : une recompense progressive (dense reward) pour guider l'entrainement puis une recompense fixe (sparse reward) pour récompenser le but 
  # Recompense progressive, plus une boules est proche de sa prise de base, elle rapporte de point (max 1); puis moyenne des 4 boules
  cfg.rewards["near_sockets"] = RewardTermCfg(
    func=mdp.balls_near_sockets, weight=1.0, params={"sockets": sockets, "std": 0.01},
  )
  # Recompense tout ou rien, chaque boule dans sa prise cible = reward de 1 (à moins de 6 mm), puis moyenne sur les 4, donc reward de 1 si les 4 sont accrochées
  cfg.rewards["hooked"] = RewardTermCfg(func=mdp.balls_hooked, weight=1.0, params={"sockets": sockets})
  if play:
    cfg.episode_length_s = 1e9 # Episode de durée infini lorsqu'on regarde le resultat de l'entrainement hors echec (toujours 10sec pour l'entrainement)
  return cfg


# Tache plus complexe, avec pour but de lever le pied gauche de la prise initiale vers la prise en haut a gauche, en gardant les 3 autre membre attaché 
# (exemple donné par Claude)
def left_foot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  wall = Wall.load(WALL)
  start = wall.start_sockets() # Tableau des position des prises de base du robot sur le mur

  # On recupere des variables de la class Hold, représentant une prise sur le mur
  foot = wall.start["left_foot"] # Prise de base du pied gauche 
  target_hold = Hold(foot.col-3, foot.row + 5) # Prise visée pour le pied gauche 
  # On verifie que la prise visé existe bien
  if target_hold not in wall.holds:
    raise ValueError(f"pas de prise {target_hold} sur le mur {WALL}")
  
  # Maillages de collision sur les prises des rangées 2 et 7 (là où passe le pied). Sans ca, la tige et le pied traversent le corps des prises (coute ~19 % de vitesse)
  cfg = _base_cfg(play, mesh_holds=tuple(h for h in wall.holds if h.row in (foot.row, target_hold.row)))

  # On creer un tableau avec les positions des prises cible pour le robot (3 prises de départ et la nouvelle prise du pied gauche)
  goal = start.copy()
  goal[LEFT_FOOT] = target_hold.socket
  goal = [g.tolist() for g in goal]

  # On modifie les observations exclusives au critic pour que les prises cibles soit celle qu'on vise dans cette tache
  cfg.observations["critic"].terms["ball_offsets"].params["sockets"] = goal
  cfg.observations["critic"].terms["hooked"].params["sockets"] = goal

  # Creation des reward exclusive à cette tache
  # Reward de maintient (contrainte douce), note de 0 à 1 en fonction de la distance d'une boule (autre que le pied gauche) avec sa prise de base (moyenne des 3)
  cfg.rewards["others_hooked"] = RewardTermCfg(
    func=mdp.balls_near_sockets, weight=1.0, params={"sockets": goal, "std": 0.01, "limbs": OTHERS_THAN_LEFT_FOOT},
  )
  # Reward progressive de guidage, note de 0 à 2 en fonction de la distance entre le pied gauche et sa prise cible
  cfg.rewards["foot_reach"] = RewardTermCfg(
    func=mdp.limb_reach, weight=2.0, params={"sockets": goal, "limb": LEFT_FOOT, "std": 0.04},
  )
  # Reward tout ou rien, 5 si la boule du pied gauche est attaché dans la prise cible (moins de 6 mm du centre de la prise)
  cfg.rewards["foot_hooked"] = RewardTermCfg(
    func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": LEFT_FOOT},
  )

  # Echec exclusif a cette tache (contrainte dure), si un des autres membre se détache de leurs prise (s’éloigne de plus de 3 cm de sa prise)
  cfg.terminations["limb_lost"] = TerminationTermCfg(
    func=mdp.limbs_lost, params={"sockets": goal, "limbs": OTHERS_THAN_LEFT_FOOT, "max_dist": 0.03},
  )
  cfg.episode_length_s = 8.0
  if play:
    cfg.episode_length_s = 1e9
  return cfg


# Tache plus complexe, avec pour but de lever le pied gauche de la prise initiale vers la prise en haut a gauche, en gardant les 3 autre membre attaché 
# (exemple donné par Claude)
def both_feet_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  wall = Wall.load(WALL)
  start = wall.start_sockets() # Tableau des position des prises de base du robot sur le mur

  # On recupere des variables de la class Hold, représentant une prise sur le mur
  left_foot = wall.start["left_foot"] # Prise de base du pied gauche
  right_foot = wall.start["right_foot"]
  left_target_hold = Hold(left_foot.col-3, left_foot.row + 5) # Prise visée pour le pied gauche 
  right_target_hold = Hold(right_foot.col, right_foot.row + 5) # Prise visée pour le pied gauche 
  # On verifie que la prise visé existe bien
  if left_target_hold not in wall.holds:
    raise ValueError(f"pas de prise {left_target_hold} sur le mur {WALL}")
  # On verifie que la prise visé existe bien
  if right_target_hold not in wall.holds:
    raise ValueError(f"pas de prise {right_target_hold} sur le mur {WALL}")
  
  # Maillages de collision sur les prises des rangées 2 et 7 (là où passe le pied). Sans ca, la tige et le pied traversent le corps des prises (coute ~19 % de vitesse)
  cfg = _base_cfg(play, mesh_holds=tuple(h for h in wall.holds if h.row in (left_foot.row, left_target_hold.row)))

  # On creer un tableau avec les positions des prises cible pour le robot (3 prises de départ et la nouvelle prise du pied gauche)
  goal = start.copy()
  goal[LEFT_FOOT] = left_target_hold.socket
  goal[RIGHT_FOOT] = right_target_hold.socket
  goal = [g.tolist() for g in goal]

  # On modifie les observations exclusives au critic pour que les prises cibles soit celle qu'on vise dans cette tache
  cfg.observations["critic"].terms["ball_offsets"].params["sockets"] = goal
  cfg.observations["critic"].terms["hooked"].params["sockets"] = goal

  # Creation des reward exclusive à cette tache
  # Reward de maintient (contrainte douce), note de 0 à 1 en fonction de la distance d'une boule (autre que le pied gauche) avec sa prise de base (moyenne des 3)
  cfg.rewards["others_hooked"] = RewardTermCfg(
    func=mdp.balls_near_sockets, weight=1.0, params={"sockets": goal, "std": 0.01, "limbs": OTHERS_THAN_BOTH_FEET},
  )
  # Reward progressive de guidage, note de 0 à 2 en fonction de la distance entre le pied gauche et sa prise cible
  cfg.rewards["left_foot_reach"] = RewardTermCfg(
    func=mdp.limb_reach, weight=2.0, params={"sockets": goal, "limb": LEFT_FOOT, "std": 0.04},
  )
  cfg.rewards["right_foot_reach"] = RewardTermCfg(
    func=mdp.limb_reach, weight=2.0, params={"sockets": goal, "limb": RIGHT_FOOT, "std": 0.04},
  )
  # Reward tout ou rien, 5 si la boule du pied gauche est attaché dans la prise cible (moins de 6 mm du centre de la prise)
  cfg.rewards["left_foot_hooked"] = RewardTermCfg(
    func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": LEFT_FOOT},
  )
  cfg.rewards["right_foot_hooked"] = RewardTermCfg(
    func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": RIGHT_FOOT},
  )

  # Echec exclusif a cette tache (contrainte dure), si un des autres membre se détache de leurs prise (s’éloigne de plus de 3 cm de sa prise)
  cfg.terminations["limb_lost"] = TerminationTermCfg(
    func=mdp.limbs_lost, params={"sockets": goal, "limbs": OTHERS_THAN_BOTH_FEET, "max_dist": 0.03},
  )
  cfg.episode_length_s = 8.0
  if play:
    cfg.episode_length_s = 1e9
  return cfg

# Pamametre de la PPO
def ppo_cfg(experiment: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
  """Les réglages de PPO (repris de mjlab_microban, la tâche de marche de Rhoban)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    wandb_project="climbing_rl",
    experiment_name=experiment,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
