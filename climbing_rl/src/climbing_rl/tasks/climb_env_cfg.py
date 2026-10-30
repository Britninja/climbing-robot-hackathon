"""Les tâches d'escalade : la configuration complète de l'environnement mjlab (les « fiches » des managers).

Deux tâches, de la plus simple à la plus dure :
  * Hang     : rester suspendu aux 4 prises de départ (vérifie que la physique tient, premier entraînement).
  * LeftFoot : sortir le pied gauche de sa prise et l'amener dans la prise juste au-dessus, les 3 autres membres
               restant accrochés (la première vraie sous-tâche d'escalade).
"""

from __future__ import annotations

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
from bam.mjlab import bam_init

from climbing_rl.robot import ACTUATED, climber_cfg, hang_pose
from climbing_rl.scene import wall_spec_fn
from climbing_rl.tasks import mdp
from climbing_rl.wall import LIMBS, Hold, Wall

WALL = "N1"
TIMESTEP = 0.004  # 2 ms, comme le simulateur officiel (contacts boule / coupe stables)
DECIMATION = 5  # 10 x 2 ms = une décision de la politique toutes les 20 ms, comme le vrai robot

LEFT_FOOT = LIMBS.index("left_foot")  # 2
OTHERS_THAN_LEFT_FOOT = tuple(i for i in range(4) if i != LEFT_FOOT)


def _base_cfg(play: bool, collisions: str = "light", max_row: int = 32) -> ManagerBasedRlEnvCfg:
  """Ce qui est commun à toutes les tâches : scène, actions, observations, événements, simulation."""
  wall = Wall.load(WALL)
  start_sockets = [s.tolist() for s in wall.start_sockets()]
  robot = SceneEntityCfg("robot", joint_names=(ACTUATED,))

  # ---------------------------------------------------------------- scène : le robot + le mur
  scene = SceneCfg(
    terrain=None,  # pas de terrain mjlab : toutes les origines à 0, le mur est au même endroit dans chaque monde
    num_envs=1,
    entities={"robot": climber_cfg(WALL, collisions=collisions, timestep=TIMESTEP)},
    spec_fn=wall_spec_fn(WALL, max_row=max_row, hold_meshes=(collisions == "full")),
  )

  # ---------------------------------------------------------------- actions : 18 positions cibles (pas la tête)
  # cible = pose accrochée + 0,5 x action (rad) : action = 0 -> le robot tient la pose de départ
  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(ACTUATED,), scale=0.5, use_default_offset=True,
    ),
  }

  # ---------------------------------------------------------------- observations
  # actor = ce que le vrai robot mesure (gyro, gravité dans le tronc, angles, vitesses), avec bruit et retards
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03), delay_min_lag=0, delay_max_lag=3,
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01), delay_min_lag=0, delay_max_lag=3,
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, params={"asset_cfg": robot}, noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, params={"asset_cfg": robot}, noise=Unoise(n_min=-0.25, n_max=0.25), delay_max_lag=1,
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  # critic = la même chose sans bruit + la vérité terrain (où sont les boules par rapport aux coupes)
  critic = {
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()},
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ball_offsets": ObservationTermCfg(func=mdp.ball_offsets, params={"sockets": start_sockets}),
    "hooked": ObservationTermCfg(func=mdp.hooked_flags, params={"sockets": start_sockets}),
  }
  observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic),
  }

  # ---------------------------------------------------------------- événements
  events = {
    # BAM écrit un frottement par environnement : ces champs du modèle doivent être dupliqués par monde
    "bam_init": EventTermCfg(func=bam_init, mode="startup"),
    # à chaque début d'épisode : retour à la pose accrochée (init_state du robot)
    "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
  }

  # ---------------------------------------------------------------- terminaisons communes
  start_z = hang_pose(WALL).trunk_pos[2]
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.trunk_dropped, params={"start_z": start_z, "max_drop": 0.05}),
    "nan": TerminationTermCfg(func=nan_detection),
  }

  # ---------------------------------------------------------------- pénalités communes
  rewards = {
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0, params={"asset_cfg": robot}),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
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
    episode_length_s=10.0,
  )


# ======================================================================================================== Hang
def hang_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rester suspendu : récompense = les 4 boules au fond de leur coupe."""
  cfg = _base_cfg(play)
  sockets = [s.tolist() for s in Wall.load(WALL).start_sockets()]
  cfg.rewards["near_sockets"] = RewardTermCfg(
    func=mdp.balls_near_sockets, weight=1.0, params={"sockets": sockets, "std": 0.01},
  )
  cfg.rewards["hooked"] = RewardTermCfg(func=mdp.balls_hooked, weight=1.0, params={"sockets": sockets})
  if play:
    cfg.episode_length_s = 1e9
  return cfg


# ==================================================================================================== LeftFoot
def left_foot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Monter le pied gauche d'une prise (5 rangées = 7,5 cm), les 3 autres membres restant accrochés."""
  cfg = _base_cfg(play)
  wall = Wall.load(WALL)
  start = wall.start_sockets()
  foot = wall.start["left_foot"]
  target_hold = Hold(foot.col-3, foot.row + 5)
  if target_hold not in wall.holds:
    raise ValueError(f"pas de prise {target_hold} sur le mur {WALL}")
  goal = start.copy()
  goal[LEFT_FOOT] = target_hold.socket  # cibles : 3 coupes de départ + la nouvelle coupe du pied gauche
  goal = [g.tolist() for g in goal]

  # le critic voit la nouvelle cible du pied
  cfg.observations["critic"].terms["ball_offsets"].params["sockets"] = goal
  cfg.observations["critic"].terms["hooked"].params["sockets"] = goal

  cfg.rewards["others_hooked"] = RewardTermCfg(
    func=mdp.balls_near_sockets, weight=1.0, params={"sockets": goal, "std": 0.01, "limbs": OTHERS_THAN_LEFT_FOOT},
  )
  cfg.rewards["foot_reach"] = RewardTermCfg(
    func=mdp.limb_reach, weight=2.0, params={"sockets": goal, "limb": LEFT_FOOT, "std": 0.04},
  )
  cfg.rewards["foot_hooked"] = RewardTermCfg(
    func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": LEFT_FOOT},
  )
  # échec : une main ou le pied droit a quitté sa prise
  cfg.terminations["limb_lost"] = TerminationTermCfg(
    func=mdp.limbs_lost, params={"sockets": goal, "limbs": OTHERS_THAN_LEFT_FOOT, "max_dist": 0.03},
  )
  cfg.episode_length_s = 8.0
  if play:
    cfg.episode_length_s = 1e9
  return cfg


# ===================================================================================================== PPO
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
