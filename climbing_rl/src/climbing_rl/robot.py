"""Le Microban grimpeur pour mjlab : modèle MJCF, collisions, servos BAM, pose accrochée de départ."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import microban_model
import mujoco
import numpy as np
from bam.mjlab import BamActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from climbing_rl.wall import LIMBS, Wall, hang_joints

#: Les 19 articulations motorisées (ordre du modèle). La tête est un servo, mais la politique ne la commande pas.
JOINTS = tuple(microban_model.JOINTS)
#: Les sites au centre des 4 boules, dans l'ordre LIMBS (sans le préfixe "robot/" ajouté par mjlab)
BALL_SITES = tuple(f"{limb}_ball" for limb in LIMBS)
#: Les géométries de collision des 4 boules (sphères de rayon 12,5 mm)
BALL_GEOMS = tuple(f"{limb}_ball_coll" for limb in LIMBS)
#: Expression régulière des articulations commandées par la politique : toutes sauf la tête
ACTUATED = r"^(?!head$).*"


# --------------------------------------------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------------------------------------------
def climber_spec(collisions: str = "light") -> mujoco.MjSpec:
  """Le MJCF du robot grimpeur (`microban_model/mjcf/robot_climb.xml`).

  collisions :
    "light" (par défaut, pour l'entraînement) : seules les 4 boules et les 7 boîtes des servos touchent le mur et les
      prises. Les 269 maillages convexes du corps sont désactivés : en milliers d'exemplaires sur GPU, ils coûtent
      très cher, et ce sont les boules qui grimpent.
    "full" : toutes les collisions du modèle (comme le simulateur officiel), pour vérifier une politique.
  """
  spec = mujoco.MjSpec.from_file(str(microban_model.mjcf(climb=True)))
  if collisions == "light":
    for g in spec.geoms:
      if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity):
        g.contype = 0
        g.conaffinity = 0
  elif collisions != "full":
    raise ValueError("collisions : 'light' ou 'full'")
  return spec


# --------------------------------------------------------------------------------------------------------------
# Pose accrochée : même méthode que ClimbingSim.reset_hanging() du simulateur officiel
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HangPose:
  trunk_pos: tuple[float, float, float]
  trunk_quat: tuple[float, float, float, float]  # w, x, y, z
  joints: dict[str, float]
  ik_error: float  # plus grande distance boule / cible restante (m)


@lru_cache(maxsize=None)
def hang_pose(wall_name: str = "N1", above: float = 0.001) -> HangPose:
  """Le robot suspendu à ses 4 prises de départ.

  1. Posture `hang.json` du simulateur, tronc à l'origine.
  2. On cale le tronc (rotation + translation, méthode de Kabsch / SVD) pour que les 4 boules tombent au mieux sur
     les 4 coupes, `above` mètres au-dessus (pas d'interpénétration au départ).
  3. Une IK (moindres carrés amortis) ajuste les articulations pour placer chaque boule exactement.
  """
  wall = Wall.load(wall_name)
  model = climber_spec("full").compile()
  data = mujoco.MjData(model)
  qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
  vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
  low, high = np.array([model.jnt_range[model.joint(n).id] for n in JOINTS]).T
  sites = [model.site(s).id for s in BALL_SITES]

  # 1. posture de référence, tronc à l'origine
  joints = hang_joints()
  data.qpos[:] = 0.0
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  data.qpos[qadr] = [joints.get(n, 0.0) for n in JOINTS]
  mujoco.mj_kinematics(model, data)

  # 2. recalage rigide du tronc sur les coupes
  P = np.array([data.site_xpos[s] for s in sites])
  S = wall.start_sockets() + np.array([0.0, 0.0, above])
  cp, cs = P.mean(0), S.mean(0)
  U, _, Vt = np.linalg.svd((P - cp).T @ (S - cs))
  R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
  data.qpos[0:3] = cs - R @ cp
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, R.flatten())
  data.qpos[3:7] = quat

  # 3. IK sur les 19 articulations, tronc fixe
  goal = S.reshape(-1)
  jac = np.zeros((3, model.nv))
  for _ in range(200):
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    err = goal - np.concatenate([data.site_xpos[s] for s in sites])
    if np.abs(err).max() < 1e-5:
      break
    J = []
    for s in sites:
      mujoco.mj_jacSite(model, data, jac, None, s)
      J.append(jac[:, vadr].copy())
    J = np.vstack(J)
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(len(err)), err)
    data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, low, high)
  mujoco.mj_kinematics(model, data)
  err = goal - np.concatenate([data.site_xpos[s] for s in sites])
  ik_error = float(np.linalg.norm(err.reshape(-1, 3), axis=1).max())

  return HangPose(
    trunk_pos=tuple(float(x) for x in data.qpos[0:3]),
    trunk_quat=tuple(float(x) for x in data.qpos[3:7]),
    joints={n: float(data.qpos[a]) for n, a in zip(JOINTS, qadr)},
    ik_error=ik_error,
  )


# --------------------------------------------------------------------------------------------------------------
# Servos : BAM (modèle m6 du XL330 identifié au banc), mêmes réglages que le simulateur officiel
# --------------------------------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class XL330ActuatorCfg(BamActuatorCfg):
  """BamActuatorCfg + la limite de courant du firmware (que le simulateur officiel règle à 1,75 A)."""

  max_current: float | None = None

  def build(self, entity, target_ids, target_names):
    actuator = super().build(entity, target_ids, target_names)
    if self.max_current is not None:
      actuator._bam_model.actuator.max_current = self.max_current
    return actuator


def xl330_actuators(timestep: float) -> XL330ActuatorCfg:
  """Les 19 servos. Latence de commande 10 à 30 ms (10 ms mesurés sur le robot), en pas de physique."""
  steps = lambda seconds: max(0, int(round(seconds / timestep)))
  return XL330ActuatorCfg(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r".*",),
    kp_fw=125.0,  # gain recommandé pour le RL (celui des politiques apprises sur le vrai robot)
    vin_range=(7.2, 7.8),  # tension d'alimentation, tirée au hasard par environnement (simulateur : 7,5 V)
    vin_drop_resistance_range=(0.03, 0.045),  # chute de tension sous charge (simulateur : 0,0366 ohm)
    vin_min=6.0,
    max_current=1.75,
    friction_scale_range=(0.9, 1.1),
    delay_min_lag=steps(0.010),
    delay_max_lag=steps(0.030),
  )


def climber_cfg(wall_name: str = "N1", collisions: str = "light", timestep: float = 0.002) -> EntityCfg:
  """L'entité « robot » de la scène, initialisée suspendue aux prises de départ du mur."""
  pose = hang_pose(wall_name)
  if pose.ik_error > 0.002:
    print(f"[climbing_rl] attention : pose accrochée imprécise ({pose.ik_error * 1000:.1f} mm)")
  return EntityCfg(
    spec_fn=lambda: climber_spec(collisions),
    init_state=EntityCfg.InitialStateCfg(
      pos=pose.trunk_pos,
      rot=pose.trunk_quat,
      joint_pos=dict(pose.joints),
      joint_vel={r".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(xl330_actuators(timestep),),
      soft_joint_pos_limit_factor=0.95,
    ),
  )
