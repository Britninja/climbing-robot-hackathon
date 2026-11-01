"""Le Microban grimpeur pour mjlab : modèle MJCF, collisions, servos BAM, pose accrochée de départ."""

from __future__ import annotations

import re
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

#: Boîtes de collision du mode "light" (repère du corps, mm : centre, demi-tailles), mesurées sur les morceaux
#: convexes `coll_foot__configuration_right_*` et `coll_radius_*` du modèle. Sans elles, le pied et la tige de la
#: boule traversent le mur et les prises (seules les boules touchent).
_FOOT_BOXES = (
  ((0.0, -16.45, 42.0), (23.1, 7.15, 45.6)),  # semelle + goussets (morceaux 17, 20, 23)
  ((0.0, 0.15, 57.25), (22.8, 9.45, 4.05)),  # traverse (18)
  ((0.0, -4.15, -1.15), (10.3, 13.75, 2.45)),  # talon, côté cheville (19)
  ((0.0, 1.4, 77.2), (6.7, 10.7, 15.9)),  # tige vers la boule (22)
  ((0.0, 12.45, 101.15), (6.7, 9.05, 17.55)),  # crochet au-dessus de la boule (00 à 16, 21)
)
_RIGHT_FOREARM_BOXES = (
  ((0.0, -47.95, 12.1), (9.4, 35.65, 11.6)),  # corps de l'avant-bras (20)
  ((0.0, -26.4, 27.1), (9.5, 34.9, 3.4)),  # flanc côté servo (19)
  ((0.0, -17.45, -1.5), (9.5, 26.95, 2.0)),  # flanc opposé (21)
  ((14.1, -77.9, 13.5), (6.1, 5.5, 6.1)),  # tige de la boule (00 à 18)
)
#: L'avant-bras gauche (`radius_mirrored`) est le droit retourné : (x, y, z) -> (-x, -y, -z)
LIGHT_BOXES = {
  "foot_climb": _FOOT_BOXES,
  "foot_climb_2": _FOOT_BOXES,
  "radius_right": _RIGHT_FOREARM_BOXES,
  "radius_left": tuple((tuple(-c for c in pos), half) for pos, half in _RIGHT_FOREARM_BOXES),
}

#: Boîtes du reste du corps, mesurées au chargement : chaque boîte est l'enveloppe (dans le repère du corps) des
#: morceaux convexes de collision du modèle dont le nom de maillage correspond à l'expression régulière. Les deux
#: côtés utilisent les mêmes maillages placés différemment : on mesure chaque corps, pas de symétrie à deviner.
#: Un groupe (motif, zmin, zmax) ne garde que les sommets dans cette tranche (mm) et étend la boîte à toute la
#: tranche : le tronc est découpé ainsi, sinon les coins de sa boîte (vides en vrai) heurtent cuisses et avant-bras.
_TRUNK = r"chest|pelvis|tt_main"
_MESH_BOX_GROUPS = {
  # bassin (étroit en x) ; caisse ; haut (étroit en y) ; poignée. Les supports d'épaule sont dans la caisse.
  "trunk": ((_TRUNK, -10, 15), (_TRUNK, 15, 100), (_TRUNK, 100, 120), r"tt_handle"),
  "head": (r".",),
  "shoulder": (r".",),  # chape de l'épaule
  "humerus": (r".",),  # entre les deux servos (déjà en boîtes dans le modèle)
  "hip": (r"_(01|04|05|08|09|11)$", r"_(00|02)$", r"_(03|06|07|10)$"),  # plaque ; les deux pattes
  "hip_block": (r"lock|horn|cap|bts2",),  # équerre + palonnier (sans les entretoises, minuscules)
  "femur": (r".",),  # cuisse
  "tibia": (r".",),  # tibia (le servo du genou est déjà une boîte)
  "ankle_block": (r"lock.*_(0[1-9]|1[01])$", r"lock.*_00$|horn|cap|bts2"),  # équerre ; flanc + palonnier
}
_BODY_GROUP = {
  "trunk": "trunk", "head": "head",
  "shoulder": "shoulder", "shoulder_2": "shoulder",
  "humerus__configuration_right": "humerus", "humerus__configuration_left": "humerus",
  "hip": "hip", "hip_2": "hip",
  "hip_block__configuration_right": "hip_block", "hip_block__configuration_left": "hip_block",
  "femur__configuration_right": "femur", "femur__configuration_left": "femur",
  "tibia__configuration_right": "tibia", "tibia__configuration_left": "tibia",
  "ankle_block__configuration_left": "ankle_block", "ankle_block__configuration_right": "ankle_block",
}

#: Bits de collision du mode "light". Le monde (mur, prises, sol) a contype = conaffinity = 1. Toutes les
#: géométries de collision du robot ont contype 3, conaffinity 2 : bit 1 -> touchent le monde, bit 2 -> se touchent
#: entre elles (comme les boules et les maillages du modèle complet). MuJoCo ignore déjà les paires parent / enfant.
ROBOT_CONTYPE, ROBOT_CONAFFINITY = 3, 2
#: Paires de corps voisins (à deux articulations l'un de l'autre) dont les boîtes se chevauchent dans la pose
#: accrochée : le chevauchement vient de la simplification en boîtes, pas d'une vraie collision. On les ignore.
SELF_EXCLUDES = (
  ("hip", "femur__configuration_right"),  # patte de la hanche / cuisse, de part et d'autre du bloc de hanche
  ("hip_2", "femur__configuration_left"),
  ("tibia__configuration_right", "foot_climb"),  # tibia / pied, de part et d'autre du bloc de cheville
  ("tibia__configuration_left", "foot_climb_2"),
)


# --------------------------------------------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def mesh_boxes() -> dict[str, tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]]:
  """Les boîtes de _MESH_BOX_GROUPS mesurées sur les maillages de collision : {corps: ((centre, demi-tailles) mm)}."""
  model = mujoco.MjSpec.from_file(str(microban_model.mjcf(climb=True))).compile()
  out = {}
  for body_name, group in _BODY_GROUP.items():
    body = model.body(body_name).id
    pieces = {}  # nom du maillage -> sommets dans le repère du corps (mm)
    for g in range(model.ngeom):
      if (model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
          or not (model.geom_contype[g] or model.geom_conaffinity[g])):
        continue
      mid = model.geom_dataid[g]
      verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, model.geom_quat[g])
      pieces[model.mesh(mid).name] = (verts @ rot.reshape(3, 3).T + model.geom_pos[g]) * 1000
    boxes = []
    for pattern in _MESH_BOX_GROUPS[group]:
      pattern, zmin, zmax = pattern if isinstance(pattern, tuple) else (pattern, -np.inf, np.inf)
      pts = np.vstack([v for name, v in pieces.items() if re.search(pattern, name)])
      lo_z, hi_z = max(zmin, pts[:, 2].min()), min(zmax, pts[:, 2].max())
      pts = pts[(pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)]
      lo, hi = pts.min(0), pts.max(0)
      lo[2], hi[2] = lo_z, hi_z
      boxes.append((tuple(float(x) for x in (lo + hi) / 2), tuple(float(x) for x in (hi - lo) / 2)))
    out[body_name] = tuple(boxes)
  return out


def self_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str, float]]:
  """Les contacts robot / robot de `data` (après mj_forward ou mj_collision) : (géom 1, géom 2, distance en m)."""
  # le tronc s'appelle "trunk" seul, "robot/trunk" dans la scène mjlab
  trunk = next(b for b in range(model.nbody) if model.body(b).name in ("trunk", "robot/trunk"))
  on_robot = lambda g: model.body_rootid[model.geom_bodyid[g]] == trunk
  name = lambda g: model.geom(g).name or f"{model.body(model.geom_bodyid[g]).name}#{g}"
  return [
    (name(c.geom1), name(c.geom2), float(c.dist))
    for c in data.contact[:data.ncon] if on_robot(c.geom1) and on_robot(c.geom2)
  ]


def climber_spec(collisions: str = "light") -> mujoco.MjSpec:
  """Le MJCF du robot grimpeur (`microban_model/mjcf/robot_climb.xml`).

  collisions :
    "light" (par défaut, pour l'entraînement) : le robot est couvert de primitives (4 boules, 7 boîtes des servos,
      boîtes des pieds et avant-bras LIGHT_BOXES, boîtes mesurées sur les maillages du reste du corps) qui touchent
      le mur, les prises, le sol et les autres membres du robot. Les 269 maillages convexes du corps sont
      désactivés : en milliers d'exemplaires sur GPU, ils coûtent très cher.
    "full" : toutes les collisions du modèle (comme le simulateur officiel), pour vérifier une politique.
  """
  spec = mujoco.MjSpec.from_file(str(microban_model.mjcf(climb=True)))
  # Pose qpos0 (liaison libre) loin du mur et du sol. Elle n'est jamais simulée (mjlab pose le robot accroché à
  # chaque reset), mais MJWarp dimensionne/vérifie njmax et nconmax sur elle : tronc à l'origine, articulations à
  # zéro, le robot est dans le sol et le mur (~134 contacts, ~540 lignes) et il faudrait des tampons bien plus gros,
  # ce qui ralentit la simulation d'environ 15 %.
  spec.body("trunk").pos = [0.4, -1.0, 1.0]
  if collisions == "light":
    for g in spec.geoms:
      if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity):
        g.contype = 0
        g.conaffinity = 0
      elif g.contype or g.conaffinity:  # boules et boîtes des servos du modèle
        g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFFINITY
    for body_name, boxes in {**mesh_boxes(), **LIGHT_BOXES}.items():
      body = spec.body(body_name)
      for k, (pos, half) in enumerate(boxes):
        body.add_geom(
          name=f"{body_name}_box{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[p / 1000 for p in pos],
          size=[h / 1000 for h in half], contype=ROBOT_CONTYPE, conaffinity=ROBOT_CONAFFINITY, group=3,
          density=0.0, rgba=[1.0, 0.2, 0.6, 0.5],
        )
    for a, b in SELF_EXCLUDES:
      spec.add_exclude(bodyname1=a, bodyname2=b)
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
