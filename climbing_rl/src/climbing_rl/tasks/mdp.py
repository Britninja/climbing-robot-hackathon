"""Fonctions de tâche (observations, récompenses, terminaisons) propres à l'escalade.

Chaque fonction reçoit `env` et rend un tenseur (num_envs, ...) : elles calculent pour tous les robots à la fois.
"Accroché" a le même sens que `ClimbingSim.hooked()` du simulateur : centre de la boule à moins de 6 mm du centre
de la coupe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from climbing_rl.robot import BALL_SITES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HOOK_TOL = 0.006
_ROBOT = SceneEntityCfg("robot")


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _ball_site_ids(env: ManagerBasedRlEnv) -> list[int]:
  """Indices (dans l'entité robot) des sites des 4 boules, ordre LIMBS. Calculés une fois puis gardés."""
  ids = getattr(env, "_climb_ball_site_ids", None)
  if ids is None:
    ids, _ = _robot(env).find_sites(BALL_SITES, preserve_order=True)
    env._climb_ball_site_ids = ids
  return ids


def _targets(env: ManagerBasedRlEnv, sockets: Sequence[Sequence[float]]) -> torch.Tensor:
  """Les positions des coupes visées (4, 3) en tenseur, sur le bon appareil."""
  return torch.tensor(sockets, dtype=torch.float32, device=env.device)


def ball_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N, 4, 3) positions monde des 4 boules."""
  return _robot(env).data.site_pos_w[:, _ball_site_ids(env)]


def ball_distances(env: ManagerBasedRlEnv, sockets) -> torch.Tensor:
  """(N, 4) distance de chaque boule à sa coupe cible."""
  return torch.norm(ball_positions(env) - _targets(env, sockets), dim=-1)


# ---- observations (pour le critic : vérité terrain, inconnue du vrai robot) --------------------------------
def ball_offsets(env: ManagerBasedRlEnv, sockets) -> torch.Tensor:
  """(N, 12) vecteur boule -> coupe cible pour chaque membre."""
  return (_targets(env, sockets) - ball_positions(env)).reshape(env.num_envs, -1)


def hooked_flags(env: ManagerBasedRlEnv, sockets, tol: float = HOOK_TOL) -> torch.Tensor:
  """(N, 4) 1 si la boule est dans sa coupe cible, 0 sinon."""
  return (ball_distances(env, sockets) < tol).float()


# ---- récompenses -----------------------------------------------------------------------------------------
def balls_near_sockets(env: ManagerBasedRlEnv, sockets, std: float = 0.01, limbs: Sequence[int] = (0, 1, 2, 3)):
  """Moyenne sur les membres choisis de exp(-d² / std²) : 1 quand la boule est au centre de sa coupe."""
  d = ball_distances(env, sockets)[:, list(limbs)]
  return torch.exp(-torch.square(d / std)).mean(dim=-1)


def balls_hooked(env: ManagerBasedRlEnv, sockets, tol: float = HOOK_TOL, limbs: Sequence[int] = (0, 1, 2, 3)):
  """Fraction des membres choisis accrochés à leur coupe cible."""
  return (ball_distances(env, sockets)[:, list(limbs)] < tol).float().mean(dim=-1)


def limb_reach(env: ManagerBasedRlEnv, sockets, limb: int, std: float = 0.03):
  """exp(-d² / std²) pour UN membre vers sa cible : guide le mouvement de loin (std plus grand)."""
  d = ball_distances(env, sockets)[:, limb]
  return torch.exp(-torch.square(d / std))


def limb_hooked(env: ManagerBasedRlEnv, sockets, limb: int, tol: float = HOOK_TOL):
  """1 si ce membre est accroché à sa cible."""
  return (ball_distances(env, sockets)[:, limb] < tol).float()


# ---- terminaisons ----------------------------------------------------------------------------------------
def trunk_dropped(env: ManagerBasedRlEnv, start_z: float, max_drop: float = 0.05) -> torch.Tensor:
  """Le tronc est descendu de plus de max_drop sous sa hauteur de départ : le robot a lâché."""
  return _robot(env).data.root_link_pos_w[:, 2] < start_z - max_drop


def limbs_lost(env: ManagerBasedRlEnv, sockets, limbs: Sequence[int], max_dist: float = 0.03) -> torch.Tensor:
  """Un des membres qui doivent rester accrochés s'est éloigné de plus de max_dist de sa coupe."""
  return (ball_distances(env, sockets)[:, list(limbs)] > max_dist).any(dim=-1)
