"""Le mur du hackathon et ses prises, lus depuis le simulateur officiel (dossier `simulator/` du dépôt).

On n'importe PAS le paquet `climbing_sim` : il exige mujoco >= 3.11, alors que mjlab 1.3 + BAM imposent mujoco 3.7.
On relit donc directement ses fichiers de données (murs JSON, géométrie des prises) et on recopie les quelques
formules de position (mêmes valeurs que `simulator/climbing_sim/wall.py`).

Repère monde (identique au simulateur) : origine au coin bas-gauche de la face avant du mur vu depuis le côté
grimpeur, x vers la droite, z vers le haut, y vers l'intérieur du mur. Le robot est du côté y < 0.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Où trouver les données du simulateur officiel ---------------------------------------------------------
# Par défaut : <racine du dépôt>/simulator/climbing_sim, ce fichier étant dans <racine>/climbing_rl/src/climbing_rl/.
# Sinon : variable d'environnement CLIMBING_SIM_DIR (le dossier qui contient walls/ et assets/).
_DEFAULT_SIM_DIR = Path(__file__).resolve().parents[3] / "simulator" / "climbing_sim"
SIM_DIR = Path(os.environ.get("CLIMBING_SIM_DIR", _DEFAULT_SIM_DIR))
WALLS_DIR = SIM_DIR / "walls"
HOLD_DIR = SIM_DIR / "assets" / "hold"
POSES_DIR = SIM_DIR / "assets" / "poses"


def check_sim_dir() -> None:
  """Erreur claire si les données du simulateur sont introuvables."""
  if not (WALLS_DIR.is_dir() and HOLD_DIR.is_dir() and POSES_DIR.is_dir()):
    raise FileNotFoundError(
      f"Données du simulateur officiel introuvables dans {SIM_DIR}.\n"
      "Ce projet doit être dans le dépôt de l'équipe, à côté du dossier simulator/ "
      "(ou définis CLIMBING_SIM_DIR=/chemin/vers/simulator/climbing_sim)."
    )


# --- Géométrie (mètres), recopiée de simulator/climbing_sim/wall.py -------------------------------------------
WALL_WIDTH = 0.800
WALL_HEIGHT = 1.400
WALL_THICKNESS = 0.010
HOLE_PITCH = 0.015
GRID_COLS, GRID_ROWS = 52, 93
GRID_X0, GRID_Z0 = 0.0175, 0.010  # centre du trou (0, 0) depuis le coin bas-gauche
SOCKET_OUT = 0.026  # centre de la coupe : sortie du mur
SOCKET_BELOW = 0.014  # centre de la coupe : sous l'axe des vis

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")


def hole_position(col: int, row: int) -> np.ndarray:
  """Position monde du trou (col, row) sur la face avant du mur."""
  if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
    raise ValueError(f"trou ({col}, {row}) hors de la grille {GRID_COLS} x {GRID_ROWS}")
  return np.array([GRID_X0 + col * HOLE_PITCH, 0.0, GRID_Z0 + row * HOLE_PITCH])


@dataclass(frozen=True)
class Hold:
  """Une prise vissée sur les trous (col, row) et (col + 2, row)."""

  col: int
  row: int

  @property
  def socket(self) -> np.ndarray:
    """Position monde du centre de la coupe (là où repose une boule accrochée)."""
    axis = (hole_position(self.col, self.row) + hole_position(self.col + 2, self.row)) / 2
    return axis + np.array([0.0, -SOCKET_OUT, -SOCKET_BELOW])


@dataclass
class Wall:
  name: str
  holds: list[Hold]
  start: dict[str, Hold]  # membre -> prise de départ

  @classmethod
  def load(cls, name: str) -> "Wall":
    """Un mur du simulateur ("N1", "N2", "N3") ou un chemin vers un fichier JSON."""
    check_sim_dir()
    path = Path(name)
    if not path.suffix:
      path = WALLS_DIR / f"{name}.json"
    d = json.loads(path.read_text())
    holds, seen = [], set()
    for h in d.get("holds", []):
      key = (int(h["col"]), int(h["row"]))
      if key not in seen:
        seen.add(key)
        holds.append(Hold(*key))
    start = {limb: Hold(int(h["col"]), int(h["row"])) for limb, h in d.get("start", {}).items()}
    for s in start.values():
      if s not in holds:
        holds.insert(0, s)
    return cls(d.get("name") or path.stem, holds, start)

  def start_sockets(self) -> np.ndarray:
    """(4, 3) positions des coupes de départ, dans l'ordre LIMBS."""
    return np.array([self.start[limb].socket for limb in LIMBS])

  def holds_up_to_row(self, max_row: int) -> list[Hold]:
    """Les prises jusqu'à une rangée donnée (pour n'en simuler que quelques-unes)."""
    return [h for h in self.holds if h.row <= max_row]


def hold_spheres() -> list[dict]:
  """Les sphères de collision de la coupe (repère de la prise, origine au centre de la coupe)."""
  check_sim_dir()
  return json.loads((HOLD_DIR / "cup_spheres.json").read_text())


def hold_mesh_files() -> list[Path]:
  """Les maillages convexes de collision de la prise (plaque, goussets, bloc arrière, peau sous la coupe)."""
  check_sim_dir()
  return sorted(HOLD_DIR.glob("hull_*.stl")) + sorted(HOLD_DIR.glob("skin_*.stl"))


def hang_joints() -> dict[str, float]:
  """Les angles de la posture accrochée du simulateur officiel (assets/poses/hang.json)."""
  check_sim_dir()
  return json.loads((POSES_DIR / "hang.json").read_text())["joints"]
