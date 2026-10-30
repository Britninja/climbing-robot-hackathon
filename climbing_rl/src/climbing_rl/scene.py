"""Le mur, ses prises et le sol, ajoutés à la scène mjlab (copie allégée de simulator/climbing_sim/scene.py).

Tout est fixe (enfants directs du monde) : MuJoCo ne calcule pas de contacts entre objets fixes, seulement entre le
robot et eux. Chaque environnement mjlab est un monde séparé : le mur est au même endroit dans tous.
"""

from __future__ import annotations

from typing import Callable, Collection

import mujoco

from climbing_rl.wall import (
  HOLD_DIR,
  WALL_HEIGHT,
  WALL_THICKNESS,
  WALL_WIDTH,
  Wall,
  hold_mesh_files,
  hold_spheres,
)


def wall_spec_fn(
  wall_name: str = "N1",
  max_row: int = 32,
  hold_meshes: bool | Collection[tuple[int, int]] = False,
  hold_friction: float = 0.5,
  floor_friction: float = 1.0,
) -> Callable[[mujoco.MjSpec], None]:
  """Une fonction qui ajoute le mur à un MjSpec (pour `SceneCfg.spec_fn`).

  max_row : on ne crée que les prises jusqu'à cette rangée (les rangées sont espacées de 5 trous, soit 7,5 cm, sur
    N1 ; départ : pieds rangée 2, mains rangée 22). Chaque prise ajoute ~30 géométries : en simuler 70 pour rien
    ralentit tout.
  hold_meshes : ajoute aussi les 14 maillages convexes de la prise (plaque, goussets, peau sous la coupe qui
    empêche d'entrer par-dessous), sinon la tige de la boule les traverse. Plus fidèle, plus lent. True : toutes les
    prises ; une liste de (col, row) : seulement celles-là (les prises que le robot peut toucher pendant la tâche).
    Sans eux, la coupe est faite de sphères seulement.
  """
  wall = Wall.load(wall_name)
  holds = wall.holds_up_to_row(max_row)
  spheres = hold_spheres()
  meshes = hold_mesh_files() if hold_meshes else []
  mesh_holds = None if hold_meshes is True else set(hold_meshes or ())  # None : toutes
  visual = HOLD_DIR / "hold_visual.stl"

  def add_wall(spec: mujoco.MjSpec) -> None:
    w = spec.worldbody
    hold_fric = [hold_friction, 0.005, 0.0001]

    # sol (le robot démarre les pieds à ~3 cm du sol)
    w.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05], pos=[WALL_WIDTH / 2, -0.5, 0.0],
      rgba=[0.55, 0.55, 0.55, 1], friction=[floor_friction, 0.005, 0.0001],
    )
    # la planche : 800 x 1400 x 10 mm, face avant dans le plan y = 0
    w.add_geom(
      name="wall", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
      pos=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
      rgba=[0.86, 0.75, 0.56, 1], condim=3,
    )

    if visual.exists():
      spec.add_mesh(name="hold_visual", file=str(visual))
    for f in meshes:
      spec.add_mesh(name="hold_" + f.stem, file=str(f))

    for k, h in enumerate(holds):
      b = w.add_body(name=f"hold{k}_c{h.col}_r{h.row}", pos=h.socket.tolist())
      b.add_site(name=f"hold{k}_socket", size=[0.002, 0, 0], rgba=[0.1, 0.9, 0.2, 1], group=4)
      if visual.exists():
        b.add_geom(
          type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold_visual", contype=0, conaffinity=0, group=1,
          rgba=[0.85, 0.35, 0.15, 1],
        )
      for sp in spheres:  # la coupe et sa lèvre
        b.add_geom(
          type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[sp["r"], 0, 0], pos=sp["pos"], group=3,
          rgba=[1.0, 0.5, 0.0, 1], friction=hold_fric, condim=3,
        )
      for f in meshes if mesh_holds is None or (h.col, h.row) in mesh_holds else []:
        b.add_geom(
          type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold_" + f.stem, group=3,
          rgba=[0.25, 0.55, 0.95, 1], friction=hold_fric, condim=3,
        )

  return add_wall
