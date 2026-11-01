"""Vérification de la scène, sans entraînement :  uv run check-scene [--num-envs 64] [--seconds 3] [--device cuda:0]

1. la pose accrochée calculée (erreur de l'IK) ;
2. la scène compilée (nombre de géométries, de prises) et les auto-contacts (robot / robot) dans la pose accrochée :
   il ne doit y en avoir aucun (les paires voisines qui se chevauchent sont exclues, voir robot.SELF_EXCLUDES) ;
3. le robot avec action nulle (les servos tiennent la pose de départ) pendant quelques secondes : les boules
   restent-elles dans les coupes ? le tronc descend-il ? des membres se touchent-ils ? combien de pas par seconde ?
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Climb-Hang-Microban")
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--seconds", type=float, default=3.0)
  p.add_argument("--device", default=None, help="cuda:0 (défaut si disponible) ou cpu")
  args = p.parse_args()

  import mujoco
  import numpy as np
  import torch

  import mjlab  # noqa: F401  (charge les tâches, dont les nôtres)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from climbing_rl.robot import hang_pose, self_contacts
  from climbing_rl.tasks import mdp
  from climbing_rl.tasks.climb_env_cfg import WALL
  from climbing_rl.wall import LIMBS, Wall

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # 1. pose
  pose = hang_pose(WALL)
  print(f"\n[1] Pose accrochée sur {WALL} : tronc à {[round(x, 4) for x in pose.trunk_pos]} m, "
        f"erreur IK {pose.ik_error * 1000:.2f} mm")

  # 2. scène
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  t0 = time.perf_counter()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  m = env.sim.mj_model
  n_coll = int(((m.geom_contype != 0) | (m.geom_conaffinity != 0)).sum())
  print(f"[2] Scène compilée en {time.perf_counter() - t0:.1f} s sur {device} : {m.ngeom} géométries dont "
        f"{n_coll} en collision, {m.nbody} corps, {args.num_envs} environnements")

  # auto-contacts dans la pose de départ (monde 0, recalculé sur CPU)
  env.reset()
  d = mujoco.MjData(m)
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  mujoco.mj_forward(m, d)
  start_self = self_contacts(m, d)
  print(f"    auto-contacts dans la pose accrochée : {len(start_self)}")
  for g1, g2, dist in start_self:
    print(f"      {g1} <-> {g2} : pénétration {-dist * 1000:.1f} mm")
  trunk = m.body("robot/trunk").id
  robot_geom = torch.tensor(m.body_rootid[m.geom_bodyid] == trunk, device=device)

  # 3. action nulle
  sockets = [s.tolist() for s in Wall.load(WALL).start_sockets()]
  z0 = env.scene["robot"].data.root_link_pos_w[:, 2].clone()
  zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
  n_steps = int(args.seconds / env.step_dt)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    env.step(zero)
  dt = time.perf_counter() - t0
  wd = env.sim.wp_data
  nacon = int(wd.nacon.numpy()[0])
  geom = torch.as_tensor(wd.contact.geom.numpy()[:nacon], device=device).long()
  world = torch.as_tensor(wd.contact.worldid.numpy()[:nacon], device=device).long()
  is_self = robot_geom[geom[:, 0]] & robot_geom[geom[:, 1]]
  self_per_world = torch.bincount(world[is_self], minlength=env.num_envs)
  nefc = np.asarray(wd.nefc.numpy())
  d = mdp.ball_distances(env, sockets) * 1000  # mm
  hooked = (d < mdp.HOOK_TOL * 1000).float().mean(0)
  drop = (z0 - env.scene["robot"].data.root_link_pos_w[:, 2]) * 1000
  print(f"[3] {args.seconds:.0f} s simulées ({n_steps} pas de politique) en {dt:.1f} s "
        f"-> {n_steps * env.num_envs / dt:.0f} pas d'environnement / s")
  for i, limb in enumerate(LIMBS):
    print(f"    {limb:<11} distance à la coupe : moy {d[:, i].mean():5.1f} mm, max {d[:, i].max():5.1f} mm, "
          f"accroché dans {hooked[i] * 100:3.0f} % des envs")
  print(f"    tronc : descendu de {drop.mean():.1f} mm en moyenne (max {drop.max():.1f} mm)")
  print(f"    auto-contacts à la fin : {int(is_self.sum())} au total, max {int(self_per_world.max())} par env ; "
        f"lignes de contrainte max {int(nefc.max())} / njmax {cfg.sim.njmax}")
  ok = bool((hooked == 1).all() and drop.max() < 20 and not start_self)
  print("\nOK : le robot tient au mur." if ok else "\nATTENTION : le robot ne tient pas bien, voir ci-dessus.")
  env.close()


if __name__ == "__main__":
  main()
