"""Évaluation chiffrée d'une politique LeftFoot :  uv run eval-policy [--checkpoint model_200.pt] [--num-envs 64]

Charge un checkpoint (par défaut le dernier model_<N>.pt du run local le plus récent), fait tourner la politique
sans bruit pendant quelques secondes, puis mesure où est la boule du pied gauche par rapport à sa coupe cible :
distance, et décalage selon les axes du mur (x : + = droite vu du grimpeur, z : + = haut, y : + = vers le mur).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path


def _latest_checkpoint(experiment: str) -> Path:
  """Le model_<N>.pt de plus grand N dans le run le plus récent de logs/rsl_rl/<experiment>/."""
  runs = sorted(p for p in (Path("logs") / "rsl_rl" / experiment).glob("*") if any(p.glob("model_*.pt")))
  if not runs:
    raise FileNotFoundError(f"aucun checkpoint dans logs/rsl_rl/{experiment}/")
  return max(runs[-1].glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Climb-LeftFoot-Microban")
  p.add_argument("--checkpoint", default=None, help="fichier .pt (défaut : le plus récent en local)")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--seconds", type=float, default=8.0)
  p.add_argument("--device", default=None, help="cuda:0 (défaut si disponible) ou cpu")
  args = p.parse_args()

  import torch

  import mjlab  # noqa: F401  (charge les tâches, dont les nôtres)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from climbing_rl.tasks import mdp
  from climbing_rl.tasks.climb_env_cfg import LEFT_FOOT
  from climbing_rl.wall import LIMBS

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = load_rl_cfg(args.task)
  ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(agent_cfg.experiment_name)
  print(f"\nCheckpoint : {ckpt}")

  # config "play" : pas de bruit d'observation ; on garde les terminaisons pour compter les chutes
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.seconds + 1.0  # pas de time_out pendant la mesure
  goal = cfg.rewards["foot_hooked"].params["sockets"]
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  base = env.unwrapped

  obs = env.get_observations()
  n_steps = int(args.seconds / base.step_dt)
  ended = torch.zeros(base.num_envs, dtype=torch.bool, device=device)  # épisode terminé (chute, main lâchée)
  first_hook = torch.full((base.num_envs,), float("nan"), device=device)  # instant du 1er accrochage (s)
  hooked_time = torch.zeros(base.num_envs, device=device)  # temps passé accroché sur la 2e moitié
  with torch.inference_mode():
    for k in range(n_steps):
      obs, _, dones, _ = env.step(policy(obs))
      ended |= dones.bool()
      d = mdp.ball_distances(base, goal)[:, LEFT_FOOT]
      hooked = (d < mdp.HOOK_TOL) & ~ended
      t = (k + 1) * base.step_dt
      first_hook = torch.where(hooked & first_hook.isnan(), torch.full_like(first_hook, t), first_hook)
      if k >= n_steps // 2:
        hooked_time += hooked.float() * base.step_dt

  ok = ~ended
  off = (mdp.ball_positions(base)[:, LEFT_FOOT] - torch.tensor(goal[LEFT_FOOT], device=device)) * 1000  # mm
  dist = off.norm(dim=-1)
  all_d = mdp.ball_distances(base, goal) * 1000
  print(f"{base.num_envs} robots, {args.seconds:.0f} s : {int(ended.sum())} épisodes terminés en cours de route "
        "(chute ou membre lâché), exclus des stats ci-dessous\n")
  if not ok.any():
    print("Aucun robot n'a tenu jusqu'au bout.")
    env.close()
    return

  off, dist, all_d = off[ok], dist[ok], all_d[ok]
  q = lambda x, f: torch.quantile(x, f).item()  # noqa: E731
  print(f"Pied gauche -> coupe cible (fin d'épisode, mm ; accroché = < {mdp.HOOK_TOL * 1000:.0f} mm, "
        "rayon de la boule 12,5 mm)")
  print(f"  distance   : moy {dist.mean():5.1f}  méd {q(dist, 0.5):5.1f}  p90 {q(dist, 0.9):5.1f}  "
        f"max {dist.max():5.1f}")
  for name, i, sens in (("x", 0, "+ = droite"), ("z", 2, "+ = haut"), ("y", 1, "+ = vers le mur")):
    c = off[:, i]
    print(f"  écart {name}    : moy {c.mean():+5.1f}  écart-type {c.std():4.1f}  min {c.min():+5.1f}  "
          f"max {c.max():+5.1f}   ({sens})")
  fh = first_hook[ok]
  print(f"  accroché à la fin : {(dist < mdp.HOOK_TOL * 1000).float().mean() * 100:.0f} % ; "
        f"temps accroché sur la 2e moitié : {hooked_time[ok].mean() / (args.seconds / 2) * 100:.0f} %")
  if (~fh.isnan()).any():
    print(f"  1er accrochage : médiane {fh.nanmedian():.2f} s ; jamais accroché : {int(fh.isnan().sum())} robots")
  print("\nAutres membres (distance à leur coupe de départ, mm) :")
  for i, limb in enumerate(LIMBS):
    if i != LEFT_FOOT:
      print(f"  {limb:<11} moy {all_d[:, i].mean():5.1f}  max {all_d[:, i].max():5.1f}")
  env.close()


if __name__ == "__main__":
  main()
