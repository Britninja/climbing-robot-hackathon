"""Enregistre les tâches auprès de mjlab (chargé automatiquement grâce au point d'entrée `mjlab.tasks`)."""

from mjlab.tasks.registry import register_mjlab_task

from climbing_rl.tasks.climb_env_cfg import hang_env_cfg, left_foot_env_cfg, ppo_cfg

register_mjlab_task(
  task_id="Climb-Hang-Microban",
  env_cfg=hang_env_cfg(),
  play_env_cfg=hang_env_cfg(play=True),
  rl_cfg=ppo_cfg("climb_hang", max_iterations=1500),
)

register_mjlab_task(
  task_id="Climb-LeftFoot-Microban",
  env_cfg=left_foot_env_cfg(),
  play_env_cfg=left_foot_env_cfg(play=True),
  rl_cfg=ppo_cfg("climb_left_foot", max_iterations=5000),
)
