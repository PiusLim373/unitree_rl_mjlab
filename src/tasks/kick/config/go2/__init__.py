from mjlab.tasks.registry import register_mjlab_task
from src.tasks.kick.rl import KickRunner

from .env_cfgs import unitree_go2_buck_env_cfg, unitree_go2_kick_env_cfg
from .rl_cfg import unitree_go2_buck_ppo_runner_cfg, unitree_go2_kick_ppo_runner_cfg

# Mirrored training as Rough-Blind (a kick with both legs has no left/right
# preference), mirror-averaged policy.onnx for deploy (see KickRunner).
register_mjlab_task(
  task_id="Unitree-Go2-DonkeyKick",
  env_cfg=unitree_go2_kick_env_cfg(),
  play_env_cfg=unitree_go2_kick_env_cfg(play=True),
  rl_cfg=unitree_go2_kick_ppo_runner_cfg(),
  runner_cls=KickRunner,
)

# The fast power kick without the handstand: same runner (mirrored training,
# exploration switch at 3000, mirror-averaged policy.onnx).
register_mjlab_task(
  task_id="Unitree-Go2-QuickBuck",
  env_cfg=unitree_go2_buck_env_cfg(),
  play_env_cfg=unitree_go2_buck_env_cfg(play=True),
  rl_cfg=unitree_go2_buck_ppo_runner_cfg(),
  runner_cls=KickRunner,
)
