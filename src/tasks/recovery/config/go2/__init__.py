from mjlab.tasks.registry import register_mjlab_task
from src.tasks.recovery.rl import RecoveryRunner

from .env_cfgs import unitree_go2_recovery_env_cfg
from .rl_cfg import unitree_go2_recovery_ppo_runner_cfg

# Mirrored training as Rough-Blind (getting up should have no left/right
# preference either), with a plain policy.onnx for deploy (see RecoveryRunner).
register_mjlab_task(
  task_id="Unitree-Go2-Recovery",
  env_cfg=unitree_go2_recovery_env_cfg(),
  play_env_cfg=unitree_go2_recovery_env_cfg(play=True),
  rl_cfg=unitree_go2_recovery_ppo_runner_cfg(),
  runner_cls=RecoveryRunner,
)
