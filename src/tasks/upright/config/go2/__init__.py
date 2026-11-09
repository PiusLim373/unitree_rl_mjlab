from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocitySymmetryRunner

from .env_cfgs import unitree_go2_upright_env_cfg
from .rl_cfg import unitree_go2_upright_ppo_runner_cfg

# Rear-leg walking and the handshake, left-right symmetric: mirrored training like
# Rough-Blind, mirror-averaged policy.onnx. Unitree-Go2-UprightWalk is stage 2 (the full
# task); -S1 is the multistage recipe's first stage (scripts/train_upright.py).
register_mjlab_task(
  task_id="Unitree-Go2-UprightWalk",
  env_cfg=unitree_go2_upright_env_cfg(),
  play_env_cfg=unitree_go2_upright_env_cfg(play=True),
  rl_cfg=unitree_go2_upright_ppo_runner_cfg(),
  runner_cls=VelocitySymmetryRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2-UprightWalk-S1",
  env_cfg=unitree_go2_upright_env_cfg(stage=1),
  play_env_cfg=unitree_go2_upright_env_cfg(play=True, stage=1),
  rl_cfg=unitree_go2_upright_ppo_runner_cfg(),
  runner_cls=VelocitySymmetryRunner,
)
