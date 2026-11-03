from mjlab.tasks.registry import register_mjlab_task
from src.tasks.kick.rl import FlipRunner, KickRunner

from .env_cfgs import (
  unitree_go2_buck_env_cfg,
  unitree_go2_front_flip_env_cfg,
  unitree_go2_kick_env_cfg,
  unitree_go2_side_flip_env_cfg,
)
from .rl_cfg import (
  unitree_go2_buck_ppo_runner_cfg,
  unitree_go2_front_flip_ppo_runner_cfg,
  unitree_go2_kick_ppo_runner_cfg,
  unitree_go2_side_flip_ppo_runner_cfg,
)

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

# A full side roll over the left side: plain training (a mirrored sample would be a
# right flip), policy.onnx = the left flip, policy_right.onnx = it mirrored (FlipRunner).
register_mjlab_task(
  task_id="Unitree-Go2-SideFlip",
  env_cfg=unitree_go2_side_flip_env_cfg(),
  play_env_cfg=unitree_go2_side_flip_env_cfg(play=True),
  rl_cfg=unitree_go2_side_flip_ppo_runner_cfg(),
  runner_cls=FlipRunner,
)

# One turn nose-down: left-right symmetric, so mirrored training and a mirror-averaged
# policy.onnx like the kicks (KickRunner).
register_mjlab_task(
  task_id="Unitree-Go2-FrontFlip",
  env_cfg=unitree_go2_front_flip_env_cfg(),
  play_env_cfg=unitree_go2_front_flip_env_cfg(play=True),
  rl_cfg=unitree_go2_front_flip_ppo_runner_cfg(),
  runner_cls=KickRunner,
)

# The multistage recipe's first three stages (scripts/train_front_flip.py), training only:
# stage 4 is Unitree-Go2-FrontFlip. Same experiment folder and runner.
for _stage in (1, 2, 3):
  register_mjlab_task(
    task_id=f"Unitree-Go2-FrontFlip-S{_stage}",
    env_cfg=unitree_go2_front_flip_env_cfg(stage=_stage),
    play_env_cfg=unitree_go2_front_flip_env_cfg(play=True, stage=_stage),
    rl_cfg=unitree_go2_front_flip_ppo_runner_cfg(),
    runner_cls=KickRunner,
  )
