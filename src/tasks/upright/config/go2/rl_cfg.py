"""RL configuration for Unitree Go2 UprightWalk."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2_upright_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """As Rough-Blind (same network and PPO), own experiment folder."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_upright"
  return cfg
