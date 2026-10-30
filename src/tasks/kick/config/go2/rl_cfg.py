"""RL configuration for Unitree Go2 DonkeyKick and QuickBuck tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2_kick_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same network and PPO settings as Go2 velocity, own experiment folder.

  The entropy coefficient here (0.01) is the starting value: KickRunner drops it
  to 0.002 at iteration 3000 (see KickRunner.entropy_switch).
  """
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_kick"
  cfg.max_iterations = 6001
  return cfg


def unitree_go2_buck_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """As DonkeyKick (same network, PPO and exploration switch), own experiment folder."""
  cfg = unitree_go2_kick_ppo_runner_cfg()
  cfg.experiment_name = "go2_buck"
  return cfg
