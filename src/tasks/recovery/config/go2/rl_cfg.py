"""RL configuration for Unitree Go2 recovery task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same network and PPO settings as Go2 velocity, own experiment folder.

  The entropy coefficient here (0.01) is the starting value: RecoveryRunner
  drops it to 0.002 at iteration 3000 (see RecoveryRunner.entropy_switch).
  """
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_recovery"
  cfg.max_iterations = 6001
  return cfg
