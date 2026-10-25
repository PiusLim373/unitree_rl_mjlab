from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.velocity.rl import VelocitySymmetryRunner


class RecoveryRunner(VelocitySymmetryRunner):
  """Mirrored training like Rough-Blind, but the deploy policy.onnx is the plain one.

  Averaging a policy with its mirror image cancels part of the roll-left/roll-right
  choice a robot on its back has to make: in the v1 eval the averaged policy
  recovered 95.9% of falls vs 97.2% plain. policy_mirror_avg.onnx is still saved.

  v3: the entropy coefficient (exploration bonus) drops from the configured 0.01
  to 0.002 at iteration 3000, when penalty_ramp reaches full strength. With 0.01
  throughout, the action noise stayed at ~0.7 and the robot never learned to lie
  still in pose (65% success on steps). With 0.002 from the start, the noise fell
  before it had learned to roll over from every fall (upright stuck at ~0.67).
  The switch is counted in env steps, so resumed runs (train.py restores the step
  counter) switch at the same point. Logged as Loss/entropy_coef.
  """

  deploy_mirror_averaged = False
  entropy_switch: tuple[int, float] | None = (3000, 0.002)  # (iteration, new coef)

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    if self.entropy_switch is None:
      return
    switch_it, late_coef = self.entropy_switch
    early_coef = self.alg.entropy_coef
    steps_per_it = self.cfg["num_steps_per_env"]
    update = self.alg.update

    def scheduled_update():
      # Called after the rollout of iteration `it`, which advanced the counter.
      it = self.env.unwrapped.common_step_counter // steps_per_it - 1
      coef = late_coef if it >= switch_it else early_coef
      if coef != self.alg.entropy_coef:
        print(f"[INFO] Iteration {it}: entropy coefficient {self.alg.entropy_coef} -> {coef}")
        self.alg.entropy_coef = coef
      if self.logger.writer is not None:
        self.logger.writer.add_scalar("Loss/entropy_coef", coef, it)
      return update()

    self.alg.update = scheduled_update
