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
  """

  deploy_mirror_averaged = False
  entropy_switch = (3000, 0.002)
