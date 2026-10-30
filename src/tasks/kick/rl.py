from src.tasks.velocity.rl import VelocitySymmetryRunner


class KickRunner(VelocitySymmetryRunner):
  """Mirrored training like Rough-Blind; policy.onnx is mirror-averaged (the default).

  Both rear legs kick together, so there is no left/right choice for averaging to
  cancel (unlike Recovery's roll): averaging makes the kick exactly symmetric.
  policy_raw.onnx is the plain policy.

  The entropy coefficient drops from 0.01 to 0.002 at iteration 3000, when
  penalty_ramp reaches full strength: Recovery v3 showed high exploration noise
  is needed early (finding the move) and gets in the way of precision later.
  """

  entropy_switch = (3000, 0.002)
