import os

from mjlab.rl.exporter_utils import attach_metadata_to_onnx

from src.tasks.velocity.rl import VelocityOnPolicyRunner, VelocitySymmetryRunner
from src.tasks.velocity.rl.runner import _export_onnx
from src.tasks.velocity.rl.symmetry import MirroredPolicy


class KickRunner(VelocitySymmetryRunner):
  """Mirrored training like Rough-Blind; policy.onnx is mirror-averaged (the default).

  Both rear legs kick together, so there is no left/right choice for averaging to
  cancel (unlike Recovery's roll): averaging makes the kick exactly symmetric.
  policy_raw.onnx is the plain policy. The front flip (nose down, no side) runs on it too.

  The entropy coefficient drops from 0.01 to 0.002 at iteration 3000, when
  penalty_ramp reaches full strength: Recovery v3 showed high exploration noise
  is needed early (finding the move) and gets in the way of precision later.
  """

  entropy_switch = (3000, 0.002)


class FlipRunner(VelocityOnPolicyRunner):
  """Plain training for a move with a side (SideFlip, over the left side).

  No mirrored samples and no mirror-averaged export: a mirrored left flip is a right
  flip, which the left flip's rewards punish, and averaging the two cancels the roll.
  Exports policy.onnx (plain, the left flip) and policy_right.onnx (MirroredPolicy:
  the same policy with its inputs and actions mirrored, the right flip).

  Entropy switch as KickRunner.
  """

  entropy_switch = (3000, 0.002)

  def export_deploy_onnx(self, policy_path: str) -> None:
    os.makedirs(policy_path, exist_ok=True)
    metadata = self._onnx_metadata()
    exports = {
      "policy.onnx": (self.alg.get_policy().as_onnx(verbose=False), "plain"),
      "policy_right.onnx": (
        MirroredPolicy(self.alg.get_policy().as_onnx(verbose=False), self.env),
        "mirrored",
      ),
    }
    for filename, (model, inference) in exports.items():
      onnx_path = os.path.join(policy_path, filename)
      _export_onnx(model, onnx_path)
      attach_metadata_to_onnx(onnx_path, {**metadata, "inference": inference})
      self._upload(policy_path, filename)
