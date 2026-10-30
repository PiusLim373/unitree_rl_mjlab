import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from .symmetry import SymmetricPolicy


def _export_onnx(model: torch.nn.Module, onnx_path: str) -> None:
  """torch.onnx.export with the settings of MjlabOnPolicyRunner.export_policy_to_onnx."""
  model.to("cpu")
  model.eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    onnx_path,
    export_params=True,
    opset_version=18,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    self.export_deploy_onnx(path.split("model")[0])

  def export_deploy_onnx(self, policy_path: str) -> None:
    """Write policy.onnx (what deploy/ loads) with the base metadata attached."""
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    attach_metadata_to_onnx(os.path.join(policy_path, filename), self._onnx_metadata())
    self._upload(policy_path, filename)

  def _uses_wandb(self) -> bool:
    # The logger only gets its type once training starts (not when re-exporting).
    return getattr(self.logger, "logger_type", None) == "wandb"

  def _onnx_metadata(self) -> dict:
    run_name: str = wandb.run.name if self._uses_wandb() and wandb.run else "local"  # type: ignore[assignment]
    return get_base_metadata(self.env.unwrapped, run_name)

  def _upload(self, policy_path: str, filename: str) -> None:
    if self._uses_wandb():
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class VelocitySymmetryRunner(VelocityOnPolicyRunner):
  """Trains on every PPO mini-batch plus its left-right mirror (rsl_rl symmetry).

  Mirrored samples take away any built-in left/right lean (v7 drifted right when
  told to go straight). Mirror rules: ``symmetry.mirror_quadruped``. With the
  mirror loss off, ``Loss/symmetry`` still logs how far the policy is from
  mirroring its own actions.

  Exports ``policy.onnx`` mirror-averaged (``symmetry.SymmetricPolicy``), which
  removes the small lean training leaves, and ``policy_raw.onnx`` plain. Python
  evals (``get_inference_policy``) use the plain policy. With
  ``deploy_mirror_averaged = False`` (subclasses), ``policy.onnx`` is the plain
  policy and ``policy_mirror_avg.onnx`` the averaged one.

  ``entropy_switch = (iteration, coef)`` (subclasses) replaces the configured
  entropy coefficient from that iteration on; see RecoveryRunner for why. It is
  counted in env steps, so resumed runs (train.py restores the step counter)
  switch at the same point, and logged as Loss/entropy_coef.
  """

  deploy_mirror_averaged: bool = True
  entropy_switch: tuple[int, float] | None = None  # (iteration, new coef)

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    train_cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": "src.tasks.velocity.rl.symmetry:mirror_quadruped",
    }
    # rsl_rl writes the env object into symmetry_cfg["_env"]. Pass copies so the
    # caller's dict (saved to agent.yaml by train.py) stays plain.
    run_cfg = dict(train_cfg)
    run_cfg["algorithm"] = dict(train_cfg["algorithm"])
    run_cfg["algorithm"]["symmetry_cfg"] = dict(train_cfg["algorithm"]["symmetry_cfg"])
    super().__init__(env, run_cfg, log_dir, device)
    if self.entropy_switch is not None:
      self._schedule_entropy(*self.entropy_switch)

  def _schedule_entropy(self, switch_it: int, late_coef: float) -> None:
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

  def export_deploy_onnx(self, policy_path: str) -> None:
    """Write policy.onnx (what deploy/ loads) plus the other variant.

    Mirror-averaged deploy: policy.onnx averaged, policy_raw.onnx plain.
    Plain deploy: policy.onnx plain, policy_mirror_avg.onnx averaged.
    Both take the same ``obs`` input and give the same ``actions`` output, so
    deploy.yaml and the C++ controller work with either.
    """
    os.makedirs(policy_path, exist_ok=True)
    metadata = self._onnx_metadata()
    averaged = (
      SymmetricPolicy(self.alg.get_policy().as_onnx(verbose=False), self.env),
      "mirror_averaged",
    )
    plain = (self.alg.get_policy().as_onnx(verbose=False), "plain")
    if self.deploy_mirror_averaged:
      exports = {"policy.onnx": averaged, "policy_raw.onnx": plain}
    else:
      exports = {"policy.onnx": plain, "policy_mirror_avg.onnx": averaged}
    for filename, (model, inference) in exports.items():
      onnx_path = os.path.join(policy_path, filename)
      _export_onnx(model, onnx_path)
      attach_metadata_to_onnx(onnx_path, {**metadata, "inference": inference})
      self._upload(policy_path, filename)
