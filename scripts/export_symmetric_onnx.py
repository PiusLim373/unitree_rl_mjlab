"""Script to re-export a trained checkpoint as deploy ONNX files.

For tasks whose runner exports mirror-averaged policies (VelocitySymmetryRunner,
e.g. Unitree-Go2-Rough-Blind), writes policy.onnx (mirror-averaged, what deploy/
loads) and policy_raw.onnx (plain) into the output folder. Use it for checkpoints
saved before that export existed (v8 and earlier); new runs save both files.
With --task Unitree-Go2-Recovery (RecoveryRunner), policy.onnx is the plain policy
and policy_mirror_avg.onnx the averaged one; use it to deploy a checkpoint other
than the run's last.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ckpt_compat import match_actor_history


@dataclass(frozen=True)
class ExportConfig:
  checkpoint: str
  """Checkpoint (model_N.pt) to export."""
  output_dir: str
  """Folder to write policy.onnx and policy_raw.onnx into (created if missing)."""
  task: str = "Unitree-Go2-Rough-Blind"
  device: str | None = None


def main():
  # Populate task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.velocity.rl import VelocitySymmetryRunner

  cfg = tyro.cli(ExportConfig, prog=sys.argv[0])
  ckpt = Path(cfg.checkpoint)
  if not ckpt.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
  runner_cls = load_runner_cls(cfg.task)
  if runner_cls is None or not issubclass(runner_cls, VelocitySymmetryRunner):
    raise ValueError(f"Task '{cfg.task}' does not export mirror-averaged policies.")
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # One robot is enough: the env only supplies observation layout, joint names
  # and metadata.
  env_cfg = load_env_cfg(cfg.task, play=True)
  match_actor_history(env_cfg, ckpt)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  runner.export_deploy_onnx(cfg.output_dir)
  env.close()
  if runner_cls.deploy_mirror_averaged:
    files = "policy.onnx (mirror-averaged) and policy_raw.onnx (plain)"
  else:
    files = "policy.onnx (plain) and policy_mirror_avg.onnx (mirror-averaged)"
  print(f"[INFO] Wrote {files} to {cfg.output_dir}")


if __name__ == "__main__":
  main()
