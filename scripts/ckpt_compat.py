"""Match a task's actor observation history to a checkpoint.

Go2 velocity actors see 47 values per step. Unitree-Go2-Rough-Blind now stacks a
5-step history (235 inputs), while older checkpoints (Flat, Rough-Blind v1-v4)
expect 47. Setting the actor history from the checkpoint lets any of them load in
either task. Only the actor's input layout changes; terrain, rewards and physics
stay as the task defines them.

Checkpoints whose input size isn't a multiple of 47 belong to other actor layouts
(e.g. Recovery: 42 x 5 = 210) and keep the task's own history; loading still
checks the sizes.
"""

from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnvCfg

ACTOR_OBS_PER_STEP = 47


def actor_history_from_checkpoint(path: str | Path) -> int | None:
  """Return how many steps of Go2 velocity actor observations the checkpoint was
  trained on, or None if its input size isn't a multiple of 47 (another layout)."""
  ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
  in_dim = ckpt["actor_state_dict"]["obs_normalizer._mean"].shape[-1]
  if in_dim % ACTOR_OBS_PER_STEP != 0:
    return None
  return in_dim // ACTOR_OBS_PER_STEP


def match_actor_history(env_cfg: ManagerBasedRlEnvCfg, path: str | Path) -> None:
  """Set the actor observation history of ``env_cfg`` to match the checkpoint."""
  history = actor_history_from_checkpoint(path)
  if history is not None:
    env_cfg.observations["actor"].history_length = history
