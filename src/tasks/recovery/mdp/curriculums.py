from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def penalty_ramp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_names: list[str],
  start_scale: float,
  ramp_steps: int,
) -> dict[str, torch.Tensor]:
  """Scale penalty weights from start_scale to 1 linearly over ramp_steps env steps.

  Full-strength penalties from the start make the "lazy" answer (lie still where
  it landed) pay best, so the policy never learns to roll over. Ramping them in
  rewards the effort first and makes it gentle later. The configured weights
  are the final ones; they are read once on the first call.
  """
  del env_ids  # Unused.
  if not hasattr(env, "_penalty_base_weights"):
    env._penalty_base_weights = {  # type: ignore[attr-defined]
      name: env.reward_manager.get_term_cfg(name).weight for name in term_names
    }
  progress = min(env.common_step_counter / ramp_steps, 1.0)
  scale = start_scale + (1.0 - start_scale) * progress
  for name, weight in env._penalty_base_weights.items():  # type: ignore[attr-defined]
    env.reward_manager.get_term_cfg(name).weight = weight * scale
  return {"penalty_scale": torch.tensor(scale)}
