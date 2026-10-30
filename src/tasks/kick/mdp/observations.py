from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def kick_phase(env: ManagerBasedRlEnv, duration: float) -> torch.Tensor:
  """Time since the kick was triggered as a fraction of duration, held at 1 after.

  Episodes start at the trigger, so this is the episode time. The deploy computes
  the same from its step counter (kick_phase in deploy/.../observations.h).
  """
  t = env.episode_length_buf.float() * env.step_dt
  return torch.clamp(t / duration, max=1.0).unsqueeze(1)
