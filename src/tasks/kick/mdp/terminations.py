from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import pitch_roll

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pitch_roll_out_of_range(
  env: ManagerBasedRlEnv,
  pitch_range: tuple[float, float],
  max_roll: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Pitch (rad, + = nose down) outside pitch_range or |roll| above max_roll.

  The handstand itself is ~45 deg nose-down, so a plain tilt limit can't tell it
  from a fall; the deploy's fall check (State_DonkeyKick) uses the same limits.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pitch, roll = pitch_roll(asset)
  return (pitch < pitch_range[0]) | (pitch > pitch_range[1]) | (torch.abs(roll) > max_roll)
