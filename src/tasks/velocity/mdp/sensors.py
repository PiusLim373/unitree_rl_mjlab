from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from mjlab.sensor import GridPatternCfg

__all__ = ["LiftedGridPatternCfg"]


@dataclass
class LiftedGridPatternCfg(GridPatternCfg):
  """Grid of parallel rays that start ``z_offset`` above the sensor frame.

  A downward ray that starts inside a geom reports that geom's bottom face, so
  with rays starting at base height, terrain taller than the base (e.g. stair
  boxes) read as the floor beneath it. Starting the rays higher lets them hit
  such terrain from above. ``height_scan`` still measures from the frame
  (base_link) position, so its value stays ``base_z - ground_z`` and simply goes
  negative where the terrain is above the base.
  """

  z_offset: float = 1.0
  """Height of the ray origins above the frame, in meters (along world z for
  ``ray_alignment`` "yaw" or "world")."""

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    local_offsets, local_directions = super().generate_rays(mj_model, device)
    local_offsets[:, 2] += self.z_offset
    return local_offsets, local_directions
