"""The heading frame of a robot that may stand on four legs or upright.

The body's left axis stays level in both postures (the body's x axis points at the sky
upright), so the heading frame comes from it: forward = left x up. cmd_vel, the tracking
rewards and the evaluation all use this frame.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply


def body_axis(asset: Entity, axis: int) -> torch.Tensor:
  """The body's x (0), y (1) or z (2) axis in the world frame."""
  v = torch.zeros_like(asset.data.root_link_pos_w)
  v[:, axis] = 1.0
  return quat_apply(asset.data.root_link_quat_w, v)


def heading_frame(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """(forward, left) unit vectors in the world frame, both level."""
  left = body_axis(asset, 1)
  left = left.clone()
  left[:, 2] = 0.0
  left = left / torch.clamp(torch.norm(left, dim=1, keepdim=True), min=1e-6)
  up = torch.zeros_like(left)
  up[:, 2] = 1.0
  return torch.cross(left, up, dim=1), left


def heading_velocity(asset: Entity) -> torch.Tensor:
  """(forward speed, left speed, yaw rate) in the heading frame, per env."""
  fwd, left = heading_frame(asset)
  v = asset.data.root_link_lin_vel_w
  w = asset.data.root_link_ang_vel_w
  return torch.stack([torch.sum(v * fwd, dim=1), torch.sum(v * left, dim=1), w[:, 2]], dim=1)


def nose_angle(asset: Entity) -> torch.Tensor:
  """The nose's angle up from the heading direction (rad): 0 on four legs, pi/2 upright,
  past pi/2 leaning back."""
  fwd, _ = heading_frame(asset)
  nose = body_axis(asset, 0)
  return torch.atan2(nose[:, 2], torch.sum(nose * fwd, dim=1))


def roll_angle(asset: Entity) -> torch.Tensor:
  """How far the left axis tips from level (rad, signed)."""
  return torch.asin(torch.clamp(body_axis(asset, 1)[:, 2], -1.0, 1.0))
