"""Standing starts for the DonkeyKick: anywhere on the tile, feet on the local ground.

The deploy triggers the kick from JointLock or from Velocity standing still, so each
start picks one of those stances. The base height comes from the stance (2-link leg
kinematics) and the highest terrain under the robot's footprint, so a robot can
start on a step or slope with its feet just above the ground.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from src.tasks.recovery.mdp.events import _footprint, _terrain_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _stance_table(asset: Entity, stances: tuple[dict[str, float], ...], device) -> torch.Tensor:
  """[num_stances, num_joints] joint positions; each stance maps joint regexes to values."""
  table = torch.zeros(len(stances), len(asset.joint_names), device=device)
  for s, stance in enumerate(stances):
    for j, name in enumerate(asset.joint_names):
      match = next((v for pattern, v in stance.items() if re.fullmatch(pattern, name)), None)
      if match is None:
        raise ValueError(f"Stance {s} has no value for joint '{name}'.")
      table[s, j] = match
  return table


def reset_standing_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stances: tuple[dict[str, float], ...],
  joint_noise: float,
  xy_range: float,
  footprint_radius: float,
  leg_lengths: tuple[float, float],
  foot_radius: float,
  drop_range: tuple[float, float],
  lin_vel: float,
  ang_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Standing robot at a random point within +-xy_range of the tile origin.

  Joints: one of `stances` (equal odds) plus uniform +-joint_noise, clamped to the
  soft limits. Base: level, random heading, its height set so the lowest foot is
  drop_range above the highest terrain under the footprint (centre + 8 points at
  footprint_radius). Base velocity: uniform +-lin_vel (m/s) and +-ang_vel (rad/s).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  env_ids = env_ids.long()
  asset: Entity = env.scene[asset_cfg.name]
  n, dev = len(env_ids), env.device

  table = _stance_table(asset, stances, dev)
  pick = torch.randint(len(stances), (n,), device=dev)
  joint_pos = table[pick] + sample_uniform(-joint_noise, joint_noise, (n, table.shape[1]), device=dev)
  limits = asset.data.soft_joint_pos_limits[env_ids]
  joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

  # Foot depth below the hip axis (~ the base origin) per leg; the longest leg sets
  # the base height.
  names = list(asset.joint_names)
  thigh = joint_pos[:, [i for i, nm in enumerate(names) if "thigh" in nm]]
  calf = joint_pos[:, [i for i, nm in enumerate(names) if "calf" in nm]]
  depth = leg_lengths[0] * torch.cos(thigh) + leg_lengths[1] * torch.cos(thigh + calf)
  stand_height = depth.amax(dim=1) + foot_radius

  origins = env.scene.env_origins
  xy_all = origins[:, :2].clone()
  xy_all[env_ids] += sample_uniform(-xy_range, xy_range, (n, 2), device=dev)
  points = xy_all.unsqueeze(1) + _footprint(footprint_radius, dev).unsqueeze(0)
  ground = _terrain_height(env, points, origins[:, 2] + 3.0)[env_ids]
  ground = torch.where(torch.isnan(ground), origins[env_ids, 2], ground)
  drop = sample_uniform(*drop_range, (n,), device=dev)
  pos = torch.cat([xy_all[env_ids], (ground + stand_height + drop).unsqueeze(1)], dim=1)

  yaw = sample_uniform(-math.pi, math.pi, (n,), device=dev)
  zero = torch.zeros_like(yaw)
  quat = quat_mul(asset.data.default_root_state[env_ids, 3:7], quat_from_euler_xyz(zero, zero, yaw))
  vel = torch.cat(
    [
      sample_uniform(-lin_vel, lin_vel, (n, 3), device=dev),
      sample_uniform(-ang_vel, ang_vel, (n, 3), device=dev),
    ],
    dim=1,
  )
  asset.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=1), env_ids=env_ids)
