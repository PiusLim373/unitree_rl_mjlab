"""UprightWalk starts: upright on the rear feet, partly risen, or standing on four legs.

Upright: nose up, the joints at the default (upright) pose plus noise. On four legs: one
of `stand_stances` level, as the kicks start. Partly risen: the nose `mid_nose_deg` up and
the joints that far from the stance to the upright pose, so the rise can be found from
its end first. The lowest foot just above the highest ground under the footprint; random
heading, small random velocities.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from src.tasks.kick.mdp.events import _stance_table
from src.tasks.recovery.mdp.events import _footprint, _terrain_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _leg_angles(asset: Entity, joint_pos: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
  names = list(asset.joint_names)
  thigh = joint_pos[:, [i for i, n in enumerate(names) if n.startswith(prefix) and "thigh" in n]]
  calf = joint_pos[:, [i for i, n in enumerate(names) if n.startswith(prefix) and "calf" in n]]
  return thigh, calf


def reset_upright_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  upright_fraction: float,
  mid_fraction: float,
  mid_nose_deg: tuple[float, float],
  stand_stances: tuple[dict[str, float], ...],
  joint_noise: float,
  xy_range: float,
  footprint_radius: float,
  leg_lengths: tuple[float, float],
  foot_radius: float,
  hip_x: float,
  drop_range: tuple[float, float],
  lin_vel: float,
  ang_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """upright_fraction of the starts upright, mid_fraction partly risen, the rest on four
  legs (see the module)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  env_ids = env_ids.long()
  asset: Entity = env.scene[asset_cfg.name]
  n, dev = len(env_ids), env.device
  r = torch.rand(n, device=dev)
  upright = r < upright_fraction
  mid = ~upright & (r < upright_fraction + mid_fraction)
  # Nose angle: 90 deg upright, 0 on four legs.
  nose = torch.where(upright, torch.full_like(r, math.pi / 2), torch.zeros_like(r))
  nose = torch.where(mid, torch.deg2rad(sample_uniform(*mid_nose_deg, (n,), device=dev)), nose)

  table = _stance_table(asset, stand_stances, dev)
  stand = table[torch.randint(len(stand_stances), (n,), device=dev)]
  w = (nose / (math.pi / 2)).unsqueeze(1)
  joint_pos = stand + w * (asset.data.default_joint_pos[env_ids] - stand)
  joint_pos = joint_pos + sample_uniform(-joint_noise, joint_noise, joint_pos.shape, device=dev)
  limits = asset.data.soft_joint_pos_limits[env_ids]
  joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

  # Lowest foot's depth below the base origin, body pitched nose-up by `nose`: each foot
  # sits at (hip x - reach along x, -reach down) in the body's x-z plane (hip abduction
  # left out); upright the rear legs reach down, level all four as reset_standing_robot.
  l1, l2 = leg_lengths
  depth = []
  for prefix, x_hip in (("F", hip_x), ("R", -hip_x)):
    thigh, calf = _leg_angles(asset, joint_pos, prefix)
    x = x_hip - (l1 * torch.sin(thigh) + l2 * torch.sin(thigh + calf))
    z = -(l1 * torch.cos(thigh) + l2 * torch.cos(thigh + calf))
    depth.append(-(x * torch.sin(nose).unsqueeze(1) + z * torch.cos(nose).unsqueeze(1)))
  depth = torch.cat(depth, dim=1).amax(dim=1) + foot_radius

  origins = env.scene.env_origins
  xy_all = origins[:, :2].clone()
  xy_all[env_ids] += sample_uniform(-xy_range, xy_range, (n, 2), device=dev)
  points = xy_all.unsqueeze(1) + _footprint(footprint_radius, dev).unsqueeze(0)
  ground = _terrain_height(env, points, origins[:, 2] + 3.0)[env_ids]
  ground = torch.where(torch.isnan(ground), origins[env_ids, 2], ground)
  drop = sample_uniform(*drop_range, (n,), device=dev)
  pos = torch.cat([xy_all[env_ids], (ground + depth + drop).unsqueeze(1)], dim=1)

  yaw = sample_uniform(-math.pi, math.pi, (n,), device=dev)
  zero = torch.zeros_like(yaw)
  heading = quat_from_euler_xyz(zero, zero, yaw)
  quat = quat_mul(heading, quat_from_euler_xyz(zero, -nose, zero))
  vel = torch.cat(
    [sample_uniform(-lin_vel, lin_vel, (n, 3), device=dev), sample_uniform(-ang_vel, ang_vel, (n, 3), device=dev)],
    dim=1,
  )
  asset.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=1), env_ids=env_ids)
