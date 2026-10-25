"""Fallen-robot reset that spawns anywhere on the tile, above the local ground.

v1 dropped every robot over the tile's central platform (spawn height relative to
the tile origin), so few falls landed on stair steps. Here the ground height under
the robot's footprint is measured with downward rays (terrain only), so a robot can
start on any step or slope, and half the starts are "settled" (low, nearly still,
like lying in an odd pose after damping) rather than tumbling drops.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp
from mujoco_warp import rays

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_vec6 = wp.types.vector(length=6, dtype=float)

# Ray buffers per env (allocated on first use): footprint rays for every world.
_RAYS: dict[int, dict] = {}


def _footprint(radius: float, device) -> torch.Tensor:
  """Centre plus 8 points on a circle: [9, 2] xy offsets."""
  angles = torch.arange(8, device=device) * (2 * math.pi / 8)
  ring = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
  return torch.cat([torch.zeros(1, 2, device=device), ring])


def _terrain_height(env: ManagerBasedRlEnv, xy: torch.Tensor, z_top: torch.Tensor) -> torch.Tensor:
  """Highest terrain z under each world's points, from straight-down rays.

  xy: [num_envs, R, 2] world points, z_top: [num_envs] ray start height.
  Returns [num_envs] (NaN where every ray missed). Terrain only (geom groups 0, 1);
  robot geoms are groups 2/3.
  """
  num_envs, num_rays = xy.shape[0], xy.shape[1]
  buf = _RAYS.get(id(env))
  if buf is None or buf["pnt"].shape[1] != num_rays:
    dev = str(env.device)
    buf = {
      "pnt": wp.zeros((num_envs, num_rays), dtype=wp.vec3, device=dev),
      "vec": wp.zeros((num_envs, num_rays), dtype=wp.vec3, device=dev),
      "dist": wp.zeros((num_envs, num_rays), dtype=float, device=dev),
      "geomid": wp.zeros((num_envs, num_rays), dtype=int, device=dev),
      "normal": wp.zeros((num_envs, num_rays), dtype=wp.vec3, device=dev),
      "bodyexclude": wp.full(num_rays, -1, dtype=int, device=dev),  # pyright: ignore[reportArgumentType]
    }
    wp.to_torch(buf["vec"]).view(num_envs, num_rays, 3)[:] = torch.tensor([0.0, 0.0, -1.0])
    _RAYS[id(env)] = buf
  pnt = wp.to_torch(buf["pnt"]).view(num_envs, num_rays, 3)
  pnt[..., :2] = xy
  pnt[..., 2] = z_top.unsqueeze(1)
  torch.cuda.synchronize(env.device)  # torch writes done before the warp launch
  rays(
    m=env.sim.wp_model,
    d=env.sim.wp_data,
    pnt=buf["pnt"],
    vec=buf["vec"],
    geomgroup=_vec6(-1, -1, 0, 0, 0, 0),  # pyright: ignore[reportArgumentType]
    flg_static=True,
    bodyexclude=buf["bodyexclude"],
    dist=buf["dist"],
    geomid=buf["geomid"],
    normal=buf["normal"],
  )
  wp.synchronize_device(str(env.device))
  dist = wp.to_torch(buf["dist"])
  hit_z = torch.where(dist >= 0, z_top.unsqueeze(1) - dist, torch.full_like(dist, -torch.inf))
  ground = hit_z.amax(dim=1)
  return torch.where(torch.isfinite(ground), ground, torch.full_like(ground, torch.nan))


def reset_fallen_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  xy_range: float,
  footprint_radius: float,
  clearance: float,
  settled_fraction: float,
  settled_drop: tuple[float, float],
  tumbling_drop: tuple[float, float],
  settled_velocity: float,
  tumbling_velocity: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Spawn at a random point within +-xy_range of the tile origin, random orientation.

  The base starts clearance + drop above the highest terrain under the footprint
  (centre + 8 points at footprint_radius), so it never starts inside a step.
  A settled_fraction of robots get a small drop and near-zero velocity
  (|v| <= settled_velocity per axis); the rest drop further and tumble
  (tumbling_velocity: x, y, z, roll, pitch, yaw ranges, world frame).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  env_ids = env_ids.long()
  asset: Entity = env.scene[asset_cfg.name]
  n, dev = len(env_ids), env.device
  origins = env.scene.env_origins

  # Ground height under each respawning robot's footprint (all worlds are cast;
  # the others just look at their own tile origin).
  xy_all = origins[:, :2].clone()
  xy_all[env_ids] += sample_uniform(-xy_range, xy_range, (n, 2), device=dev)
  points = xy_all.unsqueeze(1) + _footprint(footprint_radius, dev).unsqueeze(0)
  ground = _terrain_height(env, points, origins[:, 2] + 3.0)[env_ids]
  ground = torch.where(torch.isnan(ground), origins[env_ids, 2], ground)

  settled = torch.rand(n, device=dev) < settled_fraction
  drop = torch.where(
    settled,
    sample_uniform(*settled_drop, (n,), device=dev),
    sample_uniform(*tumbling_drop, (n,), device=dev),
  )
  pos = torch.cat([xy_all[env_ids], (ground + clearance + drop).unsqueeze(1)], dim=1)

  euler = sample_uniform(
    torch.tensor([-math.pi, -math.pi / 2, -math.pi], device=dev),
    torch.tensor([math.pi, math.pi / 2, math.pi], device=dev),
    (n, 3),
    device=dev,
  )
  default_quat = asset.data.default_root_state[env_ids, 3:7]
  quat = quat_mul(default_quat, quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))

  keys = ["x", "y", "z", "roll", "pitch", "yaw"]
  lo = torch.tensor([tumbling_velocity.get(k, (0.0, 0.0))[0] for k in keys], device=dev)
  hi = torch.tensor([tumbling_velocity.get(k, (0.0, 0.0))[1] for k in keys], device=dev)
  vel = torch.where(
    settled.unsqueeze(1),
    sample_uniform(-settled_velocity, settled_velocity, (n, 6), device=dev),
    sample_uniform(lo, hi, (n, 6), device=dev),
  )
  asset.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=1), env_ids=env_ids)
