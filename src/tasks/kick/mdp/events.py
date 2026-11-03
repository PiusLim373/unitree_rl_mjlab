"""Standing starts for the DonkeyKick: anywhere on the tile, feet on the local ground.

The deploy triggers the kick from JointLock or from Velocity standing still, so each
start picks one of those stances. The base height comes from the stance (2-link leg
kinematics) and the highest terrain under the robot's footprint, so a robot can
start on a step or slope with its feet just above the ground.

reset_mid_flip (FrontFlip v5/v6, training only): some episodes start in the air partway
through the flip instead.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform

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


def _lowest_point(
  asset: Entity,
  joint_pos: torch.Tensor,
  angle: torch.Tensor,
  leg_lengths: tuple[float, float],
  foot_radius: float,
  hip_x: float,
  body_points: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Lowest point of the robot below its base origin (m, negative = below) once turned
  `angle` (rad, + = nose down) about the left axis: the base's body_points ((x, z, radius)
  in the body frame) and each leg's knee and foot (2-link legs in the sagittal plane, hips
  at +-hip_x; hip abduction ignored, which only overstates the reach)."""
  names = list(asset.joint_names)
  dev = joint_pos.device
  pts = [torch.tensor([x, z, r], device=dev).expand(len(angle), 3) for x, z, r in body_points]
  for prefix, x in (("F", hip_x), ("R", -hip_x)):
    for side in ("L", "R"):
      q1 = joint_pos[:, next(i for i, n in enumerate(names) if n.startswith(prefix + side) and "thigh" in n)]
      q2 = joint_pos[:, next(i for i, n in enumerate(names) if n.startswith(prefix + side) and "calf" in n)]
      kx, kz = x - leg_lengths[0] * torch.sin(q1), -leg_lengths[0] * torch.cos(q1)
      fx, fz = kx - leg_lengths[1] * torch.sin(q1 + q2), kz - leg_lengths[1] * torch.cos(q1 + q2)
      r = torch.full_like(kx, foot_radius)
      pts += [torch.stack([kx, kz, r], dim=1), torch.stack([fx, fz, r], dim=1)]
  p = torch.stack(pts, dim=1)  # [n, points, (x, z, radius)]
  s, c = torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)
  z = -p[..., 0] * s + p[..., 1] * c - p[..., 2]
  return z.amin(dim=1)


def reset_mid_flip(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fraction: float,
  turns: tuple[float, float],
  ang_vel: tuple[float, float],
  up_vel: tuple[float, float],
  fwd_vel: tuple[float, float],
  liftoff: dict[str, float],
  stances: tuple[dict[str, float], ...],
  joint_noise: float,
  clearance: float,
  leg_lengths: tuple[float, float],
  foot_radius: float,
  hip_x: float,
  body_points: tuple[tuple[float, float, float], ...],
  xy_range: float,
  footprint_radius: float,
  axis: int = 1,
  direction: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reference-state starts for the front flip: `fraction` of the resets begin in the air
  partway through it, so the end of the flip is practised before the launch can reach it.
  Runs after reset_base, which places the rest.

  The state is a ballistic flight from a liftoff (liftoff: "t" since the trigger (s), the
  rotation "turns" so far, the base "height" above the ground (m)) at a rate about the
  flip's body axis from ang_vel (rad/s) and an up speed from up_vel (m/s), caught when the
  rotation reaches a value from `turns`. No roll, random heading, forward at fwd_vel (m/s).
  Joints: one of `stances` (equal odds) +- joint_noise. The base is raised if any part
  (_lowest_point) would start less than `clearance` (m) above the ground. These episodes'
  clock starts at the flight's time (env.flip_time_offset, read by _time); the others' at 0.
  """
  if axis != 1:
    raise ValueError("reset_mid_flip's ground clearance assumes a turn about the left axis.")
  if not stances:
    raise ValueError("reset_mid_flip needs at least one stance (set per robot).")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  env_ids = env_ids.long()
  dev = env.device
  if not hasattr(env, "flip_time_offset"):
    env.flip_time_offset = torch.zeros(env.num_envs, device=dev)
  env.flip_time_offset[env_ids] = 0.0
  ids = env_ids[torch.rand(len(env_ids), device=dev) < fraction]
  n = len(ids)
  if n == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]

  table = _stance_table(asset, stances, dev)
  pick = torch.randint(len(stances), (n,), device=dev)
  joint_pos = table[pick] + sample_uniform(-joint_noise, joint_noise, (n, table.shape[1]), device=dev)
  limits = asset.data.soft_joint_pos_limits[ids]
  joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)

  g = 9.81
  rate = sample_uniform(*ang_vel, (n,), device=dev)
  theta = sample_uniform(*turns, (n,), device=dev) * 2 * math.pi
  tau = (theta - liftoff["turns"] * 2 * math.pi) / rate  # time since the liftoff
  vz = sample_uniform(*up_vel, (n,), device=dev)
  height = liftoff["height"] + vz * tau - 0.5 * g * tau**2
  angle = direction * theta
  low = _lowest_point(asset, joint_pos, angle, leg_lengths, foot_radius, hip_x, body_points)
  height = torch.maximum(height, clearance - low)
  env.flip_time_offset[ids] = liftoff["t"] + tau

  origins = env.scene.env_origins
  xy_all = origins[:, :2].clone()
  xy_all[ids] += sample_uniform(-xy_range, xy_range, (n, 2), device=dev)
  points = xy_all.unsqueeze(1) + _footprint(footprint_radius, dev).unsqueeze(0)
  ground = _terrain_height(env, points, origins[:, 2] + 3.0)[ids]
  ground = torch.where(torch.isnan(ground), origins[ids, 2], ground)
  pos = torch.cat([xy_all[ids], (ground + height).unsqueeze(1)], dim=1)

  yaw = sample_uniform(-math.pi, math.pi, (n,), device=dev)
  zero = torch.zeros_like(yaw)
  heading = quat_mul(asset.data.default_root_state[ids, 3:7], quat_from_euler_xyz(zero, zero, yaw))
  quat = quat_mul(heading, quat_from_euler_xyz(zero, angle, zero))
  unit = torch.zeros(n, 3, device=dev)
  unit[:, axis] = 1.0
  flip_axis = quat_apply(heading, unit)
  unit = torch.zeros(n, 3, device=dev)
  unit[:, 0] = 1.0
  forward = quat_apply(heading, unit)
  lin = forward * sample_uniform(*fwd_vel, (n,), device=dev).unsqueeze(1)
  lin[:, 2] = vz - g * tau
  ang = flip_axis * (direction * rate).unsqueeze(1)
  asset.write_root_state_to_sim(torch.cat([pos, quat, lin, ang], dim=1), env_ids=ids)
