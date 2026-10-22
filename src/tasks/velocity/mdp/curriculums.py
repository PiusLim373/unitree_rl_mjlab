from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def vel_error_xy(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Per-step planar velocity tracking error (m/s), for terrain_levels_tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2], dim=1)


def command_speed_xy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Per-step commanded planar speed (m/s), for terrain_levels_tracking."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1)


def contact_fraction(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """Per-step share (0-1) of physics substeps in which any of the sensor's geoms
  presses on the ground harder than ``force_threshold``. Needs ``history_length``."""
  sensor: ContactSensor = env.scene[sensor_name]
  force_history = sensor.data.force_history  # [B, N, H, 3]
  assert force_history is not None, f"Sensor '{sensor_name}' needs history_length > 0."
  hit = (torch.norm(force_history, dim=-1) > force_threshold).any(dim=1)  # [B, H]
  return hit.float().mean(dim=-1)


def terrain_levels_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  promote_max_err: float,
  demote_min_err: float,
  min_cmd_speed: float,
  min_steps: int,
) -> torch.Tensor:
  """Move robots between terrain levels by how well they followed their commands.

  Uses the episode averages of the ``vel_err_xy`` and ``cmd_speed_xy`` metrics,
  which are still intact here because the curriculum runs before the metrics
  manager resets. Unlike terrain_levels_vel, this does not depend on how far the
  robot ended from its spawn point, so turning and slow commands are judged fairly.
  """
  terrain = env.scene.terrain
  assert terrain is not None

  metrics = env.metrics_manager
  steps = metrics._step_count[env_ids]
  safe_steps = steps.clamp(min=1).float()
  err = metrics._episode_sums["vel_err_xy"][env_ids] / safe_steps
  cmd_speed = metrics._episode_sums["cmd_speed_xy"][env_ids] / safe_steps
  fell = env.termination_manager.terminated[env_ids]

  started = steps > 0
  valid = steps >= min_steps
  # Promote robots that survived and tracked well while actually being asked to move.
  move_up = valid & ~fell & (err < promote_max_err) & (cmd_speed > min_cmd_speed)
  # Demote robots that fell, or tracked badly over a meaningful episode.
  move_down = started & (fell | (valid & (err > demote_min_err))) & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
