"""Rewards and the done check for getting up into the lying pose.

The goal is belly-down with legs folded in the entity's default joint pose (the
lying pose). Later goals only count once earlier ones are met: the pose only
pays when upright, and stillness only when upright and in the pose, so the
policy can't score by folding its legs while still on its back.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _tilt(asset: Entity) -> torch.Tensor:
  """Angle (rad) between the body's down axis and gravity: 0 belly-down, pi on its back."""
  cos_tilt = -asset.data.projected_gravity_b[:, 2]
  return torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))


def _upright_gate(asset: Entity, tilt_full: float, tilt_zero: float) -> torch.Tensor:
  """1 when tilt <= tilt_full, falling linearly to 0 at tilt_zero."""
  return torch.clamp((tilt_zero - _tilt(asset)) / (tilt_zero - tilt_full), 0.0, 1.0)


def _pose_score(asset: Entity, pose_var: float) -> torch.Tensor:
  err = asset.data.joint_pos - asset.data.default_joint_pos
  return torch.exp(-torch.mean(torch.square(err), dim=1) / pose_var)


def upright(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """(1 - g_z) / 2: 1 belly-down, 0.5 on its side, 0 on its back."""
  asset: Entity = env.scene[asset_cfg.name]
  return 0.5 * (1.0 - asset.data.projected_gravity_b[:, 2])


def lying_pose(
  env: ManagerBasedRlEnv,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joints close to the lying pose, counted only when upright."""
  asset: Entity = env.scene[asset_cfg.name]
  return _upright_gate(asset, tilt_full, tilt_zero) * _pose_score(asset, pose_var)


def lying_still(
  env: ManagerBasedRlEnv,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Little joint and body motion, counted only when upright and in the lying pose."""
  asset: Entity = env.scene[asset_cfg.name]
  calm = torch.exp(
    -torch.mean(torch.square(asset.data.joint_vel), dim=1)
    - torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  )
  return _upright_gate(asset, tilt_full, tilt_zero) * _pose_score(asset, pose_var) * calm


def body_impact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  force_scale: float,
) -> torch.Tensor:
  """Sum over body/leg geoms of contact force above force_threshold, in force_scale units.

  Uses the largest force over the step's physics substeps, so a brief slam counts.
  Resting on the ground stays under the threshold (whole robot ~150 N).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.norm(sensor.data.force_history, dim=-1).amax(dim=-1)  # [B, N]
  return torch.sum(torch.clamp(force - force_threshold, min=0.0), dim=1) / force_scale


def joint_vel_excess(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sum over joints of (|joint speed| - max_vel)^2, only above max_vel (rad/s).

  joint_vel_l2 weighs every bit of motion the same; this targets the fast peaks
  (v1: 95th percentile 18 rad/s while getting up) and leaves normal motion free.
  Only asset_cfg's joints (default: all), so motor groups can have their own limit.
  """
  asset: Entity = env.scene[asset_cfg.name]
  excess = torch.clamp(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - max_vel, min=0.0)
  return torch.sum(torch.square(excess), dim=1)


class torque_excess:
  """Sum over actuators of (|torque| / limit - fraction)^2, only above fraction.

  Penalizes running motors near their torque limit (v1 hit it in 9% of falls).
  Limits come from the model's actuator force ranges, in actuator order (which
  groups actuators by type, unlike the per-leg joint order).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    entity = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[entity]
    model = env.sim.mj_model
    # Compiled-model names carry the entity prefix ("robot/FL_hip_joint").
    ids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{entity}/{n}")
      for n in asset.actuator_names
    ]
    if min(ids) < 0:
      raise ValueError(f"Actuators not found in the model: {asset.actuator_names}")
    self.limits = torch.tensor(model.actuator_forcerange[ids, 1], device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    fraction: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    load = torch.abs(asset.data.actuator_force) / self.limits
    return torch.sum(torch.square(torch.clamp(load - fraction, min=0.0)), dim=1)


def _slot_columns(sensor: ContactSensor, geom_pattern: str) -> tuple[list[int], list[str]]:
  """Sensor columns (one slot per geom) whose geom name matches geom_pattern."""
  names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
  pattern = re.compile(geom_pattern)
  cols = [i for i, n in enumerate(names) if pattern.search(n.split("/")[-1])]
  if not cols:
    raise ValueError(f"No geom of sensor '{sensor.cfg.name}' matches '{geom_pattern}'.")
  return cols, [names[i] for i in cols]


class part_contact_force:
  """Sum over the matching geoms of terrain contact force above force_threshold,
  in force_scale units (largest force over the step's physics substeps).

  v2's slams came mostly from the roll itself (hips 49%, thighs 27%, head 13% of
  forces > 600 N), not from the belly landing; this targets a chosen body part.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    cols, _ = _slot_columns(sensor, cfg.params["geom_pattern"])
    self.cols = torch.tensor(cols, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    geom_pattern: str,
    force_threshold: float,
    force_scale: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force_history is not None
    force = torch.norm(sensor.data.force_history[:, self.cols], dim=-1).amax(dim=-1)
    return torch.sum(torch.clamp(force - force_threshold, min=0.0), dim=1) / force_scale


class leg_slap:
  """Squared speed (m/s) of a leg body at the moment one of its geoms starts
  touching the terrain (no contact the step before): legs slapping the ground.

  Uses the leg's own speed, not the body's: v2's hard leg contacts happened
  with the body barely moving down.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    cols, names = _slot_columns(sensor, cfg.params["geom_pattern"])
    entity = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[entity]
    model = env.sim.mj_model
    bodies = []
    for n in names:  # geom -> the body it belongs to, in the entity's body order
      # Sensor slots use bare names; the compiled model prefixes the entity.
      geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{entity}/{n.split('/')[-1]}")
      if geom_id < 0:
        raise ValueError(f"Geom '{n}' not found in the model.")
      body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id])
      bodies.append(list(asset.body_names).index(body.split("/")[-1]))
    self.cols = torch.tensor(cols, device=env.device)
    self.body_ids = torch.tensor(bodies, device=env.device)
    self.was_touching = torch.zeros((env.num_envs, len(cols)), dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.was_touching[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    geom_pattern: str,
    force_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force_history is not None
    asset: Entity = env.scene[asset_cfg.name]
    touching = torch.norm(sensor.data.force_history[:, self.cols], dim=-1).amax(dim=-1) > force_threshold
    started = touching & ~self.was_touching
    self.was_touching = touching
    speed_sq = torch.sum(torch.square(asset.data.body_link_lin_vel_w[:, self.body_ids]), dim=-1)
    return torch.sum(speed_sq * started.float(), dim=1)


def lying_still_check(
  env: ManagerBasedRlEnv,
  max_tilt: float,
  max_pose_err: float,
  max_joint_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """1 when the robot meets the deploy done check (State_Recovery), else 0.

  Tilt below max_tilt, every joint within max_pose_err of the lying pose and
  slower than max_joint_vel. The deploy also requires this for hold_time.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pose_err = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos).amax(dim=1)
  joint_vel = torch.abs(asset.data.joint_vel).amax(dim=1)
  done = (_tilt(asset) < max_tilt) & (pose_err < max_pose_err) & (joint_vel < max_joint_vel)
  return done.float()
