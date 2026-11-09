"""UprightWalk observations, rewards and the fall check.

Most rewards are gated by the up command (commands.py): `when` = "up" pays only while up
is 1 (stand upright), "down" only while it is 0 (on four legs), None always.

While walking (cmd_vel above STAND_THRESHOLD) the rear feet step in turn on the gait clock,
and both in the air costs (v1 hopped). Both only count once risen: paid in any posture,
stepping on four legs or on the knees earns the gait without ever getting up. At zero
cmd_vel it stands: both feet planted, each step costs, so it steps only to keep its
balance (the feet are points, so small wobbles are caught by swaying and bigger ones by a
step). The gait clock (`phase`) reads 0 while standing, as the velocity task's.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.kick.mdp.rewards import _in_contact
from src.tasks.velocity.mdp.rewards import variable_posture

from .commands import STAND_THRESHOLD
from .frames import body_axis, heading_frame, heading_velocity, nose_angle, roll_angle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _term(env: ManagerBasedRlEnv, command_name: str):
  return env.command_manager.get_term(command_name)


def _standing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """1 while cmd_vel is (near) zero."""
  cmd = env.command_manager.get_command(command_name)
  return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) <= STAND_THRESHOLD).float()


def _gate(env: ManagerBasedRlEnv, command_name: str, when: str | None) -> torch.Tensor:
  up = _term(env, command_name).up
  if when == "up":
    return up
  if when == "down":
    return 1.0 - up
  return torch.ones_like(up)


##
# Observations
##


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """(sin, cos) of the gait clock while walking, (0, 0) while standing."""
  t = (env.episode_length_buf * env.step_dt) % period / period
  clock = torch.stack([torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=1)
  return clock * (1.0 - _standing(env, command_name)).unsqueeze(1)


def upright_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """1 = stand upright, 0 = down on four legs."""
  return _term(env, command_name).up.unsqueeze(1)


def hand_flag(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(left, right): 1 for the paw that is out."""
  side = _term(env, command_name).hand_side
  return torch.stack([(side == 1).float(), (side == 2).float()], dim=1)


def tilted_gravity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Gravity in the body frame as this robot's slightly misaligned IMU reads it."""
  return _term(env, command_name).tilted_gravity()


def slow_history(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The last second in coarse bins (commands.py): what shows slow drift."""
  return _term(env, command_name).slow_history()


def _paw_offset(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Since the paw went out: (forward, left) moved in the heading frame (m), turn (rad)."""
  term = _term(env, command_name)
  asset: Entity = env.scene["robot"]
  fwd, left = heading_frame(asset)
  moved = asset.data.root_link_pos_w[:, :2] - term.hand_xy
  turn = torch.atan2(term.hand_fwd[:, 0] * fwd[:, 1] - term.hand_fwd[:, 1] * fwd[:, 0], torch.sum(term.hand_fwd * fwd[:, :2], dim=1))
  return torch.stack([torch.sum(moved * fwd[:, :2], dim=1), torch.sum(moved * left[:, :2], dim=1), turn], dim=1)


def drift(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """What stay_put and hand_hold charge, so the value can follow them (critic only): the
  slow velocity, and with a paw out the offset since it went out."""
  term = _term(env, command_name)
  out = (term.hand_side > 0).float().unsqueeze(1)
  return torch.cat([term.creep, _paw_offset(env, command_name) * out], dim=1)


def paw_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The simulated hand's force on the paw in the heading frame (critic only)."""
  force = getattr(env, "paw_force_h", None)
  return torch.zeros(env.num_envs, 3, device=env.device) if force is None else force


##
# Rewards
##


def track_heading_velocity(
  env: ManagerBasedRlEnv, command_name: str, std: float, axes: tuple[int, ...]
) -> torch.Tensor:
  """exp(-error^2 / std^2) over `axes` of (forward, left, yaw rate) in the heading frame."""
  asset: Entity = env.scene["robot"]
  err = env.command_manager.get_command(command_name) - heading_velocity(asset)
  return torch.exp(-torch.sum(torch.square(err[:, list(axes)]), dim=1) / std**2)


def yaw_rate_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.abs(env.command_manager.get_command(command_name)[:, 2] - heading_velocity(asset)[:, 2])


def _upright_error(asset: Entity, lean: float) -> torch.Tensor:
  """Angle (rad) between the nose and straight up tipped `lean` rad toward the belly."""
  fwd, _ = heading_frame(asset)
  target = math.cos(lean) * torch.tensor([0.0, 0.0, 1.0], device=fwd.device) + math.sin(lean) * fwd
  cos = torch.sum(body_axis(asset, 0) * target, dim=1)
  return torch.acos(torch.clamp(cos, -1.0, 1.0))


def upright(env: ManagerBasedRlEnv, command_name: str, std: float, lean: float) -> torch.Tensor:
  """Nose up (a slight lean toward the belly): exp(-angle^2 / std^2), while up."""
  angle = _upright_error(env.scene["robot"], lean)
  return torch.exp(-torch.square(angle) / std**2) * _gate(env, command_name, "up")


def upright_wide(env: ManagerBasedRlEnv, command_name: str, lean: float) -> torch.Tensor:
  """(1 + cos angle) / 2 to the same target: pays the whole way up from four legs."""
  angle = _upright_error(env.scene["robot"], lean)
  return 0.5 * (1.0 + torch.cos(angle)) * _gate(env, command_name, "up")


def level(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Body level on four legs: exp(-tilt^2 / std^2), while down."""
  up_b = body_axis(env.scene["robot"], 2)
  tilt = torch.acos(torch.clamp(up_b[:, 2], -1.0, 1.0))
  return torch.exp(-torch.square(tilt) / std**2) * _gate(env, command_name, "down")


def level_wide(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(1 + cos tilt) / 2: pays the whole way down from upright, while down."""
  up_b = body_axis(env.scene["robot"], 2)
  return 0.5 * (1.0 + up_b[:, 2]) * _gate(env, command_name, "down")


class _progress:
  """A measure's rise per second while down, its fall costs the same: holding earns nothing."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev = torch.zeros(env.num_envs, device=env.device)
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.fresh[slice(None) if env_ids is None else env_ids] = True

  def _gain(self, env: ManagerBasedRlEnv, command_name: str, value: torch.Tensor) -> torch.Tensor:
    gain = torch.where(self.fresh, 0.0, (value - self.prev) / env.step_dt)
    self.prev = value
    self.fresh[:] = False
    return gain * _gate(env, command_name, "down")


class level_progress(_progress):
  """Getting more level while down: (1 + cos tilt) / 2. Only the floor pays (a lean paid by
  level_wide trapped the come-down short of the floor)."""

  def __call__(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return self._gain(env, command_name, 0.5 * (1.0 + body_axis(env.scene["robot"], 2)[:, 2]))


class front_feet_progress(_progress):
  """The front feet getting lower while down: 1 - their mean height / max_height (clamped).
  A level body on the rear legs earns nothing (v3 came level that way, the front legs folded
  back as a counterweight)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.sites, _ = env.scene["robot"].find_sites(cfg.params["site_names"], preserve_order=True)

  def __call__(
    self, env: ManagerBasedRlEnv, command_name: str, site_names: tuple[str, ...], max_height: float
  ) -> torch.Tensor:
    del site_names  # Resolved in __init__.
    height = env.scene["robot"].data.site_pos_w[:, self.sites, 2].mean(dim=1)
    return self._gain(env, command_name, 1.0 - torch.clamp(height, 0.0, max_height) / max_height)


class base_height:
  """Base height above the lowest foot near its target (upright or on four legs):
  exp(-error^2 / std^2)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    self.sites, _ = asset.find_sites(cfg.params["site_names"], preserve_order=True)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    site_names: tuple[str, ...],
    upright_height: float,
    four_leg_height: float,
    std: float,
    foot_radius: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del site_names
    asset: Entity = env.scene[asset_cfg.name]
    ground = asset.data.site_pos_w[:, self.sites, 2].amin(dim=1) - foot_radius
    h = asset.data.root_link_pos_w[:, 2] - ground
    up = _term(env, command_name).up
    target = up * upright_height + (1.0 - up) * four_leg_height
    return torch.exp(-torch.square(h - target) / std**2)


def _risen(env: ManagerBasedRlEnv, min_nose_deg: float) -> torch.Tensor:
  return (nose_angle(env.scene["robot"]) > math.radians(min_nose_deg)).float()


def gait(
  env: ManagerBasedRlEnv,
  command_name: str,
  period: float,
  offset: list[float],
  threshold: float,
  sensor_name: str,
  min_nose_deg: float,
) -> torch.Tensor:
  """1 with both rear feet on the clock, while up, risen and walking. Both at once: counted
  per foot (the velocity task's feet_gait), a hop earns half."""
  contact = env.scene[sensor_name].data.current_contact_time > 0
  t = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
  stance = (t + torch.tensor(offset, device=env.device)) % 1.0 < threshold
  pay = (stance == contact).all(dim=1).float() * _risen(env, min_nose_deg)
  return pay * _gate(env, command_name, "up") * (1.0 - _standing(env, command_name))


def flight(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, min_nose_deg: float) -> torch.Tensor:
  """1 with both rear feet in the air, while up and risen (the rise itself may leave the
  ground)."""
  airborne = (~_in_contact(env, sensor_name, r".").any(dim=1)).float()
  return airborne * _risen(env, min_nose_deg) * _gate(env, command_name, "up")


def planted(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """1 with both rear feet on the ground, while up and standing."""
  both = _in_contact(env, sensor_name, r".").all(dim=1).float()
  return both * _gate(env, command_name, "up") * _standing(env, command_name)


def standing_steps(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """Rear-foot touchdowns this step, while up and standing (each one costs)."""
  sensor = env.scene[sensor_name]
  steps = sensor.compute_first_contact(dt=env.step_dt).float().sum(dim=1)
  return steps * _gate(env, command_name, "up") * _standing(env, command_name)


def standing_joint_speed(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Sum of joint speed^2, while up and standing: quiet, not fidgeting."""
  qd = env.scene[asset_cfg.name].data.joint_vel
  return torch.sum(torch.square(qd), dim=1) * _gate(env, command_name, "up") * _standing(env, command_name)


def standing_action_rate(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sum of action changes^2, while up and standing: a steady policy, not a vibrating one."""
  change = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(change), dim=1) * _gate(env, command_name, "up") * _standing(env, command_name)


def feet_down(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, geom_pattern: str, when: str) -> torch.Tensor:
  """Number of those feet on the ground (front feet while up: a cost; all four while down)."""
  down = _in_contact(env, sensor_name, geom_pattern).float().sum(dim=1)
  return down * _gate(env, command_name, when)


def arm_raise(
  env: ManagerBasedRlEnv, command_name: str, limit: float, min_nose_deg: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum of how far (m) the paws (asset_cfg's sites: left, right) are above limit, measured
  from the base, while up and risen. A paw out for the handshake doesn't count."""
  asset = env.scene[asset_cfg.name]
  height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - asset.data.root_link_pos_w[:, 2:3]
  side = _term(env, command_name).ramp_side
  free = torch.stack([side != 1, side != 2], dim=1).float()
  over = torch.sum(torch.clamp(height - limit, min=0.0) * free, dim=1)
  return over * _risen(env, min_nose_deg) * _gate(env, command_name, "up")


def hand_hold(env: ManagerBasedRlEnv, command_name: str, yaw_weight: float) -> torch.Tensor:
  """Distance (m) from where the paw went out plus yaw_weight x the turn since (rad), while
  a paw is out."""
  offset = _paw_offset(env, command_name)
  size = torch.norm(offset[:, :2], dim=1) + yaw_weight * torch.abs(offset[:, 2])
  return size * (_term(env, command_name).hand_side > 0).float()


def stay_put(env: ManagerBasedRlEnv, command_name: str, settle_s: float, yaw_weight: float) -> torch.Tensor:
  """Slow drift speed (m/s) plus yaw_weight x slow turn speed (rad/s), once it has stood
  upright at zero cmd_vel for settle_s. On speed, not distance: speed is what the policy can
  feel over its history."""
  term = _term(env, command_name)
  speed = torch.norm(term.creep[:, :2], dim=1) + yaw_weight * torch.abs(term.creep[:, 2])
  return speed * (term.still_for >= settle_s).float()


class posture(variable_posture):
  """The velocity task's posture term around the default (upright) pose, while up."""

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return super().__call__(env, **kwargs) * _gate(env, kwargs["command_name"], "up")


class stand_pose:
  """On four legs in `pose` (joint regex -> rad): exp(-mean squared error / pose_var), times
  exp(-sum joint speed^2 / still_var), while down."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    pose = cfg.params["pose"]
    values = [next(v for p, v in pose.items() if re.fullmatch(p, n)) for n in asset.joint_names]
    self.pose = torch.tensor(values, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    pose: dict[str, float],
    pose_var: float,
    still_var: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del pose
    asset: Entity = env.scene[asset_cfg.name]
    err = torch.mean(torch.square(asset.data.joint_pos - self.pose), dim=1)
    still = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    pay = torch.exp(-err / pose_var) * torch.exp(-still / still_var)
    return pay * _gate(env, command_name, "down")


def upright_cmd_metric(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Metric: share of the time with the up command at 1."""
  return _term(env, command_name).up


def walking_metric(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Metric: share of the time up and walking."""
  return _gate(env, command_name, "up") * (1.0 - _standing(env, command_name))


def one_foot_metric(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, min_nose_deg: float) -> torch.Tensor:
  """Metric: share of the time on exactly one rear foot, up, risen and walking (over
  walking_metric: how much it steps, not hops)."""
  one = _in_contact(env, sensor_name, r".").sum(dim=1) == 1
  return one.float() * _risen(env, min_nose_deg) * walking_metric(env, command_name)


def hand_out_metric(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Metric: share of the time with a paw out (ramp included)."""
  return _term(env, command_name).hand_ramp


##
# Terminations
##


def fell(
  env: ManagerBasedRlEnv,
  min_nose_deg: float,
  max_nose_deg: float,
  max_roll_deg: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """The nose outside [min_nose_deg, max_nose_deg] (0 = level on four legs, 90 = upright,
  past 90 = leaning back) or the left axis tipped more than max_roll_deg."""
  asset: Entity = env.scene[asset_cfg.name]
  nose = torch.rad2deg(nose_angle(asset))
  roll = torch.rad2deg(torch.abs(roll_angle(asset)))
  return (nose < min_nose_deg) | (nose > max_nose_deg) | (roll > max_roll_deg)
