"""Rewards and metrics for the kicks.

DonkeyKick: rise into a ~45 deg handstand on the front legs, hold it, kick both rear
legs straight back once, come down and stand still. QuickBuck (bottom section): one
fast strike of both rear feet at a set height, without the handstand, then stand.

The kick is triggered at t = 0 (episode start), so every phase is a time window;
the times come from KICK_TIMING (kick_env_cfg.py) / BUCK_TIMING (buck_env_cfg.py)
through each term's params.
Pitch is + when nose-down, measured against gravity (so 45 deg on a slope is 45 deg
in the world).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from src.tasks.recovery.mdp.rewards import (
  _pose_score,
  _slot_columns,
  _upright_gate,
  lying_still_check,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Contact-sensor columns per (sensor, geom pattern), resolved on first use.
_COLS: dict[tuple[int, str], torch.Tensor] = {}


def _cols(env: ManagerBasedRlEnv, sensor: ContactSensor, geom_pattern: str) -> torch.Tensor:
  key = (id(sensor), geom_pattern)
  if key not in _COLS:
    cols, _ = _slot_columns(sensor, geom_pattern)
    _COLS[key] = torch.tensor(cols, device=env.device)
  return _COLS[key]


def _in_contact(env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found[:, _cols(env, sensor, geom_pattern)] > 0  # [B, feet]


def _time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf.float() * env.step_dt


def _window(env: ManagerBasedRlEnv, start: float, end: float) -> torch.Tensor:
  t = _time(env)
  return (t >= start) & (t < end)


def pitch_roll(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """(pitch, roll) in rad from projected gravity; pitch + = nose down."""
  g = asset.data.projected_gravity_b
  pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))
  roll = torch.atan2(-g[:, 1], -g[:, 2])
  return pitch, roll


def pitch_target(
  t: torch.Tensor, rise_end: float, down_start: float, stand_start: float, pitch: float
) -> torch.Tensor:
  """Ramp 0 -> pitch over [0, rise_end], hold, ramp back to 0 over [down_start, stand_start]."""
  up = torch.clamp(t / rise_end, 0.0, 1.0)
  down = torch.clamp((t - down_start) / (stand_start - down_start), 0.0, 1.0)
  return pitch * up * (1.0 - down)


def pitch_track(
  env: ManagerBasedRlEnv,
  rise_end: float,
  down_start: float,
  stand_start: float,
  pitch: float,
  std: float,
  coarse_range: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Follow the handstand pitch profile: half exp(-err^2 / std^2) (precise), half
  1 - |err| / coarse_range (clipped at 0).

  The precise part alone pays ~0 to a robot still standing level when the target
  is 45 deg (exp(-21) at std 10 deg), leaving no signal to start tipping up; the
  linear part pays for every bit of the way.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = pitch_target(_time(env), rise_end, down_start, stand_start, pitch)
  err = pitch_roll(asset)[0] - target
  fine = torch.exp(-torch.square(err) / std**2)
  coarse = torch.clamp(1.0 - torch.abs(err) / coarse_range, min=0.0)
  return 0.5 * (fine + coarse)


def rear_feet_air(
  env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str, start: float, end: float
) -> torch.Tensor:
  """1 while both rear feet are off the ground inside [start, end)."""
  airborne = ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)
  return (airborne & _window(env, start, end)).float()


def front_feet_planted(env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str) -> torch.Tensor:
  """Share of the front feet in contact (the handstand stands on them)."""
  return _in_contact(env, sensor_name, geom_pattern).float().mean(dim=1)


def front_feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  geom_pattern: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sum over front feet in contact of foot xy speed^2 (m^2/s^2).

  Foot sites are named after their leg (FL, FR), like the foot geoms.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  key = (id(sensor), geom_pattern + "/sites")
  if key not in _COLS:
    _, names = _slot_columns(sensor, geom_pattern)
    sites = [list(asset.site_names).index(n.split("/")[-1].split("_")[0]) for n in names]
    _COLS[key] = torch.tensor(sites, device=env.device)
  speed_sq = torch.sum(torch.square(asset.data.site_lin_vel_w[:, _COLS[key], :2]), dim=-1)
  return torch.sum(speed_sq * _in_contact(env, sensor_name, geom_pattern).float(), dim=1)


def roll_level(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """roll^2: stay level side to side while balancing on the two front feet."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(pitch_roll(asset)[1])


def pitch_excess(
  env: ManagerBasedRlEnv, max_pitch: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """(pitch - max_pitch)^2 beyond max_pitch nose-down: keeps the head off the ground."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(torch.clamp(pitch_roll(asset)[0] - max_pitch, min=0.0))


def kick_score(
  asset: Entity,
  foot_ids: torch.Tensor,
  hip_ids: torch.Tensor,
  reach_min: float,
  reach_span: float,
) -> torch.Tensor:
  """Both-rear-feet kick extension in [0, 1]: the smaller of the two feet.

  Reach = foot distance behind its hip (thigh joint) along the body's long axis,
  which points back and up when the body is nose-down. 0 at <= reach_min (legs
  tucked or under the body), 1 at >= reach_min + reach_span (straight back).
  """
  d_w = asset.data.site_pos_w[:, foot_ids] - asset.data.body_link_pos_w[:, hip_ids]  # [B, 2, 3]
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, d_w.shape[1], -1)
  d_b = quat_apply_inverse(quat.reshape(-1, 4), d_w.reshape(-1, 3)).view_as(d_w)
  reach = -d_b[..., 0]
  return torch.clamp((reach - reach_min) / reach_span, 0.0, 1.0).amin(dim=1)


def _rear_leg_ids(env: ManagerBasedRlEnv, entity: str, feet, hips) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[entity]
  foot_ids = torch.tensor([list(asset.site_names).index(f) for f in feet], device=env.device)
  hip_ids = torch.tensor([list(asset.body_names).index(h) for h in hips], device=env.device)
  return foot_ids, hip_ids


class kick_snap:
  """Pays as the episode's best kick score inside [start, end) rises above its value
  at the window's first step: the total is (best - start value).

  Tucking the legs before the window lowers the start value, so tuck-then-snap
  pays most; holding the legs back or kicking again earns nothing more. The score
  only counts in the handstand (nose-down beyond min_pitch, both rear feet off the
  ground): reach is measured along the body, so random leg swings while standing
  level would otherwise pay too. The increment is divided by the step time, so
  with the reward manager's dt scaling the episode total is weight x (best -
  start value).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    entity = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    self.foot_ids, self.hip_ids = _rear_leg_ids(env, entity, cfg.params["feet"], cfg.params["hips"])
    self.best = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.best[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    feet: tuple[str, ...],
    hips: tuple[str, ...],
    start: float,
    end: float,
    reach_min: float,
    reach_span: float,
    min_pitch: float,
    sensor_name: str,
    geom_pattern: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    up = (pitch_roll(asset)[0] > min_pitch) & ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)
    score = kick_score(asset, self.foot_ids, self.hip_ids, reach_min, reach_span) * up.float()
    step = env.episode_length_buf
    first = step == round(start / env.step_dt)
    inside = (step > round(start / env.step_dt)) & (step < round(end / env.step_dt))
    gain = torch.where(inside, torch.clamp(score - self.best, min=0.0), torch.zeros_like(score))
    self.best = torch.where(first, score, torch.where(inside, torch.maximum(self.best, score), self.best))
    return gain / env.step_dt


def stand_pose(
  env: ManagerBasedRlEnv,
  start: float,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """From `start` on: joints close to the stand (default) pose, counted only when level."""
  asset: Entity = env.scene[asset_cfg.name]
  late = (_time(env) >= start).float()
  return late * _upright_gate(asset, tilt_full, tilt_zero) * _pose_score(asset, pose_var)


def stand_still(
  env: ManagerBasedRlEnv,
  start: float,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """From `start` on: little joint and body motion, counted only when level and in pose."""
  asset: Entity = env.scene[asset_cfg.name]
  calm = torch.exp(
    -torch.mean(torch.square(asset.data.joint_vel), dim=1)
    - torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  )
  return stand_pose(env, start, tilt_full, tilt_zero, pose_var, asset_cfg) * calm


##
# QuickBuck: one fast strike of both rear feet at a set height, no handstand.
##


def rear_feet_state(
  asset: Entity, foot_ids: torch.Tensor, hip_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Rear feet (each [B, 2]): height above the base link (m), reach behind the hip
  (thigh joint) and backward speed (m/s), both horizontal in the heading frame.

  The heading frame (yaw only), not the body: a target behind the robot is hit
  horizontally, whatever the body's pitch. The speed is the world foot speed.
  """
  p = asset.data.site_pos_w[:, foot_ids]  # [B, 2, 3]
  height = p[..., 2] - asset.data.root_link_pos_w[:, None, 2]
  quat = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, p.shape[1], -1).reshape(-1, 4)
  d = quat_apply_inverse(quat, (p - asset.data.body_link_pos_w[:, hip_ids]).reshape(-1, 3)).view_as(p)
  v = quat_apply_inverse(quat, asset.data.site_lin_vel_w[:, foot_ids].reshape(-1, 3)).view_as(p)
  return height, -d[..., 0], -v[..., 0]


def zone_offsets(
  height: torch.Tensor, reach: torch.Tensor, height_range, min_reach: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """How far (m) the feet are outside the strike zone, 0 inside: in height above the
  base link (below or above height_range) and in reach behind the hip (short of
  min_reach)."""
  lo, hi = height_range
  off_height = torch.clamp(torch.maximum(lo - height, height - hi), min=0.0)
  off_reach = torch.clamp(min_reach - reach, min=0.0)
  return off_height, off_reach


def strike_zone(height: torch.Tensor, reach: torch.Tensor, height_range, edge: float, min_reach: float):
  """1 inside the strike zone (height above the base link within height_range, at
  least min_reach behind the hip), fading linearly to 0 over `edge` outside it."""
  off_height, off_reach = zone_offsets(height, reach, height_range, min_reach)
  return torch.clamp(1.0 - off_height / edge, min=0.0) * torch.clamp(1.0 - off_reach / edge, min=0.0)


def zone_distance(height: torch.Tensor, reach: torch.Tensor, height_range, min_reach: float) -> torch.Tensor:
  """Straight-line distance (m) from the feet to the strike zone, 0 inside."""
  return torch.hypot(*zone_offsets(height, reach, height_range, min_reach))


def strike_score(
  asset: Entity,
  foot_ids: torch.Tensor,
  hip_ids: torch.Tensor,
  height_range: tuple[float, float],
  edge: float,
  min_reach: float,
  v_full: float,
) -> torch.Tensor:
  """Both-rear-feet strike in [0, 1]: the smaller of the two feet's
  (backward speed / v_full, capped at 1) x strike_zone."""
  height, reach, speed = rear_feet_state(asset, foot_ids, hip_ids)
  fast = torch.clamp(speed / v_full, 0.0, 1.0)
  return (fast * strike_zone(height, reach, height_range, edge, min_reach)).amin(dim=1)


class _best_in_window:
  """Base for terms paid once per episode: pays as the episode's best score inside
  [start, end) rises, so the episode total is the best score reached. A second kick
  or holding the pose earns nothing more. The increment is divided by the step
  time, so with the reward manager's dt scaling the episode total is weight x best.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    entity = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    self.foot_ids, self.hip_ids = _rear_leg_ids(env, entity, cfg.params["feet"], cfg.params["hips"])
    self.best = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.best[slice(None) if env_ids is None else env_ids] = 0.0

  def _pay(self, env: ManagerBasedRlEnv, score: torch.Tensor, start: float, end: float) -> torch.Tensor:
    inside = _window(env, start, end)
    gain = torch.where(inside, torch.clamp(score - self.best, min=0.0), torch.zeros_like(score))
    self.best = torch.where(inside, torch.maximum(self.best, score), self.best)
    return gain / env.step_dt


class strike(_best_in_window):
  """The kick: episode total = weight x the best strike_score in the kick window,
  with both rear feet off the ground."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    feet: tuple[str, ...],
    hips: tuple[str, ...],
    start: float,
    end: float,
    height_range: tuple[float, float],
    edge: float,
    min_reach: float,
    v_full: float,
    sensor_name: str,
    geom_pattern: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)
    score = strike_score(asset, self.foot_ids, self.hip_ids, height_range, edge, min_reach, v_full)
    return self._pay(env, score * airborne.float(), start, end)


class zone_approach(_best_in_window):
  """Guide to the strike zone: episode total = weight x the best closeness of the rear
  feet to the zone in the kick window, with both rear feet off the ground. Closeness
  = 1 - zone_distance / d0 (clipped at 0) of the farther foot.

  The strike pays only fast feet in or near the zone, which random kicks rarely
  reach; this pays for every bit of the way there. Height and reach together: height
  alone paid most for tucking the feet up under the belly (buck_v1), high but in
  front of the hips. Only with both rear feet off the ground: squatting the rear
  would otherwise pay too.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    feet: tuple[str, ...],
    hips: tuple[str, ...],
    start: float,
    end: float,
    height_range: tuple[float, float],
    min_reach: float,
    d0: float,
    sensor_name: str,
    geom_pattern: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    airborne = ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)
    height, reach, _ = rear_feet_state(asset, self.foot_ids, self.hip_ids)
    dist = zone_distance(height, reach, height_range, min_reach).amax(dim=1)
    score = torch.clamp(1.0 - dist / d0, 0.0, 1.0)
    return self._pay(env, score * airborne.float(), start, end)


def rear_feet_down(env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str, start: float) -> torch.Tensor:
  """From `start` on: share of the rear feet back on the ground (land soon after the kick)."""
  down = _in_contact(env, sensor_name, geom_pattern).float().mean(dim=1)
  return down * (_time(env) >= start).float()


##
# Metrics (episode averages of per-step values).
##


def handstand(
  env: ManagerBasedRlEnv,
  min_pitch: float,
  sensor_name: str,
  geom_pattern: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """1 while nose-down beyond min_pitch with both rear feet off the ground.

  Episode average x episode length = seconds in the handstand (target ~1.2 s).
  """
  asset: Entity = env.scene[asset_cfg.name]
  up = pitch_roll(asset)[0] > min_pitch
  airborne = ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)
  return (up & airborne).float()


def rear_up(env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str) -> torch.Tensor:
  """1 while both rear feet are off the ground (QuickBuck: x episode length = air time)."""
  return (~_in_contact(env, sensor_name, geom_pattern).any(dim=1)).float()


def stand_done(
  env: ManagerBasedRlEnv,
  start: float,
  max_tilt: float,
  max_pose_err: float,
  max_joint_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """1 from `start` on while the deploy done check (State_DonkeyKick) holds: level,
  every joint within max_pose_err of the stand pose and slower than max_joint_vel."""
  check = lying_still_check(env, max_tilt, max_pose_err, max_joint_vel, asset_cfg)
  return check * (_time(env) >= start).float()
