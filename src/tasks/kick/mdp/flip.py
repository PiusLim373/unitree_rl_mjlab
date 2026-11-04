"""Rewards and the fall check for the flips (SideFlip, FrontFlip, BackFlip).

A flip is a full turn about one body axis: `axis` 0 (forward, x: the side flip) or 1
(left, y: the front flip and backflip). `direction` is the turn's sign about that axis
(right-hand rule): the side flip uses -1 (over its left side; the right flip is the left
policy mirrored, MirroredPolicy), the front flip +1 (nose down), the backflip -1 (nose up).

The rotation is integrated from the body's rate about the axis (projected gravity wraps
at 180 deg), from the trigger on. It only counts after a liftoff: no limb (feet, calves,
thighs) on the ground with the body rising, so tipping over (limbs briefly off while it
falls) earns nothing.

fixed_axis (FrontFlip v2): integrate the world rate about the axis as it was at the start
(made level) instead. The body rate also counts a spin about the vertical while the axis
is tilted: FrontFlip v1 stood on its front feet and spun round instead of flipping.
upside_down (rad): progress past half a turn and the landing only count once the body has
been this far from level, so a flip has to go over its back (or side). flip_fell's
max_axis_angle (v3) holds the axis on its start direction, so it can't twist round either.

SideFlip follows Unitree's own (from video): turn_track holds the rotation to a
timeline, so spinning on past one turn or spinning too fast costs reward. FrontFlip v4
has no timeline, and nothing pays just for staying in the episode (heading_hold as a
penalty, standing only after a landing, flip_termination waives a committed crash), so
every bit more rotation pays: v3 hopped onto its front feet instead of trying.

Mid-flip starts (FrontFlip v5, events.reset_mid_flip): the rotation terms take the turn
so far from the body's pose at the first step (already lifted off), and the launch terms
(progress up to there, spin, flight height, air time) pay nothing in those episodes.

FrontFlip v6: v5 turned ~255 deg in the air and came down on its rear hips. Unitree's
flip spends part of the turn on the ground: rear_catch pays the rear feet catching the
ground late in the turn, front_pivot the turn made on the front feet before the liftoff.

FrontFlip v7: v6 caught on its rear feet, then dropped onto its chin and belly with the
front feet. The termination penalty comes before flip_landing in a step, so a touchdown
with the body down at the same step paid the landing and had its crash waived. Now the
landing needs no body contact (contact_term), a crash after the catch costs the full
penalty (catch_term), and standing pays from the catch.

FrontFlip v9: v8 landed, then knelt on its thighs and calves with the legs splayed, where
the stand pose term pays almost nothing (and knees aren't body contact). Now a thigh or
calf on the ground after the catch or landing costs (knees_down_after_landing), and a
second stand pose term with a wide pose_var pulls it up from there.

FrontFlip v11: v10 swung all four legs to one side after the liftoff, twisted in the air
and caught on one rear foot. Mirror training only makes the policy treat both sides
alike, not each flip symmetric: mirror_pose costs any difference between the two sides.

FrontFlip v12: v11 still reached one rear leg out and tapped that foot down alone, rolled
onto the front feet, and only then brought both rear feet down. rear_catch pays on either
rear foot, so rear_stagger costs the time from the first rear touchdown until both are
down together.

FrontFlip v13: v12 brought the second rear foot down sooner but kept the reach (the body
rolls ~25 deg just before the catch), and rear_stagger also charged the uneven touchdowns
rough ground and stairs force. Now mirror_pose only counts in the air, much stronger, and
rear_stagger is gone.

FrontFlip multistage v1 (scripts/train_front_flip.py): v6 to v13 all caught on one rear
foot first, a habit v6 learned because rear_catch paid either rear foot, which later
fine-tunes couldn't remove. Trained from scratch with rear_catch's pair_window: only both
rear feet landing close together count, and a crash after either touched down costs
(flip_termination's catch_any).

FrontFlip multistage v2: v1 came down with its rear legs swung forward under the belly
(rear knees a few cm off the floor at the touchdown, from stage 1 on), so they hit as soon
as the legs took the load, and the knee penalty moved the impact onto stiff front legs.
landing_pose pays the legs in the stand pose on the way down, and a foot's impact costs
as much as a knee's. Added at stage 4 it didn't change the posture, so it pays from
stage 1 (the posture is learned with the flip), and mid-flip starts practise from the
stand pose. It scores the hips too: without them the rear legs splayed out sideways.

BackFlip: the front flip's terms with the pairs trading roles (rear_catch on the front feet,
front_pivot on the rear ones). A short backflip comes down on its chin, where the LiDAR
sits, so flip_termination's head_pattern never waives a crash on the head or chin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import re

import torch

from mjlab.actuator import DcMotorActuator
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from src.tasks.recovery.mdp.rewards import _tilt

from .rewards import _in_contact, _time, _window, stand_pose, stand_still

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_axis(asset: Entity, axis: int = 0) -> torch.Tensor:
  """The body's x (0, forward) or y (1, left) axis in the world frame; a turn about that
  axis doesn't move it."""
  v = torch.zeros_like(asset.data.root_link_pos_w)
  v[:, axis] = 1.0
  return quat_apply(asset.data.root_link_quat_w, v)


def _airborne(env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str) -> torch.Tensor:
  return ~_in_contact(env, sensor_name, geom_pattern).any(dim=1)


def _level(v: torch.Tensor) -> torch.Tensor:
  """v made horizontal, unit length."""
  v = v.clone()
  v[:, 2] = 0.0
  return v / torch.clamp(torch.norm(v, dim=1, keepdim=True), min=1e-6)


def _started_mid_flip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Episodes that began in the air partway through the flip (reset_mid_flip)."""
  offset = getattr(env, "flip_time_offset", None)
  if offset is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return offset > 0.0


def _turn_angle(asset: Entity, axis: int, axis_w: torch.Tensor, direction: float) -> torch.Tensor:
  """The body's turn so far about the level axis axis_w in `direction`, in turns [0, 1),
  from where its other horizontal axis (forward for the front flip) points."""
  up = torch.zeros_like(axis_w)
  up[:, 2] = 1.0
  ref = torch.cross(axis_w, up, dim=1) if axis == 1 else torch.cross(up, axis_w, dim=1)
  perp = torch.cross(axis_w, ref, dim=1)
  other = _body_axis(asset, 1 - axis)
  angle = torch.atan2(torch.sum(other * perp, dim=1), torch.sum(other * ref, dim=1))
  return torch.remainder(direction * angle / (2.0 * math.pi), 1.0)


class _rotation:
  """Base for terms that follow the flip's rotation: integrates `direction` x the
  body's rate about `axis` (or the world rate about the start axis, fixed_axis) over
  [start, end) into `turns` (1 = a full turn), and notes a liftoff in that window
  (`lifted`): nothing matching geom_pattern in sensor_name on the ground while the base
  rises faster than min_up_vel (m/s). With upside_down, `inverted` notes the body more
  than that from level (rad) in the window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.turns = torch.zeros(env.num_envs, device=env.device)
    self.lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.inverted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.unset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.new_mid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.turns[ids] = 0.0
    self.lifted[ids] = False
    self.inverted[ids] = False
    self.unset[ids] = True

  def _update(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    asset_cfg: SceneEntityCfg,
    axis: int = 0,
    fixed_axis: bool = False,
    upside_down: float | None = None,
  ) -> torch.Tensor:
    """Advance the rotation; returns the window mask."""
    asset: Entity = env.scene[asset_cfg.name]
    inside = _window(env, start, end)
    # The start axis: the axis at the episode's first step, made level.
    first = self.unset.clone()
    self.axis_w = torch.where(first.unsqueeze(1), _level(_body_axis(asset, axis)), self.axis_w)
    self.unset[:] = False
    # A mid-flip start: the turn so far from the pose, already lifted off.
    self.new_mid = first & _started_mid_flip(env)
    if fixed_axis:
      w = torch.sum(asset.data.root_link_ang_vel_w * self.axis_w, dim=1)
    else:
      w = asset.data.root_link_ang_vel_b[:, axis]
    rate = direction * w / (2.0 * math.pi)
    self.turns = self.turns + torch.where(inside & ~self.new_mid, rate * env.step_dt, torch.zeros_like(rate))
    self.turns = torch.where(self.new_mid, _turn_angle(asset, axis, self.axis_w, direction), self.turns)
    self.lifted = self.lifted | self.new_mid
    rising = asset.data.root_link_lin_vel_w[:, 2] > min_up_vel
    self.lifted = self.lifted | (inside & rising & _airborne(env, sensor_name, geom_pattern))
    if upside_down is not None:
      self.inverted = self.inverted | (inside & (_tilt(asset) > upside_down))
      self.inverted = self.inverted | (self.new_mid & (self.turns >= 0.5))  # already over
    return inside


class turn_progress(_rotation):
  """The flip: episode total = weight x the best fraction of a full turn reached in
  [start, end), counted once the robot has left the ground (past half a turn only once
  upside down, with upside_down)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.best = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.best[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
    upside_down: float | None = None,
  ) -> torch.Tensor:
    self._update(
      env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis, upside_down
    )
    score = torch.clamp(self.turns, 0.0, 1.0) * self.lifted.float()
    if upside_down is not None:
      score = torch.where(self.inverted, score, torch.clamp(score, max=0.5))
    self.best = torch.where(self.new_mid, score, self.best)  # only the rest of a mid-flip start pays
    gain = torch.clamp(score - self.best, min=0.0)
    self.best = torch.maximum(self.best, score)
    # / step_dt: with the reward manager's dt scaling the episode total is weight x best.
    return gain / env.step_dt


class flip_landing(_rotation):
  """Once per episode: the first touchdown of a foot (touch_sensor / touch_pattern)
  after a liftoff and at least min_turns of rotation, paid if the rotation is at most
  max_turns and the body within max_tilt of level (and upside down on the way, with
  upside_down; with no body contact at that step, with contact_term). A landing on the
  side or back, a short flip or a double one earns nothing. `success` notes the robots it
  paid (stand_*_after_landing)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.done[ids] = False
    self.success[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    touch_sensor: str,
    touch_pattern: str,
    min_turns: float,
    max_turns: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
    upside_down: float | None = None,
    contact_term: str | None = None,
  ) -> torch.Tensor:
    self._update(
      env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis, upside_down
    )
    asset: Entity = env.scene[asset_cfg.name]
    feet_down = _in_contact(env, touch_sensor, touch_pattern).any(dim=1)
    touchdown = ~self.done & self.lifted & (self.turns >= min_turns) & feet_down
    landed = touchdown & (self.turns <= max_turns) & (_tilt(asset) < max_tilt)
    if upside_down is not None:
      landed = landed & self.inverted
    if contact_term is not None:
      landed = landed & ~env.termination_manager.get_term(contact_term)
    self.done = self.done | touchdown
    self.success = self.success | landed
    return landed.float() / env.step_dt


def _landed(env: ManagerBasedRlEnv, landing_term: str, catch_term: str | None = None) -> torch.Tensor:
  """1 for robots whose flip has landed (the flip_landing reward term `landing_term`
  paid), or with catch_term (FrontFlip v7) caught on the rear feet (that rear_catch term
  paid); the stand terms come first, so they read it from the previous step."""
  done = env.reward_manager.get_term_cfg(landing_term).func.success
  if catch_term is not None:
    done = done | env.reward_manager.get_term_cfg(catch_term).func.caught
  return done.float()


def stand_pose_after_landing(
  env: ManagerBasedRlEnv,
  landing_term: str,
  start: float,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  catch_term: str | None = None,
) -> torch.Tensor:
  """stand_pose, only once the flip has landed (or caught, with catch_term): standing
  after not flipping earns nothing."""
  gate = _landed(env, landing_term, catch_term)
  return gate * stand_pose(env, start, tilt_full, tilt_zero, pose_var, asset_cfg)


def stand_still_after_landing(
  env: ManagerBasedRlEnv,
  landing_term: str,
  start: float,
  tilt_full: float,
  tilt_zero: float,
  pose_var: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  catch_term: str | None = None,
) -> torch.Tensor:
  """stand_still, only once the flip has landed (or caught, with catch_term)."""
  gate = _landed(env, landing_term, catch_term)
  return gate * stand_still(env, start, tilt_full, tilt_zero, pose_var, asset_cfg)


def knees_down_after_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  geom_pattern: str,
  landing_term: str,
  catch_term: str | None = None,
) -> torch.Tensor:
  """FrontFlip v9: 1 while any thigh or calf (geom_pattern in sensor_name) is on the
  ground once the flip has landed (or caught, with catch_term)."""
  down = _in_contact(env, sensor_name, geom_pattern).any(dim=1)
  return _landed(env, landing_term, catch_term) * down.float()


def flip_landed(env: ManagerBasedRlEnv, landing_term: str, mid_start: bool) -> torch.Tensor:
  """Metric: 1 once the flip has landed, in episodes that began standing (mid_start False)
  or mid-flip (True); 0 in the others."""
  return _landed(env, landing_term) * (_started_mid_flip(env) == mid_start).float()


class flip_termination(_rotation):
  """The termination penalty: 1 when the episode ends early, except for a crash of the
  body alone (contact_term, not fall_term) after a liftoff and at least commit_turns of
  rotation: an attempt that came up short. Falls (tipped or twisted, not level in time)
  always count. With landing_term (FrontFlip v6), a crash after the landing counts too;
  with catch_term (v7), a crash after the rear feet caught the ground (rear_catch), or
  with catch_any (multistage) after either rear foot touched down (rear_catch.touched).
  With head_pattern (BackFlip), a crash with any of those geoms in head_sensor down (the
  head and chin) counts too."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    commit_turns: float,
    contact_term: str = "body_contact",
    fall_term: str = "fell",
    landing_term: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
    catch_term: str | None = None,
    catch_any: bool = False,
    head_sensor: str | None = None,
    head_pattern: str | None = None,
  ) -> torch.Tensor:
    self._update(env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis)
    tm = env.termination_manager
    committed = self.lifted & (self.turns >= commit_turns)
    if landing_term is not None:
      committed = committed & ~env.reward_manager.get_term_cfg(landing_term).func.success
    if catch_term is not None:
      catch = env.reward_manager.get_term_cfg(catch_term).func
      committed = committed & ~(catch.touched if catch_any else catch.caught)
    if head_pattern is not None:
      committed = committed & ~_in_contact(env, head_sensor, head_pattern).any(dim=1)
    crash = tm.get_term(contact_term) & ~tm.get_term(fall_term) & committed
    return (tm.terminated & ~crash).float()


class rear_catch(_rotation):
  """Once per episode (FrontFlip v6): the rear feet (touch_sensor / touch_pattern) on the
  ground after a liftoff and at least min_turns of rotation, with no body contact
  (contact_term) at that step (and upside down on the way, with upside_down). A landing
  on all four feet counts too. v5 came down on its rear hips at ~255 deg, rear legs tucked;
  Unitree's flip catches the ground with its rear feet there and rolls over them. `caught`
  notes the robots it paid.

  With pair_window (s, multistage): either rear foot alone no longer counts. Each rear
  foot's first touchdown that would count is noted, and the catch is the second one
  within pair_window of the first; it pays in full up to pair_full (s) apart, less up to
  pair_window, nothing (and never caught) after that. `touched` notes the robots with a
  rear foot down that would count, paired or not (the catch itself without pair_window).
  first_pay (multistage stage 1): this share of the weight pays at the first rear foot
  down, the rest for the pair (a stepping stone soon after going over the back).

  BackFlip: the same catch on the front feet (touch_pattern)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.caught = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.first: torch.Tensor | None = None  # [envs, rear feet] first touchdown time, inf before

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.caught[ids] = False
    self.touched[ids] = False
    if self.first is not None:
      self.first[ids] = float("inf")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    touch_sensor: str,
    touch_pattern: str,
    min_turns: float,
    contact_term: str = "body_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
    upside_down: float | None = None,
    pair_window: float | None = None,
    pair_full: float = 0.0,
    first_pay: float = 0.0,
  ) -> torch.Tensor:
    self._update(
      env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis, upside_down
    )
    down = _in_contact(env, touch_sensor, touch_pattern)
    body = env.termination_manager.get_term(contact_term)
    ready = self.lifted & (self.turns >= min_turns) & ~body
    if upside_down is not None:
      ready = ready & self.inverted
    first_touch = ~self.touched & ready & down.any(dim=1)
    self.touched = self.touched | first_touch
    if pair_window is None:
      catch = ~self.caught & self.touched
      self.caught = self.caught | catch
      return catch.float() / env.step_dt
    if self.first is None:
      self.first = torch.full(down.shape, float("inf"), device=env.device)
    new = ready[:, None] & down & torch.isinf(self.first)
    self.first = torch.where(new, _time(env)[:, None].expand_as(self.first), self.first)
    # inf (or nan) until every rear foot is down.
    gap = self.first.amax(dim=1) - self.first.amin(dim=1)
    catch = ~self.caught & torch.isfinite(gap) & (gap <= pair_window + 1e-6)
    self.caught = self.caught | catch
    pay = torch.clamp((pair_window - gap) / max(pair_window - pair_full, 1e-6), 0.0, 1.0)
    pay = first_pay * first_touch.float() + (1.0 - first_pay) * torch.where(catch, pay, torch.zeros_like(pay))
    return pay / env.step_dt


class front_pivot(_rotation):
  """FrontFlip v6: episode total = weight x the best rotation made with only the front
  feet (front_pattern, not rear_pattern, in feet_sensor) on the ground before the liftoff,
  as a fraction of max_turns (capped at 1). Unitree's flip turns ~110 deg on its front
  feet before it leaves the ground; v5 rocked back nose-up and launched level. Not in a
  mid-flip start. BackFlip: the turn on the rear feet (front_pattern / rear_pattern swapped)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.pivot = torch.zeros(env.num_envs, device=env.device)
    self.best = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.pivot[ids] = 0.0
    self.best[ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    feet_sensor: str,
    front_pattern: str,
    rear_pattern: str,
    max_turns: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
  ) -> torch.Tensor:
    before = self.turns.clone()
    self._update(env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis)
    front = _in_contact(env, feet_sensor, front_pattern).any(dim=1)
    rear = _in_contact(env, feet_sensor, rear_pattern).any(dim=1)
    on_front = front & ~rear & ~self.lifted & ~_started_mid_flip(env)
    self.pivot = self.pivot + torch.where(on_front, self.turns - before, torch.zeros_like(before))
    score = torch.clamp(self.pivot / max_turns, 0.0, 1.0)
    gain = torch.clamp(score - self.best, min=0.0)
    self.best = torch.maximum(self.best, score)
    return gain / env.step_dt


def flip_caught(env: ManagerBasedRlEnv, catch_term: str, mid_start: bool) -> torch.Tensor:
  """Metric: 1 once the rear feet have caught the ground (rear_catch term catch_term), in
  episodes that began standing (mid_start False) or mid-flip (True); 0 in the others."""
  caught = env.reward_manager.get_term_cfg(catch_term).func.caught
  return caught.float() * (_started_mid_flip(env) == mid_start).float()


class spin(_rotation):
  """Before pay_end and while short of max_turns: the whole robot's angular momentum
  about the flip's axis, in the flip's direction, as a fraction of target (N m s),
  capped at 1. Past max_turns it stops paying (SideFlip v1 spun on into a second roll).

  Whole-body, not the base's own rate: tucking the legs speeds the base up without
  adding any spin, and a learned flip needs the real thing (centroidal angular
  momentum, from a subtreeangmom sensor on the base). With fixed_axis, about the start
  axis.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    angmom_sensor: str,
    target: float,
    max_turns: float,
    pay_end: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
  ) -> torch.Tensor:
    self._update(env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis)
    asset: Entity = env.scene[asset_cfg.name]
    momentum = env.scene[angmom_sensor].data  # [B, 3] world frame, about the robot's centre of mass
    about_axis = torch.sum(momentum * (self.axis_w if fixed_axis else _body_axis(asset, axis)), dim=1)
    score = torch.clamp(direction * about_axis / target, 0.0, 1.0)
    pay = _window(env, start, pay_end) & (self.turns < max_turns) & ~_started_mid_flip(env)
    return score * pay.float()


class turn_track(_rotation):
  """Follow the flip's timeline: turns against time, piecewise linear through
  `profile` ((t, turns) points, flat before the first and after the last). Half
  exp(-err^2 / std^2) (precise), half 1 - |err| / coarse_range (clipped at 0), like
  DonkeyKick's pitch_track: the coarse part pays for every bit of the way from
  standing still. Too slow, too fast, or rolling on past one turn all lose reward."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    profile: tuple[tuple[float, float], ...],
    std: float,
    coarse_range: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
  ) -> torch.Tensor:
    self._update(env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis)
    err = self.turns - profile_target(_time(env), profile)
    fine = torch.exp(-torch.square(err) / std**2)
    coarse = torch.clamp(1.0 - torch.abs(err) / coarse_range, min=0.0)
    return 0.5 * (fine + coarse)


class over_rotated(_rotation):
  """1 while the rotation is past max_turns (metric: x episode length = seconds past)."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    direction: float,
    start: float,
    end: float,
    sensor_name: str,
    geom_pattern: str,
    min_up_vel: float,
    max_turns: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    fixed_axis: bool = False,
  ) -> torch.Tensor:
    self._update(env, direction, start, end, sensor_name, geom_pattern, min_up_vel, asset_cfg, axis, fixed_axis)
    return (self.turns > max_turns).float()


def profile_target(t: torch.Tensor, profile: tuple[tuple[float, float], ...]) -> torch.Tensor:
  """Piecewise-linear value of `profile` ((t, value) points, t increasing) at times t."""
  target = torch.full_like(t, float(profile[0][1]))
  for (t0, y0), (t1, y1) in zip(profile[:-1], profile[1:]):
    target = target + (y1 - y0) * torch.clamp((t - t0) / (t1 - t0), 0.0, 1.0)
  return target


class flight_height:
  """Once per episode: episode total = weight x the best base height gained above
  the episode's start (m), counted while airborne inside [start, end), as a fraction
  of `height` (capped at 1). More air time instead of a faster spin."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.z0 = torch.zeros(env.num_envs, device=env.device)
    self.best = torch.zeros(env.num_envs, device=env.device)
    self.unset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.best[ids] = 0.0
    self.unset[ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    geom_pattern: str,
    start: float,
    end: float,
    height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    self.z0 = torch.where(self.unset, z, self.z0)  # the base height at the first step
    self.unset[:] = False
    air = _airborne(env, sensor_name, geom_pattern) & _window(env, start, end) & ~_started_mid_flip(env)
    score = torch.clamp((z - self.z0) / height, 0.0, 1.0) * air.float()
    gain = torch.clamp(score - self.best, min=0.0)
    self.best = torch.maximum(self.best, score)
    return gain / env.step_dt


class dc_torque_saturation:
  """Motors above `fraction` of the torque they can give at their current speed (the
  DC motor's torque-speed curve, as in scripts/eval_flip.py), counted per step. v1
  held the push side's front calf there for ~0.27 s every flip; torque_excess only
  compares with the fixed maximum, which a fast motor never reaches.

  launch_term (FrontFlip v5): free until that rotation term's liftoff or launch_end (s),
  whichever comes first. The launch needs the motors at their limit; FrontFlip v4 gave up
  rotation to avoid this penalty."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    import mujoco

    entity = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[entity]
    model = env.sim.mj_model
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{entity}/{a}") for a in asset.actuator_names]
    if min(ids) < 0:
      raise ValueError(f"Actuators not found in the model: {asset.actuator_names}")
    self.limits = torch.tensor(model.actuator_forcerange[ids, 1], device=env.device, dtype=torch.float32)
    self.sat = self.limits.clone()
    self.top = torch.full_like(self.limits, float("inf"))
    for group in asset.actuators:
      if isinstance(group, DcMotorActuator):
        for i, a in enumerate(asset.actuator_names):
          if any(re.fullmatch(e, a) for e in group.cfg.target_names_expr):
            self.sat[i], self.top[i] = group.cfg.saturation_effort, group.cfg.velocity_limit
    joints = list(asset.joint_names)
    self.act_joint = torch.tensor([joints.index(a) for a in asset.actuator_names], device=env.device)

  def available(self, force: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    """Torque each motor can give in the direction of `force` at speed `vel`."""
    reach = self.top * (1 + self.limits / self.sat)
    v = torch.clamp(vel, -reach, reach)
    top = torch.clamp(self.sat * (1 - v / self.top), max=self.limits)
    bottom = torch.clamp(self.sat * (-1 - v / self.top), min=-self.limits)
    return torch.clamp(torch.where(force >= 0, top, -bottom), min=1e-3)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    fraction: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    launch_term: str | None = None,
    launch_end: float = 0.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    ratio = torch.abs(force) / self.available(force, asset.data.joint_vel[:, self.act_joint])
    count = torch.sum((ratio > fraction).float(), dim=1)
    if launch_term is not None:
      lifted = env.reward_manager.get_term_cfg(launch_term).func.lifted
      count = count * (lifted | (_time(env) >= launch_end)).float()
    return count


def airborne(
  env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str, start: float, end: float
) -> torch.Tensor:
  """1 while nothing matching geom_pattern touches the ground inside [start, end) (not in
  a mid-flip start)."""
  air = _airborne(env, sensor_name, geom_pattern) & _window(env, start, end) & ~_started_mid_flip(env)
  return air.float()


class mirror_pose:
  """FrontFlip v11: squared difference between every joint and its mirror twin (the same
  joint on the other side, by the FL/FR/RL/RR prefix), summed. Joints matching
  negate_pattern (the hips) mirror with their sign flipped. 0 for a symmetric pose. With
  sensor_name / geom_pattern (v13), only while nothing matching them touches the ground."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    names = list(asset.joint_names)
    other = {"L": "R", "R": "L"}
    twins = [names.index(re.sub(r"^([FR])([LR])_", lambda m: m[1] + other[m[2]] + "_", n)) for n in names]
    self.twin = torch.tensor(twins, device=env.device)
    negate = cfg.params.get("negate_pattern", r".*_hip_joint")
    self.sign = torch.tensor([-1.0 if re.fullmatch(negate, n) else 1.0 for n in names], device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    negate_pattern: str = r".*_hip_joint",
    sensor_name: str | None = None,
    geom_pattern: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    q = env.scene[asset_cfg.name].data.joint_pos
    cost = torch.sum(torch.square(q - q[:, self.twin] * self.sign), dim=1)
    if sensor_name is not None:
      cost = cost * _airborne(env, sensor_name, geom_pattern).float()
    return cost


class rear_stagger:
  """FrontFlip v12: 1 from the first rear foot (geom_pattern in sensor_name) touching down
  after the liftoff (launch_term's `lifted`, so it comes after that term) until both are
  down together, whether the first stays down or not (v11 tapped one down and lifted it
  again); nothing after that. `touched` / `paired` note the robots past each point."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.paired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.touched[ids] = False
    self.paired[ids] = False

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, geom_pattern: str, launch_term: str) -> torch.Tensor:
    down = _in_contact(env, sensor_name, geom_pattern)
    lifted = env.reward_manager.get_term_cfg(launch_term).func.lifted
    self.touched = self.touched | (lifted & down.any(dim=1))
    self.paired = self.paired | (lifted & down.all(dim=1))
    return (self.touched & ~self.paired).float()


class landing_pose:
  """FrontFlip multistage v2: the joints matching joint_pattern close to the stand
  (default) pose (_pose_score's exp(-mean squared error / pose_var)), from min_turns of
  launch_term's rotation after its liftoff (so it comes after that term) until anything
  matching geom_pattern in sensor_name (the legs) touches down; nothing after that, so
  landing on the knees doesn't keep it paying. `down` notes the robots past the touchdown
  (a touchdown before min_turns closes it too)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    pattern = cfg.params["joint_pattern"]
    joints = [i for i, n in enumerate(asset.joint_names) if re.fullmatch(pattern, n)]
    if not joints:
      raise ValueError(f"No joints match {pattern}")
    self.joints = torch.tensor(joints, device=env.device)
    self.down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.down[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    geom_pattern: str,
    launch_term: str,
    min_turns: float,
    pose_var: float,
    joint_pattern: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    launch = env.reward_manager.get_term_cfg(launch_term).func
    self.down = self.down | (launch.lifted & _in_contact(env, sensor_name, geom_pattern).any(dim=1))
    inside = launch.lifted & (launch.turns >= min_turns) & ~self.down
    err = asset.data.joint_pos[:, self.joints] - asset.data.default_joint_pos[:, self.joints]
    return inside.float() * torch.exp(-torch.mean(torch.square(err), dim=1) / pose_var)


class heading_hold:
  """The flip's axis (`axis`: 0 forward, 1 left) level and on its direction at the
  episode's start: cosine between the two (clipped at 0). A flip leaves its own axis
  where it was, so this only falls with the other two turns (the side flip's pitch and
  yaw, the front flip's roll and yaw). With penalty, 1 - that (0 when straight), so a
  crash doesn't lose it."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.heading = torch.zeros(env.num_envs, 3, device=env.device)
    self.unset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.unset[slice(None) if env_ids is None else env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    penalty: bool = False,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    along = _body_axis(asset, axis)
    # The start heading: the axis at the episode's first step, made level.
    self.heading = torch.where(self.unset.unsqueeze(1), _level(along), self.heading)
    self.unset[:] = False
    hold = torch.clamp(torch.sum(along * self.heading, dim=1), min=0.0)
    return 1.0 - hold if penalty else hold


class flip_fell:
  """The flip's axis more than max_axis_tilt from level (rad) at any time (the side
  flip's pitch, the front flip's roll), or the body more than max_tilt from level after
  `after` (s): by then the flip must be over. The turn itself is free while flipping.

  max_axis_angle (rad, FrontFlip v3): the axis also more than this from its direction at
  the episode's first step (made level), tipped or turned. FrontFlip v2 twisted round in
  the air while vertical (a round-off), which turns the axis without tipping it."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.start = torch.zeros(env.num_envs, 3, device=env.device)
    self.unset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.unset[slice(None) if env_ids is None else env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    max_axis_tilt: float,
    max_tilt: float,
    after: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    axis: int = 0,
    max_axis_angle: float | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    along = _body_axis(asset, axis)
    axis_tilt = torch.asin(torch.clamp(along[:, 2], -1.0, 1.0))
    late = _time(env) >= after
    fell = (torch.abs(axis_tilt) > max_axis_tilt) | (late & (_tilt(asset) > max_tilt))
    if max_axis_angle is not None:
      self.start = torch.where(self.unset.unsqueeze(1), _level(along), self.start)
      self.unset[:] = False
      off = torch.acos(torch.clamp(torch.sum(along * self.start, dim=1), -1.0, 1.0))
      fell = fell | (off > max_axis_angle)
    return fell
