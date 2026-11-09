"""UprightWalk's command term: the policy's three inputs.

- cmd_vel (vx, vy, wz) in the heading frame (frames.py), direct yaw-rate commands only.
- up: 1 = stand upright, 0 = down on four legs. Every episode starts at 1; with
  down_prob it switches to 0 at a time from down_time_range, and back to 1 with
  up_again_prob after up_again_delay. With down_start_prob it switches at once, so
  coming down is practised from every start, partly risen ones too. down_schedule turns
  these on partway through S1, once the rise is found.
- hand: 0 none, 1 left (FL), 2 right (FR). With hand_prob an episode cycles a paw out for
  hand_on_range and back for hand_off_range, only while up and settled upright (S2).
  hand_ramp follows it at 1 / ramp_s per second, the deploy's gain ramp; ramp_side is
  the leg the ramp applies to (it outlives the flag while ramping down). Going back,
  hand_pose first brings the arm to its rest pose over back_s, still soft, and only then
  does hand_ramp stiffen it over stiffen_s. With wave_prob a cycle is a wave, not a
  handshake; hand reads the same, the policy sees the arm. A wave keeps the leg's gains,
  so wave_ramp_s is all it has either way.

cmd_vel is zero while down or while a paw is out or ramping, as the deploy sends. A down
command with a paw out puts the paw back first, as the deploy does.

Also here: creep (the velocity averaged over creep_tau_s), still_for (how long it has stood
upright at zero cmd_vel), imu_tilt (each robot's IMU mounting error) and the last second of
sensors the policy reads in coarse bins (a long frame history stalls learning; a plain
average can't show drift, as a stepping leg's speed averages to zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from .frames import heading_frame, heading_velocity, nose_angle

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_INF = float("inf")
# cmd_vel (|v| + |w|) at or below this is standing.
STAND_THRESHOLD = 0.1


class UprightCommand(UniformVelocityCommand):
  cfg: UprightCommandCfg

  def __init__(self, cfg: UprightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.up = torch.ones(n, device=dev)
    self.down_at = torch.full((n,), _INF, device=dev)
    self.up_again_at = torch.full((n,), _INF, device=dev)
    self.hand_side = torch.zeros(n, dtype=torch.long, device=dev)
    self.hand_next = torch.full((n,), _INF, device=dev)
    self.hand_wave = torch.zeros(n, dtype=torch.bool, device=dev)
    self.hand_ramp = torch.zeros(n, device=dev)
    self.hand_pose = torch.zeros(n, device=dev)  # 1 = the paw's pose, 0 = the rest pose
    self.ramp_side = torch.zeros(n, dtype=torch.long, device=dev)
    self.hand_xy = torch.zeros(n, 2, device=dev)  # where the paw went out
    self.hand_fwd = torch.zeros(n, 2, device=dev)
    self.creep = torch.zeros(n, 3, device=dev)  # forward, left, yaw rate
    self.still_for = torch.zeros(n, device=dev)
    self.imu_tilt = torch.zeros(n, 3, device=dev)  # rad
    self.rear = torch.tensor([i for i, j in enumerate(self.robot.joint_names) if j.startswith("R")], device=dev)
    k = len(self.rear)
    gravity, gyro, pos, vel = self.cfg.slow_noise
    self.slow_noise = torch.tensor([gravity] * 3 + [gyro] * 3 + [pos] * k + [vel] * k, device=dev)
    self.slow_frames = torch.zeros(n, self.cfg.slow_bins * self.cfg.slow_bin_steps, len(self.slow_noise), device=dev)
    self.slow_fresh = torch.ones(n, dtype=torch.bool, device=dev)

  def _time(self) -> torch.Tensor:
    return self._env.episode_length_buf.float() * self._env.step_dt

  def _uniform(self, rng: tuple[float, float], n: int) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*rng)

  def _chance(self, p: float, n: int) -> torch.Tensor:
    return torch.rand(n, device=self.device) < p

  def tilted_gravity(self) -> torch.Tensor:
    gravity = self.robot.data.projected_gravity_b
    return gravity + torch.cross(self.imu_tilt, gravity, dim=1)

  def _slow_frame(self) -> torch.Tensor:
    """Gravity, gyro, rear joint positions and speeds, as the policy reads them."""
    data = self.robot.data
    frame = torch.cat([
      self.tilted_gravity(), data.root_link_ang_vel_b,
      (data.joint_pos - data.default_joint_pos)[:, self.rear],
      (data.joint_vel - data.default_joint_vel)[:, self.rear],
    ], dim=1)
    if self._env.cfg.observations["actor"].enable_corruption:
      frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * self.slow_noise
    return frame

  def slow_history(self) -> torch.Tensor:
    """The last second, oldest bin first. A robot just reset reads its first frame throughout,
    as a frame history does (env.reset() doesn't update the command, so it's done here)."""
    fresh = self.slow_fresh.view(-1, 1, 1)
    self.slow_frames = torch.where(fresh, self._slow_frame().unsqueeze(1), self.slow_frames)
    self.slow_fresh[:] = False
    n, _, width = self.slow_frames.shape
    return self.slow_frames.view(n, self.cfg.slow_bins, self.cfg.slow_bin_steps, width).mean(2).flatten(1)

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    n = len(env_ids)
    self.up[env_ids] = 1.0
    down_at = torch.where(self._chance(self.cfg.down_prob, n), self._uniform(self.cfg.down_time_range, n), _INF)
    self.down_at[env_ids] = torch.where(self._chance(self.cfg.down_start_prob, n), 0.0, down_at)
    self.up_again_at[env_ids] = _INF
    self.hand_side[env_ids] = 0
    self.hand_wave[env_ids] = False
    self.hand_ramp[env_ids] = 0.0
    self.hand_pose[env_ids] = 0.0
    self.ramp_side[env_ids] = 0
    self.creep[env_ids] = 0.0
    self.still_for[env_ids] = 0.0
    tilt = math.radians(self.cfg.imu_tilt_deg)
    self.imu_tilt[env_ids] = torch.empty(n, 3, device=self.device).uniform_(-tilt, tilt)
    self.slow_fresh[env_ids] = True
    self.hand_next[env_ids] = torch.where(
      self._chance(self.cfg.hand_prob, n), self._uniform(self.cfg.hand_first_range, n), _INF
    )
    return extras

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    actual = heading_velocity(self.robot)
    self.metrics["error_vel_xy"] += torch.norm(self.vel_command_b[:, :2] - actual[:, :2], dim=-1) / steps
    self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - actual[:, 2]) / steps

  def _update_command(self) -> None:
    super()._update_command()
    t, n, dt = self._time(), self.num_envs, self._env.step_dt
    # Up and down.
    down_due = (self.up > 0.5) & (t >= self.down_at)
    going_down = down_due & (self.hand_side == 0) & (self.hand_ramp <= 0.0)
    self.up = torch.where(going_down, torch.zeros_like(self.up), self.up)
    again = self._chance(self.cfg.up_again_prob, n)
    self.up_again_at = torch.where(going_down & again, t + self.cfg.up_again_delay, self.up_again_at)
    self.down_at = torch.where(going_down, torch.full_like(self.down_at, _INF), self.down_at)
    back_up = (self.up < 0.5) & (t >= self.up_again_at)
    self.up = torch.where(back_up, torch.ones_like(self.up), self.up)
    self.up_again_at = torch.where(back_up, torch.full_like(self.up_again_at, _INF), self.up_again_at)
    # Hand cycles (S2): on only while up and settled upright, with the ramp back at zero.
    due = t >= self.hand_next
    settled = (self.up > 0.5) & (nose_angle(self.robot) > math.radians(self.cfg.hand_min_nose_deg))
    turn_on = due & ~down_due & (self.hand_side == 0) & settled & (self.hand_ramp <= 0.0)
    turn_off = (due | down_due) & (self.hand_side > 0)
    side = torch.randint(1, 3, (n,), device=self.device)
    self.hand_side = torch.where(turn_on, side, self.hand_side)
    self.ramp_side = torch.where(turn_on, side, self.ramp_side)
    self.hand_wave = torch.where(turn_on, self._chance(self.cfg.wave_prob, n), self.hand_wave)
    self.hand_xy = torch.where(turn_on.unsqueeze(1), self.robot.data.root_link_pos_w[:, :2], self.hand_xy)
    self.hand_fwd = torch.where(turn_on.unsqueeze(1), heading_frame(self.robot)[0][:, :2], self.hand_fwd)
    self.hand_side = torch.where(turn_off | (self.up < 0.5), torch.zeros_like(self.hand_side), self.hand_side)
    self.hand_next = torch.where(turn_on, t + self._uniform(self.cfg.hand_on_range, n), self.hand_next)
    self.hand_next = torch.where(turn_off, t + self._uniform(self.cfg.hand_off_range, n), self.hand_next)
    # Out: pose and gains together. Back: the arm first, soft, then the gains.
    out = self.hand_side > 0
    back = torch.where(self.hand_wave, 0.0, (self.hand_pose - dt / self.cfg.back_s).clamp(min=0.0))
    self.hand_pose = torch.where(out, 1.0, back)
    target = torch.where(out, 1.0, torch.where(self.hand_pose > 0.0, self.hand_ramp, 0.0))
    ramp_s = torch.where(out, self.cfg.ramp_s, self.cfg.stiffen_s)
    step = dt / torch.where(self.hand_wave, self.cfg.wave_ramp_s, ramp_s)
    self.hand_ramp = self.hand_ramp + torch.clamp(target - self.hand_ramp, -step, step)
    self.ramp_side = torch.where(self.hand_ramp <= 0.0, torch.zeros_like(self.ramp_side), self.ramp_side)
    # cmd_vel only while up with both paws in.
    walking = (self.up > 0.5) & (self.hand_side == 0) & (self.hand_ramp <= 0.0)
    self.vel_command_b *= walking.float().unsqueeze(1)
    self.creep += (dt / self.cfg.creep_tau_s) * (heading_velocity(self.robot) - self.creep)
    cmd = self.vel_command_b
    idle = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) <= STAND_THRESHOLD
    self.still_for = torch.where(settled & idle, self.still_for + dt, torch.zeros_like(self.still_for))
    self.slow_frames = torch.cat([self.slow_frames[:, 1:], self._slow_frame().unsqueeze(1)], dim=1)

  def _debug_vis_impl(self, visualizer) -> None:
    pass  # The base draws body-frame arrows, which point at the sky upright.


@dataclass(kw_only=True)
class UprightCommandCfg(UniformVelocityCommandCfg):
  down_prob: float = 0.0
  down_time_range: tuple[float, float] = (3.0, 10.0)
  down_start_prob: float = 0.0
  up_again_prob: float = 0.0
  up_again_delay: float = 3.0
  hand_prob: float = 0.0
  hand_first_range: tuple[float, float] = (1.0, 4.0)
  hand_on_range: tuple[float, float] = (2.0, 25.0)  # the deploy has no limit: a paw can stay out
  hand_off_range: tuple[float, float] = (2.0, 6.0)
  hand_min_nose_deg: float = 70.0
  ramp_s: float = 0.5
  back_s: float = 1.5
  stiffen_s: float = 1.0
  wave_prob: float = 0.0
  wave_ramp_s: float = 0.8  # the arm goes higher than for a handshake
  creep_tau_s: float = 1.0  # long enough to average out the balance steps
  imu_tilt_deg: float = 1.5
  slow_bins: int = 10
  slow_bin_steps: int = 5  # 0.1 s bins: fine enough to show the 0.4 s steps
  slow_noise: tuple[float, float, float, float] = (0.05, 0.2, 0.01, 1.5)  # the actor's: gravity, gyro, pos, vel

  def build(self, env: ManagerBasedRlEnv) -> UprightCommand:
    return UprightCommand(self, env)


def down_schedule(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, event_name: str, stages: list[dict]
) -> dict[str, torch.Tensor]:
  """From each stage's step on, its settings: the command's (down_prob, down_start_prob,
  up_again_prob) and the starts' partly risen tilts (mid_nose_deg, on the reset event)."""
  del env_ids  # Unused.
  cfg = env.command_manager.get_term(command_name).cfg
  params = env.event_manager.get_term_cfg(event_name).params
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      for key, value in stage.items():
        if key == "mid_nose_deg":
          params[key] = value
        elif key != "step":
          setattr(cfg, key, value)
  return {"down_prob": torch.tensor(cfg.down_prob), "mid_nose_max": torch.tensor(params["mid_nose_deg"][1])}
