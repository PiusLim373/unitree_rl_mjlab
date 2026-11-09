"""UprightWalk's joint position action, with the handshake override.

The policy sets all 12 joint targets. With a paw out (the command term's hand_ramp and
ramp_side), that front leg is taken over as the deploy will: its three actuators blend
from their own gains to soft hand_kp / hand_kd aimed at paw_pose, with gravity
compensation (the leg's bias force as feedforward), so the policy's targets for it stop
counting. Once the paw is fully out, a simulated hand holds it and shakes it up and down;
in some handshakes nobody takes it, in others the hand lets go. Going back, the soft arm
first moves to the rest pose and only then do its gains return (commands.py), so a hand
still on it never meets a stiff arm on the move. A wave is the same override with a moving
target at the leg's own gains, and nobody on the paw. The policy only has to keep its
balance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .frames import heading_frame
from .power import estimated_current, motor_heat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_SIDES = {1: "FL", 2: "FR"}
_KINDS = ("hip", "thigh", "calf")
# The wave the deploy plays: arm high, swinging side to side. Pose in _KINDS order (rad, hip
# out is positive), hip and forearm swings (rad), speed (Hz).
WAVE = {"center": (0.1, -0.4, -1.3), "hip_amp": 0.45, "calf_amp": 0.0, "freq": 1.2}


class HandJointPositionAction(JointPositionAction):
  cfg: HandJointPositionActionCfg

  def __init__(self, cfg: HandJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    dev, n = self.device, self.num_envs
    entity = self._entity
    names = self._target_names
    self.cols, self.dofs, self.kinds, self.signs, paw = {}, {}, {}, {}, {}
    for side, leg in _SIDES.items():
      cols = [i for i, nm in enumerate(names) if nm.startswith(f"{leg}_")]
      self.cols[side] = torch.tensor(cols, device=dev)
      joint_ids = self._target_ids[cols]
      self.dofs[side] = entity.indexing.joint_v_adr[joint_ids].long()
      # Hip abduction mirrors with its sign flipped on the right.
      sign = {"hip": 1.0 if leg == "FL" else -1.0, "thigh": 1.0, "calf": 1.0}
      paw[side] = [sign[_kind(names[c])] * cfg.paw_pose[_kind(names[c])] for c in cols]
      self.kinds[side] = torch.tensor([_KINDS.index(_kind(names[c])) for c in cols], device=dev)
      self.signs[side] = torch.tensor([sign[_kind(names[c])] for c in cols], device=dev)
    self.paw = {s: torch.tensor(v, device=dev) for s, v in paw.items()}
    # Per actuator group: its columns among the targets, default gains, hand gains.
    self.groups = []
    for actuator in entity.actuators:
      if getattr(actuator, "stiffness", None) is None:
        continue
      act_joints = actuator.target_ids.tolist()
      kinds = [_kind(entity.joint_names[j]) for j in act_joints]
      hand_kp = torch.tensor([cfg.hand_kp[k] for k in kinds], device=dev)
      hand_kd = torch.tensor([cfg.hand_kd[k] for k in kinds], device=dev)
      legs = [entity.joint_names[j][:2] for j in act_joints]
      masks = {s: torch.tensor([lg == leg for lg in legs], device=dev, dtype=torch.float32) for s, leg in _SIDES.items()}
      self.groups.append((actuator, hand_kp, hand_kd, masks))
    self.calf_bodies = [entity.find_bodies(f"{leg}_calf")[0][0] for leg in _SIDES.values()]
    self.amp = torch.zeros(n, device=dev)
    self.freq = torch.zeros(n, device=dev)
    self.phase0 = torch.zeros(n, device=dev)
    self.hand_stiffness = torch.zeros(n, device=dev)
    self.hand_offset = torch.zeros(n, 3, device=dev)
    self.anchor = torch.zeros(n, 3, device=dev)
    self.held = torch.zeros(n, dtype=torch.bool, device=dev)
    self.let_go_at = torch.zeros(n, device=dev)
    self.release = torch.zeros(n, device=dev)
    self.last_side = torch.zeros(n, dtype=torch.long, device=dev)
    self.wave_center = torch.zeros(n, len(_KINDS), device=dev)
    self.wave_amp = torch.zeros(n, 2, device=dev)  # hip, forearm
    self.wave_freq = torch.zeros(n, device=dev)
    self.wave_lead = torch.zeros(n, device=dev)
    self.wave_t0 = torch.zeros(n, device=dev)
    self.wave_pose = torch.zeros(n, len(_KINDS), device=dev)
    env.paw_force_h = torch.zeros(n, 3, device=dev)
    # Highest over this step's physics substeps: a new target's torque spike is over by its
    # last substep, the only one a reward sees.
    self.peak_load = torch.zeros(n, device=dev)
    self.peak_amps = torch.zeros(n, device=dev)

  def _hand(self) -> tuple[torch.Tensor, torch.Tensor]:
    term = self._env.command_manager.get_term(self.cfg.command_name)
    return term.hand_ramp, term.ramp_side

  def _waving(self) -> torch.Tensor:
    return self._env.command_manager.get_term(self.cfg.command_name).hand_wave

  def fix_wave(self) -> None:
    """Every wave from now on is the deploy's (the eval and the clips)."""
    fixed = lambda v: (v, v + 1e-6)
    self.cfg.wave_center_range = tuple(fixed(v) for v in WAVE["center"])
    self.cfg.wave_hip_amp_range = fixed(WAVE["hip_amp"])
    self.cfg.wave_calf_amp_range = fixed(WAVE["calf_amp"])
    self.cfg.wave_freq_range = fixed(WAVE["freq"])

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    ramp, side = self._hand()
    soft = ramp * (~self._waving()).float()
    for actuator, hand_kp, hand_kd, masks in self.groups:
      w = sum((soft * (side == s).float()).unsqueeze(1) * m for s, m in masks.items())
      kp = actuator.default_stiffness * (1.0 - w) + hand_kp * w
      kd = actuator.default_damping * (1.0 - w) + hand_kd * w
      actuator.set_gains(slice(None), kp, kd)
    self._shake(ramp, side)
    self.peak_load.zero_()
    self.peak_amps.zero_()

  def _shake(self, ramp: torch.Tensor, side: torch.Tensor) -> None:
    """A person's hand on the paw, while it is fully out: a spring to where it took the paw
    (a hand stays put, so it doesn't drag the robot along), plus an up/down shake."""
    n, dev = self.num_envs, self.device
    new = (side > 0) & (self.last_side == 0)
    self.last_side = side.clone()
    rnd = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
    self.amp = torch.where(new, rnd(*self.cfg.shake_amp_range), self.amp)
    self.freq = torch.where(new, rnd(*self.cfg.shake_freq_range), self.freq)
    self.phase0 = torch.where(new, rnd(0.0, 2 * math.pi), self.phase0)
    self.hand_stiffness = torch.where(new, rnd(*self.cfg.hand_stiffness_range), self.hand_stiffness)
    offset = torch.stack([rnd(*r) for r in self.cfg.hand_offset_range], dim=1)
    self.hand_offset = torch.where(new.unsqueeze(1), offset, self.hand_offset)
    t = self._env.episode_length_buf.float() * self._env.step_dt
    self._wave(new, t)
    roll = rnd(0.0, 1.0)
    let_go = torch.where(roll < self.cfg.hand_let_go_prob, t + rnd(*self.cfg.hand_let_go_range), math.inf)
    let_go = torch.where(roll >= 1.0 - self.cfg.hand_free_prob, t, let_go)
    self.let_go_at = torch.where(new, let_go, self.let_go_at)
    self.release = torch.where(new, rnd(*self.cfg.hand_release_range), self.release)
    on = (ramp >= 1.0) & (side > 0)
    fwd, left = heading_frame(self._entity)
    rows, leg = torch.arange(n, device=dev), (side - 1).clamp(min=0)
    paw = self._entity.data.body_link_pos_w[:, self.calf_bodies][rows, leg]
    paw_vel = self._entity.data.body_link_lin_vel_w[:, self.calf_bodies][rows, leg]
    anchor = paw + self.hand_offset[:, :1] * fwd + self.hand_offset[:, 1:2] * left
    anchor[:, 2] += self.hand_offset[:, 2]
    self.anchor = torch.where((on & ~self.held).unsqueeze(1), anchor, self.anchor)
    self.held = on
    force_w = self.hand_stiffness.unsqueeze(1) * (self.anchor - paw) - self.cfg.hand_damping * paw_vel
    force_w[:, 2] += self.amp * torch.sin(2 * math.pi * self.freq * t + self.phase0)
    size = torch.norm(force_w, dim=1, keepdim=True).clamp(min=1e-6)
    going_back = self._env.command_manager.get_term(self.cfg.command_name).hand_pose <= self.release
    holding = on & (t < self.let_go_at) & ~self._waving() & ~going_back
    force_w = force_w * (self.cfg.hand_max_force / size).clamp(max=1.0) * holding.float().unsqueeze(1)
    self._env.paw_force_h = torch.stack(
      [torch.sum(force_w * fwd, dim=1), torch.sum(force_w * left, dim=1), force_w[:, 2]], dim=1
    )
    forces = torch.zeros(n, len(self.calf_bodies), 3, device=dev)
    for i, s in enumerate(_SIDES):
      forces[:, i] = force_w * (side == s).float().unsqueeze(1)
    self._entity.write_external_wrench_to_sim(forces, torch.zeros_like(forces), body_ids=self.calf_bodies)

  def _wave(self, new: torch.Tensor, t: torch.Tensor) -> None:
    """The wave's target: each one a new arm pose, swing and speed, with a sine on the hip
    and one on the forearm."""
    n, dev = self.num_envs, self.device
    rnd = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
    center = torch.stack([rnd(*r) for r in self.cfg.wave_center_range], dim=1)
    amp = torch.stack([rnd(*self.cfg.wave_hip_amp_range), rnd(*self.cfg.wave_calf_amp_range)], dim=1)
    self.wave_center = torch.where(new.unsqueeze(1), center, self.wave_center)
    self.wave_amp = torch.where(new.unsqueeze(1), amp, self.wave_amp)
    self.wave_freq = torch.where(new, rnd(*self.cfg.wave_freq_range), self.wave_freq)
    self.wave_lead = torch.where(new, rnd(0.0, 2 * math.pi), self.wave_lead)
    self.wave_t0 = torch.where(new, t, self.wave_t0)
    angle = 2 * math.pi * self.wave_freq * (t - self.wave_t0)
    self.wave_pose = self.wave_center.clone()
    self.wave_pose[:, 0] += self.wave_amp[:, 0] * torch.sin(angle)
    self.wave_pose[:, 2] += self.wave_amp[:, 1] * torch.sin(angle + self.wave_lead)

  def apply_actions(self) -> None:
    self.peak_load = torch.maximum(self.peak_load, motor_heat(self._env))
    self.peak_amps = torch.maximum(self.peak_amps, estimated_current(self._env))
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    effort = torch.zeros_like(target)
    ramp, side = self._hand()
    waving = self._waving().unsqueeze(1)
    out = self._env.command_manager.get_term(self.cfg.command_name).hand_pose.unsqueeze(1)
    bias = self._env.sim.data.qfrc_bias
    for s, cols in self.cols.items():
      w = (ramp * (side == s).float()).unsqueeze(1)
      rest = self._offset[:, cols]
      pose = torch.where(waving, self.wave_pose[:, self.kinds[s]] * self.signs[s], rest + out * (self.paw[s] - rest))
      target[:, cols] = target[:, cols] * (1.0 - w) + pose * w
      effort[:, cols] = bias[:, self.dofs[s]] * w
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)


def _kind(joint_name: str) -> str:
  return re.search(r"_(hip|thigh|calf)_joint$", joint_name).group(1)


@dataclass(kw_only=True)
class HandJointPositionActionCfg(JointPositionActionCfg):
  command_name: str = "twist"
  # Paw out at shoulder height, the hip turned inward (negative = toward the centre line on
  # either side) so the paw is in front of the chest, not out to the side.
  paw_pose: dict[str, float] = field(default_factory=lambda: {"hip": -0.35, "thigh": 0.5, "calf": -1.0})
  # Stiffer at the hip: sideways pulls took the soft abduction to its limit.
  hand_kp: dict[str, float] = field(default_factory=lambda: {"hip": 10.0, "thigh": 6.0, "calf": 10.0})
  hand_kd: dict[str, float] = field(default_factory=lambda: {"hip": 0.4, "thigh": 0.4, "calf": 0.6})
  shake_amp_range: tuple[float, float] = (0.0, 15.0)  # N, up/down
  shake_freq_range: tuple[float, float] = (0.8, 2.5)  # Hz
  # The hand: a spring (N/m) and damper (N s/m) to where it took the paw, shifted by this
  # much (m; forward, left, up in the heading frame), at most hand_max_force (N).
  hand_stiffness_range: tuple[float, float] = (100.0, 300.0)
  hand_damping: float = 5.0
  hand_offset_range: tuple[tuple[float, float], ...] = ((-0.05, 0.05), (-0.03, 0.03), (-0.05, 0.05))
  hand_max_force: float = 25.0
  # Shares of handshakes with nobody on the paw, and with a hand that lets go this long (s)
  # after the paw starts out; the rest hold on.
  hand_free_prob: float = 0.3
  hand_let_go_prob: float = 0.3
  hand_let_go_range: tuple[float, float] = (1.5, 3.5)
  # A hand still on the paw as it goes back lets go at this hand_pose (1 = as it starts).
  hand_release_range: tuple[float, float] = (0.3, 1.0)
  # A wave's pose in _KINDS order (rad, hip out is positive), hip and forearm swings (rad) and
  # speed (Hz). With its swing the calf stays under its limit, and the hip clear of the head
  # and LiDAR, which have no collision geometry here.
  wave_center_range: tuple[tuple[float, float], ...] = ((0.0, 0.3), (-0.7, 0.5), (-1.8, -1.25))
  wave_hip_amp_range: tuple[float, float] = (0.0, 0.5)
  wave_calf_amp_range: tuple[float, float] = (0.0, 0.4)
  wave_freq_range: tuple[float, float] = (0.5, 2.5)

  def build(self, env: ManagerBasedRlEnv) -> HandJointPositionAction:
    return HandJointPositionAction(self, env)
