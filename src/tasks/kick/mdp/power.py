"""Battery current for the flips (FrontFlip multistage v4).

The real Go2 cut all motor power in the front flip's launch: ~200 A (the 140 A sensor
reads flat; the bus sagged to ~19 V), while the side flip (~50 A) was fine. The sim's
motors give at most their limit, but a far-off PD target makes the real ones pull more:
v3's front flip asked up to ~2.6x and they gave up to ~120% of the sim's limits. So the
current here is UprightWalk's real-robot fit (upright/mdp/power.py) on the torque the real
motors would give: the PD request, capped at over_limit x the sim's speed-dependent limit
(the same as the delivered torque while nothing saturates). Averaged over each control
step's physics substeps (the bus smooths short spikes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import DcMotorActuator
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from src.tasks.upright.mdp.power import K_HEAT, bus_current

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PowerJointPositionAction(JointPositionAction):
  """JointPositionAction that also estimates the bus current at every physics substep.

  step_amps: the mean over the step's substeps so far (A); step_ask: the largest PD
  request over a motor's torque limit in them (x the limit)."""

  cfg: PowerJointPositionActionCfg

  def __init__(self, cfg: PowerJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    import mujoco

    entity = self._entity
    model = env.sim.mj_model
    names = list(entity.actuator_names)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{cfg.entity_name}/{a}") for a in names]
    self.limits = torch.tensor(model.actuator_forcerange[ids, 1], device=self.device, dtype=torch.float32)
    # The DC curve per actuator (others: their force range at any speed).
    self.sat = self.limits.clone()
    self.top = torch.full_like(self.limits, float("inf"))
    for group in entity.actuators:
      if isinstance(group, DcMotorActuator):
        self.sat[group.ctrl_ids] = group.cfg.saturation_effort
        self.top[group.ctrl_ids] = group.cfg.velocity_limit
    joints = list(entity.joint_names)
    self.act_joint = torch.tensor([joints.index(a) for a in names], device=self.device)
    n = self.num_envs
    self.amps_sum = torch.zeros(n, device=self.device)
    self.step_ask = torch.zeros(n, device=self.device)
    self.substeps = 0

  @property
  def step_amps(self) -> torch.Tensor:
    return self.amps_sum / max(self.substeps, 1)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self.amps_sum.zero_()
    self.step_ask.zero_()
    self.substeps = 0

  def apply_actions(self) -> None:
    super().apply_actions()
    data = self._entity.data
    # The PD law the actuators run next, from this substep's targets and state.
    request = torch.zeros(self.num_envs, len(self.limits), device=self.device)
    for group in self._entity.actuators:
      stiffness = getattr(group, "stiffness", None)
      if stiffness is None:
        continue
      j, c = group.target_ids, group.ctrl_ids
      err = data.joint_pos_target[:, j] - data.joint_pos[:, j]
      rate = data.joint_vel_target[:, j] - data.joint_vel[:, j]
      request[:, c] = stiffness * err + group.damping * rate + data.joint_effort_target[:, j]
    vel = data.joint_vel[:, self.act_joint]
    reach = self.top * (1 + self.limits / self.sat)
    v = torch.clamp(vel, -reach, reach)
    top = torch.clamp(self.sat * (1 - v / self.top), max=self.limits)
    bottom = torch.clamp(self.sat * (-1 - v / self.top), min=-self.limits)
    over = self.cfg.over_limit
    tau = torch.clamp(request, over * bottom, over * top)
    p = torch.sum(tau * vel, dim=1) + K_HEAT * torch.sum(torch.square(tau / self.limits), dim=1)
    self.amps_sum += bus_current(p)
    self.step_ask = torch.maximum(self.step_ask, (torch.abs(request) / self.limits).amax(dim=1))
    self.substeps += 1


@dataclass(kw_only=True)
class PowerJointPositionActionCfg(JointPositionActionCfg):
  # The share of the sim's torque limit the real motors give when asked for more (v3's
  # front flip on the robot: thighs 122%, calves 116%).
  over_limit: float = 1.2

  def build(self, env: ManagerBasedRlEnv) -> PowerJointPositionAction:
    return PowerJointPositionAction(self, env)


def _power(env: ManagerBasedRlEnv, action_name: str) -> PowerJointPositionAction:
  return env.action_manager.get_term(action_name)


def flip_current_excess(env: ManagerBasedRlEnv, budget: float, action_name: str = "joint_pos") -> torch.Tensor:
  """The step's estimated bus current above `budget` (A), as a share of it."""
  return torch.clamp(_power(env, action_name).step_amps - budget, min=0.0) / budget


def power_cut(env: ManagerBasedRlEnv, limit: float, action_name: str = "joint_pos") -> torch.Tensor:
  """Termination: the step's estimated bus current over `limit` (A), where the real robot
  would lose motor power."""
  return _power(env, action_name).step_amps > limit
