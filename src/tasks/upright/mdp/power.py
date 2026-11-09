"""Whole-robot battery current from joint torques and speeds.

The real Go2 cuts all motor power when the bus current gets too high (the front flip
at launch). Per-motor limits don't bound the total, so UprightWalk charges the estimated
bus current. The model is scripts/eval_power.py's real-robot fit (tools/flip_power):

  P_elec = sum(tau * w) + K_HEAT * sum((tau / tau_max)^2)    (W)

with the bus sagging R_BUS per amp from V_REST.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# As scripts/eval_power.py (real Go2 fit).
K_HEAT = 341.0  # W per motor at full torque
R_BUS = 0.054  # ohm
V_REST = 30.0  # V


def _limits(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
  """Each actuator's torque limit (actuator order), cached on the env."""
  limits = getattr(env, "_upright_torque_limits", None)
  if limits is None:
    import mujoco

    model = env.sim.mj_model
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{a}") for a in asset.actuator_names]
    limits = torch.tensor(model.actuator_forcerange[ids, 1], device=env.device, dtype=torch.float32)
    env._upright_torque_limits = limits
  return limits


def _action(env: ManagerBasedRlEnv):
  """The action term, which tracks the substeps' peaks (actions.py)."""
  return env.action_manager.get_term("joint_pos")


def _act_joints(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
  ids = getattr(env, "_upright_act_joints", None)
  if ids is None:
    names = list(asset.joint_names)
    ids = torch.tensor([names.index(a) for a in asset.actuator_names], device=env.device)
    env._upright_act_joints = ids
  return ids


def bus_current(p: torch.Tensor) -> torch.Tensor:
  """Bus current (A) for electrical power p (W), with the sag; capped where the bus gives out."""
  disc = torch.clamp(V_REST**2 - 4 * R_BUS * p, min=0.0)
  return (V_REST - torch.sqrt(disc)) / (2 * R_BUS)


def estimated_current(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Estimated bus current (A) now, per env."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force
  vel = asset.data.joint_vel[:, _act_joints(env, asset)]
  p = torch.sum(tau * vel, dim=1) + K_HEAT * torch.sum(torch.square(tau / _limits(env, asset)), dim=1)
  return bus_current(p)


def bus_current_excess(
  env: ManagerBasedRlEnv, budget: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Estimated bus current above `budget` (A), as a share of it, at the step's highest."""
  amps = torch.maximum(_action(env).peak_amps, estimated_current(env, asset_cfg))
  return torch.clamp(amps - budget, min=0.0) / budget


def requested_torque_excess(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Sum over motors of (PD request / limit - 1)^2 above the limit.

  The sim delivers at most the limit, so delivered-torque penalties never see a PD
  asking far beyond it; the real motors then pull ~115% (the front flip's launch).
  """
  asset: Entity = env.scene[asset_cfg.name]
  request = torch.zeros_like(asset.data.actuator_force)
  for actuator in asset.actuators:
    stiffness = getattr(actuator, "stiffness", None)
    if stiffness is None:
      continue
    j, c = actuator.target_ids, actuator.ctrl_ids
    err = asset.data.joint_pos_target[:, j] - asset.data.joint_pos[:, j]
    rate = asset.data.joint_vel_target[:, j] - asset.data.joint_vel[:, j]
    request[:, c] = stiffness * err + actuator.damping * rate + asset.data.joint_effort_target[:, j]
  over = torch.clamp(torch.abs(request) / _limits(env, asset) - 1.0, min=0.0)
  return torch.sum(torch.square(over), dim=1)


def motor_heat(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Sum over asset_cfg's actuators of (torque / limit)^2: motors at full torque."""
  asset: Entity = env.scene[asset_cfg.name]
  share = asset.data.actuator_force / _limits(env, asset)
  ids = getattr(asset_cfg, "actuator_ids", slice(None))
  return torch.sum(torch.square(share[:, ids]), dim=1)


def motor_load_excess(
  env: ManagerBasedRlEnv, start: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """All motors' load (motor_heat, eval_upright's motors_full) above `start`, at the step's
  highest."""
  load = torch.maximum(_action(env).peak_load, motor_heat(env, asset_cfg))
  return torch.clamp(load - start, min=0.0)
