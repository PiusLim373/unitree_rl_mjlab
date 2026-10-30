"""Left-right mirror augmentation for quadruped velocity policies (rsl_rl symmetry).

The mirror is a reflection across the robot's x-z plane (y -> -y). Vectors flip
their y component and angular velocities their x and z components, in the body
and the world frame alike. Legs swap sides by name (FL<->FR, RL<->RR) and hip
abduction angles change sign; thigh and calf angles keep theirs.

The gait clock (``phase``) moves half a period: ``feet_gait`` ties the diagonal
pairs to the clock (offsets 0/0.5/0.5/0), and mirroring swaps the pairs. With
that, a mirrored sample earns the same reward as the original, so PPO can reuse
the original's advantage for it.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv

_SIDE_SWAP = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}

# Sign patterns for 3-vectors: angular velocities are pseudovectors.
_ANGULAR = [-1.0, 1.0, -1.0]
_LINEAR = [1.0, -1.0, 1.0]

# Index maps per env, built on first use: group -> (perm, sign), "__actions__" too.
_MAPS: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def mirror_quadruped(
  env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Return ``[original; mirrored]`` observations and/or actions (batch doubled).

  Signature of rsl_rl's ``symmetry_cfg["data_augmentation_func"]``. Every term in
  every observation group needs a rule in ``_term_rule``; an unknown term raises.
  """
  maps = _get_maps(env)
  obs_aug = None
  if obs is not None:
    mirrored = obs.clone()
    for group in obs.keys():
      if group not in maps:
        raise KeyError(f"No mirror map for observation group '{group}'.")
      perm, sign = maps[group]
      mirrored[group] = obs[group][:, perm] * sign
    obs_aug = torch.cat([obs, mirrored], dim=0)
  actions_aug = None
  if actions is not None:
    perm, sign = maps["__actions__"]
    actions_aug = torch.cat([actions, actions[:, perm] * sign], dim=0)
  return obs_aug, actions_aug


class SymmetricPolicy(torch.nn.Module):
  """Exportable policy that averages its own answer with its mirror twin's.

  ``forward(x) = 0.5 * (pi(x) + M_a pi(M_o x))``, one batched call of 2. The result
  is exactly left-right symmetric whatever ``pi`` learned, which removes the small
  side lean that mirrored training leaves (v8: 75% of identical robots curved
  right). Wraps the module from rsl_rl's ``as_onnx()``; same input/output names
  and shapes, so the deploy controller and deploy.yaml stay unchanged.
  """

  is_recurrent: bool = False

  def __init__(self, policy: torch.nn.Module, env) -> None:
    super().__init__()
    self.policy = policy
    maps = _get_maps(env)
    obs_perm, obs_sign = maps["actor"]
    act_perm, act_sign = maps["__actions__"]
    if len(obs_perm) != policy.input_size:
      raise ValueError(f"Actor mirror has {len(obs_perm)} inputs, policy has {policy.input_size}.")
    self.register_buffer("obs_perm", obs_perm.cpu())
    self.register_buffer("obs_sign", obs_sign.cpu())
    self.register_buffer("act_perm", act_perm.cpu())
    self.register_buffer("act_sign", act_sign.cpu())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    y = self.policy(torch.cat([x, x[:, self.obs_perm] * self.obs_sign], dim=0))
    return 0.5 * (y[:n] + y[n:][:, self.act_perm] * self.act_sign)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return self.policy.get_dummy_inputs()

  @property
  def input_names(self) -> list[str]:
    return self.policy.input_names

  @property
  def output_names(self) -> list[str]:
    return self.policy.output_names


def _swap_side(name: str) -> str:
  parts = name.split("_")
  swapped = [_SIDE_SWAP.get(p, p) for p in parts]
  if swapped == parts:
    raise ValueError(f"No leg side (FL/FR/RL/RR) in name '{name}'.")
  return "_".join(swapped)


def _swap_perm(names) -> list[int]:
  index = {n: i for i, n in enumerate(names)}
  return [index[_swap_side(n)] for n in names]


def _leg_rule(names) -> tuple[list[int], list[float]]:
  """Joints or actuators: swap sides, negate hip abduction."""
  return _swap_perm(names), [-1.0 if "hip" in n.split("_") else 1.0 for n in names]


def _contact_feet(env: ManagerBasedRlEnv, sensor_name: str) -> list[str]:
  # The contact sensor lists its geoms in model order (see feet_clearance_terrain).
  asset = env.scene["robot"]
  _, geom_names = asset.find_geoms(env.scene[sensor_name].cfg.primary.pattern)
  return list(geom_names)


def _term_rule(
  env: ManagerBasedRlEnv, group: str, term: str
) -> tuple[list[int], list[float]]:
  """Permutation and signs for one history step of an observation term."""
  params = env.observation_manager.get_term_cfg(group, term).params
  if term == "base_ang_vel":
    return [0, 1, 2], _ANGULAR
  if term in ("projected_gravity", "base_lin_vel"):
    return [0, 1, 2], _LINEAR
  if term == "command":  # (vx, vy, wz)
    return [0, 1, 2], [1.0, -1.0, -1.0]
  if term == "phase":  # (sin, cos) of the gait clock, shifted half a period
    return [0, 1], [-1.0, -1.0]
  if term == "kick_phase":  # DonkeyKick timer: time, not a side
    return [0], [1.0]
  if term in ("joint_pos", "joint_vel"):
    return _leg_rule(env.scene["robot"].joint_names)
  if term == "actions":
    return _leg_rule(env.action_manager.get_term("joint_pos").target_names)
  if term == "height_scan":
    sensor = env.scene[params["sensor_name"]]
    offsets, _ = sensor.cfg.pattern.generate_rays(None, "cpu")
    xy = offsets[:, :2]
    target = xy * torch.tensor([1.0, -1.0])
    dist = (target[:, None, :] - xy[None, :, :]).abs().amax(dim=-1)  # exact, unlike cdist
    perm = dist.argmin(dim=1)
    if dist.min(dim=1).values.max() > 1e-4:
      raise ValueError(f"Ray pattern of '{params['sensor_name']}' is not left-right symmetric.")
    return perm.tolist(), [1.0] * len(perm)
  if term == "foot_height":  # one ray sensor per foot, named foot_ray_<foot>
    names = params["sensor_names"]
    return _swap_perm(names), [1.0] * len(names)
  if term in ("foot_air_time", "foot_contact"):
    feet = _contact_feet(env, params["sensor_name"])
    return _swap_perm(feet), [1.0] * len(feet)
  if term == "foot_contact_forces":  # (fx, fy, fz) per foot, sign * log1p(|f|)
    feet = _contact_feet(env, params["sensor_name"])
    foot_perm = _swap_perm(feet)
    perm = [3 * foot_perm[i] + c for i in range(len(feet)) for c in range(3)]
    return perm, [-1.0 if c == 1 else 1.0 for _ in feet for c in range(3)]
  raise KeyError(f"No mirror rule for observation term '{group}/{term}'.")


def _get_maps(env) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  base: ManagerBasedRlEnv = getattr(env, "unwrapped", env)
  key = id(base)
  if key in _MAPS:
    return _MAPS[key]
  om = base.observation_manager
  maps = {}
  for group, terms in om.active_terms.items():
    perm, sign, offset = [], [], 0
    for term, dims in zip(terms, om.group_obs_term_dim[group]):
      total = int(torch.tensor(dims).prod())
      history = max(om.get_term_cfg(group, term).history_length, 1)
      step_perm, step_sign = _term_rule(base, group, term)
      dim = total // history
      if len(step_perm) != dim or dim * history != total:
        raise ValueError(
          f"Mirror rule for '{group}/{term}' has {len(step_perm)} values, term has {dim}."
        )
      # Flattened history is history-major: [step 0 (dim), step 1 (dim), ...].
      for h in range(history):
        perm += [offset + h * dim + p for p in step_perm]
        sign += step_sign
      offset += total
    maps[group] = (
      torch.tensor(perm, device=base.device),
      torch.tensor(sign, device=base.device),
    )
  action_perm, action_sign = _leg_rule(base.action_manager.get_term("joint_pos").target_names)
  maps["__actions__"] = (
    torch.tensor(action_perm, device=base.device),
    torch.tensor(action_sign, device=base.device),
  )
  _MAPS[key] = maps
  return maps
