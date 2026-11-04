"""Script to estimate a skill policy's battery current in sim, from its joint torques and speeds.

Runs one skill per robot (the training episode, standing starts, flat ground: the real robot
tests were on a flat floor) and records every motor's torque and speed at each physics
substep. The electrical power model was fitted on the real Go2 (tools/flip_power, rosbags of
SideFlip, FrontFlip, DonkeyKick, QuickBuck on 2026-11-04; the board's motor-supply current x
voltage against the robot's tau_est and dq):

  P_elec = sum(tau * w) + K_HEAT * sum((tau / tau_max)^2)    (W; tau N·m, w rad/s)

The mechanical power passes through, and each motor at its full torque heats ~K_HEAT W
whatever its speed (the calf's gearing gives it twice the torque for the same current, so
the heat goes with the share of each motor's limit). sum((tau / tau_max)^2) reads as the
number of motors at full torque. The bus sags by R_BUS per amp, so the current is the root
of R I^2 - V0 I + P = 0. Reports, over the robots: the peak current, electrical and
mechanical power, sum |tau| and motors at full torque, the time above a current, and each
motor's peak torque. Saves the per-substep means and
the per-robot peaks to --out (.npz).

Example:
  python scripts/eval_power.py --task Unitree-Go2-SideFlip --mode mirrored \
    --policy logs/rsl_rl/go2_sideflip/<run>/model_4000.pt --out /tmp/sideflip_power.npz
"""

import sys
from dataclasses import asdict, dataclass
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# Real Go2 fit (tools/flip_power): W per motor at full torque, bus resistance (ohm).
K_HEAT = 341.0
R_BUS = 0.054
V_REST = 30.0  # V, resting bus voltage around 80% charge
CURRENT_MARKS = (50.0, 100.0)  # A: the real side flip peaked ~50 A; the front flip cut out above ~140 A


@dataclass(frozen=True)
class EvalPowerConfig:
  task: str
  policy: str
  """model_N.pt from a run of `task`."""
  mode: Literal["plain", "mirrored", "averaged"] = "averaged"
  """plain: the policy as trained; mirrored: its mirror twin (SideFlip's right flip,
  policy_right.onnx); averaged: the policy and its twin averaged (policy.onnx of the kicks,
  FrontFlip and BackFlip)."""
  num_envs: int = 512
  flat: bool = True
  skip: float = 0.02
  """s after the reset left out of the peaks: the robot settles from the reset pose."""
  out: str | None = None
  device: str | None = None


def bus_current(p: torch.Tensor) -> torch.Tensor:
  """Bus current (A) for electrical power p (W) with the sag; inf past the most the bus gives."""
  disc = V_REST**2 - 4 * R_BUS * p
  return torch.where(disc >= 0, (V_REST - torch.sqrt(disc.clamp(min=0))) / (2 * R_BUS), torch.inf)


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.velocity.rl.symmetry import _get_maps

  cfg = tyro.cli(EvalPowerConfig, prog=sys.argv[0])
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  n = cfg.num_envs

  env_cfg = load_env_cfg(cfg.task)
  run_s = env_cfg.episode_length_s
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.events.pop("mid_flip_start", None)  # standing starts only
  env_cfg.episode_length_s = run_s + 0.5  # no time-out resets inside the run
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  tg = env_cfg.scene.terrain.terrain_generator
  tg.seed = 0
  if cfg.flat:
    tg.sub_terrains = {"flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0)}
  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(cfg.task)(env, asdict(agent_cfg), device=device)
  runner.load(cfg.policy, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  base = env.unwrapped
  robot, dt = base.scene["robot"], base.step_dt
  joint_names = list(robot.joint_names)
  act_joint = torch.tensor([joint_names.index(a) for a in robot.actuator_names], device=device)
  maps = _get_maps(base)
  obs_perm, obs_sign = maps["actor"]
  act_perm, act_sign = maps["__actions__"]

  def act(obs):
    if cfg.mode == "plain":
      return policy(obs)
    flipped = obs.clone()
    flipped["actor"] = obs["actor"][:, obs_perm] * obs_sign
    twin = policy(flipped)[:, act_perm] * act_sign
    return twin if cfg.mode == "mirrored" else 0.5 * (policy(obs) + twin)

  # Every physics substep: the torque each motor applied and its speed (actuator order).
  taus, vels = [], []
  sim_step = base.sim.step

  def step_and_record(*args, **kwargs):
    out = sim_step(*args, **kwargs)
    taus.append(robot.data.actuator_force.clone())
    vels.append(robot.data.joint_vel[:, act_joint].clone())
    return out

  torch.manual_seed(0)
  with torch.no_grad():
    obs, _ = env.reset()
    base.sim.step = step_and_record
    alive, alive_steps = torch.ones(n, dtype=torch.bool, device=device), []
    for _ in range(int(round(run_s / dt))):
      alive_steps.append(alive.clone())
      obs, _, dones, _ = env.step(act(obs))
      alive &= ~dones.bool()  # a crash resets the robot: stop counting it
    base.sim.step = sim_step

  sub = len(taus) // len(alive_steps)
  tau, vel = torch.stack(taus), torch.stack(vels)  # (substeps, robots, motors)
  ok = torch.stack(alive_steps).repeat_interleave(sub, 0)
  ok[: int(round(cfg.skip * sub / dt))] = False
  t = (torch.arange(len(taus), device=device) + 1) * dt / sub
  model = base.sim.mj_model
  act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{a}") for a in robot.actuator_names]
  limits = torch.tensor(model.actuator_forcerange[act_ids, 1], device=device, dtype=torch.float32)
  mech = (tau * vel).sum(-1)
  heat = ((tau / limits) ** 2).sum(-1)  # motors at full torque
  p_elec = mech + K_HEAT * heat
  amps = bus_current(p_elec)
  tau_abs = tau.abs().sum(-1)

  def peak(x):  # per robot, over its live substeps
    return torch.where(ok, x, torch.full_like(x, -torch.inf)).amax(0)

  def held(x, s):  # mean over the last s seconds, per substep (the bus smooths short spikes)
    w = max(1, int(round(s * sub / dt)))
    c = torch.cumsum(torch.nan_to_num(x, posinf=1e4), 0)
    return torch.cat([c[:w] / torch.arange(1, w + 1, device=device)[:, None], (c[w:] - c[:-w]) / w])

  def masked_mean(x):  # over the live robots, per substep
    return (torch.where(ok, x, 0).sum(1) / ok.sum(1).clamp(min=1))

  joint_peak = torch.where(ok[..., None], tau.abs(), 0).amax(0) / limits  # (robots, motors)
  over = {a: (ok & (amps > a)).sum(0).float() * dt / sub for a in CURRENT_MARKS}
  q = lambda x: np.percentile(x[torch.isfinite(x)].cpu().numpy(), [50, 95, 100]) if torch.isfinite(x).any() else [np.inf] * 3
  collapsed = (~torch.isfinite(peak(amps))).float().mean().item()

  print(f"\n== {cfg.task} {cfg.mode}: {cfg.policy}")
  print(f"   robots {n}, crashed {(~alive).float().mean().item():.1%}, {'flat' if cfg.flat else 'training terrain'}")
  print("   per robot peak            median      p95      max")
  for label, x, f in (
    ("est. current (A)", peak(amps), "{:8.0f}"),
    ("  held 20 ms (A)", peak(held(amps, 0.02)), "{:8.0f}"),
    ("  held 50 ms (A)", peak(held(amps, 0.05)), "{:8.0f}"),
    ("est. elec power (W)", peak(p_elec), "{:8.0f}"),
    ("mech power Σ τ·ω (W)", peak(mech), "{:8.0f}"),
    ("Σ |τ| (N·m)", peak(tau_abs), "{:8.0f}"),
    ("motors at full torque", peak(heat), "{:8.1f}"),
  ):
    print(f"   {label:22s}" + "".join(f.format(v) for v in q(x)))
  if collapsed:
    print(f"   {collapsed:.1%} of robots ask for more power than the bus can give at any current (collapse)")
  for a, s in over.items():
    print(f"   time above {a:.0f} A: median {1000 * s.median().item():.0f} ms, p95 {1000 * s.quantile(0.95).item():.0f} ms")
  print("   motor            peak |τ| (% of limit): median  p95")
  jp = joint_peak.cpu().numpy()
  for j in np.argsort(-np.median(jp, 0)):
    print(f"   {robot.actuator_names[j]:16s} {100 * np.median(jp[:, j]):5.0f} {100 * np.percentile(jp[:, j], 95):5.0f}")

  if cfg.out:
    np.savez_compressed(
      cfg.out, t=t.cpu().numpy(), peak_amps_20ms=peak(held(amps, 0.02)).cpu().numpy(),
      peak_amps_50ms=peak(held(amps, 0.05)).cpu().numpy(), amps_mean=masked_mean(torch.nan_to_num(amps, posinf=1e4)).cpu().numpy(),
      tau_abs_mean=masked_mean(tau_abs).cpu().numpy(), mech_mean=masked_mean(mech).cpu().numpy(),
      heat_mean=masked_mean(heat).cpu().numpy(), peak_amps=peak(amps).cpu().numpy(),
      peak_tau_abs=peak(tau_abs).cpu().numpy(), peak_heat=peak(heat).cpu().numpy(), peak_mech=peak(mech).cpu().numpy(),
      joint_peak=jp, motors=np.array(robot.actuator_names), crashed=(~alive).cpu().numpy(),
    )
    print(f"   saved {cfg.out}")
  env.close()


if __name__ == "__main__":
  main()
