"""Script to run every UprightWalk skill back to back in one episode.

eval_upright.py runs each skill on its own; this runs them in a row from four legs, as a
person would, so a skill that leaves the next one worse off shows up. Same setup as the
eval: mirror-averaged, the task's ground, randomised robots, no pushes. Paw sides are random
per robot and phase. Per phase, of the robots that started it: alive, and done as asked (risen,
on four feet and still, or upright with the paw out), then how far it drifted and turned, the
speed it got, rear-foot steps, arm touches, knee touches and the peak current (p95).

Example:
  python scripts/chain_upright.py --task Unitree-Go2-UprightWalk \
    --policy logs/rsl_rl/go2_upright/<run>/model_14000.pt
"""

import json
import math
import sys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

PAW_S = 10.0
BACK_S = 3.5  # the arm's return, then settling
# name, seconds, kind, setting: walk's cmd_vel, paw's handshake or wave.
PHASES = (
  ("rise", 4.0, "rise", None),
  ("walk fwd", 3.0, "walk", (0.4, 0.0, 0.0)),
  ("turn", 2.0, "walk", (0.0, 0.0, 0.8)),
  ("stop", 2.0, "stand", None),
  ("shake held", PAW_S, "paw", "held"),
  ("paw back", BACK_S, "stand", None),
  ("wave", PAW_S, "paw", "wave"),
  ("arm back", BACK_S, "stand", None),
  ("wave again", PAW_S, "paw", "wave"),
  ("arm back 2", BACK_S, "stand", None),
  ("shake free", PAW_S, "paw", "free"),
  ("paw back 2", BACK_S, "stand", None),
  ("walk back", 3.0, "walk", (-0.25, 0.0, 0.0)),
  ("stop 2", 2.0, "stand", None),
  ("come down", 5.0, "down", None),
  ("rise 2", 4.0, "rise", None),
  ("shake let go", PAW_S, "paw", "let_go"),
  ("paw back 3", BACK_S, "stand", None),
  ("shake, down", 6.0, "paw", "held"),
  ("down, paw out", 8.0, "down", None),
  ("rise 3", 4.0, "rise", None),
  ("wave 3", 6.0, "paw", "wave"),
  ("down, arm up", 6.0, "down", None),
  ("rise 4", 4.0, "rise", None),
  ("walk fwd 2", 3.0, "walk", (0.4, 0.0, 0.0)),
  ("stand", 3.0, "stand", None),
)


@dataclass(frozen=True)
class ChainUprightConfig:
  policy: str
  """A model_N.pt from a run of `task`."""
  task: str = "Unitree-Go2-UprightWalk"
  num_envs: int = 512
  device: str | None = None
  json: str | None = None
  """Also write the results (read by scripts/train_upright.py's checks)."""


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.upright.mdp.frames import heading_frame, heading_velocity, nose_angle
  from src.tasks.upright.mdp.power import K_HEAT, bus_current
  from src.tasks.velocity.rl.symmetry import _get_maps

  cfg = tyro.cli(ChainUprightConfig, prog=sys.argv[0])
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  n = cfg.num_envs

  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.episode_length_s = 200.0
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.terrain.terrain_generator.seed = 0
  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(cfg.task)(env, asdict(agent_cfg), device=device)
  base = env.unwrapped
  robot, dt = base.scene["robot"], base.step_dt
  term = base.command_manager.get_term("twist")
  reset_params = base.event_manager.get_term_cfg("reset_base").params
  maps = _get_maps(base)
  obs_perm, obs_sign = maps["actor"]
  act_perm, act_sign = maps["__actions__"]
  rear, front = base.scene["rear_feet_ground"], base.scene["front_feet_ground"]
  arm_sensor, leg_sensor = base.scene["arm_touch"], base.scene["leg_ground_touch"]
  feet = base.scene["feet_ground_contact"]
  hand_act = base.action_manager.get_term("joint_pos")
  hand_act.fix_wave()
  model = base.sim.mj_model
  act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{a}") for a in robot.actuator_names]
  limits = torch.tensor(model.actuator_forcerange[act_ids, 1], device=device, dtype=torch.float32)
  joint_names = list(robot.joint_names)
  act_joint = torch.tensor([joint_names.index(a) for a in robot.actuator_names], device=device)

  term.cfg.resampling_time_range = (1000.0, 1001.0)
  term.cfg.rel_standing_envs = 0.0
  term.cfg.hand_prob = term.cfg.down_prob = term.cfg.down_start_prob = term.cfg.up_again_prob = 0.0
  term.cfg.hand_on_range = (PAW_S, PAW_S + 1e-6)
  term.cfg.hand_off_range = (1000.0, 1001.0)
  hand_act.cfg.hand_let_go_range = (5.0, 5.0 + 1e-6)
  reset_params["upright_fraction"], reset_params["mid_fraction"] = 0.0, 0.0

  peak = torch.zeros(n, device=device)
  sim_step = base.sim.step

  def step_and_record(*args, **kwargs):
    out = sim_step(*args, **kwargs)
    tau, vel = robot.data.actuator_force, robot.data.joint_vel[:, act_joint]
    p = torch.sum(tau * vel, dim=1) + K_HEAT * torch.sum(torch.square(tau / limits), dim=1)
    torch.maximum(peak, bus_current(p), out=peak)
    return out

  def yaw():
    fwd = heading_frame(robot)[0]
    return torch.atan2(fwd[:, 1], fwd[:, 0])

  def q(x, ok):
    v = x[ok & torch.isfinite(x)].cpu().numpy()
    return [float(a) for a in np.percentile(v, [50, 95])] if len(v) else [math.nan, math.nan]

  runner.load(cfg.policy, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  def act(obs):
    flipped = obs.clone()
    flipped["actor"] = obs["actor"][:, obs_perm] * obs_sign
    return 0.5 * (policy(obs) + policy(flipped)[:, act_perm] * act_sign)

  torch.manual_seed(0)
  rows = []
  with torch.no_grad():
    obs, _ = env.reset()
    term.hand_next[:] = math.inf
    base.sim.step = step_and_record
    alive = torch.ones(n, dtype=torch.bool, device=device)
    for index, (name, secs, kind, arg) in enumerate(PHASES):
      cmd = torch.tensor(arg if kind == "walk" else (0.0, 0.0, 0.0), device=device)
      now = term._time()
      if kind == "paw":
        term.cfg.wave_prob = 1.0 if arg == "wave" else 0.0
        hand_act.cfg.hand_free_prob = 1.0 if arg == "free" else 0.0
        hand_act.cfg.hand_let_go_prob = 1.0 if arg == "let_go" else 0.0
        term.hand_next[:] = now
      elif kind == "down":
        term.down_at[:] = now
        term.hand_next[:] = math.inf  # a timer left running would start a stray cycle after the rise
      elif kind == "rise" and index > 0:
        term.up_again_at[:] = now
      start = alive.clone()
      xy0, yaw_last = robot.data.root_link_pos_w[:, :2].clone(), yaw()
      z = lambda: torch.zeros(n, device=device)
      flag = lambda: torch.zeros(n, dtype=torch.bool, device=device)
      turn, drift, up_run, speed, speed_steps, leg_t, live_t, steps = z(), z(), z(), z(), z(), z(), z(), z()
      risen, arm_hit, front_on, paw_out = flag(), flag(), flag(), flag()
      in_band = alive.clone()
      peak.zero_()
      for i in range(int(round(secs / dt))):
        term.vel_command_b[:] = cmd
        obs, _, dones, _ = env.step(act(obs))
        alive &= ~dones.bool()
        nose = torch.rad2deg(nose_angle(robot))
        yaw_now = yaw()
        turn += ((yaw_now - yaw_last + math.pi) % (2 * math.pi) - math.pi) * alive.float()
        yaw_last = yaw_now
        d = torch.norm(robot.data.root_link_pos_w[:, :2] - xy0, dim=1)
        drift = torch.where(alive, torch.maximum(drift, d), drift)
        arm_hit |= alive & (torch.norm(arm_sensor.data.force_history, dim=-1) > 1.0).flatten(1).any(1)
        leg_on = (torch.norm(leg_sensor.data.force_history, dim=-1) > 1.0).flatten(1).any(1)
        leg_t += (alive & leg_on).float() * dt
        live_t += alive.float() * dt
        steps += rear.compute_first_contact(dt=dt).float().sum(1) * alive.float()
        if kind == "rise":
          up_run = torch.where(alive & (nose > 70.0), up_run + dt, torch.zeros_like(up_run))
          risen |= up_run >= 1.0
        elif kind != "down":
          in_band &= alive & (nose > 60.0) & (nose < 120.0)
          front_on |= alive & (front.data.found > 0).any(dim=1)
        if kind == "walk" and (i + 1) * dt > 1.0:
          speed += heading_velocity(robot)[:, 2 if arg[2] else 0] * alive.float()
          speed_steps += alive.float()
        paw_out |= alive & (term.hand_ramp >= 1.0)
      if kind == "down":
        still = torch.abs(robot.data.joint_vel).amax(1) < 0.5
        level = torch.abs(torch.rad2deg(nose_angle(robot))) < 20.0
        done = alive & (feet.data.found > 0).all(dim=1) & level & still
      elif kind == "rise":
        done = alive & risen
      elif kind == "paw":
        done = alive & in_band & paw_out
      else:
        done = alive & in_band
      kept = start & alive
      pct = lambda b: float(100 * (b & start).sum() / start.sum().clamp(min=1))
      rows.append({
        "phase": name, "kind": kind, "robots": int(start.sum()), "alive": pct(alive), "done": pct(done),
        "drift_m": q(drift, kept), "turn_deg": q(torch.rad2deg(turn).abs(), kept),
        "got": float((speed / speed_steps.clamp(min=1))[kept].median()) if kind == "walk" and kept.any() else math.nan,
        "steps_per_s": float(steps[kept].sum() / live_t[kept].sum().clamp(min=1e-6)) if kept.any() else math.nan,
        "arm_touch": pct(arm_hit),
        "leg_touch": float(100 * leg_t[kept].sum() / live_t[kept].sum().clamp(min=1e-6)) if kept.any() else math.nan,
        "front_touch": pct(front_on), "peak_A": q(peak, kept),
      })
    base.sim.step = sim_step
  env.close()

  print(f"{'phase':15s}{'start':>6s}{'alive':>7s}{'done':>7s}{'drift':>6s}{'p95':>6s}{'turn':>6s}{'p95':>5s}{'got':>6s}"
        f"{'steps':>6s}{'arm%':>6s}{'leg%':>6s}{'frnt%':>6s}{'A p95':>6s}")
  for r in rows:
    print(f"{r['phase']:15s}{r['robots']:6d}{r['alive']:7.1f}{r['done']:7.1f}{r['drift_m'][0]:6.2f}{r['drift_m'][1]:6.2f}"
          f"{r['turn_deg'][0]:6.0f}{r['turn_deg'][1]:5.0f}{r['got']:6.2f}{r['steps_per_s']:6.2f}"
          f"{r['arm_touch']:6.1f}{r['leg_touch']:6.1f}{r['front_touch']:6.1f}{r['peak_A'][1]:6.1f}")
  finished = float(100 * alive.float().mean())
  print(f"finished {finished:.1f}%")
  if cfg.json:
    with open(cfg.json, "w") as f:
      json.dump({"finished": finished, "phases": rows}, f, indent=2)


if __name__ == "__main__":
  main()
