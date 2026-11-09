"""Script to evaluate UprightWalk policies: walking upright, rising and coming down, the handshake.

Each policy runs mirror-averaged (policy.onnx, what deploys) on the task's ground (flat and
gentle rough), the same seeds for each. Scenarios (--stage picks those trained by then):
  - walk (S1+): upright starts, cmd_vel from the task's ranges for 20 s (some of it zero).
    Up at the end, falls, tracking error per axis while commanded, rear-foot step rate
    walking and standing (zero cmd_vel for over SETTLE_S, so the steps to stop aren't
    counted: it should stand, stepping only for balance), both feet planted while standing,
    alternation while walking (one foot down, both in the air), front-foot touches, how far
    the arms' hips turn in, how high the paws are (from the base) and the rear joints' speed
    (RMS) while standing.
  - rise (S1+): starts on four legs with the up command, then down at 6 s. Risen
    (upright by 3 s and held), time to rise, come down (on four feet, still, by 9 s),
    front-feet impact, falls, drops onto the front feet while up.
  - again (S1+): upright, down at 1.5 s, up again 3 s later, what the robot does after a
    come-down. Risen (held for 1 s by the end), falls.
  - hand (S2): upright, a paw out from 2 s for 10 s with the simulated hand shaking it.
    Upright through it (the paw fully out at some point, no fall), the base drift and turn,
    the rear-foot step rate and the share of the time with a thigh or calf on the ground
    (leg_touch, over 1 N: kneeling).
  - hand_free (S2): the same with nobody holding the paw.
  - hand_let_go (S2): the same with the hand letting go halfway.
  - hand_wave (S2): the same with the deploy's wave, on either arm; paw_sweep_m is how far
    the paw swings side to side from the base.
  - hand_long_free, hand_long_wave (S2): a paw left out for 60 s, as the deploy allows.
  - stand (S1+): upright at zero cmd_vel for 30 s, how far it drifts and turns.
  - held (S1+): each robot keeps one command for 10 s, the speed it gets along it (the
    walk scenario's mean error can't show a direction it barely moves in).
walk, rise and stand also report the share of the up time with a thigh or calf on the ground
(leg_touch: nose up on its knees reads as risen). Every scenario also reports the share of
robots whose arms touched the body or the other legs (arm_touch, over 1 N: a self-collision
on the real robot) and the estimated battery
current (scripts/eval_power.py's real-Go2 model, per physics substep): per-robot peak, held
50 ms peak, motors at full torque.

Example:
  python scripts/eval_upright.py --task Unitree-Go2-UprightWalk-S1 --stage 1 \
    --policies s1=logs/rsl_rl/go2_upright/<run>/model_6000.pt
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

HELD_S = 0.05
# Standing counts from this long after cmd_vel goes to zero (the steps to stop don't).
SETTLE_S = 1.0
# The again scenario: when the down command comes, and the up command after up_again_delay.
AGAIN_DOWN_S = 1.5
AGAIN_UP_S = AGAIN_DOWN_S + 3.0
# The hand_let_go scenario: how long after the paw starts out the hand lets go.
LET_GO_S = 5.0
# The hand_long scenarios: how long the paw stays out.
LONG_HOLD_S = 60.0
# The stand scenario: settling, then standing.
STAND_FROM_S = 3.0
STAND_S = 30.0
# The held scenario: (vx, vy, wz) by name, counted from HELD_FROM_S.
HELD_CMDS = {
  "fwd_025": (0.25, 0.0, 0.0), "fwd_05": (0.5, 0.0, 0.0), "back_025": (-0.25, 0.0, 0.0), "back_05": (-0.5, 0.0, 0.0),
  "side_03": (0.0, 0.3, 0.0), "side_05": (0.0, 0.5, 0.0), "turn_1": (0.0, 0.0, 1.0),
}
HELD_FROM_S = 2.0


@dataclass(frozen=True)
class EvalUprightConfig:
  policies: tuple[str, ...]
  """name=checkpoint pairs (model_N.pt from a run of `task`)."""
  task: str = "Unitree-Go2-UprightWalk"
  stage: int = 2
  """Scenarios trained by this stage: 1 walk, rise and come down, 2 + hand and wave."""
  num_envs: int = 512
  device: str | None = None
  json: str | None = None
  """Also write each policy's headline results (read by scripts/train_upright.py's checks)."""


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.upright.mdp.frames import heading_frame, heading_velocity, nose_angle
  from src.tasks.upright.mdp.power import K_HEAT, bus_current
  from src.tasks.velocity.rl.symmetry import _get_maps

  cfg = tyro.cli(EvalUprightConfig, prog=sys.argv[0])
  policies = dict(p.split("=", 1) for p in cfg.policies)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  n = cfg.num_envs

  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.episode_length_s = 80.0  # no time-out inside a scenario
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
  arm_sensor = base.scene["arm_touch"]
  leg_sensor = base.scene["leg_ground_touch"]
  model = base.sim.mj_model
  act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{a}") for a in robot.actuator_names]
  limits = torch.tensor(model.actuator_forcerange[act_ids, 1], device=device, dtype=torch.float32)
  joint_names = list(robot.joint_names)
  act_joint = torch.tensor([joint_names.index(a) for a in robot.actuator_names], device=device)
  feet_sites, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  paw_sites = feet_sites[:2]
  arm_hips = [joint_names.index(j) for j in ("FL_hip_joint", "FR_hip_joint")]
  rear_joints = [i for i, j in enumerate(joint_names) if j.startswith("R")]
  hand_act = base.action_manager.get_term("joint_pos")
  hand_act.fix_wave()
  held_cmd = torch.tensor(list(HELD_CMDS.values()), device=device)[torch.arange(n, device=device) % len(HELD_CMDS)]

  def yaw():
    fwd = heading_frame(robot)[0]
    return torch.atan2(fwd[:, 1], fwd[:, 0])

  # Estimated bus current at every physics substep.
  amps_log = []
  sim_step = base.sim.step

  def step_and_record(*args, **kwargs):
    out = sim_step(*args, **kwargs)
    tau, vel = robot.data.actuator_force, robot.data.joint_vel[:, act_joint]
    p = torch.sum(tau * vel, dim=1) + K_HEAT * torch.sum(torch.square(tau / limits), dim=1)
    amps_log.append((bus_current(p), torch.sum(torch.square(tau / limits), dim=1)))
    return out

  def power(alive_steps):
    sub = len(amps_log) // max(len(alive_steps), 1)
    amps = torch.stack([a for a, _ in amps_log])
    heat = torch.stack([h for _, h in amps_log])
    ok = torch.stack(alive_steps).repeat_interleave(sub, 0)[: len(amps)]
    ok[: max(1, int(round(0.02 * sub / dt)))] = False  # settling from the reset pose
    w = max(1, int(round(HELD_S * sub / dt)))
    # Held means over live substeps only (the reset spike and dead robots read 0).
    c = torch.cumsum(torch.where(ok, amps, torch.zeros_like(amps)), 0)
    held = torch.cat([c[:w] / torch.arange(1, w + 1, device=device)[:, None], (c[w:] - c[:-w]) / w])
    pk = lambda x: torch.where(ok, x, torch.full_like(x, -torch.inf)).amax(0)
    return pk(amps), pk(held), pk(heat)

  def schedule(name):
    """Scenario settings, applied before the reset (command cfg, starts) and after it (timers)."""
    term.cfg.hand_prob = term.cfg.down_prob = term.cfg.down_start_prob = 0.0
    term.cfg.up_again_prob = 1.0 if name == "again" else 0.0
    term.cfg.wave_prob = 1.0 if "wave" in name else 0.0
    hand_act.cfg.hand_free_prob = 1.0 if "free" in name else 0.0
    hand_act.cfg.hand_let_go_prob = 1.0 if name == "hand_let_go" else 0.0
    hand_act.cfg.hand_let_go_range = (LET_GO_S, LET_GO_S + 1e-6)
    term.cfg.rel_standing_envs = {"walk": 0.3, "held": 0.0}.get(name, 1.0)
    reset_params["upright_fraction"] = 0.0 if name == "rise" else 1.0
    reset_params["mid_fraction"] = 0.0

  def after_reset(name):
    if name == "rise":
      term.down_at[:] = 6.0
    if name == "again":
      term.down_at[:] = AGAIN_DOWN_S
    if name.startswith("hand"):
      term.hand_next[:] = 2.0
      on_s = LONG_HOLD_S if "long" in name else 10.0
      term.cfg.hand_on_range = (on_s, on_s + 1e-6)

  scenarios = {"walk": 20.0, "rise": 10.0, "stand": STAND_FROM_S + STAND_S, "held": 10.0, "again": AGAIN_UP_S + 3.0}
  if cfg.stage >= 2:
    scenarios |= {"hand": 12.0, "hand_free": 12.0, "hand_let_go": 12.0, "hand_wave": 12.0}
    scenarios |= {"hand_long_free": 2.0 + LONG_HOLD_S, "hand_long_wave": 2.0 + LONG_HOLD_S}

  results = {}
  for name, ckpt in policies.items():
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    def act(obs):
      flipped = obs.clone()
      flipped["actor"] = obs["actor"][:, obs_perm] * obs_sign
      return 0.5 * (policy(obs) + policy(flipped)[:, act_perm] * act_sign)

    res = {}
    for sc, run_s in scenarios.items():
      schedule(sc)
      torch.manual_seed(0)
      amps_log.clear()
      with torch.no_grad():
        obs, _ = env.reset()
        after_reset(sc)
        base.sim.step = step_and_record
        alive = torch.ones(n, dtype=torch.bool, device=device)
        alive_steps = []
        z = lambda: torch.zeros(n, device=device)
        err = torch.zeros(n, 3, device=device)
        err_steps = z()
        one_foot, both_air, steps_up, touchdowns = z(), z(), z(), z()
        front_touch = torch.zeros(n, dtype=torch.bool, device=device)
        stand_t, stand_steps, stand_planted, stand_for, stand_hip, stand_speed = z(), z(), z(), z(), z(), z()
        stand_paw = z()
        risen_at = torch.full((n,), float("nan"), device=device)
        upright_run = z()
        down_ok = torch.zeros(n, dtype=torch.bool, device=device)
        front_impact = z()
        dropped = torch.zeros(n, dtype=torch.bool, device=device)
        hand_ok = torch.ones(n, dtype=torch.bool, device=device)
        hand_seen = torch.zeros(n, dtype=torch.bool, device=device)
        xy_hand0 = yaw_last = None
        drift, turn, leg_t, up_t = z(), z(), z(), z()
        got, got_steps = torch.zeros(n, 3, device=device), z()
        paw_lo, paw_hi = torch.full((n,), math.inf, device=device), torch.full((n,), -math.inf, device=device)
        arm_hit = torch.zeros(n, dtype=torch.bool, device=device)
        for i in range(int(round(run_s / dt))):
          alive_steps.append(alive.clone())
          if sc == "held":
            term.vel_command_b[:] = held_cmd
          obs, _, dones, _ = env.step(act(obs))
          alive &= ~dones.bool()
          t = (i + 1) * dt
          if not alive.any():
            break
          nose = torch.rad2deg(nose_angle(robot))
          up_cmd = term.up > 0.5
          arm_hit |= alive & (torch.norm(arm_sensor.data.force_history, dim=-1) > 1.0).flatten(1).any(1)
          r_down = (rear.data.found > 0).float()
          f_down = (front.data.found > 0).any(dim=1)
          leg_on = alive & (torch.norm(leg_sensor.data.force_history, dim=-1) > 1.0).flatten(1).any(1)
          if sc in ("walk", "rise", "stand"):
            leg_t += (leg_on & up_cmd).float() * dt
            up_t += (alive & up_cmd).float() * dt
          if sc == "walk":
            cmd = term.vel_command_b
            active = alive & (torch.norm(cmd, dim=1) > 0.1)
            err += torch.abs(cmd - heading_velocity(robot)) * active.float().unsqueeze(1)
            err_steps += active.float()
            walking = alive & ~(torch.norm(cmd, dim=1) <= 0.1)
            standing = alive & (torch.norm(cmd, dim=1) <= 0.1)
            td = rear.compute_first_contact(dt=dt).float().sum(1)
            one_foot += (walking & (r_down.sum(1) == 1)).float()
            both_air += (walking & (r_down.sum(1) == 0)).float()
            steps_up += walking.float()
            touchdowns += td * walking.float()
            stand_for = torch.where(standing, stand_for + dt, torch.zeros_like(stand_for))
            settled = standing & (stand_for > SETTLE_S)
            stand_t += settled.float() * dt
            stand_steps += td * settled.float()
            stand_planted += (settled & (r_down.sum(1) == 2)).float() * dt
            stand_hip += torch.abs(robot.data.joint_pos[:, arm_hips]).mean(1) * settled.float() * dt
            stand_speed += torch.square(robot.data.joint_vel[:, rear_joints]).mean(1) * settled.float() * dt
            paw_height = robot.data.site_pos_w[:, paw_sites, 2] - robot.data.root_link_pos_w[:, 2:3]
            stand_paw += paw_height.mean(1) * settled.float() * dt
            front_touch |= alive & f_down
          if sc == "rise" or (sc == "again" and t > AGAIN_UP_S):
            upright_now = alive & up_cmd & (nose > 70.0)
            upright_run = torch.where(upright_now, upright_run + dt, torch.zeros_like(upright_run))
            risen_at = torch.where(torch.isnan(risen_at) & (upright_run >= 1.0), torch.full_like(risen_at, t - 1.0), risen_at)
          if sc == "rise":
            dropped |= alive & up_cmd & ~torch.isnan(risen_at) & f_down
            if t > 6.0:
              f = torch.norm(front.data.force_history, dim=-1).amax(dim=(1, 2))
              front_impact = torch.where(alive, torch.maximum(front_impact, f), front_impact)
            if abs(t - 9.0) < dt / 2:
              all_down = (base.scene["feet_ground_contact"].data.found > 0).all(dim=1)
              still = torch.abs(robot.data.joint_vel).amax(1) < 0.5
              down_ok = alive & all_down & (torch.abs(nose) < 20.0) & still
          if sc == "held" and t >= HELD_FROM_S:
            got += heading_velocity(robot) * alive.float().unsqueeze(1)
            got_steps += alive.float()
          if sc == "stand" or sc.startswith("hand"):
            if xy_hand0 is None and t >= (STAND_FROM_S if sc == "stand" else 2.0):
              xy_hand0, yaw_last = robot.data.root_link_pos_w[:, :2].clone(), yaw()
            if xy_hand0 is not None:
              d = torch.norm(robot.data.root_link_pos_w[:, :2] - xy_hand0, dim=1)
              drift = torch.where(alive, torch.maximum(drift, d), drift)
              yaw_now = yaw()
              turn += ((yaw_now - yaw_last + math.pi) % (2 * math.pi) - math.pi) * alive.float()
              yaw_last = yaw_now
          if sc.startswith("hand"):
            hand_seen |= alive & (term.hand_ramp >= 1.0)
            if xy_hand0 is not None:
              hand_ok &= alive & (nose > 60.0) & (nose < 120.0)
              stand_t += alive.float() * dt
              stand_steps += rear.compute_first_contact(dt=dt).float().sum(1) * alive.float()
              leg_t += leg_on.float() * dt
              paw_out = alive & (term.hand_ramp >= 1.0)
              paws = robot.data.site_pos_w[:, paw_sites] - robot.data.root_link_pos_w.unsqueeze(1)
              across = torch.sum(paws * heading_frame(robot)[1].unsqueeze(1), dim=2)
              across = torch.where(term.hand_side == 1, across[:, 0], across[:, 1])
              paw_lo = torch.where(paw_out, torch.minimum(paw_lo, across), paw_lo)
              paw_hi = torch.where(paw_out, torch.maximum(paw_hi, across), paw_hi)
        base.sim.step = sim_step
      peak, held, heat = power(alive_steps)
      pct = lambda b: float(100 * b.float().mean())
      q = lambda x: [float(v) for v in np.percentile(x[torch.isfinite(x)].cpu().numpy(), [50, 95])] if torch.isfinite(x).any() else [math.nan] * 2
      r = {
        "survived": pct(alive), "arm_touch": pct(arm_hit),
        "peak_A": q(peak), "held50_A": q(held), "motors_full": q(heat),
      }
      if sc in ("walk", "rise", "stand"):
        r["leg_touch"] = float(100 * leg_t.sum() / up_t.sum().clamp(min=1e-6))
      if sc == "walk":
        e = (err / err_steps.clamp(min=1).unsqueeze(1)).cpu().numpy()[err_steps.cpu().numpy() > 0]
        r |= {
          "err_vx": float(np.mean(e[:, 0])), "err_vy": float(np.mean(e[:, 1])), "err_wz": float(np.mean(e[:, 2])),
          "steps_per_s": float((touchdowns / (steps_up * dt).clamp(min=1e-6)).median()),
          "one_foot_down": float((one_foot / steps_up.clamp(min=1)).median() * 100),
          "both_air": float((both_air / steps_up.clamp(min=1)).median() * 100),
          "standing_steps_per_s": float((stand_steps.sum() / stand_t.sum().clamp(min=1e-6))),
          "standing_planted": float(100 * stand_planted.sum() / stand_t.sum().clamp(min=1e-6)),
          "standing_arm_hip": float(stand_hip.sum() / stand_t.sum().clamp(min=1e-6)),
          "standing_arm_height": float(stand_paw.sum() / stand_t.sum().clamp(min=1e-6)),
          "standing_joint_speed": math.sqrt(float(stand_speed.sum() / stand_t.sum().clamp(min=1e-6))),
          "front_touch": pct(front_touch),
        }
      if sc == "rise":
        rise_t = risen_at[~torch.isnan(risen_at)]
        r |= {
          "risen": pct(~torch.isnan(risen_at) & (risen_at <= 3.0)),
          "rise_s": float(rise_t.median()) if len(rise_t) else math.nan,
          "down": pct(down_ok), "front_impact_N": q(front_impact), "dropped": pct(dropped),
        }
      if sc == "again":
        r |= {"risen": pct(~torch.isnan(risen_at))}
      if sc == "stand":
        r |= {"drift_m": q(drift), "turn_deg": q(torch.rad2deg(turn.abs()))}
      if sc == "held":
        along = got / got_steps.clamp(min=1).unsqueeze(1) * (held_cmd / torch.norm(held_cmd, dim=1, keepdim=True))
        for k, label in enumerate(HELD_CMDS):
          group = (torch.arange(n, device=device) % len(HELD_CMDS) == k) & alive
          r[label] = float(along.sum(1)[group].median()) if group.any() else math.nan
      if sc.startswith("hand"):
        r |= {
          "upright": pct(hand_ok & hand_seen & alive), "drift_m": q(drift),
          "turn_deg": q(torch.rad2deg(turn.abs())),
          "steps_per_s": float(stand_steps.sum() / stand_t.sum().clamp(min=1e-6)),
          "leg_touch": float(100 * leg_t.sum() / stand_t.sum().clamp(min=1e-6)),
          "paw_sweep_m": q(paw_hi - paw_lo),
        }
      res[sc] = r
    results[name] = res

  env.close()
  for name, res in results.items():
    print(f"\n===== {name}: {policies[name]} ({n} robots per scenario, mirror-averaged)")
    for sc, r in res.items():
      print(f"  {sc}:")
      for k, v in r.items():
        print(f"    {k:16s} " + (f"{v[0]:.2f} (p95 {v[1]:.2f})" if isinstance(v, list) else f"{v:.2f}"))
  if cfg.json:
    flat = {}
    for name, res in results.items():
      out = {}
      for sc, r in res.items():
        for k, v in r.items():
          out[f"{sc}_{k}"] = v[0] if isinstance(v, list) else v
          if isinstance(v, list):
            out[f"{sc}_{k}_p95"] = v[1]
      flat[name] = out
    with open(cfg.json, "w") as f:
      json.dump(flat, f, indent=2)
    print(f"[INFO] results written to {cfg.json}")


if __name__ == "__main__":
  main()
