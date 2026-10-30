"""Script to evaluate kick policies (DonkeyKick or QuickBuck) on the same standing starts.

Each policy runs deterministically on the task's start distribution (same seeds, so the
same starts) for one kick per robot (the training episode: 3.5 s DonkeyKick, 2 s
QuickBuck), both plain and mirror-averaged (as policy.onnx deploys it, unless
--no-mirror-avg). Reports:
  - success: no fall, the kick, and the deploy done check (STAND_CHECK held 0.3 s after
    stand_start) by the end; by terrain and by start stance. The kick:
      DonkeyKick: a handstand (nose-down > 35 deg with both rear feet up, >= 0.4 s in
        total) and both-feet extension gained in the kick window >= 0.6 (kick_snap's
        measure)
      QuickBuck: best strike score in the kick window >= 0.6 (strike's measure: both
        feet fast and in the strike zone)
  - DonkeyKick, the handstand: time up, pitch during the hold, roll wobble
  - the kick: DonkeyKick extension gained, peak reach, left/right difference;
    QuickBuck strike score, closest approach to the zone, speed and height in the zone,
    reach, left/right difference, rear feet air time; both: peak backward foot speed
    (the slower foot), head (chin) clearance
  - gentleness: rear landing force, joint speed vs the motors' top speed, time over 80%
    torque, front-motor torque-limit hits (against the torque the motor can give at its
    speed: the DC curve for DC motors; by joint and phase)
  - failure modes, by terrain

Example:
  python scripts/eval_kick.py --policies v1=logs/rsl_rl/go2_kick/<run>/model_6000.pt
  python scripts/eval_kick.py --task Unitree-Go2-QuickBuck \
    --policies b6=logs/rsl_rl/go2_buck/<run>/model_6000.pt
"""

import collections
import re
import sys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.actuator import DcMotorActuator
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

HANDSTAND_PITCH_MIN = 0.61  # 35 deg nose-down counts as "up"
CHIN = ((0.293, 0.0, -0.06), 0.047)  # Go2 base3_collision (chin sphere, under the L1 LiDAR): centre, radius
DONE_HOLD_S = 0.3


@dataclass(frozen=True)
class EvalKickConfig:
  policies: tuple[str, ...]
  """name=checkpoint pairs (model_N.pt from a run of --task)."""
  task: str = "Unitree-Go2-DonkeyKick"
  """Unitree-Go2-DonkeyKick or Unitree-Go2-QuickBuck."""
  rounds: int = 2
  num_envs: int = 1024
  mirror_avg: bool = True
  """Also evaluate each policy mirror-averaged (what policy.onnx deploys)."""
  device: str | None = None


def main():
  # Populate task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.assets.robots.unitree_go2.go2_constants import GO2_CALF_TOP_SPEED, GO2_HIP_THIGH_TOP_SPEED
  from src.tasks.kick.buck_env_cfg import BUCK_TIMING
  from src.tasks.kick.kick_env_cfg import KICK_TIMING, STAND_CHECK
  from src.tasks.kick.mdp.rewards import kick_score, pitch_roll, rear_feet_state, strike_score, strike_zone, zone_distance
  from src.tasks.recovery.mdp.events import _terrain_height
  from src.tasks.recovery.mdp.rewards import _slot_columns
  from src.tasks.velocity.rl.symmetry import _get_maps

  cfg = tyro.cli(EvalKickConfig, prog=sys.argv[0])
  policies = dict(p.split("=", 1) for p in cfg.policies)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  n = cfg.num_envs

  # Record each terrain tile's type while the terrain is generated.
  tiles: dict[tuple[int, int], str] = {}
  create = TerrainGenerator._create_terrain_geom

  def record(self, spec, world_position, difficulty, sub_cfg, sub_row, sub_col):
    tiles[(sub_row, sub_col)] = next(k for k, v in self.cfg.sub_terrains.items() if v is sub_cfg)
    return create(self, spec, world_position, difficulty, sub_cfg, sub_row, sub_col)

  TerrainGenerator._create_terrain_geom = record

  env_cfg = load_env_cfg(cfg.task)
  buck = "strike" in env_cfg.rewards
  timing = BUCK_TIMING if buck else KICK_TIMING
  run_s = env_cfg.episode_length_s  # one kick per robot (the training episode length)
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.episode_length_s = run_s + 0.5  # no time-out resets inside the run
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  tg = env_cfg.scene.terrain.terrain_generator
  tg.seed = 0
  agent_cfg = load_rl_cfg(cfg.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = load_runner_cls(cfg.task)(env, asdict(agent_cfg), device=device)
  base = env.unwrapped
  robot, dt, tm = base.scene["robot"], base.step_dt, base.termination_manager
  feet_sensor = base.scene["feet_ground"]
  rear_cols = torch.tensor(_slot_columns(feet_sensor, r"^R[LR]_foot_collision$")[0], device=device)
  kick_cfg = base.reward_manager.get_term_cfg("strike" if buck else "kick_snap")
  kick_term = kick_cfg.func
  foot_ids, hip_ids = kick_term.foot_ids, kick_term.hip_ids
  model = base.sim.mj_model
  act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{a}") for a in robot.actuator_names]
  limits = torch.tensor(model.actuator_forcerange[act_ids, 1], device=device, dtype=torch.float32)
  # The torque each motor can give at its speed (actuator order): DC motors follow
  # their torque-speed curve (mjlab DcMotorActuator), the others their force range.
  joint_names = list(robot.joint_names)
  act_joint = torch.tensor([joint_names.index(a) for a in robot.actuator_names], device=device)
  sat_effort, top_vel = limits.clone(), torch.full_like(limits, float("inf"))
  for group in robot.actuators:
    if isinstance(group, DcMotorActuator):
      for i, a in enumerate(robot.actuator_names):
        if any(re.fullmatch(e, a) for e in group.cfg.target_names_expr):
          sat_effort[i], top_vel[i] = group.cfg.saturation_effort, group.cfg.velocity_limit

  def available_torque(force, vel):
    v = torch.clamp(vel, -top_vel * (1 + limits / sat_effort), top_vel * (1 + limits / sat_effort))
    top = torch.clamp(sat_effort * (1 - v / top_vel), max=limits)
    bottom = torch.clamp(sat_effort * (-1 - v / top_vel), min=-limits)
    return torch.clamp(torch.where(force >= 0, top, -bottom), min=1e-3)

  # Joint top speeds (Unitree go2_description.urdf), joint order.
  joint_top = torch.tensor(
    [GO2_CALF_TOP_SPEED if "calf" in j else GO2_HIP_THIGH_TOP_SPEED for j in joint_names], device=device
  )
  front_act = torch.tensor([a.startswith("F") for a in robot.actuator_names], device=device)
  front_names = [a for a in robot.actuator_names if a.startswith("F")]
  maps = _get_maps(base)
  obs_perm, obs_sign = maps["actor"]
  act_perm, act_sign = maps["__actions__"]
  chin = torch.tensor(CHIN[0], device=device).expand(n, 3)
  size, nr, nc = tg.size, tg.num_rows, tg.num_cols
  steps = int(round(run_s / dt))
  t_kick0, t_kick1 = timing["kick_start"], timing["kick_end"]
  # DonkeyKick: the handstand hold; QuickBuck: the kick and landing (roll is watched there).
  t_hold0, t_hold1 = (t_kick0, timing["stand_start"]) if buck else (timing["rise_end"], timing["down_start"])
  t_stand, t_land1 = timing["stand_start"], timing["stand_start"] + 0.3
  hold_steps = int(round(DONE_HOLD_S / dt))

  def tile_of(xy):
    r = int(np.floor((xy[0] + nr * size[0] * 0.5) / size[0]))
    c = int(np.floor((xy[1] + nc * size[1] * 0.5) / size[1]))
    return tiles.get((r, c), "border")

  variants = []
  for name, ckpt in policies.items():
    variants.append((name, ckpt, False))
    if cfg.mirror_avg:
      variants.append((f"{name}+avg", ckpt, True))

  results = {}
  for name, ckpt, averaged in variants:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] {name}: {ckpt}{' (mirror-averaged)' if averaged else ''}")

    def act(obs):
      a = policy(obs)
      if not averaged:
        return a
      mirrored = obs.clone()
      mirrored["actor"] = obs["actor"][:, obs_perm] * obs_sign
      return 0.5 * (a + policy(mirrored)[:, act_perm] * act_sign)

    rows = []
    torch.manual_seed(0)
    for _ in range(cfg.rounds):
      with torch.no_grad():
        obs, _ = env.reset()
        xy0 = robot.data.root_link_pos_w[:, :2].cpu().numpy()
        calf0 = robot.data.joint_pos[:, [i for i, j in enumerate(robot.joint_names) if "calf" in j]].mean(1)
        jointlock0 = (calf0 > -1.65).cpu().numpy()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        why = torch.zeros(n, dtype=torch.long, device=device)  # 0 alive, 1 body contact, 2 fell
        z = lambda: torch.zeros(n, device=device)
        up_time, pitch_sum, pitch_cnt, pitch_max, roll_max = z(), z(), z(), z(), z()
        chin_min, chin_t = torch.full((n,), 9.0, device=device), z()
        k_start, k_best, reach_best, reach_lr = z(), z(), z(), z()
        land_peak, tau_hit, qd_peak = z(), torch.zeros(n, dtype=torch.bool, device=device), z()
        qd_frac, tau80_time, v_back = z(), z(), z()
        s_best, s_speed, s_height, s_reach, s_lr, zone_best = z(), z(), z(), z(), z(), z()
        zone_d = torch.full((n,), 9.0, device=device)
        was_air = torch.zeros(n, dtype=torch.bool, device=device)
        tau_t, tau_j, tau_time = z(), torch.zeros(n, dtype=torch.long, device=device), z()  # first hit: time, joint
        held = torch.zeros(n, dtype=torch.int32, device=device)
        t_done = torch.full((n,), float("nan"), device=device)
        for i in range(steps):
          obs, _, dones, _ = env.step(act(obs))
          t = (i + 1) * dt
          ended = dones.bool() & alive
          why = torch.where(ended & tm.get_term("body_contact"), torch.ones_like(why), why)
          why = torch.where(ended & tm.get_term("fell") & (why == 0), torch.full_like(why, 2), why)
          alive &= ~ended
          if not alive.any():
            break
          a = alive.float()
          pitch, roll = pitch_roll(robot)
          rear_air = ~(feet_sensor.data.found[:, rear_cols] > 0).any(dim=1)
          up = rear_air if buck else (pitch > HANDSTAND_PITCH_MIN) & rear_air  # QuickBuck: air time
          up_time += (up & alive).float() * dt
          pitch_max = torch.where(alive, torch.maximum(pitch_max, pitch), pitch_max)
          if t_hold0 <= t < t_hold1:
            pitch_sum += pitch * a
            pitch_cnt += a
            roll_max = torch.where(alive, torch.maximum(roll_max, roll.abs()), roll_max)
          # Chin clearance above the ground under it.
          p = robot.data.root_link_pos_w + quat_apply(robot.data.root_link_quat_w, chin)
          ground = _terrain_height(base, p[:, None, :2], p[:, 2] + 2.0)
          clear = p[:, 2] - CHIN[1] - torch.nan_to_num(ground, nan=-9.0)
          chin_t = torch.where(alive & (clear < chin_min), torch.full_like(chin_t, t), chin_t)
          chin_min = torch.where(alive, torch.minimum(chin_min, clear), chin_min)
          # Backward foot speed (heading frame) of the slower rear foot.
          height, h_reach, speed = rear_feet_state(robot, foot_ids, hip_ids)
          v_back = torch.where(alive, torch.maximum(v_back, speed.amin(1)), v_back)
          if buck:
            # The strike: strike's gated score; speed, height, reach where it was best.
            p_s = {k: kick_cfg.params[k] for k in ("height_range", "edge", "min_reach", "v_full")}
            score = strike_score(robot, foot_ids, hip_ids, **p_s) * rear_air.float()
            zone = strike_zone(height, h_reach, p_s["height_range"], p_s["edge"], p_s["min_reach"])
            if t_kick0 <= t < t_kick1:
              better = alive & (score > s_best)
              s_speed = torch.where(better, speed.amin(1), s_speed)
              s_height = torch.where(better, height.amin(1), s_height)
              s_reach = torch.where(better, h_reach.amin(1), s_reach)
              s_lr = torch.where(better, (height[:, 0] - height[:, 1]).abs(), s_lr)
              s_best = torch.where(better, score, s_best)
              zone_air = zone.amin(1) * rear_air.float()
              zone_best = torch.where(alive, torch.maximum(zone_best, zone_air), zone_best)
              # Farther foot's distance to the zone, with both rear feet up (zone_approach's).
              d = zone_distance(height, h_reach, p_s["height_range"], p_s["min_reach"]).amax(1)
              d = torch.where(rear_air, d, torch.full_like(d, 9.0))
              zone_d = torch.where(alive, torch.minimum(zone_d, d), zone_d)
          else:
            # The kick: kick_snap's gated score, plus the raw reach.
            score = kick_score(robot, foot_ids, hip_ids, kick_cfg.params["reach_min"], kick_cfg.params["reach_span"])
            score = score * ((pitch > kick_cfg.params["min_pitch"]) & rear_air).float()
            d_w = robot.data.site_pos_w[:, foot_ids] - robot.data.body_link_pos_w[:, hip_ids]
            q = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
            reach = -quat_apply_inverse(q, d_w.reshape(-1, 3)).view(n, 2, 3)[..., 0]
            if abs(t - t_kick0) < dt / 2:
              k_start, k_best = score.clone(), score.clone()
            elif t_kick0 < t < t_kick1:
              better = alive & (reach.amin(1) > reach_best)
              reach_lr = torch.where(better, (reach[:, 0] - reach[:, 1]).abs(), reach_lr)
              reach_best = torch.where(better, reach.amin(1), reach_best)
              k_best = torch.where(alive, torch.maximum(k_best, score), k_best)
          # Gentleness. Landing: DonkeyKick from the come-down; QuickBuck once the rear
          # feet have left the ground in the kick window (not the push-off).
          was_air |= rear_air & (t >= t_kick0)
          landing = (t_hold1 <= t < t_land1) if not buck else (t < t_land1)
          if landing:
            f = torch.norm(feet_sensor.data.force_history[:, rear_cols], dim=-1).amax(dim=(1, 2))
            keep = alive & (was_air if buck else True)
            land_peak = torch.where(keep, torch.maximum(land_peak, f), land_peak)
          force = robot.data.actuator_force
          tau80_time += (alive & ((force.abs() / limits).amax(1) > 0.8)).float() * dt
          qd_frac = torch.where(alive, torch.maximum(qd_frac, (robot.data.joint_vel.abs() / joint_top).amax(1)), qd_frac)
          ratio = (force.abs() / available_torque(force, robot.data.joint_vel[:, act_joint]))[:, front_act]
          sat = (ratio.amax(1) > 0.99) & alive
          tau_time += sat.float() * dt
          first = sat & ~tau_hit
          tau_t = torch.where(first, torch.full_like(tau_t, t), tau_t)
          tau_j = torch.where(first, ratio.argmax(1), tau_j)
          tau_hit |= first
          qd_peak = torch.where(alive, torch.maximum(qd_peak, robot.data.joint_vel.abs().amax(1)), qd_peak)
          # Deploy done check.
          if t >= t_stand:
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1))
            err = (robot.data.joint_pos - robot.data.default_joint_pos).abs().amax(1)
            ok = alive & (tilt < STAND_CHECK["max_tilt"]) & (err < STAND_CHECK["max_pose_err"]) & (
              robot.data.joint_vel.abs().amax(1) < STAND_CHECK["max_joint_vel"])
            held = torch.where(ok, held + 1, torch.zeros_like(held))
            t_done = torch.where(torch.isnan(t_done) & (held >= hold_steps), torch.full_like(t_done, t), t_done)
      cols = {k: v.cpu().numpy() for k, v in dict(
        why=why, up=up_time, pitch_mean=pitch_sum / torch.clamp(pitch_cnt, min=1), pitch_max=pitch_max,
        roll=roll_max, chin=chin_min, chin_t=chin_t, kick=k_best - k_start, reach=reach_best, lr=reach_lr,
        land=land_peak, tau=tau_hit, tau_t=tau_t, tau_j=tau_j, tau_time=tau_time, qd=qd_peak, t_done=t_done,
        pitch_end=pitch_roll(robot)[0], qd_frac=qd_frac, tau80=tau80_time, v_back=v_back, strike=s_best,
        s_speed=s_speed, s_height=s_height, s_reach=s_reach, s_lr=s_lr, zone=zone_best, zone_d=zone_d).items()}
      for e in range(n):
        r = {k: float(v[e]) for k, v in cols.items()}
        r["terrain"] = tile_of(xy0[e])
        r["stance"] = "JointLock" if jointlock0[e] else "Velocity"
        kicked = r["strike"] >= 0.6 if buck else (r["up"] >= 0.4 and r["kick"] >= 0.6)
        r["success"] = (r["why"] == 0) and kicked and not np.isnan(r["t_done"])
        rows.append(r)
    results[name] = rows
  env.close()

  names = list(results)
  starts = len(results[names[0]])
  print(f"\n===== {starts} kicks per policy ({cfg.rounds} x {n}), the same starts for each =====")

  def table(title, key=None):
    print(f"\n{title}  (success | median time to done s)")
    keys = sorted({r[key] if key else "all" for r in results[names[0]]})
    for k in keys:
      cells = []
      for nm in names:
        g = [r for r in results[nm] if key is None or r[key] == k]
        s = 100 * np.mean([r["success"] for r in g])
        td = [r["t_done"] for r in g if r["success"]]
        cells.append(f"{nm}: {s:5.1f}% {np.median(td) if td else float('nan'):4.2f}")
      count = sum(1 for r in results[names[0]] if key is None or r[key] == k)
      print(f"  {k:22s} n={count:5d} | " + " | ".join(cells))

  table("overall")
  table("by start stance", "stance")
  table("by terrain", "terrain")

  def stat(label, key, fmt, keep=lambda r: r["why"] == 0, scale=1.0, over=None, tail=95):
    cells = []
    for nm in names:
      x = np.array([r[key] * scale for r in results[nm] if keep(r)])
      if not len(x):
        cells.append(f"{nm}: -")
        continue
      extra = f", >{over[0]:g}: {100 * (x > over[0]).mean():4.1f}%" if over else ""
      cells.append(f"{nm}: {np.median(x):{fmt}} / {np.percentile(x, tail):{fmt}}{extra}")
    print(f"  {label:30s} " + " | ".join(cells))

  if buck:
    print("\nthe kick  (median / 95th, robots that didn't fall)")
    stat("strike score (0-1)", "strike", "5.2f")
    stat("closest to the zone (cm)", "zone_d", "5.1f", scale=100)
    stat("strike speed in the zone (m/s)", "s_speed", "5.2f")
    stat("strike height vs base link (cm)", "s_height", "5.1f", scale=100)
    stat("reach behind hip at strike (cm)", "s_reach", "5.1f", scale=100)
    stat("left/right height gap (cm)", "s_lr", "5.1f", scale=100)
    stat("peak backward foot speed (m/s)", "v_back", "5.2f")
    stat("rear feet air time (s)", "up", "5.2f")
    stat("max pitch (deg)", "pitch_max", "5.1f", scale=180 / np.pi)
    stat("max roll, kick to stand (deg)", "roll", "5.1f", scale=180 / np.pi)
    stat("chin clearance min, med / 5th", "chin", "5.1f", keep=lambda r: True, scale=100, tail=5)
    stat("  on flat only", "chin", "5.1f", keep=lambda r: r["terrain"] == "flat", scale=100, tail=5)
  else:
    print("\nthe handstand  (median / 95th, robots that didn't fall)")
    stat("time up (s)", "up", "5.2f")
    stat("pitch during hold (deg)", "pitch_mean", "5.1f", scale=180 / np.pi)
    stat("max pitch (deg)", "pitch_max", "5.1f", scale=180 / np.pi)
    stat("max roll during hold (deg)", "roll", "5.1f", scale=180 / np.pi)
    stat("chin clearance min, med / 5th", "chin", "5.1f", keep=lambda r: True, scale=100, tail=5)
    stat("  on flat only", "chin", "5.1f", keep=lambda r: r["terrain"] == "flat", scale=100, tail=5)
    print("\nthe kick  (median / 95th, robots that didn't fall)")
    stat("extension gained (0-1)", "kick", "5.2f")
    stat("peak reach behind hip (cm)", "reach", "5.1f", scale=100)
    stat("left/right reach gap (cm)", "lr", "5.1f", scale=100)
    stat("peak backward foot speed (m/s)", "v_back", "5.2f")
  print("\ngentleness  (median / 95th, robots that didn't fall)")
  stat("rear landing force (N)", "land", "6.0f", over=(300.0,))
  stat("peak joint speed (rad/s)", "qd", "5.1f", over=(20.0,))
  stat("peak joint speed / top speed", "qd_frac", "5.2f", over=(0.8,))
  stat("time over 80% torque (ms)", "tau80", "5.0f", scale=1000)
  stat("time at torque limit (ms), hit", "tau_time", "5.0f", keep=lambda r: r["tau"], scale=1000)
  if buck:
    bounds = [(timing["kick_start"], "load"), (timing["kick_end"], "kick"),
              (timing["stand_start"], "land"), (float("inf"), "stand")]
  else:
    bounds = [(timing["rise_end"], "rise"), (timing["kick_start"], "hold"),
              (timing["kick_end"], "kick"), (timing["down_start"], "hold2"),
              (timing["stand_start"], "down"), (float("inf"), "stand")]

  def phase_of(t):
    return next(name for end, name in bounds if t < end)

  print("\nwhen the chin is lowest  (phase: % of kicks)")
  for nm in names:
    c = collections.Counter(phase_of(r["chin_t"]) for r in results[nm])
    print(f"  {nm:8s} " + ", ".join(f"{k} {100 * v / starts:.0f}%" for k, v in c.most_common()))
  print("\nfront-motor torque-limit hits  (% of kicks; first hit by joint and phase)")
  for nm in names:
    hit = [r for r in results[nm] if r["tau"]]
    joints = collections.Counter(front_names[int(r["tau_j"])] for r in hit)
    phases = collections.Counter(phase_of(r["tau_t"]) for r in hit)
    print(f"  {nm:8s} {100 * len(hit) / starts:4.1f}% | "
          + ", ".join(f"{k} {v}" for k, v in joints.most_common()) + " | "
          + ", ".join(f"{k} {v}" for k, v in phases.most_common()))

  print()
  for nm in names:
    modes, where = collections.Counter(), collections.Counter()
    for r in results[nm]:
      if r["success"]:
        continue
      if r["why"] == 1:
        mode = "face-plant / body contact"
      elif r["why"] == 2:
        mode = "fell forward / back" if r["roll"] < 0.6 else "fell sideways"
      elif buck and r["zone"] < 0.5:
        mode = "never reached the strike zone"
      elif buck and r["strike"] < 0.6:
        mode = "too slow in the zone"
      elif buck:
        mode = "not standing still at the end"
      elif r["up"] < 0.1:
        mode = "never got up"
      elif r["up"] < 0.4:
        mode = "up too briefly"
      elif r["kick"] < 0.6:
        mode = "no kick"
      else:
        mode = "not standing still at the end"
      modes[mode] += 1
      where[r["terrain"]] += 1
    print(f"failures {nm} ({sum(modes.values())}): {dict(modes.most_common())}")
    print(f"  by terrain: {dict(where.most_common())}")


if __name__ == "__main__":
  main()
