"""Script to evaluate SideFlip or FrontFlip policies on the same standing starts.

Each policy runs deterministically on the task's start distribution (same seeds, so the
same starts) for one flip per robot (the training episode). SideFlip: as trained (the
left flip, policy.onnx) and mirrored (the right flip, policy_right.onnx; unless
--no-right). FrontFlip: mirror-averaged (policy.onnx, what deploys) and plain
(policy_raw.onnx). The rotation is measured here over the whole run, not read from the
rewards, so a second turn shows: the world rate about the flip's axis as it was at the
start (made level). A spin about another axis adds nothing (FrontFlip v1 spun round on its
front feet, which the body rate counted). Reports:
  - success: no body contact or fall, a liftoff, one turn (330-414 deg) in the flip's
    direction, upside down on the way (150 deg from level), landed on the feet, and the
    deploy done check (STAND_CHECK held 0.3 s after stand_start) by the end; by terrain
    and by start stance
  - the flip: rotation, max body tilt, peak height gained, air time (no limb on the
    ground), heading drift, max tilt of the flip's axis (the side flip's pitch, the front
    flip's roll) and how far it got from its start direction (tipped or turned), travel
    along the flip and drift across it
  - gentleness: landing force per foot, thigh/calf touches (and their peak force, with
    FrontFlip's knees_ground sensor), joint speed vs the motors' top speed, time over
    90% torque, torque-limit hits against the speed-dependent DC limit (by joint); the
    front and rear knees' height above the ground (flat tiles) at the first touchdown after half a
    turn (the rear ones also over every touchdown, fell later or not) and the knee
    motors' peak torque in the 0.3 s after it (multistage v2: v1 came down with its rear
    knees a few cm off the floor)
  - FrontFlip: the rotation at liftoff (turned on the front feet first) and the rear-feet
    catch (rear feet down from 216 deg on, before any body contact: v6), with its angle,
    kneeling at the end (a thigh or calf on the ground: v8), the gap between the two rear
    feet's touchdowns and whether the second only comes after the front feet, both rear
    feet down within the catch's pair_window (multistage), landed and still up at the end
    (success without the done check), and the left/right joint mismatch (sum of
    |q - mirror(q)|^2 dt: v11; and in the air only: v13)
  - failure modes (with the body parts down at a crash: body, head, chin, hips), by terrain

Example:
  python scripts/eval_flip.py --policies v1=logs/rsl_rl/go2_sideflip/<run>/model_6000.pt
  python scripts/eval_flip.py --task Unitree-Go2-FrontFlip \
    --policies v1=logs/rsl_rl/go2_frontflip/<run>/model_6000.pt
"""

import collections
import json
import math
import re
import sys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
import tyro

from mjlab.actuator import DcMotorActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends

FULL_TURN_MIN = 330.0 / 360.0
UPSIDE_DOWN = math.radians(150.0)
DONE_HOLD_S = 0.3


@dataclass(frozen=True)
class EvalFlipConfig:
  policies: tuple[str, ...]
  """name=checkpoint pairs (model_N.pt from a run of `task`)."""
  task: str = "Unitree-Go2-SideFlip"
  """Unitree-Go2-SideFlip or Unitree-Go2-FrontFlip."""
  rounds: int = 2
  num_envs: int = 1024
  right: bool = True
  """SideFlip: also evaluate each policy mirrored (the right flip, what policy_right.onnx
  deploys). FrontFlip always runs mirror-averaged and plain."""
  device: str | None = None
  json: str | None = None
  """Also write each variant's headline results (%, ms, N) to this file (read by
  scripts/train_front_flip.py's pass checks)."""
  pair_window: float | None = None
  """Both rear feet down within this (s) count as paired (default: the task's catch
  window; a multistage stage's own, from scripts/train_front_flip.py)."""


def main():
  # Populate task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.assets.robots.unitree_go2.go2_constants import GO2_CALF_TOP_SPEED, GO2_HIP_THIGH_TOP_SPEED
  from src.tasks.kick.flip_env_cfg import FRONT_FLIP, LIFTOFF_UP_VEL, SIDE_FLIP
  from src.tasks.kick.kick_env_cfg import STAND_CHECK
  from src.tasks.recovery.mdp.rewards import _slot_columns
  from src.tasks.velocity.rl.symmetry import _get_maps

  cfg = tyro.cli(EvalFlipConfig, prog=sys.argv[0])
  policies = dict(p.split("=", 1) for p in cfg.policies)
  front = "FrontFlip" in cfg.task
  spec = FRONT_FLIP if front else SIDE_FLIP
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
  run_s = env_cfg.episode_length_s  # one flip per robot (the training episode length)
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.events.pop("mid_flip_start", None)  # standing starts only (training adds mid-flip ones)
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
  feet_sensor, limbs_sensor = base.scene["feet_ground"], base.scene["limbs_ground"]
  feet_cols = torch.tensor(_slot_columns(feet_sensor, r"^[FR][LR]_foot_collision$")[0], device=device)
  rear_cols = torch.tensor(_slot_columns(feet_sensor, r"^R[LR]_foot_collision$")[0], device=device)
  knee_cols = torch.tensor(_slot_columns(limbs_sensor, r"_(thigh|calf\d)_collision$")[0], device=device)
  # Thigh/calf contact force (FrontFlip's knees_ground sensor, v10; touches only without it).
  knee_force = base.scene.sensors.get("knees_ground")
  # The body parts down at a crash, noted at the termination check (the step resets after it).
  body_sensor = base.scene["body_ground"]
  body_cols, body_geoms = _slot_columns(body_sensor, r".")
  body_cols = torch.tensor(body_cols, device=device)
  parts = [g.split("/")[-1].replace("_collision", "") for g in body_geoms]
  parts = [{"base1": "body", "base2": "head", "base3": "chin"}.get(p, p) for p in parts]
  part_bits = 2 ** torch.arange(len(parts), device=device)
  at_check = {}
  check = tm.compute

  def check_and_note(*args, **kwargs):
    out = check(*args, **kwargs)
    at_check["parts"] = ((body_sensor.data.found[:, body_cols] > 0).long() * part_bits).sum(1)
    return out

  tm.compute = check_and_note
  full_turn_max = spec["landing"]["max_turns"]
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
  # Each joint's mirror twin (the other side; hips mirror with their sign flipped).
  other = {"L": "R", "R": "L"}
  twin = torch.tensor(
    [joint_names.index(re.sub(r"^([FR])([LR])_", lambda m: m[1] + other[m[2]] + "_", j)) for j in joint_names],
    device=device,
  )
  twin_sign = torch.tensor([-1.0 if "hip" in j else 1.0 for j in joint_names], device=device)
  # Touchdown order: RL, RR, FL, FR (physics substeps; force history index 0 is the newest).
  td_cols = torch.tensor(
    [_slot_columns(feet_sensor, rf"^{f}_foot_collision$")[0][0] for f in ("RL", "RR", "FL", "FR")], device=device
  )
  sub_dt = dt / env_cfg.decimation
  # Knees (calf body origins) and feet, front and rear pairs, for the touchdown posture.
  bodies = list(robot.body_names)
  knee_b = torch.tensor([[bodies.index(f"{f}{s}_calf") for s in "LR"] for f in "FR"], device=device)
  foot_g = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{f}_foot_collision") for f in ("FL", "FR", "RL", "RR")]
  foot_r = float(model.geom_size[foot_g[0], 0])
  knee_act = torch.tensor(
    [[i for i, a in enumerate(robot.actuator_names) if re.fullmatch(rf"{f}[LR]_calf_joint", a)] for f in "FR"], device=device
  )
  maps = _get_maps(base)
  obs_perm, obs_sign = maps["actor"]
  act_perm, act_sign = maps["__actions__"]
  size, nr, nc = tg.size, tg.num_rows, tg.num_cols
  steps = int(round(run_s / dt))
  t_spin, t_stand, t_land = (spec["timing"][k] for k in ("spin_end", "stand_start", "land_by"))
  hold_steps = int(round(DONE_HOLD_S / dt))
  # Both rear feet down within this (s) count as a paired catch (multistage).
  pair_window = cfg.pair_window or (spec.get("catch") or {}).get("pair_window") or 0.1
  axis = spec["axis"]
  x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
  flip_axis = torch.eye(3, device=device)[axis].expand(n, 3)  # the body axis the flip turns about

  def tile_of(xy):
    r = int(np.floor((xy[0] + nr * size[0] * 0.5) / size[0]))
    c = int(np.floor((xy[1] + nc * size[1] * 0.5) / size[1]))
    return tiles.get((r, c), "border")

  # (name, checkpoint, mode): plain, mirrored (SideFlip's right flip) or averaged
  # (FrontFlip's policy.onnx: the policy and its mirror twin, averaged).
  variants = []
  for name, ckpt in policies.items():
    if front:
      variants += [(f"{name}-avg", ckpt, "averaged"), (f"{name}-raw", ckpt, "plain")]
    else:
      variants.append((f"{name}-L", ckpt, "plain"))
      if cfg.right:
        variants.append((f"{name}-R", ckpt, "mirrored"))

  results = {}
  for name, ckpt, mode in variants:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    # The trained side flip goes its direction; its mirror twin the other way. A front
    # flip's mirror twin is the same flip.
    direction = -spec["direction"] if mode == "mirrored" and not front else spec["direction"]
    print(f"[INFO] {name}: {ckpt} ({mode})")

    def act(obs):
      if mode == "plain":
        return policy(obs)
      flipped = obs.clone()
      flipped["actor"] = obs["actor"][:, obs_perm] * obs_sign
      twin = policy(flipped)[:, act_perm] * act_sign
      return twin if mode == "mirrored" else 0.5 * (policy(obs) + twin)

    rows = []
    torch.manual_seed(0)
    for _ in range(cfg.rounds):
      with torch.no_grad():
        obs, _ = env.reset()
        pos0 = robot.data.root_link_pos_w.clone()
        # The ground under the standing feet (a hard landing sinks a foot a few cm into it).
        ground0 = base.sim.data.geom_xpos[:, foot_g, 2].amin(1) - foot_r
        fwd0 = quat_apply(robot.data.root_link_quat_w, x_axis).clone()
        fwd0[:, 2] = 0.0
        fwd0 = fwd0 / torch.norm(fwd0, dim=1, keepdim=True)
        side0 = torch.stack([-fwd0[:, 1], fwd0[:, 0], torch.zeros_like(fwd0[:, 0])], dim=1)  # left
        axis0 = side0 if axis else fwd0  # the flip's axis at the start, level
        # Travel along the flip: the side flip moves toward its side, the front flip forward.
        along0, across0 = (fwd0, side0) if front else (side0 * -direction, fwd0)
        calf0 = robot.data.joint_pos[:, [i for i, j in enumerate(joint_names) if "calf" in j]].mean(1)
        jointlock0 = (calf0 > -1.65).cpu().numpy()
        xy0 = pos0[:, :2].cpu().numpy()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        why = torch.zeros(n, dtype=torch.long, device=device)  # 0 alive, 1 body contact, 2 fell
        hit = torch.zeros(n, dtype=torch.long, device=device)  # body parts down at a crash (bits)
        z = lambda: torch.zeros(n, device=device)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=device)
        turns, height, air, pitch_max, land_peak, tilt_max, off_max = z(), z(), z(), z(), z(), z(), z()
        qd_frac, tau90_time, tau_time = z(), z(), z()
        lifted, landed, tau_hit, caught = b(), b(), b(), b()
        knee_hit, knee_peak = b(), z()
        knee_up = torch.full((n, 2), float("nan"), device=device)  # front, rear knees at the touchdown (m)
        knee_tau = torch.zeros(n, 2, device=device)  # front, rear knee motors' peak share after it
        t_land0 = torch.full((n,), float("nan"), device=device)
        td_t = torch.full((n, 4), float("nan"), device=device)  # first touchdown after half a turn
        mirror, mirror_air = z(), z()
        lift_turns, catch_turns = z(), z()
        tau_j = torch.zeros(n, dtype=torch.long, device=device)
        held = torch.zeros(n, dtype=torch.int32, device=device)
        t_done = torch.full((n,), float("nan"), device=device)
        t_end = torch.full((n,), float("nan"), device=device)
        for i in range(steps):
          obs, _, dones, _ = env.step(act(obs))
          t = (i + 1) * dt
          ended = dones.bool() & alive
          why = torch.where(ended & tm.get_term("body_contact"), torch.ones_like(why), why)
          hit = torch.where(ended & tm.get_term("body_contact"), at_check["parts"], hit)
          why = torch.where(ended & tm.get_term("fell") & (why == 0), torch.full_like(why, 2), why)
          t_end = torch.where(ended, torch.full_like(t_end, t), t_end)
          alive &= ~ended
          if not alive.any():
            break
          feet_down = (feet_sensor.data.found[:, feet_cols] > 0).any(dim=1)
          airborne = ~(limbs_sensor.data.found > 0).any(dim=1)
          knee_hit |= alive & (limbs_sensor.data.found[:, knee_cols] > 0).any(dim=1)
          if knee_force is not None:
            kf = torch.norm(knee_force.data.force_history, dim=-1).amax(dim=(1, 2))
            knee_peak = torch.where(alive, torch.maximum(knee_peak, kf), knee_peak)
          rate = direction * torch.sum(robot.data.root_link_ang_vel_w * axis0, dim=1) / (2 * math.pi)
          turns = torch.where(alive, turns + rate * dt, turns)
          if t < t_spin:
            rising = robot.data.root_link_lin_vel_w[:, 2] > LIFTOFF_UP_VEL
            new_lift = alive & airborne & rising & ~lifted
            lift_turns = torch.where(new_lift, turns, lift_turns)
            lifted |= new_lift
          # Rear-feet catch (the episode is still running, so no body contact yet).
          if front:
            rear_down = (feet_sensor.data.found[:, rear_cols] > 0).any(dim=1)
            new_catch = alive & lifted & rear_down & (turns >= 0.6) & ~caught
            catch_turns = torch.where(new_catch, turns, catch_turns)
            caught |= new_catch
          if front:
            on = torch.norm(feet_sensor.data.force_history[:, td_cols], dim=-1) > 0
            sub = torch.arange(on.shape[2], device=device).expand_as(on)
            k = torch.where(on, sub, torch.full_like(sub, -1)).amax(2)  # the step's earliest substep down
            first = (alive & lifted & (turns > 0.5))[:, None] & (k >= 0) & td_t.isnan()
            td_t = torch.where(first, t - k.float() * sub_dt, td_t)
            q = robot.data.joint_pos
            mismatch = torch.sum(torch.square(q - q[:, twin] * twin_sign), dim=1) * dt
            mirror += alive.float() * mismatch
            mirror_air += (alive & airborne).float() * mismatch
          air += (alive & airborne).float() * dt
          height = torch.where(alive, torch.maximum(height, robot.data.root_link_pos_w[:, 2] - pos0[:, 2]), height)
          along_axis = quat_apply(robot.data.root_link_quat_w, flip_axis)
          axis_tilt = torch.asin(torch.clamp(along_axis[:, 2], -1, 1)).abs()
          pitch_max = torch.where(alive, torch.maximum(pitch_max, axis_tilt), pitch_max)
          axis_off = torch.acos(torch.clamp(torch.sum(along_axis * axis0, dim=1), -1, 1))
          off_max = torch.where(alive, torch.maximum(off_max, axis_off), off_max)
          body_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1))
          tilt_max = torch.where(alive, torch.maximum(tilt_max, body_tilt), tilt_max)
          # Landing: first touchdown after liftoff; peak per-foot force in the 0.3 s after.
          new_land = alive & lifted & feet_down & (turns > 0.5) & ~landed
          landed |= new_land
          up = robot.data.body_link_pos_w[:, knee_b.flatten(), 2].view(n, 2, 2).mean(2) - ground0[:, None]
          knee_up = torch.where(new_land[:, None], up, knee_up)
          t_land0 = torch.where(new_land, torch.full_like(t_land0, t), t_land0)
          if bool(landed.any()):
            f = torch.norm(feet_sensor.data.force_history[:, feet_cols], dim=-1).amax(dim=(1, 2))
            land_peak = torch.where(alive & landed, torch.maximum(land_peak, f), land_peak)
          force = robot.data.actuator_force
          tau90_time += (alive & ((force.abs() / limits).amax(1) > 0.9)).float() * dt
          qd_frac = torch.where(alive, torch.maximum(qd_frac, (robot.data.joint_vel.abs() / joint_top).amax(1)), qd_frac)
          ratio = force.abs() / available_torque(force, robot.data.joint_vel[:, act_joint])
          sat = (ratio.amax(1) > 0.99) & alive
          tau_time += sat.float() * dt
          tau_j = torch.where(sat & ~tau_hit, ratio.argmax(1), tau_j)
          tau_hit |= sat
          after = (alive & (t <= t_land0 + 0.3))[:, None]  # false before the touchdown (nan)
          knee_tau = torch.where(after, torch.maximum(knee_tau, ratio[:, knee_act].amax(2)), knee_tau)
          # Deploy done check.
          if t >= t_stand:
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1))
            err = (robot.data.joint_pos - robot.data.default_joint_pos).abs().amax(1)
            ok = alive & (tilt < STAND_CHECK["max_tilt"]) & (err < STAND_CHECK["max_pose_err"]) & (
              robot.data.joint_vel.abs().amax(1) < STAND_CHECK["max_joint_vel"])
            held = torch.where(ok, held + 1, torch.zeros_like(held))
            t_done = torch.where(torch.isnan(t_done) & (held >= hold_steps), torch.full_like(t_done, t), t_done)
        fwd = quat_apply(robot.data.root_link_quat_w, x_axis)
        heading = torch.atan2(fwd0[:, 0] * fwd[:, 1] - fwd0[:, 1] * fwd[:, 0], torch.sum(fwd0[:, :2] * fwd[:, :2], 1))
        travel = robot.data.root_link_pos_w - pos0
        along = torch.sum(travel * along0, dim=1)
        across = torch.sum(travel * across0, dim=1).abs()
        kneel = alive & (limbs_sensor.data.found[:, knee_cols] > 0).any(dim=1)  # thigh or calf down at the end
        rear_gap = (td_t[:, 0] - td_t[:, 1]).abs()  # nan unless both rear feet came down
        rear_late = torch.maximum(td_t[:, 0], td_t[:, 1]) > torch.minimum(td_t[:, 2], td_t[:, 3]) + 1e-6
      cols = {k: v.cpu().numpy() for k, v in dict(
        kneel=kneel, knee_hit=knee_hit, knee_peak=knee_peak, rear_gap=rear_gap, rear_late=rear_late, mirror=mirror,
        mirror_air=mirror_air, why=why, lifted=lifted, turns=turns, height=height, air=air, pitch=pitch_max, landed=landed,
        land=land_peak, qd_frac=qd_frac, tau90=tau90_time, tau=tau_hit, tau_j=tau_j, tau_time=tau_time,
        t_done=t_done, heading=heading.abs(), along=along, across=across, tilt_max=tilt_max,
        knee_up_front=knee_up[:, 0], knee_up_rear=knee_up[:, 1], knee_tau_front=knee_tau[:, 0],
        knee_tau_rear=knee_tau[:, 1],
        off=off_max, t_end=t_end, lift_turns=lift_turns, caught=caught, catch_turns=catch_turns, hit=hit).items()}
      for e in range(n):
        r = {k: float(v[e]) for k, v in cols.items()}
        r["terrain"] = tile_of(xy0[e])
        r["stance"] = "JointLock" if jointlock0[e] else "Velocity"
        r["landed_up"] = (
          r["why"] == 0 and r["lifted"] and FULL_TURN_MIN <= r["turns"] <= full_turn_max
          and r["tilt_max"] >= UPSIDE_DOWN and r["landed"]
        )
        r["success"] = r["landed_up"] and not np.isnan(r["t_done"])
        r["paired"] = bool(r["caught"]) and r["rear_gap"] <= pair_window + 1e-6  # False when nan
        rows.append(r)
    results[name] = rows
  env.close()

  names = list(results)
  starts = len(results[names[0]])
  print(f"\n===== {starts} flips per policy ({cfg.rounds} x {n}), the same starts for each =====")

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

  print("\nthe flip  (median / 95th, robots that didn't fall)")
  stat("rotation (deg)", "turns", "6.0f", scale=360.0)
  stat("rotation, 5th percentile (deg)", "turns", "6.0f", scale=360.0, tail=5)
  for nm in names:
    over = np.mean([r["turns"] > full_turn_max for r in results[nm] if r["why"] == 0])
    print(f"  {'over-rotated (> ' + str(round(full_turn_max * 360)) + ' deg)':30s} {nm}: {100 * over:4.1f}%")
  stat("max body tilt (deg, 180 = upside down)", "tilt_max", "5.0f", scale=180 / np.pi, tail=5)
  for nm in names:
    up = np.mean([r["tilt_max"] >= UPSIDE_DOWN for r in results[nm] if r["why"] == 0])
    print(f"  {'upside down (>= 150 deg)':30s} {nm}: {100 * up:4.1f}%")
  stat("peak height gained (cm)", "height", "5.1f", scale=100)
  stat("air time (s)", "air", "5.2f")
  stat(f"max {'roll' if front else 'pitch'} of the flip axis (deg)", "pitch", "5.1f", scale=180 / np.pi)
  stat("flip axis off its start (deg)", "off", "5.1f", scale=180 / np.pi)
  stat("heading change (deg)", "heading", "5.1f", scale=180 / np.pi)
  stat(f"travel along the flip (cm, + = {'forward' if front else 'its side'})", "along", "5.1f", scale=100)
  stat("drift across it (cm)", "across", "5.1f", scale=100)
  if front:
    print("\nfront flip phases  (all flips: median / 95th)")
    stat("rotation at liftoff (deg)", "lift_turns", "6.0f", keep=lambda r: r["lifted"], scale=360.0)
    for nm in names:
      c = np.mean([r["caught"] for r in results[nm]])
      print(f"  {'rear-feet catch (% of flips)':30s} {nm}: {100 * c:5.1f}%")
    for nm in names:
      c = np.mean([r["paired"] for r in results[nm]])
      print(f"  {f'both rear within {pair_window * 1000:.0f} ms (%)':30s} {nm}: {100 * c:5.1f}%")
    for nm in names:
      c = np.mean([r["landed_up"] for r in results[nm]])
      print(f"  {'landed, up at the end (%)':30s} {nm}: {100 * c:5.1f}%")
    stat("catch at (deg)", "catch_turns", "6.0f", keep=lambda r: r["caught"], scale=360.0)
    for nm in names:
      alive = [r["kneel"] for r in results[nm] if r["why"] == 0]
      k = np.mean(alive) if alive else 0.0
      print(f"  {'kneeling at the end (% alive)':30s} {nm}: {100 * k:5.1f}%")
    both = lambda r: r["why"] == 0 and not np.isnan(r["rear_gap"])
    stat("rear feet touchdown gap (ms)", "rear_gap", "5.0f", keep=both, scale=1000)
    for nm in names:
      late = [r["rear_late"] for r in results[nm] if both(r)]
      print(f"  {'2nd rear foot after the front':30s} {nm}: {100 * np.mean(late) if late else 0.0:5.1f}%")
    stat("left/right mismatch (rad^2 s)", "mirror", "5.2f")
    stat("  of it in the air (rad^2 s)", "mirror_air", "5.2f")
  print("\ngentleness  (median / 95th, robots that didn't fall)")
  stat("landing force per foot (N)", "land", "6.0f", keep=lambda r: r["why"] == 0 and r["landed"], over=(400.0,))
  for nm in names:
    k = np.mean([r["knee_hit"] for r in results[nm]])
    print(f"  {'thigh/calf touch (% of flips)':30s} {nm}: {100 * k:5.1f}%")
  if knee_force is not None:
    stat("peak thigh/calf force (N)", "knee_peak", "6.0f", keep=lambda r: r["knee_hit"])
  landed_ok = lambda r: r["why"] == 0 and r["landed"]
  # Flat tiles only: on stairs and slopes the start's ground isn't the ground under the knees.
  flat_ok = lambda r: landed_ok(r) and r["terrain"] == "flat"
  stat("front knees up at touchdown, flat (cm)", "knee_up_front", "5.1f", keep=flat_ok, scale=100, tail=5)
  stat("rear knees up at touchdown, flat (cm)", "knee_up_rear", "5.1f", keep=flat_ok, scale=100, tail=5)
  # Every touchdown, fell later or not (multistage stage 1 crashes after it).
  flat_td = lambda r: r["landed"] and r["terrain"] == "flat"
  stat("  every touchdown, flat (cm)", "knee_up_rear", "5.1f", keep=flat_td, scale=100, tail=5)
  stat("front knee motors after it (%)", "knee_tau_front", "5.0f", keep=landed_ok, scale=100, over=(99.0,))
  stat("rear knee motors after it (%)", "knee_tau_rear", "5.0f", keep=landed_ok, scale=100, over=(99.0,))
  stat("peak joint speed / top speed", "qd_frac", "5.2f", over=(0.9,))
  stat("time over 90% torque (ms)", "tau90", "5.0f", scale=1000)
  stat("time at torque limit (ms), hit", "tau_time", "5.0f", keep=lambda r: r["tau"], scale=1000)
  print("\ntorque-limit hits  (% of flips; first hit by joint)")
  for nm in names:
    hit = [r for r in results[nm] if r["tau"]]
    joints = collections.Counter(robot.actuator_names[int(r["tau_j"])] for r in hit)
    print(f"  {nm:10s} {100 * len(hit) / starts:4.1f}% | " + ", ".join(f"{k} {v}" for k, v in joints.most_common(6)))

  print()
  for nm in names:
    modes, where = collections.Counter(), collections.Counter()
    for r in results[nm]:
      if r["success"]:
        continue
      if r["why"] == 1:
        mode = "body contact" + (" after the catch" if r["caught"] else " (after liftoff)" if r["lifted"] else "")
        mode += ": " + "+".join(p for k, p in enumerate(parts) if int(r["hit"]) >> k & 1)
      elif r["why"] == 2:
        # Before land_by only the flip axis ends it (tipped, or turned with max_axis_angle).
        mode = "fell (flip axis off)" if r["t_end"] < t_land - 1e-6 else "fell (not level by land_by)"
        mode += "" if r["lifted"] else ", no liftoff"
      elif not r["lifted"]:
        mode = "no liftoff"
      elif r["turns"] < FULL_TURN_MIN:
        mode = "partial rotation"
      elif r["turns"] > full_turn_max:
        mode = "over-rotated (more than one turn)"
      elif r["tilt_max"] < UPSIDE_DOWN:
        mode = "never upside down"
      elif not r["landed"]:
        mode = "no landing"
      else:
        mode = "not standing still at the end" + (", kneeling" if r["kneel"] else "")
      modes[mode] += 1
      where[r["terrain"]] += 1
    print(f"failures {nm} ({sum(modes.values())}): {dict(modes.most_common())}")
    print(f"  by terrain: {dict(where.most_common())}")

  if cfg.json:
    pct = lambda rows, k: float(100 * np.mean([r[k] for r in rows])) if rows else float("nan")
    med = lambda x: float(np.median(x)) if len(x) else float("nan")
    out = {}
    for nm in names:
      rows = results[nm]
      up = [r for r in rows if r["why"] == 0]
      both = [r for r in up if not np.isnan(r["rear_gap"])]
      out[nm] = {
        "success": pct(rows, "success"),
        "flat": pct([r for r in rows if r["terrain"] == "flat"], "success"),
        "landed_up": pct(rows, "landed_up"),
        "caught": pct(rows, "caught"),
        "paired": pct(rows, "paired"),
        "rear_gap_ms": med([1000 * r["rear_gap"] for r in both]),
        "rear_late": pct(both, "rear_late"),
        "knee_touch": pct(rows, "knee_hit"),
        "landing_force": med([r["land"] for r in up if r["landed"]]),
        "rear_knee_up_cm": med([100 * r["knee_up_rear"] for r in up if r["landed"] and r["terrain"] == "flat"]),
        "front_knee_up_cm": med([100 * r["knee_up_front"] for r in up if r["landed"] and r["terrain"] == "flat"]),
        "rear_knee_td_cm": med([100 * r["knee_up_rear"] for r in rows if r["landed"] and r["terrain"] == "flat"]),
        "front_knee_motor": med([100 * r["knee_tau_front"] for r in up if r["landed"]]),
        "by_terrain": {t: pct([r for r in rows if r["terrain"] == t], "success") for t in sorted({r["terrain"] for r in rows})},
      }
    with open(cfg.json, "w") as f:
      json.dump(out, f, indent=2)
    print(f"[INFO] results written to {cfg.json}")


if __name__ == "__main__":
  main()
