"""Benchmark a Go2 rough-terrain velocity policy per sub-terrain and difficulty level.

Uses the task's training terrain grid (10 levels x 20 columns) with every level
fixed (no curriculum, no pushes), the same number of robots on each tile, and the
actor's training noise unless ``--no-corruption``.

Modes:
  mix    training command mix for ``--duration-s``: falls per robot per minute, xy
         tracking error, and the share of steps with a thigh/calf on the ground.
  cross  0.5 m/s straight ahead for 20 s from the tile centre: share of robots that
         get 3.6 m out (past the stairs or slope) without falling.

Both also print the mean swing peak per tile: foot centre above the ground under
the foot (the centre ray of each foot sensor). ``--stair-height`` overrides the
tallest riser so tasks with different stairs run on the same terrain.

  python scripts/eval_terrain.py --ckpt logs/rsl_rl/go2_velocity/<run>/model_10000.pt \
    --task Unitree-Go2-Rough-Perceptive --mode cross --stair-height 0.2
"""

import sys
from dataclasses import asdict, replace
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ckpt_compat import match_actor_history

FEET = ("FR", "FL", "RR", "RL")


def main(
  ckpt: str,
  task: str = "Unitree-Go2-Rough-Blind",
  mode: Literal["mix", "cross"] = "mix",
  per_cell: int = 5,
  duration_s: float = 60.0,
  seed: int = 0,
  corruption: bool = True,
  stair_height: float | None = None,
) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(task, play=False)
  match_actor_history(cfg, ckpt)
  gen = cfg.scene.terrain.terrain_generator
  gen.seed = seed
  if stair_height is not None:
    gen.sub_terrains = dict(gen.sub_terrains)
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
      gen.sub_terrains[name] = replace(gen.sub_terrains[name], step_height_range=(0.0, stair_height))
  rows, cols = gen.num_rows, gen.num_cols
  n = rows * cols * per_cell
  cfg.scene.num_envs = n
  cfg.seed = seed
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  cfg.episode_length_s = 1e6  # every termination is a fall
  cfg.observations["actor"].enable_corruption = corruption
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (-1.0, 1.0)  # final training range (no backward stages)
  if mode == "cross":
    duration_s = 20.0
    twist.ranges.lin_vel_x = (0.5, 0.5)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_heading_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1e6, 1e6)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  agent = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent), device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # The same number of robots on every tile (level x column).
  base = env.unwrapped
  terrain = base.scene.terrain
  ids = torch.arange(n, device=base.device)
  terrain.terrain_levels[:] = (ids // per_cell) % rows
  terrain.terrain_types[:] = ids // (per_cell * rows)
  terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  base.reset()
  # Column -> sub-terrain, as the generator allocates them in curriculum mode.
  names = list(gen.sub_terrains.keys())
  props = torch.tensor([gen.sub_terrains[k].proportion for k in names])
  cum = torch.cumsum(props / props.sum(), 0)
  col_type = [int(torch.nonzero(c / cols + 0.001 < cum)[0]) for c in range(cols)]
  env_type = torch.tensor(col_type, device=base.device)[terrain.terrain_types]
  level = terrain.terrain_levels.clone()

  robot = base.scene["robot"]
  leg = base.scene["leg_ground_touch"] if "leg_ground_touch" in base.scene.sensors else None
  site_ids, _ = robot.find_sites(FEET, preserve_order=True)
  contact = base.scene["feet_ground_contact"]
  _, geoms = robot.find_geoms(contact.cfg.primary.pattern)
  contact_cols = [next(i for i, g in enumerate(geoms) if g.startswith(f"{f}_")) for f in FEET]
  rays = [base.scene[f"foot_ray_{f}"] for f in FEET]
  centre = [r.data.distances.shape[1] // 2 for r in rays]  # 0 for one ray, 12 for a 5 x 5 patch

  falls = torch.zeros(n, device=base.device)
  err = torch.zeros(n, device=base.device)
  alive_n = torch.zeros(n, device=base.device)
  leg_steps = torch.zeros(n, device=base.device)
  crossed = torch.zeros(n, dtype=torch.bool, device=base.device)
  fell_before = torch.zeros(n, dtype=torch.bool, device=base.device)
  origin = terrain.env_origins[:, :2].clone()
  peak = torch.zeros(n, 4, device=base.device)
  peak_sum = torch.zeros(n, device=base.device)
  peak_cnt = torch.zeros(n, device=base.device)
  with torch.no_grad():
    obs = env.get_observations()
    for _ in range(int(duration_s / base.step_dt)):
      obs, _, _, _ = env.step(policy(obs))
      fell = base.reset_terminated
      falls += fell
      fell_before |= fell
      alive = ~fell
      cmd = base.command_manager.get_command("twist")
      e = torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
      err += torch.where(alive, e, 0)
      alive_n += alive
      if leg is not None:
        f = torch.norm(leg.data.force_history, dim=-1)  # (n, geoms, substeps)
        leg_steps += ((f > 10.0).any(dim=-1).any(dim=-1) & alive).float()
      d = (robot.data.root_link_pos_w[:, :2] - origin).abs().amax(dim=-1)
      crossed |= (d >= 3.6) & ~fell_before & alive
      ground = torch.stack([r.data.hit_pos_w[:, c, 2] for r, c in zip(rays, centre)], 1)
      clear = robot.data.site_pos_w[:, site_ids, 2] - ground
      in_air = contact.data.found[:, contact_cols] == 0
      peak = torch.where(in_air, torch.maximum(peak, clear), peak)
      landed = contact.compute_first_contact(dt=base.step_dt)[:, contact_cols]
      swing = landed & (contact.data.last_air_time[:, contact_cols] >= 0.1) & alive[:, None]
      peak_sum += (peak * swing).sum(1)
      peak_cnt += swing.sum(1)
      peak = torch.where(landed | fell[:, None], torch.zeros_like(peak), peak)
  env.close()

  minutes = duration_s / 60
  header = f"{'terrain':22s} " + " ".join(f"L{lvl:<5d}" for lvl in range(rows))
  print(f"\n{ckpt}\n{task}, mode {mode}, {per_cell} robots per tile, {duration_s:.0f} s, "
        f"corruption {corruption}, stairs {gen.sub_terrains['pyramid_stairs'].step_height_range}")
  if mode == "mix":
    print(header + "   (falls/robot/min | xy err m/s)")
  else:
    print(header + "   (% crossed 3.6 m in 20 s without falling | falls/robot/min)")
  for t, name in enumerate(names):
    a, b = [], []
    for lvl in range(rows):
      m = (env_type == t) & (level == lvl)
      if mode == "mix":
        a.append(f"{falls[m].mean().item() / minutes:5.2f}")
        b.append(f"{(err[m].sum() / alive_n[m].sum()).item():5.2f}")
      else:
        a.append(f"{crossed[m].float().mean().item() * 100:5.0f}")
        b.append(f"{falls[m].mean().item() / minutes:5.2f}")
    print(f"{name:22s} " + " ".join(a))
    print(f"{'':22s} " + " ".join(b))
  print("mean swing peak, foot centre above the ground under the foot (cm):")
  for t, name in enumerate(names):
    cells = []
    for lvl in range(rows):
      m = (env_type == t) & (level == lvl)
      cells.append(f"{(peak_sum[m].sum() / peak_cnt[m].sum().clamp(min=1)).item() * 100:5.1f}")
    print(f"{name:22s} " + " ".join(cells))
  if leg is not None and mode == "mix":
    print("thigh/calf on the ground, share of steps, top 3 levels:")
    for t, name in enumerate(names):
      m = (env_type == t) & (level >= rows - 3)
      print(f"  {name:22s} {(leg_steps[m].sum() / alive_n[m].sum()).item() * 100:5.1f}%")
  print(f"overall falls/robot/min {falls.mean().item() / minutes:.3f}, xy err {(err.sum() / alive_n.sum()).item():.3f}")


if __name__ == "__main__":
  sys.exit(tyro.cli(main))
