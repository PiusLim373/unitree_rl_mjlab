"""Script to evaluate Recovery (get-up) policies on the same falls.

Each policy runs deterministically on the task's start distribution (same seeds, so the
same falls), with the joint damping it was trained with (read from its run's
params/env.yaml), for one 8 s fall per robot. Reports:
  - success: the deploy done check (DONE_CHECK) held for its hold time (0.5 s), and the
    time to get there, by start kind (settled / tumbling), start orientation, terrain,
    and on-steps vs platform for stair tiles
  - gentleness: peaks after the drop (joint speed, torque vs limit, body force, turn rate)
  - failure modes at the end of the 8 s
  - with --slams: touchdowns per body part (belly, head, hips, thighs, shins), their
    peak force in the next 0.1 s, and which parts cause the slams (> 600 N)

Example:
  python scripts/eval_recovery.py --policies v2=logs/.../model_6000.pt v3=logs/.../model_6000.pt --slams
"""

import collections
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.torch import configure_torch_backends

TASK = "Unitree-Go2-Recovery"
RUN_S = 8.0  # one fall per robot (the training episode length)
LAND_S = 0.6  # the drop (<= ~0.75 m) and bounce; peaks before this aren't the policy's
SLAM_N = 600.0  # ~4x the Go2's weight
TOUCH_N, GAP_STEPS, WIN_STEPS = 5.0, 5, 5  # touchdown: > 5 N after >= 0.1 s apart; peak over 0.1 s


@dataclass(frozen=True)
class EvalRecoveryConfig:
  policies: tuple[str, ...]
  """name=checkpoint pairs (model_N.pt from a Unitree-Go2-Recovery run)."""
  rounds: int = 2
  num_envs: int = 1024
  slams: bool = False
  """Also report touchdowns and slams per body part."""
  device: str | None = None


def _yaml_loader():
  class Loader(yaml.SafeLoader):
    pass

  def any_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
      return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
      return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)

  Loader.add_multi_constructor("", any_tag)
  Loader.add_multi_constructor("tag:yaml.org,2002:python/", any_tag)
  return Loader


def run_damping(ckpt: str) -> dict[str, float] | None:
  """Joint damping per actuator group ('hip' / 'thigh' / 'calf') from the run's env.yaml."""
  path = os.path.join(os.path.dirname(ckpt), "params", "env.yaml")
  if not os.path.isfile(path):
    return None
  cfg = yaml.load(open(path), Loader=_yaml_loader())
  actuators = cfg["scene"]["entities"]["robot"]["articulation"]["actuators"]
  return {
    kind: float(a["damping"])
    for a in actuators
    for kind in ("hip", "thigh", "calf")
    if kind in a["target_names_expr"][0]
  }


def set_damping(env: ManagerBasedRlEnv, kd: dict[str, float]) -> None:
  """Set Kd of the robot's position actuators (MuJoCo: biasprm[2] = -kd)."""
  robot = env.scene["robot"]
  biasprm = env.sim.model.actuator_biasprm
  for actuator in robot.actuators:
    kind = next(k for k in kd if k in actuator.cfg.target_names_expr[0])
    ids = actuator.global_ctrl_ids
    if biasprm.dim() == 3:
      biasprm[:, ids, 2] = -kd[kind]
    else:
      biasprm[ids, 2] = -kd[kind]


def _part(geom: str) -> str:
  if geom.startswith("base1"):
    return "belly/body box"
  if geom.startswith(("base2", "base3")):
    return "head (front)"
  for key, name in (("hip", "hips"), ("thigh", "thighs"), ("calf", "shins")):
    if key in geom:
      return name
  raise ValueError(geom)


def main():
  # Populate task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.recovery.recovery_env_cfg import DONE_CHECK

  cfg = tyro.cli(EvalRecoveryConfig, prog=sys.argv[0])
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

  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs, env_cfg.seed = n, 0
  env_cfg.curriculum = {}
  env_cfg.episode_length_s = RUN_S + 1.0
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  tg = env_cfg.scene.terrain.terrain_generator
  tg.seed = 0
  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device=device)
  base = env.unwrapped
  robot, sensor, dt = base.scene["robot"], base.scene["body_ground"], base.step_dt
  max_tilt, max_err, max_vel = DONE_CHECK["max_tilt"], DONE_CHECK["max_pose_err"], DONE_CHECK["max_joint_vel"]
  hold_steps = int(round(0.5 / dt))
  limits = torch.tensor([45.0 if "calf" in a else 23.5 for a in robot.actuator_names], device=device)
  geoms = list(dict.fromkeys(s.primary_name for s in sensor._slots))
  parts = list(dict.fromkeys(_part(g) for g in geoms))
  member = torch.zeros(len(geoms), len(parts), device=device)
  for i, g in enumerate(geoms):
    member[i, parts.index(_part(g))] = 1.0
  size, nr, nc = tg.size, tg.num_rows, tg.num_cols

  def tile_of(xy):
    r = int(np.floor((xy[0] + nr * size[0] * 0.5) / size[0]))
    c = int(np.floor((xy[1] + nc * size[1] * 0.5) / size[1]))
    return tiles.get((r, c), "border")

  def tilt_now():
    return torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1))

  results = {}
  for name, ckpt in policies.items():
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    kd = run_damping(ckpt)
    if kd:
      set_damping(base, kd)
    print(f"[INFO] {name}: {ckpt} (damping {kd or 'task default'})")
    policy = runner.get_inference_policy(device=device)
    rows, events = [], []
    torch.manual_seed(0)
    for rnd in range(cfg.rounds):
      with torch.no_grad():
        obs, _ = env.reset()
        tilt0 = tilt_now().cpu().numpy()
        xy0 = robot.data.root_link_pos_w[:, :2].cpu().numpy()
        off0 = (robot.data.root_link_pos_w[:, :2] - base.scene.env_origins[:, :2]).abs().amax(1).cpu().numpy()
        settled0 = (robot.data.root_link_vel_w.abs().amax(1) <= 0.1001).cpu().numpy()
        held = torch.zeros(n, dtype=torch.int32, device=device)
        t_done = torch.full((n,), float("nan"), device=device)
        peak = {k: torch.zeros(n, device=device) for k in ("qd", "tau", "imp", "w")}
        P = len(parts)
        quiet = torch.full((n, P), GAP_STEPS, device=device)
        win_left = torch.zeros(n, P, dtype=torch.int32, device=device)
        win_peak = torch.zeros(n, P, device=device)
        win_meta = torch.zeros(n, P, 3, device=device)  # down speed, tilt, done
        for i in range(int(RUN_S / dt)):
          obs, *_ = env.step(policy(obs))
          now = (i + 1) * dt
          tilt = tilt_now()
          err = (robot.data.joint_pos - robot.data.default_joint_pos).abs().amax(1)
          qd = robot.data.joint_vel.abs().amax(1)
          ok = (tilt < max_tilt) & (err < max_err) & (qd < max_vel)
          held = torch.where(ok, held + 1, torch.zeros_like(held))
          t_done = torch.where(torch.isnan(t_done) & (held >= hold_steps), torch.full_like(t_done, now), t_done)
          f_geom = torch.norm(sensor.data.force_history, dim=-1).amax(dim=-1)
          if now >= LAND_S:
            peak["qd"] = torch.maximum(peak["qd"], qd)
            peak["tau"] = torch.maximum(peak["tau"], (robot.data.actuator_force.abs() / limits).amax(1))
            peak["imp"] = torch.maximum(peak["imp"], f_geom.amax(1))
            peak["w"] = torch.maximum(peak["w"], robot.data.root_link_ang_vel_b.norm(dim=1))
          if cfg.slams:
            f_part = (f_geom.unsqueeze(-1) * member.unsqueeze(0)).amax(dim=1)
            touching = f_part > TOUCH_N
            for e, p in torch.nonzero(win_left == 1).tolist():
              m = win_meta[e, p].tolist()
              events.append((parts[p], m[0], m[1], float(win_peak[e, p]), bool(m[2]), (rnd, e)))
            win_left = torch.clamp(win_left - 1, min=0)
            win_peak = torch.where(win_left > 0, torch.maximum(win_peak, f_part), win_peak)
            new = touching & (quiet >= GAP_STEPS) & (win_left == 0) & (now >= LAND_S)
            if new.any():
              meta = torch.stack([-robot.data.root_link_lin_vel_w[:, 2], tilt, ~torch.isnan(t_done)], dim=1).float()
              win_meta = torch.where(new.unsqueeze(-1), meta.unsqueeze(1).expand(-1, P, -1), win_meta)
              win_peak = torch.where(new, f_part, win_peak)
              win_left = torch.where(new, torch.full_like(win_left, WIN_STEPS), win_left)
            quiet = torch.where(touching, torch.zeros_like(quiet), quiet + 1)
        end_tilt, end_err, end_qd = tilt.cpu().numpy(), err.cpu().numpy(), qd.cpu().numpy()
      for e in range(n):
        tile = tile_of(xy0[e])
        rows.append(dict(
          kind="settled" if settled0[e] else "tumbling",
          orient="on back" if tilt0[e] > 2.4 else ("on side" if tilt0[e] > 0.8 else "upright-ish"),
          terrain=tile,
          place=("on steps" if off0[e] > 1.5 else "platform") if "stairs" in tile else "-",
          t_done=float(t_done[e]), **{k: float(v[e]) for k, v in peak.items()},
          end_tilt=float(end_tilt[e]), end_err=float(end_err[e]), end_qd=float(end_qd[e]),
        ))
    results[name] = dict(rows=rows, events=events)
  env.close()

  names = list(results)
  falls = len(results[names[0]]["rows"])
  print(f"\n===== {falls} falls per policy ({cfg.rounds} x {n}), the same falls for each; "
        f"done check tilt < {max_tilt}, joints within {max_err} rad, slower than {max_vel} rad/s, for 0.5 s =====")

  def success(group):
    t = [r["t_done"] for r in group if not np.isnan(r["t_done"])]
    if not t:
      return 0.0, np.nan, np.nan
    return 100 * len(t) / len(group), np.median(t), np.percentile(t, 90)

  def table(title, key=None, keep=lambda r: True):
    print(f"\n{title}  (success | median s | 90th s)")
    keys = sorted({r[key] if key else "all" for r in results[names[0]]["rows"] if keep(r)})
    for k in keys:
      cells = []
      for nm in names:
        g = [r for r in results[nm]["rows"] if keep(r) and (key is None or r[key] == k)]
        s, med, p90 = success(g)
        cells.append(f"{nm}: {s:5.1f}% {med:4.2f} {p90:4.2f}")
      count = sum(1 for r in results[names[0]]["rows"] if keep(r) and (key is None or r[key] == k))
      print(f"  {k:22s} n={count:5d} | " + " | ".join(cells))

  table("overall")
  table("by start kind", "kind")
  table("by start orientation", "orient")
  table("by terrain", "terrain")
  table("stair tiles: on steps vs platform", "place", lambda r: "stairs" in r["terrain"])

  print(f"\ngentleness: peaks per fall after the first {LAND_S} s  (median / 95th, share over the line)")
  for key, label, line in (("qd", "joint speed rad/s", 20.0), ("tau", "torque / limit", 0.99),
                           ("imp", "body force N", SLAM_N), ("w", "body turn rate rad/s", None)):
    cells = []
    for nm in names:
      x = np.array([r[key] for r in results[nm]["rows"]])
      over = f", >{line:g}: {100 * (x > line).mean():4.1f}%" if line is not None else ""
      cells.append(f"{nm}: {np.median(x):7.2f} / {np.percentile(x, 95):8.2f}{over}")
    print(f"  {label:22s} " + " | ".join(cells))

  for nm in names:
    failed = [r for r in results[nm]["rows"] if np.isnan(r["t_done"])]
    modes = collections.Counter()
    for r in failed:
      if r["end_tilt"] > 2.4:
        modes["still on back"] += 1
      elif r["end_tilt"] > max_tilt:
        modes["on side / tilted"] += 1
      elif r["end_err"] > max_err:
        modes["upright, joints off pose"] += 1
      elif r["end_qd"] > max_vel:
        modes["in pose, still moving"] += 1
      else:
        modes["meets check at end, held < 0.5 s"] += 1
    print(f"failures {nm} ({len(failed)}): {dict(modes.most_common())}")

  if not cfg.slams:
    return
  for nm in names:
    ev = results[nm]["events"]
    slams = [e for e in ev if e[3] > SLAM_N]
    print(f"\n===== {nm}: touchdowns after {LAND_S} s: {len(ev)}; slams (> {SLAM_N:.0f} N within 0.1 s): "
          f"{len(slams)} = {len(slams) / falls:.2f} per fall =====")
    print(f"  {'part':16s} {'per fall':>8s} | {'down speed m/s med/95th':>23s} | {'peak N med/95th':>16s} | {'> 600 N':>7s} | share of slams")
    for p in parts:
      g = [e for e in ev if e[0] == p]
      if not g:
        continue
      v, f = np.array([e[1] for e in g]), np.array([e[3] for e in g])
      share = 100 * sum(1 for e in slams if e[0] == p) / max(len(slams), 1)
      print(f"  {p:16s} {len(g) / falls:8.2f} | {np.median(v):10.2f} / {np.percentile(v, 95):5.2f}  | "
            f"{np.median(f):6.0f} / {np.percentile(f, 95):6.0f} | {100 * (f > SLAM_N).mean():6.1f}% | {share:5.1f}%")
    head_falls = len({e[5] for e in slams if e[0] == "head (front)"})
    print(f"  falls with a head slam: {100 * head_falls / falls:.1f}%")
    print(f"  slams during recovery / after done: {sum(1 for e in slams if not e[4])} / {sum(1 for e in slams if e[4])}")


if __name__ == "__main__":
  main()
