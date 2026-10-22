"""Script to benchmark Go2 velocity policies on flat and rough terrain.

Runs each policy headless on the same terrain, seed and command ranges, and
prints fall rate and velocity tracking error side by side.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends

from ckpt_compat import match_actor_history

# Terrain name -> task whose play env provides that terrain. Both tasks share
# the same actor observations, so a checkpoint from either loads into both.
TERRAIN_TASKS = {
  "flat": "Unitree-Go2-Flat",
  "rough": "Unitree-Go2-Rough-Blind",
}


@dataclass(frozen=True)
class EvalConfig:
  flat_ckpt: str | None = None
  """Checkpoint (model_N.pt) trained on Unitree-Go2-Flat."""
  rough_blind_ckpt: str | None = None
  """Checkpoint (model_N.pt) trained on Unitree-Go2-Rough-Blind."""
  terrains: tuple[Literal["flat", "rough"], ...] = ("flat", "rough")
  num_envs: int = 1024
  duration_s: float = 60.0
  seed: int = 0
  device: str | None = None
  lin_vel_x: tuple[float, float] = (-0.5, 1.0)
  lin_vel_y: tuple[float, float] = (-0.5, 0.5)
  ang_vel_z: tuple[float, float] = (-0.5, 0.5)


def evaluate(task_id: str, ckpt: Path, cfg: EvalConfig, device: str) -> dict[str, float]:
  env_cfg = load_env_cfg(task_id, play=True)
  match_actor_history(env_cfg, ckpt)
  # Hundreds of robots need the training contact budget; a play config may size
  # it automatically for a few robots in the viewer, which runs out of GPU memory.
  env_cfg.sim.nconmax = load_env_cfg(task_id, play=False).sim.nconmax
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  # Fix the terrain layout too, so every policy sees the same rough ground.
  if env_cfg.scene.terrain is not None and env_cfg.scene.terrain.terrain_generator is not None:
    env_cfg.scene.terrain.terrain_generator.seed = cfg.seed
  twist_cmd = env_cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = cfg.lin_vel_x
  twist_cmd.ranges.lin_vel_y = cfg.lin_vel_y
  twist_cmd.ranges.ang_vel_z = cfg.ang_vel_z

  agent_cfg = load_rl_cfg(task_id)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )

  # Load the actor only, as play.py does; critic sizes differ between tasks.
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  base_env = env.unwrapped
  robot = base_env.scene["robot"]
  num_steps = int(cfg.duration_s / base_env.step_dt)
  falls = torch.zeros(cfg.num_envs, device=device)
  err_xy_sum = torch.zeros((), device=device)
  err_yaw_sum = torch.zeros((), device=device)
  num_samples = torch.zeros((), device=device)

  with torch.no_grad():
    obs = env.get_observations()
    for _ in range(num_steps):
      obs, _, _, _ = env.step(policy(obs))
      # Play episodes never time out, so every termination is a fall.
      fell = base_env.reset_terminated
      falls += fell
      # Robots that fell were just reset, so skip them for tracking error.
      alive = ~fell
      cmd = base_env.command_manager.get_command("twist")
      err_xy = torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
      err_yaw = torch.abs(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
      err_xy_sum += err_xy[alive].sum()
      err_yaw_sum += err_yaw[alive].sum()
      num_samples += alive.sum()

  env.close()
  return {
    "falls_per_robot_min": (falls.sum() / cfg.num_envs / (cfg.duration_s / 60)).item(),
    "never_fell_pct": ((falls == 0).float().mean() * 100).item(),
    "err_xy": (err_xy_sum / num_samples).item(),
    "err_yaw": (err_yaw_sum / num_samples).item(),
  }


def main():
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  # Plain tyro syntax (not mjlab.TYRO_FLAGS), so collections are passed as
  # `--terrains rough` and `--lin-vel-x -1.0 2.0`.
  cfg = tyro.cli(EvalConfig, prog=sys.argv[0])

  policies = {
    name: Path(path)
    for name, path in (("flat", cfg.flat_ckpt), ("rough_blind", cfg.rough_blind_ckpt))
    if path is not None
  }
  if not policies:
    raise ValueError("Pass at least one of --flat-ckpt or --rough-blind-ckpt.")
  for path in policies.values():
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  table = PrettyTable(
    ["Policy", "Terrain", "Falls/robot/min", "Never fell (%)", "XY vel err (m/s)", "Yaw err (rad/s)"]
  )
  for terrain in cfg.terrains:
    for name, ckpt in policies.items():
      print(f"[INFO] Evaluating {name} on {terrain} ({ckpt})")
      r = evaluate(TERRAIN_TASKS[terrain], ckpt, cfg, device)
      table.add_row([
        name,
        terrain,
        f"{r['falls_per_robot_min']:.3f}",
        f"{r['never_fell_pct']:.1f}",
        f"{r['err_xy']:.3f}",
        f"{r['err_yaw']:.3f}",
      ])

  print(
    f"\n{cfg.num_envs} robots x {cfg.duration_s:.0f} s, seed {cfg.seed}, "
    f"commands vx={cfg.lin_vel_x} vy={cfg.lin_vel_y} wz={cfg.ang_vel_z}"
  )
  print(table)


if __name__ == "__main__":
  main()
