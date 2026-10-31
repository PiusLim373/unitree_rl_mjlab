"""Script to record a skill clip for goaway_ui's skill cards.

Renders one episode of a trained policy offscreen (no viewer window; play.py --video opens
one afterwards), loading it as play.py does, and writes a muted H.264 mp4 that plays in
every browser, iPad/Safari included. The camera follows base_link from the front-left,
tilted down so the horizon (MuJoCo's black sky) stays out of frame.

--seconds is the loop length: one skill plus a short hold, starting from the episode reset.
Check the clip before using it, then copy it to goaway_ui/public/skills/ and add the skill
to SKILLS in goaway_ui's QuadrupedWidget.jsx.

Run from the repo root with MUJOCO_GL=egl. Examples (the clips goaway_ui uses):
  MUJOCO_GL=egl python scripts/record_skill_clip.py --task Unitree-Go2-DonkeyKick \
    --checkpoint logs/rsl_rl/go2_kick/<run>/model_6000.pt --output donkey_kick.mp4 --seconds 2.4
  MUJOCO_GL=egl python scripts/record_skill_clip.py --task Unitree-Go2-QuickBuck \
    --checkpoint logs/rsl_rl/go2_buck/<run>/model_6000.pt --output quick_buck.mp4 --seconds 1.6
  MUJOCO_GL=egl python scripts/record_skill_clip.py --task Unitree-Go2-Recovery \
    --checkpoint logs/rsl_rl/go2_recovery/<run>/model_5000.pt --output recovery.mp4 --seconds 1.6
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from ckpt_compat import match_actor_history


@dataclass(frozen=True)
class RecordConfig:
  task: str
  """Task the checkpoint was trained on, e.g. Unitree-Go2-DonkeyKick."""
  checkpoint: str
  """Checkpoint (model_N.pt) to play."""
  output: str
  """mp4 file to write."""
  seconds: float = 2.0
  """Clip length from the episode reset."""
  width: int = 448
  height: int = 252
  """16:9, the skill cards' shape."""
  distance: float = 2.2
  elevation: float = -28.0
  azimuth: float = 135.0
  flat: bool = True
  """Flat plane instead of the task's play terrain."""
  seed: int = 0
  device: str | None = None


def main():
  # Populate task registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = tyro.cli(RecordConfig, prog=sys.argv[0])
  ckpt = Path(cfg.checkpoint)
  if not ckpt.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  match_actor_history(env_cfg, ckpt)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  if cfg.flat and env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
  viewer = env_cfg.viewer
  viewer.width, viewer.height = cfg.width, cfg.height
  viewer.distance, viewer.elevation, viewer.azimuth = cfg.distance, cfg.elevation, cfg.azimuth
  viewer.max_extra_envs = 0

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  step_dt = env.unwrapped.step_dt
  frames = []
  with torch.inference_mode():
    for _ in range(round(cfg.seconds / step_dt)):
      env.step(policy(env.get_observations()))
      frames.append(env.unwrapped.render())
  env.close()

  imageio.mimwrite(
    cfg.output, frames, fps=round(1.0 / step_dt), codec="libx264", quality=5,
    pixelformat="yuv420p", macro_block_size=2,
    output_params=["-profile:v", "main", "-movflags", "+faststart", "-an"],
  )
  print(f"[INFO]: Wrote {cfg.output} ({len(frames)} frames, {cfg.seconds} s)")


if __name__ == "__main__":
  main()
