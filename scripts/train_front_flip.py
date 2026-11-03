"""Train the front flip from scratch in stages: the multistage recipe.

v6 to v13 built the front flip by fine-tuning one run after another, changing the rules
by hand in between. This runs the same steps as one recipe (front_flip_stage in
src/tasks/kick/flip_env_cfg.py). Each stage trains its task from the previous stage's last
checkpoint (the iteration count carries on, so the penalty ramp and the exploration switch
run as in one long run), copies the flip code into the run's code/ folder, evaluates the
last checkpoint (scripts/eval_flip.py; the checks read the mirror-averaged policy, what
deploys, and count rear feet as paired within the stage's own catch window) and stops
unless the stage's pass check holds (results at least / at most a bound).

v2: legs under the hips before the touchdown (landing_pose) from stage 1, mid-flip starts
from the stand pose, feet impact priced like the knees' in stage 4 (4000 iterations, upper
limits on knee touches and landing force), and stage 1 also checks the rear knees' height
at the touchdown (v1's folded legs came from stage 1). A first v2 with landing_pose in
stage 4 only (from v1's stage 3) didn't change the posture: kept as *_abandoned. A second,
from scratch with landing_pose on the thighs and calves only, splayed its rear legs out at
the hips and failed stage 1's knee check: kept as *_abandoned2. landing_pose now scores
all the leg joints, with a tighter pose_var.

v3: stage 4 only, from v2's stage 3: the feet's impact costs twice as much, from a lower
force (v2's front feet landed hard, the rear softly). v2's code is in
logs/rsl_rl/go2_frontflip/frontflip_v2_multistage_code_backup.

Runs are named frontflip_v<version>-<stage>_multistage. The summary (runs, checkpoints,
results, checks, status) is logs/rsl_rl/go2_frontflip/frontflip_v<version>_multistage.json;
each stage's training output goes next to it (..._stdout.log), its eval into the run
folder (eval.log, eval.json).

  python scripts/train_front_flip.py --version 1
  # Any change to the recipe is a new version; e.g. only stage 4 again, from an earlier
  # version's stage 3 checkpoint:
  python scripts/train_front_flip.py --version 3 --start-stage 4 \
    --from-checkpoint logs/rsl_rl/go2_frontflip/<run>/model_10000.pt
"""

import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro

EXPERIMENT = Path("logs/rsl_rl/go2_frontflip")
EVAL_TASK = "Unitree-Go2-FrontFlip"
# The flip's code, copied into each stage's run folder.
CODE = (
  "src/tasks/kick/mdp/flip.py",
  "src/tasks/kick/flip_env_cfg.py",
  "src/tasks/kick/config/go2/env_cfgs.py",
  "src/tasks/kick/config/go2/__init__.py",
  "scripts/eval_flip.py",
  "scripts/train_front_flip.py",
)


@dataclass(frozen=True)
class Stage:
  task: str
  iterations: int
  checks: dict[str, float]  # eval result (%) -> the least that passes
  goal: str
  limits: dict[str, float] = field(default_factory=dict)  # eval result -> the most that passes


# The lengths follow the fine-tune runs each stage repeats (v6, v8, v9, v10; stage 4 longer
# from v2: v1's was still improving at its end). Stage 1's knee height: v1's ~3 cm, the
# stand pose's well over 10.
STAGES = {
  1: Stage(
    "Unitree-Go2-FrontFlip-S1", 6000, {"paired": 70.0, "rear_knee_td_cm": 8.0},
    "over its back, caught on both rear feet, legs under the hips",
  ),
  2: Stage("Unitree-Go2-FrontFlip-S2", 1000, {"landed_up": 80.0}, "a clean landing, still up at the end"),
  3: Stage("Unitree-Go2-FrontFlip-S3", 3000, {"success": 85.0}, "up off the knees, standing still"),
  4: Stage(
    EVAL_TASK, 4000, {"success": 90.0, "flat": 99.0}, "legs under the hips, no knee slam (the full FrontFlip)",
    {"knee_touch": 25.0, "landing_force": 450.0},
  ),
}


@dataclass(frozen=True)
class TrainFrontFlipConfig:
  version: int
  """The recipe's version (runs frontflip_v<version>-<stage>_multistage)."""
  start_stage: int = 1
  from_checkpoint: str | None = None
  """Where a later start_stage begins (default: this version's previous stage)."""
  num_envs: int = 1024
  save_interval: int = 500
  iterations: tuple[int, ...] = ()
  """Override each stage's iterations (tests)."""
  checks: bool = True
  """Stop when a stage fails its pass check (off: record it and go on; tests)."""
  eval_envs: int = 1024
  eval_rounds: int = 2


def run_name(version: int, stage: int) -> str:
  return f"frontflip_v{version}-{stage}_multistage"


def last_checkpoint(run: Path) -> Path | None:
  models = sorted(run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
  return models[-1] if models else None


def main():
  from src.tasks.kick.flip_env_cfg import front_flip_stage

  cfg = tyro.cli(TrainFrontFlipConfig, prog=sys.argv[0])
  if not Path("scripts/train.py").exists():
    sys.exit("Run from the unitree_rl_mjlab folder.")
  if cfg.start_stage not in STAGES:
    sys.exit(f"--start-stage: one of {list(STAGES)}")
  EXPERIMENT.mkdir(parents=True, exist_ok=True)
  summary_path = EXPERIMENT / f"frontflip_v{cfg.version}_multistage.json"
  if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    if cfg.start_stage == 1 or any(s["stage"] >= cfg.start_stage for s in summary["stages"]):
      sys.exit(f"{summary_path} already has these stages: a changed recipe is a new --version.")
  else:
    summary = {"version": cfg.version, "stages": []}

  ckpt = Path(cfg.from_checkpoint) if cfg.from_checkpoint else None
  if ckpt is None and cfg.start_stage > 1:
    prev = [s for s in summary["stages"] if s["stage"] == cfg.start_stage - 1 and s["passed"]]
    if not prev:
      sys.exit(f"--from-checkpoint: no passed stage {cfg.start_stage - 1} in {summary_path}")
    ckpt = Path(prev[-1]["checkpoint"])
  if ckpt is not None and (not ckpt.exists() or ckpt.parent.parent.resolve() != EXPERIMENT.resolve()):
    sys.exit(f"{ckpt}: not a checkpoint in {EXPERIMENT}")

  def save(status):
    summary["status"] = status
    summary_path.write_text(json.dumps(summary, indent=2))

  for n in range(cfg.start_stage, max(STAGES) + 1):
    stage = STAGES[n]
    iterations = cfg.iterations[n - 1] if cfg.iterations else stage.iterations
    name = run_name(cfg.version, n)
    code = {f: Path(f).read_bytes() for f in CODE}  # as this stage trains with it
    save(f"stage {n} training")
    print(f"[STAGE {n}] {stage.task}: {stage.goal}; {iterations} iterations from {ckpt or 'scratch'}", flush=True)
    # One more than the stage's length, so it ends on a round checkpoint (rsl_rl saves the
    # last iteration it ran, and a resume starts again at the checkpoint's).
    cmd = [
      sys.executable, "scripts/train.py", stage.task,
      f"--env.scene.num-envs={cfg.num_envs}", "--agent.logger=tensorboard", f"--agent.run-name={name}",
      f"--agent.save-interval={cfg.save_interval}", f"--agent.max-iterations={iterations + 1}",
    ]
    if ckpt is not None:
      cmd += [
        "--agent.resume=True", f"--agent.load-run={re.escape(ckpt.parent.name)}$",
        f"--agent.load-checkpoint={re.escape(ckpt.name)}$",
      ]
    t0 = time.time()
    with open(EXPERIMENT / f"{name}_stdout.log", "w") as log:
      code_ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    runs = sorted(EXPERIMENT.glob(f"*_{name}"))
    run = runs[-1] if runs else None
    new_ckpt = last_checkpoint(run) if run else None
    if not code_ok or new_ckpt is None:
      save(f"stopped: stage {n} training failed (see {name}_stdout.log)")
      sys.exit(2)
    (run / "code").mkdir(exist_ok=True)
    for f, data in code.items():
      (run / "code" / Path(f).name).write_bytes(data)

    print(f"[STAGE {n}] trained: {new_ckpt}; evaluating", flush=True)
    result_path = run / "eval.json"
    cmd = [
      sys.executable, "scripts/eval_flip.py", "--task", EVAL_TASK, "--policies", f"s{n}={new_ckpt}",
      "--num-envs", str(cfg.eval_envs), "--rounds", str(cfg.eval_rounds), "--json", str(result_path),
      "--pair-window", str(front_flip_stage(n)["catch"]["pair_window"]),
    ]
    with open(run / "eval.log", "w") as log:
      eval_ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    results = json.loads(result_path.read_text()).get(f"s{n}-avg", {}) if eval_ok and result_path.exists() else {}
    # [result, bound, passed]; a missing or nan result fails.
    checks = {k: [results.get(k), f">= {least}", results.get(k, -1.0) >= least] for k, least in stage.checks.items()}
    checks |= {k: [results.get(k), f"<= {most}", results.get(k, math.inf) <= most] for k, most in stage.limits.items()}
    passed = eval_ok and all(ok for _, _, ok in checks.values())
    summary["stages"].append({
      "stage": n, "task": stage.task, "goal": stage.goal, "run": str(run), "from": str(ckpt) if ckpt else None,
      "checkpoint": str(new_ckpt), "train_min": round((time.time() - t0) / 60), "eval": results,
      "checks": checks, "passed": passed,
    })
    print(f"[STAGE {n}] checks: {checks} -> {'passed' if passed else 'FAILED'}", flush=True)
    if not passed and cfg.checks:
      save(f"stopped: stage {n} failed its pass check")
      sys.exit(1)
    ckpt = new_ckpt
  save("done")
  print(f"[DONE] {summary_path}", flush=True)


if __name__ == "__main__":
  main()
