"""Train UprightWalk from scratch in stages (the multistage recipe, as train_front_flip.py).

Each stage trains its task from the previous stage's last checkpoint (the iteration count
carries on, so the penalty ramp and the command curriculum run as in one long run), copies
the task's code into the run's code/ folder, evaluates its last checkpoints, newest first
(scripts/eval_upright.py, mirror-averaged: what deploys), goes on from the one that passes
the stage's check (results at least / at most a bound) furthest inside it, as the results
swing from one checkpoint to the next, and stops if none does. Stages
(src/tasks/upright/upright_env_cfg.py):
  S1 rising from four legs, standing and walking upright (from scratch, with mixed starts),
  coming down once the rise is found; S2 + the handshake and the wave.
Every stage's check includes the estimated battery current (the real Go2 cuts its motors
at a high bus current).

Runs are named uprightwalk_v<version>-<stage>_multistage. The summary (runs, checkpoints,
results, checks, status) is logs/rsl_rl/go2_upright/uprightwalk_v<version>_multistage.json;
each stage's training output goes next to it (..._stdout.log), its eval into the run
folder (eval_model_<N>.log, .json).

  python scripts/train_upright.py --version 4
  python scripts/train_upright.py --version 4 --start-stage 2 --from-checkpoint <model_N.pt>
"""

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro

EXPERIMENT = Path("logs/rsl_rl/go2_upright")
# Checkpoints tried per stage: the rise changes a lot from one to the next.
CANDIDATES = 4
# The task's code, copied into each stage's run folder.
CODE = (
  "src/tasks/upright/upright_env_cfg.py",
  "src/tasks/upright/mdp/actions.py",
  "src/tasks/upright/mdp/commands.py",
  "src/tasks/upright/mdp/events.py",
  "src/tasks/upright/mdp/frames.py",
  "src/tasks/upright/mdp/power.py",
  "src/tasks/upright/mdp/upright.py",
  "src/tasks/upright/config/go2/env_cfgs.py",
  "src/tasks/velocity/rl/symmetry.py",
  "src/assets/robots/unitree_go2/go2_constants.py",
  "scripts/eval_upright.py",
  "scripts/chain_upright.py",
  "scripts/train_upright.py",
)


@dataclass(frozen=True)
class Stage:
  task: str
  iterations: int
  checks: dict[str, float]  # eval result -> the least that passes
  goal: str
  limits: dict[str, float] = field(default_factory=dict)  # eval result -> the most that passes
  entropy_coef: float | None = None  # in place of the runner cfg's
  chain: bool = False  # also run every skill back to back (scripts/chain_upright.py)


# Battery current (A, eval_power's model): the 4-leg walk peaks ~27 A (p95 36) and holds
# ~13 A (p95 18) over 50 ms; the side flip, safe on the robot, held ~41 A. Motors at full
# torque: the walk ~1.2, DonkeyKick ~2.
def _power(scenario: str, motors_full: float = 2.0) -> dict[str, float]:
  return {f"{scenario}_peak_A_p95": 50.0, f"{scenario}_held50_A_p95": 30.0, f"{scenario}_motors_full": motors_full}


# The rises are short bursts: their motor load may run higher, their current limits stay.
_RISE_POWER = _power("rise", motors_full=2.5)
_AGAIN_POWER = _power("again", motors_full=2.5)


# Arms touching the body or rear legs (% of robots; the sim has these collisions, so the
# real robot would too). The handshake's is only reported: the simulated hand pushes the paw.
_ARMS = {"walk_arm_touch": 5.0, "rise_arm_touch": 5.0}


# A thigh or calf on the ground (% of the time): it stands, walks and rises on its feet, and
# the handshake is done standing, not kneeling.
_UP_KNEES = {"walk_leg_touch": 5.0, "rise_leg_touch": 5.0, "stand_leg_touch": 5.0}
_KNEES = {"hand_leg_touch": 5.0, "hand_free_leg_touch": 5.0, "hand_let_go_leg_touch": 5.0}


# Standing at zero cmd_vel (settled, eval_upright's SETTLE_S): small balance taps, well under
# marching on the 0.4 s clock (5 a second), and the arms down (paw height from the base, m).
# Sideways isn't checked: upright no joint pushes a foot sideways, so it's a bonus (held_side_*).
_WALK = {"walk_err_vx": 0.2, "walk_err_wz": 0.3, "walk_standing_steps_per_s": 3.0,
         "walk_steps_per_s": 6.0, "walk_standing_arm_hip": 0.25, "walk_standing_arm_height": -0.08}
# Standing steps and turning from S2 on: S1's action noise keeps them above the limits, S2's
# lower entropy brings them down (v4's S1 turned at a third of the speed, fixed in S2).
_WALK_S1 = {k: v for k, v in _WALK.items() if k not in ("walk_standing_steps_per_s", "walk_err_wz")}
# 30 s at zero cmd_vel, median, every stage: with the IMU tilt v2 drifts ~2 m and turns ~29 deg.
_STAY = {"stand_drift_m": 1.0, "stand_turn_deg": 20.0}
# Steps left and right, not a two-footed hop.
_STEPS = {"walk_survived": 95.0, "walk_one_foot_down": 50.0}
# Coming down: front feet peak force (N, p95), about twice the robot's weight.
_DOWN = {"rise_front_impact_N_p95": 300.0}
# After S1, less exploration bonus: the rise is found in S1 (runner cfg's 0.01); a later
# stage that doubled its action noise stood with more taps, one at 0.005 never rose.
_ENTROPY = 0.002
_RISE = {"rise_risen": 95.0, "rise_survived": 98.0}
_AGAIN = {"again_risen": 95.0, "again_survived": 98.0}
STAGES = {
  # 10000: the come-down starts at 1500 and is found late (v3's side tests needed ~3600).
  1: Stage(
    "Unitree-Go2-UprightWalk-S1", 10000, {**_STEPS, **_RISE, **_AGAIN, "rise_down": 98.0},
    "rising from four legs, upright on the rear legs, walking on cmd_vel, coming back down",
    {**_WALK_S1, **_STAY, **_UP_KNEES, **_ARMS, **_power("walk"), **_RISE_POWER, **_AGAIN_POWER, **_DOWN},
  ),
  # 6000: the down-with-arm-out falls and the drift settle late. Wave taps up to 3.5 a second:
  # the arm's swing sways the body, so they stay above standing's.
  2: Stage(
    "Unitree-Go2-UprightWalk", 6000,
    {**_STEPS, **_RISE, **_AGAIN, "rise_down": 98.0, "hand_upright": 95.0, "hand_free_upright": 95.0,
     "hand_let_go_upright": 95.0, "hand_wave_upright": 95.0, "hand_wave_survived": 98.0,
     "hand_wave_paw_sweep_m": 0.1, "hand_long_free_survived": 98.0, "hand_long_wave_survived": 98.0,
     "held_fwd_05": 0.35, "held_back_05": 0.3, "held_turn_1": 0.7, "chain_finished": 97.0},
    "the handshake and the wave: balance with an arm out, soft or moving",
    {**_WALK, **_STAY, "stand_drift_m": 0.6, **_UP_KNEES, **_ARMS, **_power("walk"), **_RISE_POWER, **_AGAIN_POWER,
     **_power("hand"), **_power("hand_wave"), **_DOWN, **_KNEES, "hand_wave_leg_touch": 5.0, "hand_wave_arm_touch": 5.0,
     "hand_wave_steps_per_s": 3.5, "hand_drift_m": 0.3, "hand_free_drift_m": 0.4, "hand_let_go_drift_m": 0.4,
     "hand_wave_drift_m": 0.5, "hand_turn_deg": 20.0, "hand_free_turn_deg": 20.0, "hand_let_go_turn_deg": 20.0,
     "hand_wave_turn_deg": 20.0, "hand_long_free_drift_m": 1.5, "hand_long_wave_drift_m": 1.5},
    _ENTROPY, chain=True,
  ),
}


@dataclass(frozen=True)
class TrainUprightConfig:
  version: int
  """The recipe's version (runs uprightwalk_v<version>-<stage>_multistage)."""
  start_stage: int = 1
  from_checkpoint: str | None = None
  """Where a later start_stage begins (default: this version's previous stage)."""
  num_envs: int = 1024
  save_interval: int = 500
  iterations: tuple[int, ...] = ()
  """Override each stage's iterations (tests)."""
  entropy_coefs: tuple[float, ...] = ()
  """Override each stage's entropy coefficient, as --iterations (more robots squeeze the noise harder)."""
  checks: bool = True
  """Stop when a stage fails its pass check (off: record it and go on; tests)."""
  eval_envs: int = 512


def run_name(version: int, stage: int) -> str:
  return f"uprightwalk_v{version}-{stage}_multistage"


def iteration(model: Path) -> int:
  return int(re.findall(r"\d+", model.stem)[0])


def evaluate(stage: Stage, n: int, model: Path, run: Path, eval_envs: int) -> tuple[dict, dict, bool]:
  """A checkpoint's eval: its results, the stage's checks on them and whether all hold."""
  result_path = run / f"eval_{model.stem}.json"
  cmd = [
    sys.executable, "scripts/eval_upright.py", "--task", stage.task, "--stage", str(n),
    "--policies", f"s{n}={model}", "--num-envs", str(eval_envs), "--json", str(result_path),
  ]
  with open(run / f"eval_{model.stem}.log", "w") as log:
    eval_ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
  results = json.loads(result_path.read_text()).get(f"s{n}", {}) if eval_ok and result_path.exists() else {}
  if stage.chain:
    chain_path = run / f"chain_{model.stem}.json"
    cmd = [sys.executable, "scripts/chain_upright.py", "--task", stage.task, "--policy", str(model),
           "--num-envs", str(eval_envs), "--json", str(chain_path)]
    with open(run / f"chain_{model.stem}.log", "w") as log:
      eval_ok &= subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    if chain_path.exists():
      results["chain_finished"] = json.loads(chain_path.read_text())["finished"]
  # [result, bound, passed]; a missing or nan result fails.
  checks = {k: [results.get(k), f">= {least}", results.get(k, -1.0) >= least] for k, least in stage.checks.items()}
  checks |= {k: [results.get(k), f"<= {most}", results.get(k, math.inf) <= most] for k, most in stage.limits.items()}
  return results, checks, eval_ok and all(ok for _, _, ok in checks.values())


def closeness(stage: Stage, results: dict) -> float:
  """Mean of each result over its bound (1 = on it): the lower, the further inside them."""
  ratios = [results[k] / most for k, most in stage.limits.items() if most > 0]
  ratios += [least / results[k] for k, least in stage.checks.items()]
  return sum(ratios) / len(ratios)


def main():
  cfg = tyro.cli(TrainUprightConfig, prog=sys.argv[0])
  if not Path("scripts/train.py").exists():
    sys.exit("Run from the unitree_rl_mjlab folder.")
  if cfg.start_stage not in STAGES:
    sys.exit(f"--start-stage: one of {list(STAGES)}")
  EXPERIMENT.mkdir(parents=True, exist_ok=True)
  summary_path = EXPERIMENT / f"uprightwalk_v{cfg.version}_multistage.json"
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
    entropy_coef = cfg.entropy_coefs[n - 1] if cfg.entropy_coefs else stage.entropy_coef
    name = run_name(cfg.version, n)
    code = {f: Path(f).read_bytes() for f in CODE}  # as this stage trains with it
    save(f"stage {n} training")
    print(f"[STAGE {n}] {stage.task}: {stage.goal}; {iterations} iterations from {ckpt or 'scratch'}", flush=True)
    # One more than the stage's length, so it ends on a round checkpoint.
    cmd = [
      sys.executable, "scripts/train.py", stage.task,
      f"--env.scene.num-envs={cfg.num_envs}", "--agent.logger=tensorboard", f"--agent.run-name={name}",
      f"--agent.save-interval={cfg.save_interval}", f"--agent.max-iterations={iterations + 1}",
    ]
    if entropy_coef is not None:
      cmd.append(f"--agent.algorithm.entropy-coef={entropy_coef}")
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
    # Saved by this stage, newest first (a resumed run also holds its starting checkpoint).
    start = iteration(ckpt) if ckpt else -1
    models = sorted((m for m in run.glob("model_*.pt") if iteration(m) > start), key=iteration, reverse=True) if run else []
    if not code_ok or not models:
      save(f"stopped: stage {n} training failed (see {name}_stdout.log)")
      sys.exit(2)
    (run / "code").mkdir(exist_ok=True)
    for f, data in code.items():
      (run / "code" / Path(f).name).write_bytes(data)

    tried, evaluated = {}, []
    for new_ckpt in models[:CANDIDATES] if cfg.checks else models[:1]:
      print(f"[STAGE {n}] evaluating {new_ckpt}", flush=True)
      results, checks, passed = evaluate(stage, n, new_ckpt, run, cfg.eval_envs)
      evaluated.append((new_ckpt, results, checks, passed))
      tried[new_ckpt.name] = [k for k, (_, _, ok) in checks.items() if not ok]
      print(f"[STAGE {n}] {new_ckpt.name} failed: {tried[new_ckpt.name]}", flush=True)
    passing = [e for e in evaluated if e[3]]
    new_ckpt, results, checks, passed = min(passing, key=lambda e: closeness(stage, e[1])) if passing else evaluated[0]
    summary["stages"].append({
      "stage": n, "task": stage.task, "goal": stage.goal, "run": str(run), "from": str(ckpt) if ckpt else None,
      "checkpoint": str(new_ckpt), "entropy_coef": entropy_coef,
      "train_min": round((time.time() - t0) / 60), "eval": results,
      "checks": checks, "passed": passed, "tried": tried,
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
