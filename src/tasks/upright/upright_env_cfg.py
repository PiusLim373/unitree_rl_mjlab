"""UprightWalk task configuration: standing and walking on the rear legs, with a handshake.

One policy rises from four legs, walks upright on cmd_vel (vx, vy, wz in the heading
frame, mdp/frames.py), stands quietly at zero cmd_vel (stepping only to keep its balance),
offers a soft paw and comes back down. Its inputs next to cmd_vel
are the up/down command and the hand flag (mdp/commands.py). In hand mode the paw leg is
taken over (mdp/actions.py, as the deploy will), so the policy only keeps its balance.

Built from the robot's Rough-Blind velocity configuration (to_upright_env_cfg): its
sensors, observations, randomisation and smoothness penalties stay; swapped are the
robot's default pose (upright: the action offset), the command, the starts, the rewards,
the fall check and the ground (flat and gentle rough only: the handshake is on a floor).

Multistage (scripts/train_upright.py), each stage from the previous one's checkpoint:
  S1: upright, partly risen and 4-leg starts: rising, standing and walking on cmd_vel (from
      scratch, while the exploration is wide: the rise is hard to find later); a fall onto
      the front feet is the safe exit, not a crash. Once the rise is found (DOWN_SCHEDULE),
      also the down command (come down, rise again)
  S2: also the hand mode: a soft paw out (a simulated hand shaking it), or the wave (an arm
      up and moving), held for long as the deploy has no time limit
The observations are the same in every stage (the up command and hand flag read 1 and 0
until they're used), so each stage resumes the previous one's network.

Power: the real Go2 cuts all motor power at a high bus current (the front flip's launch,
where the flips waived dc_torque_saturation). Nothing is waived here: DC saturation, the
PD-requested torque above each motor's limit and the estimated bus current above a budget
(mdp/power.py) cost in every stage and phase, ramped in only with the early penalty ramp.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
import mjlab.terrains as terrain_gen

import src.tasks.upright.mdp as mdp
from src.tasks.velocity.mdp import UniformVelocityCommandCfg

# Command ranges (m/s, rad/s): the last curriculum stage; play mode uses these.
LIN_VEL_X = (-0.5, 0.5)
LIN_VEL_Y = (-0.5, 0.5)
ANG_VEL_Z = (-1.0, 1.0)
# Smaller at first (S1 from scratch), full from iteration 3000.
COMMAND_STAGES = [
  {"step": 0, "lin_vel_x": (-0.2, 0.3), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-0.5, 0.5)},
  {"step": 1500 * 24, "lin_vel_x": (-0.3, 0.5), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": ANG_VEL_Z},
  {"step": 3000 * 24, "lin_vel_x": LIN_VEL_X, "lin_vel_y": LIN_VEL_Y, "ang_vel_z": ANG_VEL_Z},
]

# The rear feet step on this clock (s) while walking; at zero cmd_vel it stands, stepping
# only to keep its balance (v1's first try marched on the clock at zero cmd_vel too).
GAIT_PERIOD = 0.4
# Upright target: the nose this far (rad) toward the belly from straight up, so a fall
# goes forward onto the front feet.
LEAN = 0.08

# Estimated bus current (A) that starts costing: the 4-leg walk the robot runs all day
# peaks at ~27 A in sim (eval_power); the side flip, safe on the robot, held ~41 A.
CURRENT_BUDGET = 30.0
# All motors' load (sum of (torque / limit)^2) that starts costing: above the walk, so it
# only charges the rise, which speeds up in every stage without it.
MOTOR_LOAD_START = 1.5
# On only from this iteration, at these weights: from the start motor load stops the policy
# learning to balance, the others to rise (it stays on four legs, where they don't apply).
# Front feet down while up costs more from then too: v4's first S1 stopped rising when they
# came on (staying on four legs, still paid for tracking, beat rising with the paws high).
LATE_COSTS_FROM = 4000
LATE_COSTS = {"motor_load": -20.0, "arm_raise": -20.0, "stay_put": -10.0, "front_feet_down": -3.0}
# Motors above this share of the torque they can give at their speed count as saturated.
DC_SATURATION_FRACTION = 0.9
# Front feet force (N, per foot) that starts costing, in force_scale units: coming down,
# the paws are put down, not dropped onto (eval_upright's check is on the p95 peak).
FRONT_IMPACT = {"force_threshold": 200.0, "force_scale": 100.0}

# Per stage: shares of upright and partly risen starts (the rest on four legs), the partly
# risen nose range (deg), the share told to come down from the start, the episode (s), the
# up/down and hand schedules, the fall check (nose angle range in deg: 0 = level on four
# legs, 90 = upright; max roll in deg) and the changes from a step on (down_schedule). S1
# rises, stands, walks and comes down; S2 adds the handshake and the wave, its long holds
# balanced to keep walking ~40% of the time.
# S1 comes down once the rise is found, while the rising habit is young and the exploration
# wide, from low tilts up (down starts, partly risen up to nearly upright). Learnt in a stage
# of its own after S1, v3 held a lean or a level body on the rear legs in every try.
DOWN_SCHEDULE = [
  {"step": 1500 * 24, "down_start_prob": 0.5, "down_prob": 0.6, "up_again_prob": 0.4},
  {"step": 3000 * 24, "mid_nose_deg": (30.0, 88.0)},
]
_STARTS = {"upright_fraction": 0.4, "mid_fraction": 0.3}
_FELL = (-30.0, 120.0, 45.0)
STAGES = {
  1: {**_STARTS, "mid_nose_deg": (30.0, 70.0), "down_start_prob": 0.0, "episode_s": 20.0, "down_prob": 0.0,
      "down_time": (3.0, 10.0), "up_again_prob": 0.0, "hand_prob": 0.0, "wave_prob": 0.0, "fell": _FELL,
      "schedule": DOWN_SCHEDULE},
  # A smaller down share keeps the come-down from fading.
  2: {**_STARTS, "mid_nose_deg": (30.0, 88.0), "down_start_prob": 0.2, "episode_s": 30.0, "down_prob": 0.4,
      "down_time": (5.0, 25.0), "up_again_prob": 0.7, "hand_prob": 0.35, "wave_prob": 0.5, "fell": _FELL,
      "schedule": []},
}
# Upright from here (deg): both rear feet in the air costs (the rise may leave the ground).
RISEN_NOSE_DEG = 70.0
# Paw height from the base (m, negative = below it) that starts costing while up.
ARM_RAISE_LIMIT = -0.07
# Front feet height (m) from which front_feet_progress pays their way down.
FRONT_FEET_TOP = 0.5
# Standing still: after this long at zero cmd_vel (s), past the steps to stop, drift costs.
STAY_PUT_SETTLE_S = 1.5

# Penalties ramped in over the first iterations (from scratch); full strength after.
PENALTY_RAMP = {"start_scale": 0.2, "ramp_steps": 1000 * 24}


def to_upright_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  stage: int,
  rear_feet_sensor: str,
  front_feet_sensor: str,
  feet_sensor: str,
  rear_sites: tuple[str, ...],
  foot_sites: tuple[str, ...],
  rear_rays: tuple[str, ...],
  leg_sensor: str,
  arm_sensor: str,
  stand_stances: tuple[dict[str, float], ...],
  posture_std: dict[str, float],
  rear_actuators: str,
  upright_height: float,
  four_leg_height: float,
  reset: dict,
) -> ManagerBasedRlEnvCfg:
  """Turn a robot's Rough-Blind configuration (robot entity already upright, with DC
  motors; the rear/front feet contact sensors added) into UprightWalk `stage`, in place.

  rear_feet_sensor / front_feet_sensor: contact sensors on those feet (the rear one with
  air time); feet_sensor: Rough-Blind's four-foot sensor; rear_sites / foot_sites: foot
  sites; rear_rays: Rough-Blind's downward foot rays for the rear feet; leg_sensor: its
  thigh/calf sensor; arm_sensor: front legs against the rest of the robot; stand_stances: the 4-leg pose(s) (4-leg starts and the down pose);
  posture_std: per-joint std around the upright pose; rear_actuators: regex for the rear
  motors (heat); upright_height / four_leg_height: base height targets (m); reset: the
  robot's leg kinematics for the starts (leg_lengths, foot_radius, hip_x).
  """
  if stage not in STAGES:
    raise ValueError(f"UprightWalk stage {stage}: 1 or 2")
  st = STAGES[stage]
  cfg.episode_length_s = st["episode_s"]
  cmd = "twist"
  play = "penalty_ramp" not in cfg.curriculum and "command_vel" not in cfg.curriculum

  ##
  # Ground: flat and gentle rough, no curriculum.
  ##

  tg = cfg.scene.terrain.terrain_generator if cfg.scene.terrain is not None else None
  if tg is not None:
    tg.curriculum = False
    tg.sub_terrains = {
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.5),
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.5, noise_range=(0.0, 0.03), noise_step=0.01, border_width=0.25
      ),
    }
  cfg.curriculum.pop("terrain_levels", None)

  ##
  # Command, observations, actions
  ##

  cfg.commands[cmd] = mdp.UprightCommandCfg(
    entity_name="robot",
    resampling_time_range=cfg.commands["twist"].resampling_time_range,
    rel_standing_envs=0.3,  # standing is the default (joystick idle, handshake)
    heading_command=False,
    debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=LIN_VEL_X, lin_vel_y=LIN_VEL_Y, ang_vel_z=ANG_VEL_Z),
    down_prob=st["down_prob"],
    down_time_range=st["down_time"],
    down_start_prob=0.0 if play else st["down_start_prob"],  # a training aid
    up_again_prob=st["up_again_prob"],
    hand_prob=st["hand_prob"],
    wave_prob=st["wave_prob"],
  )
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = COMMAND_STAGES

  # Same layout in every stage; the up command and hand flag right after the gait clock.
  inputs = {
    "phase": ObservationTermCfg(func=mdp.phase, params={"period": GAIT_PERIOD, "command_name": cmd}),
    "upright_cmd": ObservationTermCfg(func=mdp.upright_cmd, params={"command_name": cmd}),
    "hand": ObservationTermCfg(func=mdp.hand_flag, params={"command_name": cmd}),
  }
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    new = {}
    for name, term in terms.items():
      if name == "phase":
        new.update(inputs)
      elif name == "projected_gravity" and group == "actor":
        new[name] = replace(term, func=mdp.tilted_gravity, params={"command_name": cmd})
      else:
        new[name] = term
    if group == "critic":
      new["paw_force"] = ObservationTermCfg(func=mdp.paw_force)
      new["drift"] = ObservationTermCfg(func=mdp.drift, params={"command_name": cmd})
    cfg.observations[group].terms = new
  # The frames stay short, the last second comes in coarse bins, once (a term of its own).
  # Copies: the critic shares these terms and keeps its own history.
  actor = cfg.observations["actor"]
  actor.terms = {k: replace(t, history_length=actor.history_length) for k, t in actor.terms.items()}
  actor.history_length = None
  actor.terms["slow_history"] = ObservationTermCfg(func=mdp.slow_history, params={"command_name": cmd})

  # Kicks' scale: the rise swings the rear thighs by ~1.4 rad.
  cfg.actions["joint_pos"] = mdp.HandJointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=0.5, use_default_offset=True, command_name=cmd
  )

  ##
  # Starts and pushes
  ##

  cfg.events.pop("reset_robot_joints", None)
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_upright_robot,
    mode="reset",
    params={
      "upright_fraction": st["upright_fraction"],
      "mid_fraction": st["mid_fraction"],
      "mid_nose_deg": st["mid_nose_deg"],
      "stand_stances": stand_stances,
      "joint_noise": 0.1,
      "xy_range": 0.5,
      "footprint_radius": 0.3,
      "drop_range": (0.0, 0.02),
      "lin_vel": 0.1,
      "ang_vel": 0.2,
      **reset,
    },
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.3, 0.3), "y": (-0.3, 0.3), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
    }

  ##
  # Rewards
  ##

  rear_feet = {"sensor_name": rear_feet_sensor}
  front_sites = tuple(s for s in foot_sites if s not in rear_sites)
  rewards = {
    # cmd_vel in the heading frame.
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_heading_velocity, weight=1.0, params={"command_name": cmd, "std": 0.5, "axes": (0, 1)}
    ),
    "track_forward": RewardTermCfg(
      func=mdp.track_heading_velocity, weight=0.5, params={"command_name": cmd, "std": 0.25, "axes": (0,)}
    ),
    "track_lateral": RewardTermCfg(
      func=mdp.track_heading_velocity, weight=0.5, params={"command_name": cmd, "std": 0.25, "axes": (1,)}
    ),
    "track_yaw": RewardTermCfg(
      func=mdp.track_heading_velocity, weight=1.0, params={"command_name": cmd, "std": 0.35, "axes": (2,)}
    ),
    "yaw_rate_error": RewardTermCfg(func=mdp.yaw_rate_error_l1, weight=-0.5, params={"command_name": cmd}),
    # Posture: upright while up, level on four legs while down.
    "upright": RewardTermCfg(func=mdp.upright, weight=2.0, params={"command_name": cmd, "std": 0.3, "lean": LEAN}),
    "upright_wide": RewardTermCfg(func=mdp.upright_wide, weight=1.0, params={"command_name": cmd, "lean": LEAN}),
    "level": RewardTermCfg(func=mdp.level, weight=2.0, params={"command_name": cmd, "std": 0.3}),
    # Pays the whole way down from upright (level pays only near level: v1's first S2 rose
    # but never came down). Kept small: at 5x every v3 cluster S2 held a lean that it paid.
    "level_wide": RewardTermCfg(func=mdp.level_wide, weight=1.0, params={"command_name": cmd}),
    # Pays each step toward level, nothing for holding: a full come-down from upright earns
    # about two seconds of the floor's lead over a held lean.
    "level_progress": RewardTermCfg(func=mdp.level_progress, weight=20.0, params={"command_name": cmd}),
    # The same for the front feet, as much for the whole way down: level on the rear legs alone
    # earns nothing from it.
    "front_feet_progress": RewardTermCfg(
      func=mdp.front_feet_progress,
      weight=10.0,
      params={"command_name": cmd, "site_names": front_sites, "max_height": FRONT_FEET_TOP},
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=1.0,
      params={
        "command_name": cmd, "site_names": foot_sites, "upright_height": upright_height,
        "four_leg_height": four_leg_height, "std": 0.08, "foot_radius": reset["foot_radius"],
      },
    ),
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "command_name": cmd,
        "std_standing": posture_std, "std_walking": posture_std, "std_running": posture_std,
      },
    ),
    "stand_pose": RewardTermCfg(
      func=mdp.stand_pose,
      weight=1.0,
      params={"command_name": cmd, "pose": stand_stances[0], "pose_var": 0.1, "still_var": 5.0},
    ),
    # Feet: the rear ones step on the clock while up and walking, and stay planted while up
    # and standing (each step costs: only to keep its balance); the front ones stay off the
    # ground while up and all four are down while down.
    "gait": RewardTermCfg(
      func=mdp.gait,
      weight=1.0,
      params={
        "command_name": cmd, "period": GAIT_PERIOD, "offset": [0.0, 0.5], "threshold": 0.55,
        "min_nose_deg": RISEN_NOSE_DEG, **rear_feet,
      },
    ),
    "flight": RewardTermCfg(
      func=mdp.flight, weight=-2.0, params={"command_name": cmd, "min_nose_deg": RISEN_NOSE_DEG, **rear_feet}
    ),
    "planted": RewardTermCfg(func=mdp.planted, weight=1.0, params={"command_name": cmd, **rear_feet}),
    "standing_steps": RewardTermCfg(func=mdp.standing_steps, weight=-5.0, params={"command_name": cmd, **rear_feet}),
    "standing_joint_speed": RewardTermCfg(func=mdp.standing_joint_speed, weight=-0.02, params={"command_name": cmd}),
    "standing_action_rate": RewardTermCfg(func=mdp.standing_action_rate, weight=-0.5, params={"command_name": cmd}),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance_terrain,
      weight=-1.0,
      params={
        "target_height": 0.06, "sensor_names": rear_rays, "contact_sensor_name": feet_sensor,
        "command_name": cmd, "command_threshold": 0.1, "speed_saturation": 0.2,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        **rear_feet, "command_name": cmd, "command_threshold": -1.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=rear_sites),
      },
    ),
    "soft_landing": RewardTermCfg(func=mdp.soft_landing, weight=-1e-3, params=dict(rear_feet)),
    "front_feet_down": RewardTermCfg(
      func=mdp.feet_down,
      weight=-1.0,
      params={"command_name": cmd, "sensor_name": front_feet_sensor, "geom_pattern": r".", "when": "up"},
    ),
    "four_feet_down": RewardTermCfg(
      func=mdp.feet_down,
      weight=0.25,
      params={"command_name": cmd, "sensor_name": feet_sensor, "geom_pattern": r".", "when": "down"},
    ),
    "front_impact": RewardTermCfg(
      func=mdp.part_contact_force,
      weight=-10.0,
      params={"sensor_name": front_feet_sensor, "geom_pattern": r".", **FRONT_IMPACT},
    ),
    # Arms against the body or rear legs (a real self-collision): each substep over 1 N.
    "arm_contact": RewardTermCfg(
      func=mdp.self_collision_cost, weight=-5.0, params={"sensor_name": arm_sensor, "force_threshold": 1.0}
    ),
    # Arms down unless a paw is out: without it v2's S3 raised both at zero cmd_vel.
    "arm_raise": RewardTermCfg(
      func=mdp.arm_raise,
      weight=0.0,
      params={
        "command_name": cmd, "limit": ARM_RAISE_LIMIT, "min_nose_deg": RISEN_NOSE_DEG,
        "asset_cfg": SceneEntityCfg("robot", site_names=front_sites, preserve_order=True),
      },
    ),
    # Costly enough that kneeling never pays: v2's S3 knelt through the handshake at -0.5.
    "leg_contact": RewardTermCfg(
      func=mdp.self_collision_cost, weight=-2.0, params={"sensor_name": leg_sensor, "force_threshold": 10.0}
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty, weight=-0.02, params={"sensor_name": "robot/root_angmom"}
    ),
    # A paw out: stay on the spot and keep the heading.
    "hand_hold": RewardTermCfg(func=mdp.hand_hold, weight=-3.0, params={"command_name": cmd, "yaw_weight": 1.0}),
    # Zero cmd_vel means stay: the tracking rewards are flat at creep speed.
    "stay_put": RewardTermCfg(
      func=mdp.stay_put,
      weight=0.0,
      params={"command_name": cmd, "settle_s": STAY_PUT_SETTLE_S, "yaw_weight": 0.5},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    # Power: never waived.
    "dc_torque_saturation": RewardTermCfg(
      func=mdp.dc_torque_saturation, weight=-10.0, params={"fraction": DC_SATURATION_FRACTION}
    ),
    "requested_torque": RewardTermCfg(func=mdp.requested_torque_excess, weight=-2.0),
    "motor_load": RewardTermCfg(func=mdp.motor_load_excess, weight=0.0, params={"start": MOTOR_LOAD_START}),
    "bus_current": RewardTermCfg(func=mdp.bus_current_excess, weight=-2.0, params={"budget": CURRENT_BUDGET}),
    "rear_heat": RewardTermCfg(
      func=mdp.motor_heat, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(rear_actuators,))}
    ),
  }
  cfg.rewards = rewards

  if not play:
    cfg.curriculum["penalty_ramp"] = CurriculumTermCfg(
      func=mdp.penalty_ramp,
      params={
        **PENALTY_RAMP,
        "term_names": [
          "standing_steps", "standing_joint_speed", "standing_action_rate", "flight",
          "foot_slip", "soft_landing", "arm_contact", "leg_contact", "body_ang_vel", "angular_momentum", "joint_acc_l2",
          "action_rate_l2", "dc_torque_saturation", "requested_torque", "bus_current", "rear_heat",
        ],
      },
    )
    for name, weight in LATE_COSTS.items():
      cfg.curriculum[name] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": name, "weight_stages": [{"step": LATE_COSTS_FROM * 24, "weight": weight}]},
      )
    if st["schedule"]:
      cfg.curriculum["down_schedule"] = CurriculumTermCfg(
        func=mdp.down_schedule,
        params={"command_name": cmd, "event_name": "reset_base", "stages": st["schedule"]},
      )

  ##
  # Terminations and metrics
  ##

  min_nose, max_nose, max_roll = st["fell"]
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["fell"] = TerminationTermCfg(
    func=mdp.fell, params={"min_nose_deg": min_nose, "max_nose_deg": max_nose, "max_roll_deg": max_roll}
  )
  for name in ("vel_err_xy", "cmd_speed_xy"):
    cfg.metrics.pop(name, None)
  cfg.metrics["est_current"] = MetricsTermCfg(func=mdp.estimated_current)
  cfg.metrics["arm_contact"] = MetricsTermCfg(
    func=mdp.contact_fraction, params={"sensor_name": arm_sensor, "force_threshold": 1.0}
  )
  cfg.metrics["walking"] = MetricsTermCfg(func=mdp.walking_metric, params={"command_name": cmd})
  cfg.metrics["one_foot"] = MetricsTermCfg(
    func=mdp.one_foot_metric, params={"command_name": cmd, "min_nose_deg": RISEN_NOSE_DEG, **rear_feet}
  )
  cfg.metrics["flight"] = MetricsTermCfg(
    func=mdp.flight, params={"command_name": cmd, "min_nose_deg": RISEN_NOSE_DEG, **rear_feet}
  )
  cfg.metrics["upright_cmd"] = MetricsTermCfg(func=mdp.upright_cmd_metric, params={"command_name": cmd})
  cfg.metrics["hand_out"] = MetricsTermCfg(func=mdp.hand_out_metric, params={"command_name": cmd})
  return cfg
