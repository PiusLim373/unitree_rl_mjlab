"""Recovery task configuration: get up from a fall into the lying pose.

Robots start fallen anywhere on flat, stair and slope tiles, above the local
ground, at a random orientation with random joints: half settled (small drop,
nearly still), half tumbling. They earn reward for ending belly-down, legs folded
in the default (lying) pose, and still. Penalties keep it gentle, including fast
joints, motors near their torque limit and body slams, and ramp in over training.
There is no command; episodes end only on time-out.

This module provides a factory function; robot-specific configurations set
entity, sensor and body names.
"""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import body_angular_velocity_penalty
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.recovery.mdp as mdp
from src.tasks.velocity.mdp import self_collision_cost

# Done check shared by the metric and the deploy (State_Recovery in config.yaml):
# tilt (rad), largest joint error from the lying pose (rad), largest joint speed (rad/s).
# v2: tilt 0.4 -> 0.5, since lying along a 15/30 cm staircase tilts the body ~0.47 rad.
DONE_CHECK = {"max_tilt": 0.5, "max_pose_err": 0.2, "max_joint_vel": 0.3}

# Upright gate for the pose/stillness rewards: full up to the done-check tilt (so
# lying on a slope or stairs counts), zero beyond 0.9 rad.
_GATE = {"tilt_full": DONE_CHECK["max_tilt"], "tilt_zero": 0.9, "pose_var": 0.05}

# Random difficulty per tile (no terrain curriculum). Robots spawn anywhere on
# the tile (reset_fallen_robot), so most stair-tile falls start on the steps.
# v2: stairs 30% -> 50% (stairs were the weakest terrain in v1).
RECOVERY_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  curriculum=False,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.3),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.25,
      step_height_range=(0.0, 0.15),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.25,
      step_height_range=(0.0, 0.15),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
  },
)

# Penalties ramped in by the penalty_ramp curriculum (final weights below).
PENALTY_TERMS = [
  "joint_torques",
  "joint_vel",
  "joint_vel_excess",
  "torque_excess",
  "joint_acc",
  "action_rate",
  "joint_pos_limits",
  "body_ang_vel",
  "body_impact",
  "head_contact",
  "roll_contact",
  "leg_slap",
  "self_collision",
]


def make_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base recovery task configuration."""

  ##
  # Sensors
  ##

  # Privileged terrain shape for the critic. Rays point straight down in the
  # world (yaw alignment), so it also works with the robot on its back.
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
  )

  ##
  # Observations
  ##

  # Only terms the deploy controller already provides (no command, no phase).
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  # Twice the locomotion scale: rolling over needs big joint swings. Centred on
  # the default (lying) pose.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    # Anywhere on the tile, above the local ground, at a random orientation: half
    # "settled" (small drop, nearly still), half tumbling drops.
    "reset_base": EventTermCfg(
      func=mdp.reset_fallen_robot,
      mode="reset",
      params={
        "xy_range": 3.0,
        "footprint_radius": 0.45,
        "clearance": 0.4,
        "settled_fraction": 0.5,
        "settled_drop": (0.0, 0.1),
        "tumbling_drop": (0.1, 0.35),
        "settled_velocity": 0.1,
        "tumbling_velocity": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.0),
          "roll": (-2.0, 2.0),
          "pitch": (-2.0, 2.0),
          "yaw": (-2.0, 2.0),
        },
      },
    ),
    # Tangled, odd joint configurations around the lying pose (clamped to limits).
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-1.5, 1.5),
        "velocity_range": (-1.0, 1.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "robot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "upright": RewardTermCfg(func=mdp.upright, weight=1.0),
    "lying_pose": RewardTermCfg(func=mdp.lying_pose, weight=1.0, params=dict(_GATE)),
    "lying_still": RewardTermCfg(func=mdp.lying_still, weight=1.0, params=dict(_GATE)),
    # Gentle: penalties heavier than locomotion's, ramped in by penalty_ramp.
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-4),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
    # v2: aim at the violent tail (fast joints, motors at their limit) rather than
    # all motion. Weights calibrated on v1: a typical fall pays ~0, the most violent
    # 20% pay ~40% of their task reward in these three terms (with body_impact).
    "joint_vel_excess": RewardTermCfg(
      func=mdp.joint_vel_excess, weight=-0.1, params={"max_vel": 10.0}
    ),
    "torque_excess": RewardTermCfg(
      func=mdp.torque_excess, weight=-15.0, params={"fraction": 0.8}
    ),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "body_ang_vel": RewardTermCfg(
      func=body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    # Body/leg slams above ~1.3x the robot's weight (v2: 300 -> 200 N, -0.1 -> -0.3).
    "body_impact": RewardTermCfg(
      func=mdp.body_impact,
      weight=-0.3,
      params={"sensor_name": "body_ground", "force_threshold": 200.0, "force_scale": 100.0},
    ),
    # v3: the roll itself made v2's hard contacts (hips 49%, thighs 27%, head 13% of
    # forces > 600 N), not the belly landing. Geom patterns set per-robot.
    # Calibrated on v2: a fall with a head slam (> 600 N) pays ~30% of its task
    # reward in head_contact, light head touches ~2.5% (200 N threshold separates
    # them); the 20% slam-heaviest falls pay ~33% in roll_contact + leg_slap; a
    # typical fall pays ~0 in all three.
    "head_contact": RewardTermCfg(
      func=mdp.part_contact_force,
      weight=-25.0,
      params={"sensor_name": "body_ground", "geom_pattern": "", "force_threshold": 200.0, "force_scale": 100.0},
    ),
    "roll_contact": RewardTermCfg(
      func=mdp.part_contact_force,
      weight=-0.6,
      params={"sensor_name": "body_ground", "geom_pattern": "", "force_threshold": 150.0, "force_scale": 100.0},
    ),
    "leg_slap": RewardTermCfg(
      func=mdp.leg_slap,
      weight=-5.0,
      params={"sensor_name": "body_ground", "geom_pattern": "", "force_threshold": 5.0},
    ),
    "self_collision": RewardTermCfg(
      func=self_collision_cost,
      weight=-0.5,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  # No fall terminations: the robot starts fallen and needs its body on the
  # ground. No success ending either: it should learn to hold the pose.
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "penalty_ramp": CurriculumTermCfg(
      func=mdp.penalty_ramp,
      params={
        "term_names": PENALTY_TERMS,
        "start_scale": 0.1,
        "ramp_steps": 3000 * 24,  # iterations x steps per iteration
      },
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "lying_still": MetricsTermCfg(func=mdp.lying_still_check, params=dict(DONE_CHECK)),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(RECOVERY_TERRAINS_CFG),
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.0,
      elevation=-15.0,
      azimuth=90.0,
    ),
    # Measured with random actions: <= 19 contacts and <= 77 constraint rows per
    # robot (average ~4 contacts). Contacts are pooled across robots; constraint
    # rows are per robot. Larger budgets run out of GPU memory at 1024 robots.
    sim=SimulationCfg(
      nconmax=30,
      njmax=500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=8.0,
  )
