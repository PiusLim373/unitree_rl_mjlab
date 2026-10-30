"""DonkeyKick task configuration: handstand on the front legs, one rear-leg kick.

The kick is triggered at t = 0 with the robot standing (the deploy starts it from
JointLock or from Velocity standing still). It tips up to ~45 deg nose-down onto
its front legs, holds, tucks the rear legs and snaps them straight back once,
holds, comes down and stands still in the default pose. A timer input
(kick_phase) tells the policy where it is in that sequence. Episodes end on body
or head contact, or tipping too far.

This module provides a factory function; robot-specific configurations set
entity, sensor, site and body names.
"""

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
from mjlab.tasks.velocity.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.kick.mdp as mdp
from src.tasks.velocity.mdp import foot_contact

# Phase times (s) after the trigger, shared by the rewards, the deploy
# (State_DonkeyKick in config.yaml) and scripts/eval_kick.py.
KICK_TIMING = {
  "rise_end": 0.4,     # pitch 0 -> HANDSTAND_PITCH, rear feet leave the ground
  "kick_start": 0.9,   # hold, then tuck and snap the rear legs straight back
  "kick_end": 1.3,
  "down_start": 1.6,   # hold, then pitch back to 0 and land
  "stand_start": 2.2,  # stand still in the default pose
}
HANDSTAND_PITCH = 0.79  # rad nose-down (45 deg)
# kick_phase reaches 1 when standing starts; the deploy reads the same duration.
PHASE_DURATION = KICK_TIMING["stand_start"]

# Deploy done check (hand back to JointLock/Velocity once held 0.3 s after
# stand_start) and fall limits (episode ends / deploy goes to damping).
STAND_CHECK = {"max_tilt": 0.35, "max_pose_err": 0.3, "max_joint_vel": 0.5}
FALL_LIMITS = {"pitch_range": (-0.6, 1.31), "max_roll": 0.7}  # -34..75 deg, 40 deg

_GATE = {"tilt_full": STAND_CHECK["max_tilt"], "tilt_zero": 0.7, "pose_var": 0.05}

# Mostly flat (the kick should work anywhere, but mostly runs on flat ground),
# random difficulty per tile, milder than Rough-Blind's terrain.
KICK_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  curriculum=False,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.5),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.2,
      noise_range=(0.01, 0.04),
      noise_step=0.01,
      border_width=0.25,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.075,
      slope_range=(0.0, 0.27),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.075,
      slope_range=(0.0, 0.27),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.075,
      step_height_range=(0.0, 0.10),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.075,
      step_height_range=(0.0, 0.10),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
  },
)

# Penalties ramped in by penalty_ramp (final weights below), so the handstand is
# found before it has to be gentle. The termination penalty too: at full strength
# every early failed attempt (random actions end ~20% of episodes within 1 s)
# costs more than standing still all episode earns (~10), which teaches it never
# to try. The task terms and the balance terms are full strength from the start.
PENALTY_TERMS = [
  "termination",
  "rear_landing",
  "front_slip",
  "torque_excess",
  "joint_vel_excess",
  "action_rate",
  "joint_acc",
  "joint_pos_limits",
]


def make_kick_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base DonkeyKick task configuration."""

  ##
  # Sensors
  ##

  # Privileged terrain shape for the critic (yaw-aligned, as in Recovery).
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

  # Recovery's deploy terms plus the timer.
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
    "kick_phase": ObservationTermCfg(func=mdp.kick_phase, params={"duration": PHASE_DURATION}),
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
    "foot_contact": ObservationTermCfg(
      func=foot_contact,
      params={"sensor_name": "feet_ground"},
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

  # Recovery's scale: the rise and the kick swing thighs and calves by ~1 rad.
  # Centred on the default (stand) pose.
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
    # Standing in JointLock's or Velocity's stance, anywhere on the tile.
    "reset_base": EventTermCfg(
      func=mdp.reset_standing_robot,
      mode="reset",
      params={
        "stances": (),  # Set per-robot.
        "joint_noise": 0.05,
        "xy_range": 3.0,
        "footprint_radius": 0.3,
        "leg_lengths": (0.0, 0.0),  # Set per-robot.
        "foot_radius": 0.0,  # Set per-robot.
        "drop_range": (0.0, 0.02),
        "lin_vel": 0.1,
        "ang_vel": 0.2,
      },
    ),
    "foot_friction": EventTermCfg(
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
    # Balance is sensitive to mass and motor strength: +-1 kg on the base (a point
    # mass at its COM) and +-10% PD gains.
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": (-1.0, 1.0),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        # Default selection = every actuator group (pd_gains indexes groups).
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    ),
  }

  ##
  # Rewards
  ##

  timing = KICK_TIMING
  rewards = {
    # The handstand: follow the pitch profile, rear feet up, front feet planted.
    "pitch_track": RewardTermCfg(
      func=mdp.pitch_track,
      weight=3.0,
      params={
        "rise_end": timing["rise_end"],
        "down_start": timing["down_start"],
        "stand_start": timing["stand_start"],
        "pitch": HANDSTAND_PITCH,
        "std": 0.17,  # ~10 deg
        "coarse_range": 1.0,  # rad: standing level during the hold still earns 0.1
      },
    ),
    "rear_feet_air": RewardTermCfg(
      func=mdp.rear_feet_air,
      weight=1.0,
      params={
        "sensor_name": "feet_ground",
        "geom_pattern": "",  # Set per-robot.
        "start": timing["rise_end"],
        "end": timing["down_start"],
      },
    ),
    "front_feet_planted": RewardTermCfg(
      func=mdp.front_feet_planted,
      weight=0.5,
      params={"sensor_name": "feet_ground", "geom_pattern": ""},  # Set per-robot.
    ),
    # The kick: episode total = 10 x (best - start) both-feet extension score.
    "kick_snap": RewardTermCfg(
      func=mdp.kick_snap,
      weight=10.0,
      params={
        "feet": (),  # Set per-robot.
        "hips": (),  # Set per-robot.
        "start": timing["kick_start"],
        "end": timing["kick_end"],
        "reach_min": 0.10,
        "reach_span": 0.20,
        # Only counts in the handstand: nose-down > 30 deg, rear feet off the ground.
        "min_pitch": 0.52,
        "sensor_name": "feet_ground",
        "geom_pattern": "",  # Set per-robot.
      },
    ),
    "roll_level": RewardTermCfg(func=mdp.roll_level, weight=-2.0),
    "pitch_excess": RewardTermCfg(func=mdp.pitch_excess, weight=-20.0, params={"max_pitch": 1.05}),
    # Then stand still in the default pose.
    "stand_pose": RewardTermCfg(
      func=mdp.stand_pose, weight=2.0, params={"start": timing["stand_start"], **_GATE}
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still, weight=1.0, params={"start": timing["stand_start"], **_GATE}
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
    # Gentle (ramped in): soft rear landing, no front-foot sliding, motors off
    # their limits.
    "rear_landing": RewardTermCfg(
      func=mdp.part_contact_force,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground",
        "geom_pattern": "",  # Set per-robot.
        "force_threshold": 300.0,  # ~2x the robot's weight
        "force_scale": 100.0,
      },
    ),
    "front_slip": RewardTermCfg(
      func=mdp.front_feet_slip,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground",
        "geom_pattern": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "torque_excess": RewardTermCfg(func=mdp.torque_excess, weight=-15.0, params={"fraction": 0.9}),
    "joint_vel_excess": RewardTermCfg(
      func=mdp.joint_vel_excess, weight=-0.1, params={"max_vel": 15.0}
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "body_contact": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": "body_ground", "force_threshold": 10.0},
    ),
    "fell": TerminationTermCfg(func=mdp.pitch_roll_out_of_range, params=dict(FALL_LIMITS)),
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
    # x episode length (3.5 s) = seconds in the handstand (target ~1.2 s).
    "handstand": MetricsTermCfg(
      func=mdp.handstand,
      params={"min_pitch": 0.61, "sensor_name": "feet_ground", "geom_pattern": ""},  # Set per-robot.
    ),
    "stand_done": MetricsTermCfg(
      func=mdp.stand_done, params={"start": timing["stand_start"], **STAND_CHECK}
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(KICK_TERRAINS_CFG),
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
    sim=SimulationCfg(
      nconmax=35,
      njmax=500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=3.5,
  )
