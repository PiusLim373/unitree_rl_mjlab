"""QuickBuck task configuration: a fast, hard kick of both rear legs, no handstand.

The kick is triggered at t = 0 with the robot standing (the deploy starts it from
JointLock or from Velocity standing still, like the DonkeyKick). It shifts its weight
onto the front legs, strikes back with both rear feet about 10 cm above the base
link, lands and stands still. The timer input (kick_phase) runs over BUCK_TIMING.

Built from a finished DonkeyKick configuration (to_buck_env_cfg): the start poses,
terrain, randomisation, observations, actions, contact termination, stand-still
rewards, penalties and penalty ramp are DonkeyKick's. Swapped: the timing, the kick
rewards (strike instead of the handstand), the fall limits, and 80% motor limits.
"""

import re

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.kick.mdp as mdp

# Phase times (s) after the trigger, shared by the rewards, the deploy (QuickBuck
# in config.yaml) and scripts/eval_kick.py.
BUCK_TIMING = {
  "kick_start": 0.15,  # load (weight forward); the strike counts from here...
  "kick_end": 0.7,  # ...to here, whenever the policy chooses
  "land_start": 0.8,  # rear feet back on the ground
  "stand_start": 1.2,  # stand still in the default pose
}
# kick_phase reaches 1 when standing starts; the deploy reads the same duration.
BUCK_PHASE_DURATION = BUCK_TIMING["stand_start"]
BUCK_EPISODE_S = 2.0

BUCK_FALL_LIMITS = {"pitch_range": (-0.6, 1.05), "max_roll": 0.7}  # -34..60 deg, 40 deg

# The strike zone (feet height above the base link, reach behind the hips) and the
# backward foot speed that scores full power. In the zone the leg points back about
# level (at 30 deg nose-down: thigh 1.35-2.4 rad, calf near its -0.84 rad stop), so a
# swing about the hip moves the foot up, not back; the speed comes mostly from the
# knee straightening. Best backward foot speed in the zone at 80% joint speeds, body
# still: ~3.7 m/s (planar leg model), so 3 m/s scores full power.
STRIKE = {
  "height_range": (0.05, 0.15),  # m: ~10 cm above the base link
  "edge": 0.08,  # m: soft zone edges (pays near the zone, a way in)
  "min_reach": 0.20,  # m behind the hip
  "v_full": 3.0,  # m/s
}
# zone_approach: closeness 0 at d0 from the zone (standing: ~0.39 m), 1 inside it.
ZONE_APPROACH_D0 = 0.40

# "80% power": torque and joint-speed penalties above this share of the motor limits.
# The joint-speed one holds it (the kick's power is its speed): at -1.0, thighs at
# 28 instead of 24 rad/s through a kick cost ~5, more than the strike gains from it;
# at DonkeyKick's -0.1 it cost ~0.5. Torque stays soft (push-off and landing load
# the motors at low speed; the DC curve already caps torque at speed).
POWER_FRACTION = 0.8
JOINT_VEL_EXCESS_WEIGHT = -1.0


def _group_name(actuator: DcMotorActuatorCfg) -> str:
  """Short name of an actuator group for its term name (".*hip_.*" -> "hip")."""
  return "_".join(re.sub(r"[^a-z]", "", e) for e in actuator.target_names_expr)


def to_buck_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Turn a robot's DonkeyKick configuration into QuickBuck (in place; returned).

  The rear feet, hips and contact names come from DonkeyKick's kick_snap term. The
  robot entity must use DC motor actuators (torque falls with speed): the joint-speed
  penalties are per actuator group, at POWER_FRACTION of its top speed.
  """
  timing = BUCK_TIMING
  kick = cfg.rewards["kick_snap"].params
  legs = {k: kick[k] for k in ("feet", "hips", "sensor_name", "geom_pattern")}
  contacts = {"sensor_name": kick["sensor_name"], "geom_pattern": kick["geom_pattern"]}
  window = {"start": timing["kick_start"], "end": timing["kick_end"]}

  cfg.episode_length_s = BUCK_EPISODE_S
  # One term config shared by the actor and critic groups.
  cfg.observations["actor"].terms["kick_phase"].params["duration"] = BUCK_PHASE_DURATION

  ##
  # Rewards
  ##

  rewards = cfg.rewards
  for name in ("pitch_track", "rear_feet_air", "kick_snap", "joint_vel_excess"):
    del rewards[name]
  # The kick: episode total = 10 x the best both-feet strike score (fast, in the zone).
  rewards["strike"] = RewardTermCfg(
    func=mdp.strike, weight=10.0, params={**legs, **window, **STRIKE}
  )
  zone = {k: STRIKE[k] for k in ("height_range", "min_reach")}
  rewards["zone_approach"] = RewardTermCfg(
    func=mdp.zone_approach, weight=3.0, params={**legs, **window, **zone, "d0": ZONE_APPROACH_D0}
  )
  rewards["rear_feet_down"] = RewardTermCfg(
    func=mdp.rear_feet_down, weight=0.5, params={**contacts, "start": timing["land_start"]}
  )
  # A buck, not a handstand.
  rewards["pitch_excess"].params["max_pitch"] = 0.61  # 35 deg nose-down
  for name in ("stand_pose", "stand_still"):
    rewards[name].params["start"] = timing["stand_start"]
  rewards["torque_excess"].params["fraction"] = POWER_FRACTION
  # A hard kick needs a big, sudden action: half DonkeyKick's smoothness penalties.
  rewards["action_rate"].weight = -0.05
  rewards["joint_acc"].weight = -1.25e-7

  vel_terms = []
  for actuator in cfg.scene.entities["robot"].articulation.actuators:
    assert isinstance(actuator, DcMotorActuatorCfg), "QuickBuck needs DC motor actuators"
    name = f"joint_vel_excess_{_group_name(actuator)}"
    rewards[name] = RewardTermCfg(
      func=mdp.joint_vel_excess,
      weight=JOINT_VEL_EXCESS_WEIGHT,
      params={
        "max_vel": POWER_FRACTION * actuator.velocity_limit,
        "asset_cfg": SceneEntityCfg("robot", joint_names=actuator.target_names_expr),
      },
    )
    vel_terms.append(name)

  if "penalty_ramp" in cfg.curriculum:  # not in play mode
    ramp = cfg.curriculum["penalty_ramp"].params
    ramp["term_names"] = [n for n in ramp["term_names"] if n != "joint_vel_excess"] + vel_terms

  ##
  # Terminations and metrics
  ##

  cfg.terminations["fell"].params = dict(BUCK_FALL_LIMITS)

  del cfg.metrics["handstand"]
  # x episode length (2 s) = seconds with both rear feet in the air.
  cfg.metrics["rear_up"] = MetricsTermCfg(func=mdp.rear_up, params=dict(contacts))
  cfg.metrics["stand_done"].params["start"] = timing["stand_start"]

  return cfg
