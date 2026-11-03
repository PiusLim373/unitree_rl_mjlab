"""SideFlip and FrontFlip task configurations: one full turn, landing on all four feet.

The flip is triggered at t = 0 with the robot standing (the deploy starts it from
JointLock or from Velocity standing still, like the kicks), turns once, lands on its
feet and stands still. The timer input (kick_phase) runs over the flip's timing.

SideFlip: one roll over its left side about the forward axis, in the air. The right flip
is the same policy mirrored (FlipRunner exports it as policy_right.onnx). v2 follows
Unitree's own side flip (from video): ~90 deg on the ground while one side's legs swing
up and the other side pushes, a flight with the legs spread, touchdown just short of a
full turn, standing soon after. v1 (run sideflip_v1) learned a double, low, fast flip.

FrontFlip: one turn nose-down about the left axis, over its back, landing on its feet.
Left-right symmetric, so it trains mirrored like the kicks (KickRunner). v1 (run
frontflip_v1) stood on its front feet and spun round instead: v2 counts the turn about the
fixed start axis and needs it upside down. v2 twisted round in the air instead (a
round-off): v3 holds the left axis on its start direction (the fall check) and pays more
for spin. v3 hopped onto its front feet and back (a crash cost more than not trying): v4
drops Unitree's timeline for loose deadlines and pays nothing just for staying in the
episode (see mdp/flip.py). v4 went over its back every time but crashed there, and backed
off as the motor saturation penalty ramped in: v5 frees the launch from that penalty and
starts some training episodes in the air partway through the flip (reset_mid_flip).
v6 to v13 built the flip by fine-tuning one run after another; the multistage recipe
(front_flip_stage, scripts/train_front_flip.py) trains it from scratch in stages instead.

Built from a finished DonkeyKick configuration (to_flip_env_cfg), like QuickBuck: the
start poses, terrain mix, randomisation, observations, actions, body-contact
termination, stand-still rewards, penalties and penalty ramp are DonkeyKick's.
Swapped: the timing, the flip rewards (spin reads the robot's whole-body angular
momentum sensor), the fall check (the flip's own turn is free while flipping) and 90%
motor limits.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import src.tasks.kick.mdp as mdp
from src.tasks.kick.buck_env_cfg import apply_power_fraction

##
# SideFlip
##

# Phase times (s) after the trigger, shared by the rewards, the fall check and
# scripts/eval_flip.py (and the deploy).
FLIP_TIMING = {
  "spin_end": 0.6,  # spin and air time pay until here (the flight is over by ~0.57 s)
  "land_by": 0.9,  # level again by now, or it fell
  "stand_start": 1.0,  # stand still in the default pose
}
# kick_phase reaches 1 when standing starts; the deploy reads the same duration.
FLIP_PHASE_DURATION = FLIP_TIMING["stand_start"]
FLIP_EPISODE_S = 2.0

# The rotation timeline (s, turns), from Unitree's flip: start rolling at 0.1 s, a
# quarter turn at liftoff, 0.92 at touchdown, the full turn once landed; held at 1.
ROLL_PROFILE = ((0.1, 0.0), (0.3, 0.25), (0.57, 0.92), (0.7, 1.0))
ROLL_TRACK = {"std": 0.05, "coarse_range": 1.0}  # turns (0.05 = 18 deg)

# -1: over the left side (negative about the forward axis, right-hand rule).
FLIP_DIRECTION = -1.0

# The forward axis may tip 60 deg at any time; after land_by the body must be within
# 40 deg of level. Roll is free until then.
FLIP_FALL = {"max_axis_tilt": 1.05, "max_tilt": 0.7, "after": FLIP_TIMING["land_by"]}

# A liftoff (rotation starts counting) needs the base rising faster than this (m/s):
# limbs briefly off the ground while tipping over don't count (the base falls).
LIFTOFF_UP_VEL = 0.3

# A landing counts between 300 and 414 deg of rotation, within 30 deg of level.
FLIP_LANDING = {"min_turns": 300.0 / 360.0, "max_turns": 1.15, "max_tilt": 0.52}

# Whole-body angular momentum (N m s) about the forward axis that scores full spin:
# roll inertia 0.17 kg m^2 standing, 0.13 legs tucked (go2.xml), so ~1.5 is one turn
# in a ~0.5 s flight. It stops paying past SPIN_MAX_TURNS.
SPIN_TARGET = 1.5
SPIN_MAX_TURNS = 0.85

# Base height gained above the start (m) while airborne that scores full flight height.
FLIGHT_HEIGHT = 0.20

##
# FrontFlip
##

# Loose deadlines, not Unitree's timeline (its flip is level ~1 s after the trigger).
FRONT_FLIP_TIMING = {
  "spin_end": 1.0,  # spin and air time pay until here
  "land_by": 1.4,  # level again by now, or it fell
  "stand_start": 1.5,  # stand still in the default pose
}
FRONT_FLIP_EPISODE_S = 2.5

# +1: nose down (positive about the left axis, right-hand rule).
FRONT_FLIP_DIRECTION = 1.0

# The left axis may tip or turn 40 deg from its start direction at any time (v1 rode a
# 60 deg roll limit while spinning on its front feet, v2 twisted round in the air); after
# land_by the body must be within 40 deg of level. Pitch is free until then.
FRONT_FLIP_FALL = {
  "max_axis_tilt": 0.7, "max_axis_angle": 0.7, "max_tilt": 0.7, "after": FRONT_FLIP_TIMING["land_by"]
}

# The landing counts once the robot comes within 30 deg of level on its feet from 331 deg
# (up to 414 deg): an earlier rear-feet touchdown doesn't count yet.
FRONT_FLIP_LANDING = {"min_turns": 0.92, "max_turns": 1.15, "max_tilt": 0.52}

# Pitch inertia ~0.47 kg m^2 (about 3x roll; tucking barely changes it) at the
# video's ~9 rad/s flight. Spin stops paying at three quarters, before the landing.
# v3: a higher target and weight (v2 launched too little spin to get over its back).
FRONT_FLIP_SPIN_TARGET = 5.0
FRONT_FLIP_SPIN_WEIGHT = 2.0
FRONT_FLIP_SPIN_MAX_TURNS = 0.75

# Progress past half a turn and the landing need the body this far from level (150 deg)
# on the way: over its back.
FRONT_FLIP_UPSIDE_DOWN = 2.62

# A body crash after a liftoff and this much rotation (past vertical) costs no termination
# penalty: an attempt that came up short.
FRONT_FLIP_COMMIT_TURNS = 0.25

# Mid-flip starts (training only, events.reset_mid_flip): 30% of episodes begin in the air
# partway through the flip. Multistage v2: on multistage v1's own flight (liftoff level at
# 0.22 s from 0.45 m, drifting back a little; the up speed and steady rate that put the
# base where v1's is between upside down and ~290 deg, v1 touching down at ~350 deg), from
# upside down to ~290 deg. Up to multistage v1: v5's flight (9.5-11.5 rad/s, touchdown
# ~255 deg), so they touched down before 270 deg and landing_pose never paid there. Leg
# poses (set per robot): v5's in the air with the rear legs tucked, or the stand pose
# (multistage v2; was rear legs swung back for v8's ~262 deg catch). The base is raised if
# anything starts within 8 cm of the ground (one step falls ~3 cm and turns ~16 deg).
FRONT_FLIP_MID_START = {
  "fraction": 0.3,
  "turns": (0.5, 0.8),
  "ang_vel": (13.0, 15.0),  # rad/s
  "up_vel": (1.25, 1.5),  # m/s at the liftoff
  "fwd_vel": (-0.45, -0.1),  # m/s
  "liftoff": {"t": 0.22, "turns": 0.0, "height": 0.45},
  "joint_noise": 0.25,
  "clearance": 0.08,  # m
}

# Once: the rear feet catch the ground from 216 deg on, before any body part (rear_catch).
# Worth the landing's 5, so a catch and then any crash beats v5's crash on its hips (v7: a
# crash after the catch costs the full termination penalty, ~5). Multistage: both rear feet,
# paid in full within 40 ms of each other, less up to 100 ms (v6 to v13 caught on one).
FRONT_FLIP_CATCH = {"weight": 5.0, "min_turns": 0.6, "pair_window": 0.1, "pair_full": 0.04}
# Multistage stage 1 only: finding the flip needs a reward soon after going over the back
# (a first try with the pair alone never found it, like v7). The first rear foot pays a
# share, the pair the rest, within a wider window.
FRONT_FLIP_CATCH_STAGE1 = {**FRONT_FLIP_CATCH, "pair_window": 0.3, "first_pay": 0.4}

# The turn made on the front feet before the liftoff, full at 0.3 turns (~110 deg,
# Unitree's liftoff); small, so it only nudges the launch (front_pivot).
FRONT_FLIP_PIVOT = {"weight": 2.0, "max_turns": 0.3}

# After the catch or the landing (v9): v8 knelt, thighs and calves down, legs splayed. A
# thigh or calf on the ground costs this per second (kneeling to the end ~half the
# landing's 5), and a second stand pose term, wide enough to pay from the kneel, pulls
# it up (stand_pose's 0.05 pays almost nothing there).
FRONT_FLIP_KNEE_DOWN_WEIGHT = -1.5
FRONT_FLIP_WIDE_STAND = {"weight": 2.0, "pose_var": 0.5}

# v10: v9 still caught in a sit, slamming both rear thighs down right after the rear feet
# touched (the knees took the impact, which the feet-only landing penalty let through). Any
# thigh or calf contact force costs, per 100 N: v9's slam ~4 per flip, about a crash.
FRONT_FLIP_KNEE_IMPACT = {"weight": -5.0, "force_threshold": 0.0, "force_scale": 100.0}

# v11: v10 swung its legs to one side after the liftoff and caught on one rear foot. Any
# left/right difference in the joints costs (v11: -1.5 everywhere). v13: only in the air
# (uneven ground needs uneven legs), and strong enough that v11/v12's one-leg reach just
# before the catch costs about half a crash.
FRONT_FLIP_MIRROR_POSE = {"weight": -5.0, "air_only": True}

# Multistage v2: v1 came down with its rear legs swung forward under the belly, knees a
# few cm off the floor, from stage 1 on (a fine-tune at stage 4 didn't change it). From
# stage 1, the legs near the stand pose (feet under the hips, knees up) from 270 deg until
# any leg part touches down pays this per second: ~2 a flip at most. All the leg joints
# and a tighter pose_var: on thighs and calves only, the rear legs splayed out at the hips
# (knees low again), and legs well off the pose still earned most of it.
FRONT_FLIP_LANDING_POSE = {"weight": 20.0, "min_turns": 0.75, "pose_var": 0.25, "joint_pattern": r".*_(hip|thigh|calf)_joint"}
# Multistage v2: the knee penalty moved v1's impact onto stiff front legs (feet cost a
# fifth as much per 100 N). Above LANDING_FORCE's threshold a foot now costs as much.
# Multistage v3: v2's rear feet landed softly, but the leftover spin brought the front
# feet down hard; twice the cost, from a lower force, so the rear legs brake the spin.
FRONT_FLIP_FEET_IMPACT = {"weight": -10.0, "force_threshold": 300.0}

# v12 only: each second from the first rear touchdown until both rear feet are down. It
# shortened the gap but kept the reach, and charged the uneven touchdowns rough ground and
# stairs force (v13 drops it).
FRONT_FLIP_REAR_STAGGER_WEIGHT = -30.0

##
# Shared
##

# "90% power" (user's choice): QuickBuck's speed-dependent limits and penalty weight,
# with more headroom: a flip is power-hungry.
POWER_FRACTION = 0.9
JOINT_VEL_EXCESS_WEIGHT = -1.0

# Motors above this share of the torque they can give at their speed (DC curve) count
# as saturated.
DC_SATURATION_FRACTION = 0.95

# A flip lands harder than a kick: per-foot force above this (N, ~2.7x the whole
# robot's weight on one foot) is penalised.
LANDING_FORCE = {"force_threshold": 400.0, "force_scale": 100.0}

# The reward manager scales every term by dt (0.02 s), so a one-step termination
# penalty of -20 cost only 0.4 in SideFlip v1: -250 costs ~5 at full strength.
TERMINATION_WEIGHT = -250.0

# Each flip's settings for to_flip_env_cfg and scripts/eval_flip.py. axis: the body axis
# it turns about (0 forward, 1 left); keys: its turn_track / turn_progress reward names
# (no turn_track without a profile); fixed_axis / upside_down / fall's max_axis_angle: see
# mdp/flip.py; spin_weight: 1 if not given. FrontFlip only: heading_penalty (heading_hold
# as a penalty), stand_after_landing, commit_turns (flip_termination), free_launch
# (dc_torque_saturation from the liftoff), mid_flip (reset_mid_flip's settings); v6: catch
# (rear_catch), pivot (front_pivot), front_landing (the landing counts front feet only, so
# rear feet already down from a catch don't use it up), stand_from_landing (standing pays
# from the landing, not stand_start), crash_after_landing (flip_termination charges it);
# v7: clean_landing (no landing bonus with the body down too), crash_after_catch
# (flip_termination charges a crash once the rear feet caught), stand_from_catch (standing
# pays from the catch as well as the landing); v9: knee_down (knees_down_after_landing's
# weight), wide_stand (stand_pose_wide's weight and pose_var); v10: knee_impact
# (part_contact_force on the thighs and calves: weight, threshold, scale); v11: mirror_pose
# (mirror_pose's weight, and air_only: v13); v12: rear_stagger (rear_stagger's weight);
# multistage: catch's pair_window / pair_full / first_pay (with crash_after_catch, a crash
# after either rear foot touched down costs); multistage v2: landing_pose (landing_pose's
# weight and settings), feet_impact (the feet's landing penalty weight; v3: and threshold,
# in place of LANDING_FORCE's).
SIDE_FLIP = {
  "axis": 0,
  "direction": FLIP_DIRECTION,
  "timing": FLIP_TIMING,
  "episode_s": FLIP_EPISODE_S,
  "profile": ROLL_PROFILE,
  "landing": FLIP_LANDING,
  "fall": FLIP_FALL,
  "spin_target": SPIN_TARGET,
  "spin_max_turns": SPIN_MAX_TURNS,
  "keys": ("roll_track", "roll_progress"),
}
FRONT_FLIP = {
  "axis": 1,
  "direction": FRONT_FLIP_DIRECTION,
  "timing": FRONT_FLIP_TIMING,
  "episode_s": FRONT_FLIP_EPISODE_S,
  "profile": None,
  "landing": FRONT_FLIP_LANDING,
  "fall": FRONT_FLIP_FALL,
  "spin_target": FRONT_FLIP_SPIN_TARGET,
  "spin_weight": FRONT_FLIP_SPIN_WEIGHT,
  "spin_max_turns": FRONT_FLIP_SPIN_MAX_TURNS,
  "keys": ("pitch_track", "pitch_progress"),
  "fixed_axis": True,
  "upside_down": FRONT_FLIP_UPSIDE_DOWN,
  "heading_penalty": True,
  "stand_after_landing": True,
  "commit_turns": FRONT_FLIP_COMMIT_TURNS,
  "free_launch": True,
  "mid_flip": FRONT_FLIP_MID_START,
  "catch": FRONT_FLIP_CATCH,
  "pivot": FRONT_FLIP_PIVOT,
  "front_landing": True,
  "stand_from_landing": True,
  "crash_after_landing": True,
  "clean_landing": True,
  "crash_after_catch": True,
  "stand_from_catch": True,
  "knee_down": FRONT_FLIP_KNEE_DOWN_WEIGHT,
  "wide_stand": FRONT_FLIP_WIDE_STAND,
  "knee_impact": FRONT_FLIP_KNEE_IMPACT,
  "mirror_pose": FRONT_FLIP_MIRROR_POSE,
  "landing_pose": FRONT_FLIP_LANDING_POSE,
  "feet_impact": FRONT_FLIP_FEET_IMPACT,
}

# Multistage (scripts/train_front_flip.py): FRONT_FLIP from scratch in four stages, each
# adding the rules the fine-tune runs added, in their order: 1 = v6's (finding the flip),
# 2 = v7's (a clean landing) and mirror_pose, 3 = v9's (up off the knees), 4 = v10's (no
# knee slam) with v3's feet impact, which is FRONT_FLIP (v2's landing pose from stage 1).
# The keys each stage adds:
FRONT_FLIP_STAGE_KEYS = {
  2: ("clean_landing", "crash_after_catch", "stand_from_catch", "mirror_pose"),
  3: ("knee_down", "wide_stand"),
  4: ("knee_impact", "feet_impact"),
}
# Settings a stage has in place of FRONT_FLIP's.
FRONT_FLIP_STAGE_OVERRIDES = {1: {"catch": FRONT_FLIP_CATCH_STAGE1}}


def front_flip_stage(stage: int) -> dict:
  """FRONT_FLIP without the keys later stages add, with the stage's overrides (stage 4 =
  FRONT_FLIP)."""
  if stage not in (1, 2, 3, 4):
    raise ValueError(f"FrontFlip stage {stage}: 1 to 4")
  later = {k for s, keys in FRONT_FLIP_STAGE_KEYS.items() if s > stage for k in keys}
  spec = {k: v for k, v in FRONT_FLIP.items() if k not in later}
  return {**spec, **FRONT_FLIP_STAGE_OVERRIDES.get(stage, {})}


def to_flip_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  spec: dict,
  feet_pattern: str,
  angmom_sensor: str,
  limbs_sensor: str,
  limbs_pattern: str,
  front_feet_pattern: str | None = None,
  rear_feet_pattern: str | None = None,
  knee_pattern: str | None = None,
  knee_force_sensor: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Turn a robot's DonkeyKick configuration into the flip `spec` (SIDE_FLIP,
  FRONT_FLIP) in place; returned.

  feet_pattern matches all four feet's contact geoms in DonkeyKick's feet sensor
  (front_feet_pattern / rear_feet_pattern the front and rear pairs, for the catch, the
  pivot and front_landing); angmom_sensor is the robot's subtreeangmom sensor on its base
  (whole-body angular momentum, world frame); limbs_sensor / limbs_pattern match the
  feet, calves and thighs against the terrain (airborne = none of them touching), and
  knee_pattern the thighs and calves among them (knee_down); knee_force_sensor matches
  the thighs and calves against the terrain with contact force (knee_impact). The
  robot entity must use DC motor actuators (apply_power_fraction). Mid-flip starts need
  their stances, hip_x and body_points set per robot afterwards.
  """
  timing, episode_s = spec["timing"], spec["episode_s"]
  # axis 0 is the terms' default, so SideFlip's term parameters don't list it.
  axis = {"axis": spec["axis"]} if spec["axis"] else {}
  fixed = {"fixed_axis": True} if spec.get("fixed_axis") else {}
  # Progress past half a turn and the landing only once upside down.
  gate = {"upside_down": spec["upside_down"]} if spec.get("upside_down") else {}
  sensor_name = cfg.rewards["kick_snap"].params["sensor_name"]
  feet = {"sensor_name": sensor_name, "geom_pattern": feet_pattern}
  limbs = {"sensor_name": limbs_sensor, "geom_pattern": limbs_pattern}
  # The rotation counts over the whole episode, so a second turn shows up everywhere.
  rotation = {
    "direction": spec["direction"],
    "start": 0.0,
    "end": episode_s,
    **limbs,
    "min_up_vel": LIFTOFF_UP_VEL,
    **axis,
    **fixed,
  }

  cfg.episode_length_s = episode_s
  # One term config shared by the actor and critic groups; kick_phase reaches 1 when
  # standing starts (the deploy reads the same duration).
  cfg.observations["actor"].terms["kick_phase"].params["duration"] = timing["stand_start"]

  ##
  # Rewards
  ##

  rewards = cfg.rewards
  for name in (
    "pitch_track", "rear_feet_air", "front_feet_planted", "kick_snap",
    "roll_level", "pitch_excess", "rear_landing", "front_slip",
  ):
    del rewards[name]
  track_key, progress_key = spec["keys"]
  # The flip, on Unitree's timeline (SideFlip).
  if spec["profile"] is not None:
    rewards[track_key] = RewardTermCfg(
      func=mdp.turn_track, weight=3.0, params={**rotation, "profile": spec["profile"], **ROLL_TRACK}
    )
  # Episode total = 10 x the best fraction of a full turn (after liftoff): finding the flip.
  rewards[progress_key] = RewardTermCfg(func=mdp.turn_progress, weight=10.0, params={**rotation, **gate})
  rewards["spin"] = RewardTermCfg(
    func=mdp.spin,
    weight=spec.get("spin_weight", 1.0),
    params={
      **rotation,
      "angmom_sensor": angmom_sensor,
      "target": spec["spin_target"],
      "max_turns": spec["spin_max_turns"],
      "pay_end": timing["spin_end"],
    },
  )
  # Once: episode total = 4 x the best height gained in the air (full at FLIGHT_HEIGHT).
  rewards["flight_height"] = RewardTermCfg(
    func=mdp.flight_height,
    weight=4.0,
    params={**limbs, "start": 0.0, "end": timing["spin_end"], "height": FLIGHT_HEIGHT},
  )
  rewards["airborne"] = RewardTermCfg(
    func=mdp.airborne, weight=0.5, params={**limbs, "start": 0.0, "end": timing["spin_end"]}
  )
  # Once: landed on the feet after one full turn (not two).
  touch = front_feet_pattern if spec.get("front_landing") else feet_pattern
  rewards["flip_landing"] = RewardTermCfg(
    func=mdp.flip_landing,
    weight=5.0,
    params={**rotation, "touch_sensor": sensor_name, "touch_pattern": touch, **spec["landing"], **gate},
  )
  if spec.get("clean_landing"):
    rewards["flip_landing"].params["contact_term"] = "body_contact"
  # Once: the rear feet catch the ground late in the turn, before any body part.
  if spec.get("catch"):
    catch = dict(spec["catch"])
    rewards["rear_catch"] = RewardTermCfg(
      func=mdp.rear_catch,
      weight=catch.pop("weight"),
      params={**rotation, "touch_sensor": sensor_name, "touch_pattern": rear_feet_pattern, **catch, **gate},
    )
  # The turn made on the front feet before the liftoff.
  if spec.get("pivot"):
    pivot = dict(spec["pivot"])
    rewards["front_pivot"] = RewardTermCfg(
      func=mdp.front_pivot,
      weight=pivot.pop("weight"),
      params={
        **rotation,
        "feet_sensor": sensor_name,
        "front_pattern": front_feet_pattern,
        "rear_pattern": rear_feet_pattern,
        **pivot,
      },
    )
  # A pure turn about the flip's axis: the other two turns cost.
  if spec.get("heading_penalty"):
    rewards["heading_hold"] = RewardTermCfg(
      func=mdp.heading_hold, weight=-1.0, params={**axis, "penalty": True}
    )
  else:
    rewards["heading_hold"] = RewardTermCfg(func=mdp.heading_hold, weight=1.0, params=dict(axis))
  after_landing = {"stand_pose": mdp.stand_pose_after_landing, "stand_still": mdp.stand_still_after_landing}
  for name in ("stand_pose", "stand_still"):
    rewards[name].params["start"] = 0.0 if spec.get("stand_from_landing") else timing["stand_start"]
    if spec.get("stand_after_landing"):
      rewards[name].func = after_landing[name]
      rewards[name].params["landing_term"] = "flip_landing"
      if spec.get("stand_from_catch"):
        rewards[name].params["catch_term"] = "rear_catch"
  # Up off the knees once caught or landed: a cost for kneeling, and a pull to the stand
  # pose from far off.
  if spec.get("knee_down"):
    rewards["knee_down"] = RewardTermCfg(
      func=mdp.knees_down_after_landing,
      weight=spec["knee_down"],
      params={
        "sensor_name": limbs_sensor,
        "geom_pattern": knee_pattern,
        "landing_term": "flip_landing",
        "catch_term": "rear_catch" if spec.get("catch") else None,
      },
    )
  if spec.get("wide_stand"):
    wide = dict(spec["wide_stand"])
    rewards["stand_pose_wide"] = RewardTermCfg(
      func=rewards["stand_pose"].func, weight=wide.pop("weight"), params={**rewards["stand_pose"].params, **wide}
    )
  # Knees off the ground through the catch and landing, not only after: contact force.
  if spec.get("knee_impact"):
    impact = dict(spec["knee_impact"])
    rewards["knee_impact"] = RewardTermCfg(
      func=mdp.part_contact_force,
      weight=impact.pop("weight"),
      params={"sensor_name": knee_force_sensor, "geom_pattern": knee_pattern, **impact},
    )
  # Both sides moving as mirror images (in the air only, with air_only).
  if spec.get("mirror_pose"):
    mirror = dict(spec["mirror_pose"])
    air = dict(limbs) if mirror.pop("air_only", False) else {}
    rewards["mirror_pose"] = RewardTermCfg(func=mdp.mirror_pose, weight=mirror.pop("weight"), params=air)
  # Legs under the hips on the way down (after the rotation term, which it reads), until
  # any leg part touches (a knee first closes it too).
  if spec.get("landing_pose"):
    pose = dict(spec["landing_pose"])
    rewards["landing_pose"] = RewardTermCfg(
      func=mdp.landing_pose,
      weight=pose.pop("weight"),
      params={**limbs, "launch_term": progress_key, **pose},
    )
  # Both rear feet down together at the catch, not one and then the other.
  if spec.get("rear_stagger"):
    rewards["rear_stagger"] = RewardTermCfg(
      func=mdp.rear_stagger,
      weight=spec["rear_stagger"],
      params={"sensor_name": sensor_name, "geom_pattern": rear_feet_pattern, "launch_term": progress_key},
    )
  rewards["termination"].weight = TERMINATION_WEIGHT
  if spec.get("commit_turns") is not None:
    # Same name and weight, so it stays in the penalty ramp.
    rewards["termination"].func = mdp.flip_termination
    rewards["termination"].params = {**rotation, "commit_turns": spec["commit_turns"]}
    if spec.get("crash_after_landing"):
      rewards["termination"].params["landing_term"] = "flip_landing"
    if spec.get("crash_after_catch"):
      rewards["termination"].params["catch_term"] = "rear_catch"
      # A rear foot alone isn't a catch with pair_window, but a crash after it still costs.
      if spec["catch"].get("pair_window") is not None:
        rewards["termination"].params["catch_any"] = True
  # Gentle (ramped in): soft four-foot landing, motors off their speed-dependent limit.
  impact = {"weight": -1.0, **LANDING_FORCE, **spec.get("feet_impact", {})}
  rewards["landing"] = RewardTermCfg(
    func=mdp.part_contact_force, weight=impact.pop("weight"), params={**feet, **impact}
  )
  rewards["dc_torque_saturation"] = RewardTermCfg(
    func=mdp.dc_torque_saturation, weight=-10.0, params={"fraction": DC_SATURATION_FRACTION}
  )
  if spec.get("free_launch"):
    rewards["dc_torque_saturation"].params.update(launch_term=progress_key, launch_end=timing["spin_end"])
  # A jump and a spin need a big, sudden action: QuickBuck's halved smoothness penalties.
  rewards["action_rate"].weight = -0.05
  rewards["joint_acc"].weight = -1.25e-7

  if "penalty_ramp" in cfg.curriculum:  # not in play mode
    ramp = cfg.curriculum["penalty_ramp"].params
    ramp["term_names"] = [
      "landing" if n == "rear_landing" else n for n in ramp["term_names"] if n != "front_slip"
    ] + ["dc_torque_saturation"]
  apply_power_fraction(cfg, POWER_FRACTION, JOINT_VEL_EXCESS_WEIGHT)

  ##
  # Terminations and metrics
  ##

  cfg.terminations["fell"] = TerminationTermCfg(func=mdp.flip_fell, params={**spec["fall"], **axis})

  del cfg.metrics["handstand"]
  # x episode length (2 s) = seconds with no limb on the ground.
  cfg.metrics["air"] = MetricsTermCfg(
    func=mdp.airborne, params={**limbs, "start": 0.0, "end": episode_s}
  )
  # x episode length = seconds past one turn (should stay ~0).
  cfg.metrics["over_rotated"] = MetricsTermCfg(
    func=mdp.over_rotated, params={**rotation, "max_turns": spec["landing"]["max_turns"]}
  )
  cfg.metrics["stand_done"].params["start"] = timing["stand_start"]
  # x episode length = seconds with a thigh or calf down after the catch or landing.
  if spec.get("knee_down"):
    cfg.metrics["knees_down"] = MetricsTermCfg(
      func=mdp.knees_down_after_landing, params=dict(rewards["knee_down"].params)
    )

  ##
  # Mid-flip starts (training only: play mode has no curriculum)
  ##

  if spec.get("mid_flip") and "penalty_ramp" in cfg.curriculum:
    base = cfg.events["reset_base"].params
    # After reset_base (events run in order), which places the standing starts.
    cfg.events["mid_flip_start"] = EventTermCfg(
      func=mdp.reset_mid_flip,
      mode="reset",
      params={
        **spec["mid_flip"],
        "xy_range": base["xy_range"],
        "footprint_radius": base["footprint_radius"],
        "leg_lengths": base["leg_lengths"],
        "foot_radius": base["foot_radius"],
        "stances": (),  # Set per-robot.
        "hip_x": 0.0,  # Set per-robot.
        "body_points": (),  # Set per-robot.
        "axis": spec["axis"],
        "direction": spec["direction"],
      },
    )
    # x episode length = seconds after a landing, standing starts / mid-flip starts.
    for name, mid in (("landed", False), ("landed_mid", True)):
      cfg.metrics[name] = MetricsTermCfg(
        func=mdp.flip_landed, params={"landing_term": "flip_landing", "mid_start": mid}
      )
    # x episode length = seconds after a rear-feet catch.
    if spec.get("catch"):
      for name, mid in (("caught", False), ("caught_mid", True)):
        cfg.metrics[name] = MetricsTermCfg(
          func=mdp.flip_caught, params={"catch_term": "rear_catch", "mid_start": mid}
        )

  return cfg


def to_side_flip_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  feet_pattern: str,
  angmom_sensor: str,
  limbs_sensor: str,
  limbs_pattern: str,
) -> ManagerBasedRlEnvCfg:
  """DonkeyKick configuration -> SideFlip (to_flip_env_cfg with SIDE_FLIP)."""
  return to_flip_env_cfg(cfg, SIDE_FLIP, feet_pattern, angmom_sensor, limbs_sensor, limbs_pattern)


def to_front_flip_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  feet_pattern: str,
  angmom_sensor: str,
  limbs_sensor: str,
  limbs_pattern: str,
  front_feet_pattern: str,
  rear_feet_pattern: str,
  knee_pattern: str,
  knee_force_sensor: str,
  stage: int = 4,
) -> ManagerBasedRlEnvCfg:
  """DonkeyKick configuration -> FrontFlip (to_flip_env_cfg with FRONT_FLIP, or the
  multistage recipe's earlier stage)."""
  return to_flip_env_cfg(
    cfg, front_flip_stage(stage), feet_pattern, angmom_sensor, limbs_sensor, limbs_pattern, front_feet_pattern,
    rear_feet_pattern, knee_pattern, knee_force_sensor,
  )
