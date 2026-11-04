"""Unitree Go2 DonkeyKick, QuickBuck, SideFlip, FrontFlip and BackFlip environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg

from src.assets.robots import get_go2_dc_robot_cfg, get_go2_robot_cfg
from src.tasks.kick.buck_env_cfg import to_buck_env_cfg
from src.tasks.kick.flip_env_cfg import to_back_flip_env_cfg, to_front_flip_env_cfg, to_side_flip_env_cfg
from src.tasks.kick.kick_env_cfg import make_kick_env_cfg
from src.tasks.velocity.mdp import LiftedGridPatternCfg

_FOOT_GEOMS = tuple(f"{foot}_foot_collision" for foot in ("FR", "FL", "RR", "RL"))
_REAR_FEET = r"^R[LR]_foot_collision$"
_FRONT_FEET = r"^F[LR]_foot_collision$"
_ALL_FEET = r"^[FR][LR]_foot_collision$"
_LIMBS = r"^[FR][LR]_(thigh|calf\d|foot)_collision$"
_KNEES = r"^[FR][LR]_(thigh|calf\d)_collision$"
# The head and chin (base2 / base3; the L1 LiDAR sits at the chin).
_HEAD = r"^base[23]_collision$"

# The two stances the deploy starts the kick from (joint regex -> rad).
FIXSTAND_STANCE = {r".*hip_joint": 0.0, r".*thigh_joint": 0.8, r".*calf_joint": -1.5}
VELOCITY_STANCE = {  # = INIT_STATE, the Go2 default pose
  r".*R_hip_joint": 0.1,
  r".*L_hip_joint": -0.1,
  r".*thigh_joint": 0.9,
  r".*calf_joint": -1.8,
}

# FrontFlip mid-flip starts' leg poses: FrontFlip v5's in the air (front legs forward, hips
# splayed, rear legs tucked under the belly), and the stand pose (VELOCITY_STANCE: the
# legs landing_pose pays for; multistage v2, in place of rear legs swung back for v8's
# ~262 deg catch).
FLIP_TUCKED_STANCE = {
  r"FL_hip_joint": 0.5,
  r"FR_hip_joint": -0.5,
  r"F[LR]_thigh_joint": 1.5,
  r"F[LR]_calf_joint": -1.7,
  r"RL_hip_joint": -0.3,
  r"RR_hip_joint": 0.3,
  r"R[LR]_thigh_joint": -0.15,
  r"R[LR]_calf_joint": -2.1,
}
# go2.xml: hip axes at x +-0.1934 m; base box corners, head and hips as (x, z, radius) (m).
_HIP_X = 0.1934
_BODY_POINTS = (
  (0.1881, 0.057, 0.0), (0.1881, -0.057, 0.0), (-0.1881, 0.057, 0.0), (-0.1881, -0.057, 0.0),
  (0.285, 0.01, 0.05), (0.293, -0.06, 0.047),
  (_HIP_X, 0.0, 0.05), (-_HIP_X, 0.0, 0.05),
)


def unitree_go2_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 DonkeyKick configuration.

  Standard Go2 (default pose = Velocity's stance, locomotion PD gains, full
  collisions). The deploy DonkeyKick state hands back once STAND_CHECK holds.
  """
  cfg = make_kick_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Terrain-only rays (robot geoms are groups 2/3), starting 1 m above the base
  # (as in the velocity and recovery tasks).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"
      sensor.include_geom_groups = (0, 1)
      sensor.pattern = LiftedGridPatternCfg(
        size=sensor.pattern.size,
        resolution=sensor.pattern.resolution,
        direction=sensor.pattern.direction,
        z_offset=1.0,
      )

  # Feet against the terrain (contact flags and force per foot, every substep).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground",
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  # Base (incl. the head, base2/base3) and hips against the terrain: ends the episode.
  body_ground_cfg = ContactSensorCfg(
    name="body_ground",
    primary=ContactMatch(mode="geom", entity="robot", pattern=r"^(base\d|[FR][LR]_hip)_collision$"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, body_ground_cfg)

  reset = cfg.events["reset_base"].params
  reset["stances"] = (FIXSTAND_STANCE, VELOCITY_STANCE)
  reset["leg_lengths"] = (0.213, 0.213)  # thigh, calf (m)
  reset["foot_radius"] = 0.022

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["rear_feet_air"].params["geom_pattern"] = _REAR_FEET
  cfg.rewards["front_feet_planted"].params["geom_pattern"] = _FRONT_FEET
  cfg.rewards["rear_landing"].params["geom_pattern"] = _REAR_FEET
  cfg.rewards["front_slip"].params["geom_pattern"] = _FRONT_FEET
  cfg.rewards["kick_snap"].params["feet"] = ("RL", "RR")  # foot sites
  cfg.rewards["kick_snap"].params["hips"] = ("RL_thigh", "RR_thigh")  # thigh joint bodies
  cfg.rewards["kick_snap"].params["geom_pattern"] = _REAR_FEET
  cfg.metrics["handstand"].params["geom_pattern"] = _REAR_FEET

  cfg.viewer.body_name = "base_link"

  # Play: a new kick every episode (3.5 s), on a smaller terrain.
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # mujoco_warp sizes the contact buffer from the default-pose robot at the world
    # origin, which can be inside a stair pyramid in the 5x5 play layout.
    cfg.sim.nconmax = None
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_buck_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 QuickBuck configuration.

  The DonkeyKick setup (starts, sensors, terrain, randomisation, play mode) with DC
  motor torque-speed curves, then QuickBuck's timing and rewards. The deploy
  QuickBuck state (type DonkeyKick) hands back once STAND_CHECK holds.
  """
  cfg = unitree_go2_kick_env_cfg(play)
  cfg.scene.entities = {"robot": get_go2_dc_robot_cfg()}
  return to_buck_env_cfg(cfg)


def _flip_base_cfg(play: bool) -> ManagerBasedRlEnvCfg:
  """The DonkeyKick setup (starts, sensors, terrain, randomisation, play mode) with DC
  motor torque-speed curves and the limbs_ground sensor, for the flips."""
  cfg = unitree_go2_kick_env_cfg(play)
  cfg.scene.entities = {"robot": get_go2_dc_robot_cfg()}
  # Feet, calves and thighs against the terrain: airborne means none of them touching
  # (SideFlip v1 used the feet only, so a thigh on the ground still counted as in the air).
  limbs_ground_cfg = ContactSensorCfg(
    name="limbs_ground",
    primary=ContactMatch(mode="geom", pattern=_LIMBS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (limbs_ground_cfg,)
  return cfg


# go2.xml's subtreeangmom sensor on base_link, and the flips' contact sets.
_FLIP_SENSORS = {
  "feet_pattern": _ALL_FEET,
  "angmom_sensor": "robot/root_angmom",
  "limbs_sensor": "limbs_ground",
  "limbs_pattern": _LIMBS,
}


def unitree_go2_side_flip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 SideFlip configuration (the left flip; the right one is it mirrored).

  The flips' DonkeyKick-based setup, then SideFlip's timing and rewards.
  """
  return to_side_flip_env_cfg(_flip_base_cfg(play), **_FLIP_SENSORS)


def _nose_flip_env_cfg(play: bool, stage: int, back: bool) -> ManagerBasedRlEnvCfg:
  """The flips' DonkeyKick-based setup with the knees_ground sensor, then FrontFlip's (or
  BackFlip's) timing and rewards, and the mid-flip starts' leg poses."""
  cfg = _flip_base_cfg(play)
  # Thighs and calves against the terrain with contact force (peak over the step's
  # substeps): the knee impact penalty.
  knees_ground_cfg = ContactSensorCfg(
    name="knees_ground",
    primary=ContactMatch(mode="geom", pattern=_KNEES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = cfg.scene.sensors + (knees_ground_cfg,)
  patterns = {
    **_FLIP_SENSORS, "front_feet_pattern": _FRONT_FEET, "rear_feet_pattern": _REAR_FEET,
    "knee_pattern": _KNEES, "knee_force_sensor": "knees_ground", "stage": stage,
  }
  if back:
    cfg = to_back_flip_env_cfg(cfg, **patterns, head_pattern=_HEAD)
  else:
    cfg = to_front_flip_env_cfg(cfg, **patterns)
  if "mid_flip_start" in cfg.events:  # training only
    mid = cfg.events["mid_flip_start"].params
    mid["stances"] = (FLIP_TUCKED_STANCE, VELOCITY_STANCE)
    mid["hip_x"] = _HIP_X
    mid["body_points"] = _BODY_POINTS
  return cfg


def unitree_go2_front_flip_env_cfg(play: bool = False, stage: int = 4) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 FrontFlip configuration (one turn nose-down, over its back).

  The flips' DonkeyKick-based setup, then FrontFlip's timing and rewards; stage 1 to 3
  are the multistage recipe's training stages (front_flip_stage), 4 the full FrontFlip.
  """
  return _nose_flip_env_cfg(play, stage, back=False)


def unitree_go2_back_flip_env_cfg(play: bool = False, stage: int = 4) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 BackFlip configuration (one turn nose-up, over its back).

  FrontFlip's setup with BackFlip's rewards (back_flip_stage: stage 1 to 3 train, 4 the
  full BackFlip); the mid-flip starts use FrontFlip's leg poses.
  """
  return _nose_flip_env_cfg(play, stage, back=True)
