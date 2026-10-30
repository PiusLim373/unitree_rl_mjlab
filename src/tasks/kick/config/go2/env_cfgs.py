"""Unitree Go2 DonkeyKick and QuickBuck environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg

from src.assets.robots import get_go2_dc_robot_cfg, get_go2_robot_cfg
from src.tasks.kick.buck_env_cfg import to_buck_env_cfg
from src.tasks.kick.kick_env_cfg import make_kick_env_cfg
from src.tasks.velocity.mdp import LiftedGridPatternCfg

_FOOT_GEOMS = tuple(f"{foot}_foot_collision" for foot in ("FR", "FL", "RR", "RL"))
_REAR_FEET = r"^R[LR]_foot_collision$"
_FRONT_FEET = r"^F[LR]_foot_collision$"

# The two stances the deploy starts the kick from (joint regex -> rad).
FIXSTAND_STANCE = {r".*hip_joint": 0.0, r".*thigh_joint": 0.8, r".*calf_joint": -1.5}
VELOCITY_STANCE = {  # = INIT_STATE, the Go2 default pose
  r".*R_hip_joint": 0.1,
  r".*L_hip_joint": -0.1,
  r".*thigh_joint": 0.9,
  r".*calf_joint": -1.8,
}


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
