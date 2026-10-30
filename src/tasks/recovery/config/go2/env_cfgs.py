"""Unitree Go2 recovery environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg

from src.assets.robots import get_go2_recovery_robot_cfg
from src.tasks.recovery.recovery_env_cfg import make_recovery_env_cfg
from src.tasks.velocity.mdp import LiftedGridPatternCfg

_FOOT_GEOMS = tuple(f"{foot}_foot_collision" for foot in ("FR", "FL", "RR", "RL"))


def unitree_go2_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 recovery configuration: get up into the lying pose.

  The lying pose is the crouch the deploy JointLock starts from (hips 0, thighs
  1.36, calves -2.65). The deploy Recovery state hands over to damping once the
  robot holds it (see DONE_CHECK).
  """
  cfg = make_recovery_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_recovery_robot_cfg()}

  # Terrain-only rays (robot geoms are groups 2/3), starting 1 m above the base
  # so terrain taller than the base reads correctly (as in the velocity task).
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

  # Body and leg geoms against the terrain, for body_impact (force per geom,
  # every physics substep).
  body_ground_cfg = ContactSensorCfg(
    name="body_ground",
    primary=ContactMatch(
      mode="geom", entity="robot", pattern=r".*_collision\d*$", exclude=_FOOT_GEOMS
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # The robot against itself, for self_collision.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (body_ground_cfg, self_collision_cfg)

  cfg.events["robot_friction"].params["asset_cfg"].geom_names = (r".*_collision$",)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  # Head = the front cylinder and the chin sphere (where the L1 LiDAR sits).
  cfg.rewards["head_contact"].params["geom_pattern"] = r"^base[23]_collision$"
  cfg.rewards["roll_contact"].params["geom_pattern"] = r"_(hip|thigh)_collision$"
  cfg.rewards["leg_slap"].params["geom_pattern"] = r"_(thigh|calf\d)_collision$"

  cfg.viewer.body_name = "base_link"

  # Play: a fresh fall every episode (8 s), on a smaller terrain.
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # mujoco_warp checks at startup that the default-pose robot at the world origin
    # fits in nconmax; in the 5x5 play layout the origin can be inside a stair
    # pyramid (38 contacts vs 30), so let it size itself (as Rough-Blind play does).
    cfg.sim.nconmax = None
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
