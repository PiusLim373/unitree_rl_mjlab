"""Unitree Go2 velocity environment configurations."""

from dataclasses import replace
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  CurriculumTermCfg,
  MetricsTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.mdp import (
  LiftedGridPatternCfg,
  command_speed_xy,
  contact_fraction,
  feet_clearance_terrain,
  foot_clearance,
  self_collision_cost,
  terrain_levels_tracking,
  track_forward_velocity,
  track_lateral_velocity,
  vel_error_xy,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link. Groups (0, 1) let the rays see only
  # the terrain: the default (0, 1, 2) includes the Go2's visual meshes (group 2),
  # so ~4-5 of the 187 rays stopped on its own legs. A real LiDAR height map
  # self-filters the legs too, so terrain-only matches it.
  # Rays start 1 m above the base: from base height, a ray over terrain taller
  # than the base (e.g. 15 cm stairs seen from a pit) starts inside the step box
  # and reports its bottom, so tall steps read as flat floor. Values are still
  # base_z - ground_z, now negative where the terrain is above the base.
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

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_rough_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration with a blind actor.

  Sideways commands are capped at +-0.5 with an extra sideways tracking reward,
  the actor sees a 5-step observation history, turning is tracked more strictly,
  and terrain levels follow command tracking rather than distance from spawn.
  Foot clearance is measured from the ground under each foot (one-sided, 15 cm,
  the same push at any walking speed), stairs go up to 15 cm, and
  forward/backward speed has its own tight reward.
  Episodes end only when the body (base or hips) hits the ground or the robot
  tips over; thigh and calf contacts are penalized instead.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  # Remove height scan from the actor only, so its observations match the flat
  # task and the deploy controller. The critic keeps it as privileged info.
  del cfg.observations["actor"].terms["height_scan"]

  # Forward/backward and turning up to +-1.0, sideways capped at +-0.5: v3 refused
  # blind sidestepping beyond that on rough terrain. Play mode uses these ranges
  # directly; training widens backward in steps.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  if "command_vel" in cfg.curriculum:  # Cleared in play mode.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      {"step": 1500 * 24, "lin_vel_x": (-0.75, 1.0)},
      {"step": 3000 * 24, "lin_vel_x": (-1.0, 1.0)},
    ]

  # Sideways-only tracking on top of the combined xy term, so moving sideways pays
  # off on its own (v3 and the first run barely moved sideways).
  cfg.rewards["track_lateral_velocity"] = RewardTermCfg(
    func=track_lateral_velocity,
    weight=0.5,
    params={"command_name": "twist", "std": 0.25},
  )

  # Half the robots get direct turn-rate commands (as Nav2/joystick send) instead
  # of heading control, and a stricter turning std so a constant yaw bias costs
  # reward (v4 drifted right ~2.3 deg/s when told to go straight).
  twist_cmd.rel_heading_envs = 0.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35

  # Short memory (0.1 s) so the blind actor can average noisy readings and feel
  # recent contacts. The critic keeps a single step.
  cfg.observations["actor"].history_length = 5

  # Terrain levels follow how well each robot tracked its commands over the
  # episode. Direct turn-rate commands make robots walk in circles, which the
  # distance-from-spawn rule would wrongly demote.
  cfg.metrics["vel_err_xy"] = MetricsTermCfg(
    func=vel_error_xy, params={"command_name": "twist"}
  )
  cfg.metrics["cmd_speed_xy"] = MetricsTermCfg(
    func=command_speed_xy, params={"command_name": "twist"}
  )
  if "terrain_levels" in cfg.curriculum:  # Cleared in play mode.
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=terrain_levels_tracking,
      params={
        "promote_max_err": 0.25,
        "demote_min_err": 0.5,
        "min_cmd_speed": 0.2,
        "min_steps": 50,
      },
    )

  # One downward ray per foot, so clearance is measured from the ground under the
  # foot. Groups (0, 1) hit only the terrain (robot geoms are groups 2 and 3).
  foot_rays = tuple(
    RayCastSensorCfg(
      name=f"foot_ray_{site}",
      frame=ObjRef(type="site", name=site, entity="robot"),
      pattern=GridPatternCfg(size=(0.0, 0.0)),
      ray_alignment="world",
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0, 1),
    )
    for site in ("FR", "FL", "RR", "RL")
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + foot_rays
  ray_names = tuple(ray.name for ray in foot_rays)

  # The base clearance term uses world z and is two-sided, so on raised steps any
  # lift was penalized and swings stayed low. This one only penalizes being lower
  # than 15 cm (foot center) above the ground below.
  # The foot-speed weight saturates at 0.3 m/s, so the same shortfall costs the
  # same at any walking speed (a swinging foot moves at ~2.3x body speed, i.e.
  # >0.4 m/s even at 0.2 m/s walking). Weight 2.5 roughly matches v7's push at
  # 1.0 m/s, where it lifted ~14.6 cm; v7 lifted only ~9.4 cm at 0.5 m/s.
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=feet_clearance_terrain,
    weight=-2.5,
    params={
      "target_height": 0.15,
      "sensor_names": ray_names,
      "contact_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "speed_saturation": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # The critic sees the same ground-relative foot heights (still 4 values).
  cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
    func=foot_clearance, params={"sensor_names": ray_names}
  )

  # Forward/backward-only tracking with a tight std, on top of the combined xy
  # term (std 0.5), which barely pays for closing a 0.2-0.3 m/s shortfall.
  cfg.rewards["track_forward_velocity"] = RewardTermCfg(
    func=track_forward_velocity,
    weight=0.5,
    params={"command_name": "twist", "std": 0.25},
  )

  # Stairs up to 15 cm risers (base: 10 cm). Copy the dict first: the generator
  # is a shallow copy of ROUGH_TERRAINS_CFG, shared with the Rough task.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    terrain_gen = cfg.scene.terrain.terrain_generator
    terrain_gen.sub_terrains = dict(terrain_gen.sub_terrains)
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
      terrain_gen.sub_terrains[name] = replace(
        terrain_gen.sub_terrains[name], step_height_range=(0.0, 0.15)
      )

  # End episodes only on body contact (base or hips) or tipping over. Thigh and
  # calf contacts become a penalty, so a shin bump on a stair edge is something
  # to recover from rather than an instant ending (>90% of v5/v6 endings were
  # calves touching stair edges). Split the non-foot sensor in two to do this.
  nonfoot = next(s for s in cfg.scene.sensors if s.name == "nonfoot_ground_touch")
  assert isinstance(nonfoot, ContactSensorCfg)
  body_ground_cfg = replace(
    nonfoot,
    name="body_ground_touch",
    primary=ContactMatch(
      mode="geom", entity="robot", pattern=r"^(base\d|[FR][LR]_hip)_collision$"
    ),
  )
  leg_ground_cfg = replace(
    nonfoot,
    name="leg_ground_touch",
    primary=ContactMatch(
      mode="geom", entity="robot", pattern=r"^[FR][LR]_(thigh|calf\d)_collision$"
    ),
  )
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if s.name != "nonfoot_ground_touch"
  ) + (body_ground_cfg, leg_ground_cfg)
  del cfg.terminations["illegal_contact"]
  cfg.terminations["body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": body_ground_cfg.name, "force_threshold": 10.0},
  )
  # Counts the 0-4 physics substeps per step with any thigh/calf pressing > 10 N.
  # Continuous contact costs 4 * 0.5 = 2 per second, about the whole tracking
  # reward, so walking on the shins never pays; a 0.1 s bump costs 0.2, ~20x less
  # than an episode ending (200 * 0.02 = 4).
  cfg.rewards["leg_contact"] = RewardTermCfg(
    func=self_collision_cost,
    weight=-0.5,
    params={"sensor_name": leg_ground_cfg.name, "force_threshold": 10.0},
  )
  cfg.metrics["leg_contact"] = MetricsTermCfg(
    func=contact_fraction,
    params={"sensor_name": leg_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    # mujoco_warp checks once at startup that the default-pose robot (at the world
    # origin) has at most nconmax contacts. In the random 5x5 play layout the origin
    # can be inside a stair pyramid (up to 62 contacts vs 35), so let it size itself.
    cfg.sim.nconmax = None

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
