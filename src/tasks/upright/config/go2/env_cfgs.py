"""Unitree Go2 UprightWalk environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import get_go2_upright_robot_cfg
from src.tasks.kick.config.go2.env_cfgs import VELOCITY_STANCE
from src.tasks.upright.upright_env_cfg import to_upright_env_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_blind_env_cfg

# Around the upright pose: rear legs as Rough-Blind's walking std, the arms looser, but
# their hips tight (loose, the forearms turned in toward the belly while standing).
_POSTURE_STD = {
  r"R[LR]_hip_joint": 0.15,
  r"R[LR]_thigh_joint": 0.35,
  r"R[LR]_calf_joint": 0.5,
  r"F[LR]_hip_joint": 0.15,
  r"F[LR]_(thigh|calf)_joint": 0.4,
}


def _feet_sensor(name: str, feet: tuple[str, ...], air_time: bool) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=tuple(f"{f}_foot_collision" for f in feet), entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=air_time,
    history_length=4,  # every substep (front_impact)
  )


def unitree_go2_upright_env_cfg(play: bool = False, stage: int = 2) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 UprightWalk configuration (stage 1 or 2, upright_env_cfg.STAGES).

  Rough-Blind's setup with the upright Go2 (DC motor curves, upright default pose), the
  rear and front feet contact sensors, and no height scan (its yaw-aligned grid has no
  heading upright, and the ground is flat or gently rough).
  """
  cfg = unitree_go2_rough_blind_env_cfg(play)
  cfg.scene.entities = {"robot": get_go2_upright_robot_cfg()}
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan") + (
    _feet_sensor("rear_feet_ground", ("RL", "RR"), True),
    _feet_sensor("front_feet_ground", ("FL", "FR"), False),
    # Arms against the rest of the robot (UPRIGHT_COLLISION), every substep.
    ContactSensorCfg(
      name="arm_touch",
      primary=ContactMatch(mode="geom", pattern=r"^F[LR]_(thigh|calf\d|foot)_collision$", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),  # the whole robot
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )
  cfg.observations["critic"].terms.pop("height_scan", None)
  return to_upright_env_cfg(
    cfg,
    stage,
    rear_feet_sensor="rear_feet_ground",
    front_feet_sensor="front_feet_ground",
    feet_sensor="feet_ground_contact",
    rear_sites=("RL", "RR"),
    foot_sites=("FL", "FR", "RL", "RR"),
    rear_rays=("foot_ray_RL", "foot_ray_RR"),
    leg_sensor="leg_ground_touch",
    arm_sensor="arm_touch",
    stand_stances=(VELOCITY_STANCE,),
    posture_std=_POSTURE_STD,
    rear_actuators=r"R[LR]_.*",
    upright_height=0.53,
    four_leg_height=0.29,
    reset={"leg_lengths": (0.213, 0.213), "foot_radius": 0.022, "hip_x": 0.1934},
  )
