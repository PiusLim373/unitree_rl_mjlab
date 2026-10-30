"""Unitree Go2 constants."""

from dataclasses import replace
from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

# Motor top speeds (rad/s), from Unitree's go2_description.urdf joint velocity limits.
# The calf's extra gearing trades speed for torque.
GO2_HIP_THIGH_TOP_SPEED = 30.1
GO2_CALF_TOP_SPEED = 15.7

# The same motors with a DC torque-speed curve: torque falls linearly from the limit
# at rest to 0 at the top speed (as Isaac Lab's Go2 DCMotorCfg). The position
# actuators above give full torque at any speed, which overstates fast moves
# (QuickBuck's kick). Same gains, armature and torque limits.
GO2_DC_ACTUATOR_HIP = DcMotorActuatorCfg(
  target_names_expr=GO2_ACTUATOR_HIP.target_names_expr,
  stiffness=GO2_ACTUATOR_HIP.stiffness,
  damping=GO2_ACTUATOR_HIP.damping,
  effort_limit=GO2_ACTUATOR_HIP.effort_limit,
  saturation_effort=GO2_ACTUATOR_HIP.effort_limit,
  velocity_limit=GO2_HIP_THIGH_TOP_SPEED,
  armature=GO2_ACTUATOR_HIP.armature,
)
GO2_DC_ACTUATOR_THIGH = DcMotorActuatorCfg(
  target_names_expr=GO2_ACTUATOR_THIGH.target_names_expr,
  stiffness=GO2_ACTUATOR_THIGH.stiffness,
  damping=GO2_ACTUATOR_THIGH.damping,
  effort_limit=GO2_ACTUATOR_THIGH.effort_limit,
  saturation_effort=GO2_ACTUATOR_THIGH.effort_limit,
  velocity_limit=GO2_HIP_THIGH_TOP_SPEED,
  armature=GO2_ACTUATOR_THIGH.armature,
)
GO2_DC_ACTUATOR_CALF = DcMotorActuatorCfg(
  target_names_expr=GO2_ACTUATOR_CALF.target_names_expr,
  stiffness=GO2_ACTUATOR_CALF.stiffness,
  damping=GO2_ACTUATOR_CALF.damping,
  effort_limit=GO2_ACTUATOR_CALF.effort_limit,
  saturation_effort=GO2_ACTUATOR_CALF.effort_limit,
  velocity_limit=GO2_CALF_TOP_SPEED,
  armature=GO2_ACTUATOR_CALF.armature,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

# For getting up after a fall: the robot rolls on its body and legs, so every geom
# gets friction (condim 3) and the robot collides with itself (conaffinity 1;
# MuJoCo still skips parent-child pairs). Priority 1 makes the robot's own
# friction (randomized at startup) apply against the terrain.
RECOVERY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=1,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO2_DC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_DC_ACTUATOR_HIP,
    GO2_DC_ACTUATOR_THIGH,
    GO2_DC_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=GO2_ARTICULATION.soft_joint_pos_limit_factor,
)

# The lying pose's calf angle (-2.65) is outside the 0.9 soft limits (-2.63).
# Recovery v3 doubles the joint damping (Kd 2/2/3 vs 1/1/2, same stiffness) so leg
# sweeps and the roll over the hips are physically softer; v1/v2 used 1/1/2.
GO2_RECOVERY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    replace(GO2_ACTUATOR_HIP, damping=2.0),
    replace(GO2_ACTUATOR_THIGH, damping=2.0),
    replace(GO2_ACTUATOR_CALF, damping=3.0),
  ),
  soft_joint_pos_limit_factor=0.95,
)

# Lying belly-down with legs folded: the crouch the deploy JointLock starts from.
LYING_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.15),
  joint_pos={
    ".*hip_joint": 0.0,
    ".*thigh_joint": 1.36,
    ".*calf_joint": -2.65,
  },
  joint_vel={".*": 0.0},
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


def get_go2_dc_robot_cfg() -> EntityCfg:
  """Go2 as get_go2_robot_cfg, with DC motor torque-speed curves (QuickBuck)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_DC_ARTICULATION,
  )


def get_go2_recovery_robot_cfg() -> EntityCfg:
  """Go2 for the Recovery task: lying pose as default, full contacts with friction."""
  return EntityCfg(
    init_state=LYING_STATE,
    collisions=(RECOVERY_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_RECOVERY_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
