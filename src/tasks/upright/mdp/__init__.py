from mjlab.envs.mdp import *  # noqa: F401, F403

from src.tasks.kick.mdp.flip import dc_torque_saturation as dc_torque_saturation
from src.tasks.recovery.mdp.curriculums import penalty_ramp as penalty_ramp
from src.tasks.recovery.mdp.rewards import part_contact_force as part_contact_force
from src.tasks.velocity.mdp import angular_momentum_penalty as angular_momentum_penalty
from src.tasks.velocity.mdp import body_angular_velocity_penalty as body_angular_velocity_penalty
from src.tasks.velocity.mdp import contact_fraction as contact_fraction
from src.tasks.velocity.mdp import feet_clearance_terrain as feet_clearance_terrain
from src.tasks.velocity.mdp import feet_slip as feet_slip
from src.tasks.velocity.mdp import reward_weight as reward_weight
from src.tasks.velocity.mdp import self_collision_cost as self_collision_cost
from src.tasks.velocity.mdp import soft_landing as soft_landing

from .actions import *  # noqa: F403
from .commands import *  # noqa: F403
from .events import *  # noqa: F403
from .frames import *  # noqa: F403
from .power import *  # noqa: F403
from .upright import *  # noqa: F403
