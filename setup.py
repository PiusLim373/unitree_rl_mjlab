"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    # mjlab/mujoco-warp only set lower bounds; newer releases break them.
    "mujoco==3.5.0",  # mujoco>=3.16 drops mjENBL_MULTICCD used by mujoco-warp 3.5.0
    "warp-lang==1.12.1",  # warp-lang 1.18 drops wp.context used by mjlab 1.2.0
    "scipy",  # imported by mjlab terrains but not declared by mjlab
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
