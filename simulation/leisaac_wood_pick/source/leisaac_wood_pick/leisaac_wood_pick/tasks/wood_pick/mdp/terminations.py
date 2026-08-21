"""Success predicates for the measured destination box."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..grasp_planner import stick_inside_box


def stick_in_destination_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stick"),
    box_lower_left: tuple[float, float, float] = (-0.107, 0.380, 0.0),
    box_size: tuple[float, float, float] = (0.150, 0.100, 0.085),
    wall_thickness: float = 0.003,
    maximum_linear_speed: float = 0.08,
    maximum_angular_speed: float = 1.0,
) -> torch.Tensor:
    """Return true only when the whole, settled stick lies inside the box.

    A centre-only test incorrectly accepts a stick balanced across a wall.  We
    transform all eight corners of the measured STL bounds and require every
    corner to lie within the cardboard interior and below the rim.
    """
    stick: RigidObject = env.scene[asset_cfg.name]
    relative_position = stick.data.root_pos_w - env.scene.env_origins
    inside = stick_inside_box(
        relative_position,
        stick.data.root_quat_w,
        box_lower_left=box_lower_left,
        box_size=box_size,
        wall_thickness=wall_thickness,
    )
    linear_speed = torch.linalg.vector_norm(stick.data.root_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(stick.data.root_ang_vel_w, dim=-1)
    return inside & (linear_speed <= maximum_linear_speed) & (angular_speed <= maximum_angular_speed)
