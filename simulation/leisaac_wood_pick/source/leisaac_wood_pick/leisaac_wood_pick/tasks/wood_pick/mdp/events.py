"""Reset-time randomization for scripted data generation."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..grasp_planner import StickSamplingConfig, sample_stick_root_poses


def reset_robot_to_reachable_seed(
    env: ManagerBasedRLEnv | DirectRLEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("stick"),
) -> None:
    """Reset SO-101 to a non-singular seed aimed at the sampled stick."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_names = list(robot.data.joint_names)
    stick: RigidObject = env.scene[object_cfg.name]
    relative = stick.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2]
    pan = torch.atan2(relative[:, 0], relative[:, 1]).clamp(
        min=torch.deg2rad(torch.tensor(-100.0, device=env.device)),
        max=torch.deg2rad(torch.tensor(100.0, device=env.device)),
    )
    seed_degrees = {
        "shoulder_lift": 30.0,
        "elbow_flex": 30.0,
        "wrist_flex": -60.0,
        "wrist_roll": 0.0,
        "gripper": 7.0,
    }
    joint_pos[:, joint_names.index("shoulder_pan")] = pan
    for name, degrees in seed_degrees.items():
        joint_pos[:, joint_names.index(name)] = torch.deg2rad(
            torch.tensor(degrees, device=env.device)
        )
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_stick_on_platform(
    env: ManagerBasedRLEnv | DirectRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stick"),
    platform_center_xy: tuple[float, float] = (0.235, 0.362),
    platform_size_xy: tuple[float, float] = (0.270, 0.100),
    platform_top_z: float = 0.053,
    max_object_radius: float = 0.38,
) -> None:
    """Place the stick at a random, fully supported pose on the platform.

    Sampling accounts for the yaw-dependent footprint and rejects coarse
    workspace outliers.  It writes zero root velocity so every episode starts
    from a clean physical state.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    config = StickSamplingConfig(
        platform_center_xy=platform_center_xy,
        platform_size_xy=platform_size_xy,
        platform_top_z=platform_top_z,
        max_object_radius=max_object_radius,
    )
    positions, quaternions = sample_stick_root_poses(env_ids.numel(), device=env.device, config=config)
    positions = positions + env.scene.env_origins[env_ids]
    root_pose = torch.cat((positions, quaternions), dim=-1)
    root_velocity = torch.zeros((env_ids.numel(), 6), device=env.device)

    stick: RigidObject = env.scene[asset_cfg.name]
    stick.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    stick.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
