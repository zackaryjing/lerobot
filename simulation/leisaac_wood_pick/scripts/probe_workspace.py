"""Probe SO-101 FK on a coarse joint grid without rendering cameras."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(PROJECT_DIR / "assets" / "leisaac"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
parser.add_argument("--top_k", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac_wood_pick  # noqa: F401
from leisaac_wood_pick.tasks.wood_pick import TASK_ID
from leisaac_wood_pick.tasks.wood_pick.grasp_planner import quat_apply


def main() -> None:
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    env_cfg.scene.front = None
    env_cfg.scene.wrist = None
    env_cfg.observations.policy.front = None
    env_cfg.observations.policy.wrist = None
    env_cfg.recorders = None
    env_cfg.terminations = {}
    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("gripper")
    gripper_body_id = body_ids[0]
    target = torch.tensor(args_cli.target, device=env.device)
    offset = torch.tensor((0.030, 0.010, -0.100), device=env.device)
    names = list(robot.data.joint_names)
    name_to_id = {name: index for index, name in enumerate(names)}

    pan_values = (-75, -50, -25, 0, 25, 50, 75)
    shoulder_values = (-90, -60, -30, 0, 30, 60, 90)
    elbow_values = (-90, -60, -30, 0, 30, 60, 85)
    wrist_values = (-90, -60, -30, 0, 30, 60, 90)
    results: list[tuple[float, tuple[int, int, int, int], list[float]]] = []

    for pan, shoulder, elbow, wrist in itertools.product(
        pan_values, shoulder_values, elbow_values, wrist_values
    ):
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[0, name_to_id["shoulder_pan"]] = math.radians(pan)
        joint_pos[0, name_to_id["shoulder_lift"]] = math.radians(shoulder)
        joint_pos[0, name_to_id["elbow_flex"]] = math.radians(elbow)
        joint_pos[0, name_to_id["wrist_flex"]] = math.radians(wrist)
        joint_pos[0, name_to_id["wrist_roll"]] = 0.0
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)

        body_pose = robot.data.body_pose_w[:, gripper_body_id]
        contact = body_pose[:, :3] + quat_apply(body_pose[:, 3:7], offset.expand(1, -1))
        error = float(torch.linalg.vector_norm(contact[0] - target).item())
        results.append((error, (pan, shoulder, elbow, wrist), contact[0].tolist()))

    results.sort(key=lambda item: item[0])
    print(f"Target: {args_cli.target}")
    for error, joints, contact in results[: args_cli.top_k]:
        print(
            f"error={error:.5f} m joints_deg=(pan={joints[0]}, shoulder={joints[1]}, "
            f"elbow={joints[2]}, wrist={joints[3]}) contact={[round(value, 4) for value in contact]}"
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
