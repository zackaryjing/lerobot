"""Semantic frame adapter between input devices and the SO101 gripper.

Visual forward directions (independent of IK):
- Android phone: +Y toward top edge (screen up).
- MPU6500 chip: +Z out of the die / silkscreen (green arrow in the twin).
SO101 gripper tip direction is gripper_frame_link local +Z.

Frames:
- Phone path: the device pose is expressed in the calibrated XR reference
  space and mapped through the midpoint gripper rotation.
- MPU path: the bridge already aligns the twin world frame (die forward ->
  twin +X). That twin world frame IS the robot base frame, so the die normal
  shown in the twin is used verbatim as the gripper target direction; the
  server-side calibration only gates synchronization.

This file sits ABOVE the IMU boundary: it consumes only the orientation
stream contract (absolute device quat, yaw pre-aligned, world frame == robot
base frame). Swapping the IMU (e.g. a 9-axis chip) never reaches this file
as long as the bridge keeps producing that contract; the phone path is
similarly independent of the MPU internals.
"""

from __future__ import annotations

import numpy as np

# Bump when the mapping changes; the server prints it and exposes it to the
# twin so a stale server process is visible immediately.
VERSION = 3


PHONE_FORWARD_LOCAL = np.array([0.0, 1.0, 0.0])
MPU_FORWARD_LOCAL = np.array([0.0, 0.0, 1.0])
GRIPPER_TIP_LOCAL = np.array([0.0, 0.0, 1.0])

# phone +Y -> gripper +Z; phone +Z -> gripper -X
PHONE_TO_GRIPPER_AXES = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)

DEVICE_FORWARD_LOCAL = {
    "phone": PHONE_FORWARD_LOCAL,
    "mpu6500": MPU_FORWARD_LOCAL,
}


def _device_key(device: str | None) -> str:
    return "mpu6500" if str(device or "phone").strip().lower() == "mpu6500" else "phone"


def make_xr_to_robot_rotation(
    midpoint_gripper_rotation: np.ndarray,
    calibrated_phone_rotation: np.ndarray,
    device: str | None = "phone",
) -> np.ndarray:
    """Map device reference-space vectors into robot-base coordinates."""
    if _device_key(device) == "mpu6500":
        return np.eye(3)
    return midpoint_gripper_rotation @ PHONE_TO_GRIPPER_AXES.T @ calibrated_phone_rotation.T


def phone_forward_in_robot(
    xr_to_robot_rotation: np.ndarray,
    phone_rotation: np.ndarray,
    device: str | None = "phone",
) -> np.ndarray:
    forward_local = DEVICE_FORWARD_LOCAL[_device_key(device)]
    direction = xr_to_robot_rotation @ phone_rotation @ forward_local
    return direction / np.linalg.norm(direction)


def gripper_tip_in_robot(gripper_rotation: np.ndarray) -> np.ndarray:
    direction = gripper_rotation @ GRIPPER_TIP_LOCAL
    return direction / np.linalg.norm(direction)
