"""Semantic frame adapter between Android WebXR and the SO101 gripper.

This module intentionally defines *visual forward directions*, independently of
IK. Android phone forward means the direction from the phone center toward its
top edge while the screen is facing up. SO101 gripper forward means the direction
from the closed gripper root toward its tips.
"""

from __future__ import annotations

import numpy as np


# Android/WebXR device local axes: +X right, +Y toward the top edge, +Z out of
# the screen. gripper_frame_link's visual tip direction is its local +Z.
PHONE_FORWARD_LOCAL = np.array([0.0, 1.0, 0.0])
GRIPPER_TIP_LOCAL = np.array([0.0, 0.0, 1.0])

# A proper rotation relating the remaining semantic axes. Its important part is
# phone +Y -> gripper +Z. It also maps phone +Z -> gripper -X, so a screen facing
# up corresponds to the gripper's visual top facing up in the midpoint pose.
PHONE_TO_GRIPPER_AXES = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)


def make_xr_to_robot_rotation(
    midpoint_gripper_rotation: np.ndarray, calibrated_phone_rotation: np.ndarray
) -> np.ndarray:
    """Map XR reference-space vectors into robot-base coordinates.

    At calibration time a flat phone with its top edge pointing forward maps to
    the gripper direction at the all-motors-midpoint pose.
    """
    return midpoint_gripper_rotation @ PHONE_TO_GRIPPER_AXES.T @ calibrated_phone_rotation.T


def phone_forward_in_robot(
    xr_to_robot_rotation: np.ndarray, phone_rotation: np.ndarray
) -> np.ndarray:
    direction = xr_to_robot_rotation @ phone_rotation @ PHONE_FORWARD_LOCAL
    return direction / np.linalg.norm(direction)


def gripper_tip_in_robot(gripper_rotation: np.ndarray) -> np.ndarray:
    direction = gripper_rotation @ GRIPPER_TIP_LOCAL
    return direction / np.linalg.norm(direction)
