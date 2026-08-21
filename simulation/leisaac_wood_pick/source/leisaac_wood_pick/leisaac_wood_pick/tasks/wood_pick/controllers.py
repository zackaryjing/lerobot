"""Task-specific controllers missing from the pinned Isaac Lab 2.3 API."""

from __future__ import annotations

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error


class WeightedDifferentialIKController(DifferentialIKController):
    """DLS pose IK with position prioritized over orientation.

    SO-101 has five arm joints but an absolute pose command has six task rows.
    Isaac Lab 2.3 gives all rows equal weight, so an unreachable orientation can
    leave the gripper centimetres away from the object.  Weighting both the
    rotation residual and the rotational Jacobian rows implements a soft
    orientation objective while retaining LeIsaac's existing action term.
    """

    cfg: "WeightedDifferentialIKControllerCfg"

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        position_error, axis_angle_error = compute_pose_error(
            ee_pos,
            ee_quat,
            self.ee_pos_des,
            self.ee_quat_des,
            rot_error_type="axis_angle",
        )
        orientation_weight = self.cfg.orientation_weight
        if orientation_weight > 0.0:
            weighted_error = torch.cat((position_error, orientation_weight * axis_angle_error), dim=-1)
            weighted_jacobian = jacobian.clone()
            weighted_jacobian[:, 3:6, :] *= orientation_weight
            delta_joint_pos = self._compute_delta_joint_pos(weighted_error, weighted_jacobian)
        else:
            # Strict position priority with a null-space posture bias. The
            # stock rest pose starts at shoulder/elbow limits and is close to a
            # translational singularity; the bias unfolds the arm without
            # competing with the fingertip-midpoint position task.
            jacobian_pos = jacobian[:, :3, :]
            damping = self.cfg.ik_params["lambda_val"]
            jacobian_t = jacobian_pos.transpose(1, 2)
            identity_task = torch.eye(3, device=self._device).expand(self.num_envs, -1, -1)
            jacobian_pinv = jacobian_t @ torch.linalg.inv(
                jacobian_pos @ jacobian_t + damping**2 * identity_task
            )
            delta_task = (jacobian_pinv @ position_error.unsqueeze(-1)).squeeze(-1)
            identity_joint = torch.eye(joint_pos.shape[1], device=self._device).expand(self.num_envs, -1, -1)
            null_projector = identity_joint - jacobian_pinv @ jacobian_pos
            nominal = torch.tensor(
                self.cfg.nominal_joint_pos,
                device=self._device,
                dtype=joint_pos.dtype,
            ).expand_as(joint_pos)
            delta_null = (
                null_projector @ (nominal - joint_pos).unsqueeze(-1)
            ).squeeze(-1)
            delta_joint_pos = delta_task + self.cfg.nullspace_gain * delta_null

        delta_joint_pos = delta_joint_pos.clamp(
            min=-self.cfg.maximum_joint_delta,
            max=self.cfg.maximum_joint_delta,
        )
        return joint_pos + delta_joint_pos


class WeightedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Isaac Lab 2.3 action term that actually instantiates the configured controller.

    The pinned base action hard-codes ``DifferentialIKController`` and ignores
    ``cfg.controller.class_type``.  Replacing only that instance keeps every
    other action-term behavior identical to upstream.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._ik_controller = WeightedDifferentialIKController(
            cfg=self.cfg.controller,
            num_envs=self.num_envs,
            device=self.device,
        )


@configclass
class WeightedDifferentialIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for position-prioritized DLS pose IK."""

    class_type: type = WeightedDifferentialIKController
    orientation_weight: float = 0.20
    nominal_joint_pos: tuple[float, float, float, float, float] = (
        0.0,
        0.5235987756,
        0.5235987756,
        -1.0471975512,
        0.0,
    )
    nullspace_gain: float = 0.08
    maximum_joint_delta: float = 0.15

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.orientation_weight <= 1.0:
            raise ValueError("orientation_weight must be in [0, 1]")
        if self.nullspace_gain < 0.0:
            raise ValueError("nullspace_gain must be non-negative")
        if self.maximum_joint_delta <= 0.0:
            raise ValueError("maximum_joint_delta must be positive")
