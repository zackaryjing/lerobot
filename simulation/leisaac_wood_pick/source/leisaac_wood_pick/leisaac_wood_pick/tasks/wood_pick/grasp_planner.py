"""Geometry helpers for scripted SO-101 wooden-stick demonstrations.

This module deliberately depends only on PyTorch.  It can therefore be unit
tested without launching Isaac Sim, while the state machine can use the same
batched tensors on the simulator device.

Quaternion convention is ``(w, x, y, z)`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def normalize(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize vectors along their final dimension."""
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def quat_normalize(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize a batch of WXYZ quaternions."""
    return normalize(quaternion)


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product for WXYZ quaternions."""
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by WXYZ quaternions, with broadcastable leading axes."""
    q_vector, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    q_scalar = torch.broadcast_to(quaternion[..., :1], q_vector.shape[:-1] + (1,))
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + q_scalar * twice_cross + torch.linalg.cross(q_vector, twice_cross, dim=-1)


def yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """Create WXYZ quaternions for world-Z rotations."""
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=-1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert proper 3x3 rotation matrices to normalized WXYZ quaternions."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    one = torch.ones_like(m00)
    qw = 0.5 * torch.sqrt(torch.clamp(one + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.copysign(torch.sqrt(torch.clamp(one + m00 - m11 - m22, min=0.0)), m21 - m12)
    qy = 0.5 * torch.copysign(torch.sqrt(torch.clamp(one - m00 + m11 - m22, min=0.0)), m02 - m20)
    qz = 0.5 * torch.copysign(torch.sqrt(torch.clamp(one - m00 - m11 + m22, min=0.0)), m10 - m01)
    return quat_normalize(torch.stack((qw, qx, qy, qz), dim=-1))


def interpolate_quaternion(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Shortest-arc normalized interpolation, sufficient at 30 Hz waypoints."""
    dot = (start * end).sum(dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    return quat_normalize(start + alpha * (end - start))


def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    """Map linear progress to a C2-continuous minimum-jerk blend."""
    progress = progress.clamp(0.0, 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


@dataclass(frozen=True)
class StickSamplingConfig:
    """Bounds used to reset the stick on top of the measured platform."""

    platform_center_xy: tuple[float, float] = (0.235, 0.362)
    platform_size_xy: tuple[float, float] = (0.270, 0.100)
    platform_top_z: float = 0.053
    stick_size: tuple[float, float, float] = (0.0115, 0.0115, 0.080)
    edge_margin: float = 0.004
    max_object_radius: float = 0.38
    max_resampling_rounds: int = 1024


@dataclass(frozen=True)
class GraspSamplingConfig:
    """Task-specific grasp geometry and conservative workspace filters."""

    grasp_station_offset: float = 0.030
    max_tilt_rad: float = 0.35  # 20 degrees, along the stick axis.
    # The measured platform is near the SO-101's radial limit.  A 7.5 cm
    # vertical hover made the gripper-link target ~41.8 cm from the base and
    # unreachable even though the grasp itself is reachable.
    pregrasp_clearance: float = 0.035
    lift_clearance: float = 0.140
    # Approximate fingertip midpoint relative to the ``gripper`` link.  The
    # scene's jaw detector confirms that its dominant component is local -Z;
    # forward reach on the measured platform comes from tilting the complete
    # gripper, not from a large local-Y offset.
    gripper_to_contact_tool: tuple[float, float, float] = (0.030, 0.010, -0.100)
    min_ee_z: float = 0.085
    max_ee_radius: float = 0.52
    number_of_candidates: int = 16


@dataclass
class GraspCandidates:
    """Batched candidate poses, shaped ``(num_envs, num_candidates, ...)``."""

    grasp_point_w: torch.Tensor
    grasp_ee_pos_w: torch.Tensor
    pregrasp_ee_pos_w: torch.Tensor
    ee_quat_w: torch.Tensor
    feasible: torch.Tensor
    score: torch.Tensor
    station_sign: torch.Tensor

    def select(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the lowest-score feasible candidate for every environment."""
        penalized = torch.where(self.feasible, self.score, torch.full_like(self.score, torch.inf))
        candidate_ids = penalized.argmin(dim=1)
        has_candidate = self.feasible.any(dim=1)
        batch_ids = torch.arange(candidate_ids.shape[0], device=candidate_ids.device)
        return (
            self.grasp_point_w[batch_ids, candidate_ids],
            self.grasp_ee_pos_w[batch_ids, candidate_ids],
            self.pregrasp_ee_pos_w[batch_ids, candidate_ids],
            self.ee_quat_w[batch_ids, candidate_ids],
            has_candidate,
        )


def sample_stick_root_poses(
    count: int,
    *,
    device: torch.device | str,
    config: StickSamplingConfig = StickSamplingConfig(),
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample centred stick poses whose oriented footprint stays on the platform.

    The STL long axis is local +Z.  The base quaternion rotates it onto world
    +X, after which a uniformly sampled world yaw is applied.  Samples outside
    a coarse radial workspace are rejected; exact reachability is checked by
    pose tracking in the scripted controller.
    """
    dtype = torch.float32
    device = torch.device(device)
    positions = torch.empty((count, 3), device=device, dtype=dtype)
    quaternions = torch.empty((count, 4), device=device, dtype=dtype)
    valid = torch.zeros(count, dtype=torch.bool, device=device)
    base_quaternion = torch.tensor((2**-0.5, 0.0, 2**-0.5, 0.0), device=device, dtype=dtype)
    half_long = 0.5 * config.stick_size[2]
    half_cross = 0.5 * config.stick_size[0]
    platform_half_x = 0.5 * config.platform_size_xy[0]
    platform_half_y = 0.5 * config.platform_size_xy[1]

    for _ in range(config.max_resampling_rounds):
        pending = torch.where(~valid)[0]
        if pending.numel() == 0:
            break
        yaw = (2.0 * torch.rand(pending.numel(), device=device, generator=generator) - 1.0) * torch.pi
        projected_half_x = half_long * yaw.cos().abs() + half_cross * yaw.sin().abs()
        projected_half_y = half_long * yaw.sin().abs() + half_cross * yaw.cos().abs()
        available_x = platform_half_x - config.edge_margin - projected_half_x
        available_y = platform_half_y - config.edge_margin - projected_half_y
        x = config.platform_center_xy[0] + (2.0 * torch.rand(pending.numel(), device=device, generator=generator) - 1.0) * available_x
        y = config.platform_center_xy[1] + (2.0 * torch.rand(pending.numel(), device=device, generator=generator) - 1.0) * available_y
        within_workspace = torch.sqrt(x.square() + y.square()) <= config.max_object_radius

        accepted = pending[within_workspace]
        accepted_yaw = yaw[within_workspace]
        positions[accepted, 0] = x[within_workspace]
        positions[accepted, 1] = y[within_workspace]
        positions[accepted, 2] = config.platform_top_z + half_cross
        quaternions[accepted] = quat_multiply(yaw_quaternion(accepted_yaw), base_quaternion.expand(accepted.numel(), -1))
        valid[accepted] = True

    if not bool(valid.all()):
        raise RuntimeError(
            "Unable to sample a reachable stick pose. Check platform bounds and max_object_radius."
        )
    return positions, quat_normalize(quaternions)


def sample_grasp_candidates(
    stick_pos_w: torch.Tensor,
    stick_quat_w: torch.Tensor,
    *,
    config: GraspSamplingConfig = GraspSamplingConfig(),
    generator: torch.Generator | None = None,
) -> GraspCandidates:
    """Generate collision-conscious grasp poses at the two stable ±3 cm stations.

    Local tool +X is the finger closing axis, local +Z points from the object
    toward the pregrasp pose, and local -Z is the approach direction.  Tilting
    is restricted to the vertical/stick-axis plane.  Consequently the closing
    axis stays horizontal and the two fingers cannot straddle the platform at
    different heights.
    """
    if stick_pos_w.ndim != 2 or stick_pos_w.shape[-1] != 3:
        raise ValueError("stick_pos_w must have shape (N, 3)")
    if stick_quat_w.shape != (stick_pos_w.shape[0], 4):
        raise ValueError("stick_quat_w must have shape (N, 4)")

    num_envs = stick_pos_w.shape[0]
    candidates = config.number_of_candidates
    device, dtype = stick_pos_w.device, stick_pos_w.dtype
    local_long = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
    long_axis = normalize(quat_apply(stick_quat_w, local_long.expand_as(stick_pos_w)))
    long_axis = long_axis[:, None, :].expand(-1, candidates, -1)

    candidate_ids = torch.arange(candidates, device=device)
    station_sign = torch.where(candidate_ids % 2 == 0, 1.0, -1.0).to(dtype)
    station_sign = station_sign[None, :].expand(num_envs, -1)
    # Both asymmetric gripper orientations are represented for every station.
    gripper_flip = torch.where((candidate_ids // 2) % 2 == 0, 1.0, -1.0).to(dtype)
    gripper_flip = gripper_flip[None, :].expand(num_envs, -1)

    tilt = (2.0 * torch.rand((num_envs, candidates), device=device, dtype=dtype, generator=generator) - 1.0) * config.max_tilt_rad
    # Always include vertical candidates for both stable stations and both
    # asymmetric wrist flips before introducing tilted approaches.
    tilt[:, :4] = 0.0
    world_up = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
    tool_z = normalize(world_up + torch.tan(tilt)[..., None] * long_axis)
    tool_x = normalize(torch.linalg.cross(long_axis, tool_z, dim=-1)) * gripper_flip[..., None]
    tool_y = normalize(torch.linalg.cross(tool_z, tool_x, dim=-1))
    rotation = torch.stack((tool_x, tool_y, tool_z), dim=-1)
    ee_quat_w = matrix_to_quaternion(rotation)

    grasp_point_w = stick_pos_w[:, None, :] + station_sign[..., None] * config.grasp_station_offset * long_axis
    contact_offset = torch.tensor(config.gripper_to_contact_tool, device=device, dtype=dtype)
    contact_offset_w = torch.einsum("nkij,j->nki", rotation, contact_offset)
    grasp_ee_pos_w = grasp_point_w - contact_offset_w
    pregrasp_ee_pos_w = grasp_ee_pos_w + config.pregrasp_clearance * tool_z

    # Reachability is evaluated for the virtual fingertip-midpoint frame used
    # by the action term, rather than for the gripper link origin.
    pregrasp_point_w = grasp_point_w + config.pregrasp_clearance * tool_z
    grasp_radius = torch.linalg.vector_norm(grasp_point_w[..., :2], dim=-1)
    pregrasp_radius = torch.linalg.vector_norm(pregrasp_point_w[..., :2], dim=-1)
    feasible = (
        (grasp_point_w[..., 2] >= 0.053)
        & (pregrasp_point_w[..., 2] >= config.min_ee_z)
        & (grasp_radius <= config.max_ee_radius)
        & (pregrasp_radius <= config.max_ee_radius)
    )
    # Prefer short reach and small tilt while still retaining both grasp stations.
    score = pregrasp_radius + 0.035 * tilt.abs()
    return GraspCandidates(
        grasp_point_w=grasp_point_w,
        grasp_ee_pos_w=grasp_ee_pos_w,
        pregrasp_ee_pos_w=pregrasp_ee_pos_w,
        ee_quat_w=ee_quat_w,
        feasible=feasible,
        score=score,
        station_sign=station_sign,
    )


def oriented_box_corners(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    size: tuple[float, float, float],
) -> torch.Tensor:
    """Return eight world-space corners for batched oriented boxes."""
    signs = torch.tensor(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        device=position.device,
        dtype=position.dtype,
    )
    half_size = 0.5 * torch.tensor(size, device=position.device, dtype=position.dtype)
    local_corners = signs * half_size
    return position[:, None, :] + quat_apply(quaternion[:, None, :], local_corners[None, :, :])


def stick_inside_box(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    *,
    stick_size: tuple[float, float, float] = (0.0115, 0.0115, 0.080),
    box_lower_left: tuple[float, float, float] = (-0.107, 0.380, 0.0),
    box_size: tuple[float, float, float] = (0.150, 0.100, 0.085),
    wall_thickness: float = 0.003,
    xy_margin: float = 0.001,
    floor_tolerance: float = 0.003,
    rim_tolerance: float = 0.005,
) -> torch.Tensor:
    """Check that the complete stick, rather than only its centre, is in the box."""
    corners = oriented_box_corners(position, quaternion, stick_size)
    lower = torch.tensor(box_lower_left, device=position.device, dtype=position.dtype)
    size = torch.tensor(box_size, device=position.device, dtype=position.dtype)
    interior_xy_min = lower[:2] + wall_thickness + xy_margin
    interior_xy_max = lower[:2] + size[:2] - wall_thickness - xy_margin
    xy_inside = ((corners[..., :2] >= interior_xy_min) & (corners[..., :2] <= interior_xy_max)).all(dim=-1).all(dim=-1)
    floor_top = lower[2] + wall_thickness
    z_inside = (corners[..., 2].amin(dim=-1) >= floor_top - floor_tolerance) & (
        corners[..., 2].amax(dim=-1) <= lower[2] + size[2] + rim_tolerance
    )
    return xy_inside & z_inside
