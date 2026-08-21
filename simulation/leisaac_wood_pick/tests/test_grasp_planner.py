"""CPU-only tests for wood-pick geometry; Isaac Sim is intentionally not imported."""

import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "source" / "leisaac_wood_pick"
sys.path.insert(0, str(PACKAGE_ROOT))

from leisaac_wood_pick.tasks.wood_pick.grasp_planner import (  # noqa: E402
    GraspSamplingConfig,
    StickSamplingConfig,
    oriented_box_corners,
    quat_apply,
    sample_grasp_candidates,
    sample_stick_root_poses,
    stick_inside_box,
)


def test_random_stick_footprints_stay_on_platform() -> None:
    generator = torch.Generator().manual_seed(7)
    cfg = StickSamplingConfig()
    position, quaternion = sample_stick_root_poses(2048, device="cpu", config=cfg, generator=generator)
    corners = oriented_box_corners(position, quaternion, cfg.stick_size)
    half_x, half_y = 0.5 * cfg.platform_size_xy[0], 0.5 * cfg.platform_size_xy[1]
    assert torch.all(corners[..., 0] >= cfg.platform_center_xy[0] - half_x + cfg.edge_margin - 1.0e-6)
    assert torch.all(corners[..., 0] <= cfg.platform_center_xy[0] + half_x - cfg.edge_margin + 1.0e-6)
    assert torch.all(corners[..., 1] >= cfg.platform_center_xy[1] - half_y + cfg.edge_margin - 1.0e-6)
    assert torch.all(corners[..., 1] <= cfg.platform_center_xy[1] + half_y - cfg.edge_margin + 1.0e-6)
    assert torch.all(torch.linalg.vector_norm(position[:, :2], dim=-1) <= cfg.max_object_radius + 1.0e-6)


def test_grasp_candidates_cross_the_two_stable_stations() -> None:
    position = torch.tensor([[0.16, 0.34, 0.05875]])
    # STL local +Z has been rotated to world +X.
    quaternion = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
    cfg = GraspSamplingConfig(number_of_candidates=16)
    candidates = sample_grasp_candidates(
        position,
        quaternion,
        config=cfg,
        generator=torch.Generator().manual_seed(3),
    )
    long_axis = torch.tensor([1.0, 0.0, 0.0])
    tool_x = quat_apply(candidates.ee_quat_w, torch.tensor([1.0, 0.0, 0.0]))
    tool_z = quat_apply(candidates.ee_quat_w, torch.tensor([0.0, 0.0, 1.0]))

    # Finger closing direction is transverse to the stick and horizontal.
    assert torch.allclose((tool_x * long_axis).sum(dim=-1), torch.zeros(1, 16), atol=1.0e-5)
    assert torch.allclose(tool_x[..., 2], torch.zeros(1, 16), atol=1.0e-5)
    # Pregrasp always backs away along tool +Z.
    delta = candidates.pregrasp_ee_pos_w - candidates.grasp_ee_pos_w
    assert torch.allclose(delta, cfg.pregrasp_clearance * tool_z, atol=1.0e-5)
    station_distance = torch.linalg.vector_norm(candidates.grasp_point_w - position[:, None, :], dim=-1)
    assert torch.allclose(station_distance, torch.full((1, 16), 0.03), atol=1.0e-5)
    assert bool(candidates.feasible.any())


def test_success_requires_all_stick_corners_inside_box() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # Long local Z lies vertically here, entirely under the 8.5 cm rim.
    safely_inside = torch.tensor([[-0.032, 0.430, 0.044]])
    assert bool(stick_inside_box(safely_inside, identity).item())

    # Its centre is inside, but the long axis crosses the narrow Y walls.
    horizontal_y = torch.tensor([[2**-0.5, -2**-0.5, 0.0, 0.0]])
    across_wall = torch.tensor([[-0.032, 0.472, 0.012]])
    assert not bool(stick_inside_box(across_wall, horizontal_y).item())

    # A centre over the box but still above the rim is not success.
    above_rim = torch.tensor([[-0.032, 0.430, 0.140]])
    assert not bool(stick_inside_box(above_rim, identity).item())
