#!/usr/bin/env python
"""Automatically sample safe SO101 configurations indexed by gripper direction."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import placo

from controller import ARM_JOINTS, ControlConfig, load_joint_limits
from frame_adapter import gripper_tip_in_robot
from obstacles import ObstacleEnvironment


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_URDF = REPO_ROOT / "SO101" / "so101_new_calib.urdf"
DEFAULT_COLLISIONS = REPO_ROOT / "SO101" / "collisions.json"
DEFAULT_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
DEFAULT_OUTPUT = THIS_DIR / "direction_atlas.json"


def fibonacci_sphere(count: int) -> np.ndarray:
    """Approximately equal-area unit directions, including neither duplicated pole."""
    indices = np.arange(count, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(indices * golden_angle), radius * np.sin(indices * golden_angle), z))


@dataclass
class AtlasConfig:
    samples: int = 20_000
    direction_bins: int = 256
    candidates_per_bin: int = 4
    seed: int = 20_260_717
    min_tip_height_m: float = 0.055
    min_moving_frame_height_m: float = 0.012
    branch_separation: float = 0.12


@dataclass
class AtlasCandidate:
    joints_deg: list[float]
    direction: list[float]
    tip_position_m: list[float]
    joint_margin: float
    score: float


class SO101StateValidator:
    """Joint-limit, floor-envelope, and non-adjacent self-collision validator.

    ``obstacles`` optionally adds forbidden-zone checks: the moving links are
    approximated as a sphere chain (see ``obstacles.py``) and every sample
    must keep the configured margin from every registered zone.
    """

    MOVING_FRAMES = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_frame_link")

    def __init__(
        self,
        urdf_path: Path,
        collision_pairs_path: Path,
        joint_limits: dict[str, tuple[float, float]],
        min_tip_height_m: float = 0.055,
        min_moving_frame_height_m: float = 0.012,
        obstacles: ObstacleEnvironment | None = None,
    ) -> None:
        self.robot = placo.RobotWrapper(str(urdf_path))
        self.robot.load_collision_pairs(str(collision_pairs_path))
        self.joint_limits = joint_limits
        self.low = np.array([joint_limits[name][0] for name in ARM_JOINTS])
        self.high = np.array([joint_limits[name][1] for name in ARM_JOINTS])
        self.span = self.high - self.low
        self.min_tip_height_m = min_tip_height_m
        self.min_moving_frame_height_m = min_moving_frame_height_m
        self.obstacles = obstacles
        self._collision_objects = self.robot.collision_model.geometryObjects
        self.robot.set_joint("gripper", 0.0)

    def set_configuration(self, joints_deg: np.ndarray) -> None:
        for name, value in zip(ARM_JOINTS, np.deg2rad(joints_deg), strict=True):
            self.robot.set_joint(name, float(value))
        self.robot.update_kinematics()

    def evaluate(self, joints_deg: np.ndarray) -> tuple[AtlasCandidate | None, str]:
        joints_deg = np.asarray(joints_deg, dtype=float)
        if joints_deg.shape != (5,) or not np.all(np.isfinite(joints_deg)):
            return None, "invalid_joints"
        if np.any(joints_deg < self.low) or np.any(joints_deg > self.high):
            return None, "joint_limits"
        self.set_configuration(joints_deg)
        if self.robot.self_collisions(True):
            return None, "self_collision"

        pose = self.robot.get_T_world_frame("gripper_frame_link")
        tip_position = pose[:3, 3]
        if tip_position[2] < self.min_tip_height_m:
            return None, "tip_below_floor_envelope"
        if any(
            self.robot.get_T_world_frame(frame)[:3, 3][2] < self.min_moving_frame_height_m
            for frame in self.MOVING_FRAMES
        ):
            return None, "link_below_floor_envelope"
        if self.obstacles is not None:
            distance = self.obstacles.distance_m(self.robot)
            if distance < self.obstacles.margin_m:
                return None, "obstacle_collision"

        direction = gripper_tip_in_robot(pose[:3, :3])
        normalized_margin = np.minimum(joints_deg - self.low, self.high - joints_deg) / (self.span / 2.0)
        joint_margin = float(np.clip(np.min(normalized_margin), 0.0, 1.0))
        return (
            AtlasCandidate(
                joints_deg=joints_deg.tolist(),
                direction=direction.tolist(),
                tip_position_m=tip_position.tolist(),
                joint_margin=joint_margin,
                score=0.0,
            ),
            "valid",
        )

    def collision_names(self) -> list[tuple[str, str]]:
        return [
            (self._collision_objects[item.objA].name, self._collision_objects[item.objB].name)
            for item in self.robot.self_collisions(False)
        ]


class DirectionAtlasBuilder:
    def __init__(self, validator: SO101StateValidator, config: AtlasConfig) -> None:
        self.validator = validator
        self.config = config
        self.centers = fibonacci_sphere(config.direction_bins)
        self.bins: list[list[AtlasCandidate]] = [[] for _ in range(config.direction_bins)]
        self.stats: dict[str, int] = {}

    def _insert(self, bin_index: int, candidate: AtlasCandidate) -> None:
        center = self.centers[bin_index]
        angular_error = math.acos(float(np.clip(np.dot(center, candidate.direction), -1.0, 1.0)))
        candidate.score = angular_error + 0.04 * (1.0 - candidate.joint_margin)
        entries = self.bins[bin_index]
        q = np.asarray(candidate.joints_deg)

        closest_index = None
        closest_distance = math.inf
        for index, existing in enumerate(entries):
            distance = float(
                np.linalg.norm((q - np.asarray(existing.joints_deg)) / self.validator.span) / math.sqrt(len(ARM_JOINTS))
            )
            if distance < closest_distance:
                closest_index, closest_distance = index, distance
        if closest_index is not None and closest_distance < self.config.branch_separation:
            if candidate.score < entries[closest_index].score:
                entries[closest_index] = candidate
            return
        if len(entries) < self.config.candidates_per_bin:
            entries.append(candidate)
            return
        worst = max(range(len(entries)), key=lambda index: entries[index].score)
        if candidate.score < entries[worst].score:
            entries[worst] = candidate

    def build(self) -> dict[str, Any]:
        started = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)
        for joints in rng.uniform(self.validator.low, self.validator.high, size=(self.config.samples, 5)):
            candidate, reason = self.validator.evaluate(joints)
            self.stats[reason] = self.stats.get(reason, 0) + 1
            if candidate is None:
                continue
            bin_index = int(np.argmax(self.centers @ np.asarray(candidate.direction)))
            self._insert(bin_index, candidate)

        occupied = sum(bool(entries) for entries in self.bins)
        retained = sum(len(entries) for entries in self.bins)
        return {
            "version": 1,
            "joint_names": ARM_JOINTS,
            "config": asdict(self.config),
            "stats": {
                **self.stats,
                "occupied_bins": occupied,
                "total_bins": self.config.direction_bins,
                "retained_candidates": retained,
                "elapsed_s": round(time.perf_counter() - started, 3),
            },
            "bins": [
                {"center": center.tolist(), "candidates": [asdict(candidate) for candidate in entries]}
                for center, entries in zip(self.centers, self.bins, strict=True)
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--collisions", type=Path, default=DEFAULT_COLLISIONS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AtlasConfig(
        samples=args.samples,
        direction_bins=args.bins,
        candidates_per_bin=args.per_bin,
        seed=args.seed,
    )
    limits = load_joint_limits(args.urdf, args.calibration)
    validator = SO101StateValidator(
        args.urdf,
        args.collisions,
        limits,
        min_tip_height_m=config.min_tip_height_m,
        min_moving_frame_height_m=config.min_moving_frame_height_m,
    )
    atlas = DirectionAtlasBuilder(validator, config).build()
    args.output.write_text(json.dumps(atlas, indent=2) + "\n")
    print(json.dumps(atlas["stats"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
