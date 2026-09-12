"""Forbidden collision zones for the SO101 arm (future wearable mounting).

The arm's final mounting is on a person's back with the base plate parallel to
the back, so the current robot-base +X (forward) will point up. All forbidden
zones are therefore expressed in ROBOT-BASE coordinates, and an
``ObstacleEnvironment`` can be re-posed with ``apply_transform`` when the
mounting changes; the model definitions themselves stay mounting-agnostic.

The base plate is the attachment surface and is deliberately NOT part of the
arm proxy (it is expected to touch the wearer). The moving links from the
shoulder to the gripper tip are approximated as chains of spheres along the
line between consecutive joint origins; distance is computed against every
registered zone with hppfcl.

Zones:
- ``add_box``: axis-aligned-or-rotated box, the fast primitive. The test
  pseudo-human is built from these.
- ``add_mesh``: triangle mesh loaded from an OBJ/STL file (hppfcl BVH model).
  Use this for real 3D body models later; distance queries against meshes are
  slower than against boxes, and a wearable mount would carry at most one or
  two of them.

Threading: environments are shared read-only across the validator instances
(one per placo stack). Shape building is lazy and guarded by a lock; after
configuration the environment is only read.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import hppfcl
import numpy as np

# (parent-frame link, child-frame link, proxy radius) for each moving segment.
# The endpoint of one segment is the start of the next, so the sphere chain
# covers the arm continuously from the shoulder_pan axis to the gripper tip.
ARM_SEGMENTS: tuple[tuple[str, str, float], ...] = (
    ("shoulder_link", "upper_arm_link", 0.055),  # shoulder block
    ("upper_arm_link", "lower_arm_link", 0.045),  # upper arm
    ("lower_arm_link", "wrist_link", 0.038),  # forearm
    ("wrist_link", "gripper_frame_link", 0.048),  # hand + gripper
)
SPHERE_SPACING_M = 0.03


@dataclass
class BoxZone:
    name: str
    center: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray  # 3x3, columns are the box frame axes


@dataclass
class MeshZone:
    name: str
    path: Path
    transform: np.ndarray  # 4x4 homogeneous


class ObstacleEnvironment:
    """Forbidden zones for the arm, in robot-base coordinates."""

    def __init__(self, margin_m: float = 0.02) -> None:
        if margin_m < 0.0:
            raise ValueError("margin must be non-negative")
        self.margin_m = margin_m
        self._boxes: dict[str, BoxZone] = {}
        self._meshes: dict[str, MeshZone] = {}
        self._shapes: dict[str, tuple[hppfcl.ShapeBase, hppfcl.Transform3f]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- registry

    def add_box(
        self,
        name: str,
        center: np.ndarray,
        half_extents: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> "ObstacleEnvironment":
        center = np.asarray(center, dtype=float)
        half_extents = np.asarray(half_extents, dtype=float)
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if center.shape != (3,) or half_extents.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("center/half_extents must be 3-vectors and rotation a 3x3")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_extents)) or np.any(half_extents <= 0.0):
            raise ValueError("box requires finite center and positive half extents")
        with self._lock:
            self._boxes[name] = BoxZone(name, center, half_extents, rotation)
            self._shapes.pop(name, None)
        return self

    def add_mesh(self, name: str, path: Path | str, transform: np.ndarray | None = None) -> "ObstacleEnvironment":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"mesh file not found: {path}")
        transform = np.eye(4) if transform is None else np.asarray(transform, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform must be a finite 4x4")
        with self._lock:
            self._meshes[name] = MeshZone(name, path, transform)
            self._shapes.pop(name, None)
        return self

    def remove(self, name: str) -> None:
        with self._lock:
            self._boxes.pop(name, None)
            self._meshes.pop(name, None)
            self._shapes.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._boxes.clear()
            self._meshes.clear()
            self._shapes.clear()

    def names(self) -> list[str]:
        with self._lock:
            return [*self._boxes, *self._meshes]

    def apply_transform(self, rotation: np.ndarray, translation: np.ndarray) -> "ObstacleEnvironment":
        """Return a new environment with all zones re-posed by (R, t).

        Used when the mounting changes (e.g. base parallel to the wearer's
        back): keep the human model fixed in its own frame and re-pose the
        whole environment into the new robot-base frame.
        """
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.all(np.isfinite(rotation)):
            raise ValueError("apply_transform requires a finite 3x3 rotation and a 3-vector")
        posed = ObstacleEnvironment(margin_m=self.margin_m)
        with self._lock:
            for zone in self._boxes.values():
                posed.add_box(
                    zone.name,
                    rotation @ zone.center + translation,
                    zone.half_extents,
                    rotation @ zone.rotation,
                )
            for zone in self._meshes.values():
                homogeneous = np.eye(4)
                homogeneous[:3, :3] = rotation
                homogeneous[:3, 3] = translation
                posed.add_mesh(zone.name, zone.path, homogeneous @ zone.transform)
        return posed

    # -------------------------------------------------------------- shapes

    def _shape(self, name: str) -> tuple[hppfcl.ShapeBase, hppfcl.Transform3f]:
        with self._lock:
            cached = self._shapes.get(name)
            if cached is not None:
                return cached
            if name in self._boxes:
                zone = self._boxes[name]
                shape: hppfcl.ShapeBase = hppfcl.Box(2.0 * zone.half_extents)
                transform = hppfcl.Transform3f(zone.rotation, zone.center)
            else:
                zone = self._meshes[name]
                shape = hppfcl.MeshLoader().load(str(zone.path))
                transform = hppfcl.Transform3f(
                    zone.transform[:3, :3], zone.transform[:3, 3]
                )
            self._shapes[name] = (shape, transform)
            return self._shapes[name]

    # ------------------------------------------------------------ distance

    @staticmethod
    def arm_proxy_spheres(robot) -> list[tuple[np.ndarray, float]]:
        """Sphere chain covering the moving links for the robot's current pose."""
        spheres: list[tuple[np.ndarray, float]] = []
        for parent, child, radius in ARM_SEGMENTS:
            first = robot.get_T_world_frame(parent)[:3, 3]
            second = robot.get_T_world_frame(child)[:3, 3]
            segment = float(np.linalg.norm(second - first))
            pieces = max(1, int(math.ceil(segment / SPHERE_SPACING_M)))
            for piece in range(pieces + 1):
                spheres.append((first + (second - first) * (piece / pieces), radius))
        return spheres

    def distance_m(self, robot) -> float:
        """Minimal surface distance between the arm proxy and any zone (inf if none)."""
        with self._lock:
            if not self._boxes and not self._meshes:
                return math.inf
            spheres = self.arm_proxy_spheres(robot)
            request = hppfcl.DistanceRequest()
            result = hppfcl.DistanceResult()
            closest = math.inf
            for center, radius in spheres:
                sphere = hppfcl.Sphere(radius)
                sphere_transform = hppfcl.Transform3f(np.eye(3), center)
                for name in [*self._boxes, *self._meshes]:
                    shape, transform = self._shape(name)
                    result.clear()
                    hppfcl.distance(sphere, sphere_transform, shape, transform, request, result)
                    if result.min_distance < closest:
                        closest = result.min_distance
            return closest


def make_pseudo_human() -> ObstacleEnvironment:
    """A box-composed stand-in for the wearer, behind the robot base.

    Placed for the current table mounting (forward = +X): torso behind the
    base, head and shoulders shifted to +Y so a clear reach exists on the -Y
    side. With the future back mounting, re-pose with ``apply_transform``.
    """
    environment = ObstacleEnvironment(margin_m=0.02)
    environment.add_box("torso", center=[-0.28, 0.00, 0.16], half_extents=[0.10, 0.17, 0.30])
    environment.add_box("shoulder_left", center=[-0.26, 0.20, 0.50], half_extents=[0.09, 0.09, 0.09])
    environment.add_box("shoulder_right", center=[-0.26, -0.20, 0.50], half_extents=[0.09, 0.09, 0.09])
    environment.add_box("head", center=[-0.24, 0.22, 0.70], half_extents=[0.09, 0.10, 0.12])
    return environment
