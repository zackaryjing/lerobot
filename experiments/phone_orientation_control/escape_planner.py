"""Direction-preserving constrained reconfiguration ("escape") planner.

When the live differential-IK tracker saturates against calibration limits, two
directions can still be mutually reachable through a long detour that KEEPS the
gripper tip pointed at the frozen target direction. The set of such
configurations is a 3-dimensional manifold inside the 5-joint space (the
direction task is rank-2). This module plans paths ON that manifold:

- ``project`` pulls arbitrary configurations back onto the manifold by iterating
  the exact same bounded direction-IK step the tracker uses, so "on-manifold"
  means the same thing to the planner and the runtime.
- Goals are random configurations projected onto the manifold, scored by joint
  margin and task manipulability.
- A CBiRRT-Connect-style constrained RRT connects the start to the best goal;
  every edge is sampled and checked for calibration limits, self-collision, and
  the direction cone together.

Threading contract: instances own their ``SO101StateValidator`` exclusively and
``plan()`` must be called from exactly one thread (the controller's escape
worker). Constraint strictness (floor envelopes etc.) comes entirely from the
validator instance handed in - this module adds only the direction cone.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from controller import (
    ARM_JOINTS,
    direction_tip_step,
    normalize_vector,
    rotation_vector_from_matrix,
    vector_angle,
)
from direction_atlas import SO101StateValidator
from frame_adapter import gripper_tip_in_robot

logger = logging.getLogger("phone_orientation_control.escape_planner")


@dataclass
class EscapePlannerConfig:
    direction_tolerance_rad: float = math.radians(3.0)
    projection_max_iterations: int = 24
    projection_step_rad: float = math.radians(4.0)
    projection_damping: float = 0.08
    jacobian_epsilon_deg: float = 0.15
    posture_bias_weight: float = 1e-2
    goal_samples: int = 120
    goal_candidates: int = 6
    goal_separation_normalized: float = 0.10
    rrt_step_normalized: float = 0.06
    rrt_goal_bias: float = 0.15
    rrt_max_iterations: int = 900
    rrt_connect_attempts: int = 40
    edge_resolution_deg: float = 3.0
    shortcut_attempts: int = 60
    waypoint_delta_deg: float = 2.0
    planning_timeout_s: float = 10.0
    seed: int = 20260718


@dataclass
class EscapeGoal:
    joints_deg: list[float]
    direction_error_deg: float
    joint_margin: float
    sigma_min: float
    score: float


@dataclass
class EscapePlan:
    waypoints_deg: list[list[float]]
    target_direction: list[float]
    goal: EscapeGoal
    method: str
    planning_time_s: float
    tree_nodes: int
    projections: int

    @property
    def normalized_length(self) -> float:
        points = [np.asarray(w, dtype=float) for w in self.waypoints_deg]
        return sum(
            float(np.linalg.norm(second - first)) for first, second in zip(points, points[1:])
        )


class EscapePlanner:
    """Plans collision-free paths that hold the tip direction inside a cone."""

    def __init__(self, validator: SO101StateValidator, config: EscapePlannerConfig | None = None) -> None:
        self.validator = validator
        self.config = config or EscapePlannerConfig()
        self._low = np.asarray(validator.low, dtype=float)
        self._high = np.asarray(validator.high, dtype=float)
        self._span = np.maximum(self._high - self._low, 1e-9)
        self._rng = np.random.default_rng(self.config.seed)
        self._sigma_ref: float | None = None

    # ------------------------------------------------------------------ FK

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        """4x4 gripper_frame_link pose for five arm joints in degrees."""
        self.validator.set_configuration(joints_deg)
        return self.robot_T_world_frame()

    def robot_T_world_frame(self) -> np.ndarray:
        return self.validator.robot.get_T_world_frame("gripper_frame_link")

    def tip_direction(self, joints_deg: np.ndarray) -> np.ndarray:
        return gripper_tip_in_robot(self.fk(joints_deg)[:3, :3])

    def direction_error_deg(self, joints_deg: np.ndarray, target: np.ndarray) -> float:
        return math.degrees(vector_angle(self.tip_direction(joints_deg), target))

    def _within_cone(self, joints_deg: np.ndarray, target: np.ndarray) -> bool:
        return self.direction_error_deg(joints_deg, target) <= math.degrees(
            self.config.direction_tolerance_rad
        )

    def _normalized(self, joints_deg: np.ndarray) -> np.ndarray:
        return (np.asarray(joints_deg, dtype=float) - self._low) / self._span

    def _denormalized(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=float) * self._span + self._low

    @staticmethod
    def _normalized_distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))

    # ------------------------------------------------------------- manifold

    def project(
        self,
        joints_deg: np.ndarray,
        target: np.ndarray,
        posture_target: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Iterate bounded direction steps until inside the cone; None on failure.

        The decaying null-space posture bias pulls different random seeds toward
        different branches instead of letting every projection collapse into one
        basin. Intermediate configurations are deliberately not collision-checked;
        only edges between accepted manifold nodes are.
        """
        q = np.clip(np.asarray(joints_deg, dtype=float), self._low, self._high)
        max_iter = max(1, self.config.projection_max_iterations)
        for iteration in range(max_iter):
            if self._within_cone(q, target):
                return q
            bias = None
            weight = 0.0
            if posture_target is not None and self.config.posture_bias_weight > 0.0:
                bias = np.deg2rad(np.asarray(posture_target, dtype=float) - q)
                weight = self.config.posture_bias_weight * (1.0 - iteration / max_iter)
            solution, reason = direction_tip_step(
                self.fk,
                q,
                target,
                lower_delta_rad=np.deg2rad(self._low - q),
                upper_delta_rad=np.deg2rad(self._high - q),
                damping=self.config.projection_damping,
                jacobian_epsilon_deg=self.config.jacobian_epsilon_deg,
                max_step_rad=self.config.projection_step_rad,
                fractions=(1.0, 0.5, 0.25),
                bias_rad=bias,
                bias_weight=weight,
            )
            if reason.startswith("ik_non_finite"):
                return None
            if np.allclose(solution, q, atol=1e-9):
                return q if self._within_cone(q, target) else None
            q = np.clip(solution, self._low, self._high)
        return q if self._within_cone(q, target) else None

    def _projected_jacobian(self, joints_deg: np.ndarray) -> np.ndarray:
        """Finite-difference 3x5 Jacobian of the tip direction, roll projected out."""
        epsilon_deg = self.config.jacobian_epsilon_deg
        epsilon_rad = math.radians(epsilon_deg)
        q = np.asarray(joints_deg, dtype=float)
        current_rotation = self.fk(q)[:3, :3]
        current_direction = gripper_tip_in_robot(current_rotation)
        jacobian = np.zeros((3, len(q)), dtype=float)
        for i in range(len(q)):
            perturbed = q.copy()
            perturbed[i] += epsilon_deg
            delta = rotation_vector_from_matrix(self.fk(perturbed)[:3, :3] @ current_rotation.T)
            jacobian[:, i] = delta / epsilon_rad
        roll_projection = np.eye(3) - np.outer(current_direction, current_direction)
        return roll_projection @ jacobian

    def _sigma_reference(self) -> float:
        if self._sigma_ref is None:
            midpoint = 0.5 * (self._low + self._high)
            singular_values = np.linalg.svd(self._projected_jacobian(midpoint), compute_uv=False)
            self._sigma_ref = max(float(np.mean(singular_values)), 1e-6)
        return self._sigma_ref

    def _valid_on_manifold(self, joints_deg: np.ndarray, target: np.ndarray) -> tuple[bool, float]:
        """Single validator pass deciding limits+collision(+cone) for one config."""
        candidate, _ = self.validator.evaluate(joints_deg)
        if candidate is None or not self._within_cone(joints_deg, target):
            return False, math.inf
        return True, float(candidate.joint_margin)

    def _edge_valid_manifold(
        self,
        first: np.ndarray,
        second: np.ndarray,
        target: np.ndarray,
        check_cone: bool = True,
    ) -> bool:
        """Sampled straight-line check combining validator and direction cone.

        ``check_cone=False`` validates limits+collision only; it is used for the
        hop from the (by definition off-manifold) trapped pose onto the manifold,
        which the controller replays with its own runtime validation.
        """
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        max_delta = float(np.max(np.abs(second - first)))
        steps = max(1, int(math.ceil(max_delta / self.config.edge_resolution_deg)))
        for index in range(1, steps + 1):
            sample = first + (second - first) * (index / steps)
            candidate, _ = self.validator.evaluate(sample)
            if candidate is None:
                return False
            if check_cone and vector_angle(candidate.direction, target) > self.config.direction_tolerance_rad:
                return False
        return True

    # ---------------------------------------------------------------- goals

    def _generate_goals(self, start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        candidates: list[tuple[float, float, np.ndarray]] = []  # (score, dist_from_start, q)
        attempts = 0
        while len(candidates) < self.config.goal_candidates and attempts < self.config.goal_samples:
            attempts += 1
            sample = self._rng.uniform(self._low, self._high)
            projected = self.project(sample, target, posture_target=sample)
            if projected is None:
                continue
            candidate, _ = self.validator.evaluate(projected)
            if candidate is None or not self._within_cone(projected, target):
                continue
            singular_values = np.linalg.svd(self._projected_jacobian(projected), compute_uv=False)
            sigma_min = float(singular_values[-1])
            margin = float(candidate.joint_margin)
            manipulability = min(sigma_min / self._sigma_reference(), 1.0)
            score = 0.6 * (1.0 - margin) + 0.4 * (1.0 - manipulability)
            distance = self._normalized_distance(self._normalized(projected), self._normalized(start))
            candidates.append((score, distance, projected))

        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0])
        best_score = candidates[0][0]
        # Anti-recurrence: among near-best candidates prefer the one farthest
        # from the trapped configuration so repeated escapes stop ping-ponging.
        near_best = [item for item in candidates if item[0] <= best_score + 0.2]
        kept = [max(near_best, key=lambda item: item[1])]
        for item in candidates:
            if len(kept) >= self.config.goal_candidates:
                break
            separated = all(
                self._normalized_distance(self._normalized(item[2]), self._normalized(other[2]))
                >= self.config.goal_separation_normalized
                for other in kept
            )
            if separated:
                kept.append(item)
        return [item[2] for item in kept]

    # ------------------------------------------------------------ planning

    def _extend_tree(
        self,
        tree: tuple[list[np.ndarray], list[int]],
        sample_norm: np.ndarray,
        target: np.ndarray,
        stats: SimpleNamespace,
        cancel_event: threading.Event,
        deadline: float,
    ) -> tuple[str, int]:
        nodes, parents = tree
        nearest = min(range(len(nodes)), key=lambda i: float(np.linalg.norm(nodes[i] - sample_norm)))
        parent_index = nearest
        current_norm = nodes[nearest]
        added_any = False
        while True:
            difference = sample_norm - current_norm
            distance = float(np.linalg.norm(difference))
            if distance <= 1e-9:
                return ("reached" if added_any else "advanced"), parent_index
            step = min(distance, self.config.rrt_step_normalized)
            raw_norm = current_norm + difference / distance * step
            raw_deg = self._denormalized(raw_norm)
            if cancel_event.is_set() or time.monotonic() > deadline:
                return ("advanced" if added_any else "trapped"), parent_index
            projected = self.project(raw_deg, target, posture_target=raw_deg)
            stats.projections += 1
            if projected is None:
                return ("advanced" if added_any else "trapped"), parent_index
            anchor_deg = self._denormalized(nodes[parent_index])
            if not self._edge_valid_manifold(anchor_deg, projected, target):
                return ("advanced" if added_any else "trapped"), parent_index
            nodes.append(self._normalized(projected))
            parents.append(parent_index)
            parent_index = len(nodes) - 1
            current_norm = nodes[parent_index]
            added_any = True
            if float(np.linalg.norm(sample_norm - current_norm)) <= 1e-9:
                return "reached", parent_index

    def _connect_tree(
        self,
        tree: tuple[list[np.ndarray], list[int]],
        target_norm: np.ndarray,
        target: np.ndarray,
        stats: SimpleNamespace,
        cancel_event: threading.Event,
        deadline: float,
    ) -> tuple[str, int]:
        for _ in range(self.config.rrt_connect_attempts):
            if cancel_event.is_set() or time.monotonic() > deadline:
                return "trapped", -1
            status, index = self._extend_tree(tree, target_norm, target, stats, cancel_event, deadline)
            if status == "reached":
                return "reached", index
            if status == "trapped":
                return "trapped", index
        return "trapped", -1

    @staticmethod
    def _trace(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        path = []
        while index != -1:
            path.append(nodes[index])
            index = parents[index]
        path.reverse()
        return path

    def _cbirrt_connect(
        self,
        start_deg: np.ndarray,
        goal_deg: np.ndarray,
        target: np.ndarray,
        stats: SimpleNamespace,
        cancel_event: threading.Event,
        deadline: float,
    ) -> list[np.ndarray] | None:
        start_tree: tuple[list[np.ndarray], list[int]] = ([self._normalized(start_deg)], [-1])
        goal_tree: tuple[list[np.ndarray], list[int]] = ([self._normalized(goal_deg)], [-1])
        for iteration in range(self.config.rrt_max_iterations):
            if cancel_event.is_set() or time.monotonic() > deadline:
                return None
            if self._rng.random() < self.config.rrt_goal_bias:
                sample_norm = self._normalized(goal_deg if iteration % 2 == 0 else start_deg)
            else:
                # Project the random sample onto the manifold before extending:
                # growing between on-manifold targets keeps the straight edges
                # close to the cone, which full-space sampling does not.
                raw_deg = self._rng.uniform(self._low, self._high)
                projected = self.project(raw_deg, target, posture_target=raw_deg)
                stats.projections += 1
                if projected is None or self.validator.evaluate(projected)[0] is None:
                    continue
                sample_norm = self._normalized(projected)
            growing, connecting = (
                (start_tree, goal_tree) if iteration % 2 == 0 else (goal_tree, start_tree)
            )
            status, grown_index = self._extend_tree(
                growing, sample_norm, target, stats, cancel_event, deadline
            )
            if status == "trapped" or grown_index < 0:
                continue
            bridge_norm = growing[0][grown_index]
            connect_status, connect_index = self._connect_tree(
                connecting, bridge_norm, target, stats, cancel_event, deadline
            )
            if connect_status != "reached":
                continue
            grown_path = self._trace(*growing, grown_index)
            connect_path = self._trace(*connecting, connect_index)
            if growing is start_tree:
                combined = grown_path + connect_path[::-1]
            else:
                combined = connect_path + grown_path[::-1]
            stats.tree_nodes = len(start_tree[0]) + len(goal_tree[0])
            return [self._denormalized(node) for node in combined]
        stats.tree_nodes = len(start_tree[0]) + len(goal_tree[0])
        return None

    def _shortcut(
        self,
        waypoints: list[np.ndarray],
        target: np.ndarray,
        stats: SimpleNamespace,
        cancel_event: threading.Event,
        deadline: float,
    ) -> list[np.ndarray]:
        path = [waypoint.copy() for waypoint in waypoints]
        for _ in range(self.config.shortcut_attempts):
            if len(path) < 3 or cancel_event.is_set() or time.monotonic() > deadline:
                break
            first, second = sorted(int(v) for v in self._rng.integers(0, len(path), size=2))
            if second - first < 2:
                continue
            if self._edge_valid_manifold(path[first], path[second], target):
                path = path[: first + 1] + path[second:]
        return path

    def _resample(self, waypoints: list[np.ndarray]) -> list[np.ndarray]:
        """Split segments so no consecutive pair exceeds waypoint_delta_deg."""
        result = [waypoints[0].copy()]
        for first, second in zip(waypoints, waypoints[1:]):
            segment = float(np.max(np.abs(second - first)))
            pieces = max(1, int(math.ceil(segment / self.config.waypoint_delta_deg)))
            for piece in range(1, pieces + 1):
                result.append(first + (second - first) * (piece / pieces))
        return result

    # ---------------------------------------------------------------- api

    def plan(
        self,
        start_joints_deg: np.ndarray,
        target_direction: np.ndarray,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> EscapePlan | None:
        """Plan an on-manifold detour; the result starts on the manifold.

        The trapped pose itself sits OFF the manifold by definition (it stopped
        short of the cone), so cone-constrained edges can never connect to it.
        The plan therefore begins at the collision-checked projection of the
        start onto the manifold; the hop from the live pose to that first
        waypoint is a limits+collision-only straight-line bridge validated and
        executed by the controller (``_valid_path_suffix``).
        """
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        started = time.monotonic()
        deadline = deadline if deadline is not None else started + self.config.planning_timeout_s
        target = normalize_vector(np.asarray(target_direction, dtype=float))
        start = np.clip(np.asarray(start_joints_deg, dtype=float), self._low, self._high)
        stats = SimpleNamespace(projections=0, tree_nodes=0)

        root = self.project(start, target)
        if root is None or not self._valid_on_manifold(root, target)[0]:
            logger.debug("escape planning: start could not be projected onto the manifold")
            return None
        if not self._edge_valid_manifold(start, root, target, check_cone=False):
            logger.debug("escape planning: no collision-free bridge from the start onto the manifold")
            return None

        goals = self._generate_goals(root, target)
        logger.debug(
            "escape planning: %d goals from %d samples in %.2fs",
            len(goals),
            self.config.goal_samples,
            time.monotonic() - started,
        )
        if not goals:
            return None

        best: tuple[float, list[np.ndarray], str, EscapeGoal] | None = None
        for goal_q in goals:
            if cancel_event.is_set() or time.monotonic() > deadline:
                break
            candidate, reason = self.validator.evaluate(goal_q)
            if candidate is None:
                continue
            sigma_min = float(np.linalg.svd(self._projected_jacobian(goal_q), compute_uv=False)[-1])
            manipulability = min(sigma_min / self._sigma_reference(), 1.0)
            goal_info = EscapeGoal(
                joints_deg=[float(v) for v in goal_q],
                direction_error_deg=self.direction_error_deg(goal_q, target),
                joint_margin=float(candidate.joint_margin),
                sigma_min=sigma_min,
                score=0.6 * (1.0 - float(candidate.joint_margin)) + 0.4 * (1.0 - manipulability),
            )
            if self._edge_valid_manifold(root, goal_q, target):
                segments = [root.copy(), goal_q.copy()]
                method = "direct"
            else:
                segments = self._cbirrt_connect(root, goal_q, target, stats, cancel_event, deadline)
                if segments is None:
                    continue
                method = "cbirrt"
            smoothed = self._shortcut(segments, target, stats, cancel_event, deadline)
            resampled = self._resample(smoothed)
            # Subdivision introduces new sample points; re-check each piece so
            # every emitted segment carries its own validity proof.
            valid_chain = all(
                self._edge_valid_manifold(first, second, target)
                for first, second in zip(resampled, resampled[1:])
            )
            if not valid_chain:
                continue
            length = sum(
                float(np.linalg.norm(self._normalized(b) - self._normalized(a)))
                for a, b in zip(smoothed, smoothed[1:])
            )
            if best is None or length < best[0]:
                best = (length, resampled, method, goal_info)
            if cancel_event.is_set() or time.monotonic() > deadline:
                break

        if best is None:
            return None
        _, waypoints, method, goal_info = best
        return EscapePlan(
            waypoints_deg=[[float(v) for v in waypoint] for waypoint in waypoints],
            target_direction=[float(v) for v in target],
            goal=goal_info,
            method=method,
            planning_time_s=time.monotonic() - started,
            tree_nodes=int(stats.tree_nodes),
            projections=int(stats.projections),
        )
