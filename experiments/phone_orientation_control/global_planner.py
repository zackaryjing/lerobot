"""Global direction goals and joint-space paths for the SO101 digital twin."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import placo

from controller import ARM_JOINTS
from direction_atlas import AtlasCandidate, SO101StateValidator
from frame_adapter import GRIPPER_TIP_LOCAL, gripper_tip_in_robot


@dataclass
class GlobalPlannerConfig:
    atlas_seeds: int = 16
    refined_candidates: int = 8
    refinement_iterations: int = 40
    random_seed_attempts: int = 64
    direction_tolerance_rad: float = math.radians(0.20)
    candidate_branch_separation: float = 0.08
    edge_resolution_deg: float = 3.0
    rrt_step_normalized: float = 0.08
    rrt_iterations_per_goal: int = 1_500
    rrt_goal_bias: float = 0.18
    goals_to_plan: int = 6
    seed: int = 20_260_717


@dataclass
class DirectionGoal:
    joints_deg: list[float]
    direction: list[float]
    tip_position_m: list[float]
    direction_error_deg: float
    joint_margin: float
    current_distance: float


@dataclass
class PlannedPath:
    waypoints_deg: list[list[float]]
    goal: DirectionGoal
    method: str
    normalized_length: float


class GlobalDirectionPlanner:
    """Atlas-seeded PlaCo refinement followed by bounded RRT-Connect."""

    def __init__(
        self,
        atlas_path: Path,
        validator: SO101StateValidator,
        config: GlobalPlannerConfig | None = None,
        manual_samples_path: Path | None = None,
    ) -> None:
        self.config = config or GlobalPlannerConfig()
        self.validator = validator
        self.robot = validator.robot
        self.low = validator.low
        self.high = validator.high
        self.span = validator.span
        self._rng = np.random.default_rng(self.config.seed)
        self._lock = threading.RLock()
        atlas = json.loads(atlas_path.read_text())
        self.atlas_candidates = [candidate for item in atlas["bins"] for candidate in item["candidates"]]
        self.manual_candidate_count = 0
        if manual_samples_path is not None and manual_samples_path.is_file():
            manual = json.loads(manual_samples_path.read_text())
            if manual.get("version") != 1 or not isinstance(manual.get("samples"), list):
                raise ValueError(f"unsupported manual sample file: {manual_samples_path}")
            self.atlas_candidates.extend(manual["samples"])
            self.manual_candidate_count = len(manual["samples"])

        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.mask_dof("gripper")
        self.solver.enable_joint_limits(True)
        for name in ARM_JOINTS:
            low, high = validator.joint_limits[name]
            self.robot.set_joint_limits(name, math.radians(low), math.radians(high))
        self.axis_task = self.solver.add_axisalign_task(
            "gripper_frame_link", GRIPPER_TIP_LOCAL.copy(), np.array([1.0, 0.0, 0.0])
        )
        self.axis_task.configure("visual_tip_direction", "soft", 1.0)
        self.posture_task = self.solver.add_joints_task()
        self.posture_task.configure("atlas_branch_regularization", "soft", 1e-3)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-8:
            raise ValueError("target direction must be a finite non-zero 3-vector")
        return vector / norm

    def _normalized_distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm((first - second) / self.span) / math.sqrt(len(ARM_JOINTS)))

    def _seed_candidates(self, target: np.ndarray, current: np.ndarray) -> list[dict[str, Any]]:
        ranked = sorted(
            self.atlas_candidates,
            key=lambda candidate: (
                math.acos(float(np.clip(np.dot(candidate["direction"], target), -1.0, 1.0)))
                + 0.035 * self._normalized_distance(np.asarray(candidate["joints_deg"]), current)
                + 0.02 * (1.0 - float(candidate["joint_margin"]))
            ),
        )
        selected: list[dict[str, Any]] = []
        for candidate in ranked:
            q = np.asarray(candidate["joints_deg"])
            if all(
                self._normalized_distance(q, np.asarray(existing["joints_deg"]))
                >= self.config.candidate_branch_separation
                for existing in selected
            ):
                selected.append(candidate)
            if len(selected) >= self.config.atlas_seeds:
                break
        return selected

    def _refine(self, seed: dict[str, Any], target: np.ndarray, current: np.ndarray) -> DirectionGoal | None:
        seed_q = np.asarray(seed["joints_deg"], dtype=float)
        # Wrist roll is unobservable for the visual tip-axis objective. Keep it
        # near the current arm instead of inheriting an arbitrary atlas roll.
        seed_q[4] = current[4]
        self.validator.set_configuration(seed_q)
        self.axis_task.targetAxis_world = target
        self.posture_task.set_joints(
            {name: math.radians(value) for name, value in zip(ARM_JOINTS, seed_q, strict=True)}
        )
        for _ in range(self.config.refinement_iterations):
            self.solver.solve(True)
            self.robot.update_kinematics()
            direction = gripper_tip_in_robot(self.robot.get_T_world_frame("gripper_frame_link")[:3, :3])
            if math.acos(float(np.clip(np.dot(direction, target), -1.0, 1.0))) <= self.config.direction_tolerance_rad:
                break
        refined_q = np.rad2deg([self.robot.get_joint(name) for name in ARM_JOINTS])
        evaluated, _ = self.validator.evaluate(refined_q)
        if evaluated is None:
            return None
        direction = np.asarray(evaluated.direction)
        error = math.acos(float(np.clip(np.dot(direction, target), -1.0, 1.0)))
        if error > self.config.direction_tolerance_rad:
            return None
        return DirectionGoal(
            joints_deg=refined_q.tolist(),
            direction=evaluated.direction,
            tip_position_m=evaluated.tip_position_m,
            direction_error_deg=math.degrees(error),
            joint_margin=evaluated.joint_margin,
            current_distance=self._normalized_distance(refined_q, current),
        )

    def find_goals(self, target_direction: np.ndarray, current_joints_deg: np.ndarray) -> list[DirectionGoal]:
        target = self._unit(target_direction)
        current = np.asarray(current_joints_deg, dtype=float)
        with self._lock:
            goals: list[DirectionGoal] = []
            current_candidate, _ = self.validator.evaluate(current)
            if current_candidate is not None:
                current_direction = np.asarray(current_candidate.direction)
                current_error = math.acos(
                    float(np.clip(np.dot(current_direction, target), -1.0, 1.0))
                )
                if current_error <= self.config.direction_tolerance_rad:
                    goals.append(
                        DirectionGoal(
                            joints_deg=current.tolist(),
                            direction=current_candidate.direction,
                            tip_position_m=current_candidate.tip_position_m,
                            direction_error_deg=math.degrees(current_error),
                            joint_margin=current_candidate.joint_margin,
                            current_distance=0.0,
                        )
                    )
            for seed in self._seed_candidates(target, current):
                goal = self._refine(seed, target, current)
                if goal is None:
                    continue
                q = np.asarray(goal.joints_deg)
                if any(
                    self._normalized_distance(q, np.asarray(existing.joints_deg))
                    < self.config.candidate_branch_separation
                    for existing in goals
                ):
                    continue
                goals.append(goal)
                if len(goals) >= self.config.refined_candidates:
                    break
            # Sparse atlas regions (e.g. directions past the shoulder_pan sweep)
            # hold only a handful of candidates. Refine random seeds with the
            # same PlaCo pass to fill those gaps instead of failing the plan.
            for _ in range(self.config.random_seed_attempts):
                if len(goals) >= self.config.refined_candidates:
                    break
                raw = self._rng.uniform(self.low, self.high)
                goal = self._refine(
                    {
                        "joints_deg": raw.tolist(),
                        "direction": target.tolist(),
                        "joint_margin": 0.0,
                    },
                    target,
                    current,
                )
                if goal is None:
                    continue
                q = np.asarray(goal.joints_deg)
                if any(
                    self._normalized_distance(q, np.asarray(existing.joints_deg))
                    < self.config.candidate_branch_separation
                    for existing in goals
                ):
                    continue
                goals.append(goal)
            goals.sort(key=lambda goal: goal.current_distance + 0.08 * (1.0 - goal.joint_margin))
            return goals

    def _is_valid(self, joints_deg: np.ndarray) -> bool:
        candidate, _ = self.validator.evaluate(joints_deg)
        return candidate is not None

    def _edge_valid(self, first: np.ndarray, second: np.ndarray) -> bool:
        steps = max(1, int(math.ceil(float(np.max(np.abs(second - first))) / self.config.edge_resolution_deg)))
        return all(self._is_valid(first + (second - first) * (index / steps)) for index in range(1, steps + 1))

    def _normalized(self, joints_deg: np.ndarray) -> np.ndarray:
        return (joints_deg - self.low) / self.span

    def _degrees(self, normalized: np.ndarray) -> np.ndarray:
        return self.low + normalized * self.span

    @staticmethod
    def _trace(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return result

    def _extend(
        self, nodes: list[np.ndarray], parents: list[int], target: np.ndarray
    ) -> tuple[str, int | None]:
        nearest = min(range(len(nodes)), key=lambda index: float(np.linalg.norm(nodes[index] - target)))
        delta = target - nodes[nearest]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-10:
            return "reached", nearest
        step = min(distance, self.config.rrt_step_normalized)
        new = nodes[nearest] + delta * (step / distance)
        if not self._edge_valid(self._degrees(nodes[nearest]), self._degrees(new)):
            return "trapped", None
        nodes.append(new)
        parents.append(nearest)
        return ("reached" if step >= distance - 1e-10 else "advanced"), len(nodes) - 1

    def _connect(
        self, nodes: list[np.ndarray], parents: list[int], target: np.ndarray
    ) -> tuple[str, int | None]:
        while True:
            status, index = self._extend(nodes, parents, target)
            if status != "advanced":
                return status, index

    def _rrt_connect(
        self,
        start_deg: np.ndarray,
        goal_deg: np.ndarray,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> list[np.ndarray] | None:
        if self._edge_valid(start_deg, goal_deg):
            return [start_deg, goal_deg]
        start = self._normalized(start_deg)
        goal = self._normalized(goal_deg)
        nodes_a, parents_a = [start], [-1]
        nodes_b, parents_b = [goal], [-1]
        a_is_start = True
        for _ in range(self.config.rrt_iterations_per_goal):
            if (cancel_event is not None and cancel_event.is_set()) or (
                deadline is not None and time.monotonic() > deadline
            ):
                return None
            sample = goal if self._rng.random() < self.config.rrt_goal_bias else self._rng.random(5)
            status_a, index_a = self._extend(nodes_a, parents_a, sample)
            if status_a != "trapped" and index_a is not None:
                status_b, index_b = self._connect(nodes_b, parents_b, nodes_a[index_a])
                if status_b == "reached" and index_b is not None:
                    if a_is_start:
                        start_part = list(reversed(self._trace(nodes_a, parents_a, index_a)))
                        goal_part = self._trace(nodes_b, parents_b, index_b)
                    else:
                        start_part = list(reversed(self._trace(nodes_b, parents_b, index_b)))
                        goal_part = self._trace(nodes_a, parents_a, index_a)
                    normalized_path = start_part + goal_part[1:]
                    return [self._degrees(item) for item in normalized_path]
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            a_is_start = not a_is_start
        return None

    def _shortcut(self, path: list[np.ndarray], attempts: int = 80) -> list[np.ndarray]:
        path = list(path)
        for _ in range(attempts):
            if len(path) <= 2:
                break
            first, second = sorted(self._rng.choice(len(path), size=2, replace=False).tolist())
            if second <= first + 1:
                continue
            if self._edge_valid(path[first], path[second]):
                path = path[: first + 1] + path[second:]
        return path

    def _path_length(self, path: list[np.ndarray]) -> float:
        return sum(self._normalized_distance(first, second) for first, second in zip(path, path[1:]))

    def plan(
        self,
        current_joints_deg: np.ndarray,
        target_direction: np.ndarray,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> PlannedPath | None:
        with self._lock:
            current = np.asarray(current_joints_deg, dtype=float)
            if not self._is_valid(current):
                raise ValueError("current configuration is not valid for global planning")
            goals = self.find_goals(target_direction, current)
            best: PlannedPath | None = None
            for goal in goals[: self.config.goals_to_plan]:
                if (cancel_event is not None and cancel_event.is_set()) or (
                    deadline is not None and time.monotonic() > deadline
                ):
                    break
                raw_path = self._rrt_connect(
                    current, np.asarray(goal.joints_deg), cancel_event=cancel_event, deadline=deadline
                )
                if raw_path is None:
                    continue
                path = self._shortcut(raw_path)
                length = self._path_length(path)
                method = "direct" if len(path) == 2 else "rrt_connect"
                candidate = PlannedPath(
                    waypoints_deg=[item.tolist() for item in path],
                    goal=goal,
                    method=method,
                    normalized_length=length,
                )
                if best is None or candidate.normalized_length < best.normalized_length:
                    best = candidate
            return best
