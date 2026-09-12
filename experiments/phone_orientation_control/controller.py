#!/usr/bin/env python
"""Orientation-only commanded-state planner for an SO101 follower arm.

The phone supplies orientation only. End-effector position is deliberately left
unconstrained and follows from the commanded joint-space solution. This module
has no browser or networking code so its mapping can be tested without
connecting hardware.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np

from lerobot.model.kinematics import RobotKinematics

from frame_adapter import (
    gripper_tip_in_robot,
    make_xr_to_robot_rotation,
    phone_forward_in_robot,
)


ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = [*ARM_JOINTS, "gripper"]
TrackingMode = Literal["smooth", "realtime"]
logger = logging.getLogger("phone_orientation_control.controller")

class RobotLike(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...

    def disconnect(self) -> None: ...


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-6:
        raise ValueError("quaternion norm is too small")
    return q / norm


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions."""
    ax, ay, az, aw = normalize_quat(a)
    bx, by, bz, bw = normalize_quat(b)
    return normalize_quat(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ]
        )
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array([-x, -y, -z, w])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return normalize_quat(np.asarray(q))


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(normalize_quat(a), normalize_quat(b))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def quat_slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    a, b = normalize_quat(a), normalize_quat(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quat(a + fraction * (b - a))
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    return normalize_quat((math.sin((1 - fraction) * theta) * a + math.sin(fraction * theta) * b) / math.sin(theta))


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    value = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.acos(float(np.clip(value, -1.0, 1.0)))


def rotation_vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    q = matrix_to_quat(matrix)
    if q[3] < 0:
        q = -q
    angle = 2.0 * math.acos(float(np.clip(q[3], -1.0, 1.0)))
    sin_half = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if sin_half < 1e-8:
        return 2.0 * q[:3]
    return q[:3] * (angle / sin_half)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError("direction must contain three finite non-zero values")
    return vector / norm


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(normalize_vector(a), normalize_vector(b)), -1.0, 1.0)))


def vector_slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    a, b = normalize_vector(a), normalize_vector(b)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        return normalize_vector(a + fraction * (b - a))
    if dot < -0.9995:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(a, helper))
        return normalize_vector(a * math.cos(math.pi * fraction) + axis * math.sin(math.pi * fraction))
    angle = math.acos(dot)
    return normalize_vector(
        (math.sin((1.0 - fraction) * angle) * a + math.sin(fraction * angle) * b) / math.sin(angle)
    )


def bounded_damped_least_squares(
    jacobian: np.ndarray,
    error: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    damping: float,
) -> np.ndarray:
    count = jacobian.shape[1]
    delta = np.zeros(count, dtype=float)
    free = list(range(count))
    fixed: list[int] = []
    while free:
        residual = error.copy()
        if fixed:
            residual -= jacobian[:, fixed] @ delta[fixed]
        free_jacobian = jacobian[:, free]
        delta[free] = free_jacobian.T @ np.linalg.solve(
            free_jacobian @ free_jacobian.T + damping * damping * np.eye(3),
            residual,
        )
        violations: list[tuple[float, int, float]] = []
        for index in free:
            if delta[index] < lower[index]:
                violations.append((lower[index] - delta[index], index, lower[index]))
            elif delta[index] > upper[index]:
                violations.append((delta[index] - upper[index], index, upper[index]))
        if not violations:
            break
        _, index, bound = max(violations)
        delta[index] = bound
        free.remove(index)
        fixed.append(index)
    return np.clip(delta, lower, upper)


def direction_tip_step(
    fk: Callable[[np.ndarray], np.ndarray],
    current: np.ndarray,
    desired_direction: np.ndarray,
    *,
    lower_delta_rad: np.ndarray,
    upper_delta_rad: np.ndarray,
    damping: float,
    jacobian_epsilon_deg: float,
    max_step_rad: float,
    accept_tolerance_rad: float = math.radians(0.25),
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625),
    valid: Callable[[np.ndarray], bool] | None = None,
    edge_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    bias_rad: np.ndarray | None = None,
    bias_weight: float = 0.0,
) -> tuple[np.ndarray, str]:
    """One bounded differential-IK step tracking only the visual gripper-tip axis.

    Shared verbatim by the live tracker and the escape planner so both agree on
    what "a step toward the target direction" means. Roll around the tip axis is
    free. ``valid`` gates a candidate configuration; ``edge_valid`` gates the
    straight joint-space segment from ``current``; ``None`` accepts everything.
    ``bias_rad`` optionally adds a null-space posture drift (weighted by
    ``bias_weight``) so repeated projections spread across branches instead of
    collapsing into one basin; the task step itself stays untouched.
    """
    current = np.asarray(current, dtype=float)
    current_rotation = fk(current)[:3, :3]
    current_direction = gripper_tip_in_robot(current_rotation)
    desired_direction = normalize_vector(desired_direction)
    cross = np.cross(current_direction, desired_direction)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(current_direction, desired_direction), -1.0, 1.0))
    if cross_norm < 1e-8:
        helper = (
            np.array([1.0, 0.0, 0.0])
            if abs(current_direction[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        error = (
            np.zeros(3)
            if dot > 0.0
            else normalize_vector(np.cross(current_direction, helper)) * math.pi
        )
    else:
        error = cross * (math.atan2(cross_norm, dot) / cross_norm)
    error_norm = float(np.linalg.norm(error))
    if error_norm > max_step_rad:
        error *= max_step_rad / error_norm
    epsilon_rad = math.radians(jacobian_epsilon_deg)
    jacobian = np.zeros((3, len(current)), dtype=float)
    for i in range(len(current)):
        perturbed = current.copy()
        perturbed[i] += jacobian_epsilon_deg
        perturbed_rotation = fk(perturbed)[:3, :3]
        delta = rotation_vector_from_matrix(perturbed_rotation @ current_rotation.T)
        jacobian[:, i] = delta / epsilon_rad

    # Rotation around the tip itself has no visible effect and must not
    # cause wrist-roll motion. Remove that unobservable component.
    direction_projection = np.eye(3) - np.outer(current_direction, current_direction)
    jacobian = direction_projection @ jacobian
    delta_rad = bounded_damped_least_squares(jacobian, error, lower_delta_rad, upper_delta_rad, damping)
    if bias_rad is not None and bias_weight > 0.0:
        # Task-consistent posture drift: project the bias into the Jacobian's
        # null space so it cannot fight the tracking error, then re-clip.
        pseudo_inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * damping * np.eye(3), np.eye(3)
        )
        null_projection = np.eye(len(current)) - pseudo_inverse @ jacobian
        delta_rad = np.clip(
            delta_rad + bias_weight * (null_projection @ np.asarray(bias_rad, dtype=float)),
            lower_delta_rad,
            upper_delta_rad,
        )
    raw_solution = current + np.rad2deg(delta_rad)

    if raw_solution.shape != current.shape or not np.all(np.isfinite(raw_solution)):
        return current.copy(), "ik_non_finite: differential IK returned non-finite joint values"

    def accepted(candidate: np.ndarray) -> bool:
        if valid is not None and not valid(candidate):
            return False
        return edge_valid is None or edge_valid(current, candidate)

    before_error = vector_angle(current_direction, desired_direction)
    for fraction in fractions:
        solution = current + fraction * (raw_solution - current)
        if not accepted(solution):
            continue
        reached_direction = gripper_tip_in_robot(fk(solution)[:3, :3])
        after_error = vector_angle(reached_direction, desired_direction)
        if after_error <= before_error + accept_tolerance_rad:
            return solution, ""

    return current.copy(), "collision_or_limits: no valid orientation step"


def load_joint_limits(urdf_path: Path, calibration_path: Path) -> dict[str, tuple[float, float]]:
    """Load the exact LeRobot calibration range without an artificial margin."""
    del urdf_path  # Kept in the signature for existing experiment scripts.
    calibration = json.loads(calibration_path.read_text())
    result: dict[str, tuple[float, float]] = {}
    for name in ARM_JOINTS:
        entry = calibration[name]
        half_span = (float(entry["range_max"]) - float(entry["range_min"])) * 180.0 / 4095.0
        low = -half_span
        high = half_span
        if low >= high:
            raise ValueError(f"empty safe joint range for {name}: {low:.1f}..{high:.1f}")
        result[name] = (low, high)
    return result


def load_calibration_limits(calibration_path: Path) -> dict[str, tuple[float, float]]:
    """Load the physical ranges measured during LeRobot calibration."""
    calibration = json.loads(calibration_path.read_text())
    result: dict[str, tuple[float, float]] = {}
    for name in ARM_JOINTS:
        entry = calibration[name]
        half_span = (float(entry["range_max"]) - float(entry["range_min"])) * 180.0 / 4095.0
        result[name] = (-half_span, half_span)
    return result


@dataclass
class ControlConfig:
    fps: float = 30.0
    phone_filter_tau_s: float = 0.10
    realtime_follow_tau_s: float = 0.16
    angular_speed_rad_s: float = math.radians(45.0)
    target_angle_threshold_rad: float = math.radians(2.0)
    max_phone_frame_jump_rad: float = math.radians(35.0)
    phone_stale_timeout_s: float = 0.50
    path_validation_resolution_deg: float = 0.5
    jacobian_epsilon_deg: float = 0.15
    differential_ik_damping: float = 0.08
    max_orientation_ik_step_rad: float = math.radians(3.0)
    # Escape (direction-preserving reconfiguration) tuning.
    escape_enabled: bool = True
    trap_tick_threshold: int = 15
    trap_stall_error_rad: float = math.radians(0.15)
    trap_min_error_rad: float = math.radians(0.5)
    trap_target_move_rad: float = math.radians(1.5)
    escape_cooldown_s: float = 6.0
    escape_waypoint_delta_deg: float = 2.0
    escape_target_drift_abort_rad: float = math.radians(12.0)
    escape_planning_timeout_s: float = 12.0
    escape_loop_window_s: float = 60.0
    escape_loop_limit: int = 3
    # Anticipatory free-space planning: while stall ticks accumulate toward
    # the trap threshold, a background worker plans from the current pose to
    # the current target so the trap can adopt it without a planning pause.
    preplan_enabled: bool = True
    preplan_tick_threshold: int = 2
    preplan_reuse_rad: float = math.radians(5.0)
    preplan_min_interval_s: float = 1.0
    preplan_timeout_s: float = 4.0


@dataclass
class DirectionPlan:
    start_direction: np.ndarray
    target_direction: np.ndarray
    started_at: float
    duration_s: float


class EscapePlannerLike(Protocol):
    """Structural type for the direction-preserving reconfiguration planner."""

    def plan(
        self,
        start_joints_deg: np.ndarray,
        target_direction: np.ndarray,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> Any | None: ...


class OrientationController:
    """Threaded controller; hardware access happens only in its control thread."""

    def __init__(
        self,
        kinematics: RobotKinematics,
        joint_limits: dict[str, tuple[float, float]],
        robot: RobotLike | None = None,
        config: ControlConfig | None = None,
        dry_initial_joints: np.ndarray | None = None,
        hard_joint_limits: dict[str, tuple[float, float]] | None = None,
        reset_joints: np.ndarray | None = None,
        state_validator: Callable[[np.ndarray], bool] | None = None,
        escape_planner: EscapePlannerLike | None = None,
        fallback_planner: EscapePlannerLike | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.joint_limits = joint_limits
        self.hard_joint_limits = hard_joint_limits or joint_limits
        self.robot = robot
        self.config = config or ControlConfig()
        self.hardware = robot is not None
        self._state_validator = state_validator
        self._escape_planner = escape_planner
        self._fallback_planner = fallback_planner
        self._lock = threading.Lock()
        self._kinematics_lock = threading.RLock()
        self._validation_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_phone_quat: np.ndarray | None = None
        self._latest_phone_time = 0.0
        self._phone_transport_connected = False
        self._awaiting_reconnect_pose = False
        self._phone_rebase_count = 0
        self._last_phone_rebase_deg = 0.0
        self._filtered_phone_quat: np.ndarray | None = None
        self._calib_phone_quat: np.ndarray | None = None
        self._phone_to_robot_rotation: np.ndarray | None = None
        self._calibrated = False
        self._sync_enabled = False
        self._mode: TrackingMode = "smooth"
        self._plan: DirectionPlan | None = None
        self._sim_trajectory: list[np.ndarray] | None = None
        self._sim_waypoint_index = 0
        self._sim_target_direction: np.ndarray | None = None
        self._reject_count = 0
        self._rejection_counts: dict[str, int] = {}
        self._last_rejection = ""
        self._last_error = ""
        self._last_status = "waiting for phone calibration"
        self._current_tip_direction: np.ndarray | None = None
        self._target_tip_direction: np.ndarray | None = None
        self._tip_position_m: np.ndarray | None = None
        # Escape state machine: idle -> planning -> executing -> completed/failed.
        self._escape_state: str = "idle"
        self._escape_plan_q: list[np.ndarray] | None = None
        self._escape_index = 0
        self._escape_direction: np.ndarray | None = None
        self._escape_generation = 0
        self._escape_thread: threading.Thread | None = None
        self._escape_cancel = threading.Event()
        self._escape_trap_ticks = 0
        self._escape_anchor_direction: np.ndarray | None = None
        self._escape_prev_error_rad: float | None = None
        self._escape_next_allowed_time = -math.inf
        self._escape_started_monotonic = 0.0
        self._escape_last_reason = ""
        self._escape_outcome_counts: dict[str, int] = {}
        self._escape_history: list[tuple[float, np.ndarray]] = []
        self._escape_path_payload: tuple[int, dict[str, Any]] | None = None
        self._preplan: tuple[np.ndarray, Any] | None = None
        self._preplan_worker_active = False
        self._preplan_last_started = -math.inf
        initial_joints = (
            dry_initial_joints
            if dry_initial_joints is not None
            else reset_joints
            if reset_joints is not None
            else [0, 30, -60, -30, 0, 50]
        )
        self._q = np.asarray(initial_joints, dtype=float)
        self._last_measured_q = self._q.copy()
        self._last_requested_command_q = self._q.copy()
        self._last_sent_command_q = self._q.copy()
        self._startup_q = np.asarray(reset_joints, dtype=float).copy() if reset_joints is not None else self._q.copy()
        if self._startup_q.shape != (6,):
            raise ValueError("reset pose must contain six joint values")
        if not np.all(np.isfinite(self._startup_q)) or not 0.0 <= self._startup_q[5] <= 100.0:
            raise ValueError("reset pose contains an invalid calibrated gripper target")
        self._last_tick = time.monotonic()
        self._control_time_s = 0.0
        self._control_frame = 0

    def start(self) -> None:
        if self.hardware:
            observation = self.robot.get_observation()
            measured_q = np.array([float(observation[f"{name}.pos"]) for name in ALL_JOINTS])
            self._last_measured_q = measured_q
            if not np.all(np.isfinite(measured_q)):
                raise RuntimeError("startup joint state contains non-finite values")
            startup_path = self._valid_path_suffix(measured_q[:5], self._startup_q[:5])
            if startup_path is None:
                raise RuntimeError("configured startup target is outside calibration limits or self-colliding")
            self._q = measured_q.copy()
            self._last_requested_command_q = measured_q.copy()
            self._last_sent_command_q = measured_q.copy()
            self._sim_trajectory = startup_path
            self._sim_waypoint_index = 0
            self._last_status = f"moving to valid startup target ({len(startup_path)} waypoints)"
        elif not self._target_valid(self._q[:5]) or not 0.0 <= self._q[5] <= 100.0:
            raise RuntimeError("dry-run initial target is outside calibration limits or self-colliding")
        self._thread = threading.Thread(target=self._run, name="phone-orientation-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._escape_cancel.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.hardware and self.robot.is_connected:
            self.robot.disconnect()

    def submit_phone_orientation(self, quat: list[float]) -> None:
        q = normalize_quat(np.asarray(quat, dtype=float))
        now = time.monotonic()
        with self._lock:
            if self._sync_enabled and self._latest_phone_quat is not None:
                jump = quat_angle(self._latest_phone_quat, q)
                if jump > self.config.max_phone_frame_jump_rad:
                    if self._awaiting_reconnect_pose and self._phone_to_robot_rotation is not None:
                        old_reference = (
                            self._latest_phone_quat
                            if self._filtered_phone_quat is None
                            else self._filtered_phone_quat
                        )
                        self._phone_to_robot_rotation = (
                            self._phone_to_robot_rotation
                            @ quat_to_matrix(old_reference)
                            @ quat_to_matrix(q).T
                        )
                        self._filtered_phone_quat = q.copy()
                        self._phone_rebase_count += 1
                        self._last_phone_rebase_deg = math.degrees(jump)
                        self._last_status = "phone reference frame rebased after reconnect"
                        logger.info(
                            "Rebased phone reference frame after reconnect jump of %.1f deg",
                            math.degrees(jump),
                        )
                    else:
                        self._reject_count += 1
                        self._last_rejection = f"phone_jump: {math.degrees(jump):.1f} deg"
                        self._rejection_counts["phone_jump"] = self._rejection_counts.get("phone_jump", 0) + 1
                        if self._reject_count <= 5 or self._reject_count % 30 == 0:
                            logger.warning("Rejected phone sample: %s", self._last_rejection)
                        return
            self._awaiting_reconnect_pose = False
            self._latest_phone_quat = q
            self._latest_phone_time = now

    def set_phone_transport_connected(self, connected: bool) -> None:
        with self._lock:
            was_connected = self._phone_transport_connected
            self._phone_transport_connected = connected
            if connected:
                self._awaiting_reconnect_pose = (
                    not was_connected and self._latest_phone_quat is not None
                )
                if self._sync_enabled:
                    self._last_status = "phone link restored; resuming synchronization"
            else:
                self._awaiting_reconnect_pose = self._latest_phone_quat is not None
                if self._sync_enabled:
                    self._last_status = "phone link lost; holding position"

    def calibrate(self) -> tuple[bool, str]:
        with self._lock:
            if self._sync_enabled:
                return False, "disable synchronization before recalibrating"
            if (
                self._latest_phone_quat is None
                or time.monotonic() - self._latest_phone_time > self.config.phone_stale_timeout_s
            ):
                return False, "no fresh phone orientation"
            phone_q = self._latest_phone_quat.copy()

        with self._kinematics_lock:
            # Degree zero is the midpoint encoded by LeRobot calibration. In
            # this pose gripper_frame_link +Z visually points robot-base +X.
            reference_robot_rotation = self.kinematics.forward_kinematics(np.zeros(5))[:3, :3]
        phone_rotation = quat_to_matrix(phone_q)
        with self._lock:
            self._calib_phone_quat = phone_q
            self._filtered_phone_quat = phone_q
            self._phone_to_robot_rotation = make_xr_to_robot_rotation(
                reference_robot_rotation, phone_rotation
            )
            self._target_tip_direction = gripper_tip_in_robot(reference_robot_rotation)
            self._calibrated = True
            self._sync_enabled = False
            self._plan = None
            self._last_error = ""
            self._last_status = "calibrated; synchronization is off"
        return True, "calibrated"

    def set_sync(self, enabled: bool) -> tuple[bool, str]:
        with self._lock:
            if enabled and not self._calibrated:
                return False, "calibration is required first"
            if enabled and time.monotonic() - self._latest_phone_time > self.config.phone_stale_timeout_s:
                return False, "phone orientation is stale"
            if enabled and self._sim_trajectory is not None:
                return False, "joint path is still moving to its target"
            if not enabled and self._escape_state != "idle":
                self._cancel_escape_locked("sync disabled")
            self._sync_enabled = enabled
            self._plan = None
            self._preplan = None  # direction cache is meaningless across sync edges
            if enabled:
                self._sim_trajectory = None
                self._sim_waypoint_index = 0
                self._sim_target_direction = None
            self._last_status = "synchronization enabled" if enabled else "holding position"
        return True, self._last_status

    def invalidate_calibration(self) -> None:
        """Require fresh phone calibration without interrupting a joint path."""
        with self._lock:
            if self._escape_state != "idle":
                self._cancel_escape_locked("calibration invalidated")
            self._preplan = None  # planned for a direction in the old mapping
            self._calibrated = False
            self._sync_enabled = False
            self._latest_phone_quat = None
            self._latest_phone_time = 0.0
            self._phone_transport_connected = False
            self._awaiting_reconnect_pose = False
            self._phone_rebase_count = 0
            self._last_phone_rebase_deg = 0.0
            self._filtered_phone_quat = None
            self._phone_to_robot_rotation = None
            self._target_tip_direction = None
            self._plan = None
            self._reject_count = 0
            self._rejection_counts = {}
            self._last_rejection = ""
            self._last_error = ""
            self._last_status = "new browser session; calibration required"

    def set_mode(self, mode: str) -> tuple[bool, str]:
        if mode not in ("smooth", "realtime"):
            return False, f"unknown mode: {mode}"
        with self._lock:
            self._mode = mode  # type: ignore[assignment]
            self._plan = None
        return True, f"mode set to {mode}"

    def request_reset(self) -> tuple[bool, str]:
        with self._lock:
            if self._escape_state != "idle":
                self._cancel_escape_locked("reset requested")
            self._sync_enabled = False
            self._plan = None
            self._preplan = None
            path = self._valid_path_suffix(self._q[:5], self._startup_q[:5])
            if path is None:
                return False, "reset target is outside calibration limits or self-colliding"
            self._sim_trajectory = path
            self._sim_waypoint_index = 0
            self._sim_target_direction = None
            self._last_status = f"moving to reset target ({len(path)} waypoints)"
        return True, self._last_status

    def current_arm_joints(self) -> np.ndarray:
        with self._lock:
            return self._q[:5].copy()

    def set_simulated_trajectory(
        self,
        waypoints_deg: list[list[float]],
        target_direction: list[float],
    ) -> tuple[bool, str]:
        """Load planner waypoints without adding execution-time constraints."""
        waypoints = [np.asarray(item, dtype=float) for item in waypoints_deg]
        if not waypoints or any(item.shape != (5,) or not np.all(np.isfinite(item)) for item in waypoints):
            return False, "trajectory must contain finite five-joint waypoints"
        if any(not self._target_valid(item) for item in waypoints):
            return False, "trajectory contains an out-of-calibration or self-colliding target"
        if any(not self._edge_valid(first, second) for first, second in zip(waypoints, waypoints[1:])):
            return False, "trajectory contains a self-colliding segment"
        direction = normalize_vector(np.asarray(target_direction, dtype=float))
        with self._lock:
            self._sync_enabled = False
            self._plan = None
            self._sim_trajectory = [item.copy() for item in waypoints]
            self._sim_waypoint_index = 0
            self._sim_target_direction = direction.copy()
            self._target_tip_direction = direction.copy()
            self._last_error = ""
            self._last_status = f"playing global path (0/{len(waypoints)})"
        return True, f"playing {len(waypoints)} global path waypoints"

    def cancel_simulated_trajectory(self) -> tuple[bool, str]:
        with self._lock:
            self._sim_trajectory = None
            self._sim_waypoint_index = 0
            self._sim_target_direction = None
            self._last_status = "simulation path stopped; holding position"
        return True, self._last_status

    def global_path_execution_allowed(self) -> bool:
        return True

    # -------------------------------------------------------------- escape
    # Direction-preserving reconfiguration: when the local tracker stalls
    # against limits, plan a path through the direction manifold to a better
    # conditioned configuration, play it back, then resume tracking.

    def trigger_escape(self, direction: Any = None) -> tuple[bool, str]:
        """Manually request an escape (digital-twin test hook).

        Requires calibration and synchronization, because the escape plays back
        inside the tracking loop. With ``direction`` given, the planner freezes
        exactly that target; otherwise the current filtered phone direction is
        frozen. Manual requests bypass cooldown and the stall threshold but still
        refuse while another escape is active.
        """
        target = None
        if direction is not None:
            try:
                target = normalize_vector(np.asarray(direction, dtype=float))
            except ValueError as exc:
                return False, f"invalid escape direction: {exc}"
        spawn_args = None
        with self._lock:
            if not self._calibrated:
                return False, "calibration is required first"
            if not self._sync_enabled:
                return False, "synchronization is required for escape playback"
            if self._escape_state in ("planning", "executing"):
                return False, f"escape is {self._escape_state}"
            # A finished/failed escape may be re-triggered manually immediately;
            # only automatic triggering honors the cooldown.
            if target is None:
                target = self._live_target_direction_locked()
                if target is None:
                    return False, "no phone-derived target direction is available yet"
            spawn_args = self._begin_escape_locked(target)
        if spawn_args is not None:
            threading.Thread(
                target=self._escape_worker, args=spawn_args, name="escape-planner", daemon=True
            ).start()
        return True, "escape planning started"

    def cancel_escape(self) -> tuple[bool, str]:
        with self._lock:
            active = self._escape_state != "idle"
            self._cancel_escape_locked("cancelled by request")
            if active:
                self._last_status = "escape cancelled; holding position"
        return True, ("escape cancelled" if active else "no active escape")

    def escape_path_payload(self) -> tuple[int, dict[str, Any]] | None:
        """Latest successfully planned escape for the digital twin, if any."""
        with self._lock:
            if self._escape_path_payload is None:
                return None
            generation, payload = self._escape_path_payload
            return generation, dict(payload)

    def _live_target_direction_locked(self) -> np.ndarray | None:
        mapping = self._phone_to_robot_rotation
        filtered = self._filtered_phone_quat
        if mapping is None or filtered is None:
            return None
        return phone_forward_in_robot(mapping, quat_to_matrix(filtered))

    def _reset_trap_locked(self) -> None:
        self._escape_trap_ticks = 0
        self._escape_anchor_direction = None
        self._escape_prev_error_rad = None

    def _register_stall_locked(self) -> tuple | None:
        """Count one stalled tick unless the phone target moved past the window."""
        live_target = self._live_target_direction_locked()
        if live_target is None:
            self._reset_trap_locked()
            return None
        if self._escape_anchor_direction is None:
            self._escape_anchor_direction = live_target.copy()
        elif vector_angle(live_target, self._escape_anchor_direction) > self.config.trap_target_move_rad:
            # Deliberate fast phone motion legitimately saturates joints; only a
            # stable target counts toward the trap threshold.
            self._escape_anchor_direction = live_target.copy()
            self._reset_trap_locked()
            return None
        self._escape_trap_ticks += 1
        self._maybe_start_preplan_locked()
        return self._maybe_trigger_escape_locked()

    def _maybe_start_preplan_locked(self) -> None:
        """Kick off an anticipatory free-space plan once stalls accumulate.

        Runs in the background while tracking continues; the trap adopts the
        cached plan without a planning pause when the frozen direction still
        matches. Uses the fallback (free-space) planner only: it is fast, is
        thread-safe, and the direction-preserving tier rarely converges from
        trap-like poses anyway.
        """
        cfg = self.config
        if not cfg.preplan_enabled or self._fallback_planner is None:
            return
        if self._escape_state != "idle" or self._preplan_worker_active:
            return
        if self._escape_trap_ticks < cfg.preplan_tick_threshold:
            return
        now = time.monotonic()
        if now - self._preplan_last_started < cfg.preplan_min_interval_s:
            return
        direction = self._live_target_direction_locked()
        if direction is None:
            return
        if self._preplan is not None and vector_angle(self._preplan[0], direction) <= cfg.preplan_reuse_rad:
            return  # a usable plan for this direction already exists
        self._preplan_worker_active = True
        self._preplan_last_started = now
        start_q = self._q[:5].copy()
        threading.Thread(
            target=self._preplan_worker, args=(start_q, direction), name="escape-preplan", daemon=True
        ).start()

    def _preplan_worker(self, start_q: np.ndarray, direction: np.ndarray) -> None:
        deadline = time.monotonic() + self.config.preplan_timeout_s
        try:
            plan_obj = self._fallback_planner.plan(
                start_q,
                direction,
                cancel_event=self._escape_cancel,
                deadline=deadline,
            )
        except Exception:  # noqa: BLE001 - worker must never crash silently
            logger.exception("pre-planning failed")
            plan_obj = None
        with self._lock:
            self._preplan_worker_active = False
            if plan_obj is None or self._escape_state != "idle":
                return
            live = self._live_target_direction_locked()
            if live is None or vector_angle(live, direction) > self.config.preplan_reuse_rad:
                return  # the target moved while planning; the result is stale
            if getattr(plan_obj, "method", ""):
                plan_obj.method = f"global:{plan_obj.method}"
            self._preplan = (direction, plan_obj)
            logger.info("pre-plan cached for trap adoption")

    def _note_tracking_failure(self) -> tuple | None:
        """Record a rejected IK tick; returns escape worker spawn args if triggered."""
        with self._lock:
            if self._escape_state != "idle":
                return None
            return self._register_stall_locked()

    def _note_tracking_success(self, achieved_error_rad: float) -> tuple | None:
        """Record a sent IK tick; stalls (no error improvement) also count."""
        with self._lock:
            prev = self._escape_prev_error_rad
            self._escape_prev_error_rad = achieved_error_rad
            if self._escape_state != "idle":
                return None
            if achieved_error_rad < self.config.trap_min_error_rad:
                self._reset_trap_locked()
                return None
            stalled = prev is not None and (prev - achieved_error_rad) < self.config.trap_stall_error_rad
            if not stalled:
                self._escape_trap_ticks = 0
                return None
            return self._register_stall_locked()

    def _maybe_trigger_escape_locked(self) -> tuple | None:
        cfg = self.config
        if not cfg.escape_enabled or (
            self._escape_planner is None and self._fallback_planner is None
        ):
            return None
        if self._escape_state != "idle":
            return None
        if not self._calibrated or not self._sync_enabled:
            return None
        if self._sim_trajectory is not None:
            return None
        # Use the control loop's own clock so trap timing is consistent with
        # the tick that registers stalls; wall and logical time coincide in the
        # live system but not under fast simulated test harnesses.
        now = self._last_tick
        if now - self._latest_phone_time > cfg.phone_stale_timeout_s:
            return None
        if now < self._escape_next_allowed_time:
            return None
        if self._escape_trap_ticks < cfg.trap_tick_threshold:
            return None
        target = self._live_target_direction_locked()
        if target is None:
            return None
        return self._begin_escape_locked(target)

    def _begin_escape_locked(self, direction: np.ndarray) -> tuple | None:
        """Freeze the target and switch to planning; returns worker spawn args."""
        self._escape_generation += 1
        self._escape_state = "planning"
        self._escape_plan_q = None
        self._escape_index = 0
        self._escape_direction = np.asarray(direction, dtype=float).copy()
        self._escape_started_monotonic = time.monotonic()
        self._escape_cancel = threading.Event()
        self._plan = None  # smooth-mode slerp restarts cleanly after the escape
        self._reset_trap_locked()
        self._last_error = ""
        self._last_status = "escape planning; holding last command"
        # Adopt an anticipatory pre-plan when it still matches the frozen
        # direction, so the escape starts executing without a planning pause.
        precomputed = None
        if (
            self._preplan is not None
            and vector_angle(self._preplan[0], self._escape_direction)
            <= self.config.preplan_reuse_rad
        ):
            precomputed = self._preplan[1]
        self._preplan = None
        return (
            self._q[:5].copy(),
            self._escape_direction.copy(),
            self._escape_generation,
            self.config.escape_planning_timeout_s,
            precomputed,
        )

    def _escape_worker(
        self,
        start_q: np.ndarray,
        direction: np.ndarray,
        generation: int,
        timeout_s: float,
        precomputed: Any = None,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        plan_obj = precomputed
        failure_reason = "no manifold or free-space plan within budget"
        # A precomputed anticipatory plan skips both tiers: it was already
        # validated by its planner and is re-validated at publish time.
        # Tier 1: direction-preserving manifold detour. Only converges when the
        # trapped pose sits near the frozen direction's manifold, so large
        # branch switches (e.g. pointing backward past the shoulder_pan limit)
        # usually exhaust it quickly.
        if plan_obj is None and self._escape_planner is not None and not self._escape_cancel.is_set():
            try:
                plan_obj = self._escape_planner.plan(
                    start_q,
                    direction,
                    cancel_event=self._escape_cancel,
                    deadline=deadline,
                )
            except Exception as exc:  # noqa: BLE001 - worker must never crash silently
                logger.exception("escape planning failed")
                failure_reason = f"planner error: {exc}"
                plan_obj = None
        # Tier 2: free joint-space plan (atlas seeds + PlaCo refinement +
        # RRT-Connect) to any branch pointing at the frozen direction. The tip
        # may deviate from the target mid-path; acceptable while the phone is
        # paused, and playback is guarded by the same drift abort as tier 1.
        if plan_obj is None and self._fallback_planner is not None and not self._escape_cancel.is_set():
            try:
                plan_obj = self._fallback_planner.plan(
                    start_q,
                    direction,
                    cancel_event=self._escape_cancel,
                    deadline=deadline,
                )
                if plan_obj is not None and getattr(plan_obj, "method", ""):
                    plan_obj.method = f"global:{plan_obj.method}"
            except Exception as exc:  # noqa: BLE001 - worker must never crash silently
                logger.exception("fallback escape planning failed")
                failure_reason = f"fallback planner error: {exc}"
                plan_obj = None
        with self._lock:
            self._publish_escape_plan_locked(plan_obj, generation, reason=failure_reason)

    def _publish_escape_plan_locked(self, plan_obj: Any, generation: int, reason: str | None = None) -> None:
        """Validate and adopt (or reject) a finished plan. Caller holds ``_lock``."""
        if generation != self._escape_generation or self._escape_state != "planning":
            return  # stale worker result after cancel/re-trigger
        if not self._calibrated or not self._sync_enabled:
            self._cancel_escape_locked("tracking stopped during planning")
            return
        if plan_obj is None:
            self._finish_escape_terminal_locked("failed", reason or "no manifold path within budget")
            return

        waypoints = [np.asarray(item, dtype=float) for item in plan_obj.waypoints_deg]
        invalid_index = next(
            (
                index
                for index, item in enumerate(waypoints)
                if item.shape != (5,) or not np.all(np.isfinite(item)) or not self._target_valid(item)
            ),
            None,
        )
        if invalid_index is not None:
            self._finish_escape_terminal_locked(
                "failed", f"planner waypoint {invalid_index} rejected by runtime validation"
            )
            return
        bad_edge = next(
            (
                first
                for first, (a_edge, b_edge) in enumerate(zip(waypoints, waypoints[1:]))
                if not self._edge_valid(a_edge, b_edge)
            ),
            None,
        )
        if bad_edge is not None:
            self._finish_escape_terminal_locked(
                "failed", f"planner segment {bad_edge} rejected by runtime validation"
            )
            return

        dense = self._resample_escape(waypoints)
        bridge = self._valid_path_suffix(self._q[:5], dense[0])
        if bridge is None:
            self._finish_escape_terminal_locked("failed", "bridge from current pose to escape path is invalid")
            return
        chain = [item.copy() for item in bridge + dense]
        deduped = [chain[0]]
        for item in chain[1:]:
            if not np.allclose(item, deduped[-1], atol=1e-9):
                deduped.append(item)

        self._escape_plan_q = deduped
        self._escape_index = 0
        self._escape_state = "executing"
        self._escape_last_reason = getattr(plan_obj, "method", "planned")
        self._last_error = ""
        self._last_status = f"escape executing ({len(deduped)} waypoints)"
        self._escape_path_payload = (
            generation,
            {
                "type": "escape_path",
                "waypoints_deg": [[float(v) for v in item] for item in deduped],
                "target_direction": self._escape_direction.tolist(),
                "method": self._escape_last_reason,
                "joint_names": ARM_JOINTS,
            },
        )
        logger.info("escape plan adopted: %s", self._escape_last_reason)

    def _resample_escape(self, waypoints: list[np.ndarray]) -> list[np.ndarray]:
        result = [waypoints[0].copy()]
        for first, second in zip(waypoints, waypoints[1:]):
            segment = float(np.max(np.abs(second - first)))
            pieces = max(1, int(math.ceil(segment / self.config.escape_waypoint_delta_deg)))
            for piece in range(1, pieces + 1):
                result.append(first + (second - first) * (piece / pieces))
        return result

    def _tick_escape_execution(self, q: np.ndarray) -> None:
        with self._lock:
            plan_q = self._escape_plan_q
            index = self._escape_index
            frozen = None if self._escape_direction is None else self._escape_direction.copy()
        if plan_q is None or frozen is None:
            return
        while index < len(plan_q) and np.allclose(plan_q[index], q[:5], atol=1e-8):
            index += 1
        if index >= len(plan_q):
            with self._lock:
                self._finish_escape_terminal_locked(
                    "completed", f"completed via {self._escape_last_reason}"
                )
            self._update_tip_state(q[:5])
            return
        live_target = self._live_target_direction_locked_safe()
        if (
            live_target is not None
            and vector_angle(live_target, frozen) > self.config.escape_target_drift_abort_rad
        ):
            # The user redirected the phone mid-escape; holding the old manifold
            # is pointless. Stop within one <=2-degree waypoint.
            with self._lock:
                self._cancel_escape_locked("target moved during escape", terminal=True)
            self._update_tip_state(q[:5])
            return
        command = q.copy()
        command[:5] = plan_q[index]
        sent = self._send(command)
        self._update_tip_state(sent[:5])
        with self._lock:
            self._escape_index = index + 1
            self._target_tip_direction = frozen.copy()
            self._last_error = ""
            self._last_status = f"escape executing ({index + 1}/{len(plan_q)})"

    def _live_target_direction_locked_safe(self) -> np.ndarray | None:
        with self._lock:
            return self._live_target_direction_locked()

    def _housekeep_escape(self, now: float) -> None:
        with self._lock:
            state = self._escape_state
            if state == "failed" or state == "completed":
                if now >= self._escape_next_allowed_time:
                    self._escape_state = "idle"
                return
            if math.isinf(self._escape_next_allowed_time):
                # Escape-loop latch: re-arm once the user asks for a clearly
                # different direction than the looped one.
                history = self._escape_history
                live_target = self._live_target_direction_locked()
                if history and live_target is not None:
                    last_direction = history[-1][1]
                    if vector_angle(live_target, last_direction) > math.radians(2.0):
                        self._escape_next_allowed_time = now

    def _finish_escape_terminal_locked(self, outcome: str, reason: str) -> None:
        self._escape_state = outcome
        self._escape_last_reason = reason
        self._arm_cooldown_locked()
        self._record_escape_outcome_locked(outcome)
        if outcome == "completed":
            self._plan = None
            self._last_status = "escape completed; resuming tracking"
            logger.info("escape completed")
        elif outcome == "aborted":
            self._last_status = f"escape aborted ({reason}); resuming tracking"
            logger.warning("escape aborted: %s", reason)
        else:
            self._last_status = f"escape failed ({reason}); holding position"
            logger.warning("escape failed: %s", reason)
        self._escape_plan_q = None
        self._escape_index = 0

    def _cancel_escape_locked(self, reason: str, terminal: bool = False) -> None:
        """Invalidate any active/planned escape; stale workers publish nothing."""
        self._escape_generation += 1
        self._escape_cancel.set()
        self._escape_plan_q = None
        self._escape_index = 0
        self._escape_direction = None
        self._preplan = None
        if terminal:
            self._escape_state = "failed"
            self._escape_last_reason = reason
            self._arm_cooldown_locked()
            self._record_escape_outcome_locked("aborted")
        else:
            self._escape_state = "idle"
            self._escape_last_reason = reason

    def _arm_cooldown_locked(self) -> None:
        self._escape_next_allowed_time = self._last_tick + self.config.escape_cooldown_s

    def _record_escape_outcome_locked(self, outcome: str) -> None:
        self._escape_outcome_counts[outcome] = self._escape_outcome_counts.get(outcome, 0) + 1
        now = time.monotonic()
        if self._escape_direction is not None:
            self._escape_history.append((now, self._escape_direction.copy()))
        cutoff = now - self.config.escape_loop_window_s
        self._escape_history = [entry for entry in self._escape_history if entry[0] >= cutoff]
        if not self._escape_history:
            return
        reference = self._escape_history[0][1]
        recent_same = sum(
            1
            for _, direction in self._escape_history
            if vector_angle(direction, reference) <= math.radians(2.0)
        )
        if recent_same >= self.config.escape_loop_limit:
            # Latch until the operator intervenes or asks for another direction.
            self._escape_next_allowed_time = math.inf

    def status(self) -> dict[str, Any]:
        with self._lock:
            status_q = self._q
            return {
                "hardware": self.hardware,
                "calibrated": self._calibrated,
                "sync_enabled": self._sync_enabled,
                "mode": self._mode,
                "tracking_backend": "local_ik_lerobot_action",
                "control_frame": self._control_frame,
                "global_planning": False,
                "sim_path_active": self._sim_trajectory is not None,
                "sim_waypoint": self._sim_waypoint_index,
                "sim_waypoint_count": 0 if self._sim_trajectory is None else len(self._sim_trajectory),
                "global_path_execution_allowed": self.global_path_execution_allowed(),
                "phone_fresh": self._latest_phone_quat is not None
                and time.monotonic() - self._latest_phone_time <= self.config.phone_stale_timeout_s,
                "phone_transport_connected": self._phone_transport_connected,
                "phone_stale_age_s": (
                    None
                    if self._latest_phone_quat is None
                    else round(max(0.0, time.monotonic() - self._latest_phone_time), 2)
                ),
                "phone_rebase_count": self._phone_rebase_count,
                "last_phone_rebase_deg": round(self._last_phone_rebase_deg, 1),
                "inside_operational_limits": self._within_limits(status_q[:5]),
                "inside_recorded_calibration_limits": self._within_hard_limits(status_q[:5]),
                "joints_deg": {name: round(float(status_q[i]), 2) for i, name in enumerate(ALL_JOINTS)},
                "planned_joints_deg": {
                    name: round(float(self._q[i]), 2) for i, name in enumerate(ALL_JOINTS)
                },
                "measured_joints_deg": {
                    name: round(float(self._last_measured_q[i]), 2)
                    for i, name in enumerate(ALL_JOINTS)
                },
                "requested_joints_deg": {
                    name: round(float(self._last_requested_command_q[i]), 2)
                    for i, name in enumerate(ALL_JOINTS)
                },
                "sent_joints_deg": {
                    name: round(float(self._last_sent_command_q[i]), 2)
                    for i, name in enumerate(ALL_JOINTS)
                },
                "max_command_tracking_error_deg": round(
                    float(np.max(np.abs(self._last_sent_command_q[:5] - self._last_measured_q[:5]))),
                    2,
                ),
                "phone_quat": None if self._latest_phone_quat is None else self._latest_phone_quat.tolist(),
                "current_tip_direction": (
                    None if self._current_tip_direction is None else self._current_tip_direction.tolist()
                ),
                "target_tip_direction": (
                    None if self._target_tip_direction is None else self._target_tip_direction.tolist()
                ),
                "tip_position_m": None if self._tip_position_m is None else self._tip_position_m.tolist(),
                "tip_error_deg": (
                    None
                    if self._current_tip_direction is None or self._target_tip_direction is None
                    else round(math.degrees(vector_angle(self._current_tip_direction, self._target_tip_direction)), 2)
                ),
                "reject_count": self._reject_count,
                "rejection_counts": self._rejection_counts.copy(),
                "last_rejection": self._last_rejection,
                "last_error": self._last_error,
                "status": self._last_status,
                "escape_enabled": self.config.escape_enabled
                and (self._escape_planner is not None or self._fallback_planner is not None),
                "escape_state": self._escape_state,
                "escape_last_reason": self._escape_last_reason,
                "escape_progress": {
                    "index": self._escape_index,
                    "count": 0 if self._escape_plan_q is None else len(self._escape_plan_q),
                },
                "escape_planning_elapsed_s": (
                    None
                    if self._escape_state != "planning"
                    else round(max(0.0, time.monotonic() - self._escape_started_monotonic), 1)
                ),
                "escape_trap_ticks": self._escape_trap_ticks,
                "escape_outcome_counts": dict(self._escape_outcome_counts),
                "escape_frozen_direction": (
                    None if self._escape_direction is None else self._escape_direction.tolist()
                ),
                "preplan_ready": self._preplan is not None,
                "preplan_worker_active": self._preplan_worker_active,
            }

    def _run(self) -> None:
        period = 1.0 / self.config.fps
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._tick(started)
            except Exception as exc:
                logger.exception("Control loop fault")
                with self._lock:
                    self._sync_enabled = False
                    self._reject_count += 1
                    self._rejection_counts["control_error"] = self._rejection_counts.get("control_error", 0) + 1
                    self._last_rejection = f"control_error: {exc}"
                    self._last_error = f"control error: {exc}"
                    self._last_status = "fault; synchronization disabled"
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    def _tick(self, now: float) -> None:
        dt = 1.0 / self.config.fps
        self._last_tick = now
        self._control_time_s += dt
        self._control_frame += 1
        if self.hardware:
            observation = self.robot.get_observation()
            measured_q = np.array([float(observation[f"{name}.pos"]) for name in ALL_JOINTS])
            with self._lock:
                self._last_measured_q = measured_q.copy()

        with self._lock:
            q = self._q.copy()
            enabled = self._sync_enabled
            latest_q = None if self._latest_phone_quat is None else self._latest_phone_quat.copy()
            latest_time = self._latest_phone_time
            filtered_q = None if self._filtered_phone_quat is None else self._filtered_phone_quat.copy()
            mapping = None if self._phone_to_robot_rotation is None else self._phone_to_robot_rotation.copy()
            mode = self._mode
            sim_trajectory = self._sim_trajectory
            sim_waypoint_index = self._sim_waypoint_index
            escape_state = self._escape_state

        # Publish the actual tip direction every tick, not only while
        # synchronization runs, so the twin's achieved arrow matches the
        # rendered arm pose from the very first status message.
        with self._kinematics_lock:
            current_pose = self.kinematics.forward_kinematics(q[:5])
        current_direction = gripper_tip_in_robot(current_pose[:3, :3])
        with self._lock:
            self._current_tip_direction = current_direction.copy()
            self._tip_position_m = current_pose[:3, 3].copy()

        if sim_trajectory is not None:
            self._tick_simulated_trajectory(q, sim_trajectory, sim_waypoint_index)
            return

        if not enabled:
            return
        if mapping is None:
            with self._lock:
                self._sync_enabled = False
                self._last_error = "missing calibration transform"
            return

        stale_age = math.inf if latest_q is None else now - latest_time
        if stale_age > self.config.phone_stale_timeout_s:
            with self._lock:
                self._last_error = "phone orientation is stale"
                self._last_status = "holding position; waiting for fresh phone data"
            return

        # The filter keeps running in every branch so tracking resumes without a
        # quaternion jump after an escape finishes.
        if filtered_q is None:
            filtered_q = latest_q
        alpha_phone = 1.0 - math.exp(-dt / self.config.phone_filter_tau_s)
        filtered_q = quat_slerp(filtered_q, latest_q, alpha_phone)
        with self._lock:
            self._filtered_phone_quat = filtered_q

        if escape_state == "executing":
            self._tick_escape_execution(q)
            return
        if escape_state == "planning":
            with self._lock:
                elapsed = max(0.0, now - self._escape_started_monotonic)
                self._last_error = ""
                self._last_status = f"escape planning ({elapsed:.1f}s); holding last command"
            return
        self._housekeep_escape(now)
        target_direction = phone_forward_in_robot(mapping, quat_to_matrix(filtered_q))
        with self._lock:
            self._target_tip_direction = target_direction.copy()

        desired_direction = self._next_direction(
            self._control_time_s,
            dt,
            mode,
            current_direction,
            target_direction,
        )
        if vector_angle(desired_direction, current_direction) < 1e-4:
            with self._lock:
                self._last_error = ""
                self._last_status = f"tracking ({mode}); holding calibrated pose"
            return
        solution, reason = self._tip_direction_step(q[:5], desired_direction)
        if reason:
            with self._lock:
                self._reject_count += 1
                category = reason.split(":", 1)[0]
                self._rejection_counts[category] = self._rejection_counts.get(category, 0) + 1
                self._last_rejection = reason
                self._last_error = reason
                self._last_status = "holding last finite command"
                if self._reject_count <= 5 or self._reject_count % 30 == 0:
                    logger.warning("Rejected IK target #%d: %s", self._reject_count, reason)
            spawn_args = self._note_tracking_failure()
            if spawn_args is not None:
                threading.Thread(
                    target=self._escape_worker, args=spawn_args, name="escape-planner", daemon=True
                ).start()
            return

        command = q.copy()
        command[:5] = solution
        sent = self._send(command)
        self._update_tip_state(sent[:5])
        with self._lock:
            achieved = (
                None
                if self._current_tip_direction is None
                else self._current_tip_direction.copy()
            )
            self._last_error = ""
            self._last_status = f"tracking ({mode}) via LeRobot action"
        if achieved is not None:
            spawn_args = self._note_tracking_success(vector_angle(achieved, desired_direction))
            if spawn_args is not None:
                threading.Thread(
                    target=self._escape_worker, args=spawn_args, name="escape-planner", daemon=True
                ).start()

    def _update_tip_state(self, joints_deg: np.ndarray) -> None:
        with self._kinematics_lock:
            pose = self.kinematics.forward_kinematics(joints_deg)
        with self._lock:
            self._current_tip_direction = gripper_tip_in_robot(pose[:3, :3])
            self._tip_position_m = pose[:3, 3].copy()

    def _tick_simulated_trajectory(
        self,
        q: np.ndarray,
        trajectory: list[np.ndarray],
        waypoint_index: int,
    ) -> None:
        while waypoint_index < len(trajectory) and np.allclose(
            trajectory[waypoint_index], q[:5], atol=1e-8
        ):
            waypoint_index += 1
        if waypoint_index >= len(trajectory):
            with self._lock:
                self._sim_trajectory = None
                self._sim_waypoint_index = len(trajectory)
                self._last_error = ""
                self._last_status = "global path complete; holding position"
            self._update_tip_state(q[:5])
            return

        target = trajectory[waypoint_index]
        command = q.copy()
        command[:5] = target
        sent = self._send(command)
        self._update_tip_state(sent[:5])
        with self._lock:
            self._sim_waypoint_index = waypoint_index + 1
            self._target_tip_direction = (
                None if self._sim_target_direction is None else self._sim_target_direction.copy()
            )
            self._last_error = ""
            self._last_status = f"playing global path ({waypoint_index + 1}/{len(trajectory)})"

    def _next_direction(
        self,
        now: float,
        dt: float,
        mode: TrackingMode,
        current_direction: np.ndarray,
        target_direction: np.ndarray,
    ) -> np.ndarray:
        if mode == "realtime":
            alpha = 1.0 - math.exp(-dt / self.config.realtime_follow_tau_s)
            return vector_slerp(current_direction, target_direction, alpha)

        with self._lock:
            plan = self._plan
        should_replan = plan is None
        if plan is not None:
            should_replan = (
                vector_angle(target_direction, plan.target_direction) > self.config.target_angle_threshold_rad
            )
        if should_replan:
            angle = vector_angle(current_direction, target_direction)
            duration = max(angle / self.config.angular_speed_rad_s, 1.0 / self.config.fps)
            plan = DirectionPlan(
                current_direction.copy(), target_direction.copy(), now, duration
            )
            with self._lock:
                self._plan = plan
        fraction = (now - plan.started_at) / plan.duration_s
        return vector_slerp(plan.start_direction, plan.target_direction, fraction)

    def _tip_direction_step(
        self, current: np.ndarray, desired_direction: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Track only the visual gripper-tip axis; roll around it is free."""
        with self._kinematics_lock:
            lower_delta, upper_delta = self._joint_delta_bounds(current)
            return direction_tip_step(
                self.kinematics.forward_kinematics,
                current,
                desired_direction,
                lower_delta_rad=lower_delta,
                upper_delta_rad=upper_delta,
                damping=self.config.differential_ik_damping,
                jacobian_epsilon_deg=self.config.jacobian_epsilon_deg,
                max_step_rad=self.config.max_orientation_ik_step_rad,
                valid=self._target_valid,
                edge_valid=self._edge_valid,
            )

    def _joint_delta_bounds(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.joint_limits[name][0] for name in ARM_JOINTS]) - current
        upper = np.array([self.joint_limits[name][1] for name in ARM_JOINTS]) - current
        return np.deg2rad(lower), np.deg2rad(upper)

    @staticmethod
    def _bounded_damped_least_squares(
        jacobian: np.ndarray,
        error: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        return bounded_damped_least_squares(jacobian, error, lower, upper, damping)

    def _within_limits(self, joints: np.ndarray) -> bool:
        return all(
            self.joint_limits[name][0] <= joints[i] <= self.joint_limits[name][1]
            for i, name in enumerate(ARM_JOINTS)
        )

    def _within_hard_limits(self, joints: np.ndarray) -> bool:
        return all(
            self.hard_joint_limits[name][0] <= joints[i] <= self.hard_joint_limits[name][1]
            for i, name in enumerate(ARM_JOINTS)
        )

    def _target_valid(self, joints: np.ndarray) -> bool:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (5,) or not np.all(np.isfinite(joints)) or not self._within_limits(joints):
            return False
        with self._validation_lock:
            return self._state_validator is None or self._state_validator(joints)

    def _edge_valid(self, first: np.ndarray, second: np.ndarray) -> bool:
        max_delta = float(np.max(np.abs(second - first)))
        steps = max(1, int(math.ceil(max_delta / self.config.path_validation_resolution_deg)))
        return all(
            self._target_valid(first + (second - first) * (index / steps))
            for index in range(1, steps + 1)
        )

    def _valid_path_suffix(self, current: np.ndarray, target: np.ndarray) -> list[np.ndarray] | None:
        """Return the valid suffix of a straight joint path to a valid target.

        An invalid measured start cannot mathematically belong to an all-valid
        path.  In that case only the suffix after the final invalid sample is
        returned; every commanded waypoint is still range- and collision-valid.
        """
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        if not self._target_valid(target):
            return None
        max_delta = float(np.max(np.abs(target - current)))
        steps = max(1, int(math.ceil(max_delta / self.config.path_validation_resolution_deg)))
        samples = [current + (target - current) * (index / steps) for index in range(steps + 1)]
        last_invalid = max(
            (index for index, sample in enumerate(samples) if not self._target_valid(sample)),
            default=-1,
        )
        return [sample.copy() for sample in samples[last_invalid + 1 :]]

    def _send(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=float)
        if command.shape != (6,) or not np.all(np.isfinite(command)):
            raise ValueError("refusing malformed or non-finite joint command")
        if not self._target_valid(command[:5]):
            raise ValueError("refusing out-of-calibration or self-colliding joint target")
        # LeRobot exposes the calibrated gripper range as 0..100 rather than degrees.
        if not 0.0 <= command[5] <= 100.0:
            raise ValueError("refusing out-of-calibration gripper target")
        action = {f"{name}.pos": float(command[i]) for i, name in enumerate(ALL_JOINTS)}
        sent_action = action
        if self.hardware:
            sent_action = self.robot.send_action(action)
        sent_q = np.array(
            [float(sent_action.get(f"{name}.pos", command[index])) for index, name in enumerate(ALL_JOINTS)]
        )
        with self._lock:
            self._last_requested_command_q = command.copy()
            self._last_sent_command_q = sent_q
            self._q = sent_q.copy()
        return sent_q
