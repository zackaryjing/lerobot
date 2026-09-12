#!/usr/bin/env python
"""Reproduce the minimal failing case: phone forward -> slow 180 deg clockwise
rotation with pauses. Diagnosis harness (not a unit test).

Stage A probes the escape planner directly from a tracker-trapped pose toward
azimuths beyond the shoulder_pan limit.
Stage B runs the full controller loop with real-time pacing, exactly the
user scenario.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller import (
    ARM_JOINTS,
    ControlConfig,
    OrientationController,
    load_joint_limits,
    load_calibration_limits,
    vector_angle,
)
from direction_atlas import SO101StateValidator
from escape_planner import EscapePlanner, EscapePlannerConfig
from lerobot.model.kinematics import RobotKinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URDF = ROOT / "SO101/so101_new_calib.urdf"
COLLISIONS = ROOT / "SO101/collisions.json"
CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
RESET = HERE / "reset_pose_myfollower01.json"


def azimuth(direction: np.ndarray) -> float:
    """Angle of the direction projected on the robot X-Y plane, in degrees."""
    d = direction / np.linalg.norm(direction)
    return math.degrees(math.atan2(d[1], d[0]))


def make_controller(escape_planner=None, fallback_planner=None):
    limits = load_joint_limits(URDF, CALIBRATION)
    hard_limits = load_calibration_limits(CALIBRATION)
    import json

    reset_data = json.loads(RESET.read_text())
    reset_joints = [float(reset_data["joints_deg"][name]) for name in
                    ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
    kinematics = RobotKinematics(str(URDF), target_frame_name="gripper_frame_link", joint_names=ARM_JOINTS)
    validator = SO101StateValidator(
        URDF, COLLISIONS, limits, min_tip_height_m=-math.inf, min_moving_frame_height_m=-math.inf
    )
    controller = OrientationController(
        kinematics,
        limits,
        config=ControlConfig(),
        hard_joint_limits=hard_limits,
        reset_joints=np.asarray(reset_joints),
        state_validator=lambda joints: validator.evaluate(joints)[0] is not None,
        escape_planner=escape_planner,
        fallback_planner=fallback_planner,
    )
    return controller, validator, limits


def run_ticks(controller, count, dt=1.0 / 30.0):
    for _ in range(count):
        now = controller._last_tick + dt
        controller._latest_phone_time = now
        controller._tick(now)


def phone_yaw_quat(theta_deg: float) -> np.ndarray:
    """Phone quaternion for a rotation about its +Z (up, screen-up) axis."""
    theta = math.radians(theta_deg)
    return np.array([0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)])


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    if stage == "A":
        return stage_a()
    return stage_b()


def stage_a() -> int:
    planner = EscapePlanner(SO101StateValidator(
        URDF, COLLISIONS,
        load_joint_limits(URDF, CALIBRATION),
        min_tip_height_m=-math.inf, min_moving_frame_height_m=-math.inf,
    ))
    controller, _, limits = make_controller(escape_planner=planner)

    print("limits:", {name: (round(limits[name][0], 1), round(limits[name][1], 1)) for name in ARM_JOINTS})

    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]
    run_ticks(controller, 5)

    start_direction = controller.status()["current_tip_direction"]
    print(f"start: tip azimuth {azimuth(start_direction):7.1f} deg, q={np.round(controller.current_arm_joints(), 1)}")

    # Drive the phone in one big jump to -150 deg azimuth and let the greedy
    # tracker stall against the shoulder_pan limit.
    controller._latest_phone_quat = phone_yaw_quat(-150.0)
    controller._filtered_phone_quat = controller._latest_phone_quat.copy()
    run_ticks(controller, 600)
    trapped_q = controller.current_arm_joints().copy()
    trapped_direction = controller.status()["current_tip_direction"]
    target_direction = controller.status()["target_tip_direction"]
    err = math.degrees(vector_angle(trapped_direction, target_direction))
    print(f"trapped: tip azimuth {azimuth(trapped_direction):7.1f} deg, err {err:5.1f} deg, "
          f"target azimuth {azimuth(target_direction):7.1f}")
    print(f"trapped q = {np.round(trapped_q, 1)}")
    print(f"trapped margins = {np.round(np.minimum(trapped_q - np.array([limits[n][0] for n in ARM_JOINTS]), np.array([limits[n][1] for n in ARM_JOINTS]) - trapped_q), 1)}")

    # Probe the escape planner at several azimuths beyond the limit.
    for theta in (-130.0, -150.0, -170.0, -180.0):
        target = controller._phone_to_robot_rotation @ quat_to_matrix(phone_yaw_quat(theta)) @ np.array([0.0, 1.0, 0.0])
        target /= np.linalg.norm(target)
        started = time.monotonic()
        plan_obj = planner.plan(trapped_q, target)
        elapsed = time.monotonic() - started
        if plan_obj is None:
            print(f"escape to {theta:6.1f} deg: FAILED after {elapsed:5.1f}s")
            continue
        print(f"escape to {theta:6.1f} deg: OK method={plan_obj.method:7s} "
              f"{len(plan_obj.waypoints_deg):3d} waypoints, {elapsed:5.1f}s, "
              f"goal q={np.round(np.asarray(plan_obj.goal.joints_deg), 1)}, "
              f"goal err={plan_obj.goal.direction_error_deg:5.2f} deg")
    return 0


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    from controller import quat_to_matrix as qtm
    return qtm(q)


def stage_b() -> int:
    # Full scenario with real-time pacing. Escape planner runs in its worker
    # thread, so ticks must not run faster than real time.
    from global_planner import GlobalDirectionPlanner

    limits = load_joint_limits(URDF, CALIBRATION)
    planner = EscapePlanner(SO101StateValidator(
        URDF, COLLISIONS, limits,
        min_tip_height_m=-math.inf, min_moving_frame_height_m=-math.inf,
    ))
    fallback = GlobalDirectionPlanner(HERE / "direction_atlas.json", SO101StateValidator(
        URDF, COLLISIONS, limits,
        min_tip_height_m=-math.inf, min_moving_frame_height_m=-math.inf,
    ))
    original_plan = fallback.plan

    def traced_plan(start_q, direction, cancel_event=None, deadline=None):
        started = time.monotonic()
        result = original_plan(start_q, direction, cancel_event=cancel_event, deadline=deadline)
        print(f"  [fallback] start_q={np.round(start_q, 1)} target_az="
              f"{azimuth(direction):7.1f} -> {'OK ' + result.method if result else 'FAIL'} "
              f"({time.monotonic() - started:.1f}s)")
        return result

    fallback.plan = traced_plan
    controller, _, _ = make_controller(escape_planner=planner, fallback_planner=fallback)
    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]

    fps = controller.config.fps
    dt = 1.0 / fps
    segments = [(0.0, -45.0, 3.0), (-45.0, -90.0, 3.0), (-90.0, -135.0, 3.0),
                (-135.0, -180.0, 3.0)]  # (from, to, seconds)

    print(f"{'t(s)':>6} {'phone':>7} {'tip_az':>7} {'err':>6} {'escape_state':>12} {'trap':>4} {'reason'}")
    start_wall = time.monotonic()
    for lo, hi, seconds in segments:
        frames = int(seconds * fps)
        for frame in range(frames):
            frac = frame / frames
            theta = lo + (hi - lo) * frac
            controller.submit_phone_orientation(phone_yaw_quat(theta).tolist())
            now = controller._last_tick + dt
            controller._latest_phone_time = time.monotonic()
            controller._tick(now)
            status = controller.status()
            if frame % 10 == 0 or status["escape_state"] != "idle":
                print(f"{now:6.1f} {theta:7.1f} {azimuth(status['current_tip_direction']):7.1f} "
                      f"{status['tip_error_deg']:6.1f} {status['escape_state']:>12} "
                      f"{status['escape_trap_ticks']:4d} {status['escape_last_reason'][:30]}")
            time.sleep(dt)
        # Pause: hold the phone still for 4 seconds.
        for _ in range(int(4.0 * fps)):
            controller.submit_phone_orientation(phone_yaw_quat(hi).tolist())
            now = controller._last_tick + dt
            controller._latest_phone_time = time.monotonic()
            controller._tick(now)
            status = controller.status()
            if status["escape_state"] != "idle":
                print(f"{now:6.1f} {hi:7.1f} {azimuth(status['current_tip_direction']):7.1f} "
                      f"{status['tip_error_deg']:6.1f} {status['escape_state']:>12} "
                      f"{status['escape_trap_ticks']:4d} {status['escape_last_reason'][:30]}")
            time.sleep(dt)

    status = controller.status()
    current = status["current_tip_direction"]
    print(f"\nFINAL: phone -180.0, tip azimuth {azimuth(current):7.1f}, err {status['tip_error_deg']:5.1f} deg")
    print(f"outcomes: {status['escape_outcome_counts']}")
    print(f"wall time: {time.monotonic() - start_wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
