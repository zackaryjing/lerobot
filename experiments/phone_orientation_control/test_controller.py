#!/usr/bin/env python

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from controller import (
    ALL_JOINTS,
    ARM_JOINTS,
    ControlConfig,
    OrientationController,
    RobotKinematics,
    load_calibration_limits,
    matrix_to_quat,
    normalize_quat,
    quat_angle,
    quat_slerp,
    quat_to_matrix,
    vector_angle,
    load_joint_limits,
)
from escape_planner import EscapePlanner, EscapePlannerConfig
from frame_adapter import GRIPPER_TIP_LOCAL, PHONE_FORWARD_LOCAL
from direction_atlas import AtlasConfig, DirectionAtlasBuilder, SO101StateValidator, fibonacci_sphere
from global_planner import GlobalDirectionPlanner, GlobalPlannerConfig
from record_direction_samples import ManualSampleStore


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def test_quaternion_roundtrip_and_slerp():
    q = normalize_quat(np.array([0.2, -0.3, 0.1, 0.9]))
    roundtrip = matrix_to_quat(quat_to_matrix(q))
    assert quat_angle(q, roundtrip) < 1e-7
    halfway = quat_slerp(np.array([0, 0, 0, 1.0]), np.array([0, 0, 1.0, 0]), 0.5)
    assert np.isclose(quat_angle(np.array([0, 0, 0, 1.0]), halfway), np.pi / 2)


def test_joint_limits_match_exact_calibration_range_without_margin():
    limits = load_joint_limits(
        ROOT / "SO101/so101_new_calib.urdf",
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    )
    assert set(limits) == set(ARM_JOINTS)
    assert all(low < 0 < high for low, high in limits.values())
    assert limits["shoulder_pan"] == (-104.0, 104.0)
    assert np.isclose(limits["shoulder_lift"][1], 104.65934065934066)
    assert np.isclose(limits["elbow_flex"][1], 97.18681318681318)


def test_configured_reset_is_inside_exact_calibration_and_self_collision_free():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf",
        ROOT / "SO101/collisions.json",
        limits,
        min_tip_height_m=-np.inf,
        min_moving_frame_height_m=-np.inf,
    )
    reset_data = json.loads((HERE / "reset_pose_myfollower01.json").read_text())
    reset = np.array([reset_data["joints_deg"][name] for name in ARM_JOINTS])

    candidate, reason = validator.evaluate(reset)

    assert candidate is not None, reason
    assert all(limits[name][0] <= reset[index] <= limits[name][1] for index, name in enumerate(ARM_JOINTS))
    assert validator.collision_names() == []


def test_ik_updates_kinematics_before_first_solve():
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    current = np.array([7.824, -103.868, 96.835, 66.769, -1.978])
    desired = kinematics.forward_kinematics(current)
    solution = kinematics.inverse_kinematics(current, desired, position_weight=1.0, orientation_weight=1.0)
    reached = kinematics.forward_kinematics(solution)
    assert np.linalg.norm(reached[:3, 3] - desired[:3, 3]) < 0.005


def test_action_target_must_stay_inside_exact_calibration_limits():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, hard_joint_limits=hard_limits)
    outside = np.array([150.0, -150.0, 150.0, -150.0, 150.0, 50.0])
    assert not controller._within_limits(outside[:5])
    with pytest.raises(ValueError, match="out-of-calibration"):
        controller._send(outside)
    with pytest.raises(ValueError, match="gripper"):
        controller._send(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.01]))


def test_tip_direction_step_reduces_error_without_position_or_roll_target():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config, hard_joint_limits=hard_limits)
    current = np.array([7.824, -103.868, 96.835, 66.769, -1.978])
    current_rotation = kinematics.forward_kinematics(current)[:3, :3]
    angle = np.deg2rad(8.0)
    rotation_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    current_direction = current_rotation @ GRIPPER_TIP_LOCAL
    desired_direction = rotation_z @ current_direction

    solution, reason = controller._tip_direction_step(current, desired_direction)
    reached_direction = kinematics.forward_kinematics(solution)[:3, :3] @ GRIPPER_TIP_LOCAL

    assert reason == ""
    assert vector_angle(reached_direction, desired_direction) < vector_angle(current_direction, desired_direction)
    assert not np.allclose(solution, current)


def test_orientation_ik_stays_inside_exact_calibration_limits():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config, hard_joint_limits=hard_limits)
    current = np.array([7.824, -103.868, 96.835, 66.769, -1.978])
    current_rotation = kinematics.forward_kinematics(current)[:3, :3]

    for axis in np.eye(3):
        for sign in (-1.0, 1.0):
            angle = sign * np.deg2rad(3.0)
            cross = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
            )
            offset = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
            desired_direction = offset @ current_rotation @ GRIPPER_TIP_LOCAL
            solution, reason = controller._tip_direction_step(current, desired_direction)
            assert reason == ""
            assert controller._within_limits(solution)


def test_calibration_maps_phone_top_to_midpoint_gripper_tip():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config)
    phone_quat = normalize_quat(np.array([0.2, -0.3, 0.1, 0.9]))
    controller.submit_phone_orientation(phone_quat.tolist())

    ok, reason = controller.calibrate()

    assert ok, reason
    mapped_phone_top = controller._phone_to_robot_rotation @ quat_to_matrix(phone_quat) @ PHONE_FORWARD_LOCAL
    midpoint_tip = kinematics.forward_kinematics(np.zeros(5))[:3, :3] @ GRIPPER_TIP_LOCAL
    assert vector_angle(mapped_phone_top, midpoint_tip) < 1e-7


def test_fibonacci_sphere_is_unit_length_and_spans_both_hemispheres():
    directions = fibonacci_sphere(64)
    assert directions.shape == (64, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[:, 2].min() < -0.9
    assert directions[:, 2].max() > 0.9


def test_state_validator_allows_midpoint_and_rejects_known_self_collision():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf", ROOT / "SO101/collisions.json", limits
    )
    midpoint, reason = validator.evaluate(np.zeros(5))
    assert midpoint is not None, reason
    collision, reason = validator.evaluate(np.array([-69.183, 41.502, 81.727, -15.028, 53.576]))
    assert collision is None
    assert reason == "self_collision"


def test_small_direction_atlas_retains_multiple_safe_candidates():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf", ROOT / "SO101/collisions.json", limits
    )
    atlas = DirectionAtlasBuilder(
        validator, AtlasConfig(samples=300, direction_bins=32, candidates_per_bin=3, seed=7)
    ).build()
    assert atlas["stats"]["valid"] > 100
    assert atlas["stats"]["occupied_bins"] > 10
    assert atlas["stats"]["retained_candidates"] >= atlas["stats"]["occupied_bins"]


def test_global_planner_finds_multiple_branches_and_safe_path_to_back_direction():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf", ROOT / "SO101/collisions.json", limits
    )
    planner = GlobalDirectionPlanner(
        HERE / "direction_atlas.json",
        validator,
        GlobalPlannerConfig(refined_candidates=5, goals_to_plan=4, rrt_iterations_per_goal=500),
    )
    current = np.array([0.0, 30.0, -60.0, -30.0, 0.0])
    target = np.array([-1.0, 0.0, 0.0])

    goals = planner.find_goals(target, current)
    result = planner.plan(current, target)

    assert len(goals) >= 2
    assert all(goal.direction_error_deg <= 0.2 for goal in goals)
    assert result is not None
    assert result.goal.direction_error_deg <= 0.2
    assert len(result.waypoints_deg) >= 2
    assert all(validator.evaluate(np.asarray(waypoint))[0] is not None for waypoint in result.waypoints_deg)

    already_forward = planner.plan(np.zeros(5), np.array([1.0, 0.0, 0.0]))
    assert already_forward is not None
    assert np.allclose(already_forward.goal.joints_deg, np.zeros(5), atol=1e-6)
    assert already_forward.normalized_length == 0.0


def test_manual_sample_store_appends_and_undoes_atomically(tmp_path):
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf", ROOT / "SO101/collisions.json", limits
    )
    candidate, reason = validator.evaluate(np.zeros(5))
    assert candidate is not None, reason
    output = tmp_path / "samples.json"

    store = ManualSampleStore(output)
    store.add(candidate, 42.0)
    reloaded = ManualSampleStore(output)

    assert reloaded.count == 1
    assert reloaded.payload["samples"][0]["gripper_deg"] == 42.0
    assert reloaded.undo()
    assert ManualSampleStore(output).count == 0


def test_dry_run_previews_global_path_waypoints_to_completion():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config)
    start = controller.current_arm_joints()
    goal = start + np.array([4.0, -3.0, 2.0, 1.0, -2.0])
    ok, reason = controller.set_simulated_trajectory(
        [start.tolist(), goal.tolist()], [1.0, 0.0, 0.0]
    )
    assert ok, reason

    now = controller._last_tick
    for _ in range(10):
        now += 1.0 / config.fps
        controller._tick(now)
        if not controller.status()["sim_path_active"]:
            break

    status = controller.status()
    assert not status["sim_path_active"]
    assert np.max(np.abs(controller.current_arm_joints() - goal)) < 0.08
    assert status["current_tip_direction"] is not None


def test_hardware_and_dry_run_accept_the_same_finite_waypoints():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    start = np.array([0.0, 30.0, -60.0, -30.0, 0.0])
    goal = start + np.array([2.0, 0.0, 0.0, 0.0, 0.0])

    locked = OrientationController(kinematics, limits, robot=object())  # type: ignore[arg-type]
    ok, reason = locked.set_simulated_trajectory([start.tolist(), goal.tolist()], [1.0, 0.0, 0.0])
    assert ok, reason


def test_sync_does_not_move_before_the_first_planned_ik_step():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config, hard_joint_limits=hard_limits)
    folded = np.array([7.824, -103.868, 89.835, 66.769, -1.978, 1.568])
    controller._q = folded.copy()
    controller._last_measured_q = folded.copy()
    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]

    assert np.allclose(controller.current_arm_joints(), folded[:5])
    assert controller.status()["tracking_backend"] == "local_ik_lerobot_action"


def test_local_following_reduces_folded_pose_direction_error_without_hidden_path():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config, hard_joint_limits=hard_limits)
    folded = np.array([7.824, -103.868, 89.835, 66.769, -1.978, 1.568])
    controller._q = folded.copy()
    controller._last_measured_q = folded.copy()
    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_mode("realtime")[0]
    assert controller.set_sync(True)[0]

    target = kinematics.forward_kinematics(np.zeros(5))[:3, :3] @ GRIPPER_TIP_LOCAL
    before = kinematics.forward_kinematics(folded[:5])[:3, :3] @ GRIPPER_TIP_LOCAL
    before_error = vector_angle(before, target)
    now = controller._last_tick
    for _ in range(90):
        now += 1.0 / config.fps
        controller._latest_phone_time = now
        controller._tick(now)

    after_q = controller.current_arm_joints()
    after = kinematics.forward_kinematics(after_q)[:3, :3] @ GRIPPER_TIP_LOCAL
    assert vector_angle(after, target) < before_error - np.deg2rad(20.0)
    assert not controller.status()["sim_path_active"]
    assert controller.status()["tracking_backend"] == "local_ik_lerobot_action"


def test_phone_link_loss_holds_immediately_and_keeps_sync_armed():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config)
    current = controller.current_arm_joints()
    direction = kinematics.forward_kinematics(current)[:3, :3] @ GRIPPER_TIP_LOCAL
    angle = np.deg2rad(10.0)
    rotation_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    now = controller._last_tick
    controller._sync_enabled = True
    controller._phone_to_robot_rotation = np.eye(3)
    controller._latest_phone_quat = np.array([0.0, 0.0, 0.0, 1.0])
    controller._latest_phone_time = now - 1.0
    controller._target_tip_direction = rotation_z @ direction
    before = controller.current_arm_joints()

    controller._tick(now + 1.0 / config.fps)
    assert np.allclose(controller.current_arm_joints(), before)
    assert controller.status()["sync_enabled"]
    assert "waiting for fresh phone data" in controller.status()["status"]


def test_new_browser_session_does_not_cancel_startup_or_reset_path():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits)
    controller._sim_trajectory = [np.zeros(5)]

    controller.invalidate_calibration()

    assert controller._sim_trajectory is not None
    assert len(controller._sim_trajectory) == 1
    assert not controller.status()["calibrated"]


def test_reconnect_large_phone_jump_rebases_without_target_jump():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    controller.submit_phone_orientation(identity.tolist())
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]
    old_direction = (
        controller._phone_to_robot_rotation @ quat_to_matrix(identity) @ PHONE_FORWARD_LOCAL
    )

    controller.set_phone_transport_connected(False)
    controller.set_phone_transport_connected(True)
    angle = np.deg2rad(60.0)
    reconnected = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
    controller.submit_phone_orientation(reconnected.tolist())
    rebased_direction = (
        controller._phone_to_robot_rotation
        @ quat_to_matrix(reconnected)
        @ PHONE_FORWARD_LOCAL
    )

    assert vector_angle(old_direction, rebased_direction) < 1e-7
    assert controller.status()["reject_count"] == 0
    assert controller.status()["phone_rebase_count"] == 1
    assert controller.status()["last_phone_rebase_deg"] == 60.0


def test_invalid_measured_start_recovers_through_only_valid_command_targets():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf",
        ROOT / "SO101/collisions.json",
        limits,
        min_tip_height_m=-np.inf,
        min_moving_frame_height_m=-np.inf,
    )
    old_self_colliding_reset = np.array([7.824, -103.868, 96.835, 66.769, -1.978])
    reset_data = json.loads((HERE / "reset_pose_myfollower01.json").read_text())
    valid_reset = np.array([reset_data["joints_deg"][name] for name in ARM_JOINTS])
    controller = OrientationController(
        kinematics,
        limits,
        state_validator=lambda joints: validator.evaluate(joints)[0] is not None,
    )
    path = controller._valid_path_suffix(old_self_colliding_reset, valid_reset)

    assert path
    assert np.allclose(path[-1], valid_reset)
    assert all(validator.evaluate(waypoint)[0] is not None for waypoint in path)
    assert all(controller._within_limits(waypoint) for waypoint in path)


def test_hardware_action_is_passed_to_lerobot_without_custom_joint_controller():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    measured = np.array([0.0, 10.0, -20.0, -10.0, 0.0, 25.0])

    class StaticRobot:
        is_connected = True

        def __init__(self):
            self.actions = []

        def get_observation(self):
            return {f"{name}.pos": measured[index] for index, name in enumerate(ALL_JOINTS)}

        def send_action(self, action):
            self.actions.append(action.copy())
            return action

        def disconnect(self):
            self.is_connected = False

    robot = StaticRobot()
    controller = OrientationController(
        kinematics,
        limits,
        robot=robot,
        config=config,
        hard_joint_limits=hard_limits,
    )
    controller._q = measured.copy()
    command = measured + np.array([1.25, -0.75, 0.5, 1.0, -1.5, 0.0])
    controller._send(command)

    assert len(robot.actions) == 1
    actual = np.array([robot.actions[0][f"{name}.pos"] for name in ALL_JOINTS])
    assert np.allclose(actual, command)
    assert np.allclose(controller._last_requested_command_q, command)
    assert np.allclose(controller._last_sent_command_q, command)


def test_dry_run_and_hardware_generate_identical_actions_despite_hardware_feedback():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    urdf = str(ROOT / "SO101/so101_new_calib.urdf")
    initial = np.array([7.824, -103.868, 89.835, 66.769, -1.978, 1.568])
    static_feedback = np.array([-20.0, 40.0, -30.0, 15.0, 10.0, 1.568])

    class StaticRobot:
        is_connected = True

        def __init__(self):
            self.actions = []

        def get_observation(self):
            return {
                f"{name}.pos": static_feedback[index]
                for index, name in enumerate(ALL_JOINTS)
            }

        def send_action(self, action):
            self.actions.append(action.copy())
            return action

        def disconnect(self):
            self.is_connected = False

    robot = StaticRobot()
    dry = OrientationController(
        RobotKinematics(urdf, joint_names=ARM_JOINTS),
        limits,
        config=config,
        hard_joint_limits=hard_limits,
        dry_initial_joints=initial,
        reset_joints=initial,
    )
    hardware = OrientationController(
        RobotKinematics(urdf, joint_names=ARM_JOINTS),
        limits,
        robot=robot,
        config=config,
        hard_joint_limits=hard_limits,
        dry_initial_joints=initial,
        reset_joints=initial,
    )
    for controller in (dry, hardware):
        controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
        assert controller.calibrate()[0]
        assert controller.set_mode("smooth")[0]
        assert controller.set_sync(True)[0]
        controller._last_tick = 0.0

    dry_now = 0.0
    hardware_now = 0.0
    for frame in range(45):
        dry_now += 1.0 / config.fps
        hardware_now += (0.6 if frame % 2 == 0 else 1.4) / config.fps
        dry._latest_phone_time = dry_now
        hardware._latest_phone_time = hardware_now
        dry._tick(dry_now)
        hardware._tick(hardware_now)
        assert np.allclose(hardware.current_arm_joints(), dry.current_arm_joints(), atol=1e-10)
        if robot.actions:
            action = np.array([robot.actions[-1][f"{name}.pos"] for name in ALL_JOINTS])
            assert np.allclose(action, dry._last_sent_command_q, atol=1e-10)

    assert np.allclose(
        list(hardware.status()["joints_deg"].values()),
        list(dry.status()["joints_deg"].values()),
    )
    assert not np.allclose(hardware._last_measured_q[:5], hardware.current_arm_joints())


def test_phone_sync_uses_local_ik_without_global_path_playback():
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig()
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(kinematics, limits, config=config)
    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]
    before = controller.current_arm_joints()
    now = controller._last_tick + 1.0 / config.fps
    controller._tick(now)
    controller._tick(now + 1.0 / config.fps)

    status = controller.status()
    assert not np.allclose(controller.current_arm_joints(), before)
    assert status["sync_enabled"]
    assert not status["sim_path_active"]
    assert not status["global_planning"]
    assert status["tracking_backend"] == "local_ik_lerobot_action"


# --------------------------------------------------------------------------
# Direction-preserving escape planner
# --------------------------------------------------------------------------

FOLDED = np.array([7.824, -103.868, 89.835, 66.769, -1.978])
# Joint-safe step from the folded pose: shoulder_lift has only ~0.75 degrees of
# negative headroom against the real calibration, so it must move positive.
ESC_DELTA = np.array([2.0, 0.6, 1.5, -2.0, 2.0])


def _make_escape_planner(**overrides):
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    validator = SO101StateValidator(
        ROOT / "SO101/so101_new_calib.urdf",
        ROOT / "SO101/collisions.json",
        limits,
        min_tip_height_m=-np.inf,
        min_moving_frame_height_m=-np.inf,
    )
    return EscapePlanner(validator, EscapePlannerConfig(**overrides)), validator, limits


class FakeEscapePlanner:
    """Deterministic planner stub returning a fixed waypoint list."""

    def __init__(self, waypoints_deg, method="direct"):
        self.waypoints_deg = waypoints_deg
        self.method = method
        self.calls = []

    def plan(self, start_joints_deg, target_direction, cancel_event=None, deadline=None):
        self.calls.append((np.asarray(start_joints_deg).copy(), np.asarray(target_direction).copy()))
        return SimpleNamespace(
            waypoints_deg=[list(map(float, waypoint)) for waypoint in self.waypoints_deg],
            target_direction=list(map(float, target_direction)),
            goal=SimpleNamespace(direction_error_deg=0.0),
            method=self.method,
            planning_time_s=0.0,
            tree_nodes=0,
            projections=0,
        )


class FailingEscapePlanner:
    def plan(self, start_joints_deg, target_direction, cancel_event=None, deadline=None):
        return None


def _escape_controller(planner, robot=None, **config_overrides):
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
    config = ControlConfig(**config_overrides)
    limits = load_joint_limits(ROOT / "SO101/so101_new_calib.urdf", calibration)
    hard_limits = load_calibration_limits(calibration)
    kinematics = RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS)
    controller = OrientationController(
        kinematics,
        limits,
        robot=robot,
        config=config,
        hard_joint_limits=hard_limits,
        dry_initial_joints=np.array([*FOLDED, 1.568]),
        escape_planner=planner,
    )
    controller.submit_phone_orientation([0.0, 0.0, 0.0, 1.0])
    assert controller.calibrate()[0]
    assert controller.set_sync(True)[0]
    return controller


def test_escape_planner_projects_perturbed_configs_into_direction_cone():
    planner, validator, limits = _make_escape_planner()
    base = np.array([0.0, 30.0, -60.0, -30.0, 0.0])
    direction = planner.tip_direction(base)
    rng = np.random.default_rng(7)

    checked = 0
    valid = 0
    for _ in range(8):
        perturbed = np.clip(base + rng.uniform(-8.0, 8.0, size=5), [limits[n][0] for n in ARM_JOINTS],
                            [limits[n][1] for n in ARM_JOINTS])
        projected = planner.project(perturbed, direction)
        if projected is None:
            continue
        checked += 1
        assert planner.direction_error_deg(projected, direction) <= math.degrees(
            EscapePlannerConfig().direction_tolerance_rad
        )
        if validator.evaluate(projected)[0] is not None:
            valid += 1
    assert checked >= 5, "projection failed to converge from most seeds"
    assert valid >= max(1, checked - 2), "too many projected configs landed invalid"


def test_escape_planner_finds_separated_goals_for_one_direction():
    planner, _, _ = _make_escape_planner(goal_samples=60, goal_candidates=4)
    start = FOLDED.copy()
    direction = planner.tip_direction(start)

    goals = planner._generate_goals(start, direction)

    assert len(goals) >= 2
    normalized = [planner._normalized(goal) for goal in goals]
    for first in range(len(normalized)):
        for second in range(first + 1, len(normalized)):
            separation = float(np.linalg.norm(normalized[first] - normalized[second]))
            assert separation >= planner.config.goal_separation_normalized - 1e-9


def test_escape_planner_connects_same_direction_branches_with_valid_path():
    planner, validator, _ = _make_escape_planner(
        goal_samples=40, goal_candidates=3, rrt_max_iterations=300, planning_timeout_s=6.0
    )
    # Two distinct configurations that point the same way exercise the
    # constrained search; the straight joint-space line between distant
    # branches usually violates the cone or collision checks.
    pairs = [
        (FOLDED.copy(), np.array([0.0, 30.0, -60.0, -30.0, 0.0])),
        (np.array([40.0, -20.0, 40.0, -50.0, 10.0]), FOLDED.copy()),
    ]
    planned = None
    for start, goal_config in pairs:
        direction = planner.tip_direction(goal_config)
        plan_obj = planner.plan(start, direction)
        if plan_obj is None:
            continue
        waypoints = [np.asarray(w, dtype=float) for w in plan_obj.waypoints_deg]
        # The plan starts ON the manifold (never at the off-cone trapped pose);
        # the hop from the trapped pose is the controller's collision-only
        # bridge, so it must exist but is exempt from the cone.
        assert validator.evaluate(waypoints[0])[0] is not None
        assert planner.direction_error_deg(waypoints[0], direction) <= math.degrees(
            planner.config.direction_tolerance_rad
        )
        assert planner._edge_valid_manifold(start, waypoints[0], direction, check_cone=False), "no bridge"
        assert all(validator.evaluate(w)[0] is not None for w in waypoints), "invalid waypoint"
        for waypoint in waypoints:
            error = vector_angle(planner.tip_direction(waypoint), direction)
            assert error <= planner.config.direction_tolerance_rad + 1e-6, "cone violated"
        deltas = [float(np.max(np.abs(b - a))) for a, b in zip(waypoints, waypoints[1:])]
        assert max(deltas) <= planner.config.waypoint_delta_deg + 1e-9
        planned = plan_obj
        break
    assert planned is not None, "planner failed to connect same-direction branches"


def test_escape_edge_validation_rejects_cone_violation_and_accepts_safe_pair():
    planner, _, _ = _make_escape_planner()
    base = np.array([0.0, 30.0, -60.0, -30.0, 0.0])
    direction = planner.tip_direction(base)
    nearby = planner.project(base + np.array([2.0, -2.0, 1.0, 1.0, -1.0]), direction)
    assert nearby is not None

    far = np.clip(base + np.array([70.0, 0.0, 0.0, 0.0, 0.0]), [-104, -104, -97, -104, -160], [104, 104, 97, 104, 160])
    assert planner._edge_valid_manifold(base, nearby, direction)
    assert not planner._edge_valid_manifold(base, far, direction)


def test_escape_waypoints_are_resampled_to_max_joint_delta():
    planner, _, _ = _make_escape_planner()
    segment = [np.zeros(5), np.array([45.0, -30.0, 20.0, -15.0, 10.0])]
    resampled = planner._resample(segment)

    assert np.allclose(resampled[0], segment[0])
    deltas = [float(np.max(np.abs(b - a))) for a, b in zip(resampled, resampled[1:])]
    assert max(deltas) <= planner.config.waypoint_delta_deg + 1e-9
    assert np.allclose(resampled[-1], segment[-1])


def test_trap_detection_triggers_escape_after_threshold():
    fake = FakeEscapePlanner([[*FOLDED], [*FOLDED + ESC_DELTA]])
    controller = _escape_controller(fake)
    assert controller.status()["escape_state"] == "idle"

    spawn_args = None
    for _ in range(controller.config.trap_tick_threshold):
        spawn_args = controller._note_tracking_failure()
        if spawn_args is not None:
            break

    assert spawn_args is not None, "escape did not trigger after threshold"
    assert controller.status()["escape_state"] == "planning"
    assert controller.status()["sync_enabled"]
    controller._escape_worker(*spawn_args)  # synchronous publish
    assert controller.status()["escape_state"] == "executing"
    assert fake.calls, "planner was never invoked"
    frozen = controller.status()["escape_frozen_direction"]
    assert frozen is not None and np.allclose(frozen, fake.calls[0][1], atol=1e-9)


def test_escape_execution_completes_and_resumes_tracking():
    delta = ESC_DELTA
    fake = FakeEscapePlanner([[*FOLDED], [*(FOLDED + delta)]])
    # Short cooldown keeps the simulated tick loop small while still proving
    # the completed -> idle housekeeping transition.
    controller = _escape_controller(fake, escape_cooldown_s=0.5)
    config = controller.config

    spawn_args = None
    for _ in range(controller.config.trap_tick_threshold):
        spawn_args = controller._note_tracking_failure()
        if spawn_args is not None:
            break
    assert spawn_args is not None
    controller._escape_worker(*spawn_args)  # synchronous publish
    assert controller.status()["escape_state"] == "executing"
    assert controller.status()["escape_progress"]["count"] >= 2

    joints_at_completion = None
    now = controller._last_tick
    for _ in range(int(config.escape_cooldown_s * config.fps) + 10):
        now += 1.0 / config.fps
        controller._latest_phone_time = now
        controller._tick(now)
        status = controller.status()
        if status["escape_state"] == "completed" and joints_at_completion is None:
            joints_at_completion = controller.current_arm_joints()[:5].copy()
        if status["escape_state"] == "idle":
            break

    assert controller.status()["escape_state"] == "idle"
    assert controller.status()["sync_enabled"]
    assert controller.status()["escape_outcome_counts"].get("completed", 0) == 1
    assert joints_at_completion is not None
    assert not np.allclose(controller.current_arm_joints()[:5], joints_at_completion), (
        "tracking did not resume after the escape finished"
    )


def test_escape_cancelled_by_calibration_invalidation_or_sync_off():
    for cancel in ("calibration", "sync"):
        delta = ESC_DELTA
        fake = FakeEscapePlanner([[*FOLDED], [*(FOLDED + delta)]])
        controller = _escape_controller(fake)
        spawn_args = None
        for _ in range(controller.config.trap_tick_threshold):
            spawn_args = controller._note_tracking_failure()
            if spawn_args is not None:
                break
        controller._escape_worker(*spawn_args)
        assert controller.status()["escape_state"] == "executing"

        if cancel == "calibration":
            controller.invalidate_calibration()
        else:
            controller.set_sync(False)
        assert controller.status()["escape_state"] == "idle"
        assert np.allclose(controller.current_arm_joints()[:5], FOLDED, atol=1e-9)


def test_escape_target_drift_aborts_execution():
    delta = ESC_DELTA
    fake = FakeEscapePlanner([[*FOLDED], [*(FOLDED + delta)], [*(FOLDED + 2 * delta)]])
    controller = _escape_controller(fake)
    spawn_args = None
    for _ in range(controller.config.trap_tick_threshold):
        spawn_args = controller._note_tracking_failure()
        if spawn_args is not None:
            break
    controller._escape_worker(*spawn_args)
    assert controller.status()["escape_state"] == "executing"

    # Rotate the phone mapping so the live target leaves the frozen direction.
    angle = np.deg2rad(60.0)
    rotation_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    with controller._lock:
        controller._phone_to_robot_rotation = rotation_z @ controller._phone_to_robot_rotation
    now = controller._last_tick + 1.0 / controller.config.fps
    controller._latest_phone_time = now
    controller._tick(now)

    assert controller.status()["escape_state"] == "failed"
    assert "target moved" in controller.status()["escape_last_reason"]
    assert controller.status()["escape_outcome_counts"].get("aborted", 0) == 1


def test_escape_trigger_guards_cooldown_and_manual_bypass():
    failing = FailingEscapePlanner()
    controller = _escape_controller(failing)
    direction = [1.0, 0.0, 0.0]

    uncalibrated = OrientationController(
        RobotKinematics(str(ROOT / "SO101/so101_new_calib.urdf"), joint_names=ARM_JOINTS),
        load_joint_limits(ROOT / "SO101/so101_new_calib.urdf",
                          Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"),
        escape_planner=failing,
    )
    ok, message = uncalibrated.trigger_escape(direction)
    assert not ok and "calibration" in message

    unsynced = _escape_controller(failing)
    with unsynced._lock:
        unsynced._sync_enabled = False
    ok, message = unsynced.trigger_escape(direction)
    assert not ok and "synchronization" in message

    # While planning: refuse a second request (state forced deterministically
    # because the stub planner finishes faster than the test can re-request).
    ok, message = controller.trigger_escape(direction)
    assert ok, message
    controller.cancel_escape()
    with controller._lock:
        controller._escape_state = "planning"
        controller._escape_generation += 1
    ok, message = controller.trigger_escape(direction)
    assert not ok and "planning" in message
    with controller._lock:
        controller._escape_state = "idle"

    # A failing plan arms the cooldown...
    ok, message = controller.trigger_escape(direction)
    assert ok, message
    controller._escape_worker(
        controller._q[:5].copy(), np.asarray(direction, dtype=float), controller._escape_generation, 1.0
    )
    assert controller.status()["escape_state"] == "failed"
    # ...which blocks automatic re-triggering but not an explicit manual one.
    with controller._lock:
        controller._escape_trap_ticks = controller.config.trap_tick_threshold
        auto = controller._maybe_trigger_escape_locked()
    assert auto is None
    ok, message = controller.trigger_escape(direction)
    assert ok, message  # manual requests bypass the cooldown latch
    controller.cancel_escape()


def test_escape_publish_revalidates_waypoints_and_bridges_from_live_commanded_q():
    # A plan starting away from the commanded pose gets bridged from _q.
    delta = 1.5 * ESC_DELTA
    fake = FakeEscapePlanner([[*(FOLDED + delta)], [*(FOLDED + 1.5 * delta)]])
    controller = _escape_controller(fake)
    spawn_args = None
    for _ in range(controller.config.trap_tick_threshold):
        spawn_args = controller._note_tracking_failure()
        if spawn_args is not None:
            break
    controller._escape_worker(*spawn_args)

    assert controller.status()["escape_state"] == "executing"
    plan_q = controller._escape_plan_q
    assert np.allclose(plan_q[0][:5], controller.current_arm_joints()[:5])
    assert np.allclose(plan_q[-1][:5], FOLDED + 1.5 * delta, atol=1e-9)

    # An out-of-limits waypoint must be rejected at publish time, not executed.
    ok, _ = controller.cancel_escape()
    assert ok
    bad = FakeEscapePlanner([[200.0, 0.0, 0.0, 0.0, 0.0], [*FOLDED]])
    controller._escape_planner = bad
    ok, message = controller.trigger_escape([1.0, 0.0, 0.0])
    assert ok, message
    worker_args = (controller._q[:5].copy(), np.asarray([1.0, 0.0, 0.0]), controller._escape_generation, 1.0)
    controller._escape_worker(*worker_args)
    assert controller.status()["escape_state"] == "failed"
    assert "rejected by runtime validation" in controller.status()["escape_last_reason"]
    assert np.allclose(controller.current_arm_joints()[:5], FOLDED, atol=1e-9)


def test_dry_run_and_hardware_perform_identical_escape_actions():
    delta = ESC_DELTA

    class StaticRobot:
        is_connected = True

        def __init__(self):
            self.actions = []

        def get_observation(self):
            measured = np.array([-20.0, 40.0, -30.0, 15.0, 10.0, 1.568])
            return {f"{name}.pos": measured[index] for index, name in enumerate(ALL_JOINTS)}

        def send_action(self, action):
            self.actions.append(action.copy())
            return action

        def disconnect(self):
            self.is_connected = False

    def run(robot):
        fake = FakeEscapePlanner([[*FOLDED], [*(FOLDED + delta)]])
        controller = _escape_controller(fake, robot=robot)
        sent = []
        spawn_args = None
        for _ in range(controller.config.trap_tick_threshold):
            spawn_args = controller._note_tracking_failure()
            if spawn_args is not None:
                break
        controller._escape_worker(*spawn_args)
        now = controller._last_tick
        for _ in range(10):
            now += 1.0 / controller.config.fps
            controller._latest_phone_time = now
            controller._tick(now)
            sent.append(controller._last_sent_command_q.copy())
            if controller.status()["escape_state"] != "executing":
                break
        return sent

    dry_sent = run(None)
    robot = StaticRobot()
    hardware_sent = run(robot)

    assert len(dry_sent) == len(hardware_sent) > 1
    for dry_command, hardware_command in zip(dry_sent, hardware_sent):
        assert np.allclose(dry_command, hardware_command, atol=1e-10)
