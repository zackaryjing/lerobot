#!/usr/bin/env python

import json
from pathlib import Path

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
