"""Scripted SO-101 policy for generating wood pick-and-place demonstrations."""

from __future__ import annotations

from enum import Enum, auto

import torch
from isaaclab.managers import SceneEntityCfg
from leisaac.datagen.state_machine.base import StateMachineBase

from .grasp_planner import (
    GraspSamplingConfig,
    interpolate_quaternion,
    minimum_jerk,
    quat_apply,
    quat_conjugate,
    quat_multiply,
    sample_grasp_candidates,
)
from .mdp import stick_in_destination_box


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0


class Phase(Enum):
    SETTLE = auto()
    MOVE_PREGRASP = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    MOVE_CLEAR = auto()
    MOVE_ABOVE_BOX = auto()
    LOWER_TO_BOX = auto()
    RELEASE = auto()
    RETREAT = auto()
    SETTLE_RESULT = auto()
    DONE = auto()


_PHASE_ORDER = (
    Phase.SETTLE,
    Phase.MOVE_PREGRASP,
    Phase.DESCEND,
    Phase.CLOSE,
    Phase.LIFT,
    Phase.MOVE_CLEAR,
    Phase.MOVE_ABOVE_BOX,
    Phase.LOWER_TO_BOX,
    Phase.RELEASE,
    Phase.RETREAT,
    Phase.SETTLE_RESULT,
    Phase.DONE,
)

# With the scene's dt=1/120 and decimation=4, one policy step is 1/30 s.
_PHASE_STEPS = {
    Phase.SETTLE: 20,
    Phase.MOVE_PREGRASP: 120,
    Phase.DESCEND: 60,
    Phase.CLOSE: 25,
    Phase.LIFT: 65,
    Phase.MOVE_CLEAR: 60,
    Phase.MOVE_ABOVE_BOX: 90,
    Phase.LOWER_TO_BOX: 45,
    Phase.RELEASE: 25,
    Phase.RETREAT: 45,
    Phase.SETTLE_RESULT: 75,
}

_TRACKING_TOLERANCE_M = {
    Phase.MOVE_PREGRASP: 0.035,
    # The virtual TCP is still awaiting exact USD calibration.  Continue far
    # enough to test physical closure/lift, then reject the episode if the
    # object-height check below shows that no grasp occurred.
    Phase.DESCEND: 0.060,
    Phase.LIFT: 0.030,
    Phase.MOVE_CLEAR: 0.035,
    Phase.MOVE_ABOVE_BOX: 0.030,
    Phase.LOWER_TO_BOX: 0.025,
    Phase.RETREAT: 0.030,
}


class WoodPickStateMachine(StateMachineBase):
    """Generate one conservative, feedback-validated scripted demonstration.

    The LeIsaac ``so101_state_machine`` action term performs damped least-
    squares IK each simulator step.  This class samples geometrically valid
    pose candidates, sends smooth Cartesian waypoints, and treats failure to
    track any critical waypoint as an IK/reachability failure.  The recorder
    can consequently export successful episodes only.

    Version one intentionally supports a single camera environment at a time.
    The phase scalar is shared, so using ``num_envs > 1`` would couple failure
    handling across environments and is rejected explicitly.
    """

    def __init__(self, grasp_config: GraspSamplingConfig = GraspSamplingConfig()) -> None:
        self._grasp_config = grasp_config
        self._gripper_body_id: int | None = None
        self._phase = Phase.SETTLE
        self._phase_step = 0
        self._episode_done = False
        self._failure_reason: str | None = None
        self._planned = False
        self._phase_initialized = False
        self._phase_start_pos_w: torch.Tensor | None = None
        self._phase_start_quat_w: torch.Tensor | None = None
        self._last_target_pos_w: torch.Tensor | None = None
        self._last_target_quat_w: torch.Tensor | None = None
        self._targets: dict[Phase, tuple[torch.Tensor, torch.Tensor]] = {}

    def setup(self, env) -> None:
        if env.num_envs != 1:
            raise ValueError("WoodPickStateMachine currently requires --num_envs 1")
        robot = env.scene["robot"]
        body_ids, body_names = robot.find_bodies("gripper")
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected exactly one SO-101 gripper body, found {body_names}")
        self._gripper_body_id = int(body_ids[0])

    def _current_ee_pose(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gripper_body_id is None:
            raise RuntimeError("Call setup(env) before generating actions")
        pose = env.scene["robot"].data.body_pose_w[:, self._gripper_body_id]
        body_pos, body_quat = pose[:, :3], pose[:, 3:7]
        contact_offset = torch.tensor(
            self._grasp_config.gripper_to_contact_tool,
            device=body_pos.device,
            dtype=body_pos.dtype,
        ).expand_as(body_pos)
        contact_pos = body_pos + quat_apply(body_quat, contact_offset)
        return contact_pos.clone(), body_quat.clone()

    def _plan_episode(self, env) -> None:
        stick = env.scene["stick"]
        stick_pos_w = stick.data.root_pos_w.clone()
        stick_quat_w = stick.data.root_quat_w.clone()
        gripper_pose = env.scene["robot"].data.body_pose_w[:, self._gripper_body_id]
        jaw_detection_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        jaw_from_gripper_local = quat_apply(
            quat_conjugate(gripper_pose[:, 3:7]),
            jaw_detection_w - gripper_pose[:, :3],
        )
        print(
            "  tool calibration: "
            f"jaw_detection_w={[round(float(v), 4) for v in jaw_detection_w[0].tolist()]}, "
            f"jaw_from_gripper_local={[round(float(v), 4) for v in jaw_from_gripper_local[0].tolist()]}",
            flush=True,
        )
        candidates = sample_grasp_candidates(stick_pos_w, stick_quat_w, config=self._grasp_config)
        grasp_point, _grasp_ee_pos, _pregrasp_ee_pos, pick_quat, has_candidate = candidates.select()
        if not bool(has_candidate.all()):
            self._fail("no grasp candidate passed the coarse workspace filter")
            return

        dtype, device = stick_pos_w.dtype, stick_pos_w.device
        env_origin = env.scene.env_origins
        world_up = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).expand_as(stick_pos_w)
        tool_z = quat_apply(pick_quat, world_up)
        pregrasp_point = grasp_point + self._grasp_config.pregrasp_clearance * tool_z
        lifted_grasp = grasp_point + self._grasp_config.lift_clearance * world_up

        # Canonical drop orientation: tool +Y (and therefore the held stick)
        # lies along box +X, while tool +Z remains upright.
        half_sqrt = 2**-0.5
        drop_quat = torch.tensor((half_sqrt, 0.0, 0.0, half_sqrt), device=device, dtype=dtype).repeat(
            env.num_envs, 1
        )
        clear_contact = env_origin + torch.tensor((0.070, 0.340, 0.225), device=device, dtype=dtype)
        box_center_xy = torch.tensor((-0.032, 0.430), device=device, dtype=dtype)
        above_box_contact = torch.cat(
            (
                env_origin[:, :2] + box_center_xy,
                env_origin[:, 2:3] + torch.full((env.num_envs, 1), 0.185, device=device, dtype=dtype),
            ),
            dim=-1,
        )
        release_contact = above_box_contact.clone()
        release_contact[:, 2] = env_origin[:, 2] + 0.110

        retreat_contact = release_contact + 0.100 * world_up

        self._targets = {
            Phase.MOVE_PREGRASP: (pregrasp_point, pick_quat),
            Phase.DESCEND: (grasp_point, pick_quat),
            Phase.CLOSE: (grasp_point, pick_quat),
            Phase.LIFT: (lifted_grasp, pick_quat),
            Phase.MOVE_CLEAR: (clear_contact, pick_quat),
            Phase.MOVE_ABOVE_BOX: (above_box_contact, drop_quat),
            Phase.LOWER_TO_BOX: (release_contact, drop_quat),
            Phase.RELEASE: (release_contact, drop_quat),
            Phase.RETREAT: (retreat_contact, drop_quat),
            Phase.SETTLE_RESULT: (retreat_contact, drop_quat),
        }
        self._planned = True

    def _fail(self, reason: str) -> None:
        self._failure_reason = reason
        self._episode_done = True
        self._phase = Phase.DONE

    def _validate_previous_phase(self, env, previous_phase: Phase) -> None:
        tolerance = _TRACKING_TOLERANCE_M.get(previous_phase)
        if tolerance is not None and self._last_target_pos_w is not None:
            actual_pos, _ = self._current_ee_pose(env)
            error = torch.linalg.vector_norm(actual_pos - self._last_target_pos_w, dim=-1)
            if bool((error > tolerance).any()):
                actual_xyz = [round(float(value), 4) for value in actual_pos[0].tolist()]
                target_xyz = [round(float(value), 4) for value in self._last_target_pos_w[0].tolist()]
                jaw_xyz = [
                    round(float(value), 4)
                    for value in env.scene["ee_frame"].data.target_pos_w[0, 1].tolist()
                ]
                stick_xyz = [
                    round(float(value), 4)
                    for value in env.scene["stick"].data.root_pos_w[0].tolist()
                ]
                self._fail(
                    f"{previous_phase.name} was not IK-reachable: "
                    f"position error={float(error.max().item()):.4f} m > {tolerance:.4f} m; "
                    f"actual={actual_xyz}, target={target_xyz}, "
                    f"jaw_detection={jaw_xyz}, stick={stick_xyz}"
                )
                return

        if previous_phase is Phase.LIFT:
            stick = env.scene["stick"]
            relative_height = stick.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            if bool((relative_height < 0.095).any()):
                self._fail("grasp failed: stick did not rise with the gripper")

    def _initialize_phase(self, env) -> None:
        previous_phase = _PHASE_ORDER[_PHASE_ORDER.index(self._phase) - 1] if self._phase is not Phase.SETTLE else None
        if previous_phase is not None:
            self._validate_previous_phase(env, previous_phase)
            if self._episode_done:
                return
        self._phase_start_pos_w, self._phase_start_quat_w = self._current_ee_pose(env)
        if self._phase is Phase.SETTLE:
            self._targets[Phase.SETTLE] = (self._phase_start_pos_w, self._phase_start_quat_w)
        self._phase_initialized = True

    def _phase_gripper_command(self) -> float:
        if self._phase in (
            Phase.CLOSE,
            Phase.LIFT,
            Phase.MOVE_CLEAR,
            Phase.MOVE_ABOVE_BOX,
            Phase.LOWER_TO_BOX,
        ):
            return _GRIPPER_CLOSE
        return _GRIPPER_OPEN

    def _target_for_current_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._phase_start_pos_w is None or self._phase_start_quat_w is None:
            raise RuntimeError("Phase was not initialized")
        target_pos, target_quat = self._targets[self._phase]
        duration = _PHASE_STEPS[self._phase]
        progress = torch.tensor(
            min((self._phase_step + 1) / duration, 1.0),
            device=target_pos.device,
            dtype=target_pos.dtype,
        )
        alpha = minimum_jerk(progress).reshape(1, 1)
        position = self._phase_start_pos_w + alpha * (target_pos - self._phase_start_pos_w)
        quaternion = interpolate_quaternion(self._phase_start_quat_w, target_quat, alpha)
        return position, quaternion

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        if not self._planned and not self._episode_done:
            self._plan_episode(env)
        if self._episode_done:
            current_pos, current_quat = self._current_ee_pose(env)
            return self._world_pose_action(env, current_pos, current_quat, _GRIPPER_OPEN)
        if not self._phase_initialized:
            self._initialize_phase(env)
        if self._episode_done:
            current_pos, current_quat = self._current_ee_pose(env)
            return self._world_pose_action(env, current_pos, current_quat, _GRIPPER_OPEN)

        target_pos, target_quat = self._target_for_current_step()
        self._last_target_pos_w = target_pos.clone()
        self._last_target_quat_w = target_quat.clone()
        return self._world_pose_action(env, target_pos, target_quat, self._phase_gripper_command())

    def _world_pose_action(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_command: float,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        inverse_root = quat_conjugate(root_quat_w)
        target_pos_b = quat_apply(inverse_root, target_pos_w - root_pos_w)
        target_quat_b = quat_multiply(inverse_root, target_quat_w)
        gripper = torch.full(
            (env.num_envs, 1), gripper_command, device=env.device, dtype=target_pos_b.dtype
        )
        return torch.cat((target_pos_b, target_quat_b, gripper), dim=-1)

    def advance(self) -> None:
        if self._episode_done:
            return
        self._phase_step += 1
        if self._phase_step < _PHASE_STEPS[self._phase]:
            return
        current_index = _PHASE_ORDER.index(self._phase)
        self._phase = _PHASE_ORDER[current_index + 1]
        self._phase_step = 0
        self._phase_initialized = False
        if self._phase is Phase.DONE:
            self._episode_done = True

    def check_success(self, env) -> bool:
        if self._failure_reason is not None:
            return False
        success = stick_in_destination_box(env, asset_cfg=SceneEntityCfg("stick"))
        return bool(success.all().item())

    def reset(self) -> None:
        self._phase = Phase.SETTLE
        self._phase_step = 0
        self._episode_done = False
        self._failure_reason = None
        self._planned = False
        self._phase_initialized = False
        self._phase_start_pos_w = None
        self._phase_start_quat_w = None
        self._last_target_pos_w = None
        self._last_target_quat_w = None
        self._targets = {}

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def phase_step(self) -> int:
        return self._phase_step

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason
