"""Task-specific grasp sampling and Mink IK for the SO-101 wood-pick scene."""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from .env import JOINT_NAMES, STICK_HALF_SIZE, WoodPickEnv


ARM_JOINTS = JOINT_NAMES[:5]
PLATFORM_CENTER_XY = np.array((0.235, 0.362), dtype=np.float64)
PLATFORM_HALF_XY = np.array((0.135, 0.050), dtype=np.float64)
PLATFORM_TOP_Z = 0.053
BOX_CENTER_XY = np.array((-0.032, 0.430), dtype=np.float64)

# First 56 real training episodes: median open follower state=16.86 and
# median closed leader command=6.28 in LeRobot's calibrated 0--100 motor
# coordinates. The real-policy bridge maps those to 8.5 and -3.1 physical
# degrees respectively.
OPEN_GRIPPER = np.deg2rad(8.5)
CLOSED_GRIPPER = np.deg2rad(-3.1)
CONTACT_GRIPPER = CLOSED_GRIPPER
SCRIPTED_HOME = np.deg2rad(np.array((0.8, -99.2, 88.8, 52.8, -2.8, 25.0)))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-10:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def set_grasp_weld(env: WoodPickEnv, active: bool) -> None:
    """Activate/deactivate a weld while preserving the current relative pose."""
    equality_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "scripted_grasp_weld"
    )
    if equality_id < 0:
        raise RuntimeError("scripted_grasp_weld equality is missing from the model")
    if active:
        gripper_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        stick_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        gripper_rotation = env.data.xmat[gripper_id].reshape(3, 3)
        relative_position = gripper_rotation.T @ (
            env.data.xpos[stick_id] - env.data.xpos[gripper_id]
        )
        gripper_quaternion = env.data.xquat[gripper_id]
        stick_quaternion = env.data.xquat[stick_id]
        relative_quaternion = quaternion_multiply(
            np.array(
                (
                    gripper_quaternion[0],
                    -gripper_quaternion[1],
                    -gripper_quaternion[2],
                    -gripper_quaternion[3],
                )
            ),
            stick_quaternion,
        )
        relative_quaternion /= np.linalg.norm(relative_quaternion)
        # Compiled weld layout is anchor-on-body2, anchor-on-body1,
        # relative quaternion, torque scale (3 + 3 + 4 + 1). This differs
        # from the compact seven-value MJCF ``relpose`` representation.
        env.model.eq_data[equality_id, 0:3] = 0.0
        env.model.eq_data[equality_id, 3:6] = relative_position
        env.model.eq_data[equality_id, 6:10] = relative_quaternion
    env.data.eq_active[equality_id] = active
    mujoco.mj_forward(env.model, env.data)


def minimum_jerk(progress: float | np.ndarray) -> float | np.ndarray:
    progress = np.clip(progress, 0.0, 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def sample_stick_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a horizontal stick pose whose full footprint stays on the platform."""
    for _ in range(1024):
        yaw = rng.uniform(-np.pi, np.pi)
        half_long = STICK_HALF_SIZE[2]
        half_cross = STICK_HALF_SIZE[0]
        projected = np.array(
            (
                half_long * abs(np.cos(yaw)) + half_cross * abs(np.sin(yaw)),
                half_long * abs(np.sin(yaw)) + half_cross * abs(np.cos(yaw)),
            )
        )
        available = PLATFORM_HALF_XY - projected - 0.005
        if np.any(available <= 0.0):
            continue
        xy = PLATFORM_CENTER_XY + rng.uniform(-available, available)
        # The far-right/far-forward corner is outside the practical SO-101 workspace.
        if np.linalg.norm(xy) > 0.445:
            continue
        position = np.array((xy[0], xy[1], PLATFORM_TOP_Z + half_cross))
        yaw_quat = np.array((np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        lay_flat_quat = np.array((2**-0.5, 0.0, 2**-0.5, 0.0))
        return position, quaternion_multiply(yaw_quat, lay_flat_quat)
    raise RuntimeError("Could not sample a stick pose on the platform")


@dataclass(frozen=True)
class GraspCandidate:
    contact_tcp: np.ndarray
    pregrasp_tcp: np.ndarray
    lift_tcp: np.ndarray
    above_box_tcp: np.ndarray
    lower_box_tcp: np.ndarray
    rotation: np.ndarray
    approach_axis: np.ndarray
    closing_axis: np.ndarray
    station_offset: float
    tilt_rad: float
    wrist_flip: int


@dataclass(frozen=True)
class IKResult:
    joint_positions: np.ndarray
    position_error: float
    approach_error_rad: float
    closing_error_rad: float
    iterations: int

    @property
    def feasible(self) -> bool:
        return (
            self.position_error <= 0.012
            # SO-101 has five arm DoFs, so exact 6D pose tracking is
            # over-constrained. Position and finger-closing alignment are hard
            # requirements; approach is deliberately allowed to lean.
            and self.approach_error_rad <= np.deg2rad(80.0)
            and self.closing_error_rad <= np.deg2rad(45.0)
        )


@dataclass
class TrajectoryPhase:
    name: str
    joint_targets: np.ndarray


@dataclass
class ScriptedPlan:
    phases: list[TrajectoryPhase]
    candidate: GraspCandidate
    ik_report: dict[str, IKResult]
    candidate_index: int
    seed_index: int

    @property
    def total_steps(self) -> int:
        return sum(len(phase.joint_targets) for phase in self.phases)


def sample_grasp_candidates(
    stick_position: np.ndarray,
    stick_rotation: np.ndarray,
    rng: np.random.Generator,
    count: int = 32,
    grasp_depth: float = 0.010,
) -> list[GraspCandidate]:
    """Sample TCP poses inside the two regular sections of the stick bounding box."""
    long_axis = normalize(stick_rotation[:, 2])
    world_up = np.array((0.0, 0.0, 1.0))
    candidates: list[GraspCandidate] = []
    for index in range(count):
        station_sign = 1.0 if index % 2 == 0 else -1.0
        wrist_flip = 1 if (index // 2) % 2 == 0 else -1
        tilt = 0.0 if index < 4 else rng.uniform(-np.deg2rad(20), np.deg2rad(20))
        station = station_sign * rng.uniform(0.026, 0.034)
        local_point = np.array(
            (rng.uniform(-0.0015, 0.0015), rng.uniform(-0.0015, 0.0015), station)
        )
        nominal_tcp = stick_position + stick_rotation @ local_point

        # Site local +X points from the wrist toward the fingertips; local +Z
        # is the finger closing axis. Keep both fingers at comparable height.
        approach = normalize(-world_up + np.tan(tilt) * long_axis)
        closing = wrist_flip * normalize(np.cross(long_axis, approach))
        tool_y = normalize(np.cross(closing, approach))
        rotation = np.column_stack((approach, tool_y, closing))
        # At the calibrated aperture the fixed fingertip lies on the local -Z
        # side of the planner TCP. Moving the desired TCP 8 mm in that same
        # direction translates the fixed jaw away from the stick, leaving the
        # moving jaw to establish contact during closure instead of letting the
        # fixed jaw press down on the stick during descent.
        nominal_tcp = nominal_tcp - 0.008 * closing
        # Keep the hover unchanged, then descend past the geometric section
        # centre. A few millimetres of platform contact is intentional: the
        # real position-controlled arm will be stopped by that surface too.
        pregrasp = nominal_tcp - 0.045 * approach
        contact_tcp = nominal_tcp - np.array((0.0, 0.0, grasp_depth))
        lift = contact_tcp + np.array((0.0, -0.025, 0.115))
        # Keep the stick centre, not the off-centre grasp station, above the
        # destination centre during release.
        grasp_offset = contact_tcp - stick_position
        above_box = np.array((BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.155)) + grasp_offset
        lower_box = np.array((BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.105)) + grasp_offset
        candidates.append(
            GraspCandidate(
                contact_tcp=contact_tcp,
                pregrasp_tcp=pregrasp,
                lift_tcp=lift,
                above_box_tcp=above_box,
                lower_box_tcp=lower_box,
                rotation=rotation,
                approach_axis=approach,
                closing_axis=closing,
                station_offset=station,
                tilt_rad=tilt,
                wrist_flip=wrist_flip,
            )
        )
    return candidates


class MinkIKSolver:
    """Position-prioritized constrained IK on the MuJoCo gripper TCP site."""

    def __init__(self, env: WoodPickEnv) -> None:
        self.model = env.model
        self.configuration = mink.Configuration(self.model, q=env.data.qpos.copy())
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.qpos_addresses = env._qpos_addresses.copy()
        self.arm_qpos_addresses = self.qpos_addresses[:5]
        self.arm_dof_addresses = self.model.jnt_dofadr[env._joint_ids[:5]].copy()
        allowed_dofs = set(int(index) for index in self.arm_dof_addresses)
        frozen_dofs = [index for index in range(self.model.nv) if index not in allowed_dofs]

        self.frame_task = mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=8.0,
            orientation_cost=0.16,
            gain=0.8,
            lm_damping=1.0e-4,
        )
        self.posture_task = mink.PostureTask(
            self.model, cost=2.0e-3, gain=0.2, lm_damping=1.0e-4
        )
        self.freeze_task = mink.DofFreezingTask(self.model, frozen_dofs)
        velocities = {name: 1.6 for name in ARM_JOINTS}
        self.limits = [
            mink.ConfigurationLimit(self.model, min_distance_from_limits=np.deg2rad(0.5)),
            mink.VelocityLimit(self.model, velocities),
        ]
        self.base_q = env.data.qpos.copy()
        self.tcp_offset_in_site = self._calibrate_contact_tcp(env)

    def _calibrate_contact_tcp(self, env: WoodPickEnv) -> np.ndarray:
        """Return site-to-midpoint offset at the 11.5 mm contact aperture."""
        saved_qpos = env.data.qpos.copy()
        env.data.qpos[env._qpos_addresses[-1]] = CONTACT_GRIPPER
        mujoco.mj_forward(env.model, env.data)
        fixed = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1"
        )
        moving = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_sph_tip1"
        )
        midpoint = 0.5 * (env.data.geom_xpos[fixed] + env.data.geom_xpos[moving])
        site_rotation = env.data.site_xmat[self.site_id].reshape(3, 3)
        offset = site_rotation.T @ (midpoint - env.data.site_xpos[self.site_id])
        env.data.qpos[:] = saved_qpos
        mujoco.mj_forward(env.model, env.data)
        return offset

    def update_base_state(self, qpos: np.ndarray) -> None:
        self.base_q = np.asarray(qpos, dtype=np.float64).copy()

    def site_target_for_tcp(self, tcp_position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(tcp_position) - np.asarray(rotation) @ self.tcp_offset_in_site

    def solve(
        self,
        tcp_position: np.ndarray,
        rotation: np.ndarray,
        seed: np.ndarray,
        *,
        max_iterations: int = 160,
    ) -> IKResult:
        q = self.base_q.copy()
        q[self.qpos_addresses] = seed
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)
        site_target = self.site_target_for_tcp(tcp_position, rotation)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rotation), site_target
        )
        self.frame_task.set_target(target)

        iterations = 0
        for iterations in range(1, max_iterations + 1):
            velocity = mink.solve_ik(
                self.configuration,
                [self.frame_task, self.posture_task],
                dt=0.05,
                solver="daqp",
                damping=1.0e-5,
                limits=self.limits,
                constraints=[self.freeze_task],
            )
            self.configuration.integrate_inplace(velocity, 0.05)
            position_error, approach_error, closing_error = self._errors(
                tcp_position, rotation
            )
            if (
                position_error < 0.0025
                and approach_error < np.deg2rad(18.0)
                and closing_error < np.deg2rad(18.0)
            ):
                break

        position_error, approach_error, closing_error = self._errors(tcp_position, rotation)
        result_q = seed.copy()
        result_q[:5] = self.configuration.q[self.arm_qpos_addresses]
        return IKResult(
            joint_positions=result_q,
            position_error=position_error,
            approach_error_rad=approach_error,
            closing_error_rad=closing_error,
            iterations=iterations,
        )

    def _errors(
        self, tcp_target: np.ndarray, rotation_target: np.ndarray
    ) -> tuple[float, float, float]:
        site_position = self.configuration.data.site_xpos[self.site_id]
        site_rotation = self.configuration.data.site_xmat[self.site_id].reshape(3, 3)
        tcp_position = site_position + site_rotation @ self.tcp_offset_in_site
        position_error = float(np.linalg.norm(tcp_position - tcp_target))
        approach_error = float(
            np.arccos(np.clip(np.dot(site_rotation[:, 0], rotation_target[:, 0]), -1.0, 1.0))
        )
        closing_error = float(
            np.arccos(np.clip(np.dot(site_rotation[:, 2], rotation_target[:, 2]), -1.0, 1.0))
        )
        return position_error, approach_error, closing_error


class ScriptedPlanner:
    def __init__(
        self,
        env: WoodPickEnv,
        rng: np.random.Generator,
        grasp_depth: float = 0.010,
        closed_gripper: float = CLOSED_GRIPPER,
    ) -> None:
        self.env = env
        self.rng = rng
        self.grasp_depth = grasp_depth
        self.closed_gripper = closed_gripper
        self.ik = MinkIKSolver(env)
        self._arm_low = env.model.jnt_range[env._joint_ids[:5], 0]
        self._arm_high = env.model.jnt_range[env._joint_ids[:5], 1]
        self._stick_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom"
        )
        self._collision_data = mujoco.MjData(env.model)

    def _segment_hits_stick(
        self, start: np.ndarray, end: np.ndarray, steps: int = 120
    ) -> bool:
        """Kinematically reject a free-space joint segment that hits the stick.

        IK endpoint feasibility alone does not protect the object from the arm
        sweeping through it en route to the pregrasp. This check keeps the
        measured object pose fixed and evaluates MuJoCo's actual collision
        geometry along the same minimum-jerk segment used by execution.
        """
        data = self._collision_data
        data.qpos[:] = self.env.data.qpos
        for target in self._joint_segment("collision_check", start, end, steps).joint_targets:
            data.qpos[self.env._qpos_addresses] = target
            mujoco.mj_forward(self.env.model, data)
            for contact in data.contact:
                if contact.geom1 == self._stick_geom_id:
                    other_geom = int(contact.geom2)
                elif contact.geom2 == self._stick_geom_id:
                    other_geom = int(contact.geom1)
                else:
                    continue
                # World geoms support the object and are expected. Any contact
                # with a robot body before descend invalidates this transit.
                if self.env.model.geom_bodyid[other_geom] != 0:
                    return True
        return False

    @staticmethod
    def _phase_feasible(phase: str, result: IKResult) -> bool:
        # Contact must be much more accurate than free-space transport. The
        # destination box has centimetres of clearance, while the 11.5 mm
        # stick cross-section does not.
        position_limit = {
            "pregrasp": 0.010,
            "grasp": 0.006,
            "lift": 0.012,
            "above_box": 0.020,
            "lower_box": 0.020,
        }[phase]
        closing_limit = np.deg2rad(35.0 if phase in ("pregrasp", "grasp") else 50.0)
        return (
            result.position_error <= position_limit
            and result.approach_error_rad <= np.deg2rad(82.0)
            and result.closing_error_rad <= closing_limit
        )

    def plan(self, candidate_count: int = 32, seeds_per_candidate: int = 3) -> ScriptedPlan:
        self.ik.update_base_state(self.env.data.qpos)
        stick_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "stick"
        )
        stick_position = self.env.data.xpos[stick_body].copy()
        stick_rotation = self.env.data.xmat[stick_body].reshape(3, 3).copy()
        candidates = sample_grasp_candidates(
            stick_position,
            stick_rotation,
            self.rng,
            count=candidate_count,
            grasp_depth=self.grasp_depth,
        )

        best: tuple[float, int, int, GraspCandidate, dict[str, IKResult]] | None = None
        failure_counts = {
            name: 0
            for name in ("pregrasp", "transit", "grasp", "lift", "above_box", "lower_box")
        }
        nearest_failure: tuple[float, str, IKResult] | None = None
        nearest_by_phase: dict[str, IKResult] = {}
        for candidate_index, candidate in enumerate(candidates):
            for seed_index in range(seeds_per_candidate):
                seed = SCRIPTED_HOME.copy()
                if seed_index:
                    seed[:5] = self.rng.uniform(self._arm_low, self._arm_high)
                report: dict[str, IKResult] = {}
                report["pregrasp"] = self.ik.solve(
                    candidate.pregrasp_tcp, candidate.rotation, seed
                )
                if not self._phase_feasible("pregrasp", report["pregrasp"]):
                    failure_counts["pregrasp"] += 1
                    value = report["pregrasp"].position_error
                    if (
                        "pregrasp" not in nearest_by_phase
                        or value < nearest_by_phase["pregrasp"].position_error
                    ):
                        nearest_by_phase["pregrasp"] = report["pregrasp"]
                    if nearest_failure is None or value < nearest_failure[0]:
                        nearest_failure = (value, "pregrasp", report["pregrasp"])
                    continue
                if self._segment_hits_stick(
                    SCRIPTED_HOME, report["pregrasp"].joint_positions
                ):
                    failure_counts["transit"] += 1
                    continue
                report["grasp"] = self.ik.solve(
                    candidate.contact_tcp,
                    candidate.rotation,
                    report["pregrasp"].joint_positions,
                )
                if not self._phase_feasible("grasp", report["grasp"]):
                    failure_counts["grasp"] += 1
                    value = report["grasp"].position_error
                    if "grasp" not in nearest_by_phase or value < nearest_by_phase["grasp"].position_error:
                        nearest_by_phase["grasp"] = report["grasp"]
                    if nearest_failure is None or value < nearest_failure[0]:
                        nearest_failure = (value, "grasp", report["grasp"])
                    continue
                report["lift"] = self.ik.solve(
                    candidate.lift_tcp, candidate.rotation, report["grasp"].joint_positions
                )
                if not self._phase_feasible("lift", report["lift"]):
                    failure_counts["lift"] += 1
                    value = report["lift"].position_error
                    if "lift" not in nearest_by_phase or value < nearest_by_phase["lift"].position_error:
                        nearest_by_phase["lift"] = report["lift"]
                    if nearest_failure is None or value < nearest_failure[0]:
                        nearest_failure = (value, "lift", report["lift"])
                    continue
                report["above_box"] = self.ik.solve(
                    candidate.above_box_tcp,
                    candidate.rotation,
                    report["lift"].joint_positions,
                )
                if not self._phase_feasible("above_box", report["above_box"]):
                    failure_counts["above_box"] += 1
                    value = report["above_box"].position_error
                    if (
                        "above_box" not in nearest_by_phase
                        or value < nearest_by_phase["above_box"].position_error
                    ):
                        nearest_by_phase["above_box"] = report["above_box"]
                    if nearest_failure is None or value < nearest_failure[0]:
                        nearest_failure = (value, "above_box", report["above_box"])
                    continue
                report["lower_box"] = self.ik.solve(
                    candidate.lower_box_tcp,
                    candidate.rotation,
                    report["above_box"].joint_positions,
                )
                if not self._phase_feasible("lower_box", report["lower_box"]):
                    failure_counts["lower_box"] += 1
                    value = report["lower_box"].position_error
                    if (
                        "lower_box" not in nearest_by_phase
                        or value < nearest_by_phase["lower_box"].position_error
                    ):
                        nearest_by_phase["lower_box"] = report["lower_box"]
                    if nearest_failure is None or value < nearest_failure[0]:
                        nearest_failure = (value, "lower_box", report["lower_box"])
                    continue
                score = (
                    sum(result.position_error for result in report.values())
                    + 0.004 * abs(candidate.tilt_rad)
                    + 0.002
                    * np.linalg.norm(report["pregrasp"].joint_positions[:5] - SCRIPTED_HOME[:5])
                )
                if best is None or score < best[0]:
                    best = (score, candidate_index, seed_index, candidate, report)

        if best is None:
            nearest = "none"
            if nearest_failure is not None:
                _, phase, result = nearest_failure
                nearest = (
                    f"{phase}: position={result.position_error * 1000:.1f} mm, "
                    f"approach={np.rad2deg(result.approach_error_rad):.1f} deg, "
                    f"closing={np.rad2deg(result.closing_error_rad):.1f} deg"
                )
            per_phase = {
                phase: (
                    round(result.position_error * 1000, 1),
                    round(float(np.rad2deg(result.approach_error_rad)), 1),
                    round(float(np.rad2deg(result.closing_error_rad)), 1),
                )
                for phase, result in nearest_by_phase.items()
            }
            raise RuntimeError(
                f"No IK-feasible grasp/transport route among "
                f"{candidate_count * seeds_per_candidate} trials; "
                f"failure_counts={failure_counts}; nearest={nearest}; "
                f"nearest_by_phase_mm_deg={per_phase}"
            )
        _, candidate_index, seed_index, candidate, report = best
        phases = self._build_trajectory(candidate, report)
        return ScriptedPlan(phases, candidate, report, candidate_index, seed_index)

    def _joint_segment(
        self, name: str, start: np.ndarray, end: np.ndarray, steps: int
    ) -> TrajectoryPhase:
        progress = minimum_jerk(np.linspace(1.0 / steps, 1.0, steps))[:, None]
        targets = start[None, :] + progress * (end - start)[None, :]
        return TrajectoryPhase(name, targets)

    def _build_trajectory(
        self, candidate: GraspCandidate, report: dict[str, IKResult]
    ) -> list[TrajectoryPhase]:
        home = SCRIPTED_HOME.copy()
        pregrasp = report["pregrasp"].joint_positions.copy()
        grasp = report["grasp"].joint_positions.copy()
        lift = report["lift"].joint_positions.copy()
        above = report["above_box"].joint_positions.copy()
        lower = report["lower_box"].joint_positions.copy()
        for target in (home, pregrasp, grasp, lift, above, lower):
            target[5] = OPEN_GRIPPER

        closed_grasp = grasp.copy()
        closed_grasp[5] = self.closed_gripper
        closed_lift = lift.copy()
        closed_lift[5] = self.closed_gripper
        closed_above = above.copy()
        closed_above[5] = self.closed_gripper
        closed_lower = lower.copy()
        closed_lower[5] = self.closed_gripper
        retreat = above.copy()
        retreat[5] = OPEN_GRIPPER

        return [
            TrajectoryPhase("settle", np.repeat(home[None, :], 15, axis=0)),
            self._joint_segment("move_pregrasp", home, pregrasp, 120),
            self._joint_segment("descend", pregrasp, grasp, 50),
            self._joint_segment("close", grasp, closed_grasp, 35),
            self._joint_segment("lift", closed_grasp, closed_lift, 70),
            self._joint_segment("move_above_box", closed_lift, closed_above, 120),
            self._joint_segment("lower_to_box", closed_above, closed_lower, 50),
            self._joint_segment("release", closed_lower, lower, 35),
            self._joint_segment("retreat", lower, retreat, 50),
            TrajectoryPhase("settle_result", np.repeat(retreat[None, :], 60, axis=0)),
        ]
