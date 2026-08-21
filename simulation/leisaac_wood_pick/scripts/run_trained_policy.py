"""Run a real-robot LeRobot ACT checkpoint closed-loop in the wood-pick scene.

The policy uses LeRobot motor coordinates while Isaac stores radians.  This
runner performs the same calibrated conversion as LeIsaac's dataset exporter,
executes at 30 Hz, replans after a configurable number of ACT chunk actions,
and records synchronized front/wrist videos plus episode diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
LEISAAC_ASSETS_DIR = PROJECT_DIR / "assets" / "leisaac"
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "outputs/train/act_so101_test/checkpoints/025000/pretrained_model"
)
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(LEISAAC_ASSETS_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--episode_seconds", type=float, default=40.0)
parser.add_argument("--actions_per_chunk", type=int, default=50)
parser.add_argument(
    "--replan_overlap",
    type=int,
    default=25,
    help="Replan when this many queued actions remain; 25 mirrors the real client threshold of 0.5 for 50 actions.",
)
parser.add_argument(
    "--blend_new_weight",
    type=float,
    default=0.7,
    help="Weight of the new plan in fixed mode; 0.7 mirrors the real async client.",
)
parser.add_argument(
    "--blend_mode",
    choices=("cosine", "fixed"),
    default="cosine",
    help="Cosine cross-fade guarantees a continuous handoff; fixed reproduces the real client's weighted average.",
)
parser.add_argument("--max_relative_target", type=float, default=10.0)
parser.add_argument("--settle_steps", type=int, default=1)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument(
    "--video_dir",
    type=Path,
    default=Path("outputs/trained_policy_sim"),
)
parser.add_argument(
    "--metrics_file",
    type=Path,
    default=Path("outputs/trained_policy_sim/metrics.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

import leisaac_wood_pick  # noqa: F401
from leisaac_wood_pick.tasks.wood_pick import TASK_ID


# Mean of frame zero over all 56 real training episodes, in LeRobot motor
# coordinates.  Resetting here is materially less out-of-distribution than
# Isaac's all-zero default pose.
POLICY_HOME_MOTOR = np.asarray(
    (0.734447, -99.198679, 98.762425, 55.575562, -1.768954, 16.508600),
    dtype=np.float32,
)


class EpisodeVideoRecorder:
    """Write synchronized camera streams without retaining frames in RAM."""

    def __init__(self, directory: Path, fps: int = 30) -> None:
        self.directory = directory.resolve()
        self.fps = fps
        self._writers: dict[str, object] = {}
        self._paths: dict[str, Path] = {}

    def start(self, episode: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for camera in ("front", "wrist"):
            path = self.directory / f"episode_{episode:03d}_{camera}_pending.mp4"
            self._paths[camera] = path
            self._writers[camera] = imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

    @staticmethod
    def _rgb(image: torch.Tensor) -> np.ndarray:
        array = image[0, ..., :3].detach().cpu().numpy()
        return np.clip(array, 0, 255).astype(np.uint8)

    def append(self, observation: dict[str, torch.Tensor]) -> None:
        for camera, writer in self._writers.items():
            writer.append_data(self._rgb(observation[camera]))

    def finish(self, result: str) -> None:
        for writer in self._writers.values():
            writer.close()
        for camera, pending in self._paths.items():
            final = pending.with_name(pending.name.replace("pending", result))
            pending.rename(final)
            print(f"Saved video: {final}", flush=True)
        self._writers.clear()
        self._paths.clear()


def _motor_to_sim(motor: np.ndarray, device: str) -> torch.Tensor:
    motor_batch = np.asarray(motor, dtype=np.float32).reshape(-1, 6)
    radians = convert_lerobot_action_to_leisaac(motor_batch)
    return torch.as_tensor(radians, device=device, dtype=torch.float32)


def _sim_to_motor(radians: torch.Tensor) -> torch.Tensor:
    motor = convert_leisaac_action_to_lerobot(radians)
    return torch.as_tensor(motor, dtype=torch.float32)


def _policy_image(image: torch.Tensor) -> torch.Tensor:
    image = image[0, ..., :3].permute(2, 0, 1).contiguous().float()
    if float(image.max().item()) > 1.5:
        image = image / 255.0
    return image


def _predict_chunk(
    policy,
    preprocessor,
    postprocessor,
    policy_observation: dict[str, torch.Tensor],
    robot_joint_pos: torch.Tensor,
    actions_per_chunk: int,
) -> tuple[deque[torch.Tensor], float]:
    state_motor = _sim_to_motor(robot_joint_pos)[0]
    observation = {
        "observation.state": state_motor,
        "observation.images.wrist": _policy_image(policy_observation["wrist"]),
        "observation.images.front": _policy_image(policy_observation["front"]),
    }

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        processed = preprocessor(observation)
        normalized_chunk = policy.predict_action_chunk(processed)
        count = min(actions_per_chunk, normalized_chunk.shape[1])
        actions = [
            postprocessor(normalized_chunk[:, index, :]).squeeze(0).cpu()
            for index in range(count)
        ]
    torch.cuda.synchronize()
    return deque(actions), time.perf_counter() - started


def _overlap_action_chunks(
    queued_actions: deque[torch.Tensor],
    incoming_actions: deque[torch.Tensor],
    new_weight: float,
    blend_mode: str,
) -> deque[torch.Tensor]:
    """Align two plans at the current step and blend their shared horizon.

    Fixed mode mirrors the real async client's ``weighted_average`` aggregation;
    cosine mode provides a continuous old-to-new handoff. Simulation uses
    queue-relative indices because it has no network timestamps. The unshared
    tail always comes from the newly inferred plan.
    """
    if not queued_actions:
        return deque(action.clone() for action in incoming_actions)
    old = list(queued_actions)
    new = list(incoming_actions)
    overlap = min(len(old), len(new))
    if blend_mode == "cosine":
        if overlap == 1:
            weights = (1.0,)
        else:
            # Smoothly retain the immediately pending old action, then reach
            # the new plan exactly at the end of the shared horizon.
            weights = tuple(
                0.5 - 0.5 * np.cos(np.pi * index / (overlap - 1))
                for index in range(overlap)
            )
    else:
        weights = (new_weight,) * overlap
    blended = [
        (1.0 - weights[index]) * old[index] + weights[index] * new[index]
        for index in range(overlap)
    ]
    blended.extend(new[overlap:])
    return deque(blended)


def _set_policy_home(env) -> dict[str, list[float]]:
    robot = env.scene["robot"]
    home = _motor_to_sim(POLICY_HOME_MOTOR, env.device)
    velocity = torch.zeros_like(home)
    robot.write_joint_state_to_sim(home, velocity)
    robot.set_joint_position_target(home)
    return {
        "motor": POLICY_HOME_MOTOR.tolist(),
        "radians": home[0].detach().cpu().tolist(),
    }


def main() -> None:
    if not 0 <= args_cli.replan_overlap < args_cli.actions_per_chunk:
        raise ValueError("replan_overlap must be in [0, actions_per_chunk)")
    if not 0.0 <= args_cli.blend_new_weight <= 1.0:
        raise ValueError("blend_new_weight must be in [0, 1]")
    if not args_cli.model_path.is_dir():
        raise FileNotFoundError(f"ACT checkpoint not found: {args_cli.model_path}")
    robot_asset = LEISAAC_ASSETS_DIR / "robots" / "so101_follower.usd"
    if not robot_asset.is_file():
        raise FileNotFoundError(f"SO-101 asset not found: {robot_asset}")

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    # Model outputs are absolute calibrated joint targets. Isaac Lab's joint
    # action defaults to adding the articulation's default pose as an offset,
    # which would effectively command ``home + home`` and pin several joints
    # at their limits.
    env_cfg.actions.arm_action.use_default_offset = False
    env_cfg.actions.gripper_action.use_default_offset = False
    # The real STS3215 servos hold their commanded position against gravity.
    # LeIsaac already disables gravity for its IK/state-machine control mode;
    # do the same for direct learned joint targets or the arm falls onto its
    # shoulder/elbow limits before the first policy observation.
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    # Required by ``WoodPickSceneCfg.robot_contact``.  The stock SO-101 USD
    # does not apply PhysX contact-reporter APIs to its rigid links itself.
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    # The USD's coarse link colliders overlap in the tightly folded real-data
    # home pose and otherwise pin shoulder/elbow at their limits.  External
    # collisions with the table, platform, stick and box remain enabled.
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    # This runner owns its lightweight video/metrics output.  Disable the
    # default in-memory Isaac recorder terms to avoid retaining an unused copy
    # of every 640x480 observation during long episodes.
    env_cfg.recorders.record_initial_state = None
    env_cfg.recorders.record_post_step_states = None
    env_cfg.recorders.record_pre_step_actions = None
    env_cfg.recorders.record_pre_step_flat_policy_observations = None
    env_cfg.recorders.record_post_step_processed_actions = None
    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped

    print(f"Loading ACT checkpoint: {args_cli.model_path.resolve()}", flush=True)
    policy = get_policy_class("act").from_pretrained(args_cli.model_path)
    policy.to(args_cli.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args_cli.model_path,
        preprocessor_overrides={"device_processor": {"device": args_cli.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    video = EpisodeVideoRecorder(args_cli.video_dir, fps=30)
    episode_reports: list[dict] = []
    max_steps = round(args_cli.episode_seconds / env.step_dt)

    try:
        for episode in range(1, args_cli.num_episodes + 1):
            observation, _ = env.reset()
            policy.reset()
            home = _set_policy_home(env)
            home_action = _motor_to_sim(POLICY_HOME_MOTOR, env.device)
            for _ in range(args_cli.settle_steps):
                observation, _, _, _, _ = env.step(home_action)

            policy_observation = observation["policy"]
            initial_motor = _sim_to_motor(env.scene["robot"].data.joint_pos)[0]
            initial_stick = env.scene["stick"].data.root_pos_w[0].detach().cpu()
            action_queue: deque[torch.Tensor] = deque()
            raw_action_queue: deque[torch.Tensor] = deque()
            inference_times: list[float] = []
            predicted_min = torch.full((6,), torch.inf)
            predicted_max = torch.full((6,), -torch.inf)
            clipped_steps = 0
            max_stick_height = float(initial_stick[2].item())
            success = False
            actual_motor_trace: list[np.ndarray] = []
            commanded_motor_trace: list[np.ndarray] = []
            predicted_motor_trace: list[np.ndarray] = []
            effective_motor_trace: list[np.ndarray] = []
            joint_velocity_trace: list[np.ndarray] = []
            stick_position_trace: list[np.ndarray] = []
            ee_pose_trace: list[np.ndarray] = []
            ee_velocity_trace: list[np.ndarray] = []
            applied_torque_trace: list[np.ndarray] = []
            computed_torque_trace: list[np.ndarray] = []
            contact_force_trace: list[np.ndarray] = []
            chunk_id_trace: list[int] = []
            chunk_action_index_trace: list[int] = []
            clipped_joint_trace: list[np.ndarray] = []
            robot = env.scene["robot"]
            gripper_body_ids, gripper_body_names = robot.find_bodies("gripper")
            if len(gripper_body_ids) != 1:
                raise RuntimeError(
                    f"Expected one gripper body, found {gripper_body_names}"
                )
            gripper_body_id = int(gripper_body_ids[0])
            joint_names = list(robot.data.joint_names)
            contact_sensor = env.scene["robot_contact"]
            contact_body_names = list(contact_sensor.body_names)
            chunk_id = -1
            chunk_action_index = 0
            video.start(episode)

            for step in range(max_steps):
                should_replan = not action_queue or (
                    args_cli.replan_overlap > 0
                    and len(action_queue) <= args_cli.replan_overlap
                )
                if should_replan:
                    chunk_id += 1
                    chunk_action_index = 0
                    incoming_actions, inference_time = _predict_chunk(
                        policy,
                        preprocessor,
                        postprocessor,
                        policy_observation,
                        env.scene["robot"].data.joint_pos,
                        args_cli.actions_per_chunk,
                    )
                    action_queue = _overlap_action_chunks(
                        action_queue,
                        incoming_actions,
                        args_cli.blend_new_weight,
                        args_cli.blend_mode,
                    )
                    # Preserve the unblended newest plan in parallel so the
                    # diagnostics can measure how much smoothing removed.
                    raw_action_queue = deque(action.clone() for action in incoming_actions)
                    inference_times.append(inference_time)
                    print(
                        f"Episode {episode}, step {step}: inferred {len(incoming_actions)} actions, "
                        f"effective queue={len(action_queue)}, overlap={args_cli.replan_overlap}, "
                        f"blend={args_cli.blend_mode}, "
                        f"in {inference_time * 1000.0:.1f} ms",
                        flush=True,
                    )

                effective_motor = action_queue.popleft().float()
                predicted_motor = raw_action_queue.popleft().float()
                predicted_min = torch.minimum(predicted_min, predicted_motor)
                predicted_max = torch.maximum(predicted_max, predicted_motor)
                current_motor = _sim_to_motor(env.scene["robot"].data.joint_pos)[0]
                delta = effective_motor - current_motor
                safe_delta = delta.clamp(
                    min=-args_cli.max_relative_target,
                    max=args_cli.max_relative_target,
                )
                if not torch.equal(delta, safe_delta):
                    clipped_steps += 1
                clipped_joints = torch.ne(delta, safe_delta)
                safe_motor = current_motor + safe_delta
                action = _motor_to_sim(safe_motor.numpy(), env.device)

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                observation, _, terminated, _, _ = env.step(action)
                policy_observation = observation["policy"]
                video.append(policy_observation)
                actual_motor = _sim_to_motor(env.scene["robot"].data.joint_pos)[0]
                stick_position = env.scene["stick"].data.root_pos_w[0].detach().cpu()
                actual_motor_trace.append(actual_motor.numpy().copy())
                commanded_motor_trace.append(safe_motor.numpy().copy())
                predicted_motor_trace.append(predicted_motor.numpy().copy())
                effective_motor_trace.append(effective_motor.numpy().copy())
                joint_velocity_trace.append(
                    env.scene["robot"].data.joint_vel[0].detach().cpu().numpy().copy()
                )
                stick_position_trace.append(stick_position.numpy().copy())
                ee_pose_trace.append(
                    robot.data.body_pose_w[0, gripper_body_id]
                    .detach().cpu().numpy().copy()
                )
                ee_velocity_trace.append(
                    robot.data.body_vel_w[0, gripper_body_id]
                    .detach().cpu().numpy().copy()
                )
                applied_torque_trace.append(
                    robot.data.applied_torque[0].detach().cpu().numpy().copy()
                )
                computed_torque_trace.append(
                    robot.data.computed_torque[0].detach().cpu().numpy().copy()
                )
                contact_force_trace.append(
                    contact_sensor.data.net_forces_w[0]
                    .detach().cpu().numpy().copy()
                )
                chunk_id_trace.append(chunk_id)
                chunk_action_index_trace.append(chunk_action_index)
                clipped_joint_trace.append(clipped_joints.numpy().copy())
                chunk_action_index += 1
                stick_height = float(stick_position[2].item())
                max_stick_height = max(max_stick_height, stick_height)
                if bool(terminated[0].item()):
                    success = True
                    break

            result = "success" if success else "failed"
            video.finish(result)
            final_stick = env.scene["stick"].data.root_pos_w[0].detach().cpu()
            final_motor = _sim_to_motor(env.scene["robot"].data.joint_pos)[0]
            actual_array = np.stack(actual_motor_trace)
            commanded_array = np.stack(commanded_motor_trace)
            predicted_array = np.stack(predicted_motor_trace)
            effective_array = np.stack(effective_motor_trace)
            velocity_array = np.stack(joint_velocity_trace)
            stick_array = np.stack(stick_position_trace)
            ee_pose_array = np.stack(ee_pose_trace)
            ee_velocity_array = np.stack(ee_velocity_trace)
            applied_torque_array = np.stack(applied_torque_trace)
            computed_torque_array = np.stack(computed_torque_trace)
            contact_force_array = np.stack(contact_force_trace)
            initial_array = initial_motor.numpy()[None, :]
            state_before_action = np.concatenate((initial_array, actual_array[:-1]), axis=0)
            intended_motion = commanded_array - state_before_action
            actual_motion = actual_array - state_before_action
            opposite_motion = (
                (intended_motion * actual_motion < 0.0)
                & (np.abs(intended_motion) > 0.1)
                & (np.abs(actual_motion) > 0.01)
            )
            tracking_error = np.abs(commanded_array - actual_array)
            diagnostics_path = (
                args_cli.video_dir.resolve() / f"episode_{episode:03d}_diagnostics.npz"
            )
            np.savez_compressed(
                diagnostics_path,
                actual_motor=actual_array,
                commanded_motor=commanded_array,
                predicted_motor=predicted_array,
                effective_motor_target=effective_array,
                actual_joint_position_rad=_motor_to_sim(actual_array, "cpu").numpy(),
                commanded_joint_position_rad=_motor_to_sim(commanded_array, "cpu").numpy(),
                predicted_joint_position_rad=_motor_to_sim(predicted_array, "cpu").numpy(),
                joint_velocity_radians_s=velocity_array,
                stick_position_world=stick_array,
                ee_pose_world=ee_pose_array,
                ee_velocity_world=ee_velocity_array,
                applied_joint_torque=applied_torque_array,
                computed_joint_torque=computed_torque_array,
                robot_contact_force_world=contact_force_array,
                chunk_id=np.asarray(chunk_id_trace, dtype=np.int32),
                chunk_action_index=np.asarray(chunk_action_index_trace, dtype=np.int32),
                clipped_joint=np.stack(clipped_joint_trace),
                joint_names=np.asarray(joint_names),
                gripper_body_name=np.asarray([gripper_body_names[0]]),
                contact_body_names=np.asarray(contact_body_names),
                fps=np.asarray([round(1.0 / env.step_dt)], dtype=np.int32),
            )
            print(f"Saved diagnostics: {diagnostics_path}", flush=True)
            report = {
                "episode": episode,
                "success": success,
                "frames": step + 1,
                "duration_s": (step + 1) * env.step_dt,
                "policy_home": home,
                "initial_state_motor": initial_motor.tolist(),
                "final_state_motor": final_motor.tolist(),
                "initial_stick_position": initial_stick.tolist(),
                "final_stick_position": final_stick.tolist(),
                "max_stick_height": max_stick_height,
                "predicted_action_min": predicted_min.tolist(),
                "predicted_action_max": predicted_max.tolist(),
                "relative_target_clipped_steps": clipped_steps,
                "tracking_error_motor": {
                    "mean_per_joint": tracking_error.mean(axis=0).tolist(),
                    "max_per_joint": tracking_error.max(axis=0).tolist(),
                    "opposite_motion_steps": int(np.any(opposite_motion, axis=1).sum()),
                },
                "inference_ms": {
                    "count": len(inference_times),
                    "mean": float(np.mean(inference_times) * 1000.0),
                    "max": float(np.max(inference_times) * 1000.0),
                },
            }
            episode_reports.append(report)
            print(json.dumps(report, indent=2), flush=True)
    finally:
        # Isaac Kit may terminate the interpreter as part of app shutdown, so
        # persist results before closing it.  Keeping this in ``finally`` also
        # preserves all completed episode reports if a later episode fails.
        if episode_reports:
            metrics = {
                "model_path": str(args_cli.model_path.resolve()),
                "seed": args_cli.seed,
                "actions_per_chunk": args_cli.actions_per_chunk,
                "replan_overlap": args_cli.replan_overlap,
                "blend_new_weight": args_cli.blend_new_weight,
                "blend_mode": args_cli.blend_mode,
                "max_relative_target": args_cli.max_relative_target,
                "episodes": episode_reports,
                "success_rate": sum(item["success"] for item in episode_reports)
                / len(episode_reports),
            }
            args_cli.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with args_cli.metrics_file.open("w") as stream:
                json.dump(metrics, stream, indent=2)
            print(f"Saved metrics: {args_cli.metrics_file.resolve()}", flush=True)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
