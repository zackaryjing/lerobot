"""Run the real-robot ACT checkpoint closed-loop in the MuJoCo workcell."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

# These must be selected before importing MuJoCo. They work with NVIDIA through
# WSL's D3D12 Mesa backend while keeping the run headless.
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from mujoco_wood_pick.env import CONTROL_FPS, JOINT_NAMES, WoodPickEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPOSITORY_ROOT / "outputs/train/act_so101_test/checkpoints/025000/pretrained_model"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/mujoco_trained_policy"

# Mean frame-zero state of the 56 real training episodes. This is in the
# calibrated SO-101 motor coordinate system stored by LeRobot.
POLICY_HOME_MOTOR = np.asarray(
    (0.734447, -99.198679, 98.762425, 55.575562, -1.768954, 16.508600),
    dtype=np.float32,
)

# Calibration used by the real-data/Isaac bridge. The first row is the motor
# coordinate range and the second is the corresponding physical joint angle.
MOTOR_LIMIT = np.asarray(
    ((-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (0, 100)),
    dtype=np.float32,
)
PHYSICAL_LIMIT_DEG = np.asarray(
    ((-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100)),
    dtype=np.float32,
)


def motor_to_sim(motor: np.ndarray) -> np.ndarray:
    motor = np.asarray(motor, dtype=np.float32)
    fraction = (motor - MOTOR_LIMIT[:, 0]) / (MOTOR_LIMIT[:, 1] - MOTOR_LIMIT[:, 0])
    degrees = PHYSICAL_LIMIT_DEG[:, 0] + fraction * (
        PHYSICAL_LIMIT_DEG[:, 1] - PHYSICAL_LIMIT_DEG[:, 0]
    )
    return np.deg2rad(degrees).astype(np.float64)


def sim_to_motor(radians: np.ndarray) -> np.ndarray:
    degrees = np.rad2deg(np.asarray(radians, dtype=np.float64))
    fraction = (degrees - PHYSICAL_LIMIT_DEG[:, 0]) / (
        PHYSICAL_LIMIT_DEG[:, 1] - PHYSICAL_LIMIT_DEG[:, 0]
    )
    return (
        MOTOR_LIMIT[:, 0] + fraction * (MOTOR_LIMIT[:, 1] - MOTOR_LIMIT[:, 0])
    ).astype(np.float32)


class VideoRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.writers: dict[str, object] = {}
        self.paths: dict[str, Path] = {}

    def start(self, episode: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for camera in ("front", "wrist"):
            path = self.output_dir / f"episode_{episode:03d}_{camera}_pending.mp4"
            self.paths[camera] = path
            self.writers[camera] = imageio.get_writer(
                path, fps=round(CONTROL_FPS), codec="libx264", quality=8, macro_block_size=None
            )

    def append(self, observation: dict[str, np.ndarray]) -> None:
        for camera, writer in self.writers.items():
            writer.append_data(observation[f"observation.images.{camera}"])

    def finish(self, result: str) -> list[str]:
        final_paths = []
        for writer in self.writers.values():
            writer.close()
        for camera, pending in self.paths.items():
            final = pending.with_name(pending.name.replace("pending", result))
            pending.rename(final)
            final_paths.append(str(final))
            print(f"Saved video: {final}", flush=True)
        self.writers.clear()
        self.paths.clear()
        return final_paths

    def abort(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


class CameraViewer:
    """Small Tk window for the exact RGB frames passed to the policy."""

    def __init__(self, title: str) -> None:
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._photo = None
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda _event: self.close())
        self._root.bind("q", lambda _event: self.close())

    def update(self, rgb: np.ndarray) -> bool:
        if self._closed:
            return False
        from PIL import Image, ImageTk

        try:
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._label.configure(image=self._photo)
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def policy_image(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255.0)


def predict_chunk(
    policy,
    preprocessor,
    postprocessor,
    observation: dict[str, np.ndarray],
    actions_per_chunk: int,
    device: str,
) -> tuple[deque[np.ndarray], float]:
    policy_observation = {
        "observation.state": torch.from_numpy(sim_to_motor(observation["observation.state"])),
        "observation.images.wrist": policy_image(observation["observation.images.wrist"]),
        "observation.images.front": policy_image(observation["observation.images.front"]),
    }
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        processed = preprocessor(policy_observation)
        normalized_chunk = policy.predict_action_chunk(processed)
        count = min(actions_per_chunk, normalized_chunk.shape[1])
        actions = deque(
            postprocessor(normalized_chunk[:, index, :]).squeeze(0).cpu().numpy().astype(np.float32)
            for index in range(count)
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return actions, time.perf_counter() - started


def overlap_chunks(
    queued: deque[np.ndarray], incoming: deque[np.ndarray], new_weight: float
) -> deque[np.ndarray]:
    """Reproduce the real client's fixed weighted-average overlap."""
    if not queued:
        return deque(action.copy() for action in incoming)
    old, new = list(queued), list(incoming)
    overlap = min(len(old), len(new))
    blended = [
        (1.0 - new_weight) * old[index] + new_weight * new[index] for index in range(overlap)
    ]
    blended.extend(new[overlap:])
    return deque(blended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=40.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--replan-overlap", type=int, default=25)
    parser.add_argument("--blend-new-weight", type=float, default=0.7)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo's interactive 3D viewer and pace execution close to 30 Hz.",
    )
    parser.add_argument(
        "--camera-view",
        choices=("none", "front", "wrist", "both"),
        default="none",
        help="Show live policy camera input in a separate window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"ACT checkpoint not found: {args.model_path}")
    if not 0 <= args.replan_overlap < args.actions_per_chunk:
        raise ValueError("replan-overlap must be in [0, actions-per-chunk)")
    if not 0 <= args.blend_new_weight <= 1:
        raise ValueError("blend-new-weight must be in [0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    print(f"Loading policy: {args.model_path.resolve()}", flush=True)
    policy = get_policy_class("act").from_pretrained(args.model_path)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = WoodPickEnv(render_images=True, episode_seconds=args.episode_seconds)
    home_radians = motor_to_sim(POLICY_HOME_MOTOR)
    gripper_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    stick_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "stick")
    dof_addresses = env.model.jnt_dofadr[env._joint_ids]
    recorder = VideoRecorder(args.output_dir)
    reports: list[dict] = []
    viewer = None
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(env.model, env.data)
    camera_viewer = CameraViewer("LeRobot policy cameras") if args.camera_view != "none" else None

    try:
        for episode in range(1, args.num_episodes + 1):
            observation, _ = env.reset(
                seed=args.seed + episode - 1, options={"joint_positions": home_radians}
            )
            policy.reset()
            for _ in range(round(args.settle_seconds * CONTROL_FPS)):
                env.advance_physics(home_radians)
            observation = env.observe()
            initial_stick = env.data.xpos[stick_body_id].copy()
            recorder.start(episode)

            queue: deque[np.ndarray] = deque()
            raw_queue: deque[np.ndarray] = deque()
            inference_times: list[float] = []
            actual_trace: list[np.ndarray] = []
            command_trace: list[np.ndarray] = []
            predicted_trace: list[np.ndarray] = []
            joint_rad_trace: list[np.ndarray] = []
            velocity_trace: list[np.ndarray] = []
            stick_trace: list[np.ndarray] = []
            ee_pose_trace: list[np.ndarray] = []
            clipped_trace: list[np.ndarray] = []
            chunk_trace: list[int] = []
            chunk_id = -1
            success = False
            viewer_closed = False
            wall_started = time.perf_counter()

            for step in range(round(args.episode_seconds * CONTROL_FPS)):
                step_started = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    viewer_closed = True
                    break
                if not queue or (args.replan_overlap and len(queue) <= args.replan_overlap):
                    chunk_id += 1
                    incoming, inference_seconds = predict_chunk(
                        policy, preprocessor, postprocessor, observation,
                        args.actions_per_chunk, args.device,
                    )
                    queue = overlap_chunks(queue, incoming, args.blend_new_weight)
                    raw_queue = deque(action.copy() for action in incoming)
                    inference_times.append(inference_seconds)
                    print(
                        f"episode={episode} step={step} chunk={chunk_id} "
                        f"inference={inference_seconds * 1000:.1f} ms queue={len(queue)}",
                        flush=True,
                    )

                effective_motor = queue.popleft()
                predicted_motor = raw_queue.popleft()
                current_motor = sim_to_motor(env.joint_positions)
                delta = effective_motor - current_motor
                safe_delta = np.clip(delta, -args.max_relative_target, args.max_relative_target)
                clipped = delta != safe_delta
                safe_motor = current_motor + safe_delta

                observation, _, success, truncated, _ = env.step(motor_to_sim(safe_motor))
                recorder.append(observation)
                actual_trace.append(sim_to_motor(env.joint_positions))
                command_trace.append(safe_motor.copy())
                predicted_trace.append(predicted_motor.copy())
                joint_rad_trace.append(env.joint_positions)
                velocity_trace.append(env.data.qvel[dof_addresses].copy())
                stick_trace.append(env.data.xpos[stick_body_id].copy())
                ee_pose_trace.append(
                    np.concatenate((env.data.xpos[gripper_body_id], env.data.xquat[gripper_body_id])).copy()
                )
                clipped_trace.append(clipped)
                chunk_trace.append(chunk_id)
                if viewer is not None:
                    viewer.sync()
                if camera_viewer is not None:
                    if args.camera_view == "both":
                        camera_frame = np.concatenate(
                            (
                                observation["observation.images.front"],
                                observation["observation.images.wrist"],
                            ),
                            axis=1,
                        )
                    else:
                        camera_frame = observation[f"observation.images.{args.camera_view}"]
                    if not camera_viewer.update(camera_frame):
                        viewer_closed = True
                if viewer is not None or camera_viewer is not None:
                    time.sleep(max(0.0, 1.0 / CONTROL_FPS - (time.perf_counter() - step_started)))
                if success or truncated or viewer_closed:
                    break

            if not actual_trace:
                recorder.abort()
                break
            result = "success" if success else "failed"
            video_paths = recorder.finish(result)
            trace_path = args.output_dir / f"episode_{episode:03d}_diagnostics.npz"
            np.savez_compressed(
                trace_path,
                actual_motor=np.stack(actual_trace),
                commanded_motor=np.stack(command_trace),
                predicted_motor=np.stack(predicted_trace),
                actual_joint_position_rad=np.stack(joint_rad_trace),
                joint_velocity_radians_s=np.stack(velocity_trace),
                stick_position_world=np.stack(stick_trace),
                ee_pose_world=np.stack(ee_pose_trace),
                clipped_joint=np.stack(clipped_trace),
                chunk_id=np.asarray(chunk_trace, dtype=np.int32),
                joint_names=np.asarray(JOINT_NAMES),
                fps=np.asarray([CONTROL_FPS], dtype=np.float32),
            )
            actual = np.stack(actual_trace)
            commanded = np.stack(command_trace)
            report = {
                "episode": episode,
                "success": bool(success),
                "frames": step + 1,
                "sim_duration_s": (step + 1) / CONTROL_FPS,
                "wall_duration_s": time.perf_counter() - wall_started,
                "initial_stick_position": initial_stick.tolist(),
                "final_stick_position": env.data.xpos[stick_body_id].tolist(),
                "max_stick_height": float(np.max(np.stack(stick_trace)[:, 2])),
                "relative_target_clipped_steps": int(np.any(np.stack(clipped_trace), axis=1).sum()),
                "tracking_error_motor_mean": np.abs(commanded - actual).mean(axis=0).tolist(),
                "inference_ms_mean": float(np.mean(inference_times) * 1000),
                "inference_ms_max": float(np.max(inference_times) * 1000),
                "videos": video_paths,
                "diagnostics": str(trace_path),
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if viewer_closed:
                break
    finally:
        recorder.abort()
        if viewer is not None:
            viewer.close()
        if camera_viewer is not None:
            camera_viewer.close()
        env.close()
        if reports:
            metrics = {
                "model_path": str(args.model_path.resolve()),
                "appearance_path": str(env.appearance_path.resolve()),
                "device": args.device,
                "actions_per_chunk": args.actions_per_chunk,
                "replan_overlap": args.replan_overlap,
                "blend_new_weight": args.blend_new_weight,
                "max_relative_target": args.max_relative_target,
                "episodes": reports,
                "success_rate": sum(report["success"] for report in reports) / len(reports),
            }
            metrics_path = args.output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
