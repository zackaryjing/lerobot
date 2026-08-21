#!/usr/bin/env python3
"""Run the sampled-grasp Mink IK baseline in the MuJoCo wood-pick scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import imageio.v2 as imageio
import mujoco
import numpy as np

from mujoco_wood_pick.env import CONTROL_FPS, JOINT_NAMES, WoodPickEnv
from mujoco_wood_pick.scripted_control import (
    SCRIPTED_HOME,
    ScriptedPlan,
    ScriptedPlanner,
    sample_stick_pose,
    set_grasp_weld,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/mujoco_scripted_control"


class VideoRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.writers: dict[str, object] = {}
        self.paths: dict[str, Path] = {}

    def start(self, episode: int, replay: int = 0) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"episode_{episode:03d}"
        if replay:
            prefix += f"_replay_{replay:03d}"
        for camera in ("front", "wrist"):
            path = self.output_dir / f"{prefix}_{camera}_pending.mp4"
            self.paths[camera] = path
            self.writers[camera] = imageio.get_writer(
                path,
                fps=round(CONTROL_FPS),
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

    def append(self, observation: dict[str, np.ndarray]) -> None:
        for camera, writer in self.writers.items():
            writer.append_data(observation[f"observation.images.{camera}"])

    def finish(self, result: str) -> list[str]:
        paths: list[str] = []
        for writer in self.writers.values():
            writer.close()
        for pending in self.paths.values():
            final = pending.with_name(pending.name.replace("pending", result))
            pending.rename(final)
            paths.append(str(final))
            print(f"Saved video: {final}", flush=True)
        self.writers.clear()
        self.paths.clear()
        return paths

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


class RestartController:
    """Thread-safe bridge between GUI callbacks and the simulation loop."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def consume(self) -> bool:
        requested = self._event.is_set()
        self._event.clear()
        return requested


class CameraViewer:
    def __init__(self, restart: RestartController) -> None:
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._root = tk.Tk()
        self._root.title("Scripted control cameras: front | wrist")
        controls = tk.Frame(self._root)
        controls.pack(fill="x", padx=6, pady=6)
        tk.Button(controls, text="重新开始 (R)", command=restart.request).pack(side="left")
        self._status = tk.StringVar(value="准备中")
        tk.Label(controls, textvariable=self._status).pack(side="left", padx=12)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._photo = None
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda _event: self.close())
        self._root.bind("q", lambda _event: self.close())
        self._root.bind("r", lambda _event: restart.request())

    def set_status(self, text: str) -> None:
        if not self._closed:
            self._status.set(text)

    def pump(self) -> bool:
        if self._closed:
            return False
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
        return not self._closed

    def update(self, observation: dict[str, np.ndarray], selection: str) -> bool:
        if self._closed:
            return False
        from PIL import Image, ImageTk

        if selection == "both":
            frame = np.concatenate(
                (
                    observation["observation.images.front"],
                    observation["observation.images.wrist"],
                ),
                axis=1,
            )
        else:
            frame = observation[f"observation.images.{selection}"]
        try:
            self._photo = ImageTk.PhotoImage(Image.fromarray(frame))
            self._label.configure(image=self._photo)
            self.pump()
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


def snapshot_viewer_camera(viewer) -> dict[str, object]:
    """Copy every user-controlled 3-D camera field."""
    with viewer.lock():
        camera = viewer.cam
        return {
            "type": camera.type,
            "fixedcamid": camera.fixedcamid,
            "trackbodyid": camera.trackbodyid,
            "lookat": camera.lookat.copy(),
            "distance": camera.distance,
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
            "orthographic": camera.orthographic,
        }


def restore_viewer_camera(viewer, state: dict[str, object]) -> None:
    with viewer.lock():
        camera = viewer.cam
        camera.type = state["type"]
        camera.fixedcamid = state["fixedcamid"]
        camera.trackbodyid = state["trackbodyid"]
        camera.lookat[:] = state["lookat"]
        camera.distance = state["distance"]
        camera.azimuth = state["azimuth"]
        camera.elevation = state["elevation"]
        camera.orthographic = state["orthographic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--seeds-per-candidate", type=int, default=3)
    parser.add_argument(
        "--grasp-depth-mm",
        type=float,
        default=10.0,
        help="Lower the grasp TCP below the nominal stick-section centre.",
    )
    parser.add_argument(
        "--closed-gripper-deg",
        type=float,
        default=-3.1,
        help="Position target used while holding the stick.",
    )
    parser.add_argument("--max-planning-attempts", type=int, default=12)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--camera-view", choices=("none", "front", "wrist", "both"), default="none"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace to 30 Hz even when no GUI is open.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--grasp-assist",
        action="store_true",
        help="Explicitly enable the legacy contact-triggered weld (disabled by default).",
    )
    parser.add_argument(
        "--no-grasp-assist",
        action="store_false",
        dest="grasp_assist",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def plan_report(plan: ScriptedPlan) -> dict:
    return {
        "candidate_index": plan.candidate_index,
        "seed_index": plan.seed_index,
        "total_steps": plan.total_steps,
        "duration_s": plan.total_steps / CONTROL_FPS,
        "station_offset_m": plan.candidate.station_offset,
        "tilt_deg": float(np.rad2deg(plan.candidate.tilt_rad)),
        "wrist_flip": plan.candidate.wrist_flip,
        "contact_tcp": plan.candidate.contact_tcp.tolist(),
        "pregrasp_tcp": plan.candidate.pregrasp_tcp.tolist(),
        "lift_tcp": plan.candidate.lift_tcp.tolist(),
        "above_box_tcp": plan.candidate.above_box_tcp.tolist(),
        "lower_box_tcp": plan.candidate.lower_box_tcp.tolist(),
        "ik": {
            phase: {
                "position_error_mm": result.position_error * 1000.0,
                "approach_error_deg": float(np.rad2deg(result.approach_error_rad)),
                "closing_error_deg": float(np.rad2deg(result.closing_error_rad)),
                "iterations": result.iterations,
                "joint_positions_rad": result.joint_positions.tolist(),
            }
            for phase, result in plan.ik_report.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("num-episodes must be positive")
    if not 0.0 <= args.grasp_depth_mm <= 15.0:
        raise ValueError("grasp-depth-mm must be in [0, 15]")
    if not -10.0 <= args.closed_gripper_deg <= 10.0:
        raise ValueError("closed-gripper-deg must be in [-10, 10]")
    rng = np.random.default_rng(args.seed)
    render_images = not args.no_video or args.camera_view != "none"
    env = WoodPickEnv(render_images=render_images, episode_seconds=30.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = None if args.no_video else VideoRecorder(args.output_dir)
    restart = RestartController()
    camera_viewer = CameraViewer(restart) if args.camera_view != "none" else None
    viewer = None
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        def on_viewer_key(keycode: int) -> None:
            if keycode == ord("R"):
                restart.request()

        viewer = mujoco_viewer.launch_passive(
            env.model,
            env.data,
            key_callback=on_viewer_key,
        )
    controls = camera_viewer

    stick_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "stick")
    stick_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom")
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    reports: list[dict] = []

    try:
        for episode in range(1, args.num_episodes + 1):
            plan = None
            planning_failures: list[str] = []
            planning_started = time.perf_counter()
            for attempt in range(1, args.max_planning_attempts + 1):
                stick_position, stick_quaternion = sample_stick_pose(rng)
                env.reset(
                    seed=args.seed + episode - 1,
                    options={
                        "joint_positions": SCRIPTED_HOME,
                        "stick_position": stick_position,
                        "stick_quaternion": stick_quaternion,
                    },
                )
                env.advance_physics(SCRIPTED_HOME, round(0.5 * CONTROL_FPS))
                try:
                    planner = ScriptedPlanner(
                        env,
                        rng,
                        grasp_depth=args.grasp_depth_mm / 1000.0,
                        closed_gripper=np.deg2rad(args.closed_gripper_deg),
                    )
                    plan = planner.plan(args.candidate_count, args.seeds_per_candidate)
                    break
                except RuntimeError as error:
                    planning_failures.append(str(error))
                    print(f"Planning attempt {attempt} rejected: {error}", flush=True)
            if plan is None:
                raise RuntimeError(
                    f"Failed to plan episode {episode} after {args.max_planning_attempts} object poses"
                )
            planning_seconds = time.perf_counter() - planning_started
            print(
                f"Episode {episode}: planned {plan.total_steps} steps in {planning_seconds:.2f}s; "
                f"candidate={plan.candidate_index}, station={plan.candidate.station_offset:+.3f}m, "
                f"tilt={np.rad2deg(plan.candidate.tilt_rad):+.1f}deg",
                flush=True,
            )
            for phase, result in plan.ik_report.items():
                print(
                    f"  IK {phase:>10}: pos={result.position_error * 1000:5.1f}mm "
                    f"approach={np.rad2deg(result.approach_error_rad):5.1f}deg "
                    f"closing={np.rad2deg(result.closing_error_rad):5.1f}deg",
                    flush=True,
                )

            replay = 0
            stopped = False
            while True:
                # A replay deliberately uses the exact same sampled object pose and
                # exact same planned joint trajectory. Only the MuJoCo physics state
                # is reset, making repeated visual comparisons meaningful.
                if replay:
                    camera_state = snapshot_viewer_camera(viewer) if viewer is not None else None
                    env.reset(
                        seed=args.seed + episode - 1,
                        options={
                            "joint_positions": SCRIPTED_HOME,
                            "stick_position": stick_position,
                            "stick_quaternion": stick_quaternion,
                        },
                    )
                    env.advance_physics(SCRIPTED_HOME, round(0.5 * CONTROL_FPS))
                    if viewer is not None and camera_state is not None:
                        restore_viewer_camera(viewer, camera_state)
                        viewer.sync()

                restart.consume()  # Clear a stale double-click before starting.
                if recorder is not None:
                    recorder.start(episode, replay)
                actual_trace: list[np.ndarray] = []
                command_trace: list[np.ndarray] = []
                stick_trace: list[np.ndarray] = []
                tcp_trace: list[np.ndarray] = []
                phase_trace: list[str] = []
                contact_count_trace: list[int] = []
                stick_contact_count_trace: list[int] = []
                stick_normal_force_trace: list[float] = []
                stick_tangent_force_trace: list[float] = []
                fixed_jaw_normal_force_trace: list[float] = []
                moving_jaw_normal_force_trace: list[float] = []
                platform_normal_force_trace: list[float] = []
                stick_max_penetration_trace: list[float] = []
                gripper_actuator_force_trace: list[float] = []
                gripper_constraint_force_trace: list[float] = []
                grasp_assist_used = False
                restart_requested = False
                wall_started = time.perf_counter()
                if controls is not None:
                    controls.set_status(f"第 {replay + 1} 次运行")

                for phase in plan.phases:
                    if phase.name == "release" and grasp_assist_used:
                        set_grasp_weld(env, False)
                    print(
                        f"  phase: {phase.name} ({len(phase.joint_targets)} steps)",
                        flush=True,
                    )
                    if controls is not None:
                        controls.set_status(f"运行中：{phase.name}")
                    phase_stick_contacts: set[str] = set()
                    phase_jaw_contact = False
                    for target in phase.joint_targets:
                        step_started = time.perf_counter()
                        observation, _, _, _, _ = env.step(target)
                        if recorder is not None:
                            recorder.append(observation)
                        if viewer is not None:
                            if not viewer.is_running():
                                stopped = True
                            else:
                                viewer.sync()
                        if camera_viewer is not None:
                            if not camera_viewer.update(observation, args.camera_view):
                                stopped = True

                        site_rotation = env.data.site_xmat[site_id].reshape(3, 3)
                        tcp = (
                            env.data.site_xpos[site_id]
                            + site_rotation @ planner.ik.tcp_offset_in_site
                        )
                        actual_trace.append(env.joint_positions)
                        command_trace.append(target.copy())
                        stick_trace.append(env.data.xpos[stick_body_id].copy())
                        tcp_trace.append(tcp.copy())
                        phase_trace.append(phase.name)
                        contact_count_trace.append(env.data.ncon)
                        stick_contacts = 0
                        stick_normal_force = 0.0
                        stick_tangent_force = 0.0
                        fixed_jaw_normal_force = 0.0
                        moving_jaw_normal_force = 0.0
                        platform_normal_force = 0.0
                        stick_max_penetration = 0.0
                        for contact_index in range(env.data.ncon):
                            contact = env.data.contact[contact_index]
                            if contact.geom1 == stick_geom_id:
                                other_geom = int(contact.geom2)
                            elif contact.geom2 == stick_geom_id:
                                other_geom = int(contact.geom1)
                            else:
                                continue
                            stick_contacts += 1
                            contact_force = np.zeros(6, dtype=np.float64)
                            mujoco.mj_contactForce(
                                env.model, env.data, contact_index, contact_force
                            )
                            normal_force = max(0.0, float(contact_force[0]))
                            stick_normal_force += normal_force
                            stick_tangent_force += float(
                                np.linalg.norm(contact_force[1:3])
                            )
                            stick_max_penetration = max(
                                stick_max_penetration, max(0.0, -float(contact.dist))
                            )
                            name = mujoco.mj_id2name(
                                env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                            )
                            phase_stick_contacts.add(name or f"geom_{other_geom}")
                            other_body = int(env.model.geom_bodyid[other_geom])
                            other_body_name = mujoco.mj_id2name(
                                env.model, mujoco.mjtObj.mjOBJ_BODY, other_body
                            )
                            if other_body_name in ("gripper", "moving_jaw_so101_v1"):
                                phase_jaw_contact = True
                            if other_body_name == "gripper":
                                fixed_jaw_normal_force += normal_force
                            elif other_body_name == "moving_jaw_so101_v1":
                                moving_jaw_normal_force += normal_force
                            elif other_body == 0:
                                platform_normal_force += normal_force
                        stick_contact_count_trace.append(stick_contacts)
                        stick_normal_force_trace.append(stick_normal_force)
                        stick_tangent_force_trace.append(stick_tangent_force)
                        fixed_jaw_normal_force_trace.append(fixed_jaw_normal_force)
                        moving_jaw_normal_force_trace.append(moving_jaw_normal_force)
                        platform_normal_force_trace.append(platform_normal_force)
                        stick_max_penetration_trace.append(stick_max_penetration)
                        gripper_actuator_force_trace.append(
                            float(env.data.actuator_force[env._actuator_ids[-1]])
                        )
                        gripper_constraint_force_trace.append(
                            float(
                                env.data.qfrc_constraint[
                                    env.model.jnt_dofadr[env._joint_ids[-1]]
                                ]
                            )
                        )

                        if restart.consume():
                            restart_requested = True
                        if args.realtime or viewer is not None or camera_viewer is not None:
                            time.sleep(
                                max(
                                    0.0,
                                    1.0 / CONTROL_FPS
                                    - (time.perf_counter() - step_started),
                                )
                            )
                        if stopped or restart_requested:
                            break
                    if phase_stick_contacts:
                        print(
                            f"    stick contacts: {sorted(phase_stick_contacts)}",
                            flush=True,
                        )
                    if (
                        phase.name == "close"
                        and phase_jaw_contact
                        and args.grasp_assist
                        and not restart_requested
                    ):
                        set_grasp_weld(env, True)
                        grasp_assist_used = True
                        print("    grasp assist: weld activated after jaw contact", flush=True)
                    if stopped or restart_requested:
                        break

                if restart_requested:
                    if recorder is not None:
                        recorder.finish("restarted")
                    replay += 1
                    print(
                        "Restart requested: restoring the same initial state and trajectory.",
                        flush=True,
                    )
                    continue

                success = env.is_success()
                result_name = "success" if success else "failed"
                videos = recorder.finish(result_name) if recorder is not None else []
                prefix = f"episode_{episode:03d}"
                if replay:
                    prefix += f"_replay_{replay:03d}"
                diagnostics_path = args.output_dir / f"{prefix}_diagnostics.npz"
                np.savez_compressed(
                    diagnostics_path,
                    actual_joint_position_rad=np.stack(actual_trace),
                    commanded_joint_position_rad=np.stack(command_trace),
                    stick_position_world=np.stack(stick_trace),
                    tcp_position_world=np.stack(tcp_trace),
                    phase=np.asarray(phase_trace),
                    contact_count=np.asarray(contact_count_trace, dtype=np.int32),
                    stick_contact_count=np.asarray(
                        stick_contact_count_trace, dtype=np.int32
                    ),
                    stick_normal_force_n=np.asarray(
                        stick_normal_force_trace, dtype=np.float64
                    ),
                    stick_tangent_force_n=np.asarray(
                        stick_tangent_force_trace, dtype=np.float64
                    ),
                    fixed_jaw_normal_force_n=np.asarray(
                        fixed_jaw_normal_force_trace, dtype=np.float64
                    ),
                    moving_jaw_normal_force_n=np.asarray(
                        moving_jaw_normal_force_trace, dtype=np.float64
                    ),
                    platform_normal_force_n=np.asarray(
                        platform_normal_force_trace, dtype=np.float64
                    ),
                    stick_max_penetration_m=np.asarray(
                        stick_max_penetration_trace, dtype=np.float64
                    ),
                    gripper_actuator_force_nm=np.asarray(
                        gripper_actuator_force_trace, dtype=np.float64
                    ),
                    gripper_constraint_force_nm=np.asarray(
                        gripper_constraint_force_trace, dtype=np.float64
                    ),
                    joint_names=np.asarray(JOINT_NAMES),
                    fps=np.asarray([CONTROL_FPS], dtype=np.float32),
                )
                stick_array = np.stack(stick_trace)
                episode_report = {
                    "episode": episode,
                    "replay": replay,
                    "success": bool(success),
                    "stopped_by_user": stopped,
                    "frames": len(actual_trace),
                    "sim_duration_s": len(actual_trace) / CONTROL_FPS,
                    "wall_duration_s": time.perf_counter() - wall_started,
                    "planning_seconds": planning_seconds,
                    "planning_rejections": len(planning_failures),
                    "grasp_depth_mm": args.grasp_depth_mm,
                    "closed_gripper_deg": args.closed_gripper_deg,
                    "grasp_assist_enabled": args.grasp_assist,
                    "grasp_assist_used": grasp_assist_used,
                    "stick_initial_position": stick_array[0].tolist(),
                    "stick_final_position": stick_array[-1].tolist(),
                    "stick_max_height": float(stick_array[:, 2].max()),
                    "plan": plan_report(plan),
                    "videos": videos,
                    "diagnostics": str(diagnostics_path),
                }
                reports.append(episode_report)
                print(json.dumps(episode_report, ensure_ascii=False, indent=2), flush=True)

                # For the usual single-episode interactive inspection, keep the
                # final state visible. The user can replay it indefinitely or
                # close either GUI to finish the process.
                interactive_replay = (
                    args.num_episodes == 1
                    and (viewer is not None or camera_viewer is not None)
                    and not stopped
                )
                if interactive_replay:
                    if controls is not None:
                        controls.set_status(
                            f"本次{'成功' if success else '失败'}；点击重新开始或按 R"
                        )
                    print(
                        "Run finished. Press R/click Restart to replay; close the GUI to exit.",
                        flush=True,
                    )
                    while True:
                        if viewer is not None:
                            if not viewer.is_running():
                                stopped = True
                                break
                            viewer.sync()
                        if controls is not None and not controls.pump():
                            stopped = True
                            break
                        if restart.consume():
                            restart_requested = True
                            break
                        time.sleep(1.0 / CONTROL_FPS)
                    if restart_requested:
                        replay += 1
                        continue
                break
            if stopped:
                break
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        if camera_viewer is not None:
            camera_viewer.close()
        env.close()
        if reports:
            metrics = {
                "seed": args.seed,
                "candidate_count": args.candidate_count,
                "seeds_per_candidate": args.seeds_per_candidate,
                "grasp_depth_mm": args.grasp_depth_mm,
                "closed_gripper_deg": args.closed_gripper_deg,
                "grasp_assist_enabled": args.grasp_assist,
                "episodes": reports,
                "success_rate": sum(report["success"] for report in reports) / len(reports),
            }
            metrics_path = args.output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
