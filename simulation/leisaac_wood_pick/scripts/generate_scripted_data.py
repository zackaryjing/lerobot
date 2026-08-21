"""Generate successful SO-101 wooden-stick demonstrations with a state machine.

The first output format is Isaac Lab HDF5.  It preserves simulator state and
actions for replay and can subsequently be converted to LeRobot Dataset v3 by
LeIsaac's standard converter.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LEISAAC_ASSETS_DIR = PROJECT_DIR / "assets" / "leisaac"
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(LEISAAC_ASSETS_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_demos", type=int, default=1, help="Number of successful episodes to export.")
parser.add_argument("--max_attempts", type=int, default=25, help="Stop instead of retrying forever.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--record", action="store_true", help="Write successful demonstrations to HDF5.")
parser.add_argument(
    "--dataset_file",
    type=Path,
    default=Path("datasets/wood_pick_scripted.hdf5"),
    help="New HDF5 output path. Existing files are never overwritten.",
)
parser.add_argument("--realtime", action="store_true", help="Throttle to the scene's 30 Hz control rate.")
parser.add_argument(
    "--video_dir",
    type=Path,
    default=None,
    help="If set, save one 30 FPS front MP4 and wrist MP4 for every attempted episode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import DatasetExportMode, EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

import leisaac_wood_pick  # noqa: F401
from leisaac_wood_pick.tasks.wood_pick import TASK_ID
from leisaac_wood_pick.tasks.wood_pick import mdp as wood_mdp
from leisaac_wood_pick.tasks.wood_pick.controllers import (
    WeightedDifferentialIKControllerCfg,
    WeightedDifferentialInverseKinematicsAction,
)
from leisaac_wood_pick.tasks.wood_pick.recorders import PostStepJointPositionTargetsRecorderCfg
from leisaac_wood_pick.tasks.wood_pick.scripted_policy import WoodPickStateMachine


class EpisodeVideoRecorder:
    """Stream both camera observations to per-attempt H.264 videos."""

    def __init__(self, directory: Path | None, fps: int = 30) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self.fps = fps
        self.attempt = 0
        self._writers: dict[str, object] = {}
        self._temporary_paths: dict[str, Path] = {}

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def start(self, attempt: int) -> None:
        if not self.enabled or self._writers:
            return
        self.attempt = attempt
        self.directory.mkdir(parents=True, exist_ok=True)
        for camera in ("front", "wrist"):
            path = self.directory / f"episode_{attempt:03d}_{camera}_pending.mp4"
            self._temporary_paths[camera] = path
            self._writers[camera] = imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

    @staticmethod
    def _rgb_array(image: torch.Tensor) -> np.ndarray:
        array = image[0].detach().cpu().numpy()
        if array.shape[-1] == 4:
            array = array[..., :3]
        return np.clip(array, 0, 255).astype(np.uint8)

    def append(self, policy_observation: dict[str, torch.Tensor]) -> None:
        if not self._writers:
            return
        for camera, writer in self._writers.items():
            writer.append_data(self._rgb_array(policy_observation[camera]))

    def finish(self, result: str) -> None:
        if not self._writers:
            return
        for writer in self._writers.values():
            writer.close()
        for camera, temporary_path in self._temporary_paths.items():
            final_path = self.directory / f"episode_{self.attempt:03d}_{camera}_{result}.mp4"
            temporary_path.rename(final_path)
            print(f"Saved video: {final_path}", flush=True)
        self._writers.clear()
        self._temporary_paths.clear()


def _constant_success_term(success: bool) -> TerminationTermCfg:
    return TerminationTermCfg(
        func=lambda env: torch.full((env.num_envs,), success, dtype=torch.bool, device=env.device)
    )


def _set_episode_result(env, success: bool) -> None:
    env.termination_manager.set_term_cfg("success", _constant_success_term(success))
    env.termination_manager.compute()


def _configure_recording(env_cfg) -> None:
    if not args_cli.record:
        env_cfg.recorders = None
        return
    dataset_path = args_cli.dataset_file.resolve()
    if dataset_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dataset_path}")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    env_cfg.recorders.dataset_filename = dataset_path.stem
    # Replace the internal 8D Cartesian command with the 6D joint target that
    # has the same semantics and ordering as a real SO-101 LeRobot dataset.
    env_cfg.recorders.record_pre_step_actions = None
    env_cfg.recorders.record_post_step_processed_actions = None
    env_cfg.recorders.record_post_step_joint_targets = PostStepJointPositionTargetsRecorderCfg()


def main() -> None:
    robot_asset = LEISAAC_ASSETS_DIR / "robots" / "so101_follower.usd"
    if not robot_asset.is_file():
        raise FileNotFoundError(f"Missing SO-101 asset: {robot_asset}; run scripts/download_assets.py")

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101_state_machine")
    env_cfg.actions.arm_action.controller = WeightedDifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.04},
        orientation_weight=0.0,
    )
    env_cfg.actions.arm_action.class_type = WeightedDifferentialInverseKinematicsAction
    env_cfg.actions.arm_action.body_name = "gripper"
    env_cfg.actions.arm_action.body_offset = DifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.030, 0.010, -0.100)
    )
    env_cfg.events.scripted_ready_pose = EventTermCfg(
        func=wood_mdp.reset_robot_to_reachable_seed,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("stick"),
        },
    )
    env_cfg.seed = args_cli.seed
    # The state machine owns episode boundaries.  Keep a placeholder success
    # term so the recorder receives an explicit true/false result at reset.
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = _constant_success_term(False)
    _configure_recording(env_cfg)

    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    sm = WoodPickStateMachine()
    sm.setup(env)
    env.reset()
    sm.reset()
    _set_episode_result(env, False)

    attempts = 0
    successes = 0
    last_phase = sm.phase
    next_wall_step = time.monotonic()
    video_recorder = EpisodeVideoRecorder(args_cli.video_dir, fps=30)

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and successes < args_cli.num_demos:
                if sm.is_episode_done:
                    attempts += 1
                    success = sm.check_success(env)
                    video_recorder.finish("success" if success else "failed")
                    _set_episode_result(env, success)
                    if success:
                        successes += 1
                        print(f"Episode {attempts}: success ({successes}/{args_cli.num_demos})", flush=True)
                    else:
                        reason = sm.failure_reason or "stick was not settled fully inside the box"
                        print(f"Episode {attempts}: failed: {reason}", flush=True)

                    # Reset commits/discards the just-finished episode according
                    # to EXPORT_SUCCEEDED_ONLY, then triggers a new randomized pose.
                    env.reset()
                    sm.reset()
                    _set_episode_result(env, False)
                    last_phase = sm.phase
                    if attempts >= args_cli.max_attempts and successes < args_cli.num_demos:
                        raise RuntimeError(
                            f"Reached --max_attempts={args_cli.max_attempts} with only {successes} successes"
                        )
                    continue

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")
                action = sm.get_action(env)
                video_recorder.start(attempts + 1)
                observation, _, _, _, _ = env.step(action)
                video_recorder.append(observation["policy"])
                sm.advance()
                if sm.phase is not last_phase:
                    print(f"  phase: {last_phase.name} -> {sm.phase.name}", flush=True)
                    last_phase = sm.phase

                if args_cli.realtime:
                    next_wall_step += env.step_dt
                    delay = next_wall_step - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    elif delay < -1.0:
                        next_wall_step = time.monotonic()
    finally:
        video_recorder.finish("incomplete")
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()

    if args_cli.record:
        print(f"Saved {successes} successful demonstrations to {args_cli.dataset_file.resolve()}")


if __name__ == "__main__":
    main()
