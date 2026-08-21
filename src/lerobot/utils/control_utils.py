# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import os
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


class _TerminalKeyboardListener:
    """Read LeRobot recording shortcuts directly from a POSIX terminal."""

    def __init__(self, on_key):
        import termios

        self._on_key = on_key
        self._fd = sys.stdin.fileno()
        self._termios = termios
        self._old_settings = None
        self._restore_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lerobot-terminal-keyboard", daemon=True)

    def start(self):
        import tty

        self._old_settings = self._termios.tcgetattr(self._fd)
        # cbreak disables line buffering and echo while keeping Ctrl+C as SIGINT.
        tty.setcbreak(self._fd)
        self._thread.start()
        return self

    def _restore_terminal(self):
        with self._restore_lock:
            if self._old_settings is not None:
                settings, self._old_settings = self._old_settings, None
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, settings)

    def _run(self):
        import select

        buffer = b""
        escape_started_at = None
        try:
            while not self._stop_event.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.02)
                if readable:
                    data = os.read(self._fd, 32)
                    if data:
                        buffer += data
                        if buffer.startswith(b"\x1b") and escape_started_at is None:
                            escape_started_at = time.monotonic()

                while buffer:
                    if buffer.startswith(b"\x1b[C"):
                        self._on_key("right")
                        buffer = buffer[3:]
                        escape_started_at = time.monotonic() if buffer.startswith(b"\x1b") else None
                    elif buffer.startswith(b"\x1b[D"):
                        self._on_key("left")
                        buffer = buffer[3:]
                        escape_started_at = time.monotonic() if buffer.startswith(b"\x1b") else None
                    elif buffer.startswith(b"\x1b"):
                        # Wait briefly to distinguish a standalone Escape key from an arrow sequence.
                        if escape_started_at is None:
                            escape_started_at = time.monotonic()
                        if len(buffer) < 3 and time.monotonic() - escape_started_at < 0.15:
                            break
                        if buffer == b"\x1b":
                            self._on_key("esc")
                            buffer = b""
                        else:
                            # Ignore unsupported terminal escape sequences (for example up/down arrows).
                            buffer = buffer[3:]
                        escape_started_at = time.monotonic() if buffer.startswith(b"\x1b") else None
                    else:
                        # Recording control only uses arrows and Escape; discard other terminal input.
                        buffer = buffer[1:]
        except Exception:
            logging.exception("Terminal keyboard listener stopped unexpectedly.")
        finally:
            self._restore_terminal()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._restore_terminal()


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    def handle_key(key_name: str):
        if key_name == "right":
            print("Right arrow key pressed. Exiting loop...")
            events["exit_early"] = True
        elif key_name == "left":
            print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif key_name == "esc":
            print("Escape key pressed. Stopping data recording...")
            events["stop_recording"] = True
            events["exit_early"] = True

    # Wayland blocks pynput's global keyboard hook. Read escape sequences from the
    # focused terminal instead, which also avoids printing literals such as "^[[C".
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and sys.stdin.isatty():
        listener = _TerminalKeyboardListener(handle_key).start()
        logging.info("Using terminal keyboard controls for the Wayland session (focus this terminal).")
        return listener, events

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                handle_key("right")
            elif key == keyboard.Key.left:
                handle_key("left")
            elif key == keyboard.Key.esc:
                handle_key("esc")
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
