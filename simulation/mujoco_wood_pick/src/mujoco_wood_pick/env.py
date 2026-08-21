"""Gymnasium-style wrapper around the measured MuJoCo workcell."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import gymnasium as gym

# WSLg otherwise prefers the Intel iGPU on this dual-GPU laptop. Users can
# override this before importing the package on a different machine.
if "microsoft" in platform.release().lower():
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "wood_pick.xml"
DEFAULT_APPEARANCE_PATH = PROJECT_DIR / "config" / "appearance.json"
LOCAL_APPEARANCE_PATH = PROJECT_DIR / "config" / "appearance.local.json"

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CAMERA_NAMES = ("front", "wrist")
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONTROL_FPS = 30.0
PHYSICS_SUBSTEPS = 16
EPISODE_SECONDS = 25.0

# Folded reset used by the previous Isaac scene and by the real policy tests.
HOME_QPOS = np.deg2rad(np.array((0.0, -99.0, 90.0, 53.0, 0.0, 7.0), dtype=np.float64))
STICK_START_POS = np.array((0.235, 0.362, 0.05875), dtype=np.float64)
STICK_START_QUAT = np.array((2**-0.5, 0.0, 2**-0.5, 0.0), dtype=np.float64)
STICK_HALF_SIZE = np.array((0.00575, 0.00575, 0.040), dtype=np.float64)
BOX_INNER_MIN = np.array((-0.104, 0.383, 0.003), dtype=np.float64)
BOX_INNER_MAX = np.array((0.040, 0.477, 0.085), dtype=np.float64)


class WoodPickEnv(gym.Env):
    """Single SO-101 environment with LeRobot-compatible observations.

    Actions are six physical joint positions in radians. Observations use HWC
    uint8 RGB arrays under the same dotted keys used by LeRobot datasets.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_images: bool = True,
        episode_seconds: float = EPISODE_SECONDS,
        appearance_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_images = render_images
        self.max_steps = round(episode_seconds * CONTROL_FPS)
        self.control_step = 0
        self._renderer: mujoco.Renderer | None = None
        self._visual_options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._visual_options)
        self._visual_options.geomgroup[:] = 0
        self._visual_options.geomgroup[:3] = 1
        # The bright scripted_tcp site is a 3-D-viewer debugging aid. Never
        # leak it into front/wrist observations consumed by a policy.
        self._visual_options.sitegroup[2] = 0

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._stick_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "stick_free")
        self._stick_qpos_address = int(self.model.jnt_qposadr[self._stick_joint_id])
        self._stick_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "stick")

        if appearance_path is None:
            appearance_path = LOCAL_APPEARANCE_PATH if LOCAL_APPEARANCE_PATH.is_file() else DEFAULT_APPEARANCE_PATH
        self.appearance_path = Path(appearance_path)
        if self.appearance_path.is_file():
            self.apply_appearance(self.appearance_path)

        control_range = self.model.actuator_ctrlrange[self._actuator_ids].astype(np.float32)
        self.action_space = spaces.Box(control_range[:, 0], control_range[:, 1], dtype=np.float32)
        observation_spaces: dict[str, spaces.Space[Any]] = {
            "observation.state": spaces.Box(-np.inf, np.inf, shape=(len(JOINT_NAMES),), dtype=np.float32),
        }
        if render_images:
            image_space = spaces.Box(0, 255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            observation_spaces |= {
                "observation.images.front": image_space,
                "observation.images.wrist": image_space,
            }
        self.observation_space = spaces.Dict(observation_spaces)

    def apply_appearance(self, path: str | Path) -> None:
        """Apply material and lighting overrides from a small JSON file."""
        path = Path(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        for name, values in config.get("materials", {}).items():
            material_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if material_id < 0:
                continue
            if "rgb" in values:
                self.model.mat_rgba[material_id, :3] = np.asarray(values["rgb"], dtype=np.float32)
            if "rgba" in values:
                self.model.mat_rgba[material_id] = np.asarray(values["rgba"], dtype=np.float32)
            if "roughness" in values:
                self.model.mat_roughness[material_id] = float(values["roughness"])

        lighting = config.get("lighting", {})
        if "headlight_ambient" in lighting:
            self.model.vis.headlight.ambient[:] = np.asarray(lighting["headlight_ambient"], dtype=np.float32)
        if "headlight_diffuse" in lighting:
            self.model.vis.headlight.diffuse[:] = np.asarray(lighting["headlight_diffuse"], dtype=np.float32)
        for light_name in ("key", "fill"):
            key = f"{light_name}_diffuse"
            if key not in lighting:
                continue
            light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
            if light_id >= 0:
                self.model.light_diffuse[light_id] = np.asarray(lighting[key], dtype=np.float32)
        self.appearance_path = path

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._qpos_addresses].copy()

    def _set_stick_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        start = self._stick_qpos_address
        self.data.qpos[start : start + 3] = position
        self.data.qpos[start + 3 : start + 7] = quaternion / np.linalg.norm(quaternion)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        joint_positions = np.asarray(options.get("joint_positions", HOME_QPOS), dtype=np.float64)
        if joint_positions.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected reset joint_positions shape {(len(JOINT_NAMES),)}, "
                f"got {joint_positions.shape}"
            )
        self.data.qpos[self._qpos_addresses] = joint_positions
        self.data.ctrl[self._actuator_ids] = joint_positions
        self._set_stick_pose(
            np.asarray(options.get("stick_position", STICK_START_POS), dtype=np.float64),
            np.asarray(options.get("stick_quaternion", STICK_START_QUAT), dtype=np.float64),
        )
        self.control_step = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), self.info()

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Expected action shape {(len(JOINT_NAMES),)}, got {action.shape}")
        low = self.model.actuator_ctrlrange[self._actuator_ids, 0]
        high = self.model.actuator_ctrlrange[self._actuator_ids, 1]
        clipped = np.clip(action, low, high)
        self.data.ctrl[self._actuator_ids] = clipped
        return clipped

    def advance_physics(self, action: np.ndarray, control_steps: int = 1) -> None:
        """Advance without rendering, useful for settling and benchmarks."""
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=PHYSICS_SUBSTEPS * control_steps)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        clipped = self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=PHYSICS_SUBSTEPS)
        self.control_step += 1
        success = self.is_success()
        truncated = self.control_step >= self.max_steps
        info = self.info()
        info["action_clipped"] = not np.array_equal(np.asarray(action), clipped)
        return self.observe(), float(success), success, truncated, info

    def _get_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        return self._renderer

    def render_camera(self, camera: str) -> np.ndarray:
        if camera not in CAMERA_NAMES and camera != "overview":
            raise ValueError(f"Unknown camera {camera!r}; expected one of {CAMERA_NAMES + ('overview',)}")
        renderer = self._get_renderer()
        renderer.update_scene(self.data, camera=camera, scene_option=self._visual_options)
        return np.asarray(renderer.render()).copy()

    def observe(self) -> dict[str, np.ndarray]:
        observation = {"observation.state": self.joint_positions.astype(np.float32)}
        if self.render_images:
            observation["observation.images.front"] = self.render_camera("front")
            observation["observation.images.wrist"] = self.render_camera("wrist")
        return observation

    def is_success(self) -> bool:
        """True when every stick corner is inside the box and nearly settled."""
        center = self.data.xpos[self._stick_body_id]
        rotation = self.data.xmat[self._stick_body_id].reshape(3, 3)
        signs = np.array(
            [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
        )
        corners = center + (signs * STICK_HALF_SIZE) @ rotation.T
        # Resting contacts have sub-millimetre solver penetration. Keep XY
        # strict, but do not reject a correctly settled stick because its
        # lowest corner is tens of microns below the nominal box-floor top.
        lower = BOX_INNER_MIN.copy()
        lower[2] -= 0.001
        contained = bool(np.all(corners >= lower) and np.all(corners <= BOX_INNER_MAX))
        speed = float(np.linalg.norm(self.data.cvel[self._stick_body_id]))
        return contained and speed < 0.10

    def info(self) -> dict[str, Any]:
        return {
            "success": self.is_success(),
            "control_step": self.control_step,
            "sim_time": float(self.data.time),
            "joint_names": JOINT_NAMES,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
