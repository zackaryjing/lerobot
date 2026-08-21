"""Launch the scene and save one frame from each simulated camera."""

import argparse
import os
from pathlib import Path

# LeIsaac otherwise resolves assets relative to whichever Git repository is the
# current working directory. Pin it before importing Isaac/LeIsaac modules.
PROJECT_DIR = Path(__file__).resolve().parents[1]
LEISAAC_ASSETS_DIR = PROJECT_DIR / "assets" / "leisaac"
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(LEISAAC_ASSETS_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=Path, default=Path("outputs/scene_preview"))
parser.add_argument(
    "--camera",
    choices=("both", "front", "wrist"),
    default="front",
    help="Enable both cameras, or only one camera to reduce rendering memory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from isaaclab_tasks.utils import parse_env_cfg
from PIL import Image

import leisaac_wood_pick  # noqa: F401
from leisaac_wood_pick.tasks.wood_pick import TASK_ID
from leisaac_wood_pick.tasks.wood_pick.scene_parameters import WOODEN_TARGET_USD_PATH


def _save_rgb(image, path: Path) -> None:
    array = image[0].detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    robot_asset = LEISAAC_ASSETS_DIR / "robots" / "so101_follower.usd"
    if not robot_asset.is_file():
        raise FileNotFoundError(
            f"Missing official SO-101 asset: {robot_asset}\n"
            "Run `python scripts/download_assets.py` first."
        )
    if not WOODEN_TARGET_USD_PATH.is_file():
        raise FileNotFoundError(
            f"Missing converted wooden-target asset: {WOODEN_TARGET_USD_PATH}\n"
            "Run `python scripts/prepare_assets.py --headless` first."
        )

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.camera != "both":
        disabled_camera = "wrist" if args_cli.camera == "front" else "front"
        setattr(env_cfg.scene, disabled_camera, None)
        setattr(env_cfg.observations.policy, disabled_camera, None)
    # Joint-position actions let the preview hold the measured rest pose.
    env_cfg.use_teleop_device("so101leader")
    env = gym.make(TASK_ID, cfg=env_cfg)

    try:
        observation, _ = env.reset()
        stick = env.unwrapped.scene["stick"]
        print(f"Stick initial position (m): {stick.data.root_pos_w[0].tolist()}", flush=True)
        hold_action = env.unwrapped.scene["robot"].data.default_joint_pos.clone()
        for _ in range(args_cli.steps):
            observation, _, _, _, _ = env.step(hold_action)
        print(f"Stick final position (m): {stick.data.root_pos_w[0].tolist()}", flush=True)

        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        policy_observation = observation["policy"]
        if args_cli.camera in ("both", "front"):
            _save_rgb(policy_observation["front"], args_cli.output_dir / "front.png")
        if args_cli.camera in ("both", "wrist"):
            _save_rgb(policy_observation["wrist"], args_cli.output_dir / "wrist.png")
        print(f"Saved camera previews to {args_cli.output_dir.resolve()}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
