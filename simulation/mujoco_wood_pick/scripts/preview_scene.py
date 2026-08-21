#!/usr/bin/env python3
"""Save camera previews or open MuJoCo's interactive passive viewer."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", choices=("egl", "glfw", "osmesa"), default="egl")
    parser.add_argument("--adapter", default="NVIDIA", help="WSL D3D12 adapter name substring.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mujoco_scene_preview"))
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--viewer", action="store_true", help="Open an interactive WSLg window instead of saving PNGs.")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer else args.gl)
    if args.adapter:
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", args.adapter)

    import mujoco
    import numpy as np
    from PIL import Image

    from mujoco_wood_pick import CONTROL_FPS, WoodPickEnv
    from mujoco_wood_pick.env import HOME_QPOS

    env = WoodPickEnv(render_images=False)
    try:
        env.reset()
        env.advance_physics(HOME_QPOS, max(0, round(args.settle_seconds * CONTROL_FPS)))

        if args.viewer:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    started = time.perf_counter()
                    env.advance_physics(env.joint_positions)
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / CONTROL_FPS - (time.perf_counter() - started)))
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)
        frames = {name: env.render_camera(name) for name in ("front", "wrist", "overview")}
        for name, frame in frames.items():
            path = args.output_dir / f"{name}.png"
            Image.fromarray(frame).save(path)
            print(path.resolve())

        combined = np.concatenate((frames["front"], frames["wrist"]), axis=1)
        combined_path = args.output_dir / "front_wrist.png"
        Image.fromarray(combined).save(combined_path)
        print(combined_path.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()
