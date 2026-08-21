#!/usr/bin/env python3
"""Measure local physics and two-camera offscreen throughput."""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", choices=("egl", "glfw", "osmesa"), default="egl")
    parser.add_argument("--adapter", default="NVIDIA")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", args.gl)
    if args.adapter:
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", args.adapter)

    from OpenGL import GL

    from mujoco_wood_pick import WoodPickEnv
    from mujoco_wood_pick.env import HOME_QPOS

    env = WoodPickEnv(render_images=False)
    try:
        env.reset()
        env.render_camera("front")  # Context and shader warm-up.
        renderer = GL.glGetString(GL.GL_RENDERER)

        started = time.perf_counter()
        for _ in range(args.frames):
            env.advance_physics(HOME_QPOS)
        physics_seconds = time.perf_counter() - started

        started = time.perf_counter()
        for _ in range(args.frames):
            env.advance_physics(HOME_QPOS)
            env.render_camera("front")
            env.render_camera("wrist")
        camera_seconds = time.perf_counter() - started

        print(f"Renderer: {renderer.decode(errors='replace') if renderer else 'unknown'}")
        print(f"Physics control steps/s: {args.frames / physics_seconds:.1f}")
        print(f"Dual-camera control steps/s: {args.frames / camera_seconds:.1f}")
        print(f"Individual 640x480 images/s: {2 * args.frames / camera_seconds:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
