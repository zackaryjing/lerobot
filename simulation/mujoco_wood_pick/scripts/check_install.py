#!/usr/bin/env python3
"""Compile the workcell and smoke-test both camera render paths."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", choices=("egl", "glfw", "osmesa"), default="egl")
    parser.add_argument("--adapter", default="NVIDIA", help="WSL D3D12 adapter name substring.")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", args.gl)
    if args.adapter:
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", args.adapter)

    import mujoco

    from mujoco_wood_pick import MODEL_PATH, WoodPickEnv

    env = WoodPickEnv(render_images=False)
    try:
        observation, _ = env.reset()
        print(f"MuJoCo: {mujoco.__version__}")
        print(f"MUJOCO_GL: {os.environ['MUJOCO_GL']}")
        print(f"Model: {MODEL_PATH}")
        print(
            f"Model sizes: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}, "
            f"nbody={env.model.nbody}, ngeom={env.model.ngeom}"
        )
        print(f"State: {observation['observation.state'].shape}")
        if not args.no_render:
            for camera in ("front", "wrist"):
                image = env.render_camera(camera)
                print(
                    f"Camera {camera}: shape={image.shape}, dtype={image.dtype}, "
                    f"range=[{image.min()}, {image.max()}]"
                )
            try:
                from OpenGL import GL

                renderer = GL.glGetString(GL.GL_RENDERER)
                vendor = GL.glGetString(GL.GL_VENDOR)
                if renderer is not None and vendor is not None:
                    print(f"OpenGL vendor: {vendor.decode(errors='replace')}")
                    print(f"OpenGL renderer: {renderer.decode(errors='replace')}")
            except Exception as error:  # Rendering success is authoritative.
                print(f"OpenGL renderer query unavailable: {error}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
