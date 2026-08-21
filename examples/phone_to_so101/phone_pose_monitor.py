#!/usr/bin/env python
"""
Phone pose monitor — visualizes raw ARCore/WebXR pose WITHOUT connecting to robot.
Use this to see how jittery the phone tracking really is.

Run: python examples/phone_to_so101/phone_pose_monitor.py
"""

import time
import numpy as np

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone

FPS = 30
PHONE_OS = PhoneOS.ANDROID  # change to PhoneOS.IOS for iPhone


def main():
    teleop_config = PhoneConfig(phone_os=PHONE_OS)
    phone = Phone(teleop_config)
    phone.connect()

    if not phone.is_connected:
        raise RuntimeError("Phone not connected!")

    print("Phone connected. Move your phone...\n")
    print(f"{'time':>8s}  {'enabled':>7s}  {'pos_x':>8s}  {'pos_y':>8s}  {'pos_z':>8s}  "
          f"{'|pos|':>8s}  {'jump_warn':>9s}")
    print("-" * 80)

    last_pos = None
    last_rot = None

    while True:
        t0 = time.perf_counter()

        action = phone.get_action()
        if not action:
            time.sleep(0.01)
            continue

        pos = action["phone.pos"]
        rot = action["phone.rot"]
        enabled = action["phone.enabled"]

        # Compute delta since last frame
        delta_pos = np.linalg.norm(pos - last_pos) if last_pos is not None else 0.0
        jump_warn = "⚠ JUMP!" if delta_pos > 0.05 else ""

        print(f"{time.perf_counter():8.3f}  {str(enabled):>7s}  "
              f"{pos[0]:8.4f}  {pos[1]:8.4f}  {pos[2]:8.4f}  "
              f"{np.linalg.norm(pos):8.4f}  {jump_warn:>9s}")

        last_pos = pos.copy()
        last_rot = rot

        dt = time.perf_counter() - t0
        sleep_t = max(1.0 / FPS - dt, 0.001)
        time.sleep(sleep_t)


if __name__ == "__main__":
    main()
