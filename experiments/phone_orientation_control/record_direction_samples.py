#!/usr/bin/env python
"""Teleoperate SO101 with its leader and capture useful direction configurations.

ENTER records the follower's current safe configuration, ``u`` removes the last
sample, and ``q`` exits. The output is appended and atomically saved after every
change, so it remains useful if the process or robot power is interrupted.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from controller import ALL_JOINTS, ARM_JOINTS, ControlConfig, load_joint_limits
from direction_atlas import DEFAULT_COLLISIONS, DEFAULT_URDF, SO101StateValidator


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
)
DEFAULT_OUTPUT = THIS_DIR / "manual_direction_samples.json"


class ManualSampleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("version") != 1 or not isinstance(payload.get("samples"), list):
                raise ValueError(f"unsupported manual sample file: {path}")
            self.payload = payload
        else:
            self.payload = {
                "version": 1,
                "source": "so101_leader_follower_teleoperation",
                "joint_names": ARM_JOINTS,
                "samples": [],
            }

    @property
    def count(self) -> int:
        return len(self.payload["samples"])

    def add(self, candidate: object, gripper_deg: float) -> None:
        item = asdict(candidate)  # AtlasCandidate
        item["gripper_deg"] = float(gripper_deg)
        item["captured_at_unix_s"] = time.time()
        self.payload["samples"].append(item)
        self.save()

    def undo(self) -> bool:
        if not self.payload["samples"]:
            return False
        self.payload["samples"].pop()
        self.save()
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload, indent=2) + "\n")
        temporary.replace(self.path)


def read_commands(commands: queue.SimpleQueue[str]) -> None:
    for line in sys.stdin:
        commands.put(line.strip().lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--follower-id", default="myfollower01")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-id", default="myleader01")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--collisions", type=Path, default=DEFAULT_COLLISIONS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

    limits = load_joint_limits(args.urdf, args.calibration)
    validator = SO101StateValidator(args.urdf, args.collisions, limits)
    store = ManualSampleStore(args.output)
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True)
    )
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()

    print("Connecting leader and follower; no sample is recorded automatically.")
    leader.connect(calibrate=False)
    try:
        follower.connect(calibrate=False)
        if not leader.is_calibrated or not follower.is_calibrated:
            raise RuntimeError("leader/follower calibration mismatch; refusing to teleoperate")
        threading.Thread(target=read_commands, args=(commands,), daemon=True).start()
        print(
            f"Ready. Existing samples: {store.count}. ENTER=capture, u+ENTER=undo, q+ENTER=quit."
        )
        period = 1.0 / args.fps
        running = True
        while running:
            started = time.perf_counter()
            follower.send_action(leader.get_action())
            observation = follower.get_observation()
            while not commands.empty():
                command = commands.get()
                if command == "q":
                    running = False
                elif command == "u":
                    print("Removed last sample." if store.undo() else "No sample to remove.")
                elif command == "":
                    joints = np.array([float(observation[f"{name}.pos"]) for name in ARM_JOINTS])
                    candidate, reason = validator.evaluate(joints)
                    if candidate is None:
                        print(f"Rejected capture: {reason}; joints={np.round(joints, 2).tolist()}")
                    else:
                        store.add(candidate, float(observation[f"{ALL_JOINTS[-1]}.pos"]))
                        print(
                            f"Captured #{store.count}: direction={np.round(candidate.direction, 3).tolist()}, "
                            f"tip_z={candidate.tip_position_m[2]:.3f}m"
                        )
            time.sleep(max(0.0, period - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()
    print(f"Saved {store.count} samples to {args.output}")


if __name__ == "__main__":
    main()
