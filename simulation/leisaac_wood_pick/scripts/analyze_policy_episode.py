"""Diagnose joint/chunk/contact failures from a trained-policy episode NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# LeIsaac's linear mapping between normalized real-motor coordinates and the
# physical joint angles declared by the stock SO-101 USD.
USD_LIMIT_DEG = np.asarray(
    ((-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100)),
    dtype=float,
)
MOTOR_LIMIT = np.asarray(
    ((-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (0, 100)),
    dtype=float,
)


def _motor_to_degree(value: np.ndarray) -> np.ndarray:
    motor_fraction = (value - MOTOR_LIMIT[:, 0]) / (MOTOR_LIMIT[:, 1] - MOTOR_LIMIT[:, 0])
    return USD_LIMIT_DEG[:, 0] + motor_fraction * (USD_LIMIT_DEG[:, 1] - USD_LIMIT_DEG[:, 0])


def _top_rows(values: np.ndarray, count: int = 12) -> list[int]:
    return np.argsort(values)[-count:][::-1].astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--actions_per_chunk", type=int, default=50)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    source = args.diagnostics.resolve()
    output_dir = (args.output_dir or source.with_suffix("")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(source)

    actual = data["actual_motor"].astype(float)
    commanded = data["commanded_motor"].astype(float)
    predicted = data["predicted_motor"].astype(float)
    effective = data["effective_motor_target"].astype(float) if "effective_motor_target" in data else predicted
    actual_deg = _motor_to_degree(actual)
    commanded_deg = _motor_to_degree(commanded)
    predicted_deg = _motor_to_degree(predicted)
    effective_deg = _motor_to_degree(effective)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    names = data["joint_names"].astype(str).tolist() if "joint_names" in data else list(DEFAULT_NAMES)
    frames = len(actual)
    time_s = np.arange(frames) / fps

    # Frame i stores state after action i.  Differences therefore identify
    # action-target discontinuities and the corresponding physical response.
    predicted_jump = np.vstack((np.zeros((1, 6)), np.diff(predicted, axis=0)))
    effective_jump = np.vstack((np.zeros((1, 6)), np.diff(effective, axis=0)))
    command_jump = np.vstack((np.zeros((1, 6)), np.diff(commanded, axis=0)))
    actual_jump = np.vstack((np.zeros((1, 6)), np.diff(actual, axis=0)))
    tracking_error = commanded - actual
    predicted_jump_norm = np.linalg.norm(predicted_jump, axis=1)
    effective_jump_norm = np.linalg.norm(effective_jump, axis=1)
    command_jump_norm = np.linalg.norm(command_jump, axis=1)
    actual_jump_norm = np.linalg.norm(actual_jump, axis=1)
    error_norm = np.linalg.norm(tracking_error, axis=1)

    if "chunk_id" in data:
        chunk_id = data["chunk_id"].astype(int)
        chunk_boundaries = np.flatnonzero(np.r_[False, np.diff(chunk_id) != 0])
    else:
        chunk_id = np.arange(frames) // args.actions_per_chunk
        chunk_boundaries = np.arange(args.actions_per_chunk, frames, args.actions_per_chunk)

    boundary_mask = np.zeros(frames, dtype=bool)
    boundary_mask[chunk_boundaries] = True
    non_boundary = ~boundary_mask
    non_boundary[0] = False

    contact_norm = None
    contact_peak_by_body = None
    contact_names: list[str] = []
    if "robot_contact_force_world" in data:
        contact = data["robot_contact_force_world"].astype(float)
        contact_norm_by_body = np.linalg.norm(contact, axis=-1)
        contact_norm = contact_norm_by_body.max(axis=1)
        contact_peak_by_body = contact_norm_by_body.max(axis=0)
        contact_names = data["contact_body_names"].astype(str).tolist()

    ee_step_mm = None
    ee_speed = None
    if "ee_pose_world" in data:
        ee_pos = data["ee_pose_world"][:, :3].astype(float)
        ee_step_mm = np.r_[0.0, np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) * 1000.0]
        if "ee_velocity_world" in data:
            ee_speed = np.linalg.norm(data["ee_velocity_world"][:, :3], axis=1)

    events = []
    interesting = sorted(
        set(_top_rows(predicted_jump_norm) + _top_rows(error_norm) + chunk_boundaries.tolist())
    )
    for frame in interesting:
        pred_joint = int(np.argmax(np.abs(predicted_jump[frame])))
        error_joint = int(np.argmax(np.abs(tracking_error[frame])))
        events.append(
            {
                "frame": frame,
                "time_s": frame / fps,
                "chunk": int(chunk_id[frame]),
                "is_chunk_boundary": bool(boundary_mask[frame]),
                "largest_policy_jump_joint": names[pred_joint],
                "largest_policy_jump": float(predicted_jump[frame, pred_joint]),
                "policy_jump_norm": float(predicted_jump_norm[frame]),
                "effective_target_jump_norm": float(effective_jump_norm[frame]),
                "command_jump_norm": float(command_jump_norm[frame]),
                "actual_jump_norm": float(actual_jump_norm[frame]),
                "largest_tracking_error_joint": names[error_joint],
                "largest_tracking_error": float(tracking_error[frame, error_joint]),
                "tracking_error_norm": float(error_norm[frame]),
                "ee_step_mm": None if ee_step_mm is None else float(ee_step_mm[frame]),
                "contact_force_n": None if contact_norm is None else float(contact_norm[frame]),
            }
        )

    boundary_policy = predicted_jump_norm[boundary_mask]
    regular_policy = predicted_jump_norm[non_boundary]
    report = {
        "source": str(source),
        "frames": frames,
        "fps": fps,
        "duration_s": frames / fps,
        "joint_names": names,
        "chunk_boundaries": chunk_boundaries.tolist(),
        "policy_jump_motor_units": {
            "boundary_median": float(np.median(boundary_policy)) if len(boundary_policy) else None,
            "boundary_max": float(np.max(boundary_policy)) if len(boundary_policy) else None,
            "non_boundary_median": float(np.median(regular_policy)) if len(regular_policy) else None,
            "non_boundary_p95": float(np.percentile(regular_policy, 95)) if len(regular_policy) else None,
            "max_per_joint": dict(zip(names, np.max(np.abs(predicted_jump), axis=0).tolist(), strict=True)),
        },
        "effective_target_jump_motor_units": {
            "boundary_median": float(np.median(effective_jump_norm[boundary_mask])) if np.any(boundary_mask) else None,
            "boundary_max": float(np.max(effective_jump_norm[boundary_mask])) if np.any(boundary_mask) else None,
            "non_boundary_p95": float(np.percentile(effective_jump_norm[non_boundary], 95)) if np.any(non_boundary) else None,
            "max_per_joint": dict(zip(names, np.max(np.abs(effective_jump), axis=0).tolist(), strict=True)),
        },
        "tracking_error_motor_units": {
            "mean_abs_per_joint": dict(zip(names, np.mean(np.abs(tracking_error), axis=0).tolist(), strict=True)),
            "max_abs_per_joint": dict(zip(names, np.max(np.abs(tracking_error), axis=0).tolist(), strict=True)),
        },
        "end_effector": None if ee_step_mm is None else {
            "max_step_mm": float(np.max(ee_step_mm)),
            "max_speed_m_s": None if ee_speed is None else float(np.max(ee_speed)),
        },
        "contacts": None if contact_norm is None else {
            "max_force_n": float(np.max(contact_norm)),
            "peak_force_per_body_n": dict(zip(contact_names, contact_peak_by_body.tolist(), strict=True)),
        },
        "top_events": sorted(events, key=lambda item: item["policy_jump_norm"], reverse=True)[:20],
    }

    with (output_dir / "analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)

    # Full, spreadsheet-friendly trajectory.  Degrees are physical USD joint
    # angles; the source NPZ retains the original normalized real-motor values.
    full_rows = []
    for frame in range(frames):
        row: dict[str, float | int] = {
            "frame": frame,
            "time_s": frame / fps,
            "chunk_id": int(chunk_id[frame]),
            "chunk_action_index": int(data["chunk_action_index"][frame])
            if "chunk_action_index" in data
            else frame % args.actions_per_chunk,
        }
        for joint, name in enumerate(names):
            row[f"{name}.policy_deg"] = float(predicted_deg[frame, joint])
            row[f"{name}.effective_target_deg"] = float(effective_deg[frame, joint])
            row[f"{name}.command_deg"] = float(commanded_deg[frame, joint])
            row[f"{name}.actual_deg"] = float(actual_deg[frame, joint])
            row[f"{name}.tracking_error_deg"] = float(commanded_deg[frame, joint] - actual_deg[frame, joint])
            if "applied_joint_torque" in data:
                row[f"{name}.applied_torque_nm"] = float(data["applied_joint_torque"][frame, joint])
        if "ee_pose_world" in data:
            for index, label in enumerate(("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")):
                row[f"ee.{label}"] = float(data["ee_pose_world"][frame, index])
        if "ee_velocity_world" in data:
            for index, label in enumerate(("vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s")):
                row[f"ee.{label}"] = float(data["ee_velocity_world"][frame, index])
        if contact_norm is not None:
            row["robot.max_contact_force_n"] = float(contact_norm[frame])
        for index, label in enumerate(("x_m", "y_m", "z_m")):
            row[f"stick.{label}"] = float(data["stick_position_world"][frame, index])
        full_rows.append(row)
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=full_rows[0].keys())
        writer.writeheader()
        writer.writerows(full_rows)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    for joint, name in enumerate(names):
        ax = axes.flat[joint]
        ax.plot(time_s, predicted_deg[:, joint], lw=0.55, alpha=0.45, label="raw policy")
        if "effective_motor_target" in data:
            ax.plot(time_s, effective_deg[:, joint], lw=0.75, alpha=0.85, label="blended target")
        ax.plot(time_s, commanded_deg[:, joint], lw=0.8, label="command")
        ax.plot(time_s, actual_deg[:, joint], lw=0.8, label="actual")
        ax.set_title(name)
        ax.set_ylabel("joint angle (deg)")
        ax.grid(alpha=0.2)
        for frame in chunk_boundaries:
            ax.axvline(frame / fps, color="k", lw=0.25, alpha=0.15)
    axes.flat[0].legend(ncol=3, fontsize=8)

    ax = axes.flat[6]
    ax.plot(time_s, predicted_jump_norm, label="policy target jump")
    if "effective_motor_target" in data:
        ax.plot(time_s, effective_jump_norm, label="blended target jump")
    ax.plot(time_s, actual_jump_norm, label="actual joint step", alpha=0.8)
    ax.plot(time_s, error_norm, label="tracking error", alpha=0.8)
    ax.set_title("Joint-space discontinuity")
    ax.set_ylabel("motor units / frame")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes.flat[7]
    if ee_step_mm is not None:
        ax.plot(time_s, ee_step_mm, label="EE displacement (mm/frame)")
    if contact_norm is not None:
        ax.plot(time_s, contact_norm, label="max robot contact force (N)")
    if ee_step_mm is None and contact_norm is None:
        ax.text(0.5, 0.5, "Re-run with enriched diagnostics", ha="center", va="center")
    ax.set_title("Cartesian motion and collision")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_diagnostics.png", dpi=160)
    plt.close(fig)

    if "ee_pose_world" in data:
        ee_pos = data["ee_pose_world"][:, :3].astype(float)
        fig = plt.figure(figsize=(15, 11))
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], lw=0.8)
        ax3d.scatter(
            ee_pos[chunk_boundaries, 0],
            ee_pos[chunk_boundaries, 1],
            ee_pos[chunk_boundaries, 2],
            s=12,
            c="red",
            label="replan boundary",
        )
        ax3d.set(xlabel="world X (m)", ylabel="world Y (m)", zlabel="world Z (m)", title="End-effector path")
        ax3d.legend(fontsize=8)

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], lw=0.8)
        ax.scatter(ee_pos[chunk_boundaries, 0], ee_pos[chunk_boundaries, 1], s=12, c="red")
        ax.set(xlabel="world X (m)", ylabel="world Y (m)", title="Top view", aspect="equal")
        ax.grid(alpha=0.2)

        ax = fig.add_subplot(2, 2, 3)
        for index, label in enumerate(("X", "Y", "Z")):
            ax.plot(time_s, ee_pos[:, index], label=label)
        ax.set(xlabel="time (s)", ylabel="position (m)", title="Cartesian coordinates")
        ax.legend()
        ax.grid(alpha=0.2)

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(time_s, ee_step_mm, label="displacement (mm/frame)")
        if ee_speed is not None:
            ax.plot(time_s, ee_speed * 1000.0 / fps, label="velocity equivalent (mm/frame)", alpha=0.7)
        if contact_norm is not None:
            ax.plot(time_s, contact_norm, label="contact force (N)")
        for frame in chunk_boundaries:
            ax.axvline(frame / fps, color="red", lw=0.35, alpha=0.2)
        ax.set(xlabel="time (s)", title="Motion spikes versus replanning")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "end_effector_trajectory.png", dpi=160)
        plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
