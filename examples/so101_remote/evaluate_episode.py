#!/usr/bin/env python

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ACT trajectory evaluation on a recorded episode.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metric_block(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = prediction - target
    absolute_error = error.abs()
    return {
        "mae": absolute_error.mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "p95_absolute_error": torch.quantile(absolute_error.flatten(), 0.95).item(),
        "per_joint_mae": dict(zip(JOINT_NAMES, absolute_error.mean(dim=0).tolist(), strict=True)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.output_dir / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    dataset = LeRobotDataset(
        "local/so101_offical_04",
        root=args.dataset_root,
        episodes=[args.episode],
        video_backend="torchcodec",
    )
    actual = torch.stack(list(dataset.hf_dataset["action"])).float()
    episode_length, action_dim = actual.shape
    if action_dim != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} action dimensions, got {action_dim}")

    policy = get_policy_class("act").from_pretrained(args.model_path)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    anchors = list(range(0, episode_length, args.stride))
    stitched = torch.full_like(actual, torch.nan)
    chunk_predictions = []
    chunk_targets = []
    hold_baselines = []
    inference_times = []

    for anchor in anchors:
        sample = dataset[anchor]
        observation = {key: sample[key] for key in policy.config.input_features}

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            processed_observation = preprocessor(observation)
            normalized_chunk = policy.predict_action_chunk(processed_observation)
            predicted_steps = [
                postprocessor(normalized_chunk[:, step, :]).squeeze(0)
                for step in range(min(args.horizon, normalized_chunk.shape[1]))
            ]
            predicted_chunk = torch.stack(predicted_steps)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - start)

        valid_length = min(len(predicted_chunk), episode_length - anchor)
        predicted_chunk = predicted_chunk[:valid_length]
        target_chunk = actual[anchor : anchor + valid_length]
        hold_chunk = sample["observation.state"].float().repeat(valid_length, 1)

        chunk_predictions.append(predicted_chunk)
        chunk_targets.append(target_chunk)
        hold_baselines.append(hold_chunk)

        for offset in range(valid_length):
            index = anchor + offset
            if torch.isnan(stitched[index]).all():
                stitched[index] = predicted_chunk[offset]
            else:
                stitched[index] = 0.3 * stitched[index] + 0.7 * predicted_chunk[offset]

    all_predictions = torch.cat(chunk_predictions)
    all_targets = torch.cat(chunk_targets)
    all_hold = torch.cat(hold_baselines)

    valid_stitched = ~torch.isnan(stitched).any(dim=1)
    stitched_metrics = metric_block(stitched[valid_stitched], actual[valid_stitched])
    chunk_metrics = metric_block(all_predictions, all_targets)
    hold_metrics = metric_block(all_hold, all_targets)

    max_horizon = min(args.horizon, max(len(chunk) for chunk in chunk_predictions))
    horizon_mae = []
    hold_horizon_mae = []
    for offset in range(max_horizon):
        pred_at_offset = []
        target_at_offset = []
        hold_at_offset = []
        for predicted, target, hold in zip(
            chunk_predictions, chunk_targets, hold_baselines, strict=True
        ):
            if offset < len(predicted):
                pred_at_offset.append(predicted[offset])
                target_at_offset.append(target[offset])
                hold_at_offset.append(hold[offset])
        horizon_mae.append((torch.stack(pred_at_offset) - torch.stack(target_at_offset)).abs().mean().item())
        hold_horizon_mae.append(
            (torch.stack(hold_at_offset) - torch.stack(target_at_offset)).abs().mean().item()
        )

    action_range = actual.max(dim=0).values - actual.min(dim=0).values
    stitched_joint_mae = torch.tensor(list(stitched_metrics["per_joint_mae"].values()))
    normalized_joint_mae = 100 * stitched_joint_mae / action_range.clamp_min(1e-8)

    metrics = {
        "episode": args.episode,
        "frames": episode_length,
        "duration_s": episode_length / dataset.fps,
        "fps": dataset.fps,
        "anchors": anchors,
        "stride": args.stride,
        "horizon": args.horizon,
        "inference_time_s": {
            "mean": float(np.mean(inference_times)),
            "p95": float(np.quantile(inference_times, 0.95)),
            "min": float(np.min(inference_times)),
            "max": float(np.max(inference_times)),
        },
        "teacher_forced_receding_horizon": stitched_metrics,
        "all_action_chunks": chunk_metrics,
        "hold_current_pose_baseline": hold_metrics,
        "model_mae_vs_hold_ratio": chunk_metrics["mae"] / hold_metrics["mae"],
        "per_joint_mae_as_percent_of_episode_range": dict(
            zip(JOINT_NAMES, normalized_joint_mae.tolist(), strict=True)
        ),
        "horizon_mae": horizon_mae,
        "hold_horizon_mae": hold_horizon_mae,
    }

    with (args.output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        actual=actual.numpy(),
        stitched=stitched.numpy(),
        anchors=np.asarray(anchors),
    )

    time_axis = np.arange(episode_length) / dataset.fps
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(time_axis, actual[:, joint_index], label="recorded", linewidth=1.5)
        axis.plot(time_axis, stitched[:, joint_index], label="model", linewidth=1.0, alpha=0.85)
        axis.set_title(JOINT_NAMES[joint_index])
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.supxlabel("time (s)")
    figure.supylabel("normalized joint position")
    figure.tight_layout()
    figure.savefig(args.output_dir / "trajectory_overlay.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    horizon_axis = np.arange(max_horizon) / dataset.fps
    axis.plot(horizon_axis, horizon_mae, label="ACT")
    axis.plot(horizon_axis, hold_horizon_mae, label="hold current pose")
    axis.set_xlabel("prediction horizon (s)")
    axis.set_ylabel("mean absolute error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "horizon_error.png", dpi=160)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
