# SO-101 wood-pick digital twin

This external LeIsaac/Isaac Lab project models the measured real workcell used
by the two-camera ACT dataset. It also includes a first scripted grasp and
successful-only data generator; see [`SCRIPTED_DATA.md`](SCRIPTED_DATA.md) for
its geometry, IK strategy, action-label semantics, and cuRobo upgrade path.

## Implemented scene

- official LeIsaac SO-101 follower asset, with white printed parts and black
  motors/gripper applied non-destructively at startup;
- measured overhead RGB camera at 640x480 and 30 FPS, mounted 90 degrees
  clockwise; the DFOV=100-degree source view is centrally cropped to 60% width
  and height to match the recorded training frames;
- official LeIsaac wrist camera as a temporary mount calibration;
- video-matched tabletop and white placement platform;
- open cardboard destination box with a video-matched yellow bottom;
- measured `target/wooden_target.stl`, converted to a dynamic USD rigid body;
- configurable dome/key/fill lighting;
- LeIsaac observations compatible with later LeRobot Dataset v3 recording.

## Coordinate frame

All configuration values use SI units. The supplied measurements are converted
from centimetres in `scene_parameters.py`.

- origin: midpoint of the rear edge of the follower-arm base;
- `+X`: right;
- `+Y`: forward;
- `+Z`: upward;
- `Z=0`: tabletop surface.

Measured geometry:

| Item | Lower-left / position (m) | Size (m) |
|---|---|---|
| destination box | `(-0.107, 0.380, 0)` | `0.150 x 0.100 x 0.085` |
| placement platform | `(0.100, 0.312, 0)` | `0.270 x 0.100 x 0.053` |
| overhead camera | `(0.060, 0.230, 0.595)` | optical axis along `-Z` |
| wooden target STL | initially centred on platform | `0.0115 x 0.0115 x 0.080` |

The box length and platform length are currently interpreted along `+X`; their
widths are along `+Y`. The wooden target is initially aligned with `+X`.

## Parameters still awaiting calibration

- exact overhead-camera principal point and distortion coefficients (the
  reported DFOV, EFL and 4:3 crop now determine the pinhole projection);
- wrist-camera physical transform and intrinsics;
- exact transform between the LeIsaac SO-101 USD root and the measured rear
  base-edge origin;
- real light positions, exposure, and white balance;
- real wooden-target mass (currently 30 g) and cardboard wall thickness
  (currently 3 mm).

All of these are centralized in
`source/leisaac_wood_pick/leisaac_wood_pick/tasks/wood_pick/scene_parameters.py`.

## Runtime recommendation

Use a separate `leisaac` conda environment on a Linux RTX machine. An RTX 4090
is not required for this single-environment preview: 8 GB VRAM is the practical
minimum for the measured 5.6 GiB peak, while 16 GB or more leaves useful room
for parallel environments and later data generation. The local RTX 3050 Ti
4 GB is used only for editing.

Isaac Sim 5.1 has no software fallback for these RTX camera observations.
Passing `--device cpu` only moves environment tensors to the CPU; Vulkan/RTX
camera startup and the current PhysX stack still require a CUDA-capable GPU.

The tested AutoDL installation occupies about 24 GB for the conda environment;
LeIsaac's source checkout itself is only about 10 MB. With one simulated camera
enabled at a time, the cached scene starts and writes a frame in about 27
seconds and peaks at 5.3--5.4 GiB of VRAM. Both cameras together peak at about
5.6 GiB. A pressure test with only 4.3 GiB free failed in both PhysX and Vulkan,
so a 4 GB GPU is not sufficient even for sequential offline rendering with the
current quality settings.

Do not install Isaac Sim into the existing training `lerobot` environment.
Generate the dataset in `leisaac`, then train it in `lerobot`.

## Install on the RTX machine

LeIsaac 0.4.0 currently pins Isaac Sim 5.1 / Isaac Lab 2.3:

```bash
conda create -n leisaac python=3.11
conda activate leisaac
conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install \
  'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' \
  --extra-index-url https://pypi.nvidia.com
pip install -e simulation/leisaac_wood_pick/source/leisaac_wood_pick
```

Download LeIsaac's official SO-101 asset:

```bash
python simulation/leisaac_wood_pick/scripts/download_assets.py
```

The preview script pins `LEISAAC_ASSETS_ROOT` to the downloaded asset directory,
avoiding LeIsaac's current-working-directory-dependent default resolution.

## Convert the measured STL

Run this once after installation:

```bash
python simulation/leisaac_wood_pick/scripts/prepare_assets.py --headless
```

The converter recentres the binary STL, converts millimetres to metres, adds a
convex-hull collider and a 30 g rigid body, and writes:

```text
simulation/leisaac_wood_pick/source/leisaac_wood_pick/
  leisaac_wood_pick/assets/wooden_target.usd
```

## Render the first validation images

Render the two views sequentially so both cameras are never active at once:

```bash
python simulation/leisaac_wood_pick/scripts/preview_scene.py \
  --headless --enable_cameras --steps 60 --camera front \
  --output_dir outputs/scene_preview_front
python simulation/leisaac_wood_pick/scripts/preview_scene.py \
  --headless --enable_cameras --steps 60 --camera wrist \
  --output_dir outputs/scene_preview_wrist
```

`front` is the default camera. The `both` mode remains available for profiling,
but is not used by the normal offline-preview workflow.

The first validation criterion is geometry and framing; material and lighting
values should only be tuned after the camera axes and intrinsics are confirmed.

## Evaluate the real-robot ACT checkpoint

The policy runner uses receding-horizon execution by default: it predicts 50
actions, replans when 25 remain, and cosine-cross-fades the shared horizon. This
prevents the large shoulder/elbow discontinuity produced by hard chunk swaps.

```bash
python simulation/leisaac_wood_pick/scripts/run_trained_policy.py \
  --headless --enable_cameras --device cuda:0 \
  --num_episodes 1 --episode_seconds 40 \
  --actions_per_chunk 50 --replan_overlap 25 --blend_mode cosine \
  --video_dir outputs/trained_policy_overlap25_cosine_40s \
  --metrics_file outputs/trained_policy_overlap25_cosine_40s/metrics.json
```

Use `--blend_mode fixed --blend_new_weight 0.7` to reproduce the real async
client's `0.3 * old + 0.7 * new` aggregation. Use `--replan_overlap 0` only for
hard-switch diagnostics. The runner records raw and blended targets, actual
joint angles, end-effector pose/velocity, applied torque and per-link contact
forces. Generate plots and a full per-frame CSV with:

```bash
python simulation/leisaac_wood_pick/scripts/analyze_policy_episode.py \
  outputs/trained_policy_overlap25_cosine_40s/episode_001_diagnostics.npz \
  --output_dir outputs/trained_policy_overlap25_cosine_40s/analysis
```
