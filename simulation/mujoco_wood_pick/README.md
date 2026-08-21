# 本地 MuJoCo SO-101 木条抓取环境

这个目录是 Isaac/LeIsaac 场景的轻量本地替代版本。第一阶段包含：

- Google DeepMind MuJoCo Menagerie 的 SO-101 MJCF、网格和简化碰撞体；
- 测量得到的桌面、白色平台、开放纸盒和木条位置/尺寸；
- 与真机画面匹配的颜色、顶部相机旋转和中央 60% 裁剪后 FOV；
- 顶部 `front`、腕部 `wrist` 两路 640x480 RGB；
- 120 Hz 物理、30 Hz 控制和 Gymnasium 风格接口；
- 木条完整落入盒内且基本静止的成功判定。

SO-101 上游资产及许可证位于 `assets/robotstudio_so101/`。`models/so101.xml`
是针对本场景改色、修改网格路径和相机参数后的本地副本。

## 安装

MuJoCo 装在现有 `lerobot` Conda 环境中，不需要新建大型环境：

```bash
conda run -n lerobot python -m pip install -e simulation/mujoco_wood_pick
```

## 检查 EGL 和双相机

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/check_install.py --gl egl
```

脚本在 WSL 中默认设置 `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`，避免双显卡笔记本
把 OpenGL/EGL 自动分配给 Intel 核显。可用 `--adapter ""` 取消，或传入其他设备名。

如果以后需要 CPU 软件渲染，先安装系统包 `libosmesa6`，再执行：

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/check_install.py --gl osmesa
```

当前这台 WSL 已验证的是 NVIDIA EGL 路径，未安装 OSMesa 系统库。

## 交互调整颜色与灯光

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/tune_appearance.py
```

界面可以切换顶部、腕部和概览相机，并实时调整材质 RGB、粗糙度、环境光、
相机补光、主灯和补光。点击“保存本地配置”会写入被 Git 忽略的
`config/appearance.local.json`；其他预览和环境代码下次启动时会自动加载它。

## 保存三张预览图

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/preview_scene.py \
  --gl egl --output-dir outputs/mujoco_scene_preview
```

输出 `front.png`、`wrist.png`、`overview.png` 和并排的 `front_wrist.png`。

## 性能检查

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/benchmark.py --frames 100
```

## Run the real-trained ACT policy

The runner converts between the calibrated LeRobot motor coordinates and
MuJoCo radians, executes the checkpoint closed-loop at 30 Hz, and records both
camera streams plus a compressed joint/trajectory trace:

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/run_trained_policy.py
```

Add `--viewer` to watch the policy in MuJoCo's interactive 3D window while it
runs (closing the window stops the run):

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/run_trained_policy.py --viewer
```

The policy's exact visual inputs can be shown as one or two live camera views:

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/run_trained_policy.py \
  --viewer --camera-view both
```

Use `front` or `wrist` instead of `both` to show only one stream. Press `q` or
Escape in the camera window to stop the episode.

By default it runs one 40-second episode with the local checkpoint at
`outputs/train/act_so101_test/checkpoints/025000/pretrained_model` and writes
results to `outputs/mujoco_trained_policy/`.

## Run the scripted grasp baseline

This mode randomizes the stick on the platform, samples TCP poses in its two
regular ±3 cm sections, filters the complete grasp/lift/transport route with
Mink constrained IK, and executes a minimum-jerk waypoint trajectory:

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/run_scripted_control.py \
  --viewer --camera-view both
```

Click `重新开始` in the optional camera window, or focus the MuJoCo 3-D viewer
and press `R`, to replay the same object pose and planned trajectory. No separate
reset-only window is opened. Replay is
available both during execution and after it finishes. It resets the robot,
stick, controller, contacts, weld state, and simulation clock while preserving
the current user-controlled 3-D viewing camera. Close either interactive window
to exit. Replayed recordings use an `_replay_NNN` filename suffix.

The bright magenta sphere in the MuJoCo 3-D viewer is the scripted planner TCP:
the midpoint of the two central fingertip collision markers at a `-3.1` degree
gripper command. It is intentionally excluded from front/wrist camera images.

It records the same front/wrist videos plus commanded/actual joints, TCP and
stick trajectories in `outputs/mujoco_scripted_control/`. This first baseline
uses local IK and task-specific clearance waypoints; it does not yet provide a
global collision-free planning guarantee.

## 交互查看原始场景

WSLg 图形窗口可用时：

```bash
conda run -n lerobot python simulation/mujoco_wood_pick/scripts/preview_scene.py --viewer
```

当前腕部相机沿用 Menagerie 的物理安装座，顶部相机内参和腕部精确外参仍应在
相机标定完成后更新。机器人基座原点相对真实“底座后缘中点”的毫米级偏移也仍需实测。
