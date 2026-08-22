#!/bin/bash
# LeRobot+MuJoCo 容器一键启动脚本
# 用法:
#   ./docker/run.sh                 # 交互式 bash (GPU)
#   ./docker/run.sh python xxx.py   # 直接执行命令
#   NO_GPU=1 ./docker/run.sh        # 不启用 GPU
set -e
IMAGE="${IMAGE:-lerobot-mujoco:v0.1}"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DOCKER_ARGS=( --rm )
if [ -t 0 ]; then
    DOCKER_ARGS+=( -it )
fi
if [ "${NO_GPU:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    DOCKER_ARGS+=( --gpus all )
    # WSL2 (Docker Desktop) 下必须覆盖其内置的过期驱动库, 否则 CUDA 初始化失败;
    # 原生 Linux 宿主机上该目录不存在, 此挂载自动跳过。
    if [ -d /usr/lib/wsl ]; then
        DOCKER_ARGS+=( -v /usr/lib/wsl:/usr/lib/wsl )
    fi
fi

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
