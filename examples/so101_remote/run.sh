#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
local_python="/home/jing/miniconda3/envs/lerobot/bin/python"
remote_repo="/home/jing/allprojects/robo_project/lerobot/lerobot_joyand"

case "${1:-}" in
  server)
    cd "$repo_dir"
    exec "$local_python" -m lerobot.async_inference.policy_server \
      --host=127.0.0.1 \
      --port=8080 \
      --fps=30
    ;;
  tunnel)
    exec ssh -N \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -R 18080:127.0.0.1:8080 \
      jing
    ;;
  client)
    exec ssh -t jing \
      "cd '$remote_repo' && exec /home/jing/miniconda3/envs/lerobot/bin/python -m lerobot.async_inference.robot_client --config_path examples/so101_remote/jing_client.yaml"
    ;;
  *)
    echo "Usage: $0 {server|tunnel|client}" >&2
    exit 2
    ;;
esac
