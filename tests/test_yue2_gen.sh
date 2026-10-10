#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
export PATH="$ROOT/.zig-out/bin:$PATH"
export MLX_ROOT="$ROOT"

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
MODEL=${YUE2_TEST_MODEL:-}
FIX=${YUE2_FIXTURES:-}
if [[ -z "$MODEL" || -z "$FIX" ]]; then
  echo "SKIP: YUE2_TEST_MODEL or YUE2_FIXTURES not set"
  exit 0
fi

# ensure converted pack is readable
test -f "$MODEL/config.json"
test -f "$MODEL/ar.safetensors"
test -f "$MODEL/nar.safetensors"
test -f "$MODEL/vae.safetensors"
test -f "$MODEL/qwen.tiktoken"

mlx-serve --port "$PORT" --models-dir "$(dirname "$MODEL")" >/tmp/mlx-serve-yue2.$PORT.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q yue2 || (cat /tmp/mlx-serve-yue2.$PORT.log; false)
echo "ok"
