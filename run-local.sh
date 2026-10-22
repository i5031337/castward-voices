#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
speech_python="${PYTHON:-$PWD/.venv/bin/python}"
native_binary="${AUDIOCPP_SERVER:-$PWD/../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server}"
speech_config="${SPEECH_CONFIG:-$PWD/backend.json}"
native_config="${AUDIOCPP_CONFIG:-$PWD/audiocpp-server.local.json}"
native_pid=""
facade_pid=""
cleanup() {
    if [[ -n "$facade_pid" ]]; then kill "$facade_pid" 2>/dev/null || true; fi
    if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! "$speech_python" - "$speech_config" <<'PY'
import json,sys,httpx
try:
    response=httpx.get(json.load(open(sys.argv[1]))['backend_url'].rstrip('/')+'/health',timeout=2)
    response.raise_for_status()
except httpx.HTTPError:
    sys.exit(1)
PY
then
    export LD_LIBRARY_PATH="${AUDIOCPP_CUDA_LIB:-/usr/local/cuda-13.1/lib64}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
    "$native_binary" --config "$native_config" --no-ui > audiocpp-backend.log 2>&1 &
    native_pid=$!
    "$speech_python" - "$speech_config" <<'PY'
import json,sys,time,httpx
url=json.load(open(sys.argv[1]))['backend_url'].rstrip('/')+'/health'
for _ in range(60):
    try:
        httpx.get(url,timeout=1).raise_for_status()
        break
    except httpx.HTTPError:
        time.sleep(0.25)
else:
    raise SystemExit('audio.cpp did not become ready; see audiocpp-backend.log')
PY
fi
"$speech_python" server.py --config "$speech_config" "$@" &
facade_pid=$!
wait "$facade_pid"
