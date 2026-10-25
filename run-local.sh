#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
speech_python="${PYTHON:-$PWD/.venv/bin/python}"
native_binary="${AUDIOCPP_SERVER:-$PWD/../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server}"
speech_config="${SPEECH_CONFIG:-$PWD/backend.json}"
native_config="${AUDIOCPP_CONFIG:-$PWD/audiocpp-server.local.json}"
if [[ -f "$PWD/.cache/install/settings.json" ]]; then
    if [[ -z "${AUDIOCPP_SERVER:-}" ]]; then
        IFS= read -r native_binary < "$PWD/.cache/install/server-path"
    fi
    speech_config="${SPEECH_CONFIG:-$PWD/.cache/install/backend.json}"
    native_config="${AUDIOCPP_CONFIG:-$PWD/.cache/install/native.json}"
fi
native_pid=""
cpu_pid=""
facade_pid=""
runtime_dir=""
kokoro_cpu=0
gpu_device=0
facade_args=()
while (($#)); do
    case "$1" in
        --kokoro-cpu) kokoro_cpu=1; shift ;;
        --gpu-device)
            if (($# < 2)); then echo '--gpu-device requires an index' >&2; exit 2; fi
            gpu_device="$2"; shift 2 ;;
        --config)
            if (($# < 2)); then echo '--config requires a path' >&2; exit 2; fi
            speech_config="$2"; shift 2 ;;
        --config=*) speech_config="${1#--config=}"; shift ;;
        -h|--help)
            echo 'Castward Voices · local speech server'
            echo 'Usage: ./run-local.sh [--kokoro-cpu] [--gpu-device INDEX] [server.py options]'
            echo '  --kokoro-cpu       CPU Kokoro/PocketTTS + CUDA character speech, one resident GPU model'
            echo '  --gpu-device INDEX GPU for CPU-narration mode (default: 0)'
            echo '  --port PORT        Compatibility API port (default: 8001)'
            exit 0 ;;
        *) facade_args+=("$1"); shift ;;
    esac
done
cleanup() {
    if [[ -n "$facade_pid" ]]; then kill "$facade_pid" 2>/dev/null || true; fi
    if [[ -n "$cpu_pid" ]]; then kill "$cpu_pid" 2>/dev/null || true; fi
    if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || true; fi
    for pid in "$facade_pid" "$cpu_pid" "$native_pid"; do
        if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
    done
    if [[ -n "$runtime_dir" ]]; then rm -rf -- "$runtime_dir"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export LD_LIBRARY_PATH="${AUDIOCPP_CUDA_LIB:-/usr/local/cuda/lib64}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
wait_native() {
    "$speech_python" - "$1" "$2" "$3" "$4" <<'PY'
import os,sys,time,httpx
url,pid,expected,logfile=sys.argv[1:]
with httpx.Client(timeout=1) as client:
    for _ in range(120):
        try:
            os.kill(int(pid),0)
        except ProcessLookupError:
            raise SystemExit(f'audio.cpp exited; see {logfile}')
        try:
            response=client.get(url+'/health')
            response.raise_for_status()
            if expected and response.json().get('backend') != expected:
                raise SystemExit(f'Unexpected backend on {url}; see {logfile}')
            break
        except httpx.HTTPError:
            time.sleep(0.25)
    else:
        raise SystemExit(f'audio.cpp did not become ready; see {logfile}')
PY
}
if ((kokoro_cpu)); then
    mkdir -p "$PWD/.cache"
    runtime_dir="$(mktemp -d "$PWD/.cache/launch.XXXXXX")"
    gpu_port="${AUDIOCPP_GPU_PORT:-7864}"
    cpu_port="${AUDIOCPP_CPU_PORT:-7863}"
    "$speech_python" launch_config.py "$native_config" "$speech_config" "$runtime_dir" \
        --gpu-device "$gpu_device" --gpu-port "$gpu_port" --cpu-port "$cpu_port"
    native_config="$runtime_dir/gpu.json"
    speech_config="$runtime_dir/backend.json"
    "$native_binary" --config "$native_config" --no-ui > audiocpp-backend-cpu-mode-gpu.log 2>&1 &
    native_pid=$!
    "$native_binary" --config "$runtime_dir/cpu.json" --no-ui > audiocpp-backend-cpu.log 2>&1 &
    cpu_pid=$!
    wait_native "http://127.0.0.1:$gpu_port" "$native_pid" cuda audiocpp-backend-cpu-mode-gpu.log
    wait_native "http://127.0.0.1:$cpu_port" "$cpu_pid" cpu audiocpp-backend-cpu.log
    echo "Kokoro/PocketTTS: CPU; other character speech: CUDA device $gpu_device (one resident GPU model)"
elif ! "$speech_python" - "$speech_config" <<'PY'
import json,sys,httpx
try:
    response=httpx.get(json.load(open(sys.argv[1]))['backend_url'].rstrip('/')+'/health',timeout=2)
    response.raise_for_status()
except httpx.HTTPError:
    sys.exit(1)
PY
then
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
"$speech_python" server.py --config "$speech_config" "${facade_args[@]}" &
facade_pid=$!
children=("$facade_pid")
if [[ -n "$native_pid" ]]; then children+=("$native_pid"); fi
if [[ -n "$cpu_pid" ]]; then children+=("$cpu_pid"); fi
wait -n "${children[@]}"
