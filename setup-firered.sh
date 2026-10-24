#!/usr/bin/env bash
# Build all Castward model families and fetch the pinned native FireRed Instruct package.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cuda=ON
case "${1:-}" in
    --cpu) cuda=OFF; shift ;;
    -h|--help)
        echo 'Usage: ./setup-firered.sh [--cpu]'
        echo 'AUDIOCPP_SOURCE selects the audio.cpp checkout (default: ../Ming/audio.cpp).'
        echo 'AUDIOCPP_BUILD_DIR selects a build directory (default: existing source/build/ming-cuda).'
        echo 'BUILD_JOBS selects build parallelism (default: 2; increase for faster builds).'
        echo 'CUDA_ARCHITECTURES overrides GPU targets (default: keep cached targets; native for fresh builds).'
        echo '--cpu uses .cache/audio.cpp-cpu-build by default; launch with a CPU native config.'
        exit 0 ;;
esac
if (($#)); then echo "Unexpected argument: $1" >&2; exit 2; fi
source_dir="${AUDIOCPP_SOURCE:-$PWD/../Ming/audio.cpp}"
if [[ ! -f "$source_dir/src/models/fireredtts3/session.cpp" ]]; then
    echo "FireRed-capable audio.cpp source is missing: $source_dir" >&2
    exit 1
fi
command -v hf >/dev/null || { echo 'Install the Hugging Face hf CLI first.' >&2; exit 1; }
build_dir="${AUDIOCPP_BUILD_DIR:-$source_dir/build/ming-cuda}"
if [[ "$cuda" == OFF ]]; then
    build_dir="${AUDIOCPP_BUILD_DIR:-$PWD/.cache/audio.cpp-cpu-build}"
fi
build_args=("-DENGINE_ENABLE_CUDA=$cuda")
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    build_args+=(-DCMAKE_BUILD_TYPE=Release)
    if [[ "$cuda" == ON ]]; then
        build_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-native}")
    fi
elif [[ "$cuda" == ON && -n "${CUDA_ARCHITECTURES:-}" ]]; then
    build_args+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
fi
echo "Updating audio.cpp in $build_dir (existing compiler and target settings are retained)."
cmake -S "$source_dir" -B "$build_dir" \
    -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=ming3_tts,kokoro_tts,qwen3_tts,fireredtts3 \
    "${build_args[@]}"
cmake --build "$build_dir" --target audiocpp_server --parallel "${BUILD_JOBS:-2}"
HF_HOME="$PWD/.cache/huggingface" hf download audio-cpp/audio.cpp-gguf \
    FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-q8_0.gguf \
    --revision ef1ab2598f6d4d101486e4c5b9e605c687f20642 --local-dir models/audiocpp-extra
echo 'FireRed is installed. Configure model paths/devices in audiocpp-server.local.json, then run ./run-local.sh.'
echo "For a custom build directory, set AUDIOCPP_SERVER=$build_dir/bin/audiocpp_server when launching."
