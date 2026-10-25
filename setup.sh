#!/usr/bin/env bash
# One installer for the shared server and its optional models.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
exec uv run --no-project --python 3.11 "$PWD/setup_models.py" "$@"
