#!/usr/bin/env bash
# Compatibility entry point; all installation logic lives in the shared setup.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./setup.sh --add pocket "$@"
