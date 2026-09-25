#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_compiled_policy.sh <pi05_compiled_regular|pi05_compiled_optimized>}"
case "$VARIANT" in
  pi05_compiled_regular)
    ARTIFACT="${PI05_REGULAR_ARTIFACT:-$HOME/pi05-compiled-regular-openvino-a}"
    ;;
  pi05_compiled_optimized)
    ARTIFACT="${PI05_OPTIMIZED_ARTIFACT:-$HOME/pi05-ssog-a-mc2-mux}"
    ;;
  *)
    echo "unknown compiled policy variant: $VARIANT" >&2
    exit 2
    ;;
esac

exec "${OPENPI_GATEWAY_PYTHON:-$HOME/.venv-compiled-policy/bin/openpi-serve-compiled}" \
  --variant "$VARIANT" \
  --artifact-dir "$ARTIFACT" \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8000
