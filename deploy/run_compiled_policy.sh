#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_compiled_policy.sh <pi05_compiled_regular|pi05_compiled_optimized>}"
case "$VARIANT" in
  pi05_compiled_regular)
    ARTIFACT=/var/lib/spring-data/models/pi05-compiled-regular-openvino-a
    ;;
  pi05_compiled_optimized)
    ARTIFACT=/home/sentradel/pi05-ssog-a-mc2-mux
    ;;
  *)
    echo "unknown compiled policy variant: $VARIANT" >&2
    exit 2
    ;;
esac

exec /home/sentradel/.venv-compiled-policy/bin/openpi-serve-compiled \
  --variant "$VARIANT" \
  --artifact-dir "$ARTIFACT" \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8000
