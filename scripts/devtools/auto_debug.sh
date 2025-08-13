#!/usr/bin/env bash
set -euo pipefail

TARGETS=("core" "infrastructure" "strategies" "trading_system")
echo "[auto-debug] starting loop (Ctrl+C to stop)"
while true; do
  echo "---- lint (errors only) ----"
  pylint --errors-only "${TARGETS[@]}" || true
  echo "---- quick tests ----"
  pytest -q || true
  sleep 10
done
