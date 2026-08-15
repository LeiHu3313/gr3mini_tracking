#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec bash
fi

case "$1" in
  Gr3Mini-Tracking-Teacher|Gr3Mini-Tracking-Adapter)
    set -- gr3mini-train "$@"
    ;;
esac

exec "$@"
