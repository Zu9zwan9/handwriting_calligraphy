#!/usr/bin/env bash
# Portable launcher to avoid hardcoded /usr/bin/python issues.
# Uses the current environment's Python (e.g., from an activated venv).

set -euo pipefail

# Try preferred python executables in order
PY=""
for cand in python python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "Error: No python interpreter found in PATH. Activate your venv or install Python 3." >&2
  exit 1
fi

exec "$PY" "$(dirname "$0")/animate_text.py" "$@"
