#!/usr/bin/env bash
set -euo pipefail

# Runs this repo's `instance_setup.py` on a remote machine over SSH.
#
# Usage:
#   ./remote_instance_setup.sh
#   ./remote_instance_setup.sh --tmux
#
# Notes:
# - Assumes a Debian/Ubuntu-like remote (uses apt-get).
# - Requires you can SSH to the host below.

HOST="${HOST:-root@142.170.89.112}"
PORT="${PORT:-15447}"

REPO_URL="${REPO_URL:-https://github.com/rtavasso/audioEngram.git}"
REPO_DIR="${REPO_DIR:-audioEngram}"

MODE="${1:-}"

remote_cmd_common=$(
  cat <<'SH'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates git curl wget tar \
  python3 python3-venv

rm -rf "$REPO_DIR"
git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

python3 instance_setup.py
SH
)

remote_cmd_tmux=$(
  cat <<'SH'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates git curl wget tar tmux \
  python3 python3-venv

rm -rf "$REPO_DIR"
git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

tmux new -d -s setup "python3 instance_setup.py |& tee setup.log"
tmux attach -t setup
SH
)

if [[ "$MODE" == "--tmux" ]]; then
  ssh -p "$PORT" "$HOST" "REPO_URL='$REPO_URL' REPO_DIR='$REPO_DIR' bash -lc $(printf %q "$remote_cmd_tmux")"
elif [[ -z "$MODE" ]]; then
  ssh -p "$PORT" "$HOST" "REPO_URL='$REPO_URL' REPO_DIR='$REPO_DIR' bash -lc $(printf %q "$remote_cmd_common")"
else
  echo "Unknown arg: $MODE" >&2
  echo "Usage: $0 [--tmux]" >&2
  exit 2
fi

