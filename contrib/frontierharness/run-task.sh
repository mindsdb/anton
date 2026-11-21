#!/usr/bin/env bash
# Run one FrontierHarness Eval task with anton through Harbor (Terminal-Bench) or Pier (DeepSWE).
# The eval's run-trials.sh calls it through --cmd; see README.md.
#
# Usage: run-task.sh <suite> <task> <model> <jobs-dir>
set -euo pipefail

[ $# -eq 4 ] || { echo "usage: run-task.sh <suite> <task> <model> <jobs-dir>" >&2; exit 2; }
suite=$1 task=$2 model=$3 jobs=$4
here=$(cd "$(dirname "$0")" && pwd)

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
# Task containers install the same anton commit this script was checked out at.
export ANTON_FH_REF=${ANTON_FH_REF:-$(git -C "$here" rev-parse HEAD)}
export ANTON_FH_BASE_URL=${ANTON_FH_BASE_URL:-https://api.mindshub.ai/v1}
# On Runta the key is a stub that the egress proxy swaps for the stored secret.
export ANTON_FH_API_KEY=${ANTON_FH_API_KEY:-${MINDS_API_KEY:-${FIREWORKS_API_KEY:-}}}
DEEP_SWE=${DEEP_SWE:-/work/deep-swe}
# Written by the eval's provisioning on Runta, where egress TLS goes through a proxy CA.
CA_OVERLAY=${CA_OVERLAY:-/work/runta-ca-overlay.yaml}

case "$suite" in
  terminal-bench)
    overlay=()
    [ -f "$CA_OVERLAY" ] && overlay=(--extra-docker-compose "$CA_OVERLAY")
    exec harbor run -d terminal-bench@2.0 -i "$task" -a anton_harbor:AntonHarbor -m "$model" \
      --jobs-dir "$jobs" ${overlay[@]+"${overlay[@]}"} -r 2 -y ;;
  datacurve)
    exec pier run -p "$DEEP_SWE/tasks/$task" --agent-import-path anton_pier:AntonPier -m "$model" \
      --jobs-dir "$jobs" -y ;;
  *)
    echo "run-task.sh: unknown suite $suite" >&2; exit 2 ;;
esac
