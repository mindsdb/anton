#!/usr/bin/env bash
# The eval's --install-script: runs in /work/harness, the anton clone at the pinned commit,
# after Harbor and Pier are installed. Anton itself is installed per task, inside each task
# container, so this only checks that both runners can load the adapter.
set -euo pipefail

here=/work/harness/contrib/frontierharness
export PATH="$HOME/.local/bin:$PATH"
[ -f "$here/run-task.sh" ] || { echo "install.sh: $here not found; is /work/harness the anton clone?" >&2; exit 1; }

tools=$(uv tool dir)
for pair in harbor:anton_harbor datacurve-pier:anton_pier; do
  tool=${pair%%:*} module=${pair#*:}
  PYTHONPATH="$here" "$tools/$tool/bin/python" -c "import $module" \
    || { echo "install.sh: $tool cannot import $module" >&2; exit 1; }
done
echo "anton adapter ready at $here (anton $(git -C /work/harness rev-parse --short HEAD))"
