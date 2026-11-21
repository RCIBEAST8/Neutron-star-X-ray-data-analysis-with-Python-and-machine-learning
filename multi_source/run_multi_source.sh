#!/usr/bin/env bash
# multi-source extension, runs the same model on GX 339-4, V404 Cyg and Sco X-1 alongside Aql X-1
# ms_data.py downloads any raw file that's missing (URLs are in scripts/ms_config.py)
# run run_all.sh first, the Aql X-1 files come from the main scripts

# stops straight away if any step fails
set -euo pipefail
PY="${PY:-$HOME/venvs/jax/bin/python}"
cd "$(dirname "$0")/scripts"

# ms_no_splice.py isn't in here, it's a separate re-check that gets run on its own
for s in ms_data ms_experiments ms_seed_check ms_figures ms_verify; do
    echo "=== $s"
    "$PY" "$s.py"
done
