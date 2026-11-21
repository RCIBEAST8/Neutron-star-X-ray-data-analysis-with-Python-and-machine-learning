#!/usr/bin/env bash
# runs all the Aql X-1 scripts in order, then the checks in checks.py
# needs a python with jax, flax, optax, astropy, pandas and matplotlib (set PY to point at it if it's not ~/venvs/jax)

# stops straight away if any step fails
set -euo pipefail
PY="${PY:-$HOME/venvs/jax/bin/python}"
cd "$(dirname "$0")/scripts"

# neural_network.py isn't in the list, it only defines the network and training.py imports it
for s in importing_data fourier_features training testing_and_plots forecasting checks; do
    echo "=== $s"
    "$PY" "$s.py"
done
