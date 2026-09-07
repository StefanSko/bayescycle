#!/usr/bin/env bash
# Run from a fresh copy of this workspace containing scripts/model/input only.
# Existing outputs are evidence: never delete them to rerun this script.
set -euo pipefail
cd "$(dirname "$0")"
for path in results design.json provenance.json; do
  if [[ -e "$path" ]]; then echo "Refusing existing $path" >&2; exit 1; fi
done
export JAX_ENABLE_X64=true MPLBACKEND=Agg OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export BAYESCYCLE_NO_AUTO_PROVISION=1
mkdir -p logs
uv run --project .. python prepare_design.py
uv run --project .. bayescycle prior-predictive model.py --data design.json -o results/prior --backend bayesjax --seed 4200 --draws 200 > logs/prior-design.log 2>&1
uv run --project .. python analysis.py prior > logs/prior-analysis.log 2>&1
{ time uv run --project .. bayescycle sample model.py --data input/recovery.json -o results/recovery --backend bayesjax --seed 4301 --chains 4 --warmup 500 --draws 1000 --target-accept 0.9 --max-treedepth 10; } > logs/recovery.log 2>&1
uv run --project .. python analysis.py recovery > logs/recovery-analysis.log 2>&1
{ time uv run --project .. bayescycle sample model.py --data input/data.json -o results/initial --backend bayesjax --seed 4201 --chains 4 --warmup 500 --draws 1000 --target-accept 0.9 --max-treedepth 10; } > logs/initial.log 2>&1
# CLI standalone pinned uvx export/visualization boundary. Failure is retained,
# not repaired by switching inference backends; supplementary trace remains.
if ! { time uv run --project .. bayescycle plot trace results/initial -o results/initial/cli-trace.png --no-auto-provision; } > logs/cli-plot.log 2>&1; then
  echo 'CLI visualization failed; inspect logs/cli-plot.log' >&2
fi
uv run --project .. python analysis.py initial > logs/initial-analysis.log 2>&1
uv run --project .. python record_provenance.py
